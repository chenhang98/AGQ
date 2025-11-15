# Modified from https://github.com/bytedance/kmax-deeplab/blob/main/kmax_deeplab/modeling/criterion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
from skimage.measure import label as skimage_label
import numpy as np

from interface import AutoLossMixin
from .sparse_inst import (
    compute_mask_iou, 
    is_dist_avail_and_initialized, 
    get_world_size,
    # dice_score
)


# https://www.tensorflow.org/api_docs/python/tf/math/divide_no_nan
def divide_no_nan(x: torch.Tensor, y: torch.Tensor):
    return torch.nan_to_num(x / y, nan=0.0, posinf=0.0, neginf=0.0)


def dice_score(inputs, targets):
    # inputs = inputs.sigmoid()
    inputs = F.softmax(inputs, dim=0)
    inputs = inputs.flatten(1) # N x HW

    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (
        inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        matched_cls_prob: torch.Tensor,
        pixel_gt_void_mask: torch.Tensor,
        reduction: str='sum'
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # inputs = inputs.sigmoid()
    inputs = inputs.softmax(1) # B N HW
    if pixel_gt_void_mask is not None:
        # https://github.com/google-research/deeplab2/blob/main/model/loss/base_loss.py#L111
        inputs = inputs.masked_fill(pixel_gt_void_mask.unsqueeze(1), 0) # remove void pixels.

    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)

    loss *= matched_cls_prob

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    return loss.sum()



def softmax_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        pixel_gt_void_mask: torch.Tensor,
        reduction: str = 'mean',
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.cross_entropy(inputs, targets, reduction="none") # B x HW
    if pixel_gt_void_mask is not None:
        loss = loss.masked_fill(pixel_gt_void_mask, 0) # remove void pixels.

    num_non_zero = (loss != 0.0).to(loss).sum(-1) # B
    num_non_zero = torch.clamp(num_non_zero, min=1.0)
    loss_sum_per_sample = loss.sum(-1) # B
    loss = divide_no_nan(loss_sum_per_sample, num_non_zero)

    if reduction == 'none':
        return loss
    return loss.mean()


def _gumbel_topk_sample(logits: torch.Tensor, k: int):
    """Samples k points from the softmax distribution with Gumbel-Top-k trick."""
    # Note that torch.rand is [0, 1), we need to make it (0, 1) to ensure the log is valid.
    gumbel_noise = torch.rand(size=logits.shape, dtype=logits.dtype, device=logits.device)
    gumbel_noise = -torch.log(-torch.log(gumbel_noise))
    _, indices = torch.topk(logits + gumbel_noise, k)
    return indices


def pixelwise_insdis_loss(
        pixel_feature: torch.Tensor,
        gt_mask: torch.Tensor,
        sample_temperature: float,
        sample_k: int,
        instance_discrimination_temperature: float,
    ):
    # pixel_feature: B x C x D x H x W
    # gt_mask: B x N x D x H x W

    _SOFTMAX_MASKING_CONSTANT = -99999.0

    pixel_gt_void_mask = (gt_mask.sum(1) < 1) # B x D x H x W

    mask_gt_area = gt_mask.sum(2).sum(2).sum(2)         # B x N
    pixel_gt_area = torch.einsum('bndhw,bn->bdhw', gt_mask, mask_gt_area) # B x D x H x W
    inverse_gt_mask_area = (
            pixel_gt_area.shape[1] * pixel_gt_area.shape[2] * pixel_gt_area.shape[3]
        ) / torch.clamp(pixel_gt_area, min=1.0) # B x D x H x W
    gt_mask = gt_mask.flatten(2) # B x N x DHW

    pixel_gt_void_mask = (gt_mask.sum(1) < 1) # B x D x H x W
    pixel_gt_void_mask = pixel_gt_void_mask.flatten(1) # B x DHW
    inverse_gt_mask_area = inverse_gt_mask_area.flatten(1) # B x DHW

    pixel_feature = pixel_feature.flatten(2) # B x C x DHW

    sample_logits = torch.log(inverse_gt_mask_area) * sample_temperature # B x DHW
    # sample_logits.masked_fill_(pixel_gt_void_mask, float('-inf'))
    sample_logits += pixel_gt_void_mask.to(sample_logits) * _SOFTMAX_MASKING_CONSTANT

    sample_indices = _gumbel_topk_sample(sample_logits, sample_k) # B x K
    # Sample ground truth one-hot encodings and compute gt_similarity.
    pixel_gt_sampled_feature = torch.gather(gt_mask, dim=2, index=sample_indices.unsqueeze(1).repeat(1, gt_mask.shape[1], 1)) # B x N x K
    sampled_gt_similarity = torch.einsum('bnk,bnj->bkj', pixel_gt_sampled_feature, pixel_gt_sampled_feature) # B x K x K; 1 iif belong to the same instance

    # Normalize the ground truth similarity into a distribution (sum to 1).
    pixel_normalizing_constant = sampled_gt_similarity.sum(dim=1, keepdim=True) # B x 1 x K
    sampled_gt_similarity /= torch.clamp(pixel_normalizing_constant, min=1.0) # B x K x K

    # Sample predicted features and compute pred_similarity.
    pixel_pred_sampled_feature = torch.gather(pixel_feature, dim=2, index=sample_indices.unsqueeze(1).repeat(1, pixel_feature.shape[1], 1)) # B x C x K
    sampled_pred_similarity = torch.einsum('bck,bcj->bkj', pixel_pred_sampled_feature, pixel_pred_sampled_feature) # B x K x K
    sampled_pred_similarity /= instance_discrimination_temperature # B x K x K
    loss = F.cross_entropy(sampled_pred_similarity, sampled_gt_similarity, reduction="none") # B x K

    num_non_zero = (loss != 0.0).to(loss).sum(-1) # B
    num_non_zero = torch.clamp(num_non_zero, min=1.0)
    loss_sum_per_sample = loss.sum(-1) # B
    return divide_no_nan(loss_sum_per_sample, num_non_zero).mean() # 1



class PanopticCriterion3D(nn.Module):
    # This part is partially derivated from: https://github.com/facebookresearch/detr/blob/main/models/detr.py
    default_weight_dict = dict(loss_mask=0.3, loss_dice=3.0, loss_matchness=1.0, loss_pixel=0.1, loss_kernel=0.0, loss_affinity=1.0)


    def __init__(self, matcher, **update_weight_dict):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = deepcopy(self.default_weight_dict)
        self.weight_dict.update(update_weight_dict)


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def loss_matchness(self, outputs, indices):
        src_idx = self._get_src_permutation_idx(indices)

        # Bx100xDxHxW
        assert "pred_match" in outputs

        src_match = outputs["pred_match"]
        tgt_match = src_match.new_zeros(src_match.size())
        tgt_match[src_idx] = 1

        losses = {
            "loss_matchness": F.binary_cross_entropy_with_logits(src_match, tgt_match, reduction='mean'),
        }
        return losses


    def loss_kernel(self, outputs, bg_index=0):
        # pred_kernel: BxNxC
        assert bg_index == 0

        pred_kernel = outputs["pred_kernel"]
        if pred_kernel.size(1) == outputs['pred_masks'].size(1):    # contaion bg kernel
            pred_kernel = pred_kernel[:, 0:, :]

        pred_kernel = pred_kernel / pred_kernel.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        cosine_sim = torch.bmm(pred_kernel, pred_kernel.transpose(1, 2))

        B, N, C = pred_kernel.shape
        label = torch.arange(N, device=cosine_sim.device).repeat(B, 1)

        losses = {
            "loss_kernel": F.cross_entropy(cosine_sim, label)
        }
        return losses


    def loss_masks_with_iou_objectness(self, outputs, target_masks, indices, num_instances, bg_index):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # Bx100xDxHxW
        assert "pred_masks" in outputs

        src_masks = outputs["pred_masks"]

        # Pad and permute the target_mask to B x N x D x H x W
        pad_target_masks = torch.zeros_like(src_masks)
        pad_target_masks_o = torch.cat([
                    m[J] for m, (_, J) in zip(target_masks, indices)]).to(pad_target_masks)
        pad_target_masks[src_idx] = pad_target_masks_o

        # Note that for instance segmentation masks may overlap with each other, here we normalize
        # the mask to ensure they sum to one
        pad_target_masks = pad_target_masks / torch.clamp(pad_target_masks.sum(1, keepdim=True), min=1.0)

        src_masks = src_masks.flatten(2)                                # B, N, DHW
        pad_target_masks = pad_target_masks.flatten(2)                  # B, N, DHW
        pixel_gt_void_mask = (pad_target_masks.sum(1) < 1).flatten(1)   # B x DHW,  BG mask
        # with torch.no_grad():
        #     ious = compute_mask_iou(src_masks, pad_target_masks)

        # fill channel bg_mask to background
        pad_target_masks[:,bg_index] = 1 - pad_target_masks.sum(1)


        matched_cls_prob = torch.full(
            src_masks.shape[:2], 0, dtype=src_masks.dtype, device=src_masks.device
        ) # B x N
        matched_cls_prob[src_idx] = 1

        losses = {
            # "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            # "loss_dice": dice_loss(src_masks, pad_target_masks) / num_instances,
            "loss_dice": dice_loss(src_masks, pad_target_masks, matched_cls_prob, pixel_gt_void_mask, reduction='mean'),
            # "loss_mask": softmax_ce_loss(src_masks, pad_target_masks, pixel_gt_void_mask)
            "loss_mask": softmax_ce_loss(src_masks, pad_target_masks, None),
        }

        # instance dis. loss
        if outputs.get('pixel_feature', None) is not None:
            B, _, D, H, W = outputs['pixel_feature'].shape
            losses["loss_pixel"] = pixelwise_insdis_loss(
                outputs['pixel_feature'], 
                pad_target_masks[:,1:].view(B, -1, D, H, W),            # don't calculate inst. dis. loss on BG
                sample_temperature=0.6,
                sample_k=4096,
                instance_discrimination_temperature=0.3,
            )

        return losses


    def get_num_instances(self, target_masks, outputs):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_instances = sum(t.size(0) for t in target_masks)        # num_instances in this GPU
        num_instances = torch.as_tensor(
            [num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(
            num_instances / get_world_size(), min=1).item()         # average num_instances per GPU
        return num_instances


    def _get_loss(self, outputs, target_masks, indices, num_instances, bg_index=0):
        losses = dict()
        losses.update(
            self.loss_masks_with_iou_objectness(
                    outputs, target_masks, indices, num_instances, bg_index=bg_index)
        )
        if self.weight_dict.get("loss_kernel", 0) > 0:
            losses.update(
                self.loss_kernel(outputs, bg_index=bg_index)
            )
        return losses


    def loss_affinity(self, outputs, targets):
        pred_affinity = outputs["pred_affinity"]
        gt_affinity = targets[1]
        eps = 1e-5

        gt_affinity = F.interpolate(
            gt_affinity.to(pred_affinity),
            size=pred_affinity.shape[2:], 
            mode="trilinear", 
            align_corners=False
        )
        loss = F.binary_cross_entropy_with_logits(pred_affinity, gt_affinity.clamp(eps,1-eps))
        return {"loss_affinity": loss}


    def _forward_one_group(self, outputs, targets, weight, bg_index=0):
        assert outputs.get('pred_match', None) is None

        target_masks = self.prepare_target_masks(targets, outputs)
        num_instances = self.get_num_instances(target_masks, outputs)

        # DEBUG
        # mask_int = targets[0][0].cpu().numpy()
        # for j in np.unique(mask_int):
        #     if j == 0:
        #         continue
        #     cc_masks = skimage_label(mask_int == j)
        #     if cc_masks.max() > 1:
        #         print(j, cc_masks.max(), (cc_masks == 2).sum())

        losses = dict()
        indices = None

        # Process aux. losses
        for i, aux_output in enumerate(outputs['aux_outputs']):
            # Retrieve the matching between the outputs of the last layer and the targets
            if indices is None:
                indices = self.matcher(aux_output, target_masks)

            # Compute all the requested losses
            aux_losses = self._get_loss(aux_output, target_masks, indices, num_instances, bg_index=bg_index)
            for k in aux_losses.keys():
                if k in self.weight_dict:
                    losses[f"{k}_{i}"] = aux_losses[k] * self.weight_dict[k]

        # Last layer
        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if indices is None:
            indices = self.matcher(outputs_without_aux, target_masks)

        # Compute all the requested losses
        losses.update(
            self._get_loss(outputs, target_masks, indices, num_instances, bg_index=bg_index)
        )

        if "pred_affinity" in outputs and self.weight_dict.get("loss_affinity", 0) > 0:
            losses.update(
                self.loss_affinity(outputs, targets)
            )

        # calculate loss by module
        for out in outputs.get("auto_loss_callbacks", []):
            loss_dict = out.get_loss(targets)
            losses.update(loss_dict)

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses


    def forward(self, outputs, targets, weight, bg_index=0):
        """process multi. group outputs; compatibility with pytorch-connectomics"""
        output_aux_groups = outputs.pop('aux_groups', [])
        losses = self._forward_one_group(outputs, targets, weight, bg_index)

        for i, output_group_i in enumerate(output_aux_groups):
            losses_i = self._forward_one_group(output_group_i, targets, weight, bg_index)
            for k, v in losses_i.items():
                losses[f"{k}_aux_group_{i}"] = v
        return sum(losses.values()), losses


    def prepare_target_masks(self, targets, outputs):
        """
        Prepare and resize target masks. (1) decompose all instance masks 
        for the tensor of instance ids. (2) resize.
        shape change: (B, D, H', W') -> list of (Ni, D, H, W)
        """
        pred_masks = outputs["pred_masks"]     # B, N, D, H, W

        out = []
        for tgt in targets[0]:
            ids = tgt.unique()
            fg_masks = []
            for i in ids:
                if i > 0:
                    fg_masks.append(tgt == i)
            fg_masks = torch.stack(fg_masks, dim=0)     # (Ni, D, H', W')
            fg_masks = F.interpolate(
                        fg_masks[:, None].to(pred_masks), 
                        size=pred_masks.shape[2:], 
                        mode="trilinear", 
                        align_corners=False
                ).squeeze(1)
            out.append(fg_masks)
        return out



class PanopticMatcher3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.mask_score = dice_score

    def forward(self, outputs, tgt_masks, bg_index=0):
        with torch.no_grad():
            B, N, D, H, W = outputs["pred_masks"].shape
            pred_masks = outputs['pred_masks'].float()

            indices = []
            for b in range(B):
                this_pred_masks = pred_masks[b].flatten(1)          # N, DHW
                this_tgt_masks = tgt_masks[b].flatten(1)            # K, DHW

                with autocast(enabled=False):
                    mask_score = self.mask_score(this_pred_masks, this_tgt_masks)
                    mask_score[bg_index] = 0        # avoid background prediction matching with any fg masks

                # Nx(Number of gts)
                C = mask_score.cpu()

                # hungarian matching
                row_ind, col_ind = linear_sum_assignment(C, maximize=True)
                indices.append((row_ind, col_ind))

        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
        return indices
