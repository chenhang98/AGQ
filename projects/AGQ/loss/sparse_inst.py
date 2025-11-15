# Modified from https://github.com/hustvl/SparseInst
# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

import torch
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
import torchvision



def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def aligned_bilinear(tensor, factor):
    # borrowed from Adelaidet: https://github1s.com/aim-uofa/AdelaiDet/blob/HEAD/adet/utils/comm.py
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow),
        mode='bilinear',
        align_corners=True
    )
    tensor = F.pad(
        tensor, pad=(factor // 2, 0, factor // 2, 0),
        mode="replicate"
    )

    return tensor[:, :, :oh - 1, :ow - 1]


def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_score(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (
        inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def dice_loss(inputs, targets, reduction='sum'):
    inputs = inputs.sigmoid()
    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    return loss.sum()


class SparseInstCriterion3D(nn.Module):
    # This part is partially derivated from: https://github.com/facebookresearch/detr/blob/main/models/detr.py
    default_weight_dict = dict(loss_mask=5.0, loss_dice=2.0, loss_objectness=1.0, loss_matchness=1.0)

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


    def loss_masks_with_iou_objectness(self, outputs, target_masks, indices, num_instances):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # Bx100xDxHxW
        assert "pred_masks" in outputs
        assert "pred_scores" in outputs

        src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]
        num_masks = [m.size(0) for m in target_masks]

        target_masks = torch.cat(target_masks, dim=0).to(src_masks)       # N, D, H, W
        if len(target_masks) == 0:
            losses = {
                "loss_dice": src_masks.sum() * 0.0,
                "loss_mask": src_masks.sum() * 0.0,
                "loss_objectness": src_iou_scores.sum() * 0.0
            }
            return losses

        src_masks = src_masks[src_idx]
        target_masks = F.interpolate(target_masks[:, None], 
                            size=src_masks.shape[-3:], mode="trilinear", align_corners=False).squeeze(1)

        src_masks = src_masks.flatten(1)
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])
        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        target_masks = target_masks[mix_tgt_idx].flatten(1)

        with torch.no_grad():
            ious = compute_mask_iou(src_masks, target_masks)

        tgt_iou_scores = ious
        src_iou_scores = src_iou_scores[src_idx]
        tgt_iou_scores = tgt_iou_scores.flatten(0)
        src_iou_scores = src_iou_scores.flatten(0)

        losses = {
            "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            "loss_dice": dice_loss(src_masks, target_masks) / num_instances,
            "loss_mask": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        }
        return losses


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


    def forward(self, outputs, targets, weight):

        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}
        target_masks = self.prepare_target_masks(targets)

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, target_masks)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_instances = sum(t.size(0) for t in target_masks)        # num_instances in this GPU
        num_instances = torch.as_tensor(
            [num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(
            num_instances / get_world_size(), min=1).item()         # average num_instances per GPU
        # Compute all the requested losses
        losses = dict()
        losses.update(
            self.loss_masks_with_iou_objectness(
                        outputs, target_masks, indices, num_instances)
        )
        losses.update(
            self.loss_matchness(outputs, indices)
        )

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return sum(losses.values()), losses


    def prepare_target_masks(self, targets):
        # decompose all instance masks for the tensor of instance ids
        # shape change: (B, D, H, W) -> list of (Ni, D, H, W)
        out = []
        for tgt in targets[0]:
            ids = tgt.unique()
            masks = []
            for i in ids:
                if i > 0:
                    masks.append(tgt == i)
            masks = torch.stack(masks, dim=0)
            out.append(masks)
        return out


class SparseInstMatcher3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.mask_score = dice_score

    def forward(self, outputs, target_masks):
        with torch.no_grad():
            B, N, D, H, W = outputs["pred_masks"].shape
            pred_masks = outputs['pred_masks']

            num_insts = [m.size(0) for m in target_masks]       # #instances in each image
            if sum(num_insts) == 0:
                return [(torch.as_tensor([]).to(pred_masks), torch.as_tensor([]).to(pred_masks))] * B

            tgt_masks = torch.cat(target_masks, dim=0).to(pred_masks)       # N, D, H, W
            tgt_masks = F.interpolate(tgt_masks[:, None], 
                                size=pred_masks.shape[2:], mode="trilinear", align_corners=False).squeeze(1)

            pred_masks = pred_masks.view(B * N, -1)
            tgt_masks = tgt_masks.flatten(1)
            with autocast(enabled=False):
                pred_masks = pred_masks.float()
                tgt_masks = tgt_masks.float()
                mask_score = self.mask_score(pred_masks, tgt_masks)
                # Nx(Number of gts)
                C = mask_score

            C = C.view(B, N, -1).cpu()
            # hungarian matching
            indices = [linear_sum_assignment(c[i], maximize=True)
                       for i, c in enumerate(C.split(num_insts, -1))]
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(
                j, dtype=torch.int64)) for i, j in indices]
            return indices
