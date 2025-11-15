import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.distributed as dist
from fvcore.nn.weight_init import c2_msra_fill
from skimage.measure import label
import mahotas
from scipy import ndimage

from connectomics.model.utils import model_init
from connectomics.model.arch import UNet3D
from connectomics.model.block import conv3d_norm_act
from connectomics.model.block import *

from interface import E2EMixin


def get_connected_components(affinity):
    # affinity: shape N, 3, D, H, W; unnormalized
    out = []
    with torch.no_grad():
        affinity = affinity.detach().sigmoid()
        bool_masks = affinity.mean(1) > affinity.mean()        # N, D, H, W
        for bm in bool_masks:
            bm = bm.cpu().numpy().astype(np.int32)
            masks = label(bm)
            out.append(masks)
        out = torch.from_numpy(np.stack(out, axis=0)).to(affinity.device)
    return out


def get_seeds(boundary, method='grid', next_id = 1,
             seed_distance = 10):
    if method == 'grid':
        height = boundary.shape[0]
        width  = boundary.shape[1]

        seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x*num_seeds_y
        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y,num_seeds_x))

    if method == 'minima':
        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    if method == 'maxima_distance':
        distance = mahotas.distance(boundary<0.5)
        maxima = mahotas.regmax(distance)
        seeds, num_seeds = mahotas.label(maxima)
        seeds += next_id
        seeds[seeds==next_id] = 0
    return seeds, num_seeds


def watershed(affs, seed_method, use_mahotas_watershed = True):
    affs_xy = 1.0 - 0.5*(affs[1] + affs[2])
    depth  = affs_xy.shape[0]
    fragments = np.zeros_like(affs[0]).astype(np.uint64)
    next_id = 1
    for z in range(depth):
        seeds, num_seeds = get_seeds(affs_xy[z], next_id=next_id, method=seed_method)
        if use_mahotas_watershed:
            fragments[z] = mahotas.cwatershed(affs_xy[z], seeds)
        else:
            fragments[z] = ndimage.watershed_ift((255.0*affs_xy[z]).astype(np.uint8), seeds)
        next_id += num_seeds

    return fragments


def get_watershed_fragments(affinity):
    # affinity: shape N, 3, D, H, W; unnormalized
    out = []
    with torch.no_grad():
        affinity_np = affinity.detach().sigmoid().cpu().numpy()
        for af in affinity_np:
            masks = watershed(af, 'maxima_distance', use_mahotas_watershed=True)
            out.append(masks.astype(np.int64))
        out = torch.from_numpy(np.stack(out, axis=0)).to(affinity.device)
    return out


class FusionBlock(nn.Module):
    def __init__(self, 
            in_filters=[28, 36, 48, 64, 80],
            out_level=0, 
            out_filter=64, 
            mid_filter=64, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            shared_kwargs={}
        ):
        super().__init__()

        self.out_level = out_level
        self.shared_kwargs = shared_kwargs

        self.pre_convs = nn.ModuleList([
            conv3d_norm_act(f, mid_filter, kernel_size=kernel_size,
                padding=padding, **self.shared_kwargs) \
            for f in in_filters
        ])
        self.post_convs = nn.Sequential(
            conv3d_norm_act(mid_filter, out_filter, kernel_size=kernel_size,
                padding=padding, **self.shared_kwargs)
        )

    def forward(self, feats):
        target_size = feats[self.out_level].shape[2:]

        out = 0
        for i, x in enumerate(feats):
            x = self.pre_convs[i](x)
            x = F.interpolate(x, size=target_size, 
                    mode='trilinear', align_corners=True)
            out += x
        return self.post_convs(out)


class SelfAttention(nn.Module):
    # https://github1s.com/karpathy/nanoGPT/blob/HEAD/model.py#L31-L76

    def __init__(self, n_embd, n_head=8, dropout=0.0, bias=True):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                            dropout_p=self.dropout if self.training else 0, 
                            is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, n_embd, bias=True, dropout=0.0, ratio=4):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, ratio * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(ratio * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SelfAttentionBlock(nn.Module):

    def __init__(self, n_embd, n_heads=8, bias=True, dropout=0.0, pre_norm=False):
        super().__init__()
        self.pre_norm = pre_norm
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_heads, dropout, bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x):
        if self.pre_norm:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = self.ln_1(x + self.attn(x))
            x = self.ln_2(x + self.mlp(x))
        return x


class ConnectedComponetsHead(nn.Module):
    """Convert predicted affinity to coarse init. masks and gather the crossponding instance features."""

    def __init__(self, 
            num_convs=4, 
            num_masks=100,
            num_learned_masks=20,
            kernel_dim=64,
            in_filter=64, 
            out_filter=64, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            shared_kwargs={},
            init_mask_method='connected_components',
            mask_area_thresh=100, 
            return_init_mask=False
        ):
        super().__init__()

        self.shared_kwargs = shared_kwargs
        self.init_mask_method = init_mask_method
        self.mask_area_thresh = mask_area_thresh
        self.return_init_mask = return_init_mask

        in_filters = [in_filter] + [out_filter] * (num_convs - 1)
        self.inst_convs = nn.Sequential(*[
            conv3d_norm_act(f, out_filter, kernel_size=kernel_size,
                    padding=padding, **self.shared_kwargs) \
            for f in in_filters
        ])

        self.num_masks = num_masks
        self.num_learned_masks = num_learned_masks

        if self.num_learned_masks > 0:
            self.learned_masks_embed = nn.Embedding(self.num_learned_masks, out_filter)
        else:
            self.learned_masks_embed = None

        # outputs
        self.mask_kernel = nn.Linear(out_filter, kernel_dim)

        if dist.is_initialized():
            self.norm_logits = nn.SyncBatchNorm(1, eps=1e-3, momentum=0.01)
            self.norm_kernel = nn.SyncBatchNorm(64, eps=1e-3, momentum=0.01)
        else:
            self.norm_logits = nn.BatchNorm2d(1, eps=1e-3, momentum=0.01)
            self.norm_kernel = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)

        if self.num_learned_masks == 0:
            del self.norm_kernel        # not available for only one init. mask

        self.prior_prob = 0.01
        self._init_weights()


    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):        # TODO
                c2_msra_fill(m)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
            # init.constant_(module.bias, bias_value)
        # init.normal_(self.iam_conv.weight, std=0.01)

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)


    def _sample_init_masks(self, pred_affinity):
        if self.init_mask_method == "connected_components":
            init_masks = get_connected_components(pred_affinity)
        elif self.init_mask_method == "watershed":
            init_masks = get_watershed_fragments(pred_affinity)
        return init_masks


    def forward(self, features, mask_features, pred_affinity, init_masks=None):
        # instance features (x4 convs)
        features = self.inst_convs(features)

        # we back to learned_masks_embed only if (1) pred_affinity=None, or (2) len(inds)==0 (no valid init. masks) (3) self.num_masks == 0
        if (pred_affinity is None) or (self.num_masks == 0):
            inst_features = []
            recorded_init_masks = []
            for b in range(features.size(0)):
                # only use learned masks
                inst_features.append(self.learned_masks_embed.weight)

        else:
            if init_masks is None:
                init_masks = self._sample_init_masks(pred_affinity)        # B, D, H, W

            # sample init masks
            B, D, H, W = init_masks.shape

            inst_features = []
            recorded_init_masks = []
            for b, init_masks_per_image in enumerate(init_masks):
                inds, areas = torch.unique(init_masks_per_image, return_counts=True)

                if inds[0] == 0:
                    inds = inds[1:]      # ignore bg
                    areas = areas[1:]

                # filter too small regions
                inds = inds[areas > self.mask_area_thresh]
                inds = inds[torch.randperm(len(inds))]

                if len(inds):       # if init. masks is empty, use learnable embedding instead
                    inst_feature_per_image = torch.stack([
                            features[b, :, init_masks_per_image==i].mean(1) \
                            for i in inds[:self.num_masks]
                        ], dim=0)

                    if self.return_init_mask:
                        recorded_init_masks.append(
                            torch.stack([
                                init_masks_per_image==i \
                                for i in inds[:self.num_masks]
                            ], dim=0)
                        )

                    if self.learned_masks_embed is not None:
                        # add aditional learned masks
                        inst_feature_per_image = torch.cat([
                                    inst_feature_per_image, 
                                    self.learned_masks_embed.weight
                                ], dim=0)
                    inst_features.append(inst_feature_per_image)
                else:
                    inst_features.append(self.learned_masks_embed.weight)

        inst_features = torch.stack(inst_features, dim=0)     # B, N, C

        # predict classification & segmentation kernel & objectness
        pred_kernel = self.mask_kernel(inst_features)
        if self.num_learned_masks > 0:
            pred_kernel = self.norm_kernel(pred_kernel.transpose(1,2)).transpose(1,2)

        N = pred_kernel.shape[1]
        # mask_features: BxCxDxHxW
        B, C, D, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, 
                mask_features.view(B, C, D * H * W)
            ).unsqueeze(dim=1)              # B, 1, N, DHW

        pred_masks = self.norm_logits(pred_masks).view(B, N, D, H, W)

        if self.return_init_mask:
            return features, inst_features, pred_masks, pred_kernel, recorded_init_masks
        return features, inst_features, pred_masks, pred_kernel, None


class Predictor(nn.Module):
    def __init__(self, out_filter, kernel_dim, norm_kernel=True, norm_logits=True):
        super().__init__()
        # outputs
        self.fc_kernel = nn.Linear(out_filter, kernel_dim)

        if dist.is_initialized():
            self.norm_logits = nn.SyncBatchNorm(1, eps=1e-3, momentum=0.01) if norm_logits else None
            self.norm_kernel = nn.SyncBatchNorm(64, eps=1e-3, momentum=0.01) if norm_kernel else None
        else:
            self.norm_logits = nn.BatchNorm2d(1, eps=1e-3, momentum=0.01) if norm_logits else None
            self.norm_kernel = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01) if norm_kernel else None

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.fc_kernel.weight, std=0.01)
        init.constant_(self.fc_kernel.bias, 0.0)

    def forward(self, obj_feat, mask_features, return_kernel=False):
        # obj_feat: 

        # predict classification & segmentation kernel & objectness
        pred_kernel = self.fc_kernel(obj_feat)
        if self.norm_kernel is not None:
            pred_kernel = self.norm_kernel(pred_kernel.transpose(1,2)).transpose(1,2)

        N = pred_kernel.shape[1]
        # mask_features: BxCxDxHxW
        B, C, D, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, 
                mask_features.view(B, C, D * H * W)
            ).unsqueeze(dim=1)              # B, 1, N, DHW

        if self.norm_logits is not None:
            pred_masks = self.norm_logits(pred_masks).view(B, N, D, H, W)
        else:
            pred_masks = pred_masks.view(B, N, D, H, W)

        if return_kernel:
            return pred_masks, pred_kernel
        return pred_masks


class KernelUpdator(nn.Module):

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 gate_sigmoid=True,
                 gate_norm_act=False,
                 activate_out=False,
                 ):
        super(KernelUpdator, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out,
                                     1)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        if self.gate_norm_act:
            self.gate_norm = nn.LayerNorm(self.feat_channels)

        self.norm_in = nn.LayerNorm(self.feat_channels)
        self.norm_out = nn.LayerNorm(self.feat_channels)
        self.input_norm_in = nn.LayerNorm(self.feat_channels)
        self.input_norm_out = nn.LayerNorm(self.feat_channels)

        self.activation = nn.ReLU(inplace=True)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = nn.LayerNorm(self.out_channels)

    def forward(self, update_feature, input_feature):
        B, N, C = update_feature.shape

        update_feature = update_feature.reshape(-1, self.in_channels)       # BN, C
        num_proposals = update_feature.size(0)                  # BN
        parameters = self.dynamic_layer(update_feature)         # BN, 2*C
        param_in = parameters[:, :self.num_params_in].view(
            -1, self.feat_channels)                             # BN, C
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels)                             # BN, C

        input_feats = self.input_layer(
            input_feature.reshape(num_proposals, -1, self.feat_channels))
        input_in = input_feats[..., :self.num_params_in]        # BN, 1, C
        input_out = input_feats[..., -self.num_params_out:]     # BN, 1, C

        gate_feats = input_in * param_in.unsqueeze(-2)          # BN, 1, C
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = input_gate.sigmoid()
            update_gate = update_gate.sigmoid()
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = update_gate * param_out.unsqueeze(
            -2) + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features.reshape(B, N, C)


class InstDecoder(nn.Module):
    def __init__(self, 
            fusion_channel, 
            num_iter=2, 
            init_mask_method='connected_components', 
            num_masks=100, 
            num_learned_masks=100,
            return_init_mask=False,
            share_weights=False,
            bg_conv_as_bias=False,
            bg_conv_share=False,
            bg_conv_norm='bn'
        ):
        super().__init__()
        kernel_dim = 64

        self.rpn = ConnectedComponetsHead(
            num_convs=4, num_masks=num_masks, kernel_dim=64, 
            in_filter=fusion_channel+3, out_filter=fusion_channel, 
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            init_mask_method=init_mask_method,
            num_learned_masks=num_learned_masks,
            return_init_mask=return_init_mask
        )
        self.num_iter = num_iter
        
        self.share_weights = share_weights
        self.bg_conv_as_bias = bg_conv_as_bias
        self.bg_conv_share = bg_conv_share
        self.bg_conv_norm = bg_conv_norm

        if self.share_weights:
            self.predictor = Predictor(fusion_channel, kernel_dim, norm_kernel=(num_learned_masks > 0))
            self.kernel_interactor = SelfAttentionBlock(fusion_channel)
            self.kernel_updator = KernelUpdator(fusion_channel, fusion_channel)

            if bg_conv_as_bias:
                self.bg_bias = nn.Parameter(torch.zeros(1, requires_grad=True))
            elif bg_conv_share:
                self.bg_conv = conv3d_norm_act(fusion_channel, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), norm_mode=bg_conv_norm)    # norm_mode='bn' (by default)
            else:
                self.bg_convs = nn.ModuleList(
                    [conv3d_norm_act(fusion_channel, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), norm_mode=bg_conv_norm) \
                        for _ in range(num_iter + 1)])          # norm_mode='bn'
        else:
            self.predictors = nn.ModuleList(
                [Predictor(fusion_channel, kernel_dim, norm_kernel=(num_learned_masks > 0)) \
                 for _ in range(num_iter)])
            self.kernel_interactors = nn.ModuleList(
                [SelfAttentionBlock(fusion_channel) for _ in range(num_iter)])
            self.kernel_updators = nn.ModuleList(
                [KernelUpdator(fusion_channel, fusion_channel) for _ in range(num_iter)]
            )
            if bg_conv_as_bias:
                self.bg_bias = nn.Parameter(torch.zeros(1, requires_grad=True))
            else:
                self.bg_convs = nn.ModuleList(
                    [conv3d_norm_act(fusion_channel, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1), norm_mode=bg_conv_norm) \
                        for _ in range(num_iter + 1)])          # norm_mode='bn'


    def forward(self, x, mask_features, pred_affinity, init_masks=None):
        mask_features = F.normalize(mask_features, p=2, dim=1)

        if self.bg_conv_as_bias:
            bg_mask = torch.ones_like(mask_features[:,:1]) * self.bg_bias
        elif self.bg_conv_share:
            bg_mask = self.bg_conv(mask_features)
        else:
            bg_mask = self.bg_convs[-1](mask_features)

        pred_masks_list = []
        pred_kernel_list = []

        # iter-0: get init masks with sparse inst-style head
        # img_features, mask_features: BxCxDxHxW, pred_kernels: BxNxC, pred_masks: BxNxDxHxW
        img_features, obj_feat, pred_masks, pred_kernel, recorded_init_masks \
                = self.rpn(x, mask_features, pred_affinity, init_masks=init_masks)
        pred_masks_list.append(torch.cat([bg_mask, pred_masks], dim=1))
        pred_kernel_list.append(pred_kernel)

        for i in range(self.num_iter):
            # gather inst. features using masks
            x_feat = torch.einsum('bndhw,bcdhw->bnc', pred_masks.softmax(1), img_features)        # F^K


            if self.share_weights:
                # kernel update: generate new obj_feat
                obj_feat = self.kernel_updator(x_feat, obj_feat)
                # kernel interaction: self-attention -> FFN
                obj_feat = self.kernel_interactor(obj_feat)
                pred_masks, pred_kernel = self.predictor(obj_feat, mask_features, return_kernel=True)
            else:
                # kernel update: generate new obj_feat
                obj_feat = self.kernel_updators[i](x_feat, obj_feat)       # BxNxC
                # kernel interaction: self-attention -> FFN
                obj_feat = self.kernel_interactors[i](obj_feat)
                pred_masks, pred_kernel = self.predictors[i](obj_feat, mask_features, return_kernel=True)

            # bg_mask = self.bg_convs[i](mask_features)
            if self.bg_conv_as_bias:
                bg_mask = torch.ones_like(mask_features[:,:1]) * self.bg_bias
            elif self.bg_conv_share:
                bg_mask = self.bg_conv(mask_features)
            else:
                bg_mask = self.bg_convs[i](mask_features)

            pred_masks_list.append(torch.cat([bg_mask, pred_masks], dim=1))
            pred_kernel_list.append(pred_kernel)

        return pred_masks_list, pred_kernel_list, recorded_init_masks


class MaskBranch(nn.Module):

    def __init__(self, 
            num_convs=4, 
            kernel_dim=64,
            in_filter=64, 
            out_filter=64, 
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            shared_kwargs={}
        ):
        super().__init__()

        in_filters = [in_filter] + [out_filter] * (num_convs - 1)
        self.mask_convs = nn.Sequential(*[
            conv3d_norm_act(f, out_filter, kernel_size=kernel_size,
                    padding=padding, **shared_kwargs) \
            for f in in_filters
        ])
        self.projection = nn.Conv3d(out_filter, kernel_dim, 
                kernel_size=1, padding=0)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv3d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)


class AGQ(UNet3D, E2EMixin):
    """
    Image -> UNet3D -> Fusion -> inst_convs -> [iam] -> [kernel], [score]
                           \                               \ 
                            \--> mask_convs -------------> (*) --> [masks]
    """
    def __init__(self,
                 block_type='residual',
                 in_channel: int = 1,
                 out_channel: int = 3,
                 filters: List[int] = [28, 36, 48, 64, 80],
                 is_isotropic: bool = False,
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 init_mode: str = 'orthogonal',
                 pooling: bool = False,
                 blurpool: bool = False,
                 return_feats: Optional[list] = None,
                 use_matchness: bool = False,
                 with_affinity: bool = False,
                 num_iter: int = 2,
                 affinity_convs: int = 4,
                 affinity_for_mask: bool = False,
                 init_mask_method: str = 'connected_components',
                 num_masks: int = 100,
                 num_learned_masks: int = 100,
                 return_init_mask: bool = False,
                 inference_without_bg: bool = False,
                 inst_decoder_share_weights: bool = False,
                 inst_decoder_bg_conv_as_bias: bool = False,
                 inst_decoder_bg_conv_share: bool = False,
                 inst_decoder_bg_conv_norm: str = 'bn',
                 **kwargs):
        super(UNet3D, self).__init__()

        self.depth = len(filters)
        self.do_return_feats = (return_feats is not None)
        self.return_feats = return_feats
        print(f"Return feature maps from 3D U-Net? {self.do_return_feats}")

        if is_isotropic:
            isotropy = [True] * self.depth
        assert len(filters) == len(isotropy)

        block = self.block_dict[block_type]
        self.pooling, self.blurpool = pooling, blurpool
        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        # input and output layers
        kernel_size_io, padding_io = self._get_kernal_size(
            is_isotropic, io_layer=True)
        self.conv_in = conv3d_norm_act(in_channel, filters[0], kernel_size_io,
                                padding=padding_io, **self.shared_kwargs)

        # encoding path
        self.down_layers = nn.ModuleList()
        for i in range(self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[i])
            previous = max(0, i-1)
            stride = self._get_stride(isotropy[i], previous, i)
            layer = nn.Sequential(
                self._make_pooling_layer(isotropy[i], previous, i),
                conv3d_norm_act(filters[previous], filters[i], kernel_size,
                                stride=stride, padding=padding, **self.shared_kwargs),
                block(filters[i], filters[i], **self.shared_kwargs))
            self.down_layers.append(layer)

        # decoding path
        self.up_layers = nn.ModuleList()
        for j in range(1, self.depth):
            kernel_size, padding = self._get_kernal_size(isotropy[j])
            layer = nn.ModuleList([
                conv3d_norm_act(filters[j], filters[j-1], kernel_size,
                                padding=padding, **self.shared_kwargs),
                block(filters[j-1], filters[j-1], **self.shared_kwargs)])
            self.up_layers.append(layer)

        # task branches
        fusion_channel = 64
        fusion_level = 1

        self.fusion_block = FusionBlock(
            filters, fusion_level, 
            fusion_channel, fusion_channel, 
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            shared_kwargs=self.shared_kwargs
        )

        # initialization
        model_init(self, mode=init_mode)

        self.shared_kwargs['norm_mode'] = 'none'
        self.inst_decoder = InstDecoder(fusion_channel, num_iter, init_mask_method, num_masks, num_learned_masks, return_init_mask, 
                                        share_weights=inst_decoder_share_weights,
                                        bg_conv_as_bias=inst_decoder_bg_conv_as_bias,
                                        bg_conv_share=inst_decoder_bg_conv_share,
                                        bg_conv_norm=inst_decoder_bg_conv_norm)
        self.mask_branch = MaskBranch(
            num_convs=4, kernel_dim=64, 
            in_filter=fusion_channel+3, out_filter=fusion_channel, 
            kernel_size=(1, 3, 3), padding=(0, 1, 1),
            shared_kwargs=self.shared_kwargs
        )

        self.with_affinity = with_affinity
        if self.with_affinity:
            self.affinity_branch = MaskBranch(
                num_convs=affinity_convs, kernel_dim=3, 
                in_filter=fusion_channel+3, out_filter=fusion_channel, 
                kernel_size=(3, 3, 3), padding=(1, 1, 1),
                shared_kwargs=self.shared_kwargs
            )

        self.inference_without_bg = inference_without_bg


    @staticmethod
    def parse_config(cfg, kwargs):
        kwargs['num_iter'] = cfg.MODEL.NUM_ITER
        kwargs['with_affinity'] = cfg.MODEL.WITH_AFFINITY
        kwargs['affinity_convs'] = cfg.MODEL.AFFINITY_CONVS
        kwargs['init_mask_method'] = cfg.MODEL.INIT_MASK_METHOD
        kwargs['num_masks'] = cfg.MODEL.NUM_MASKS
        kwargs['num_learned_masks']= cfg.MODEL.NUM_LEARNED_MASKS
        kwargs['return_init_mask']= cfg.MODEL.RETURN_INIT_MASK
        kwargs['inference_without_bg']= cfg.MODEL.INFERENCE_WITHOUT_BG
        kwargs['inst_decoder_share_weights']= cfg.MODEL.INST_DECODER_SHARE_WEIGHTS
        kwargs['inst_decoder_bg_conv_share']= cfg.MODEL.INST_DECODER_BG_CONV_SHARE
        kwargs['inst_decoder_bg_conv_as_bias'] = cfg.MODEL.INST_DECODER_BG_CONV_AS_BIAS
        kwargs['inst_decoder_bg_conv_norm']= cfg.MODEL.INST_DECODER_BG_CONV_NORM
        return kwargs


    def _forward_backbone(self, x):
        x = self.conv_in(x)

        down_x = [None] * (self.depth-1)
        for i in range(self.depth-1):
            x = self.down_layers[i](x)
            down_x[i] = x

        out = []
        x = self.down_layers[-1](x)
        out.append(x)

        for j in range(self.depth-1):
            i = self.depth-2-j
            x = self.up_layers[i][0](x)
            x = self._upsample_add(x, down_x[i])
            x = self.up_layers[i][1](x)
            out.append(x)

        return out[::-1]    # P0, P1, P2, P3, P4


    @torch.no_grad()
    def compute_coordinates(self, x):
        d, h, w = x.size(2), x.size(3), x.size(4)
        z_loc = torch.linspace(-1, 1, d, device=x.device)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        
        z_loc, y_loc, x_loc = torch.meshgrid(z_loc, y_loc, x_loc)
        z_loc = z_loc.expand([x.shape[0], 1, -1, -1, -1])
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1, -1])

        locations = torch.cat([z_loc, x_loc, y_loc], 1)
        return locations.to(x)


    def forward(self, x, label=None):

        # backbone
        feats = self._forward_backbone(x)
        x = self.fusion_block(feats)

        # feature extraction
        coord_features = self.compute_coordinates(x)
        x = torch.cat([coord_features, x], dim=1)

        # affinity prediction
        if self.with_affinity:
            pred_affinity = self.affinity_branch(x)
        else:
            pred_affinity = None

        # instance segmentation
        mask_features = self.mask_branch(x)     # B, C, D, H, W
        pred_masks, pred_kernels, recorded_init_masks = self.inst_decoder(x, mask_features, pred_affinity)

        if self.inference_without_bg:
            pred_masks[-1][0,0].zero_()     # remove Background prediction

        output = {
            "pred_masks": pred_masks[-1],
            "pred_kernel": pred_kernels[-1],
            "pixel_feature": mask_features,
            "aux_outputs": [
                {'pred_masks': m, 'pred_kernel': k} \
                    for m, k in zip(
                        pred_masks[:-1], 
                        pred_kernels[:-1]
                    )
            ],
            "recorded_init_masks": recorded_init_masks      # None if not set
        }

        if self.with_affinity:
            output['pred_affinity'] = pred_affinity

        return output
