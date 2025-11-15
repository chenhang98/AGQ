from __future__ import print_function, division
from typing import Optional, Union, List
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_dilation
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

from .data_misc import split_masks

import logging
import torch
import torch.nn.functional as F


LOG_ONCE = False 

def log_once(info):
    global LOG_ONCE
    if not LOG_ONCE:
        logging.warning(info)
    LOG_ONCE = True 


def seg_to_weights(targets, wopts, mask=None, seg=None):
    # input: list of targets
    out = [None]*len(wopts)
    for wid, wopt in enumerate(wopts):
        out[wid] = seg_to_weight(targets[wid], wopt, mask, seg)
    return out


def seg_to_weight(target, wopts, mask=None, seg=None):
    out = [None]*len(wopts)
    foo = np.zeros((1), int)
    for wid, wopt in enumerate(wopts):
        if wopt[0] == '1':  # 1: by gt-target ratio
            dilate = (wopt == '1-1')
            out[wid] = weight_binary_ratio(target.copy(), mask, dilate)
        elif wopt[0] == '2':  # 2: unet weight
            assert seg is not None
            _, w0, w1 = wopt.split('-')
            out[wid] = weight_unet3d(seg, float(w0), float(w1))

        # boundary weight
        elif wopt[0] == '3':
            args = wopt.split('-')
            if len(args) == 3:
                _, width, ratio = args[0], int(args[1]), float(args[2])
            elif len(args) == 1:
                width, ratio = 11, 0.5
            else:
                raise NotImplementedError
            out[wid] = weight_boundary(target, width=(1, width, width), ratio=ratio)

        # area weight
        elif wopt[0] == '4':
            args = wopt.split('-')
            if len(args) == 3:
                _, gamma, ratio = args[0], float(args[1]), float(args[2])
            elif len(args) == 1:
                gamma, ratio = 0.5, 0.5
            else:
                raise NotImplementedError
            out[wid] = weight_area(seg, gamma=gamma, ratio=ratio)

        # boundary weight v4
        elif wopt[0] == '6':
            args = wopt.split('-')
            if len(args) == 3:
                _, width, loss_weight = args[0], int(args[1]), float(args[2])
            elif len(args) == 1:
                width, loss_weight = 11, 10
            else:
                raise NotImplementedError
            out[wid] = weight_boundary_v4(target, width=(1, width, width), loss_weight=loss_weight)

        # boundary weight v2
        elif wopt[0] == '7':
            args = wopt.split('-')
            if len(args) == 3:
                _, alpha, ratio = args[0], int(args[1]), float(args[2])
            elif len(args) == 1:
                alpha, ratio = 11, 0.5
            else:
                raise NotImplementedError
            out[wid] = weight_boundary_v2(target, alpha=alpha, ratio=ratio)

        # boundary weight v3
        elif wopt[0] == '8':
            args = wopt.split('-')
            if len(args) == 3:
                _, alpha, ratio = args[0], int(args[1]), float(args[2])
            elif len(args) == 1:
                alpha, ratio = 11, 0.5
            else:
                raise NotImplementedError
            out[wid] = weight_boundary_v3(target, alpha=alpha, ratio=ratio)

        # area weight v3
        elif wopt[0] == '9':
            args = wopt.split('-')
            if len(args) == 3:
                _, gamma, ratio = args[0], float(args[1]), float(args[2])
            elif len(args) == 1:
                gamma, ratio = 1, 1
            else:
                raise NotImplementedError
            out[wid] = weight_area_v3(seg, gamma=gamma, ratio=ratio)

        else:  # no weight map
            out[wid] = foo
    return out


def weight_area(label, gamma=0.5, ratio=0.5):
    log_once(f"Use area weight: gamma={gamma:.3f}, ratio={ratio:.3f}")

    from skimage.measure import label as skimage_label

    weight_map = []
    for z in range(label.shape[0]):
        conn_label = skimage_label(label[z])
        ids, areas = np.unique(conn_label, return_counts=True)
        
        wm = np.zeros(label[z].shape, dtype=np.float32)
        for t, s in zip(ids, areas):
            wm[conn_label==t] = 1.0 / np.power(s, gamma)
        weight_map.append(wm)
    weight_map = np.stack(weight_map)       # D, H, W

    weight_map /= weight_map.mean()
    weight_map = ratio * weight_map + (1 - ratio)
    return weight_map[None,...]


def weight_area_v3(label, gamma=0.5, ratio=0.5):
    log_once(f"Use area weight v3: gamma={gamma:.3f}, ratio={ratio:.3f}")

    from skimage.measure import label as skimage_label

    weight_map = []
    for z in range(label.shape[0]):
        conn_label = skimage_label(label[z])
        ids, areas = np.unique(conn_label, return_counts=True)
        
        wm = np.zeros(label[z].shape, dtype=np.float32)
        for t, s in zip(ids, areas):
            s = min(max(s, 90), 360) / 90   # 1 ~ 4
            wm[conn_label==t] = 4.0 / np.power(s, gamma) - 1        # 3 ~ 0
        weight_map.append(wm)
    weight_map = np.stack(weight_map)       # D, H, W

    weight_map = ratio * weight_map + 1
    return weight_map[None,...]


def weight_area_v4(label, thresh=1000, loss_weight=10):
    log_once(f"Use area weight v4: thresh={thresh:.3f}, loss_weight={loss_weight:.3f}")

    from skimage.measure import label as skimage_label

    weight_map = []
    for z in range(label.shape[0]):
        conn_label = skimage_label(label[z])
        ids, areas = np.unique(conn_label, return_counts=True)
        
        wm = np.zeros(label[z].shape, dtype=np.float32)
        for t, s in zip(ids, areas):
            if t > 0 and s < thresh:
                wm[conn_label==t] = 1
        weight_map.append(wm)
    weight_map = np.stack(weight_map)       # D, H, W

    weight_map = (loss_weight - 1) * weight_map + 1
    return weight_map[None,...]




def weight_boundary(affinity, width=(1, 11, 11), ratio=0.5):
    log_once(f"Use boundary weight: width={width}, ratio={ratio:.3f}")

    aff_gt_tsr = torch.Tensor(affinity)[:,None]    # 3, 1, D, H, W

    kernel = aff_gt_tsr.new_ones((1, 1, *width))
    bias = aff_gt_tsr.new_zeros((1,))

    pad = []
    for w in width[::-1]:   # F.pad is applied from last to fist
        pad.append(w//2)
        pad.append(w//2)

    aff_gt_tsr = F.pad(aff_gt_tsr, pad=pad, mode="reflect")
    boundary_mask = F.conv3d(aff_gt_tsr, kernel, bias, stride=1, padding=0) 

    bml = torch.abs(boundary_mask - np.prod(width))
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms).sqrt().sqrt() / np.sqrt(np.sqrt((np.prod(width)/2)))
    fbmask = fbmask[:,0].numpy()

    weight = ratio * fbmask + (1 - ratio) 
    weight /= weight.mean()

    return weight



def weight_boundary_v2(affinity, alpha=20, ratio=0.5):
    log_once(f"Use boundary weight v2: alpha={alpha}, ratio={ratio:.3f}")

    border_mask = affinity < 0.5        # 3, D, H, W
    border_skeleton = np.zeros_like(border_mask)
    distance_map = np.zeros_like(border_mask, dtype=np.float32)

    # skeletonize
    for i in range(border_mask.shape[0]):
        for j in range(border_mask.shape[1]):
            border_skeleton[i,j] = skeletonize(border_mask[i,j])
            distance_map[i, j] = distance_transform_edt(~border_skeleton[i,j])

    weight_function = lambda d: np.exp(-d / alpha)
    weight_map = weight_function(distance_map)

    weight = ratio * weight_map + (1 - ratio) 
    weight /= weight.mean()

    return weight



def weight_boundary_v3(affinity, alpha=20, ratio=0.5):
    log_once(f"Use boundary weight v3: alpha={alpha}, ratio={ratio:.3f}")

    border_mask = affinity < 0.5        # 3, D, H, W
    border_skeleton = np.zeros_like(border_mask)
    distance_map = np.zeros_like(border_mask, dtype=np.float32)

    # skeletonize
    for i in range(border_mask.shape[0]):
        for j in range(border_mask.shape[1]):
            border_skeleton[i,j] = skeletonize(border_mask[i,j])
            distance_map[i, j] = distance_transform_edt(~border_skeleton[i,j])

    weight_function = lambda d: np.exp(-d / alpha)
    weight_map = weight_function(distance_map)

    weight = ratio * weight_map + 1

    return weight



def weight_boundary_v4(affinity, width=(1, 11, 11), loss_weight=10):
    log_once(f"Use boundary weight: width={width}, loss_weight={loss_weight:.3f}")

    aff_gt_tsr = torch.Tensor(affinity)[:,None]    # 3, 1, D, H, W

    kernel = aff_gt_tsr.new_ones((1, 1, *width))
    bias = aff_gt_tsr.new_zeros((1,))

    pad = []
    for w in width[::-1]:   # F.pad is applied from last to fist
        pad.append(w//2)
        pad.append(w//2)

    aff_gt_tsr = F.pad(aff_gt_tsr, pad=pad, mode="reflect")
    boundary_mask = F.conv3d(aff_gt_tsr, kernel, bias, stride=1, padding=0) 

    bml = torch.abs(boundary_mask - np.prod(width))
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms).sqrt().sqrt() / np.sqrt(np.sqrt((np.prod(width)/2)))
    fbmask = fbmask[:,0].numpy()

    weight = (loss_weight - 1) * fbmask + 1
    return weight




def weight_binary_ratio(label, mask=None, dilate=False):
    if label.max() == label.min():
        # uniform weights for single-label volume
        return np.ones_like(label, np.float32)

    min_ratio = 5e-2
    label = (label != 0).astype(np.float64)  # foreground
    if mask is not None:
        mask = mask.astype(label.dtype)[np.newaxis, :]
        ww = (label*mask).sum() / mask.sum()
    else:
        ww = label.sum() / np.prod(label.shape)
    ww = np.clip(ww, a_min=min_ratio, a_max=1-min_ratio)
    weight_factor = max(ww, 1-ww)/min(ww, 1-ww)

    if dilate:
        N = label.ndim
        assert N in [3, 4]
        struct = np.ones([1]*(N-2) + [3, 3])

        label = (label != 0)
        label = binary_dilation(label, struct).astype(np.float64)

    # Case 1 -- Affinity Map
    # In that case, ww is large (i.e., ww > 1 - ww), which means the high weight
    # factor should be applied to background pixels.

    # Case 2 -- Contour Map
    # In that case, ww is small (i.e., ww < 1 - ww), which means the high weight
    # factor should be applied to foreground pixels.

    if ww > 1-ww:
        # switch when foreground is the dominate class
        label = 1 - label
    weight = weight_factor*label + (1-label)

    if mask is not None:
        weight = weight*mask

    return weight.astype(np.float32)


def weight_unet3d(seg, w0=10.0, w1=5.0, sigma=5):
    out = np.ones_like(seg).astype(np.float32)
    zid = np.where((seg > 0).max(axis=1).max(axis=1) > 0)[0]
    for z in zid:
        out[z] = weight_unet2d(seg[z], w0, w1, sigma)
    return out[np.newaxis]


def weight_unet2d(seg, w0=10.0, w1=5.0, sigma=5):
    min_val = 1.0
    max_val = max(w0, w1)

    masks = split_masks(seg)
    N, H, W = masks.shape
    if N < 2:  # Number of foreground segments is smaller than 2.
        weight_map = (seg != 0).astype(np.float32) * w1
        return np.clip(weight_map, min_val, max_val)

    distance = []
    foreground = np.zeros((H, W), dtype=np.uint8)
    for i in range(N):
        binary = (masks[i] != 0).astype(np.uint8)
        foreground = np.maximum(foreground, binary)
        dist = distance_transform_edt(1-binary)
        distance.append(dist)

    distance = np.stack(distance, 0)
    distance = np.partition(distance, 1, axis=0)
    d1 = distance[0, :, :]
    d2 = distance[1, :, :]
    weight_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    weight_map = weight_map * (1-foreground).astype(np.float32)
    weight_map += foreground.astype(np.float32) * w1

    return np.clip(weight_map, min_val, max_val)
