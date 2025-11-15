import pickle as pkl
import numpy as np
import waterz as w
import os
import os.path as osp
from tqdm import tqdm
from skimage.measure import label as skimage_label
import argparse

from scripts import eval
from inference_mp import _index_to_location, count_volume


def zyx_from_fn(filename):
    z, y, x = map(int, os.path.splitext(os.path.basename(filename))[0].split('_'))
    return z, y, x


def get_block_pos_list(sample_volume_size=(17,257,257), volume_size=(100,1024,1024), sample_stride=(16,256,256)):
    volume_size = np.array(volume_size)
    sample_stride = np.array(sample_stride)
    sample_size = count_volume(volume_size, sample_volume_size, sample_stride)

    sample_num = np.prod(sample_size)
    sample_size_test = np.array([np.prod(sample_size[1:]), sample_size[2]])   # y*x, x

    out = []
    for i in range(sample_num):
        pos = _index_to_location(i, sample_size_test)
        for i in range(3):
            if pos[i] != sample_size[i] - 1:
                pos[i] = int(pos[i] * sample_stride[i])
            else:
                pos[i] = int(volume_size[i] - sample_volume_size[i])
        out.append(pos)
    return out


def get_blocks(pkl_path, sample_volume_size=(17,257,257), volume_size=(100,1024,1024), sample_stride=(16,256,256)):
    # note that the blocks must be put on in order
    out = []
    pos_list = get_block_pos_list(sample_volume_size, volume_size, sample_stride)
    for z, y, x in pos_list:
        path = osp.join(pkl_path, f'{z}_{x}_{y}.pkl')
        if osp.exists(path):
            out.append(path)
        else:
            print(f"Warning: {path} not available")
    return out


def reallocate_ids(X, start_id=1):
    """Avoid id conflicts."""
    new_X = np.zeros_like(X)
    mask = X != 0

    non_zero_elements = X[mask]
    unique_elements = np.unique(non_zero_elements)

    new_values = np.arange(start_id, start_id + len(unique_elements))
    mapping = dict(zip(unique_elements, new_values))

    new_X[mask] = np.vectorize(mapping.get)(non_zero_elements)
    return new_X, new_values[-1] + 1


def concat(
        pkl_path='exps/full_pipeline/AC3AC4/outputs/affinity_knet_cc_half_esc', 
        out_path = 'exps/full_pipeline/AC3AC4/outputs/full_affinity_knet_cc_half_esc',
        block_shape=(17,257,257), 
        full_shape=(100,1024,1024),
        sample_stride=(16,256,256)
    ):
    """ Concat all block predictions from pkl_path to out_path / mask_int.pkl. 
    Or only return the concatted predictions if out_path is None.
    """
    D, H, W = full_shape
    d, h, w = block_shape
    if out_path:
        os.makedirs(out_path, exist_ok=True)

    # --------------------------------------------------------------------------
    all_mask_int = np.zeros((D, H, W), dtype=np.int32)
    border_mask  = np.zeros((3, D, H, W), dtype=bool)
    all_affinity = np.zeros((3, D, H, W), dtype=np.float32)
    all_count = np.zeros((D, H, W), dtype=np.float32)
    start_id = 1

    # --------------------------------------------------------------------------
    print('Concatenating mask_int ...')
    all_blocks = get_blocks(pkl_path, block_shape, full_shape, sample_stride)

    for filename in tqdm(all_blocks):
        with open(filename, 'rb') as f:
            mask_int = pkl.load(f)['mask_int']
        z, y, x = zyx_from_fn(filename)

        all_mask_int[z:z+d, y:y+h, x:x+w], start_id = reallocate_ids(mask_int[:d,:h,:w], start_id)
        border_mask[0, z, y:y+h, x:x+w] = True
        border_mask[1, z:z+d, y, x:x+w] = True
        border_mask[2, z:z+d, y:y+h, x] = True

    if out_path:
        with open(osp.join(out_path, 'mask_int.pkl'), 'wb') as f:
            pkl.dump(all_mask_int, f)
    # --------------------------------------------------------------------------

    if out_path:
        with open(osp.join(out_path, 'border_mask.pkl'), 'wb') as f:
            pkl.dump(border_mask, f)

    # --------------------------------------------------------------------------
    print('Concatenating affinity ...')

    for filename in tqdm(os.listdir(pkl_path)):
        with open(osp.join(pkl_path, filename), 'rb') as f:
            try:
                data = pkl.load(f)
                z0, y0, x0 = zyx_from_fn(filename)
                z1, y1, x1 = z0 + d, y0 + h, x0 + w
                all_count[z0:z1, y0:y1, x0:x1] += 1
                all_affinity[:, z0:z1, y0:y1, x0:x1] += data['affinity'][:,:d,:h,:w]
            except Exception as e:
                print(e)

    all_affinity = all_affinity / np.clip(all_count[None], a_max=np.inf, a_min=1e-4)            # TODO: optimize this
    if out_path:
        with open(osp.join(out_path, 'affinity.pkl'), 'wb') as f:
            pkl.dump(all_affinity, f)

    print('Concatenating done.')
    return all_mask_int, border_mask, all_affinity



def getScoreFunc(scoreF):
    # from LSD
    waterz_merge_functions = {
        'hist_quant_10': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>',
        'hist_quant_10_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>',
        'hist_quant_25': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>',
        'hist_quant_25_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>',
        'hist_quant_50': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>',
        'hist_quant_50_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>',
        'hist_quant_75': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>',
        'hist_quant_75_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>',
        'hist_quant_90': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>',
        'hist_quant_90_initmax': 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>',
        'mean': 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>',
    }
    if scoreF in waterz_merge_functions:
        return waterz_merge_functions[scoreF]
    
    # aff50_his256
    config = {x[:3]: x[3:] for x in scoreF.split('_')}
    if 'aff' in config:
        if 'his' in config and config['his']!='0':
            return 'OneMinus<HistogramQuantileAffinity<RegionGraphType, %s, ScoreValue, %s>>' % (config['aff'],config['his'])
        else:
            return 'OneMinus<QuantileAffinity<RegionGraphType, '+config['aff']+', ScoreValue>>'
    elif 'max' in config:
            return 'OneMinus<MeanMaxKAffinity<RegionGraphType, '+config['max']+', ScoreValue>>'
    else:
        return scoreF



def merge(
        mask_int,
        affinity,
        border_mask,
        out_path,
        thresh_border = 0.3,
        thresh_other  = 0.0,
        sf = 'mean'
    ):
    """Merge the concatted predictions (mask_int) into a full prediction using affinity."""
    if type(mask_int) == str:
        mask_int = eval.load(mask_int)
    if type(affinity) == str:
        affinity = eval.load(affinity)
    if type(border_mask) == str:
        border_mask = eval.load(border_mask)
    if border_mask.ndim == 3:
        border_mask = border_mask[None]

    print("preparing merging data ... ")

    # a tricky way to implement two thresholds
    masked_affinity = affinity.copy()
    # this transformation keeps order and range, equivalent to using different thresholds for different regions
    masked_affinity[border_mask] = (masked_affinity[border_mask]  + thresh_border) / 2
    masked_affinity[~border_mask] = (masked_affinity[~border_mask] + thresh_other) / 2
    thresh = 0.5           # a fixed value, don't change

    mask_int = skimage_label(mask_int)
    print(f'Agglomerate using score function {sf}, thresh={thresh:.2f}')

    res = w.agglomerate(
        masked_affinity.astype(np.float32), 
        [float(thresh)], 
        gt=None, 
        fragments=mask_int.astype(np.uint64), 
        aff_threshold_low=0.0001, 
        aff_threshold_high=0.9999, 
        scoring_function=getScoreFunc(sf),
        discretize_queue=256,
    )
    res = list(res)
    mask_int_merged = res[0]

    print("Merging done. ")

    if out_path:
        with open(osp.join(out_path, 'mask_int_merged.pkl'), 'wb') as f:
            pkl.dump(mask_int_merged, f)

    return mask_int_merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Concat, merge and eval the blockwise predictions.")
    parser.add_argument("--pkl_path", type=str)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--thresh_border", type=float, default=0.3)
    parser.add_argument("--thresh_other", type=float, default=0.0)
    parser.add_argument("--image_path", type=str, default="datasets/AC3-AC4/AC3_inputs.h5")
    parser.add_argument("--label_path", type=str, default="datasets/AC3-AC4/AC3_labels.h5")
    parser.add_argument("--block_shape", type=str, default="17,257,257")
    parser.add_argument("--full_shape", type=str, default="100,1024,1024")
    parser.add_argument("--sample_stride", type=str, default="16,256,256")
    args = parser.parse_args()

    block_shape = tuple(map(int, args.block_shape.split(',')))
    full_shape = tuple(map(int, args.full_shape.split(',')))
    sample_stride = tuple(map(int, args.sample_stride.split(',')))

    all_mask_int, border_mask, all_affinity = concat(
        pkl_path=args.pkl_path,
        out_path = args.out_path,
        block_shape=block_shape, 
        full_shape=full_shape,
        sample_stride=sample_stride
    )
    mask_int_merged = merge(
        all_mask_int,
        all_affinity,
        border_mask,
        out_path = args.out_path,
        thresh_border = args.thresh_border,
        thresh_other  = args.thresh_other,
        sf = 'mean'
    )

    image = eval.load(args.image_path)
    label = eval.load(args.label_path)

    # metrics = eval._eval_cremi(mask_int_merged.astype(np.int64), label, 0, print_metrics=False)       # same as _eval_skimage
    metrics = eval._eval_skimage(mask_int_merged.astype(np.int64), label, print_metrics=False)

    ss = list()
    for key, value in metrics.items():
        fv = "{:.3f}".format(value)
        ss.append(key + ": " + fv)
    print(' | '.join(ss))
