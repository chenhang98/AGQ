import numpy as np 
from PIL import Image
import h5py
import pickle as pkl
from tabulate import tabulate
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import h5py
import numpy as np
import imageio


def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    try:
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            images.append(np.array(img))
        return np.array(images)
    except Exception as e:
        print("Call imageio.imread:", e)
        return imageio.imread(path)
    

def load(fn):
    if fn.endswith('.npy'):
        return np.load(fn)
    elif fn.endswith('.pkl'):
        return pkl.load(open(fn, 'rb'))
    elif fn.endswith('.tiff') or fn.endswith('.tif'):
        return read_tiff(fn)
    elif fn.endswith('.h5'):
        with h5py.File(fn, 'r') as f:
            ks = list(f.keys())
            assert len(ks) == 1
            return np.array(f[ks[0]])
    else:
        raise NotImplementedError


def print_table(list_dicts):
    keys = list(list_dicts[0].keys())
    table = [[d[k] for k in keys] for i, d in enumerate(list_dicts)]
    print(tabulate(table, headers=keys, floatfmt=".3f", tablefmt="pipe"))


def eval_waterz(fn_dt, fn_gt, ignore_border=25/4.0):
    import waterz
    from waterz.seg_util import create_border_mask

    # use evaluation api from https://github.com/zudi-lin/waterz
    dt = load(fn_dt).astype(np.uint64)
    gt = load(fn_gt).astype(np.uint64)

    # ignore boundary within `ignore_border` voxels
    gt = create_border_mask(gt, ignore_border, np.uint64(0))

    if dt.ndim == 3:
        return waterz.evaluate_total_volume(dt, gt)
    elif dt.ndim == 4:
        out = []
        for i in range(dt.shape[0]):
            out.append(
                waterz.evaluate_total_volume(dt[i], gt))
        return out
    else:
        raise NotImplementedError
    

def _eval_skimage(dt, gt, print_metrics=True):

    from skimage.metrics import adapted_rand_error as adapted_rand_ref
    from skimage.metrics import variation_of_information as voi_ref

    gt_seg = gt
    segmentation = dt.astype(np.int64)

    arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    voi_sum = voi_split + voi_merge

    metrics = {
        'voi_split': voi_split,
        'voi_merge': voi_merge,
        'voi_sum': voi_sum,
        'adapted_RAND': arand
    }
    if print_metrics:
        print('evaluated with skimage api\n', metrics)
    return metrics


def _eval_cremi(dt, gt, ignore_border, print_metrics=True):
    from cremi.evaluation import NeuronIds
    from cremi import Volume

    dt = Volume(dt)
    gt = Volume(gt - 1)       # cremi expects np.uint64(-1) for background, (i.e. self.gt += 1 in NeuronIds.__init__)

    neuron_ids_evaluation = NeuronIds(gt, ignore_border)

    (voi_split, voi_merge) = neuron_ids_evaluation.voi(dt)
    adapted_rand = neuron_ids_evaluation.adapted_rand(dt)

    metrics = {
        'voi_split': voi_split,
        'voi_merge': voi_merge,
        'adapted_RAND': adapted_rand
    }
    if print_metrics:
        print('evaluated with cremi api\n', metrics)
    return metrics


def eval_cremi(fn_dt, fn_gt, ignore_border=0):
    # use evaluation api from https://github.com/cremi/cremi_python/tree/python3
    dt = load(fn_dt).astype(np.uint64)
    gt = load(fn_gt).astype(np.uint64)

    if dt.ndim == 3:
        return _eval_cremi(dt, gt, ignore_border)
    elif dt.ndim == 4:
        out = []
        for i in range(dt.shape[0]):
            out.append(
                _eval_cremi(dt[i], gt, ignore_border))
        print_table(out)
        return out
    else:
        raise NotImplementedError


def eval_skimage(fn_dt, fn_gt):
    from skimage.metrics import adapted_rand_error
    from skimage.metrics import variation_of_information 

    dt = load(fn_dt).astype(np.uint64)
    gt = load(fn_gt).astype(np.uint64)

    if dt.ndim == 3:
        adapted_rand_score = adapted_rand_error(gt, dt)[0]
        voi_split, voi_merge = variation_of_information(gt, dt)
        print('adapted_RAND: {}'.format(adapted_rand_score))
        print('VI (split): {},  VI (merge): {}'.format(voi_split, voi_merge))
        return adapted_rand_score
    elif dt.ndim == 4:
        out = []
        for i in range(dt.shape[0]):
            adapted_rand_score = adapted_rand_error(gt, dt[i])[0]
            voi_split, voi_merge = variation_of_information(gt, dt[i])
            print('slice {}: adapted_RAND = {}'.format(i, adapted_rand_score))
            print('VI (split): {},  VI (merge): {}'.format(voi_split, voi_merge))
            out.append(adapted_rand_score)
        print_table(out)
        return out 
    else:
        raise NotImplementedError
