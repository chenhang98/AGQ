import pickle as pkl
import numpy as np
from tqdm import tqdm
import numpy as np 
from PIL import Image
import h5py
import os.path as osp
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import sys
import os
import time
import sys
import argparse
from torch.utils.data import DataLoader
from connectomics.model import update_state_dict

sys.path.insert(0, 'projects/AGQ')
from main import *


def get_voxel_offset(roi):
    x, y, z = roi.attrs['offset']
    dx, dy, dz = roi.attrs['resolution']
    return z // dz, y // dy, x // dx

def count_volume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)

def crop_volume(data, sz, st=(0, 0, 0)):
    # must be (z, y, x) or (c, z, y, x) format 
    assert data.ndim in [3, 4]
    st = np.array(st).astype(np.int32)

    if data.ndim == 3:
        return data[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]
    else: # crop spatial dimensions
        return data[:, st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]

def _index_to_location(index, sz):
    # index -> z,y,x
    # sz: [y*x, x]
    pos = [0, 0, 0]
    pos[0] = int(np.floor(index/sz[0]))
    pz_r = index % sz[0]
    pos[1] = int(np.floor(pz_r/sz[1]))
    pos[2] = pz_r % sz[1]
    return pos

def build_model_from_cfg(cfg, checkpoint, rank=0, mode='test'):
    """build the model and load weights from checkpoint"""
    # build model
    device = "cuda"
    model = build_model(cfg, device)

    # load pre-trained model
    print('Load pretrained checkpoint: ', checkpoint)
    checkpoint = torch.load(checkpoint, map_location=device)
    print('checkpoints: ', checkpoint.keys())

    # update model weights
    pretrained_dict = checkpoint['state_dict']
    pretrained_dict = update_state_dict(cfg, pretrained_dict, mode=mode)
    # model_dict = model.module.state_dict()  # nn.DataParallel
    model_dict = model.state_dict()  # nn.DataParallel

    # show model keys that do not match pretrained_dict
    if not model_dict.keys() == pretrained_dict.keys():
        print("Module keys in model.state_dict() do not exactly "
                        "match the keys in pretrained_dict!")
        for key in model_dict.keys():
            if not key in pretrained_dict:
                print(key)

    # 1. filter out unnecessary keys by name
    pretrained_dict = {k: v for k,
                        v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict (if size match)
    for param_tensor in pretrained_dict:
        if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
            model_dict[param_tensor] = pretrained_dict[param_tensor]
    # 3. load the new state dict
    # model.module.load_state_dict(model_dict)  # nn.DataParallel
    model.load_state_dict(model_dict)  # nn.DataParallel
    model.eval()

    return model


class Dataset:
    def __init__(self, cfg, image):
        self.image = image

        self.data_mean = cfg.DATASET.MEAN
        self.data_std = cfg.DATASET.STD
        self.sample_volume_size = cfg.MODEL.INPUT_SIZE

        self.volume_size = np.array(self.image.shape)
        # self.sample_stride = np.array([x//2 for x in self.sample_volume_size])
        self.sample_stride = np.array(cfg.INFERENCE.STRIDE)
        self.sample_size = count_volume(self.volume_size, self.sample_volume_size, self.sample_stride)

        self.sample_num = np.prod(self.sample_size)
        self.sample_size_test = np.array([np.prod(self.sample_size[1:]), self.sample_size[2]])   # y*x, x

        print("Full volume size: ", self.volume_size)
        print("Sample volume size: ", self.sample_volume_size, "; Sample stride: ", self.sample_stride)
        print("Number of samples: ", self.sample_size, "; total: ", self.sample_num)

    def _process_image(self, x: np.array):
        x = np.expand_dims(x, 0) # (z,y,x) -> (c,z,y,x)
        x = (x - self.data_mean) / self.data_std
        return x
    
    def __getitem__(self, index):
        pos = _index_to_location(index, self.sample_size_test)    # pos: z, y, x;
        for i in range(3):
            if pos[i] != self.sample_size[i] - 1:
                pos[i] = int(pos[i] * self.sample_stride[i])
            else:
                pos[i] = int(self.volume_size[i] - self.sample_volume_size[i])

        out_image = (crop_volume(self.image, self.sample_volume_size, pos)/255.0).astype(np.float32)
        out_image = self._process_image(out_image)

        return pos, out_image

    def __len__(self):
        return self.sample_num


class TestBatch:
    def __init__(self, batch):
        self._handle_batch(*zip(*batch))

    def _handle_batch(self, pos, out_input):
        self.pos = pos
        self.out_input = torch.from_numpy(np.stack(out_input, 0))

    # custom memory pinning method on custom type
    def pin_memory(self):
        self._pin_batch()
        return self

    def _pin_batch(self):
        self.out_input = self.out_input.pin_memory()


def collect_fn(batch):
    return TestBatch(batch)


# inference the model and save the out_path/pred_mask_int and out_path/pred_affinity
def inference(model, dataloader, out_dir, rank, save_float_mask=False, area_thresh=40):
    device = next(model.parameters()).device

    if rank == 0:
        dataloader = tqdm(dataloader)

    for data in dataloader:
        z, y, x = data.pos[0]
        out_path = osp.join(out_dir, f'{z}_{y}_{x}.pkl')

        inputs = data.out_input.to(device)
        with torch.no_grad():
            output = model(inputs)

        out_size = data.out_input.shape[-3:]

        pred_masks = output['pred_masks']
        pred_masks = F.interpolate(pred_masks, size=out_size, 
                        mode='trilinear', align_corners=False)
        
        pred_masks_soft = pred_masks.softmax(1)
        areas = pred_masks_soft.flatten(2).sum(-1)
        # import matplotlib.pyplot as plt
        # plt.hist(areas.cpu().numpy()[0], bins=50)

        pred_masks_soft[areas < area_thresh] = 0
        resized_mask_int = pred_masks_soft[0].cpu().argmax(0).numpy()

        if 'pred_affinity' in output:
            pred_affinity = output['pred_affinity'].sigmoid()
            resized_affinity = F.interpolate(pred_affinity, size=out_size, 
                            mode='trilinear', align_corners=False)
            resized_affinity = resized_affinity.cpu().numpy()[0]
        else:
            resized_affinity = None

        if save_float_mask:
            with open(out_path, 'wb') as f:
                pkl.dump(dict(
                    affinity=resized_affinity,    # 3, D, H, W
                    mask_int=resized_mask_int,    # D, H, W
                    mask_float=pred_masks[0].cpu().numpy(),    # N, D, H, W
                    pos=(z, y, x),
                    patch_size=resized_mask_int.shape
                ), f)
        else:
            with open(out_path, 'wb') as f:
                pkl.dump(dict(
                    affinity=resized_affinity,    # 3, D, H, W
                    mask_int=resized_mask_int,    # D, H, W
                    pos=(z, y, x),
                    patch_size=resized_mask_int.shape
                ), f)
    print("Model inference done")


def read_tiff(path):
    """
    path - Path to the multipage-tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)


def load(fn):
    if fn.endswith('.npy'):
        return np.load(fn)
    elif fn.endswith('.tiff') or fn.endswith('.tif'):
        return read_tiff(fn)
    elif fn.endswith('.h5'):
        with h5py.File(fn, 'r') as f:
            ks = list(f.keys())
            assert len(ks) == 1
            return np.array(f[ks[0]])
    else:
        raise NotImplementedError


def create_dataset(cfg):
    """Inputs are specified via INFERENCE.INPUT_PATH and INFERENCE.IMAGE_NAME."""
    if cfg.INFERENCE.INPUT_PATH:
        image_path = osp.join(cfg.INFERENCE.INPUT_PATH, cfg.INFERENCE.IMAGE_NAME)
    else:
        image_path = cfg.INFERENCE.IMAGE_NAME
    dataset_dict = {
        'image': load(image_path),
    }
    return Dataset(cfg, **dataset_dict)


def processor(rank, world_size, cfg, checkpoint, out_files, save_float_mask=False, area_thresh=40):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    model = build_model_from_cfg(cfg, checkpoint, rank=rank)

    dataset = create_dataset(cfg)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collect_fn,
                             sampler=sampler, num_workers=4, pin_memory=True, prefetch_factor=4)

    inference(model, dataloader, out_files, rank, save_float_mask, area_thresh)

    dist.destroy_process_group()


def get_args():
    parser = argparse.ArgumentParser(description="Model Training & Inference")
    parser.add_argument('--config-file', type=str,
                        help='configuration file (yaml)')
    parser.add_argument('--config-base', type=str,
                        help='base configuration file (yaml)', default=None)
    parser.add_argument('--inference', action='store_true',
                        help='inference mode')
    parser.add_argument('--distributed', action='store_true',
                        help='distributed training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to load the checkpoint')
    parser.add_argument('--manual-seed', type=int, default=None)
    parser.add_argument('--local_world_size', type=int, default=1,
                        help='number of GPUs each process.')
    parser.add_argument('--local_rank', type=int, default=None,
                        help='node rank for distributed training')
    parser.add_argument('--debug', action='store_true',
                        help='run the scripts in debug mode')
    parser.add_argument('--save_float_mask', action='store_true',
                        help='save float mask')
    parser.add_argument('--area_thresh', type=int, default=40,
                        help='area threshold for mask')
    # Merge configs from command line (e.g., add 'SYSTEM.NUM_GPUS 8').
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    cfg = load_cfg(args, add_cfg_func=add_custom_config)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(random.randint(10000, 20000))
    # os.environ['CUDA_VISIBLE_DEVICES'] = '8,9'

    checkpoint = args.checkpoint

    out_dir = cfg.INFERENCE.OUTPUT_PATH
    label_path = cfg.INFERENCE.LABEL_PATH

    print(f'inference results will be saved to {out_dir}')

    os.makedirs(out_dir, exist_ok=True)
    num_processes = torch.cuda.device_count()

    print(f'inference model with {num_processes} GPUs')

    if num_processes > 1:
        mp.spawn(processor,
                args=(num_processes, cfg, checkpoint, out_dir, args.save_float_mask, args.area_thresh),
                nprocs=num_processes,
                join=True)
    else:
        model = build_model_from_cfg(cfg, checkpoint, rank=0)

        dataset = create_dataset(cfg)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collect_fn,
                                num_workers=4, pin_memory=True, prefetch_factor=4)
        inference(model, dataloader, out_dir, 0, args.save_float_mask, args.area_thresh)
