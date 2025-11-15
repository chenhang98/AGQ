import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import autocast, GradScaler
import random
from typing import Optional
from yacs.config import CfgNode
import time
import math
import GPUtil
from collections import defaultdict
import matplotlib

from connectomics.utils.system import get_args, init_devices
from connectomics.config import load_cfg, save_all_cfg
from connectomics.engine import Trainer
from connectomics.engine.base import TrainerBase
from connectomics.engine.solver import *
from connectomics.model import *
from connectomics.utils.monitor import build_monitor
from connectomics.data.augmentation import build_train_augmentor, TestAugmentor
from connectomics.data.dataset import build_dataloader, get_dataset

from connectomics.data.utils import build_blending_matrix, writeh5, get_padsize, array_unpad

import sys 
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cur_dir)

import model
from loss import build_criterion
from config import add_custom_config


def get_lookup(label):
    N = int(np.max(label)) + 1
    lookup = 0.2 + 0.8 * np.random.rand(N, 3)
    lookup[:, 1] = 0.8
    lookup[:, 2] = 0.8
    lookup = matplotlib.colors.hsv_to_rgb(lookup)
    lookup[0, :] = np.ones(3)
    return lookup


class E2ETrainer(Trainer):
    r"""Trainer class for supervised learning.

    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        device (torch.device): model running device. GPUs are recommended for model training and inference.
        mode (str): running mode of the trainer (``'train'`` or ``'test'``). Default: ``'train'``
        rank (int, optional): node rank for distributed training. Default: `None`
        checkpoint (str, optional): the checkpoint file to be loaded. Default: `None`
    """

    def __init__(self,
                 cfg: CfgNode,
                 device: torch.device,
                 mode: str = 'train',
                 rank: Optional[int] = None,
                 checkpoint: Optional[str] = None):
        self.init_basics(cfg, device, mode, rank)

        self.model = build_model(self.cfg, self.device, rank)
        if not rank:
            print(self.model)
            try:
                from mmcv.cnn.utils import get_model_complexity_info
                input_shape = tuple([1] + cfg.MODEL.INPUT_SIZE)
                flops, params = get_model_complexity_info(self.model, input_shape)
                split_line = '=' * 30
                print(f'{split_line}\nInput shape: {input_shape}\nFlops: {flops}\nParams: {params}\n{split_line}')
            except ImportError:
                print('Please install mmcv to get model complexity information.')
        
        if self.mode == 'train':
            self.optimizer = build_optimizer(self.cfg, self.model)
            self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
            self.scaler = GradScaler() if cfg.MODEL.MIXED_PRECESION else None
            self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER
            self.update_checkpoint(checkpoint)

            # stochastic weight averaging
            if self.cfg.SOLVER.SWA.ENABLED:
                self.swa_model, self.swa_scheduler = build_swa_model(
                    self.cfg, self.model, self.optimizer)

            self.augmentor = build_train_augmentor(self.cfg)
            # self.criterion = Criterion.build_from_cfg(self.cfg, self.device)
            self.criterion = build_criterion('panoptic')
            # 
            if self.is_main_process:
                self.monitor = build_monitor(self.cfg)
                self.monitor.load_info(self.cfg, self.model)

            self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
            self.total_time = 0
        else:
            self.update_checkpoint(checkpoint)
            # build test-time augmentor and update output filename
            self.augmentor = TestAugmentor.build_from_cfg(cfg, activation=True)
            if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
                self.test_filename = self.cfg.INFERENCE.OUTPUT_NAME
                self.test_filename = self.augmentor.update_name(self.test_filename)

        self.dataset, self.dataloader = None, None
        if not self.cfg.DATASET.DO_CHUNK_TITLE and not self.inference_singly:
            self.dataloader = build_dataloader(
                self.cfg, self.augmentor, self.mode, rank=rank,
                dataset_options={"ensure_single_connected": True})
            self.dataloader = iter(self.dataloader)
            if self.mode == 'train' and cfg.DATASET.VAL_IMAGE_NAME is not None:
                self.val_loader = build_dataloader(
                    self.cfg, None, mode='val', rank=rank,
                dataset_options={"ensure_single_connected": True})


    def train(self):
        r"""Training function of the trainer class.
        """
        self.model.train()

        for i in range(self.total_iter_nums):
            iter_total = self.start_iter + i
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            # load data
            sample = next(self.dataloader)
            volume = sample.out_input
            target, weight = sample.out_target_l, sample.out_weight_l
            self.data_time = time.perf_counter() - self.start_time

            # prediction
            volume = volume.to(self.device, non_blocking=True)
            with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                pred = self.model(volume, label=target[0])       # also feed gt inst. id label
                loss, losses_vis = self.criterion(pred, target, weight)

            self._train_misc(loss, pred, volume, target, weight,
                             iter_total, losses_vis)

        self.maybe_save_swa_model()


    def _train_misc(self, loss, pred, volume, target, weight,
                    iter_total, losses_vis):
        self.backward_pass(loss)  # backward pass

        # logging and update record
        if hasattr(self, 'monitor'):
            do_vis = self.monitor.update(iter_total, loss, losses_vis,
                                         self.optimizer.param_groups[0]['lr'])
            if do_vis:
                # self.monitor.visualize(
                #     volume, target, pred, weight, iter_total)
                if torch.cuda.is_available():
                    GPUtil.showUtilization(all=True)

        # Save model
        if (iter_total+1) % self.cfg.SOLVER.ITERATION_SAVE == 0:
            self.save_checkpoint(iter_total)

        if (iter_total+1) % self.cfg.SOLVER.ITERATION_VAL == 0:
            self.validate(iter_total)

        # update learning rate
        self.maybe_update_swa_model(iter_total)
        self.scheduler_step(iter_total, loss)

        if self.is_main_process:
            self.iter_time = time.perf_counter() - self.start_time
            self.total_time += self.iter_time
            avg_iter_time = self.total_time / (iter_total+1-self.start_iter)
            est_time_left = avg_iter_time * \
                (self.total_iter_nums+self.start_iter-iter_total-1) / 3600.0
            info = [
                '[Iteration %05d]' % iter_total, 'Data time: %.4fs' % self.data_time,
                'Iter time: %.4fs' % self.iter_time, 'Avg iter time: %.4fs' % avg_iter_time,
                'Time Left %.2fh' % est_time_left,
                'loss: %.3f' % loss.item(),
                *['%s: %.3f' % (k, v) for (k, v) in losses_vis.items()]]
            print(', '.join(info))

        # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        del volume, target, pred, weight, loss, losses_vis


    def _visualize(self, volume, label, pred, weight, iter_total,
                  suffix: Optional[str] = None,
                  additional_image_groups: Optional[dict] = None,
                ) -> None:
        writer = self.monitor.logger.log_tb
        self.monitor.vis.visualize_image_groups(writer, iter_total, additional_image_groups)

        volume = self.monitor.vis._denormalize(volume)
        if isinstance(label, list):
            label = label[0]

        def colorize(x):
            x = x.cpu().numpy()
            x = get_lookup(x)[x]
            x = torch.from_numpy(x).permute(0, 4, 1, 2, 3)   # BDHWC -> BCDHW
            return x

        pred_masks_list = []
        for out in (pred['aux_outputs'] + [pred]):
            pred_masks = F.interpolate(out['pred_masks'], 
                size=volume.shape[-3:], mode="trilinear", align_corners=False)
            pred_masks = pred_masks.detach().argmax(1)
            pred_masks_list.append(colorize(pred_masks))

        label = colorize(label)

        volume = self.monitor.vis.permute_truncate(volume, is_3d=True)
        label = self.monitor.vis.permute_truncate(label, is_3d=True)
        pred_masks_list = [
            self.monitor.vis.permute_truncate(m, is_3d=True) for m in pred_masks_list]

        sz = volume.size()  # z,c,y,x
        canvas = []
        volume_visual = volume.detach().cpu().expand(sz[0], 3, sz[2], sz[3])
        canvas.append(volume_visual)

        def maybe2rgb(temp):
            if temp.shape[1] == 2: # 2d affinity map has two channels
                temp = torch.cat([temp, torch.zeros(
                    sz[0], 1, sz[2], sz[3]).type(temp.dtype)], dim=1)
            return temp

        pred_masks_list_visual = [
            maybe2rgb(m.detach().cpu()) for m in pred_masks_list]
        label_visual = [maybe2rgb(label.detach().cpu())]

        canvas = canvas + pred_masks_list_visual + label_visual
        canvas_merge = torch.cat(canvas, 0)
        canvas_show = torchvision.utils.make_grid(
            canvas_merge, nrow=8, normalize=True, scale_each=True)

        writer.add_image('Consecutive_%s' % suffix, canvas_show, iter_total)


    def validate(self, iter_total):
        r"""Validation function of the trainer class.
        """
        if not hasattr(self, 'val_loader'):
            return

        self.model.eval()
        with torch.no_grad():
            val_losses = defaultdict(float)
            for i, sample in enumerate(self.val_loader):
                volume = sample.out_input
                target, weight = sample.out_target_l, sample.out_weight_l

                # prediction
                volume = volume.to(self.device, non_blocking=True)
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    pred = self.model(volume)
                    loss, losses_vis = self.criterion(pred, target, weight)
                    val_losses['Validation_Loss'] += loss.item()
                    for k, v in losses_vis.items():
                        val_losses[f'{k}_val'] += v.item()

        if hasattr(self, 'monitor'):
            for k, v in val_losses.items():
                self.monitor.logger.log_tb.add_scalar(
                    k, v / len(self.val_loader), iter_total)
            # self.monitor.visualize(volume, target, pred,
            #                        weight, iter_total, suffix='Val')
            self._visualize(volume, target, pred,
                            weight, iter_total, suffix='Val')

        val_loss = val_losses['Validation_Loss'] / len(self.val_loader)
        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = val_loss

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(iter_total, is_best=True)

        # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        del pred, loss, val_loss

        # model.train() only called at the beginning of Trainer.train().
        self.model.train()


def main():
    args = get_args()
    cfg = load_cfg(args, add_cfg_func=add_custom_config)
    device = init_devices(args, cfg)

    if args.local_rank == 0 or args.local_rank is None:
        # In distributed training, only print and save the configurations 
        # using the node with local_rank=0.
        print("PyTorch: ", torch.__version__)
        print(cfg)

        if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
            print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
            os.makedirs(cfg.DATASET.OUTPUT_PATH)
            save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    # start training or inference
    mode = 'test' if args.inference else 'train'
    trainer = E2ETrainer(cfg, device, mode,
                      rank=args.local_rank,
                      checkpoint=args.checkpoint)

    # Start training or inference:
    if cfg.DATASET.DO_CHUNK_TITLE == 0:
        test_func = trainer.test_singly if cfg.INFERENCE.DO_SINGLY else trainer.test
        test_func() if args.inference else trainer.train()
    else:
        trainer.run_chunk(mode)

    print("Rank: {}. Device: {}. Process is finished!".format(
          args.local_rank, device))


if __name__ == "__main__":
    main()
