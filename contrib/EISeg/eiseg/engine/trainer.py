import time
import os
import random
import logging
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

import tqdm
import cv2
import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
from visualdl import LogWriter

from util.vis import draw_probmap, draw_points
from util.misc import save_checkpoint
from util.distributed import reduce_loss_dict
from paddleseg.utils import get_sys_env, logger
from .optimizer import get_optimizer_lr

env_info = get_sys_env()
info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
info = '\n'.join(['', format('Environment Information', '-^48s')] + info + ['-' * 48])
logger.info(info)
place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info['GPUs used'] else 'cpu'
paddle.set_device(place)


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer_params=None,
                 checkpoint_interval=10,
                 max_interactive_points=0,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points'),
                 max_num_next_clicks=3,
                 prev_mask_drop_prob=0.0,
                 with_prev_mask = False
                 ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks
        self.with_prev_mask = with_prev_mask

        self.prev_mask_drop_prob = prev_mask_drop_prob

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.train_sample = paddle.io.DistributedBatchSampler(self.trainset,
                                                         batch_size=cfg.batch_size,
                                                         shuffle=True,
                                                         drop_last=True)
        self.train_data = paddle.io.DataLoader(self.trainset, batch_sampler=self.train_sample,num_workers=4)

        self.valset = valset
        self.val_sample = paddle.io.DistributedBatchSampler(self.valset,
                                                       batch_size=cfg.batch_size,
                                                       shuffle=False,
                                                       drop_last=True)
        self.val_data = paddle.io.DataLoader(self.valset, batch_sampler=self.val_sample, num_workers=4)
        
        if self.is_master:
            vdlpath = os.path.join(self.cfg.LOGS_PATH, "vdl")
            if not os.path.exists(vdlpath):
                os.makedirs(vdlpath)
            self.sw = LogWriter(vdlpath)

        logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
        logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')
        
        self.lr = optimizer_params['learning_rate']
        backbone_params, other_params = get_optimizer_lr(model)
        self.lr_scheduler1 = paddle.optimizer.lr.MultiStepDecay(
            learning_rate=optimizer_params['learning_rate'],
            milestones=[49, 50, 53], gamma=0.1)
        self.lr_scheduler2 = paddle.optimizer.lr.MultiStepDecay(
            learning_rate=optimizer_params['learning_rate'] * 0.1,
            milestones=[49, 50, 53], gamma=0.1)

        optimizer1 = paddle.optimizer.Adam(learning_rate=self.lr_scheduler1, parameters=other_params)
        optimizer2 = paddle.optimizer.Adam(learning_rate=self.lr_scheduler2, parameters=backbone_params)
        self.optim = [optimizer1, optimizer2]
        
        if self.cfg.weights is not None:
            print(self.cfg.weights)
            model = self._load_weights(model)

        nranks = paddle.distributed.ParallelEnv().nranks
        if nranks == 1:
            self.net = model
        
        else:
            self.net = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized():
                paddle.distributed.init_parallel_env()
                self.net = paddle.DataParallel(model)
            else:
                self.net = paddle.DataParallel(model)

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')
        for epoch in range(start_epoch, num_epochs):
            
            self.training(epoch)
        
            if isinstance(self.optim[0]._learning_rate, paddle.optimizer.lr.LRScheduler):
                self.optim[0]._learning_rate.step()
            if isinstance(self.optim[1]._learning_rate, paddle.optimizer.lr.LRScheduler):
                self.optim[1]._learning_rate.step()

            if validation:
                self.validation(epoch)

    def training(self, epoch, log_iters=10, save_epoch=1):
 
        if self.cfg.distributed:
            self.train_sample.set_epoch(epoch)

        for metric in self.train_metrics:
            metric.reset_epoch_stats()
        log_prefix = 'Train' + self.task_prefix.capitalize()
        
        self.net.train()
        train_loss = 0.0
        for i, batch_data in enumerate(self.train_data):
            global_step = epoch * len(self.train_data) + i
            loss, losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data)

            self.optim[0].clear_grad()
            self.optim[1].clear_grad()
            loss.backward()
            self.optim[0].step()
            self.optim[1].step()
            losses_logging['overall'] = loss
            losses_logging = reduce_loss_dict(losses_logging)
            train_loss += losses_logging['overall']
            lr = self.optim[0].get_lr()

            if self.is_master:
                if global_step % log_iters == 0:
                    logger.info('Epoch={}, Step={}, loss={:.4f}, lr={}'.format(epoch, global_step, float(loss), lr))
                    for loss_name, loss_value in losses_logging.items():

                        self.sw.add_scalar(f'{log_prefix}Losses/{loss_name}', loss_value.numpy(), global_step)
                        self.sw.add_scalar('Train/lr', lr, global_step)
                    for metric in self.train_metrics:
                        metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)
                    
        if self.is_master:
            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval

            if epoch % checkpoint_interval == 0:
                print('saving model .........')
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,
                                epoch=epoch, multi_gpu=self.cfg.multi_gpu)
                print('finish save model!')



            
    def validation(self, epoch, log_iters=10, save_epoch=1):
        log_prefix = 'Val' + self.task_prefix.capitalize()
       
        for metric in self.val_metrics:
            metric.reset_epoch_stats()
        val_loss = 0
        losses_logging = defaultdict(list)
        self.net.eval()
        
        with paddle.no_grad():
            for i, batch_data in enumerate(self.val_data):
                val_global_step = epoch * len(self.val_data) + i

                loss, batch_losses_logging, splitted_batch_data, outputs = \
                    self.batch_forward(batch_data, validation=True)
                batch_losses_logging['overall'] = loss
                reduce_loss_dict(batch_losses_logging)
                for loss_name, loss_value in batch_losses_logging.items():
                    losses_logging[loss_name].append(loss_value.numpy())

                val_loss += batch_losses_logging['overall'].numpy()

                if self.is_master:
                    logger.info(f'Epoch {epoch}, validation loss: {val_loss[0]/(i + 1):.4f}')
                    for metric in self.val_metrics:
                        metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', val_global_step)

            if self.is_master:
                for loss_name, loss_values in losses_logging.items():
                    self.sw.add_scalar(f'{log_prefix}Losses/{loss_name}', np.array(loss_values).mean(), epoch)

                for metric in self.val_metrics:
                    self.sw.add_scalar(f'{log_prefix}Metrics/{metric.name}', metric.get_epoch_value(), epoch)


                if val_global_step % log_iters == 0 and self.is_master:
                    logger.info('Epoch={}, Step={}, loss={:.4f}'.format(epoch, val_global_step, float(loss)))

                    for metric in self.val_metrics:
                        metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', val_global_step)
      

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()
        image, points, gt_mask = batch_data
        orig_gt_mask = gt_mask.clone()
        prev_output = paddle.zeros_like(image, dtype='float32')[:, :1, :, :]
        last_click_indx = None

        with paddle.no_grad():
            num_iters = random.randint(0, self.max_num_next_clicks-1)
            for click_indx in range(num_iters):
                last_click_indx = click_indx
                if not validation:
                    self.net.eval()
                net_input = paddle.concat((image, prev_output), axis=1) if self.with_prev_mask else image
                prev_output = F.sigmoid(self.net(net_input, points)['instances'])
                points = get_next_points(prev_output, orig_gt_mask, points, click_indx + 1)

            if self.with_prev_mask and self.prev_mask_drop_prob > 0 and last_click_indx is not None:
                zero_mask = np.random.random(size=prev_output.shape[0]) < self.prev_mask_drop_prob
                for num, drop in enumerate(zero_mask):
                    if drop == True:
                        prev_output[num] = paddle.zeros_like(prev_output[num])
            if not validation:
                self.net.train()
        batch_data[1] = points
        
        net_input = paddle.concat((image, prev_output), axis=1) if self.with_prev_mask else image
        output = self.net(net_input, points)


        loss = 0.0
        loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                             lambda: (output['instances'], batch_data[2]))
        loss = self.add_loss('instance_aux_loss', loss, losses_logging, validation,
                             lambda: (output['instances_aux'], batch_data[2]))

        if self.is_master:
            with paddle.no_grad():
                for m in metrics:
                    m.update(output['instances'], batch_data[2])
        return loss, losses_logging, batch_data, output

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = paddle.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss

    def save_visualization(self, splitted_batch_data, outputs, global_step, prefix):
        output_images_path = self.cfg.VIS_PATH / prefix
        if self.task_prefix:
            output_images_path /= self.task_prefix

        if not output_images_path.exists():
            output_images_path.mkdir(parents=True)
        image_name_prefix = f'{global_step:06d}'

        def _save_image(suffix, image):
            cv2.imwrite(str(output_images_path / f'{image_name_prefix}_{suffix}.jpg'),
                        image, [cv2.IMWRITE_JPEG_QUALITY, 85])

        images = splitted_batch_data[0]
        points = splitted_batch_data[1]
        instance_masks = splitted_batch_data[2]

        gt_instance_masks = instance_masks.numpy()
        predicted_instance_masks = F.sigmoid(outputs[2]).detach().numpy()
        points = points.detach().numpy()

        image_blob, points = images[0], points[0]
        gt_mask = np.squeeze(gt_instance_masks[0], axis=0)
        predicted_mask = np.squeeze(predicted_instance_masks[0], axis=0)

        image = image_blob.numpy() * 255
        image = image.transpose((1, 2, 0))

        image_with_points = draw_points(image, points[:self.max_interactive_points], (0, 255, 0))
        image_with_points = draw_points(image_with_points, points[self.max_interactive_points:], (0, 0, 255))

        gt_mask[gt_mask < 0] = 0.25
        gt_mask = draw_probmap(gt_mask)
        predicted_mask = draw_probmap(predicted_mask)
        viz_image = np.hstack((image_with_points, gt_mask, predicted_mask)).astype(np.uint8)

        _save_image('instance_segmentation', viz_image[:, :, ::-1])

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at f'{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net, str(checkpoint_path))
        return net

    @property
    def is_master(self):
        return paddle.distributed.ParallelEnv().local_rank == 0


def get_next_points(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.numpy()[:, 0, :, :]
    gt = gt.numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.shape[1] // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

    return points


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = paddle.load(path_to_weights)
    current_state_dict.update(new_state_dict)
    model.set_state_dict(current_state_dict)
    logger.info(f'Load checkpoint from path: {path_to_weights}')