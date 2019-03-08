# Author: An Jiaoyang
# 11.13 21:17 
# =============================
"""
训练
"""
import random
import numpy as np
import torch
import torch.cuda
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import os
import math
import time
import argparse
from data.loader.loader import get_all_loader_annotations
from data.dataset import Dataset
from data.sampler import RandomBatchSampler
from model.utils import net_utils
from model.nets.vgg16 import Vgg16
from utils import utils
from utils.logger import Logger
from config.base import cfg


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=None, type=str, help='配置文件路径')
    parser.add_argument('--debug', default=False, type=utils.str2bool, help='debug 模式')
    args, _ = parser.parse_known_args()

    cfg.merge_from_file(args.config_path)
    cfg.debug = args.debug


def main():
    logger = Logger(cfg.log_folder, cfg.log_name, use_pprint=True)

    logger.print('Training Model on: {}'.format(cfg.dataset_name))
    logger.print('Using the specified args:')
    logger.print(cfg)

    # 调试模式
    cudnn.deterministic = cfg.cudnn_deterministic if not cfg.debug else True
    cudnn.benchmark = cfg.cudnn_benchmark if not cfg.debug else False
    if cudnn.deterministic:
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        np.random.seed(1)
        random.seed(1)

    use_gpu = torch.cuda.is_available() and cfg.cuda
    device = torch.device('cuda' if use_gpu else 'cpu')

    # 数据读取
    batch_size = cfg.train.batch_size
    annotations = get_all_loader_annotations(print_fn=logger.print, training=True)
    dataset = Dataset(annotations, batch_size, sub_means=True, training=True)
    sampler = RandomBatchSampler(len(dataset), batch_size)
    data_loader = data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    data_iterator = iter(data_loader)

    # 模型
    if cfg.model == 'vgg16':
        model = Vgg16()  # net 用于获取模型参数, model 用于训练
    else:
        model = None
    model.init_weights()  # 网络参数初始化

    # gpu 运行
    model = model.to(device)

    # 优化器, 学习率调度. 注: 先移动模型到 cuda, 再创建优化器
    optimizer = net_utils.get_optimizer(model)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg.train.lr_steps, cfg.train.gamma)

    # 恢复模型参数
    start_iteration = 1
    if cfg.train.resume:
        start_iteration = net_utils.load_model(cfg.train.resume, model, optimizer, lr_scheduler) + 1
        model.freeze_to(cfg.train.freeze_to)  # 固定模型部分层参数
    elif cfg.train.base_net:
        # 使用 base net 模型的参数. strict=False 允许两个模型之间的参数不完全匹配
        net_utils.load_model(cfg.train.base_net, model, strict=False)
        model.freeze_to(cfg.train.freeze_to)  # 固定模型部分层参数
    start_iteration = cfg.train.start_iter if cfg.train.start_iter != -1 else start_iteration

    # 并行
    parallel_model = use_gpu and cfg.multi_gpu
    if parallel_model:
        model = torch.nn.DataParallel(model)
    model.train()  # 训练模式

    # 1 个 epoch 迭代次数
    epoch_size = math.ceil(len(dataset) / batch_size)

    t = time.time()
    for i, iteration in enumerate(range(start_iteration, cfg.train.max_iter + 1), start=1):
        lr_scheduler.step(iteration)

        items = next(data_iterator)
        images, im_info, gt_boxes, num_boxes = [x.to(device) for x in items]

        if i % epoch_size == 0:
            data_iterator = iter(data_loader)  # 迭代完 1 个 epoch

        # 前向
        result = model(images, im_info, gt_boxes, num_boxes)
        rois, pred_cls_prob, pred_loc, rpn_cls_loss, rpn_loc_loss, cls_loss, loc_loss, rois_label = result
        if parallel_model:
            rpn_cls_loss, rpn_loc_loss = rpn_cls_loss.mean(), rpn_loc_loss.mean()
            cls_loss, loc_loss = cls_loss.mean(), loc_loss.mean()
        loss = rpn_cls_loss + rpn_loc_loss + cls_loss + loc_loss
        num_fg_roi = rois_label.ne(0).sum().item()
        num_bg_roi = rois_label.numel() - num_fg_roi

        # 反向
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        if cfg.train.gradient_clipping:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_norm)
        optimizer.step()  # 更新参数

        if iteration % 1 == 0:
            text = 'epoch {}/{}={} || iter {} || fg/bg: {}/{} || rpn_cls: {:.4f} || rpn_loc: {:.4f} || ' \
                   'cls: {:.4f} || loc: {:.4f} || total: {:.4f} {}||time: {} ms ||'.format(
                iteration, epoch_size, math.ceil(iteration / epoch_size),
                (iteration - 1) % epoch_size + 1,
                num_fg_roi, num_bg_roi,
                rpn_cls_loss.item(), rpn_loc_loss.item(), cls_loss.item(), loc_loss.item(), loss.item(),
                '|| norm: {:.4f} '.format(total_norm) if cfg.train.gradient_clipping else '',
                int((time.time() - t) * 1000)
            )
            logger.print(text)
        t = time.time()

        if iteration % 30 == 0:
            torch.cuda.empty_cache()

        # 检查是否需要保存模型
        stop_file = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'stop')
        have_stop = os.path.isfile(stop_file)

        if iteration % cfg.train.save_interval == 0 or have_stop:
            path = os.path.join(cfg.train.save_folder, 'model-{}.pth')
            net_utils.save_model(path.format(iteration),
                                 model.module if use_gpu and cfg.multi_gpu else model,
                                 optimizer, lr_scheduler, iteration)
            if have_stop:
                os.remove(stop_file)
                exit(0)


if __name__ == '__main__':
    init()
    main()
