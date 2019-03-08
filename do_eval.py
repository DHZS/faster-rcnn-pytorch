# Author: An Jiaoyang
# 1.3 17:01
# =============================
"""
测试
"""
import random
import numpy as np
import torch
import torch.cuda
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import os
import pickle
import time
import argparse
from data.loader.loader import get_all_loader_annotations
from data.loader import pascal_voc, coco
from data.eval.pascal_voc import evaluate_detections
from data.dataset import Dataset
from model.utils import net_utils, inference_utils
from model.nets.vgg16 import Vgg16
from utils.logger import Logger
from utils import utils
from config.base import cfg


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=None, type=str, help='配置文件路径')
    args, _ = parser.parse_known_args()

    cfg.merge_from_file(args.config_path)
    utils.mkdir(cfg.test.output_dir)


def main():
    logger = Logger(cfg.log_folder, cfg.log_name, use_pprint=True)

    logger.print('Evaluating Model on: {}'.format(cfg.dataset_name))
    logger.print('Using the specified args:')
    logger.print(cfg)

    # 调试模式
    cudnn.deterministic = cfg.cudnn_deterministic
    cudnn.benchmark = cfg.cudnn_benchmark
    if cudnn.deterministic:
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        np.random.seed(1)
        random.seed(1)

    use_gpu = torch.cuda.is_available() and cfg.cuda
    device = torch.device('cuda' if use_gpu else 'cpu')

    # 数据读取
    annotations = get_all_loader_annotations(print_fn=logger.print, training=False)
    dataset = Dataset(annotations, batch_size=1, sub_means=True, training=False)
    data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    data_iterator = iter(data_loader)

    # 模型
    if cfg.model == 'vgg16':
        model = Vgg16()  # net 用于获取模型参数, model 用于训练
    else:
        model = None

    model = model.to(device)  # gpu 运行
    net_utils.load_model(cfg.test.model, model, use_gpu=use_gpu)  # 恢复模型参数
    model.eval()  # 测试模式

    num_images = len(dataset)
    num_classes = (len(coco.CLASSES), len(pascal_voc.CLASSES))[cfg.test.metric == 'voc']
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]  # [cls, img]

    eval_time = time.time()
    for i in range(num_images):
        items = next(data_iterator)
        images, im_info, gt_boxes, num_boxes = [x.to(device) for x in items]

        # 前向
        det_time = time.time()
        with torch.no_grad():
            result = model(images, im_info, gt_boxes, num_boxes)
        rois, pred_cls_prob, pred_loc, _, _, _, _, _ = result

        # 恢复预测结果
        # [k, n], [k, 4]
        im_info, rois, pred_cls_prob, pred_loc = im_info.cpu(), rois.cpu(), pred_cls_prob.cpu(), pred_loc.cpu()
        scores, boxes = inference_utils.process_boxes(im_info, rois, pred_cls_prob, pred_loc)

        det_time = int(1000 * (time.time() - det_time))
        misc_time = time.time()

        # 逐类别 nms
        results = inference_utils.nms_all(scores, boxes)
        results = inference_utils.get_top_k_boxes(results)
        for j in range(len(results)):
            all_boxes[j][i] = results[j]  # [?, 5]

        misc_time = int(1000 * (time.time() - misc_time))
        logger.print('Detecting {}/{} || detection time: {}ms || process time: {}ms'.format(
            i+1, num_images, det_time, misc_time))

    # 保存检测结果
    file_name = '_'.join([cfg.dataset_name[0], 'detections.pkl'])
    file_name = os.path.join(cfg.test.output_dir, file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # with open(file_name, 'rb') as f:
    #     all_boxes = pickle.load(f)
    #
    # # 为什么过滤掉低分的检测结果反而 AP 变低了?
    # for i in range(num_classes):
    #     for j in range(num_images):
    #         if len(all_boxes[i][j]) > 0:
    #             keep = all_boxes[i][j][:, 4] > 0.2
    #             all_boxes[i][j] = all_boxes[i][j][keep, :]

    # 评估
    logger.print('Evaluating detections')
    evaluate_detections(all_boxes, logger.print)

    eval_time = int(1000 * (time.time() - eval_time))
    print('Evaluation time: {}ms'.format(eval_time))


if __name__ == '__main__':
    init()
    main()
