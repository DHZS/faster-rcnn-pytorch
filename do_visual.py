# Author: An Jiaoyang
# 1.3 17:01
# =============================
"""
可视化
"""
import random
import numpy as np
import torch
import torch.cuda
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import argparse
from data.loader.loader import get_all_loader_annotations
from data.dataset import Dataset
from model.utils import net_utils, inference_utils
from model.nets.vgg16 import Vgg16
from utils import utils, visualization
from config.base import cfg


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=None, type=str, help='配置文件路径')
    args, _ = parser.parse_known_args()

    cfg.merge_from_file(args.config_path)
    utils.mkdir(cfg.test.output_dir)


def main():
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
    annotations = get_all_loader_annotations(print_fn=print, training=False)
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
    for i in range(num_images):
        items = next(data_iterator)
        images, im_info, gt_boxes, num_boxes = [x.to(device) for x in items]

        # 前向
        with torch.no_grad():
            result = model(images, im_info, gt_boxes, num_boxes)
        rois, pred_cls_prob, pred_loc, _, _, _, _, _ = result

        # 恢复预测结果
        # [k, num_cls], [k, 4]
        im_info, rois, pred_cls_prob, pred_loc = im_info.cpu(), rois.cpu(), pred_cls_prob.cpu(), pred_loc.cpu()
        scores, boxes = inference_utils.process_boxes(im_info, rois, pred_cls_prob, pred_loc)

        # 逐类别 nms
        results = inference_utils.nms_all(scores, boxes, score_threshold=0.05)
        results = inference_utils.get_top_k_boxes(results)

        # 整理成可视化需要的格式
        pred_boxes = np.zeros([0, 4], dtype=np.float32)
        pred_scores = np.zeros([0], dtype=np.float32)
        pred_labels = []

        for j in range(len(results)):
            if len(results[j]) == 0:
                continue
            obj_boxes = results[j]
            pred_boxes = np.concatenate([pred_boxes, obj_boxes[:, 0:4]], axis=0)
            pred_scores = np.concatenate([pred_scores, obj_boxes[:, 4]], axis=0)
            pred_labels += [j] * len(obj_boxes)

        pred_boxes = pred_boxes.reshape(-1, 4)
        pred_scores = pred_scores.reshape(-1)
        pred_labels = np.array(pred_labels, dtype=np.int32).reshape(-1)

        image = dataset.load_image(i)[:, :, ::-1]  # bgr -> rgb
        gt_boxes = np.concatenate([annotations[i]['boxes'], annotations[i]['labels'].reshape(-1, 1)], axis=1)

        visualization.show_prediction(image, (pred_boxes, pred_labels, pred_scores), gt_boxes, add_mean=False)


if __name__ == '__main__':
    init()
    main()
