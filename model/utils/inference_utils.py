# Author: An Jiaoyang
# 1.3 21:07 
# =============================
"""Inference 阶段工具"""
import torch
import numpy as np
from model.utils import box_utils
from config.base import cfg
from model.cpp.faster_rcnn import nms


def _un_normalize_deltas(deltas):
    """对预测的坐标偏移去标准化. 参考 proposal_match_layer 中的 `_compute_reg_target()`"""
    if cfg.train.bbox_normalize_targets_precomputed:
        device = deltas.device
        mean = torch.tensor(cfg.train.bbox_normalize_means, dtype=torch.float32, device=device)
        stddev = torch.tensor(cfg.train.bbox_normalize_stds, dtype=torch.float32, device=device)
        if cfg.class_agnostic:
            deltas = deltas * stddev + mean  # [1, k, 4]
        else:
            (n, k), m = deltas.size()[0: 2], cfg.num_classes
            deltas = deltas.view(n, k, m, 4)
            deltas = deltas * stddev + mean
            deltas = deltas.view(n, k, m * 4)
    return deltas


def process_boxes(im_info, rois, pred_cls, pred_loc):
    """根据预测值, 恢复预测结果. 仅支持 batch_size=1

    pred_cls: 为 SoftMax 的输出, 表示预测概率
    """
    # 获取 boxes, 预测类别概率, 预测坐标偏移
    boxes = rois.detach()[:, :, 1:5]  # [1, k, 4], k=rpn_post_nms_top_k
    scores = pred_cls.detach()  # [1, k, n], n=num_classes
    deltas = _un_normalize_deltas(pred_loc.detach())  # [1, k, (1 or num_cls)*4]

    # 解码 boxes
    boxes = box_utils.boxes_decode(boxes, deltas)  # [1, k, (1 or num_cls)*4]
    boxes = box_utils.clip_boxes(boxes, im_info)

    # 恢复 boxes 坐标为原始图像比例
    boxes = boxes / im_info.detach()[0, 2]

    scores = scores[0]  # [k, n]
    boxes = boxes[0]  # [k, (1 or num_cls)*4]

    return scores, boxes


def nms_all(scores, boxes, score_threshold=0.0):
    """对每个类别 boxes 执行 nms

    scores: [k, n]
    boxes: [k, (num_cls or 1)*4]
    score_threshold: 过滤掉得分低于 score_threshold 的结果
    select_top_score: 每个预测结果取得分最高的, 相当于共 k 个结果, 然后过滤 cls=0 的结果

    返回 list. 长度为 num_classes.
        list 中的元素为: ndarray 类型大小为 [k, 5] 的过滤后的结果. k >= 0, 5 表示 (x, y, x, y, score)
    """
    k = scores.size(0)  # 结果个数
    num_classes = scores.size(1)
    results = [[] for _ in range(num_classes)]  # num_class 个检测结果

    if cfg.test.test_mode == 'select_top_score':
        scores_max, idx = torch.max(scores, dim=1)  # [k]
        if not cfg.class_agnostic:
            boxes = boxes.view(k, num_classes, 4)
            boxes = boxes[torch.arange(k), idx, :]  # [k, 4]

        fg_idx = idx != 0  # 忽略背景
        labels = idx[fg_idx]  # [m], m 表示正样本个数
        scores = scores_max[fg_idx]  # [m]
        boxes = boxes[fg_idx]  # [m, 4]

        # 0 为背景, 忽略
        for i in range(1, num_classes):
            idx = torch.nonzero((labels == i) & (scores > score_threshold)).view(-1)  # 检查 cls_id=i 并且得分大于阈值的检测结果
            if idx.numel() == 0:
                # 返回 [0, 5] 的空结果
                results[i] = np.zeros([0, 5], dtype=np.float32)
            else:
                cls_scores = scores[idx]
                cls_boxes = boxes[idx]
                _, order = torch.sort(cls_scores, dim=0, descending=True)  # [t]
                # 按得分降序排序
                cls_scores = cls_scores[order]
                cls_boxes = cls_boxes[order]
                cls_det = torch.cat([cls_boxes, cls_scores.view(-1, 1)], dim=1)  # [t, 5]
                # nms
                keep = nms(cls_boxes, cls_scores, cfg.test.nms_threshold).long()  # [k]
                cls_det = cls_det[keep, :]  # [k, 5]
                results[i] = cls_det.cpu().numpy()

    else:  # select_top_score == False
        assert cfg.test.test_mode == 'default'
        # 0 为背景, 忽略
        for i in range(1, num_classes):
            # 获取第 i 类得分大于阈值的结果 index
            idx = torch.nonzero(scores[:, i] > score_threshold).view(-1)  # [t]
            if idx.numel() == 0:
                # 返回 [0, 5] 的空结果
                results[i] = np.zeros([0, 5], dtype=np.float32)
            else:
                cls_scores = scores[idx, i]  # [t]
                _, order = torch.sort(cls_scores, dim=0, descending=True)  # [t]
                if cfg.class_agnostic:
                    cls_boxes = boxes[idx, :]  # [t, 4]
                else:
                    cls_boxes = boxes[idx, i * 4:(i + 1) * 4]  # [t, 4]
                # 按得分降序排序
                cls_scores = cls_scores[order]
                cls_boxes = cls_boxes[order]
                cls_det = torch.cat([cls_boxes, cls_scores.view(-1, 1)], dim=1)  # [t, 5]

                # nms
                keep = nms(cls_boxes, cls_scores, cfg.test.nms_threshold).long()  # [k]
                cls_det = cls_det[keep, :]  # [k, 5]
                results[i] = cls_det.cpu().numpy()

    return results


def get_top_k_boxes(det):
    """获取得分前 k 个 检测结果

    det: nms() 的返回值

    返回 数据格式与 nms() 返回值相同
    """
    if cfg.test.max_per_image > 0:
        num_classes = len(det)
        scores = np.concatenate([det[i][:, 4] for i in range(1, num_classes)], axis=0)  # 所有 score 组成一个 array
        if scores.size > cfg.test.max_per_image:
            # 存在多余的检测结果, 进行过滤
            det_ = [[] for _ in range(num_classes)]
            score_threshold = np.sort(scores)[-cfg.test.max_per_image]  # 获取第 max_per_image 个得分
            for i in range(1, num_classes):
                keep = det[i][:, 4] >= score_threshold
                det_[i] = det[i][keep, :]
            return det_
    return det













