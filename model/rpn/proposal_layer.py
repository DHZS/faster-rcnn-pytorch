# Author: An Jiaoyang
# 12.22 21:15 
# =============================
"""
对 RPN 的预测结果进行过滤, 生成 proposals.

1. 根据预测特征图大小 (h, w) 生成 anchor box, shape: [n, hwk, 4]
2. 将预测的偏移转换为 bbox 坐标 (xyxy)
3. 根据预测 score 获取 top k1 的 bbox, 过滤部分 bbox
4. 根据 pred_box, pred_score, nms_threshold, 执行 NMS 算法
5. 根据预测 score 获取 top k2 的 bbox, 过滤部分 bbox
6. 返回过滤的 proposals, shape [n, post_nms_top_k, 5], 5 表示 (batch_id, x, y, x, y)
"""
import torch
from config.base import cfg
from model.rpn.anchor_box import create_anchors, create_anchor_boxes
from model.utils.box_utils import boxes_decode, clip_boxes
from model.cpp.faster_rcnn import nms


class ProposalLayer(object):
    def __init__(self):
        self.base_anchors = torch.from_numpy(create_anchors()).float()  # 预测特征图 (0, 0) 点的 anchor

    def __call__(self, pred_prob, pred_loc, im_info, phase):
        """
        pred_prob: 预测的类别概率/得分, shape: [n, k, h, w]
        pred_loc: 预测的坐标偏移, shape: [n, k4, h, w]
        im_info: batch 内 image 大小, shape: [n, 3], 2 表示 (h, w, scale). scale 原始图像到当前图像的缩放比, 用不到
        phase: 阶段 'test' 或 'train'
        """
        n, _, h, w = pred_prob.size()  # batch size
        k = self.base_anchors.size(0)  # base anchor boxes 数量

        # 预测为目标的概率. 注: 前 k 个为背景, 后 k 个为目标
        pred_prob = pred_prob[:, k:, :, :]  # [n, k, h, w]

        # NMS 配置
        pre_nms_top_k = cfg[phase].rpn_pre_nms_top_k
        post_nms_top_k = cfg[phase].rpn_post_nms_top_k
        nms_threshold = cfg[phase].rpn_nms_threshold
        min_size = cfg[phase].rpn_min_size

        # 生成 anchor boxes
        device = pred_prob.device
        anchors = create_anchor_boxes(self.base_anchors, (h, w), device, batch_size=n)  # [n, hwk, 4]

        # 对结转置, 调整 size, 保持和 anchor 的顺序一致
        pred_prob = pred_prob.permute(0, 2, 3, 1).contiguous().view(n, -1)  # [n, h, w, k] -> [n, hwk]
        pred_loc = pred_loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # [n, hwk, 4]

        # bbox 解码: (dx,dy,dh,dw) -> (xyxy)
        proposals = boxes_decode(anchors, pred_loc)  # [n, hwk, 4]
        # 裁剪 proposals 坐标到 image size 范围内
        proposals = clip_boxes(proposals, im_info)

        # 根据得分, NMS 过滤 bboxes
        _, order = torch.sort(pred_prob, dim=1, descending=True)  # [n, hwk]

        # 过滤后的 proposals, shape: [n, top_k, 5], 5 表示 (0 开始的 batch_id, xyxy), 不足的用 0 填充
        output = pred_prob.new_zeros(n, post_nms_top_k, 5)
        for i in range(n):
            # 获取 batch 中每一张图像的结果
            proposals_single = proposals[i]  # [hwk, 4]
            scores_single = pred_prob[i]  # [hwk]
            order_single = order[i]  # [hwk]

            # 获取 NMS 之前 top k 的 bboxes
            if 0 < pre_nms_top_k < pred_prob.numel():
                order_single = order_single[:pre_nms_top_k]
            proposals_single = proposals_single[order_single, :]  # 获取前 k 个 [top_k1, 4], 按 cls 得分从高到低排序
            scores_single = scores_single[order_single]  # [top_k1]

            # NMS 算法
            # 返回保留的 proposals 的 index, 如 [1, 4, 7]
            keep_idx = nms(proposals_single, scores_single, nms_threshold).long()  # shape: [?, ]

            #  获取 NMS 之后 top k 的 bboxes
            if post_nms_top_k > 0:
                keep_idx = keep_idx[:post_nms_top_k]  # 个数不足 top k 时会取所有值
            proposals_single = proposals_single[keep_idx, :]  # [top_k2, 4]
            scores_single = scores_single[keep_idx]  # [top_k2]

            # 将结果加入 output, 空白位置为 0
            num_proposal = proposals_single.size(0)
            output[i, :, 0] = i  # batch_id
            output[i, :num_proposal, 1:] = proposals_single

        return output





