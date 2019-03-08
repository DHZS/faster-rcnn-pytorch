# Author: An Jiaoyang
# 12.26 17:33 
# =============================
"""
根据第一阶段预测的 proposals(以下简称为 roi), 生成第二阶段的分类和回归 ground truth.

1. 将 roi 和 gt boxes 合并, 作为新的 roi
2. 计算 roi 与 gt boxes 的 IoU, 根据 IoU 阈值划分正负样本
3. 根据配置中设置的正负样本个数, 采样指定数量的 roi 正负样本. 注意: (1) 可能有重复项 (2) 由于合并, 正样本可能为 gt boxes
4. 计算 3. 中正负样本的 labels, 正样本的回归偏移 gt, 回归权重(同 RPN 中, 包括 inside_weights, outside_weights)
   注意: 如果设置了 train_bbox_normalize_targets_precomputed = True, 回归偏移会进行归一化(间均值, 除标准差)
5. 返回 sampled_rois, gt_labels, reg_target, inside_weights, outside_weights
   (1) sampled_rois:    [n, r, 5], 5 表示 (batch_id, x, y, x, y). r 表示配置中设置的二阶段正负样本总数(rois_per_image),
                        二阶段中采样的正负样本 boxes 坐标.
   (2) gt_labels:       [n, r]. 正负样本的类别标签
   (3) reg_target:      [n, r, 4]. 回归偏移 ground truth, 负样本位置为 0
   (4) inside_weights:  [n, r, 4]. 正样本 mask
   (5) outside_weights: [n, r, 4]. 正样本损失权重, 设为 1
"""
import torch
from config.base import cfg
from model.utils.box_utils import boxes_iou_batch, boxes_encode_batch


class ProposalMatchLayer(object):
    def __init__(self):
        # 类别个数, 包括背景
        self.num_classes = cfg.num_classes
        # 是否使用预计算的归一化项
        self.train_bbox_normalize_targets_precomputed = cfg.train.bbox_normalize_targets_precomputed
        # 样本均值
        self.train_bbox_normalize_means = torch.tensor(cfg.train.bbox_normalize_means, dtype=torch.float32)
        # 样本标准差
        self.train_bbox_normalize_stds = torch.tensor(cfg.train.bbox_normalize_stds, dtype=torch.float32)

    def __call__(self, rois, gt_boxes):
        """
        rois, 即 proposals: [n, top_k, 5], 5 -> (batch_id, x, y, x, y)
        gt_boxes: [n, b, 5], 5 -> (x, y, x, y, label)
        """
        self.train_bbox_normalize_means = self.train_bbox_normalize_means.type_as(gt_boxes)
        self.train_bbox_normalize_stds = self.train_bbox_normalize_stds.type_as(gt_boxes)

        # 将 rois 与 gt boxes 连接在一起???
        gt_boxes_append = gt_boxes.new_zeros(gt_boxes.size())
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, 0:4]

        # 使 rois 包含 gt boxes???
        rois = torch.cat([rois, gt_boxes_append], dim=1)  # [top_k+b, 5]

        rois_per_image = cfg.train.rois_per_image  # 一幅图像 roi 样本个数
        fg_roi_per_image = int(cfg.train.fg_fraction * rois_per_image)
        fg_roi_per_image = 1 if fg_roi_per_image == 0 else fg_roi_per_image  # ???

        output = self._sample_rois(rois, gt_boxes, fg_roi_per_image, rois_per_image)
        gt_labels, sampled_rois, reg_target, inside_weights = output

        # 正样本权重, 设为 1, [n, r, 4]
        outside_weights = (inside_weights > 0).float()

        return sampled_rois, gt_labels, reg_target, inside_weights, outside_weights

    def _compute_reg_target(self, roi, gt_boxes):
        """计算 roi boxes 的回归坐标偏移的 gt 值

        roi:      roi boxes, [n, r, 4], r 表示 rois_per_image
        gt_boxes: gt boxes, [n, r, 4]
        """
        n, r, _ = roi.size()
        target = boxes_encode_batch(roi, gt_boxes)  # 坐标编码, [n, r, 4]

        if self.train_bbox_normalize_targets_precomputed:
            # 回归时, 使用之前统计的均值和标准差对 target 进行标准化
            mean = self.train_bbox_normalize_means.view(1, 1, 4).expand(n, r, 4)
            std = self.train_bbox_normalize_stds.view(1, 1, 4).expand(n, r, 4)
            target = (target - mean) / std

        return target

    def _get_reg_weights(self, reg_target, labels_batch):
        """负样本位置回归偏移设为 0, 生成正负样本 mask

        reg_target:   [n, r, 4]
        labels_batch: [n, r]

        返回:
             new_reg_target, [n, r, 4]
             inside_weights, [n, r, 4]
        """
        n = labels_batch.size(0)  # batch size
        r = labels_batch.size(1)  # rois per image

        # 负样本位置回归偏移设为 0
        new_reg_target = reg_target.new_zeros(n, r, 4)
        # 正负样本 mask
        inside_weights = reg_target.new_zeros(n, r, 4)

        for i in range(n):
            if labels_batch[i].sum() == 0:
                # 没有正样本
                continue
            idx = torch.nonzero(labels_batch[i] > 0).view(-1)  # nf 个正样本 index, [nf]
            if idx.numel() > 0:
                new_reg_target[i, idx, :] = reg_target[i, idx, :]
                inside_weights[i, idx, :] = 1  # 正样本 mask 为 1

        # ????? 上述做法的意义是什么 ??? 下面的方法更简单 ???
        # inside_weights = (labels_batch > 0).type_as(reg_target).float().view(n, r, 1).expand(n, r, 4)
        # new_reg_target = reg_target * inside_weights

        return new_reg_target, inside_weights

    def _sample_rois(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image):
        """对 rois 随机采样, 获得正负样本

        all_rois:          [n, top_k+b, 5], 简写为 [n, k, 5], 5 表示 (?, x, y, x, y)
        gt_boxes:          [n, b, 5], 5 表示 (x, y, x, y, cls_id)
        fg_rois_per_image: 配置文件中设定的最大正样本 roi 数量
        rois_per_image:    配置文件中设定的一幅图像中用于训练的 roi 总数, 记为 r
        """
        # n: batch size, k: roi/proposal 数量, b: 一幅图像 gt boxes 数量
        n, k, b = all_rois.size(0), all_rois.size(1), gt_boxes.size(1)

        iou = boxes_iou_batch(all_rois, gt_boxes)  # [n, k, b]
        # gt_assignment 表示 roi 对应的最大 IoU 的 gt boxes 的 index
        iou_max, gt_assignment = torch.max(iou, dim=2)  # [n, k]

        # 计算 gt boxes 下标
        offset = torch.arange(0, n) * b
        offset = offset.view(n, 1).expand(n, k).type_as(gt_assignment) + gt_assignment  # [n, k]
        offset = offset.view(n*k)  # [nk]

        labels = gt_boxes[:, :, 4].contiguous().view(-1)  # [b]
        labels = labels[offset].view(n, k)  # [n, k]

        # roi 训练样本 label 的 gt 值
        gt_labels_batch = labels.new_zeros(n, rois_per_image)
        # roi 训练样本的 roi boxes
        rois_batch = all_rois.new_zeros(n, rois_per_image, 5)
        # roi 训练样本的 gt boxes. 负样本对应的 gt boxes 无意义
        gt_boxes_batch = all_rois.new_zeros(n, rois_per_image, 5)

        # 防止一张图像中的正样本 roi 个数超过 fg_rois_per_image
        for i in range(n):
            # 获取正样本 roi 索引
            fg_idx = torch.nonzero(iou_max[i] >= cfg.train.fg_threshold).view(-1)  # [num_fg_roi]
            num_fg_roi = fg_idx.size(0)  # num_fg_roi 个正样本 roi

            # 获取负样本 roi 索引
            bg_idx = torch.nonzero((iou_max[i] >= cfg.train.bg_threshold[0]) & (iou_max[i] < cfg.train.bg_threshold[1]))
            bg_idx = bg_idx.view(-1)  # [num_bg_roi]
            num_bg_roi = bg_idx.size(0)  # num_bg_roi 个负样本 roi

            # 因为正负样本数量可能超出设定值, 所以需要对正负样本进行随机采样
            if num_fg_roi > 0 and num_bg_roi > 0:
                # 采样正样本, 防止正样本数量超过 fg_rois_per_image
                fg_roi_this_img = min(fg_rois_per_image, num_fg_roi)
                rand_num = torch.randperm(num_fg_roi).type_as(gt_boxes).long()
                fg_idx = fg_idx[rand_num[:fg_roi_this_img]]

                # 采样负样本
                # 由于可能存在 bg_roi_this_image > num_bg_roi 的情况,
                # 即需要的负样本 roi 比当前的负样本 roi 数量多, 会出现重复负样本
                # 因此使用生成 (0, 1) 随机数乘以 num_bg_roi 的方法采样负样本
                bg_roi_this_image = rois_per_image - fg_roi_this_img
                # shape [bg_roi_this_image], 取值范围 [0, num_bg_roi - 1]
                rand_num = torch.floor(torch.rand(bg_roi_this_image) * num_bg_roi).type_as(gt_boxes).long()
                bg_idx = bg_idx[rand_num]

            elif num_fg_roi > 0 and num_bg_roi == 0:
                # 没有负样本的情况, 只随机采样正样本. 可能有重复项
                rand_num = torch.floor(torch.rand(rois_per_image) * num_fg_roi).type_as(gt_boxes).long()
                fg_idx = fg_idx[rand_num]
                fg_roi_this_img = rois_per_image
                bg_roi_this_image = 0

            elif num_bg_roi > 0 and num_fg_roi == 0:
                # 没有正样本的情况, 只随机采样负样本, 可能有重复项
                rand_num = torch.floor(torch.rand(rois_per_image) * num_bg_roi).type_as(gt_boxes).long()
                bg_idx = bg_idx[rand_num]
                fg_roi_this_img = 0
                bg_roi_this_image = rois_per_image

            else:
                raise ValueError()

            # 合并正负样本索引
            keep_idx = torch.cat([fg_idx, bg_idx], dim=0)  # [r]

            # 创建 target labels
            gt_labels_batch[i].copy_(labels[i, keep_idx])  # 为什么用 copy?

            # 设置负样本的 label 为 0
            if fg_roi_this_img < rois_per_image:
                gt_labels_batch[i, fg_roi_this_img:] = 0

            rois_batch[i] = all_rois[i, keep_idx]
            rois_batch[i, :, 0] = i

            gt_boxes_batch[i] = gt_boxes[i, gt_assignment[i, keep_idx]]

        # 计算回归偏移, [n, r, 4]
        reg_target = self._compute_reg_target(rois_batch[:, :, 1:5], gt_boxes_batch[:, :, 0:4])

        # 获得正样本 weights/mask
        reg_target, inside_weights = self._get_reg_weights(reg_target, gt_labels_batch)  # [n, r, 4]

        return gt_labels_batch, rois_batch, reg_target, inside_weights

