# Author: An Jiaoyang
# 12.24 21:16 
# =============================
"""
根据 ground truth, 对每个 anchor box 分配正负样本.

1. 根据预测特征图大小, 生成 anchor boxes ,共计 [h, w, k] 个
2. 忽略部分区域落在 image 之外的 anchor boxes, 共计 [m] 个
3. 计算每个 anchor boxes 与每个 gt boxes 的 IoU, 得到 [n, m, b] 个 IoU 结果
4. 创建 labels, 表示每个 anchor boxes 负责预测的类别, 1(前景), 0(背景), -1(忽略)
   (1) 保证每个 gt box 有与之匹配的 anchor box, 即使两者 IoU 阈值小于正样本阈值
   (2) 大于正样本阈值的 anchor box 为正, 小于负样本阈值的 anchor box 为负, 忽略其他 anchor box, 记为 -1
5. 根据配置文件, RPN 阶段训练的正负样本总数为 num, 正负样本比例为 r, 根据 num 和 r 调整每个 image 中的正负样本的数量
   (1) 如果超过配置中的最大数量, faster r-cnn 的策略是 "随机丢弃" 部分正/负样本, 以满足要求的数量
   (2) 这里应该可以有更好的采样办法, 比如丢弃简单样本, 相似的样本等...
6. 计算回归的 ground truth, (dx, dy, dw, dh)
7. 计算正样本 mask, 正负样本损失权重 mask
8. 将上述 labels, 回归 gt, 正样本 mask, 权重 mask 的大小变换为与网络的输出 tensor 相同的形式, [n, c, h, w]
9. 返回 8. 中的四个值
"""
import torch
from config.base import cfg
from model.rpn.anchor_box import create_anchors, create_anchor_boxes
from model.utils.box_utils import boxes_iou_batch, boxes_encode_batch


class AnchorMatchLayer(object):
    def __init__(self):
        self.base_anchors = torch.from_numpy(create_anchors()).float()  # 预测特征图 (0, 0) 点的 anchor

        # 是否保留坐标在图像之外, allowed_border 范围内的预测 box
        self.allowed_border = 0.0

    def __call__(self, pred_prob, gt_boxes, im_info):
        # pred_prob 没有参与计算, 只是获取了其部分属性

        n, _, h, w = pred_prob.size()
        b = gt_boxes.size(1)  # gt boxes 个数
        k = self.base_anchors.size(0)  # base anchor 数量, 即论文中的 k

        # 生成 anchor box
        device = pred_prob.device
        anchors = create_anchor_boxes(self.base_anchors, (h, w), device).squeeze(dim=0)  # [1, hwk, 4] -> [hwk, 4]
        hwk = anchors.size(0)  # anchor boxes 总数量

        # 忽略落在 (图像区域 + allowed_border) 之外的 anchor box. keep 为 保留的 mask, shape: [hwk]
        keep = ((anchors[:, 0] >= -self.allowed_border) &
                (anchors[:, 1] >= -self.allowed_border) &
                (anchors[:, 2] <= (im_info[0][1] + self.allowed_border)) &  # w
                (anchors[:, 3] <= (im_info[0][0] + self.allowed_border)))  # h

        # 保留落在 image 内的 anchor boxes
        idx_inside = torch.nonzero(keep).view(-1)  # 保留项的 index, [m, 1] -> [m]
        anchors = anchors[idx_inside, :]  # [m, 4]
        m = anchors.size(0)  # 保留的 anchor boxes 数量

        # 初始化 anchor boxes 的 label. 1->正样本, 0->负样本, -1->忽略
        labels = gt_boxes.new_full(size=[n, m], fill_value=-1)  # [n, m]
        box_inside_weights = gt_boxes.new_zeros(n, m)  # [n, m]
        box_outside_weights = gt_boxes.new_zeros(n, m)  # [n, m]

        # 计算 anchor boxes 与 gt boxes 的 IoU
        iou = boxes_iou_batch(anchors, gt_boxes)  # [n, m, b], b 为 gt_boxes 数量

        # 计算每个 anchor box 与哪个 gt boxes 的 IoU 最大
        iou_max, iou_max_idx = torch.max(iou, dim=2)  # [n, m]

        # 计算每个 gt boxes 所有 anchor boxes 的 IoU 最大, 保证每个 gt boxes 都有 anchor boxes 对应
        gt_iou_max, _ = torch.max(iou, dim=1)  # [n, b]

        # 是否使用严格的正样本 anchor box 条件, 默认为 False, 即: 即使 IoU < 正样本阈值, 也能保证每个 gt box 都有匹配的 anchor box
        if not cfg.train.rpn_clobber_positives:
            labels[iou_max < cfg.train.rpn_negative_overlap] = 0

        # gt_iou_max == 0 表示用来 padding 的 gt boxes, 设为 1e-5 是为了过滤掉对应的 gt box
        gt_iou_max[gt_iou_max == 0] = 1e-5

        # 计算每个 anchor box 是几个 gt box 的最佳匹配(匹配的含义是: 对于任意 gt box, 和其有最大 IoU 的 anchor box)
        keep = torch.sum(iou == gt_iou_max.view(n, 1, b).expand(n, m, b), dim=2)  # [n, m]

        # sum(keep) = 0 表示所有的 anchor box 与所有的 gt box 的 IoU 都为 0
        if torch.sum(keep) > 0:
            # 和每个 gt box 有最大 IoU 的 anchor box 为正样本(即: 不考虑 IoU 阈值, 保证每个 gt box 至少与一个 anchor box 匹配)
            labels[keep > 0] = 1

        # IoU 大于阈值的 anchor box 为正样本
        labels[iou_max >= cfg.train.rpn_positive_overlap] = 1

        if cfg.train.rpn_clobber_positives:
            labels[iou_max < cfg.train.rpn_negative_overlap] = 0

        # 正样本 anchor box 最大个数
        max_fg = int(cfg.train.rpn_batch_size * cfg.train.rpn_fg_fraction)

        # 每幅图像中正/负样本 anchor box 个数
        sum_fg = torch.sum(labels == 1, dim=1)  # [n]
        sum_bg = torch.sum(labels == 0, dim=1)

        # 确保正负样本比与 cfg 中的配置相符, 如果超过最大样本个数限制, 随机忽略一些 anchor box
        for i in range(n):
            # 如果正样本太多, 随机忽略一部分
            if sum_fg[i] > max_fg:
                fg_idx = torch.nonzero(labels[i] == 1).view(-1)
                num_fg = fg_idx.size(0)  # 正样本个数, 等于 sum_fg[i]
                rand_num = torch.randperm(num_fg).type_as(gt_boxes).long()  # 随机打乱正样本索引
                disable_idx = fg_idx[rand_num[:num_fg-max_fg]]  # 要忽略的 anchor box 索引
                labels[i, disable_idx] = -1  # 忽略

            # 最大负样本个数 = num_max - num_pos
            max_bg = cfg.train.rpn_batch_size - torch.sum(labels[i] == 1)

            # 如果负样本太多, 随机忽略一部分
            if sum_bg[i] > max_bg:
                bg_idx = torch.nonzero(labels[i] == 0).view(-1)
                num_bg = bg_idx.size(0)
                rand_num = torch.randperm(num_bg).type_as(gt_boxes).long()
                disable_idx = bg_idx[rand_num[:num_bg-max_bg]]
                labels[i][disable_idx] = -1

        # 取出每个 anchor box 对应的 gt box
        offset = torch.arange(0, n) * b  # 下标偏移
        offset = offset.view(n, 1).expand(n, m).type_as(iou_max_idx)  # [n, m]
        iou_max_idx = iou_max_idx + offset  # [n, m]
        matched_gt_boxes = gt_boxes.view(n * b, 5)[iou_max_idx.view(n * m), :].view(n, m, 5)  # [n, m, 5]

        # 计算坐标回归的 ground truth
        boxes_target = boxes_encode_batch(anchors, matched_gt_boxes[:, :, 0:4])  # [n, m, 4], (dx, dy, dw, dh)

        # 只有正样本才有坐标回归损失
        box_inside_weights[labels == 1] = 1

        # 计算回归损失的权重项为: 1/样本个数, 设定正负样本权重相同
        avg_examples = torch.sum(labels >= 0).float() / n  # 一幅图像中平均正负样本总数
        positive_weights = 1.0 / avg_examples
        negative_weights = 1.0 / avg_examples

        box_outside_weights[labels == 1] = positive_weights
        box_outside_weights[labels == 0] = negative_weights

        # 构造和原始 anchor map 大小相同的数据, [n, hwk, ...]
        labels = _unmap(data=labels, count=hwk, index=idx_inside, fill=-1)  # [n, hwk]
        boxes_target = _unmap(boxes_target, hwk, idx_inside, fill=0)  # [n, hwk, 4]
        box_inside_weights = _unmap(box_inside_weights, hwk, idx_inside, fill=0)  # [n, hwk]
        box_outside_weights = _unmap(box_outside_weights, hwk, idx_inside, fill=0)  # [n, hwk]

        # 输出
        outputs = []

        labels = labels.view(n, h, w, k).permute(0, 3, 1, 2).contiguous()
        labels = labels.view(n, 1, k * h, w)
        outputs += [labels]  # [n, 1, kh, w]

        boxes_target = boxes_target.view(n, h, w, k * 4).permute(0, 3, 1, 2).contiguous()
        outputs += [boxes_target]  # [n, k4, h, w]

        box_inside_weights = box_inside_weights.view(n, hwk, 1).expand(n, hwk, 4).contiguous()
        box_inside_weights = box_inside_weights.view(n, h, w, k * 4).permute(0, 3, 1, 2).contiguous()
        outputs += [box_inside_weights]  # [n, k4, h, w]

        box_outside_weights = box_outside_weights.view(n, hwk, 1).expand(n, hwk, 4).contiguous()
        box_outside_weights = box_outside_weights.view(n, h, w, k * 4).permute(0, 3, 1, 2).contiguous()
        outputs += [box_outside_weights]  # [n, k4, h, w]

        return outputs


def _unmap(data, count, index, fill=0):
    """根据 index, 将 data 映射回原始大小为 [n, count] 或 [n, count, p] 的 tensor

    data: 要映射回的数据, [n, m] 或 [n, m, p]
    count: 原始 tensor 大小
    index: data 在原始 tensor 的索引, [m]
    fill: 剩余位置的填充值
    """
    n = data.size(0)  # batch size
    if data.dim() == 2:
        ret = data.new_full(size=[n, count], fill_value=fill)
        ret[:, index] = data
    else:
        assert data.dim() == 3
        p = data.size(2)  # data 第 2 维大小
        ret = data.new_full(size=[n, count, p], fill_value=fill)
        ret[:, index, :] = data
    return ret








