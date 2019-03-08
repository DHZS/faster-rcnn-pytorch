# Author: An Jiaoyang
# 12.21 21:07 
# =============================
"""
RPN 网络

1. 使用 backbone 最后一层特征图进行预测, 得到预测的 cls, loc
2. 根据 cls, loc 生成 tok k 个 proposals
3. 根据生成的 anchor boxes 和给定的 gt boxes 计算 RPN 阶段的 ground truth, 包括标签 labels, 回归偏移 boxes_target
4. 计算 cls 和 loc 的损失
5. 返回 proposals, 两个损失
"""
import torch.nn as nn
import torch.nn.functional as F
from config.base import cfg
from model.rpn.proposal_layer import ProposalLayer
from model.rpn.anchor_match_layer import AnchorMatchLayer
from model.losses import smooth_l1


class RPN(nn.Module):
    def __init__(self, base_net_out_dim):
        super(RPN, self).__init__()
        self.base_net_out_dim = base_net_out_dim  # base net 连接 RPN 特征图的输出通道
        self.anchor_scales = cfg.anchor_scales
        self.anchor_ratios = cfg.anchor_ratios

        # 预测层
        self.num_anchor = len(self.anchor_scales) * len(self.anchor_ratios)
        self.rpn_conv = nn.Conv2d(self.base_net_out_dim, 512, kernel_size=3, padding=1)
        # 注意 padding = 0, 这里和 ssd 不太一样
        self.pred_cls = nn.Conv2d(512, 2 * self.num_anchor, kernel_size=1, padding=0)  # 顺序 [n, 2k, h, w]
        self.pred_loc = nn.Conv2d(512, 4 * self.num_anchor, kernel_size=1, padding=0)  # 顺序 [n, k4, h, w]

        # 对预测的 box 解码, 根据得分, nms 过滤部分结果
        self.rpn_proposal_layer = ProposalLayer()

        # 给每个 anchor box 分配标签, 坐标回归偏移, 损失函数权重
        self.rpn_anchor_match_layer = AnchorMatchLayer()

    def forward(self, x, gt_boxes, im_info):
        x = F.relu(self.rpn_conv(x), inplace=True)
        pred_cls = self.pred_cls(x)  # [n, 2k, h, w]
        pred_loc = self.pred_loc(x)  # [n, k4, h, w]

        # 类别预测 softmax, 获得预测概率, 用于过滤 proposals
        n, _, h, w = pred_cls.size()
        k = self.num_anchor
        pred_prob = F.softmax(pred_cls.view(n, 2, k, h, w), dim=1).view(n, 2*k, h, w)

        # 获取过滤后的 proposals
        phase = 'train' if self.training else 'test'
        # !!!! 关键, 需要 proposals.requires_grad=False. Mask RCNN 中提到, 不需要计算 RoI Align 层 关于 proposals 坐标的偏导
        proposals = self.rpn_proposal_layer(pred_prob.detach(), pred_loc.detach(), im_info, phase)  # [n, top_k, 5]

        # 计算 RPN 的损失
        rpn_cls_loss = 0
        rpn_loc_loss = 0

        if self.training:
            rpn_data = self.rpn_anchor_match_layer(pred_prob.detach(), gt_boxes, im_info)
            # [n, 1, kh, w], [n, k4, h, w], [n, k4, h, w], [n, k4, h, w]
            labels, boxes_target, box_inside_weights, box_outside_weights = rpn_data

            # 计算分类损失
            pred_cls = pred_cls.view(n, 2, k*h*w).permute(0, 2, 1).contiguous().view(-1, 2)  # [nkhw, 2]
            labels = labels.view(-1)  # [nkhw]

            cls_mask = labels != -1  # [nkhw]
            pred_cls = pred_cls[cls_mask, :]  # [d, 2], d 为正负样本总数
            labels = labels[cls_mask].long()  # [d]
            # cls 损失为所有训练样本的平均
            rpn_cls_loss = F.cross_entropy(pred_cls, labels, reduction='mean')

            # 计算回归损失
            # [n, k4, h, w]
            loc_loss = smooth_l1(pred_loc, boxes_target, box_inside_weights, box_outside_weights, sigma=3.)
            # 求每个 batch 的平均 loss. 说明: box_inside_weights 保证训练样本才有损失, box_outside_weights 用于设置每个样本的权重
            rpn_loc_loss = loc_loss.view(n, -1).sum(dim=1).mean()

        return proposals, rpn_cls_loss, rpn_loc_loss




