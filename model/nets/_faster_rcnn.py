# Author: An Jiaoyang
# 12.21 15:37 
# =============================
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base import cfg
from model.rpn.rpn import RPN
from model.rpn.proposal_match_layer import ProposalMatchLayer
from model.cpp.faster_rcnn import ROIPool, ROIAlign
from model.losses import smooth_l1
from model.utils import net_utils


class FasterRCNN(nn.Module):
    def __init__(self, base_net_out_dim):
        super(FasterRCNN, self).__init__()
        self.num_classes = cfg.num_classes  # 类别个数, 包括背景
        self.class_agnostic = cfg.class_agnostic  # 坐标回归是否为类别不可知
        self.base_net_out_dim = base_net_out_dim  # base net 输出特征图通道数
        self.input_channels = cfg.input_channels
        self.use_bn = cfg.use_bn  # base net 是否使用 bn
        self.rpn_feat_size = cfg.rpn_feat_size  # RoI Pooling 输出特征图的高宽
        self.rpn_pooling_mode = cfg.rpn_pooling_mode  # RoI Pool 模式
        self.rpn_spatial_scale = cfg.rpn_spatial_scale  # RoI pooling 层 stride

        # RPN 网络
        self.rpn = RPN(self.base_net_out_dim)
        # 根据 rpn 的输出 roi, 采样得到第二阶段的训练样本和 ground truth
        self.proposal_match_layer = ProposalMatchLayer()

        if self.rpn_pooling_mode == 'pool':
            self.roi_pool = ROIPool(output_size=self.rpn_feat_size, spatial_scale=self.rpn_spatial_scale)
        elif self.rpn_pooling_mode == 'align':
            # sampling_ratio 参数作用???
            self.roi_pool = ROIAlign(output_size=self.rpn_feat_size, spatial_scale=self.rpn_spatial_scale,
                                     sampling_ratio=0)
        else:
            raise ValueError()

    def forward(self, x, im_info, gt_boxes, num_boxes):
        # 输入到 RPN 的特征图
        base_feat = self._base_net(x)  # [n, 512, h/16, w/16]
        rois, rpn_cls_loss, rpn_loc_loss = self.rpn(base_feat, gt_boxes, im_info)

        if self.training:
            # 匹配第二阶段正负样本
            output = self.proposal_match_layer(rois, gt_boxes)
            # 用采样得到的 rois 替换 rpn 输出的 rois
            rois, rois_label, rois_loc_target, inside_weights, outside_weights = output

            # n 为 batch size, r 为第二阶段样本总数
            n, r = rois_label.size()

            rois_label = rois_label.view(n*r).long()
            rois_loc_target = rois_loc_target.view(n*r, 4)
            inside_weights = inside_weights.view(n*r, 4)
            outside_weights = outside_weights.view(n*r, 4)
        else:
            rois_label = None
            rois_loc_target = None
            inside_weights = None
            outside_weights = None
            rpn_cls_loss = 0
            rpn_loc_loss = 0

        # roi pooling, batch_id 用于 roi 对应 base_feat 特征
        # base_feat:   [n, xxx, h/ratio, w/ratio]
        # rois:        [batch_size * num_rois, (batch_id, x, y, x, y)], 简写为 [nr, 5]
        # pooled_feat: [nr, xxx, ph, pw], (ph, pw) 为配置的 pooled 特征大小
        pooled_feat = self.roi_pool(base_feat, rois.view(-1, 5))
        pooled_feat = self._head_net(pooled_feat)  # [nr, xxx]

        # 分类回归预测
        pred_cls = self._pred_cls_net(pooled_feat)  # [nr, num_cls]
        pred_cls_prob = F.softmax(pred_cls, dim=1)  # 预测概率, [nr, num_cls]
        pred_loc = self._pred_loc_net(pooled_feat)  # [nr, (1 or num_cls)*4]
        if self.training and not self.class_agnostic:
            # 如果是类别可知, 处理 pred_loc
            nr = pred_loc.size(0)
            pred_loc = pred_loc.view(nr, self.num_classes, 4)
            # 选择 gt label 对应的 pred loc
            pred_loc = pred_loc[torch.arange(nr), rois_label, :]  # [nr, 4]

        # 计算损失
        cls_loss = 0
        loc_loss = 0

        if self.training:
            cls_loss = F.cross_entropy(pred_cls, rois_label)
            # 求所有样本的平均损失(loss_sum / num_roi)
            loc_loss = smooth_l1(pred_loc, rois_loc_target, inside_weights, outside_weights).sum(dim=1).mean()
            # num_pos = (rois_label > 0).sum().float()  # TODO 官方实现?
            # loc_loss = smooth_l1(pred_loc, rois_loc_target, inside_weights, outside_weights).sum() / num_pos

        # n 为 batch size, r 为预测的 rois 数量
        n, r = rois.size(0), rois.size(1)

        pred_cls_prob = pred_cls_prob.view(n, r, self.num_classes)
        pred_loc = pred_loc.view(n, r, -1)  # 测试时 shape 为 [n, r, (1 or num_cls)*4]

        return rois, pred_cls_prob, pred_loc, rpn_cls_loss, rpn_loc_loss, cls_loss, loc_loss, rois_label

    def init_weights(self, module_class=None):
        """初始化权重, 默认为正太分布"""
        if cfg.train.model_init == 'normal':
            def initializer(m): net_utils.init_normal(m, 0, 0.01, cfg.train.truncated)
        elif cfg.train.model_init == 'kaiming':
            initializer = net_utils.init_kaiming
        else:
            assert cfg.train.model_init == 'xavier'
            initializer = net_utils.init_xavier

        module_class = module_class or (nn.Linear, nn.Conv2d, nn.BatchNorm2d)
        for layer in self.modules():
            if isinstance(layer, module_class):
                initializer(layer)

    def freeze_to(self, layer_name=None):
        """冻结第一层到 layer_name 范围内的所有层"""
        pass

    def _base_net(self, x):
        """特征提取 + anchor 预测"""
        raise NotImplementedError()

    def _head_net(self, pooled_feat):
        """处理 roi pooling 之后的特征, 准备输入到分类回归分支"""
        raise NotImplementedError()

    def _pred_cls_net(self, x):
        """分类分支"""
        raise NotImplementedError()

    def _pred_loc_net(self, x):
        """回归分支"""
        raise NotImplementedError()


