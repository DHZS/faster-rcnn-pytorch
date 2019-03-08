# Author: An Jiaoyang
# 12.14 20:30 
# =============================
import torch.nn as nn
from functools import partial
from collections import OrderedDict, Iterable
from model.layers.base import conv2d, fc
from model.nets._faster_rcnn import FasterRCNN
from model.utils import net_utils
from config.base import cfg


class Vgg16(FasterRCNN):
    def __init__(self):
        self.base_net_out_dim = 512  # base net 输出特征图通道数
        super(Vgg16, self).__init__(self.base_net_out_dim)

        # vgg net
        self.base_layers = nn.ModuleDict(self._create_base_net())  # out: [n, 512, h/16, w/16]
        self.head_layers = nn.ModuleDict(self._create_head_layers())  # out: [n, 4096]
        # cls, loc
        self.pred_cls = nn.Linear(4096, self.num_classes)
        if self.class_agnostic:
            self.pred_loc = nn.Linear(4096, 4)
        else:
            self.pred_loc = nn.Linear(4096, self.num_classes * 4)  # 后续处理顺序为 [num_cls, 4]

    def _create_base_net(self):
        """特征提取 + anchor 预测"""
        base_layers = OrderedDict()
        conv_ = partial(conv2d, kernel_size=3, padding=1, use_bn=self.use_bn, layers=base_layers)
        # conv1
        conv_(self.input_channels, 64, name='conv1_1')
        conv_(64, 64, name='conv1_2')
        base_layers['pool1'] = nn.MaxPool2d(2, 2)
        # conv2
        conv_(64, 128, name='conv2_1')
        conv_(128, 128, name='conv2_2')
        base_layers['pool2'] = nn.MaxPool2d(2, 2)
        # conv3
        conv_(128, 256, name='conv3_1')
        conv_(256, 256, name='conv3_2')
        conv_(256, 256, name='conv3_3')
        base_layers['pool3'] = nn.MaxPool2d(2, 2)
        # conv4
        conv_(256, 512, name='conv4_1')
        conv_(512, 512, name='conv4_2')
        conv_(512, 512, name='conv4_3')
        base_layers['pool4'] = nn.MaxPool2d(2, 2)
        # conv5
        conv_(512, 512, name='conv5_1')
        conv_(512, 512, name='conv5_2')
        conv_(512, self.base_net_out_dim, name='conv5_3')
        return base_layers

    def _create_head_layers(self):
        """proposal 特征提取"""
        ph, pw = self.rpn_feat_size  # RoI Pooling 之后高宽
        head_layers = OrderedDict()
        fc_ = partial(fc, use_bn=False, drop_prob=0.5, layers=head_layers)
        # 全连接层
        fc_(512 * ph * pw, 4096, name='fc6')
        fc_(4096, 4096, name='fc7')
        return head_layers

    def _base_net(self, x):
        # override
        names, layers = net_utils.flatten_dict(self.base_layers)
        for name, layer in zip(names, layers):
            x = layer(x)
        return x

    def _head_net(self, pooled_feat):
        # override
        # x 表示 pooled_feat: [nr, 512, ph, pw]
        x = pooled_feat.view(pooled_feat.size(0), -1)
        names, layers = net_utils.flatten_dict(self.head_layers)
        for name, layer in zip(names, layers):
            x = layer(x)
        return x

    def _pred_cls_net(self, x):
        # override
        pred_cls = self.pred_cls(x)  # [nr, num_cls]
        return pred_cls

    def _pred_loc_net(self, x):
        # override
        pred_loc = self.pred_loc(x)  # [nr, 4]
        return pred_loc

    def freeze_to(self, layer_name=None):
        # override
        if not layer_name:
            return

        def no_grad(layer_):
            for p in layer_.parameters():
                p.requires_grad = False

        for k, layer in self.base_layers.items():
            if isinstance(layer, Iterable):
                for sub_layer in layer.values():
                    no_grad(sub_layer)
            else:
                no_grad(layer)
            if k == layer_name:
                break

    def init_weights(self, module_class=None):
        super(Vgg16, self).init_weights(module_class)
        # 与 tf-faster-rcnn 相同
        net_utils.init_normal(self.pred_loc, 0, 0.001, cfg.train.truncated)




