# Author: An Jiaoyang
# 12.21 14:30 
# =============================
"""基本的层
"""
import torch.nn as nn
from collections import OrderedDict


def fc(
        in_features,
        out_features,
        use_bias=True,
        use_bn=False,
        use_relu=True,
        drop_prob=None,
        layers=None,
        name=None):
    """
    使用 partial 来包裹该函数, 必需定义 layers
    """
    sub_layers = OrderedDict()
    sub_layers['fc'] = nn.Linear(in_features, out_features, use_bias)
    if use_bn:
        sub_layers['bn'] = nn.BatchNorm1d(num_features=out_features)
    if use_relu:
        sub_layers['relu'] = nn.ReLU(inplace=True)
    if drop_prob is not None:
        sub_layers['drop'] = nn.Dropout(drop_prob)
    layers[name] = nn.ModuleDict(sub_layers)


def conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        use_bias=True,
        use_bn=False,
        use_relu=True,
        layers=None,
        name=None):
    """
    使用 partial 来包裹该函数, 必需定义 layers
    """
    sub_layers = OrderedDict()
    sub_layers['conv'] = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=use_bias)
    if use_bn:
        sub_layers['bn'] = nn.BatchNorm2d(num_features=out_channels)
    if use_relu:
        sub_layers['relu'] = nn.ReLU(inplace=True)
    layers[name] = nn.ModuleDict(sub_layers)




