# Author: An Jiaoyang
# 11.13 21:57 
# =============================
"""网络相关工具"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import os
from utils import utils
from config.base import cfg


def init_xavier(module):
    """Xavier 初始化"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()


def init_kaiming(module):
    """Kaiming 初始化"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(module.weight.data, nonlinearity='relu')
        if module.bias is not None:
            module.bias.data.zero_()


def init_normal(module, mean, stddev, truncated=False):
    """正态分布"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if truncated:
            module.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            module.weight.data.normal_(mean, stddev)
        if module.bias is not None:
            module.bias.data.zero_()


def flatten_dict(layer, layer_name='', sep='.'):
    """展开嵌套的 ModuleDict"""
    keys = []
    layers = []
    for k, v in layer.items():
        k = k if layer_name == '' else layer_name + sep + k
        if isinstance(v, nn.ModuleDict):
            _keys, _layers = flatten_dict(v, k)
            keys += _keys
            layers += _layers
        else:
            keys += [k]
            layers += [v]
    return keys, layers


def save_model(path, model, optimizer=None, scheduler=None, iteration=None):
    """保存模型"""
    folder = os.path.split(path)[0]
    utils.mkdir(folder)
    state = dict()
    state['model'] = model.state_dict()
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    if iteration is not None:
        state['iteration'] = iteration
    torch.save(state, path)


def load_model(path, model, optimizer=None, scheduler=None, default_iter=0, strict=True, use_gpu=True):
    """加载模型, 返回 iteration"""
    state_dict = torch.load(path, map_location=None if use_gpu else 'cpu')
    if 'model' not in state_dict:
        # state_dict 就是模型的参数
        model.load_state_dict(state_dict, strict=strict)
        return default_iter
    model.load_state_dict(state_dict['model'])
    if 'optimizer' in state_dict and optimizer:
        optimizer.load_state_dict(state_dict['optimizer'])
    if 'scheduler' in state_dict and scheduler:
        scheduler.load_state_dict(state_dict['scheduler'])
    if 'iteration' in state_dict:
        default_iter = state_dict['iteration']
    return default_iter


def get_optimizer(model):
    """根据配置获取优化器"""
    lr = cfg.train.lr
    momentum = cfg.train.momentum
    weight_decay = cfg.train.weight_decay

    if cfg.train.optimizer == 'adam':
        lr = lr * 0.1  # 减小 10 倍学习率

    params = []

    for key, value in dict(model.named_parameters()).items():
            if ('.bias' in key) and ('bn.bias' not in key or cfg.train.bn_double_bias):
                params += [{'params': [value],
                            'lr': lr * (2 if cfg.train.double_bias else 1),
                            'weight_decay': cfg.train.weight_decay if cfg.train.bias_decay else 0}]
            else:
                params += [{'params': [value],
                            'lr': lr,
                            'weight_decay': cfg.train.weight_decay}]

    if cfg.train.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr, momentum, weight_decay=weight_decay)
    else:
        assert cfg.train.optimizer == 'adam'
        optimizer = optim.Adam(params, lr, weight_decay=weight_decay)

    return optimizer




