# Author: An Jiaoyang
# 12.26 10:50 
# =============================
"""Smooth L1 Loss"""
import torch


def smooth_l1_loss(input_, target, weight_in=1.0, weight_out=1.0, sigma=1.0):
    """Smooth L1 损失

    input_: a
    target: b
    weight_in: w_in
    weight_out: w_out
    sigma: s

    返回: 与 input_ 大小相同

    x = a - b
    x' = w_in * x
    smooth_l1 = 0.5 * (s * x')^2     if |x| < 1/s^2
                |x'| - 0.5 * 1/s^2   if |x| >= 1/s^2
    output = w_out * smooth_l1

    公式细节, 查看这里: https://blog.csdn.net/wfei101/article/details/77778462
    """
    s2 = sigma ** 2
    x = (input_ - target) * weight_in
    abs_x = torch.abs(x)
    sign = (abs_x < (1. / s2)).detach().float()

    loss = 0.5 * s2 * torch.pow(x, 2) * sign + (abs_x - 0.5 / s2) * (1. - sign)
    loss = loss * weight_out

    return loss
