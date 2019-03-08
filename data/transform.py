# Author: An Jiaoyang
# 1.1 21:01 
# =============================
"""数据变换
"""
import cv2
import numpy as np
from config.base import cfg


class Compose(object):
    """组合多个 Transform"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes=None, labels=None):
        for fun in self.transforms:
            image, boxes, labels = fun(image, boxes, labels)
        return image, boxes, labels


class Resize(object):
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_AREA = cv2.INTER_AREA
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4

    def __init__(self, size=None, interpolation=INTER_LINEAR):
        """
        调整图像大小
        :param size: int 或 (width, height)
        :param interpolation: INTER_NEAREST - 最近邻插值
                              INTER_LINEAR - 线性插值(默认)
                              INTER_AREA - 区域插值
                              INTER_CUBIC - 三次样条插值
                              INTER_LANCZOS4 - Lanczos 插值
        """
        # size 为 None 时, 返回原图
        self.size = size if isinstance(size, (tuple, list)) else (size, size)
        self.interpolation = interpolation

    def __call__(self, image, boxes=None, labels=None):
        image = image if self.size[0] is None else cv2.resize(image, self.size, interpolation=self.interpolation)
        return image, boxes, labels


class SubtractMeans(object):
    """减均值"""
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image, boxes, labels


def subtract_means_transform():
    """获取减均值的 transform"""
    transform = SubtractMeans(np.array(cfg.pixel_means, dtype=np.float32))
    return transform
