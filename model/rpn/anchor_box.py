# Author: An Jiaoyang
# 11.10 19:14 
# =============================
"""生成 anchor boxes
输入图像的大小可能不相同, 即预测特征图大小特征可能不相同, 因此只输出预测特征图 (0, 0) 点对应到原图位置的 anchor box
坐标为在原图的绝对坐标值, 从 0 开始, 包括两个端点, 格式 (xyxy)

生成的结果与 py-faster-rcnn 相同, 代码中使用了 round() 函数进行四舍五入, 因此得到的 anchor 坐标不如理论值精确
"""
import torch
import numpy as np
from config.base import cfg


class AnchorBox(object):
    def __init__(self, anchor_scales, anchor_ratios, feat_stride):
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.feat_stride = feat_stride

    def create(self):
        # base box 的宽高为 self.feat_stride, 计算不同比例的 anchor box
        ratio = np.array(self.anchor_ratios, dtype=np.float32).reshape([-1, 1])
        w = np.round(self.feat_stride / np.sqrt(ratio))  # round, 因此结果不精确
        h = np.round(w * ratio)

        # 计算不同大小的 anchor box
        scale = np.array(self.anchor_scales, dtype=np.float32).reshape([1, -1])
        w = np.reshape(w * scale, [-1])
        h = np.reshape(h * scale, [-1])

        # 预测特征图 (0,0) 点在原图对应的窗口中心坐标
        cx = cy = (self.feat_stride - 1) / 2

        # 计算每个 anchor 的左上角和右下角坐标
        x_min = cx - (w - 1) / 2
        y_min = cy - (h - 1) / 2
        x_max = cx + (w - 1) / 2
        y_max = cy + (h - 1) / 2
        xyxy = np.stack([x_min, y_min, x_max, y_max], axis=1)
        return xyxy


def create_anchors(anchor_scales=None, anchor_ratios=None, feat_stride=None):
    """生成 (0, 0) 位置的 k 个 anchor box

    返回值: shape [k, 4]
    """
    anchor_scales = cfg.anchor_scales if anchor_scales is None else anchor_scales
    anchor_ratios = cfg.anchor_ratios if anchor_ratios is None else anchor_ratios
    feat_stride = cfg.feat_stride if feat_stride is None else feat_stride

    anchor_box = AnchorBox(anchor_scales, anchor_ratios, feat_stride)
    anchors = anchor_box.create()
    return anchors


def create_anchor_boxes(base_anchor_boxes, feature_size, device, batch_size=None, feat_stride=None):
    """生成预测层上的所有位置的 anchor boxes

    base_anchor_box: (0, 0) 位置的 k 个 anchor box
    feature_size: 预测特征图大小 (h, w)
    device: device
    batch_size: 如果为空, 返回 [1, hwk, 4], 否则为 [n, hwk, 4]
    feat_stride: 如果为空, 取 cfg 中的值
    """
    feat_stride = cfg.feat_stride if feat_stride is None else feat_stride
    h, w = feature_size
    a = h * w  # 预测特征图大小
    k = base_anchor_boxes.size(0)  # anchor 种类个数

    # 生成不同位置 anchor 的偏移
    y, x = np.mgrid[:h, :w].astype(np.float32) * feat_stride
    shifts = np.stack([x.ravel(), y.ravel(), x.ravel(), y.ravel()], axis=1)  # [hw, 4]
    shifts = torch.from_numpy(shifts).contiguous().to(device).float()

    base_anchor_box = base_anchor_boxes.to(device).float()  # [k, 4]
    anchors = base_anchor_box.view(1, k, 4) + shifts.view(a, 1, 4)  # 广播, [hw, k, 4]
    anchors = anchors.view(1, a * k, 4)  # [1, hwk, 4]
    if batch_size is not None:
        anchors = anchors.expand(batch_size, a * k, 4)  # [n, hwk, 4]
    return anchors


if __name__ == '__main__':
    an = AnchorBox(None)
    print(an.create())
    # [[ -84.  -40.   99.   55.]
    # [-176.  -88.  191.  103.]
    # [-360. -184.  375.  199.]
    # [ -56.  -56.   71.   71.]
    # [-120. -120.  135.  135.]
    # [-248. -248.  263.  263.]
    # [ -36.  -80.   51.   95.]
    # [ -80. -168.   95.  183.]
    # [-168. -344.  183.  359.]]


