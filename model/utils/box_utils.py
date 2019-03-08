# Author: An Jiaoyang
# 12.24 11:46 
# =============================
"""
对 bounding boxes 的各种操作
"""
import torch


def clip_boxes(boxes, im_shape):
    """裁剪 bbox 坐标到 image size 范围内

    boxes: [n, hwk, 4]
    im_shape: [n, 2], size 顺序 (h, w)
    """
    batch_size = boxes.size(0)
    for i in range(batch_size):
        h, w = im_shape[i, 0], im_shape[i, 1]
        boxes[i, :, 0::4].clamp_(0, w - 1)  # x_min
        boxes[i, :, 1::4].clamp_(0, h - 1)  # y_min
        boxes[i, :, 2::4].clamp_(0, w - 1)  # x_max
        boxes[i, :, 3::4].clamp_(0, h - 1)  # x_max
    return boxes


def boxes_encode_batch(boxes, gt_boxes):
    """坐标编码 (xyxy) -> (dx,dy,dh,dw)

    boxes: anchor boxes, shape [n, k, 4] 或 [k, 4]
    gt_boxes: ground truth boxes, shape [n, k, 4]

    返回: [n, k, 4], 4 表示 (dx, dy, dw, dh)
    """
    n, k = gt_boxes.size(0), gt_boxes.size(1)  # n 为 batch size, k 为 anchor box 个数
    if boxes.dim() == 2:
        # anchor boxes (w, h, cx, cy), shape [k]
        aw = boxes[:, 2] - boxes[:, 0] + 1.0
        ah = boxes[:, 3] - boxes[:, 1] + 1.0
        acx = boxes[:, 0] + 0.5 * aw
        acy = boxes[:, 1] + 0.5 * ah

        # 调整 size 为 [n, k]
        aw, ah, acx, acy = [x.view(1, k).expand(n, k) for x in [aw, ah, acx, acy]]

        # gt boxes (w, h, cx, cy), shape [n, k]
        gw = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1.0
        gh = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1.0
        gcx = gt_boxes[:, :, 0] + 0.5 * gw
        gcy = gt_boxes[:, :, 1] + 0.5 * gh

        # 计算偏移
        dx = (gcx - acx) / aw
        dy = (gcy - acy) / ah
        dw = torch.log(gw / aw)
        dh = torch.log(gh / ah)

    elif gt_boxes.dim() == 3:
        # shape [n, k]
        aw = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
        ah = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
        acx = boxes[:, :, 0] + 0.5 * aw
        acy = boxes[:, :, 1] + 0.5 * ah

        # shape [n, k]
        gw = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1.0
        gh = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1.0
        gcx = gt_boxes[:, :, 0] + 0.5 * gw
        gcy = gt_boxes[:, :, 1] + 0.5 * gh

        dx = (gcx - acx) / aw
        dy = (gcy - acy) / ah
        dw = torch.log(gw / aw)
        dh = torch.log(gh / ah)

    else:
        raise ValueError()

    d_xywh = torch.stack([dx, dy, dw, dh], dim=2)  # [n, k, 4]
    return d_xywh


def boxes_decode(boxes, deltas):
    """坐标解码 (dx,dy,dh,dw) -> (xyxy) 坐标为在原图上的绝对坐标

    boxes: anchor box, 顺序 (xyxy), shape [n, hwk, 4]
    deltas: RPN anchor 回归阶段为  预测的坐标, 宽高偏移, 顺序 (dx,dy,dh,dw), shape [n, hwk, 4]
            RoI 预测坐标回归阶段为  [n, r, (1 or num_cls)*4]

    返回 [n, hwk, 4] 或 [n, hwk, (1 or num_cls)*4]
    """
    w = boxes[:, :, 2] - boxes[:, :, 0] + 1.0  # [n, hwk]
    h = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    cx = boxes[:, :, 0] + 0.5 * w
    cy = boxes[:, :, 1] + 0.5 * h
    # 扩展最后一维, 方便广播
    w, h, cx, cy = [x.unsqueeze(2) for x in [w, h, cx, cy]]  # [n, hwk, 1]

    dx = deltas[:, :, 0::4]  # [n, hwk, (1 or num_cls)]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    # 预测的 (w,h,cx,cy)
    pred_cx = dx * w + cx  # [n, hwk, (1 or num_cls)]
    pred_cy = dy * h + cy
    pred_w = torch.exp(dw) * w
    pred_h = torch.exp(dh) * h

    # 转换为 xyxy
    x_min = pred_cx - 0.5 * pred_w  # [n, hwk, (1 or num_cls)]
    y_min = pred_cy - 0.5 * pred_h
    x_max = pred_cx + 0.5 * pred_w
    y_max = pred_cy + 0.5 * pred_h

    # 按读取顺序构造解码后的 boxes 坐标
    xyxy = torch.empty_like(deltas)  # [n, hwk, 4] 或 [n, hwk, (1 or num_cls)*4]
    xyxy[:, :, 0::4] = x_min
    xyxy[:, :, 1::4] = y_min
    xyxy[:, :, 2::4] = x_max
    xyxy[:, :, 3::4] = y_max
    return xyxy


def boxes_iou_batch(anchor_boxes, gt_boxes):
    """计算每个 anchor box 与所有 gt boxes 的 IoU

    anchor_boxes: [k, 4] 或 [n, k, 4] 或 [n, k, 5]  # k 为 anchor boxes 数量
    gt_boxes: [n, b, 4]  # n 为 batch size

    返回: [n, k, b]
    """
    n = gt_boxes.size(0)  # batch size
    b = gt_boxes.size(1)  # gt boxes 总数

    if anchor_boxes.dim() == 2:
        k = anchor_boxes.size(0)  # anchor boxes 总数

        anchor_boxes = anchor_boxes.view(1, k, 4).expand(n, k, 4).contiguous()  # [n, k, 4]
        gt_boxes = gt_boxes[:, :, 0:4].contiguous()

        # 计算 anchor boxes 宽高, shape [n, k]
        aw = anchor_boxes[:, :, 2] - anchor_boxes[:, :, 0] + 1
        ah = anchor_boxes[:, :, 3] - anchor_boxes[:, :, 1] + 1
        anchor_area = (aw * ah).view(n, k, 1)

        # 计算 gt boxes 宽高, shape [n, b]
        gw = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1
        gh = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1
        gt_area = (gw * gh).view(n, 1, b)

        # 边长为 0 的 boxes mask
        anchor_zero = (aw == 1) & (ah == 1)  # [n, k]
        anchor_zero = anchor_zero.view(n, k, 1).expand(n, k, b)  # [n, k, b]
        gt_zero = (gw == 1) & (gh == 1)  # [n, b]
        gt_zero = gt_zero.view(n, 1, b).expand(n, k, b)  # [n, k, b]

        # 计算 IoU
        boxes = anchor_boxes.view(n, k, 1, 4).expand(n, k, b, 4)
        query_boxes = gt_boxes.view(n, 1, b, 4).expand(n, k, b, 4)

        # 计算交集区域坐标
        x_min = torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0])
        y_min = torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1])
        x_max = torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2])
        y_max = torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3])

        # 交集区域边长
        iw = x_max - x_min + 1
        ih = y_max - y_min + 1

        # 边长为负数, 调整为 0
        iw[iw < 0] = 0
        ih[ih < 0] = 0

        # 计算 IoU
        inter = iw * ih  # [n, k, b]
        union = anchor_area + gt_area - inter  # [n, k, b]
        iou = inter / union

        # 填充边长为 0 的 boxes 计算得到的 IoU 为 0/-1.  为什么填充的值不一样?
        iou.masked_fill_(mask=gt_zero, value=0)
        iou.masked_fill_(mask=anchor_zero, value=-1)

    elif anchor_boxes.dim() == 3:
        k = anchor_boxes.size(1)  # anchor boxes 总数

        if anchor_boxes.size(2) == 4:
            anchor_boxes = anchor_boxes.contiguous()  # [n, k, 4]
        else:
            anchor_boxes = anchor_boxes[:, :, 1:5].contiguous()  # [n, k, 4]

        gt_boxes = gt_boxes[:, :, 0:4].contiguous()

        # 计算 anchor boxes 宽高, shape [n, k]
        aw = anchor_boxes[:, :, 2] - anchor_boxes[:, :, 0] + 1
        ah = anchor_boxes[:, :, 3] - anchor_boxes[:, :, 1] + 1
        anchor_area = (aw * ah).view(n, k, 1)

        # 计算 gt boxes 宽高, shape [n, b]
        gw = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1
        gh = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1
        gt_area = (gw * gh).view(n, 1, b)

        # 边长为 0 的 boxes mask
        anchor_zero = (aw == 1) & (ah == 1)  # [n, k]
        anchor_zero = anchor_zero.view(n, k, 1).expand(n, k, b)  # [n, k, b]
        gt_zero = (gw == 1) & (gh == 1)  # [n, b]
        gt_zero = gt_zero.view(n, 1, b).expand(n, k, b)  # [n, k, b]

        # 计算 IoU
        boxes = anchor_boxes.view(n, k, 1, 4).expand(n, k, b, 4)
        query_boxes = gt_boxes.view(n, 1, b, 4).expand(n, k, b, 4)

        # 计算交集区域坐标
        x_min = torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0])
        y_min = torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1])
        x_max = torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2])
        y_max = torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3])

        # 交集区域边长
        iw = x_max - x_min + 1
        ih = y_max - y_min + 1

        # 边长为负数, 调整为 0
        iw[iw < 0] = 0
        ih[ih < 0] = 0

        # 计算 IoU
        inter = iw * ih  # [n, k, b]
        union = anchor_area + gt_area - inter  # [n, k, b]
        iou = inter / union

        # 填充边长为 0 的 boxes 计算得到的 IoU 为 0/-1.  为什么填充的值不一样?
        # 先填充 0, 即在 [k, b] b (无效的 boxes, 纵向)方向填充全零
        # 再填充 -1, 即在 [k, b] k (无效的 anchor, 横向)方向填充-1
        # 保证取每个roi对应最大IoU时(b中/横向取最大), 无效的anchor取到的最大box都是第0个.
        iou.masked_fill_(mask=gt_zero, value=0)
        iou.masked_fill_(mask=anchor_zero, value=-1)

    else:
        raise ValueError()

    return iou






























