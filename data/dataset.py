# Author: An Jiaoyang
# 12.28 22:29 
# =============================
"""
根据数据集的标注文件创建 Dataset, 属于同一个 batch 的图像比例相似, 处理后图像尺寸相同
"""
import math
import cv2
import torch
import numpy as np
import torch.utils.data.dataset as dataset
from data.transform import subtract_means_transform
from config.base import cfg


class Dataset(dataset.Dataset):
    def __init__(self, annotations, batch_size, transform=None, sub_means=True, training=True):
        self.data = annotations
        self.num_data = len(self.data)
        self.batch_size = batch_size
        self.training = training
        self.sub_means = sub_means
        self.sub_means_transform = subtract_means_transform()
        self.transform = transform
        # 每幅图像需要调整到的 ratio. 相同 batch 的图像 ratio 相同
        self.target_ratio = self._get_batch_ratio()
        # 每个样本 gt_boxes 长度. 相同 batch 的值一样, 为最大的那个.
        self.max_num_boxes_in_batch = self._get_batch_num_boxes()

    def _get_batch_ratio(self):
        """计算同一个 batch 中每幅图像需要调整到的 ratio"""
        batch_ratio = np.zeros([self.num_data], dtype=np.float32)
        num_batch = math.ceil(self.num_data / self.batch_size)

        for i in range(num_batch):
            i0 = i * self.batch_size  # 每个 batch 开始的 index
            i1 = min((i + 1) * self.batch_size - 1, self.num_data - 1)  # 每个 batch 最后的 index

            left_ratio = self.data[i0]['ratio']
            right_ratio = self.data[i1]['ratio']

            if right_ratio < 1:
                # 使用最左侧的 ratio 作为该 batch 的 ratio. 小于最低 ratio 时, 目标 ratio 设置为最低 ratio
                target_ratio = max(cfg.ratio_lowest, left_ratio)
            elif left_ratio > 1:
                # 使用最右侧的 ratio 作为该 batch 的 ratio. 大于最高 ratio 时, 目标 ratio 设置为最高 ratio
                target_ratio = min(cfg.ratio_highest, right_ratio)
            else:
                # ratio 范围在 1 附近, 目标 ratio 设为 1
                target_ratio = 1

            batch_ratio[i0: i1 + 1] = target_ratio
        return batch_ratio

    def _get_batch_num_boxes(self):
        """计算同一个 batch 中 gt_boxes 个数最大值"""
        """计算同一个 batch 中每幅图像需要调整到的 ratio"""
        batch_num_boxes = np.zeros([self.num_data], dtype=np.int32)
        num_batch = math.ceil(self.num_data / self.batch_size)

        for i in range(num_batch):
            i0 = i * self.batch_size  # 每个 batch 开始的 index
            i1 = min((i + 1) * self.batch_size, self.num_data)  # 每个 batch 最后的 index, 不包括该 index

            max_num_boxes = max([self.data[j]['boxes'].shape[0] for j in range(i0, i1)])

            batch_num_boxes[i0: i1] = max_num_boxes
        return batch_num_boxes

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        """
        返回 image       [3, h, w]
            im_info     [2, ], (h, w)
            gt_boxes    [max_boxes, 5]
            num_boxes   [1, ]
        """
        item = self.data[index]

        # 得到指定短边长度的 image
        image = self.load_image(index)
        if self.sub_means:
            image, _, _ = self.sub_means_transform(image)
        image, im_scale = self._scale_image(image)
        (h, w) = image.shape[0: 2]

        # 根据 scale 处理 boxes
        num_obj = item['boxes'].shape[0]
        assert num_obj > 0
        gt_boxes = np.zeros([num_obj, 5], dtype=np.float32)
        gt_boxes[:, 0:4] = item['boxes'] * im_scale
        gt_boxes[:, 4] = item['labels']

        # 如果是训练阶段, 随机打乱 boxes, crop 图像
        if self.training:
            target_ratio = self.target_ratio[index]
            need_crop = item['need_crop']
            image, gt_boxes, (h, w) = self._crop_image(image, gt_boxes, target_ratio, need_crop)
            gt_boxes, num_boxes = self._padding_boxes(index, gt_boxes)
        else:
            # 测试阶段, 使用没有实际含义的值填充
            gt_boxes = np.zeros([1, 5], dtype=np.float32)
            num_boxes = 0

        # 使用 transform
        if self.transform is not None:
            boxes = gt_boxes[:, 0:4]
            labels = gt_boxes[:, 4:]
            image, boxes, labels = self.transform(image, boxes, labels)
            gt_boxes = np.concatenate([boxes, labels], axis=1)

        # 整理数据格式
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()  # [3, h, w]
        im_info = torch.tensor([h, w, im_scale], dtype=torch.float32)  # (h, w, im_scale)
        gt_boxes = torch.from_numpy(gt_boxes).contiguous()  # [max_boxes, 5]
        num_boxes = torch.tensor([num_boxes], dtype=torch.int32)  # [1, ]

        return image, im_info, gt_boxes, num_boxes

    def load_image(self, index):
        """使用 cv2 读取图像, 通道顺序为 bgr"""
        image = cv2.imread(self.data[index]['path'], cv2.IMREAD_COLOR)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=2)  # [h, w, 3]
        if self.data[index]['flipped']:
            image = image[:, ::-1, :]
        return image

    def _padding_boxes(self, index, gt_boxes):
        """对齐, 过滤, 填充 bounding boxes

        返回  gt_boxes_padding
             num_boxes: 实际 boxes 数量
        """
        # 检查 bounding boxes 是否有效
        keep = (gt_boxes[:, 0] < gt_boxes[:, 2]) & (gt_boxes[:, 1] < gt_boxes[:, 3])

        if cfg.keep_all_gt_boxes:
            # 忽略 max_num_gt_boxes 参数. 保留所有有效的 boxes
            max_boxes = self.max_num_boxes_in_batch[index]
            gt_boxes_padding = np.zeros([max_boxes, gt_boxes.shape[1]], dtype=np.float32)
            if np.sum(keep) != 0:
                gt_boxes = gt_boxes[keep]
                num_boxes = gt_boxes.shape[0]
                gt_boxes_padding[:num_boxes, :] = gt_boxes
            else:
                num_boxes = 0

        else:
            # 固定 gt_boxes 最大长度, 对长度不足的 gt_boxes 填充 0
            gt_boxes_padding = np.zeros([cfg.max_num_gt_boxes, gt_boxes.shape[1]], dtype=np.float32)

            if np.sum(keep) != 0:
                gt_boxes = gt_boxes[keep]
                num_boxes = min(gt_boxes.shape[0], cfg.max_num_gt_boxes)
                gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]
            else:
                num_boxes = 0

        return gt_boxes_padding, num_boxes

    @staticmethod
    def _crop_image(image, gt_boxes, target_ratio, need_crop):
        """裁剪图像

        target_ratio: 目标 ratio
        need_crop: 是否需要裁剪 image

        返回: padding_image    一个 batch 内大小统一的 image
             gt_boxes         裁剪 image 后更新 bounding boxes 坐标
             (h, w)           裁剪后的 image 尺寸
        """
        h, w = image.shape[0: 2]
        ratio = target_ratio  # 图像的目标比例 w: h

        # 随机重排 gt_boxes, 因为限制了一张图像上 gt_boxes 的最大数量
        np.random.shuffle(gt_boxes)

        # 如果需要, crop 图像
        if need_crop:
            if ratio < 1:
                # w < h, 需要对 h 进行裁剪, 检查 gt_boxes 的 y 坐标范围
                y_min = int(np.min(gt_boxes[:, 1]))
                y_max = int(np.max(gt_boxes[:, 3]))
                box_region = y_max - y_min + 1
                trim_size = int(w / ratio)
                if trim_size > h:
                    # 合法性检查
                    trim_size = h
                    print('INFO: trim_size > h')
                if y_min == 0:
                    y0 = 0  # 裁剪的起始坐标
                else:
                    # 如果 boxes y 坐标不从 0 开始
                    if box_region < trim_size:
                        # boxes 坐标范围 < 裁剪区域长度, 可以裁剪的 y 坐标范围
                        y0_min = max(0, y_max - trim_size)
                        y0_max = min(y_min, h - trim_size)
                        if y0_min == y0_max:
                            y0 = y0_min
                        else:
                            y0 = np.random.choice(range(y0_min, y0_max))
                    else:
                        # boxes 范围 > 裁剪长度, 考虑裁掉 boxes 范围 [0, (box_region-trim_size)/2] 内的一个值
                        y_min_add = int((box_region - trim_size) / 2)
                        if y_min_add == 0:
                            y0 = y_min
                        else:
                            y0 = np.random.choice(range(y_min, y_min + y_min_add))
                # crop 图像
                image = image[y0:y0+trim_size, :, :]

                # 更新 boxes y 坐标
                gt_boxes[:, 1] = np.clip(gt_boxes[:, 1] - y0, a_min=0, a_max=trim_size-1)
                gt_boxes[:, 3] = np.clip(gt_boxes[:, 3] - y0, a_min=0, a_max=trim_size-1)

            else:
                # w:h >=1, 需要为 w 裁剪, 原理同上
                x_min = int(np.min(gt_boxes[:, 0]))
                x_max = int(np.max(gt_boxes[:, 2]))
                box_region = x_max - x_min + 1
                trim_size = int(h * ratio)
                if trim_size > w:
                    # 合法性检查
                    trim_size = w
                    print('INFO: trim_size > w')
                if x_min == 0:
                    x0 = 0  # 裁剪的起始坐标
                else:
                    if box_region < trim_size:
                        # boxes 坐标范围 < 裁剪区域长度, 可以裁剪的 y 坐标范围
                        x0_min = max(0, x_max - trim_size)
                        x0_max = min(x_min, w - trim_size)
                        if x0_min == x0_max:
                            x0 = x0_min
                        else:
                            x0 = np.random.choice(range(x0_min, x0_max))
                    else:
                        x_min_add = int((box_region - trim_size) / 2)
                        if x_min_add == 0:
                            x0 = x_min
                        else:
                            x0 = np.random.choice(range(x_min, x_min + x_min_add))
                # crop 图像
                image = image[:, x0:x0+trim_size, :]

                # 更新 boxes x 坐标
                gt_boxes[:, 0] = np.clip(gt_boxes[:, 0] - x0, a_min=0, a_max=trim_size-1)
                gt_boxes[:, 2] = np.clip(gt_boxes[:, 2] - x0, a_min=0, a_max=trim_size-1)
        # end if need_crop

        # 根据 ratio, 对图像填充
        if ratio < 1:
            # w < h, 该 batch 内所有 image 的 w=600, r_i <= r_i+1, 即 h_i >= h_i+1.
            # 故令该 batch 内 image 的 h=h_i, 多余部分用 0 填充
            h_max = int(math.ceil(w / ratio))
            padding_image = np.zeros([h_max, w, 3], dtype=np.float32)
            padding_image[0:h, :, :] = image
            h = h_max  # 更新 h
        elif ratio > 1:
            # w > h, 该 batch 内所有 image 的 h=600, r_i <= r_i+1, 即 w_i <= w_i+1.
            # 故令该 batch 内 image 的 w=w_i+n, (n=batch_size-1), 多余部分用 0 填充
            w_max = int(math.ceil(h * ratio))
            padding_image = np.zeros([h, w_max, 3], dtype=np.float32)
            padding_image[:, 0:w, :] = image
            w = w_max  # 更新 h
        else:
            # r_min <= 1, r_max >= 1, 因此统一 w, h = min(w_i, h_i) = 600, 裁剪掉多余的部分
            assert ratio == 1
            wh_min = min(w, h)
            padding_image = image[0:wh_min, 0:wh_min, :]
            # 更新 gt_boxes
            gt_boxes = gt_boxes.copy()
            gt_boxes[:, 0:4] = np.clip(gt_boxes[:, 0:4], a_min=0, a_max=wh_min-1)
            # 更新 w, h
            w = h = wh_min

        return padding_image, gt_boxes, (h, w)

    @staticmethod
    def _scale_image(image):
        """缩放图像"""
        h, w = image.shape[0: 2]
        short = min(h, w)  # 短边
        target_short = cfg.train.shortest_side  # 目标长度
        scale = float(target_short / short)  # 目标尺度
        # 根据 scale 缩放图像
        im = cv2.resize(image, dsize=None, dst=None, fx=scale, fy=scale)
        return im, scale
















