# Author: An Jiaoyang
# 12.28 21:48 
# =============================
"""Loader

标注格式
{
    'id': ,
    'path': ,
    'size': ,
    'ratio':,
    'boxes': ,
    'labels': ,
    'is_hard': ,
    'overlaps': ,
    'seg_areas': ,
    'flipped': ,
    'need_crop':
}

"""
import os
import pickle
import numpy as np
from config.base import cfg


class Loader(object):
    def __init__(self):
        self.name = None  # 数据库名称
        self.cache_path = None  # 缓存目录
        self.image_ids = None  # image id 编号
        self.all_annotations = None  # 数据集中所有图像的信息
        self.is_flipped = False  # 是否翻转
        self.is_sorted = False  # 是否已根据 ratio 排序
        self.ratio_index = None  # 排序结果在原 array 的 index
        self.use_difficult = None  # 是否使用困难标注样本
        self.training = None  # 是否是训练

    def _load_annotation(self, id_):
        """根据 image_id 读取 image 标注"""
        raise NotImplementedError()

    def get_all_annotations(self):
        """返回所有数据集中所有 annotations, 优先读取缓存文件, 不存在时会自动创建"""
        if self.all_annotations is not None:
            return self.all_annotations
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.all_annotations = pickle.load(f)
                assert len(self.all_annotations) == len(self.image_ids)
        else:
            self.all_annotations = [self._load_annotation(id_) for id_ in self.image_ids]
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.all_annotations, f, pickle.HIGHEST_PROTOCOL)
        return self.all_annotations

    def append_flipped_images(self):
        """增加水平翻转后的 image"""
        assert self.all_annotations is not None
        if self.is_flipped:
            return
        self.is_flipped = True
        for i in range(len(self.image_ids)):
            data = self.all_annotations[i]
            im_size = data['size'].copy()
            boxes = data['boxes'].copy()
            # 翻转 boxes
            boxes[:, 0], boxes[:, 2] = im_size[1] - boxes[:, 2] - 1, im_size[1] - boxes[:, 0] - 1
            result = {
                'id': data['id'],
                'path': data['path'],
                'size': im_size,
                'ratio': data['ratio'],
                'boxes': boxes,
                'labels': data['labels'],
                'is_hard': data['is_hard'],
                'overlaps': data['overlaps'],
                'seg_areas': data['seg_areas'],
                'flipped': True,
                'need_crop': data['need_crop']}
            self.all_annotations.append(result)
        self.image_ids = self.image_ids * 2


def sort_by_ratio(annotations):
    """根据图像宽高比 (w: h) 对数据升序排序"""
    ratio_list = []
    for i, data in enumerate(annotations):
        ratio = data['ratio']
        ratio_list += [ratio]

        # 是否需要裁剪
        if ratio > cfg.ratio_highest:
            data['need_crop'] = True
        elif ratio < cfg.ratio_lowest:
            data['need_crop'] = True
        else:
            data['need_crop'] = False

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)  # 从小到大排序, 返回排序结果在原 array 的 index

    # 对 annotations 和 image_ids 排序
    sorted_annotations = []
    for idx in ratio_index:
        sorted_annotations.append(annotations[idx])

    return sorted_annotations


def get_all_loader(training):
    """获取所有 loader"""
    from data.loader.pascal_voc import PascalVoc
    from data.loader.coco import COCO

    loader_list = []
    for i in range(len(cfg.dataset_type)):
        if cfg.dataset_type[i] == 'voc':
            loader = PascalVoc(cfg.image_set[i], cfg.dataset_path[i],
                               cfg.use_difficult[i], cfg.dataset_name[i], training)
        else:
            assert cfg.dataset_type[i] == 'coco'
            loader = COCO()
        loader_list += [loader]
    return loader_list


def get_all_loader_annotations(output=True, print_fn=None, training=True):
    """获取所有 loader 的 annotations"""
    # 自定义 print 函数
    if output is False:
        def print_(x): return x
    elif print_fn is not None:
        print_ = print_fn
    else:
        print_ = print

    annotation_list = []
    loaders = get_all_loader(training)

    for i, loader in enumerate(loaders):
        text = 'Loading {}/{} | name: {} | image set: {} | use hard: {}'.format(
            i + 1, len(loaders), cfg.dataset_name[i], cfg.image_set[i], cfg.use_difficult[i])
        if training:
            text += ' | flip: {}'.format(cfg.flip)
        print_(text)

        annotations = loader.get_all_annotations()
        if training and cfg.flip:
            loader.append_flipped_images()
            annotations = loader.get_all_annotations()
        annotation_list += annotations
        print_('Done, loaded {} images'.format(len(annotations)))
    if training:
        annotation_list = sort_by_ratio(annotation_list)  # 根据 ratio 升序排序, 确定是否需要裁剪
    print_('Done. Total training images: {}'.format(len(annotation_list)))

    return annotation_list


