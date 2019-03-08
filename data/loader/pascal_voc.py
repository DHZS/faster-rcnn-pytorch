# Author: An Jiaoyang
# 12.28 16:55 
# =============================
"""
读取 PASCAL VOC 数据集
"""
import os
import cv2
import numpy as np
import scipy.sparse
from xml.etree import ElementTree
from config.base import cfg
from data.loader.loader import Loader
from utils import utils


CLASSES = (
    '__background__',  # 索引从 0 开始, 0 为背景
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


class PascalVoc(Loader):
    def __init__(self, image_set, path, use_difficult=False, name=None, training=True):
        super(PascalVoc, self).__init__()
        self.training = training  # super
        assert image_set in ('train', 'trainval', 'val', 'test')
        self.image_set = image_set
        self.use_difficult = use_difficult
        self.name = (name or 'pascal_voc') + '_' + self.image_set + \
                    ('_no_hard', '_use_hard')[self.use_difficult] + \
                    (('_no_flip', '_use_flip')[cfg.flip] if self.training else '')  # super
        self.path = path
        self.image_path = os.path.join(self.path, 'JPEGImages', '{}.jpg')
        self.xml_path = os.path.join(self.path, 'Annotations', '{}.xml')
        utils.mkdir(cfg.cache_path)
        self.cache_path = os.path.join(cfg.cache_path, '{}_annotations.pkl'.format(self.name))  # super
        self.image_set_path = ''
        self.num_class = len(CLASSES)
        self.class_to_index = dict(zip(CLASSES, range(self.num_class)))  # {'cls': id, ...}
        self.image_ids = self._load_image_id()  # super
        self.all_annotations = None  # super

    def _load_image_id(self):
        """加载所有 image id"""
        ids = []
        self.image_set_path = os.path.join(self.path, 'ImageSets', 'Main', '{}.txt'.format(self.image_set))
        with open(self.image_set_path, mode='rt') as f:
            ids += [id_ for id_ in f.read().strip().split('\n')]
        return ids

    def _load_annotation(self, id_):
        """读取 xml 标注数据, 读取图像尺寸"""
        # 读取图像尺寸
        img_path = self.image_path.format(id_)
        img = cv2.imread(img_path)
        im_size = np.array([img.shape[0], img.shape[1]], dtype=np.float32)

        xml_path = self.xml_path.format(id_)
        xml = ElementTree.parse(xml_path)
        objects = xml.findall('object')
        if not self.use_difficult:
            # 忽略标注为困难的样本
            objects = [obj for obj in objects if int(obj.find('difficult').text) == 0]
        num_obj = len(objects)

        # 需要读取的数据
        boxes = np.zeros([num_obj, 4], dtype=np.float32)
        gt_labels = np.zeros([num_obj], dtype=np.int32)
        overlaps = np.zeros([num_obj, self.num_class], dtype=np.float32)
        seg_areas = np.zeros([num_obj], dtype=np.float32)
        is_hard = np.zeros([num_obj], dtype=np.int32)

        for i, obj in enumerate(objects):
            hard = obj.find('difficult')
            difficult = 0 if hard is None else int(hard.text)  # 0 or 1
            is_hard[i] = difficult

            bbox = obj.find('bndbox')
            # 使坐标从 0 开始
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes[i, :] = [x1, y1, x2, y2]

            cls_id = self.class_to_index[obj.find('name').text.lower().strip()]
            gt_labels[i] = cls_id

            overlaps[i, cls_id] = 1.0
            overlaps = scipy.sparse.lil_matrix(overlaps)  # 转换为稀疏矩阵, 节省内存

            seg_areas[i] = (x2 - x1 + 1) * (y2 - y1 + 1)

        result = {
            'id': id_,
            'path': img_path,
            'size': im_size,  # (h, w)
            'ratio': im_size[1] / im_size[0],  # w/h
            'boxes': boxes,
            'labels': gt_labels,
            'is_hard': is_hard,
            'overlaps': overlaps,
            'seg_areas': seg_areas,
            'flipped': False,
            'need_crop': None}
        return result



