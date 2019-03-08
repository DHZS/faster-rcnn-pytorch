# Author: An Jiaoyang
# 12.28 21:01 
# =============================
"""
读取 COCO 数据集
"""
from data.loader.loader import Loader


CLASSES = (
    '__background__',  # 索引从 0 开始, 0 为背景
    )


class COCO(Loader):
    def _load_annotation(self, id_):
        pass
