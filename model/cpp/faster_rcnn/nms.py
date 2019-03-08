# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from model.cpp.faster_rcnn import _C

nms = _C.nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
