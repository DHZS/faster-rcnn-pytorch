# Faster R-CNN in PyTorch

[中文](README_zh.md)

This is a simple implementation of Faster R-CNN. I mainly referred to the following repositories:
- [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)

## Progress

- [x] Training and testing on VOC
- [x] VGG16
- [ ] Training and testing on COCO
- [ ] ResNet

## Prerequisites

- python 3
- pytorch 1.0
- python-opencv
- matplotlib
- yacs

## Installation

1. Clone the repository

```
git clone https://github.com/DHZS/faster-rcnn-pytorch.git
```

2. Compile modules

```
cd faster-rcnn-pytorch/model/cpp
python setup.py build develop
```


## Training on PASCAL VOC

### Prepare the data

Configure your dataset path in `yaml` config file. e.g.
```
# Train on VOC07&12
dataset_name: ['voc_2007', 'voc_2012']
dataset_type: ['voc', 'voc']
dataset_path: ['/dataset/pascal_voc/VOC2007', '/dataset/pascal_voc/VOC2012']
image_set: ['trainval', 'trainval']
use_difficult: [True, True]
```

### Download the model pretrained on ImageNet
[VGG16_no_bn](https://drive.google.com/file/d/1Fhb3AM5BcYEFW0g2yHv5aMwVsOl36JT8/view?usp=sharing)(Google Drive)

Configure the model file path. e.g.
```
train:
  base_net: './output/faster_rcnn_base_vgg16_caffe_no_bn.pth'
```

### Train & Evaluation & Visualization
```
# train
python do_train.py --config_path=config/_train_voc.yaml

# evaluation
python do_eval.py --config_path=config/_eval_voc.yaml

# visualization
python do_visual.py --config_path=config/_visual_voc.yaml
```

## Result

VGG16(conv5_3), Roi Align, VOC07 trainval/VOC07 test
```
V0C07 metric? Yes
AP for aeroplane = 0.7222
AP for bicycle = 0.7745
AP for bird = 0.6855
AP for boat = 0.5490
AP for bottle = 0.5627
AP for bus = 0.7931
AP for car = 0.8554
AP for cat = 0.8283
AP for chair = 0.4850
AP for cow = 0.7905
AP for diningtable = 0.6418
AP for dog = 0.7935
AP for horse = 0.8343
AP for motorbike = 0.7754
AP for person = 0.7725
AP for pottedplant = 0.4553
AP for sheep = 0.6845
AP for sofa = 0.6376
AP for train = 0.7485
AP for tvmonitor = 0.7229
Mean AP = 0.7056
```






