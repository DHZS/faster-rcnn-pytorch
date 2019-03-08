# Faster R-CNN in PyTorch

Faster R-CNN PyTorch 实现, 代码包含了大量中文注释, 便于理解. 欢迎大家学习, 交流~

部分代码参考了如下项目:
- [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)

## 进度

- [x] VOC 数据集
- [x] VGG16
- [ ] COCO 数据集
- [ ] ResNet

## 依赖库

- python 3
- pytorch 1.0
- python-opencv
- matplotlib
- yacs

## 安装

1. 克隆项目

```
git clone https://github.com/DHZS/faster-rcnn-pytorch.git
```

2. 编译 NMS, RoI Pooling/Align. 代码所属 [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

```
cd faster-rcnn-pytorch/model/cpp
python setup.py build develop
```


## 训练

### 准备数据

在 `yaml` 文件中配置数据集. 如 voc07 12 trainval
```
dataset_name: ['voc_2007', 'voc_2012']
dataset_type: ['voc', 'voc']
dataset_path: ['/dataset/pascal_voc/VOC2007', '/dataset/pascal_voc/VOC2012']
image_set: ['trainval', 'trainval']
use_difficult: [True, True]
```

### 下载 VGG16 预训练模型.
[VGG16_no_bn](https://drive.google.com/file/d/1Fhb3AM5BcYEFW0g2yHv5aMwVsOl36JT8/view?usp=sharing)(Google Drive)

配置预训练模型路径
```
train:
  base_net: './output/faster_rcnn_base_vgg16_caffe_no_bn.pth'
```

### 训练 & 测试 & 可视化
```
# 训练
python do_train.py --config_path=config/_train_voc.yaml

# 测试
python do_eval.py --config_path=config/_eval_voc.yaml

# 可视化
python do_visual.py --config_path=config/_visual_voc.yaml
```

## 实验结果

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






