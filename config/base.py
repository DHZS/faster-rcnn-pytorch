# Author: An Jiaoyang
# 12.21 21:44 
# =============================
from yacs.config import CfgNode as CN


# ========= 全局变量 ==========
_c = CN()
_c.train = CN()
_c.test = CN()
cfg = _c
# =============================


# ========== 其他 ==========
# debug 模式
_c.debug = False

# 读取数据 worker 数量
_c.num_workers = 4
# 是否使用 CUDA
_c.cuda = True
# 是否使用多 GPU
_c.multi_gpu = True
# 是否使用 cudnn 加速
_c.cudnn_benchmark = True
# 是否使用可复现的结果, 会降低速度
_c.cudnn_deterministic = False
# ------ 日志 ------
# 日志路径
_c.log_folder = ''
# 日志文件名, 默认为创建日志的时间, 如果最后有 "+" 为追加模式
_c.log_name = ''
# =============================


# ========== 训练参数 ==========
# 优化器
_c.train.optimizer = 'sgd'
# 一次训练 batch_size 幅图像
_c.train.batch_size = 1
# 学习率
_c.train.lr = 0.001
# 学习率衰减率
_c.train.gamma = 0.1
# 学习率衰减的 step
_c.train.lr_steps = [60000, ]
# 优化器动量
_c.train.momentum = 0.9
# 权重衰减
_c.train.weight_decay = 0.0005
# 迭代次数
_c.train.max_iter = 80000
# 偏置使用 2 倍学习率
_c.train.double_bias = True
# 如果有 bn, 其偏置是否使用 2 倍学习率
_c.train.bn_double_bias = False
# 偏置是否使用 weight decay
_c.train.bias_decay = False
# 是否裁剪梯度
_c.train.gradient_clipping = False
# 允许的最大 2 范数
_c.train.clip_norm = 10.

# 模型
_c.model = 'vgg16'
# 第二阶段回归是否是类别不可知的
_c.class_agnostic = False
# 模型初始化方式. 'normal': 正态分布. 'kaiming': kaiming 正态分布. 'xavier': xavier 正态分布
_c.train.model_init = 'normal'
# 是否截断正太分布. 正太分布的值如果与均值的差值小于两倍的标准差: (x - mean) < 2 * stddev
_c.train.truncated = False

# 从模型 state_dict 的快照中恢复训练
_c.train.resume = ''
# 使用 base net 从头开始训练
_c.train.base_net = ''
# 从指定迭代次数开始训练, -1 为忽略该参数, 使用 resume 模型中的值
_c.train.start_iter = -1
# 冻结指定层以及其之前的层, 不更新参数
_c.train.freeze_to = 'conv2_2'

# 模型保存路径
_c.train.save_folder = ''
# 保存间隔
_c.train.save_interval = 500
# =============================


# ========== 数据集 ==========
# 数据集类型. 'voc' 或 'coco'
_c.dataset_type = ['voc', ]
# 数据集路径
_c.dataset_path = ['', ]
# 数据集 set
_c.image_set = ['', ]
# 是否使用标注为 'difficult' 的数据
_c.use_difficult = [False, ]
# 数据集名称
_c.dataset_name = ['', ]

# 是否翻转 image
_c.flip = True

# 类别个数, 包括背景
_c.num_classes = 21

# w:h > 2 是裁剪图像
_c.ratio_highest = 2.

# w:h < 0.5 时裁剪图像
_c.ratio_lowest = 0.5

# 生成 dataset 时, 一幅图像保留的最大 boxes 个数. 个数不足的图像 boxes 会用 0 补齐. 设置的值越大, 计算量越大.
_c.max_num_gt_boxes = 20
# 不设置 max_num_gt_boxes, 生成 dataset 时. 设置 boxes 个数为同一个 batch 中的最大值. 不足的补 0.
_c.keep_all_gt_boxes = False

# 缓存目录
_c.cache_path = ''

# 减图像均值, 颜色通道顺序为 (bgr)
_c.pixel_means = [102.9801, 115.9465, 122.7717]
# ------------------

# 图像短边缩放到的大小
_c.train.shortest_side = 600

# =============================


# ========== anchor 参数 ==========
# anchor 大小比例
_c.anchor_scales = [8, 16, 32]

# anchor 宽高比 (w: h)
_c.anchor_ratios = [0.5, 1, 2]

# RPN 特征图 stride, 4 次下采样
_c.feat_stride = 16
# =============================


# ========== 网络参数 ==========
# 输入图像通道数
_c.input_channels = 3

# 是否使用 BN
_c.use_bn = False
# =============================


# ========== RPN 阶段配置 ==========
# RoI Pooling 之后特征图宽高 (h, w)
_c.rpn_feat_size = (7, 7)

# RoI pooling/align 模式, 'pool' 或 'align'
_c.rpn_pooling_mode = 'align'

# RoI pooling 层特征的 stride, 用于将原图的 roi 坐标对应到 roi pooling 层
_c.rpn_spatial_scale = 1 / 16.

# IoU >= 阈值: 正样本
_c.train.rpn_positive_overlap = 0.7

# IoU < 阈值: 负样本
_c.train.rpn_negative_overlap = 0.3

# 是否使用严格的正样本 anchor box 条件
# True:  确保正样本 anchor box 的 IoU 一定大于正样本阈值
# False: 如果某个 gt box 的最大 IoU < 正本阈值, 为了确保每个 gt box 有对应的 anchor box 进行匹配/预测, 设置该 anchor box 为正样本
_c.train.rpn_clobber_positives = False

# RPN 阶段, 参与训练的 anchor box 总数
_c.train.rpn_batch_size = 256

# 参与训练的 anchor box 中, 正样本所占的最大比例
_c.train.rpn_fg_fraction = 0.5

# 训练时, RPN 中 NMS 之前保留的预测概率 top k 的结果
_c.train.rpn_pre_nms_top_k = 12000

# 训练时, RPN 中 NMS 之后保留的预测概率 top k 的结果
_c.train.rpn_post_nms_top_k = 2000

# 训练时, RPN 中 NMS 阈值
_c.train.rpn_nms_threshold = 0.7

# 训练时, RPN 中 保留 h, w 都大于阈值的 proposal, 长度为在原图中的绝对长度(没有使用)
_c.train.rpn_min_size = 8

# ------ 测试 ------

_c.test.rpn_pre_nms_top_k = 6000
_c.test.rpn_post_nms_top_k = 300
_c.test.rpn_nms_threshold = 0.7

# (没有使用)
_c.test.rpn_min_size = 16
# =============================


# ========== 第二阶段 ==========
# 第二阶段每幅图像采样 roi 的样本数量, 第二阶段总的 batch size 为 batch_size * rois_per_image
_c.train.rois_per_image = 256

# 一个 batch 中正样本个数所占比例, 即 class_id > 0
_c.train.fg_fraction = 0.25

# roi 正样本 IoU 阈值
_c.train.fg_threshold = 0.5

# roi 负样本 IoU 阈值. iou∈[low, high), 则 class_id=0
_c.train.bg_threshold = (0.0, 0.5)

# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
_c.train.bbox_normalize_targets_precomputed = True

# 计算得到的回归偏移均值
_c.train.bbox_normalize_means = (0.0, 0.0, 0.0, 0.0)

# 计算得到的回归偏移标准差
_c.train.bbox_normalize_stds = (0.1, 0.1, 0.2, 0.2)
# =============================


# ========== 测试 ==========
# 模型路径
_c.test.model = ''

# 输出路径
_c.test.output_dir = ''

# 评估指标. 'voc' 或 'coco'
_c.test.metric = 'voc'
# 使用 PASCAL VOC 2007 或 2012 的评估指标
_c.test.use_voc_2007_metric = True

# 测试 nms 阈值
_c.test.nms_threshold = 0.3
# 一幅图像保留的最大目标数
_c.test.max_per_image = 100

# 测试模式. 'default': 默认的测试模式。 ‘select_top_score’： [n, 21] 的得分取每项最高的变为 [n] 项预测结果
_c.test.test_mode = 'default'
# =============================




























