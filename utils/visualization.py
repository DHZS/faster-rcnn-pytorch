# =============================
"""工具类
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
from data.transform import SubtractMeans
from data.loader.pascal_voc import CLASSES
from config.base import cfg


# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors


colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


# =========================================================================== #
# Matplotlib show...
# =========================================================================== #
class Plot(object):
    def __init__(self, dpi=50, line_width=1.5, order='xy', labels_map=None):
        self.dpi = dpi
        self.line_width = line_width
        self.order = 'xy' if order == 'xy' else 'yx'
        self.height = 0
        self.width = 0
        self.labels_map = labels_map
        self.have_image = False

    def plot_image(self, image):
        self.height, self.width = int(image.shape[0]), int(image.shape[1])
        fig_size = self.width / self.dpi, self.height / self.dpi
        plt.figure(figsize=fig_size)
        plt.axis('off')
        plt.tight_layout()  # 尽可能减小边距
        plt.imshow(image, cmap=plt.cm.gray)
        self.have_image = True

    def plot_boxes(self, bboxes, labels=None, is_fraction=False, scores=None, color_idx=None):
        colors = dict()
        for i in range(bboxes.shape[0]):
            cls_id = 0 if labels is None else int(labels[i])
            if cls_id >= 0:
                score = 1. if scores is None else float(scores[i])
                if cls_id not in colors:
                    color_id = color_idx or cls_id  # 如果指定颜色 id，所有类别使用相同的颜色
                    colors[cls_id] = [c / 255 for c in colors_tableau[color_id]]
                # 计算 box 大小
                idx = [0, 1, 2, 3] if self.order == 'xy' else [1, 0, 3, 2]
                if is_fraction:
                    box = bboxes[i, idx] * [self.width, self.height, self.width, self.height]
                else:
                    box = bboxes[i, idx]
                x_min, y_min, x_max, y_max = [int(c) for c in box]
                xy = (x_min, y_min)
                width = x_max - x_min + 1
                height = y_max - y_min + 1

                rect = plt.Rectangle(xy, width, height, fill=False, edgecolor=colors[cls_id], linewidth=self.line_width)
                plt.gca().add_patch(rect)
                class_name = str(cls_id) if (self.labels_map is None or labels is None) else self.labels_map[cls_id]
                text = []
                if labels is not None:
                    text += [class_name]
                if scores is not None:
                    text += ['{:.3f}'.format(score)]
                text = ' | '.join(text)
                if text != '':
                    plt.gca().text(x_min, y_min - 2,
                                   text,
                                   bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                   fontsize=10, color='white')

    def show(self):
        if self.have_image:
            plt.show()
            plt.close()
            self.have_image = False

    def save(self, path):
        if self.have_image:
            plt.savefig(path)
            plt.close()
            self.have_image = False


def plt_bboxes(img, classes, bboxes,
               scores=None, class_name_map=None, order='xy',
               color_idx=None, line_width=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    dpi = 50.
    height, width = int(img.shape[0]), int(img.shape[1])
    fig_size = width / dpi, height / dpi

    plt.figure(figsize=fig_size)
    plt.axis('off')
    plt.tight_layout()  # 尽可能减小边距
    plt.imshow(img, cmap=plt.cm.gray)
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = 1 if scores is None else scores[i]
            if cls_id not in colors:
                # colors[cls_id] = (random.random(), random.random(), random.random())
                color_id = color_idx or cls_id  # 如果指定颜色 id，所有类别使用相同的颜色
                colors[cls_id] = [c / 255 for c in colors_tableau[color_id]]
            if order == 'xy':
                y_min = int(bboxes[i, 1] * height)
                x_min = int(bboxes[i, 0] * width)
                y_max = int(bboxes[i, 3] * height)
                x_max = int(bboxes[i, 2] * width)
            else:
                assert order == 'yx'
                y_min = int(bboxes[i, 0] * height)
                x_min = int(bboxes[i, 1] * width)
                y_max = int(bboxes[i, 2] * height)
                x_max = int(bboxes[i, 3] * width)

            rect = plt.Rectangle((x_min, y_min), x_max - x_min,
                                 y_max - y_min, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=line_width)
            plt.gca().add_patch(rect)
            class_name = str(cls_id) if class_name_map is None else class_name_map[cls_id]
            plt.gca().text(x_min, y_min - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=10, color='white')
    plt.show()


def pre_process(image, bboxes=None):
    """预处理 image"""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = np.transpose(image, [1, 2, 0])
    if bboxes is not None:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.cpu().numpy()
    return image, bboxes


def show_target(image, bboxes, add_mean=False, is_fraction=False, print_bboxes=False):
    """可视化ground truth"""
    image, bboxes = pre_process(image, bboxes)
    image = image if not add_mean else SubtractMeans((-104, -117, -123))(image)[0]
    image = image.astype(np.int32)
    bboxes, labels = bboxes[:, 0:4], bboxes[:, 4].astype(np.int32)
    if print_bboxes:
        for i in range(bboxes.shape[0]):
            print(bboxes[i], labels[i])
    plot = Plot(labels_map=CLASSES)
    plot.plot_image(image)
    plot.plot_boxes(bboxes, labels, is_fraction)
    plot.show()


def show_prediction(image, preds, gt_bboxes=None, add_mean=False, save_path=None):
    """可视化预测结果"""
    image = image if not add_mean else SubtractMeans(-np.array(cfg.pixel_means))(image)[0]
    image = image.astype(np.int32)
    if gt_bboxes is not None:
        gt_bboxes, gt_labels = gt_bboxes[:, 0:4], gt_bboxes[:, 4].astype(np.int32)
    pred_boxes, pred_cls, pred_scores = preds
    # 可视化
    plot = Plot(labels_map=CLASSES)
    plot.plot_image(image)
    if gt_bboxes is not None:
        plot.plot_boxes(gt_bboxes, gt_labels, color_idx=1)
    plot.plot_boxes(pred_boxes, pred_cls, scores=pred_scores, color_idx=3)
    if save_path is not None:
        plot.save(save_path)
    else:
        plot.show()















