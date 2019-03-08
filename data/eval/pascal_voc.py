# Author: An Jiaoyang
# 1.4 9:44 
# =============================
"""评估 PASCAL VOC 检测结果"""
import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from utils import utils
from data.loader import pascal_voc
from config.base import cfg


def _parse_rec(filename):
    """解析 xml 标注"""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects


def _voc_ap(rec, prec, use_07_metric=False):
    """通过给定的 recall 和 precision 计算 voc ap.
    如果 use_07_metric 为 True, 使用 VOC 07 的 11 点计算方法."""
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _voc_eval(det_path, anno_path, image_set_file, class_name, cache_file, ovthresh=0.5, use_07_metric=False):
    """
    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
            detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
            annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
    """
    # first load gt
    cache_file = '{}_annots.pkl'.format(cache_file)
    # read list of images
    with open(image_set_file, 'r') as f:
        lines = f.readlines()
    image_names = [x.strip() for x in lines]

    if not os.path.isfile(cache_file):
        # load annotations
        recs = {}
        for i, image_name in enumerate(image_names):
            recs[image_name] = _parse_rec(anno_path.format(image_name))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(image_names)))
        # save
        print('Saving cached annotations to {:s}'.format(cache_file))
        with open(cache_file, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cache_file, 'rb') as f:
            try:
                recs = pickle.load(f)
            except:
                recs = pickle.load(f, encoding='bytes')

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for image_name in image_names:
        R = [obj for obj in recs[image_name] if obj['name'] == class_name]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[image_name] = {'bbox': bbox,
                                  'difficult': difficult,
                                  'det': det}

    # read dets
    det_file = det_path.format(class_name)
    with open(det_file, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = _voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def _get_voc_results_file_template():
    """获取检测结果文件路径模板"""
    # {output_dir}/detection_result/{dataset_name}_{image_set}_{cls_name}.txt
    path = os.path.join(cfg.test.output_dir, 'detection_result')
    utils.mkdir(path)
    file_name = '_'.join([cfg.dataset_name[0], cfg.image_set[0], '{}.txt'])
    file_path = os.path.join(path, file_name)
    return file_path


def _write_voc_results_file(all_boxes, loader, print_):
    """将检测结果写入 txt 文件"""
    for i, cls in enumerate(pascal_voc.CLASSES):
        if cls == '__background__':
            continue
        print_('Writing {} VOC results file'.format(cls))
        filename = _get_voc_results_file_template().format(cls)
        with open(filename, 'wt') as f:
            for j, image_id in enumerate(loader.image_ids):
                dets = all_boxes[i][j]
                if len(dets) == 0:
                    continue
                # VOC 的坐标从 1 开始
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        image_id, dets[k, -1], dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1))


def _do_python_eval(loader, print_):
    cache_dir = os.path.join(cfg.cache_path, 'annotations_cache')
    utils.mkdir(cache_dir)
    cache_file = os.path.join(cache_dir, loader.name)
    output_dir = os.path.join(cfg.test.output_dir, 'evaluation')
    utils.mkdir(output_dir)

    # The PASCAL VOC metric changed in 2010
    use_07_metric = cfg.test.use_voc_2007_metric
    print_('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

    aps = []
    for i, cls in enumerate(pascal_voc.CLASSES):
        if cls == '__background__':
            continue
        filename = _get_voc_results_file_template().format(cls)
        rec, prec, ap = _voc_eval(filename, loader.xml_path, loader.image_set_path, cls, cache_file,
                                  ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print_('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    print_('Mean AP = {:.4f}'.format(np.mean(aps)))
    print_('~~~~~~~~')
    print_('Results:')
    for ap in aps:
        print_('{:.3f}'.format(ap))
    print_('{:.3f}'.format(np.mean(aps)))
    print_('~~~~~~~~')
    print_('')
    print_('--------------------------------------------------------------')
    print_('Results computed with the **unofficial** Python eval code.')
    print_('Results should be very close to the official MATLAB eval code.')
    print_('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print_('-- Thanks, The Management')
    print_('--------------------------------------------------------------')


def evaluate_detections(all_boxes, print_fn=None):
    print_ = print_fn or print
    # 只能评估一个数据集
    loader = pascal_voc.PascalVoc(cfg.image_set[0], cfg.dataset_path[0], cfg.use_difficult[0], cfg.dataset_name[0])
    _write_voc_results_file(all_boxes, loader, print_)
    _do_python_eval(loader, print_)


