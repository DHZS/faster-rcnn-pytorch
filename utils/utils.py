# Author: An Jiaoyang
# 11.15 10:32 
# =============================
"""工具
"""
import os
import time


def str2bool(s):
    return s if isinstance(s, bool) else s.lower() in ('yes', 'true', 't', '1')


def str2list(type_, sep=','):
    def fn(s, typ=type_):
        if isinstance(s, (list, tuple)):
            return s
        if typ in (str, int, float, bool):
            if typ == bool:
                typ = str2bool
            params = [typ(p.strip()) for p in s.strip().split(sep) if p.strip() != '']
            return params
        if typ in (list, tuple):
            return eval(s)
    return fn


def mkdir(p):
    if isinstance(p, str) and not os.path.exists(p):
        os.makedirs(p)


def name_with_time(prefix='', suffix='', file_ext='', time_format=None):
    """返回带有时间的文件名"""
    time_format = time_format or '%Y%m%d_%H%M%S'
    if prefix.strip() != '':
        prefix = prefix.strip() + '_'
    if suffix.strip() != '':
        suffix = '_' + suffix.strip()
    if file_ext.strip() != '':
        file_ext = file_ext.strip()
        if len(file_ext) > 0 and file_ext[0] != '.':
            file_ext = '.' + file_ext
    return prefix + time.strftime(time_format, time.localtime()) + suffix + file_ext


def flatten_list(list_, type_=list):
    """将嵌套的 list 转换为 1 维"""
    result = list()
    for item in list_:
        if isinstance(item, (tuple, list)):
            if len(item) > 0:
                result += flatten_list(item)
        else:
            result += [item]
    return type_(result)


def main():
    pass


if __name__ == '__main__':
    main()














