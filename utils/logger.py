# Author: An Jiaoyang
# 11.15 22:44 
# =============================
"""日志工具"""
import os
import sys
import time
from utils import utils


class Logger:
    def __init__(self, folder, file_name=None, append=None, time_format=None, use_pprint=False):
        utils.mkdir(folder.strip())
        self.folder = folder.strip()
        if file_name and file_name.strip()[-1] == '+':
            self.file_name = file_name.strip()[:-1]
            self.append = False if append is False else True
        else:
            self.file_name = None
            self.append = bool(append)
        self.time_format = time_format or '%Y%m%d_%H%M%S.log'
        self.path = None
        self.file = None
        self.init = True
        if use_pprint:
            from pprint import pprint
            self.print_fn = lambda text, stream=sys.stdout: print(text, file=stream) \
                if isinstance(text, str) else pprint(text, stream, width=200)
        else:
            self.print_fn = lambda text, stream=sys.stdout: print(text, file=stream)

    def _init(self):
        if self.init:
            self.init = False
            if self.file_name is None:
                self.file_name = time.strftime(self.time_format, time.localtime())
            self.path = os.path.join(self.folder, self.file_name)
            self.file = open(self.path, 'at' if self.append else 'wt', encoding='utf-8')

    def write(self, text, print_=False):
        """write with/without print"""
        self._init()
        self.print_fn(text, stream=self.file)
        self.file.flush()
        if print_:
            self.print_fn(text)

    def print(self, text):
        """print and write"""
        self.write(text, print_=True)
