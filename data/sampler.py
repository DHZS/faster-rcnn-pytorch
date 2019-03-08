# Author: An Jiaoyang
# 1.2 10:52 
# =============================
"""
PyTorch Dataset Sampler
在采样 item 时, 以 batch 为单位打乱数据, 如

原始数据 [(1, 2), (3, 4), (5, 6), ...]
随机采样 [(5, 6), (1, 2), (3, 4), ...]

其中同一个 batch 内的数据不变
"""
import torch
import torch.utils.data as data


class RandomBatchSampler(data.sampler.Sampler):
    def __init__(self, num_item, batch_size):
        self.num_item = num_item
        self.batch_size = batch_size
        self.num_batch = int(self.num_item / self.batch_size)
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if self.num_item % batch_size != 0:
            # 最后一个 batch 数据个数不足
            self.leftover = torch.arange(self.num_batch * batch_size, self.num_item).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(-1, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), dim=0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_item



