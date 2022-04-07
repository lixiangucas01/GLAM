#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import torch
from torch.utils.data import Dataset

# label_dict = {
#     'neutral': torch.Tensor([0]),
#     'happy': torch.Tensor([1]),
#     'sad': torch.Tensor([2]),
#     'angry': torch.Tensor([3]),
# }

class DataSet(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x = self.X[index]
        x = torch.from_numpy(x)
        x = x.float()
        y = self.Y[index]
        y = torch.LongTensor([y])
        return x, y

    def __len__(self):
        return len(self.X)
