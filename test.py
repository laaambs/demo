# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 10:33
# @Author  : lambs
# @File    : test.py
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
tensor = torch.stack((-x, -y, x, y)).T
print(tensor)
tensor = tensor.repeat(3, 2)
print(tensor)
print(tensor[:])