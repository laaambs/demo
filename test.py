# -*- coding: utf-8 -*-
# @Time    : 2023/6/18 15:40
# @Author  : lambs
# @File    : test.py
import torch

# x = torch.arange(0,12).reshape(3,4)
# print(x)
# print(x.is_contiguous())
# y = torch.transpose(x,dim0=0,dim1=1)
# print(y)
# print(y.is_contiguous())
# y = y.contiguous() #新开辟一块内存，重新按语义顺序按行存储
# print(y.is_contiguous())

x = torch.arange(0, 3 * 4 * 5).reshape(3, 4, 5)
print(x)
y1 = torch.permute(x, dims=(2, 1, 0))
print(y1)
print(y1.is_contiguous())
y2 = torch.transpose(x, dim0=2, dim1=0)
print(y2 == y1)
print(y2.is_contiguous())
