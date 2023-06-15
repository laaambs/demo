# -*- coding: utf-8 -*-
# @Time    : 2023/3/15 14:34
# @Author  : lambs
# @File    : semi_hard_triplet_loss.py
import torch
import torch.nn as nn
from torch.autograd import Variable


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(self.margin)

    def forward(self, inputs, labels):
        # 计算样本两两之间的距离
        n = inputs.shape[0]
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist.clamp_(min=1e-12).sqrt()
        # 计算mask矩阵
        mask = (labels.expand(n, n) == labels.expand(n, n).t())
        # pick the hardest triplets
        # 或者使用facenet的策略：1. 使用所有positive; 2. 使用semi-hard
        dist_ap = []
        dist_an = []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap).reshape(-1)
        dist_an = torch.stack(dist_an).reshape(-1)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss, prec
