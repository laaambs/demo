# -*- coding: utf-8 -*-
# @Time    : 2023/2/27 18:30
# @Author  : lambs
# @File    : sampler.py
import os
import numpy as np
import torch
from torch.utils.data import Sampler
from dataloader.dataset import MiniImageNet, NovelDataset


class CategorySampler:
    """ Usage: this class is a batch sampler, a batch equals an episode,
        thus the size of a batch is n ways*(k shot+q query).
        Params: source label, num_batch, batch_size
    """

    def __init__(self, label, num_batch, n_way, samples_per_class):
        self.category_indices = self.classify(label)
        self.num_batch = num_batch
        self.n_way = n_way
        self.samples_per_class = samples_per_class

    def classify(self, label):
        category_indices = []
        categories = len(set(label))
        for i in range(categories):
            category_indices.append(np.argwhere(np.array(label) == i))
        return category_indices

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        for i in range(self.num_batch):
            batch = []
            # randomly pick n classes
            classes = torch.randperm(len(self.category_indices))[:self.n_way]
            for c in classes:
                indices = self.category_indices[int(c)]
                # randomly pick s samples for each class
                samples = torch.randperm(len(indices))[:self.samples_per_class]
                batch.append(torch.Tensor(indices[samples]).squeeze())
            # batch_1 = torch.cat(batch, dim=0).reshape(-1)
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class NovelSampler:
    """Usage: this class is a batch sampler.
        In triplet loss, we use this sampler to sample novel data
        but only sample its support data
    """

    def __init__(self, support, num_batch, n_class, k_shot):
        self.support = support
        self.num_batch = num_batch
        self.n_class = n_class
        self.k_shot = k_shot

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        for i in range(self.num_batch):
            batch = []
            classes = torch.randperm(len(self.support))[:self.n_class]
            for c in classes:
                batch.append(torch.Tensor(self.support[c][:self.k_shot]).squeeze())
            batch = torch.stack(batch).reshape(-1)
            yield batch


if __name__ == '__main__':
    # dataset = MiniImageNet("train")
    # print(next(iter(CategorySampler(dataset.labels, 20, 5, 20))).reshape(-1, 5).t())
    dataset = NovelDataset(k_shot=5)
    print(next(iter(NovelSampler(dataset.support, num_batch=20, n_class=3, k_shot=5))))
