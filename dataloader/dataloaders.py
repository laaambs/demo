# -*- coding: utf-8 -*-
# @Time    : 2023/2/27 20:30
# @Author  : lambs
# @File    : dataloaders.py
import os.path

import numpy as np
import torch

from dataloader.dataset import MiniImageNet
from dataloader.sampler import CategorySampler
from torch.utils.data import DataLoader


def get_dataloader(num_episodes=100, num_val_episodes=600, num_test_episodes=10000,
                   n_way=5, k_shot=5, q_query=15, num_workers=4):
    train_dataset = MiniImageNet("train")
    train_sampler = CategorySampler(train_dataset.labels, num_batch=num_episodes,
                                    n_way=n_way, samples_per_class=k_shot + q_query)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=True)
    val_dataset = MiniImageNet("val")
    val_sampler = CategorySampler(val_dataset.labels, num_batch=num_val_episodes,
                                  n_way=n_way, samples_per_class=k_shot + q_query)
    val_dataloader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,
                                num_workers=num_workers, pin_memory=True)
    test_dataset = MiniImageNet("test")
    test_sampler = CategorySampler(test_dataset.labels, num_batch=num_test_episodes,
                                   n_way=n_way, samples_per_class=k_shot + q_query)
    test_dataloader = DataLoader(dataset=test_dataset, batch_sampler=test_sampler,
                                 num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = get_dataloader()
    # for images, labels in val_dataloader:
    #     print(labels)
    #     break
    print(len(val_dataloader))
