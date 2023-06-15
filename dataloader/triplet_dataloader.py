# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 15:55
# @Author  : lambs
# @File    : triplet_dataloader.py
import os.path

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader.dataset import NovelDataset, MiniImageNet
from dataloader.sampler import NovelSampler, CategorySampler
from PIL import Image
from torchvision import transforms


def get_transforms(img_size=84):
    transforms_list = [
        transforms.Resize(92),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]
    return transforms_list


def get_aug_transforms(img_size=84):
    crop_transforms = [[transforms.CenterCrop(img_size)], [transforms.RandomCrop(img_size)]]
    aug_transforms = [
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0, 180))
    ]
    transforms_list = [[transforms.Resize(112)]]
    crop_index = np.random.permutation(len(crop_transforms))[0]
    transforms_list.append(crop_transforms[crop_index])
    aug_index = np.random.permutation(len(aug_transforms))[:2]
    transforms_list.append(list(aug_transforms[i] for i in aug_index))
    transforms_list.append([transforms.ToTensor()])
    return [t for l in transforms_list for t in l]


def data_augmentation(img_path, aug_times, img_size=84):
    img = Image.open(img_path)
    data = []
    for i in range(aug_times):
        if i == 0:
            transforms_list = get_transforms(img_size)
        else:
            transforms_list = get_aug_transforms(img_size)
        transforms_lists = transforms.Compose(transforms_list + [
            transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                 np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
        ])
        data.append(transforms_lists(img))
    return torch.stack(data)


def prepare_data(novel_class=3, k_shot=5, base_class=5, samples_per_class=20, num_batch=20,
                 num_workers=4, pin_memory=True):
    novel_dataset = NovelDataset(k_shot)
    novel_sampler = NovelSampler(novel_dataset.support, num_batch, novel_class, k_shot)
    novel_dataloader = DataLoader(novel_dataset, batch_sampler=novel_sampler,
                                  num_workers=int(num_workers / 2), pin_memory=pin_memory)
    base_dataset = MiniImageNet("train")
    base_classes = base_dataset.num_classes
    base_sampler = CategorySampler(base_dataset.labels, num_batch, n_way=base_class,
                                   samples_per_class=samples_per_class)
    base_dataloader = DataLoader(base_dataset, batch_sampler=base_sampler,
                                 num_workers=num_workers, pin_memory=pin_memory)

    return novel_dataloader, base_dataloader, base_classes


def sample_triplet_batch(novel_iter, base_iter, base_classes, k_shot=5,
                         samples_per_class=20, num_batch=20, img_size=84):
    for i in range(num_batch):
        novel_data, novel_label = next(novel_iter)
        base_data, base_label = next(base_iter)
        aug_novel_data = []
        for novel_img in novel_data:
            aug_times = int(samples_per_class / k_shot)
            aug_novel_data.append(data_augmentation(novel_img, aug_times, img_size))
        aug_novel_data = torch.stack(aug_novel_data).reshape(-1, 20, 3, 84, 84).transpose(0, 1)
        aug_novel_data = aug_novel_data.contiguous().reshape(-1, 3, 84, 84)
        aug_novel_label = novel_label.reshape(3, -1).repeat(1, 4).t().reshape(-1)
        aug_novel_label = aug_novel_label + base_classes
        batch_data = torch.cat((aug_novel_data, base_data), dim=0)
        batch_label = torch.cat((aug_novel_label, base_label), dim=0)
        yield batch_data, batch_label


if __name__ == "__main__":
    # img = Image.open(os.path.join("./data/miniimagenet/images/","n0153282900000005.jpg"))
    # img.show()
    # transforms_list = transforms.Compose(get_aug_transforms())
    # toPIL = transforms.ToPILImage()
    # toPIL(transforms_list(img)).show()
    prepare_data()
