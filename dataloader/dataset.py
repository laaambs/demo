# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 17:09
# @Author  : lambs
# @File    : dataset.py
import os.path

import torch
from PIL import Image
import numpy as np
import numpy.random as random
from torch.utils.data import Dataset
from torchvision import transforms

split_list = ['train', 'val', 'test']


class MiniImageNet(Dataset):
    def __init__(self, split, augment=False):
        self.images, self.labels, self.classes_dict = self.parse_file(
            os.path.join("../data/miniimagenet/split/", "{}.csv".format(split)))
        self.num_classes = len(self.classes_dict.keys())
        self.transforms = self.get_transforms(augment, split)

    def parse_file(self, split_file_path):
        """ Usage: return images and labels
        """
        images = []
        labels = []
        label_id_dict = dict()
        id_count = 0
        with open(split_file_path, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                image, label = lines[i].strip().split(",")
                if label not in label_id_dict.keys():
                    label_id_dict[label] = id_count
                    id_count += 1
                images.append(image)
                labels.append(label_id_dict[label])
        return images, labels, label_id_dict

    def __len__(self):
        return len(self.labels)

    def get_transforms(self, augment, split="train"):
        img_size = 84
        if augment and split == "train":
            transform_list = [
                transforms.RandomResizedCrop(img_size),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.toTensor()
            ]
        else:
            transform_list = [
                transforms.Resize(92),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ]

        transform_list = transforms.Compose(transform_list + [
            transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                 np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
        ])
        return transform_list

    def __getitem__(self, index):
        image, label = self.images[int(index)], self.labels[int(index)]
        img = Image.open(os.path.join("../data/miniimagenet/images/", image))
        image = self.transforms(img)
        return image, label


class NovelDataset(Dataset):
    def __init__(self, k_shot):
        self.novel_dataset = MiniImageNet('test', augment=False)
        self.support, self.query = self.divide(k_shot)

    def divide(self, k_shot):
        labels = self.novel_dataset.labels
        category_indices = []
        support = []
        query = []
        for c in range(len(set(labels))):
            category_indices.append(np.argwhere(np.array(labels) == c))
            class_indices = random.permutation(category_indices[c])
            support.append(class_indices[:k_shot])
            query.append(class_indices[k_shot:])
        return support, query

    def __len__(self):
        return len(self.novel_dataset.labels)

    def __getitem__(self, index):
        """
        :param index:
        :return: return the image path and image label
        """
        image, label = self.novel_dataset.images[int(index)], self.novel_dataset.labels[int(index)]
        img = os.path.join("../data/miniimagenet/images/", image)
        return img, label


if __name__ == '__main__':
    dataset = NovelDataset(k_shot=5)
    print(len(dataset.support))
    img, label = dataset[0]
    print(label)
    image = Image.open(img)
    image.show()
