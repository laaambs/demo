# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 10:33
# @Author  : lambs
# @File    : kmeans_train.py
import os

import torch

from models.demo import Demo


def prepare_model(best_weights, way=5, shot=5, query=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Demo(way, shot, query)
    model_dict = torch.load(best_weights, map_location=device)['state']
    model.load_state_dict(model_dict)
    model.to(device)
    return model


class Trainer:
    def __init__(self):
        self.model = prepare_model(os.path.join('./checkpoint', "max_acc.pth"))
        self.model.eval()
