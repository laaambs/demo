# -*- coding: utf-8 -*-
# @Time    : 2023/3/15 16:21
# @Author  : lambs
# @File    : triplet_train.py
import os
from torch import optim
from dataloader.triplet_dataloader import prepare_data, sample_triplet_batch
import torch
import torch.nn.functional as F
from models.demo import Demo
from models.semi_hard_triplet_loss_2 import TripletSemihardLoss
from models.semi_hard_triplet_loss import SemiHardTripletLoss
from models.triplet_loss import TripletLoss
from train import Averager


def prepare_model(best_weights, way=5, shot=5, query=15, margin=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Demo(way, shot, query)
    model_dict = torch.load(best_weights, map_location=device)['state']
    model.load_state_dict(model_dict)
    model.to(device)
    triplet_loss = TripletSemihardLoss(margin)
    triplet_loss.to(device)
    no_free_parameters = []
    for p in model.named_parameters():
        if 'layer4.0.downsample' in p[0] or 'attention' in p[0]:
            p[1].requires_grad = True
            no_free_parameters.append(p[1])
        else:
            p[1].requires_grad = False
    return model, triplet_loss, no_free_parameters


def prepare_optimizer(parameters, lr=0.00005, weight_decay=0.01,
                      step_size=20, gamma=0.5):
    optimizer = optim.Adam(
        params=parameters,
        lr=lr,
        weight_decay=weight_decay
    )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(step_size),
        gamma=gamma
    )
    return optimizer, lr_scheduler


def train():
    num_batch = 20
    max_epoch = 30
    model, triplet_loss, no_free_parameters = prepare_model(os.path.join('./checkpoint', "max_acc.pth"))
    optimizer, lr_scheduler = prepare_optimizer(no_free_parameters)
    novel_dataloader, base_dataloader, base_classes = prepare_data(num_batch=num_batch)
    best_log = dict()
    best_log['min_loss'] = 1e10
    best_log['min_loss_epoch'] = 0
    for epoch in range(1, max_epoch + 1):
        triplet_batch_sampler = sample_triplet_batch(iter(novel_dataloader), iter(base_dataloader), base_classes)
        model.train()
        total_loss = 0.0
        for i in range(num_batch):
            batch_data, batch_label = next(triplet_batch_sampler)
            if torch.cuda.is_available():
                batch_data = batch_data.cuda()
                batch_label = batch_label.cuda()
            embeddings = model(batch_data, triplet=True)
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            loss = triplet_loss(embeddings, batch_label)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('triplet_train epoch {}, batch {}, loss={:.4f}'.format(epoch, i, loss))
        lr_scheduler.step()
        avg_loss = total_loss / num_batch
        print('triplet_train epoch {}, avg_loss={:.4f}'.format(epoch, avg_loss))
        if avg_loss <= best_log['min_loss']:
            best_log['min_loss'] = avg_loss
            best_log['min_loss_epoch'] = epoch
            torch.save({'epoch': epoch, 'state': model.state_dict()},
                       os.path.join("./checkpoint/", "triplet_min_loss.pth"))

    torch.save(model, './checkpoint/triplet_epoch-last.pth')


if __name__ == '__main__':
    # num_batch = 20
    # max_epoch = 20
    # novel_iter, base_iter, base_classes = prepare_data(num_batch=num_batch)
    # triplet_batch_sampler = sample_triplet_batch(novel_iter, base_iter, base_classes)
    # for epoch in range(1, max_epoch + 1):
    train()
    # prepare_model(os.path.join('./checkpoint', "max_acc.pth"))
