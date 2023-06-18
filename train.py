# -*- coding: utf-8 -*-
# @Time    : 2023/3/7 17:18
# @Author  : lambs
# @File    : train.py
import json
import os
import numpy as np
import torch
from torch import optim
from models.demo import Demo
from dataloader.dataloaders import get_dataloader


def prepare_model(init_weights, way=5, shot=5, query=15):
    # 获取model的参数字典
    model = Demo(way, shot, query)
    model_dict = model.state_dict()
    # 加载预训练模型参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_model_dict = torch.load(init_weights, map_location=device)['params']
    pre_model_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict.keys()}
    # model加载预训练模型参数
    model_dict.update(pre_model_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    return model


def prepare_optimizer(model, lr=0.002, lr_mul=5, mom=0.9, weight_decay=0.0005,
                      milestones=[40, 70, 100, 130], gamma=0.5):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]
    optimizer = optim.SGD(
        [{'params': model.encoder.parameters()},
         {'params': top_para, 'lr': lr * lr_mul}],
        lr=lr,
        momentum=mom,
        nesterov=True,
        weight_decay=weight_decay
    )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma
    )
    return optimizer, lr_scheduler


def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Trainer:
    def __init__(self):
        self.init_weights = os.path.join("./checkpoint/", "Res12-pre.pth")
        self.way = 5
        self.shot = 5
        self.query = 15
        self.num_episodes = 100
        self.num_val_episodes = 600
        self.num_test_episodes = 10000
        self.lr = 0.0004
        self.lr_mul = 10
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.milestones = [40, 70, 100, 130]
        self.gamma = 0.5
        self.max_epoch = 160
        self.eval_interval = 1
        self.num_workers = 4
        self.model = prepare_model(self.init_weights, self.way, self.shot, self.query)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = get_dataloader()
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, self.lr, self.lr_mul,
                                                              self.momentum, self.weight_decay,
                                                              self.milestones, self.gamma)
        self.best_log = dict()
        self.best_log['max_acc'] = 0.0
        self.best_log['max_acc_epoch'] = 0
        self.best_log['max_acc_interval'] = 0.0

    def try_evaluate(self, epoch):
        if epoch % self.eval_interval == 0:
            vl, va, vap = self.evaluate(self.val_dataloader)
            print('epoch {}, val, loss={:.4f} acc={:.4f}+{:.4f}'.format(epoch, vl, va, vap))

            if va >= self.best_log['max_acc']:
                self.best_log['max_acc'] = va
                self.best_log['max_acc_interval'] = vap
                self.best_log['max_acc_epoch'] = epoch
                torch.save({'epoch': epoch, 'state': self.model.state_dict()},
                           os.path.join("./checkpoint/", "max_acc_02.pth"))

    def evaluate(self, val_dataloader):
        self.model.eval()
        record = np.zeros((self.num_val_episodes, 2))
        labels = self.model.generate_labels()
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader, 1):
                data = batch[0]
                if torch.cuda.is_available():
                    labels = labels.cuda()
                    data = data.cuda()
                logits = self.model(data)
                loss = self.model.compute_loss(logits, labels)
                acc = self.model.compute_accuracy(logits, labels)
                record[i - 1, 0] = loss
                record[i - 1, 1] = acc

        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        return vl, va, vap

    def train(self):
        file_path = os.path.join("./checkpoint", "best_log.json")
        # start training
        for epoch in range(1, self.max_epoch + 1):
            self.model.train()
            total_loss = Averager()
            total_acc = Averager()
            for data, gt_labels in self.train_dataloader:
                labels = self.model.generate_labels()
                if torch.cuda.is_available():
                    data = data.cuda()
                    gt_labels = gt_labels.cuda()
                    labels = labels.cuda()
                logits = self.model(data)
                loss = self.model.compute_loss(logits, labels)
                acc = self.model.compute_accuracy(logits, labels)
                total_loss.add(loss)
                total_acc.add(acc)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step()
            self.try_evaluate(epoch)

        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.best_log['max_acc_epoch'],
            self.best_log['max_acc'],
            self.best_log['max_acc_interval']))
        with open(file_path, 'w') as f:
            json.dump(self.best_log, f)
        torch.save(self.model, 'epoch-last_02.pth')

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join("./checkpoint/",
                                                           "triplet_min_loss.pth"))['state'])
        self.model.eval()
        labels = self.model.generate_labels()
        if torch.cuda.is_available():
            labels = labels.cuda()
        batch_count = 0
        total_loss = 0.0
        total_acc = 0.0
        with torch.no_grad():
            for i, batch in enumerate(self.test_dataloader, 1):
                data = batch[0]
                if torch.cuda.is_available():
                    data = data.cuda()
                logits = self.model(data)
                loss = self.model.compute_loss(logits, labels)
                acc = self.model.compute_accuracy(logits, labels)
                total_loss += loss.item()
                total_acc += acc
                print('Test batch:{}, loss=:{:.4f}, acc={:.4f}'.format(batch_count, loss, acc))
                batch_count += 1
        print('Test total loss={:.4f}, total acc={:.4f}'.format(total_loss, total_acc))
        print('Test average loss={:.4f}, average acc={:.4f}'.format(total_loss / batch_count, total_acc / batch_count))


if __name__ == '__main__':
    trainer = Trainer()
    # trainer.train()
    trainer.test()
