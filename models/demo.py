# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 16:13
# @Author  : lambs
# @File    : demo.py
import os

import torch
import torch.nn as nn
import numpy as np
from dataloader.dataloaders import get_dataloader
from networks.res12 import ResNet


class ScaledDotProductionAttention(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(ScaledDotProductionAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v):
        attention_score = torch.bmm(q, torch.transpose(k, -2, -1))
        scale = np.sqrt(k.shape[-1])
        attention_score = self.softmax(attention_score / scale)
        attention_score = self.dropout(attention_score)
        output = torch.bmm(attention_score, v)
        return output, attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, input_dim, k_dim, v_dim, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.input_dim = input_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.w_q = nn.Linear(input_dim, n_head * k_dim, bias=False)
        self.w_k = nn.Linear(input_dim, n_head * k_dim, bias=False)
        self.w_v = nn.Linear(input_dim, n_head * v_dim, bias=False)
        self.attention = ScaledDotProductionAttention()
        self.fc = nn.Linear(n_head * v_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_dim)

        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2 / (input_dim + k_dim)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2 / (input_dim + k_dim)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2 / (input_dim + v_dim)))
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, triplet=False):
        n_head = self.n_head
        k_dim = self.k_dim
        v_dim = self.v_dim

        residual = q
        q_len, _ = q.shape
        k_len, _ = k.shape
        v_len, _ = v.shape
        q = self.w_q(q).view(q_len, n_head, k_dim)
        k = self.w_k(k).view(k_len, n_head, k_dim)
        v = self.w_v(v).view(v_len, n_head, v_dim)

        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        output, attention_score = self.attention(q, k, v)
        output = output.permute(1, 0, 2).contiguous().view(q_len, -1)
        if not triplet:
            output = self.layer_norm(self.dropout(self.fc(output)) + residual)
            return output
        else:
            output = self.dropout(self.fc(output)) + residual
            return output


class Demo(nn.Module):
    def __init__(self, way, shot, query):
        super(Demo, self).__init__()
        self.encoder = ResNet()
        self.emb_features = 640
        self.way = way
        self.shot = shot
        self.query = query
        self.attention = MultiHeadAttention(n_head=1, input_dim=self.emb_features,
                                            k_dim=self.emb_features, v_dim=self.emb_features)
        self.loss_function = nn.CrossEntropyLoss()

    def split_instances(self):
        """
        :return: support_indices, query_indices
        """
        way, shot, query = self.way, self.shot, self.query
        support_idx = range(0, way * shot)
        query_idx = range(way * shot, way * (shot + query))
        return support_idx, query_idx

    def split_novel_instances(self):
        novel_class = 3
        base_class = 5
        samples_per_class = 20
        novel_idx = range(0, novel_class * samples_per_class)
        base_idx = range(novel_class * samples_per_class, (novel_class + base_class) * samples_per_class)
        return novel_idx, base_idx

    def forward(self, x, mode="base"):
        if mode == "base":
            return self.base_forward(x)
        elif mode == "triplet":
            return self.triplet_forward(x)
        elif mode == "kmeans":
            return self.kmeans_forward(x)

    def base_forward(self, x):
        # feature extraction
        batch_emb = self.encoder(x)
        # split support and query
        support_idx, query_idx = self.split_instances()
        support_emb = batch_emb[support_idx]
        query_emb = batch_emb[query_idx]
        # feature adaption
        adapted_support = self.attention(support_emb, query_emb, query_emb)
        adapted_query = self.attention(query_emb, support_emb, support_emb)
        # compute prototype
        prototype = adapted_support.contiguous().view(self.shot, self.way, -1)
        prototype = prototype.permute(1, 0, 2).mean(dim=1)
        # nn classify
        logits = - self.euclidean_dist(adapted_query, prototype)
        return logits

    def kmeans_forward(self, x):
        # feature extraction
        batch_emb = self.encoder(x)
        # split support and query
        support_idx, query_idx = self.split_instances()
        support_emb = batch_emb[support_idx]
        query_emb = batch_emb[query_idx]
        # initialize cluster centers
        center = support_emb.reshape(5, 5, 640).permute(1, 0, 2).mean(dim=1)

        return center

    def triplet_forward(self, x):
        # feature extraction
        batch_emb = self.encoder(x)
        # split novel and base
        novel_idx, base_idx = self.split_novel_instances()
        novel_emb = batch_emb[novel_idx]
        base_emb = batch_emb[base_idx]
        # get negative samples
        negative_emb = self.attention(base_emb, novel_emb, novel_emb, triplet=True)
        return torch.cat((novel_emb, negative_emb), dim=0)

    def euclidean_dist(self, x, y):
        x_len = x.shape[0]
        y_len = y.shape[0]
        assert x.shape[1] == y.shape[1]
        dim = x.shape[1]
        x = x.unsqueeze(1).expand(x_len, y_len, dim)
        y = y.unsqueeze(0).expand(x_len, y_len, dim)
        return torch.pow(x - y, 2).sum(2)

    def generate_labels(self):
        labels = torch.arange(self.way, dtype=torch.int16).repeat(self.query)
        labels = labels.type(torch.LongTensor)

        return labels

    def compute_loss(self, logits, labels):
        loss = self.loss_function(logits, labels)
        return loss

    def compute_accuracy(self, logits, labels):
        pred = torch.argmax(logits, dim=1)
        accuracy = (pred == labels).type(torch.FloatTensor).mean().item()
        return accuracy


if __name__ == "__main__":
    # prepare data
    train_dataloader, val_dataloader, test_dataloader = get_dataloader()

    # prepare model
    model = Demo(way=5, shot=5, query=15)
    model_dict = model.state_dict()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_model_dict = torch.load(os.path.join("../checkpoint/", "Res12-pre.pth"), map_location=device)['params']
    pre_model_dict = {k: v for k, v in pre_model_dict.items() if k in model_dict.keys()}
    model_dict.update(pre_model_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    # print loss and accuracy
    for images, labels in test_dataloader:
        print(labels)

        # # 查看embedding中特征维度的均值分布
        # batch_emb = model(images, mode="kmeans")
        # distribution = batch_emb.reshape(20,5,640).permute(1,0,2).mean(dim=1)
        # large_feature = torch.argwhere(distribution>1).tolist()
        # print(large_feature)
        # for row,col in large_feature:
        #     print(distribution[row,col])
        # logits = model(images)
        # labels = model.generate_labels()
        # loss = model.compute_loss(logits, labels)
        # acc = model.compute_accuracy(logits, labels)
        # print(loss, acc)
        break
