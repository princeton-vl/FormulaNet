import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


def l2norm2d(inputs, k):
    # k dimension to normalize
    norm = torch.sqrt(torch.sum(inputs * inputs, k)) + 1e-12
    return inputs / norm.expand_as(inputs)

class ClassifyLoss(nn.Module):
    # max pooling over nodes in one statement.
    def __init__(self, nFeats_in=None, nFeats_out=None, layer_list=None,
                 dropout=0, bias=False):
        super(ClassifyLoss, self).__init__()
        self.dropout = dropout
        if layer_list is None:
            self.list = False
            self.l1 = nn.Linear(nFeats_in, nFeats_out, bias=bias)
            if self.dropout > 0:
                self.l1dropout = nn.Dropout(self.dropout, inplace=True)
            init.kaiming_normal(self.l1.weight)
            self.l2 = nn.Linear(nFeats_out, 2)
            init.kaiming_normal(self.l2.weight)
            self.bn1 = nn.BatchNorm1d(nFeats_out)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.list = True
            self.hids = nn.ModuleList()
            self.bns = nn.ModuleList()
            if self.dropout > 0:
                self.dropout_l = nn.Dropout(inplace=True)
            for i in range(len(layer_list) - 1):
                self.hids.append(nn.Linear(layer_list[i], layer_list[i+1], bias=False))
                init.kaiming_normal(self.hids[-1].weight)
                self.bns.append(nn.BatchNorm1d(layer_list[i+1]))
            self.lout = nn.Linear(layer_list[-1], 2)
            init.kaiming_normal(self.lout.weight)
        self.crossentropy = nn.CrossEntropyLoss()
        self.score = None

    def check_result(self, y):
        y = torch.cat(y).data
        correct = self.score.eq(y).cpu().sum()
        return correct

    def forward(self, x, y):
        y_ = torch.cat(y)
        x = torch.cat(x, 0)
        if not self.list:
            x = self.l1(x)
            x = self.bn1(x)
            x = self.relu(x)
            if self.dropout > 0:
                x = self.l1dropout(x)
            x = self.l2(x)
        else:
            for i in range(len(self.hids)):
                x = self.hids[i](x)
                x = self.bns[i](x)
                x = F.relu(x, inplace=True)
                if self.dropout > 0:
                    x = self.dropout_l(x)
            x= self.lout(x)

        self.score = x.data.max(1)[1]
        loss = self.crossentropy(x, y_)
        return loss



