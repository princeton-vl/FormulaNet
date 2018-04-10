import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo


class input_data(object):
    def __init__(self, G):
        self.onehot  # Tensor (num_nodes*input_dim) one hot from dict
        self.index1  # Long Tensor (num_pairs) (source) self.index2  # Long Tensor (num_pairs) (target)
        self.mat  #  Tensor for index add (num_nodes * num_pairs)


class ForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim, nLayers=2, bias=False):
        super(ForwardBlock, self).__init__()
        self.linear = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.relu = nn.ModuleList()
        self.nLayers = nLayers
        for i in range(nLayers):
            if i == 0: l = input_dim
            else: l = output_dim
            self.linear.append(nn.Linear(l, output_dim, bias))
            self.bn.append(nn.BatchNorm1d(output_dim))
            self.relu.append(nn.ReLU(inplace=True))
            init.kaiming_normal(self.linear[-1].weight)

    def forward(self, x):
        for i in range(self.nLayers):
            x = self.linear[i](x)
            x = self.bn[i](x)
            x = self.relu[i](x)
        return x


class FullyConnectedNet(nn.Module):
    def __init__(self, *layers):
        '''
        layers : list of int
            There are dimensions in the sequence
        '''
        super(FullyConnectedNet, self).__init__()
        self.linear = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.relu = nn.ModuleList()
        pre_dim = layers[0]
        self.nLayers = 0
        for dim in layers[1:]:
            self.linear.append(nn.Linear(pre_dim, dim, bias=False))
            self.bn.append(nn.BatchNorm1d(dim))
            self.relu.append(nn.ReLU(inplace=True))
            init.kaiming_normal(self.linear[-1].weight)
            self.nLayers += 1
            pre_dim = dim

    def forward(self, x):
        for i in range(self.nLayers):
            x = self.linear[i](x)
            x = self.bn[i](x)
            x = self.relu[i](x)
        return x


def maxpoolcat(x1, x2):
    x1 = x1.max(0)[0]
    x2 = x2.max(0)[0]
    x = torch.cat((x1, x2), 1)
    return x


def concat_em_uc(stmt, conj):
    stmt = stmt.max(0)[0]
    conj = conj.max(0)[0]
    cov = stmt * conj
    x = torch.cat((cov, conj), 1)
    return x


def em(stmt, conj):
    stmt = stmt.max(0)[0]
    conj = conj.max(0)[0]
    cov = stmt * conj
    return cov


def dot_max(x1, x2):
    return torch.mm(x1, x2.t()).max()

def dot_mean(x1, x2):
    return torch.mm(x1, x2.t()).mean()

def meanpoolcat(x1, x2):
    x1 = x1.mean(0)
    x2 = x2.mean(0)
    x = torch.cat((x1, x2), 1)
    return x


def maxpoolpair(conj, stmt):
    conj = conj.max(0)[0]
    conj = conj.repeat(stmt.size()[0], 1)
    return torch.cat((conj, stmt), 1)


class GraphNet(nn.Module):
    def __init__(self,
                 input_dim,
                 nFeats,
                 nLayers,
                 block='normal',
                 depth=2,
                 bias=False,
                 short_cut=False,
                 direction=False,
                 loss=None,
                 binary=False,
                 no_step_supervision=False,
                 tied_weight=False,
                 compatible=False):  # compatible is used to run old model
        super(GraphNet, self).__init__()
        self.no_step_supervision = no_step_supervision
        self.input_dim = input_dim
        self.nFeats = nFeats
        self.nLayers = nLayers
        self.step = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        if compatible:
            self.l1 = nn.Linear(input_dim, nFeats, bias=False)
        else:
            self.l1 = nn.Embedding(input_dim, nFeats, sparse=False)
        init.kaiming_normal(self.l1.weight)
        self.l2 = nn.ModuleList()
        self.bn = nn.BatchNorm1d(nFeats)
        self.short_cut = short_cut
        self.direction = direction
        self.binary = binary
        if loss == 'pair':
            self.pair_forward = PairForward(2 * nFeats, nFeats, nFeats // 2,
                                            nFeats // 2)
        if self.direction or self.binary:
            self.step_out = nn.ModuleList()
        if self.binary:
            self.step_binary = nn.ModuleList()
        self.tied_weight = tied_weight
        if tied_weight:
            self.step = block_dict[block](nFeats * 2, nFeats, depth, bias)
            if self.direction or self.binary:
                self.step_out = block_dict[block](nFeats * 2, nFeats, depth, bias)
            if self.binary:
                self.step_binary = block_dict[block](nFeats*3, nFeats*3, depth, bias)
            if short_cut:
                self.l2 = block_dict[block](nFeats,  nFeats, 1 , bias)
        else:
            for i in range(nLayers):
                self.step.append(block_dict[block](nFeats * 2, nFeats, depth, bias))
                if self.direction or self.binary:
                    self.step_out.append(
                        block_dict[block](nFeats * 2, nFeats, depth, bias))
                if self.binary:
                    self.step_binary.append(block_dict[block](nFeats*3, nFeats*3, depth, bias))
                if short_cut and i < nLayers-1:
                    self.l2.append(block_dict[block](nFeats,  nFeats, 1 , bias))

    def forward(self, data, conj=None):
        # [onehot , index1 , index2 , mat]
        # if self.direction == true:
        # [onehot, iindex1, iindex2, imat, oindex1, oindex2, omat]
        #print (len(data))
        x = self.l1(data[0])
        x = self.bn(x)
        x = self.relu(x)
        if self.no_step_supervision:
            out = None
        else:
            out = [x]
        if self.tied_weight:
            for i in range(self.nLayers):
                if conj is not None:
                    z = torch.cat((x, conj), 0)
                else:
                    z = x
                y = self.step(torch.cat((z[data[1]], z[data[2]]), 1))
                z = torch.mm(data[3], y)
                if self.direction or self.binary:
                    y_out = self.step_out(torch.cat((x[data[4]], x[data[5]]), 1))
                    z_out = torch.mm(data[6], y_out)
                    z = z + z_out
                if self.binary and data[7].size()[0] > 1:
                    y_bi = self.step_binary(torch.cat( (x[data[7][0] ], x[data[7][1] ], x[data[7][2] ]) , 1))
                    z_bi = torch.mm(data[8], y_bi.view(-1 , self.nFeats))
                    z = z + z_bi
                if self.no_step_supervision:
                    out = [z]
                else:
                    out.append(z)
                if self.short_cut and i < self.nLayers-1:
                    x = x + z
                    x = self.l2(x)
                else:
                    x = z
        else:
            for i in range(self.nLayers):
                if conj is not None:
                    z = torch.cat((x, conj), 0)
                else:
                    z = x
                y = self.step[i](torch.cat((z[data[1]], z[data[2]]), 1))
                z = torch.mm(data[3], y)
                if self.direction or self.binary:
                    y_out = self.step_out[i](torch.cat((x[data[4]], x[data[5]]), 1))
                    z_out = torch.mm(data[6], y_out)
                    z = z + z_out
                if self.binary and data[7].size()[0] > 1:
                    y_bi = self.step_binary[i](torch.cat( (x[data[7][0] ], x[data[7][1] ], x[data[7][2] ]) , 1))
                    z_bi = torch.mm(data[8], y_bi.view(-1 , self.nFeats))
                    z = z + z_bi
                if self.no_step_supervision:
                    out = [z]
                else:
                    out.append(z)
                if self.short_cut and i < self.nLayers-1:
                    x = x + z
                    x = self.l2[i](x)
                else:
                    x = z
        return out


class PairForward(nn.Module):
    def __init__(self, nFeats_in, nFeats1, nFeats2, nFeats_out, bias=False):
        super(PairForward, self).__init__()
        self.l1 = nn.Linear(nFeats_in, nFeats1, bias=bias)
        init.kaiming_normal(self.l1.weight)
        self.bn1 = nn.BatchNorm1d(nFeats1)
        self.relu1 = nn.ReLU(inplace=True)

        self.l2 = nn.Linear(nFeats1, nFeats2, bias=bias)
        init.kaiming_normal(self.l2.weight)
        self.bn2 = nn.BatchNorm1d(nFeats2)
        self.relu2 = nn.ReLU(inplace=True)

        self.l3 = nn.Linear(nFeats2, nFeats_out, bias=bias)
        init.kaiming_normal(self.l3.weight)
        self.bn3 = nn.BatchNorm1d(nFeats_out)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, data):
        x = self.l1(data)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.l3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x.max(0)[0]


block_dict = {'normal': ForwardBlock}
