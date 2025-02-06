import sys
_module = sys.modules[__name__]
del sys
GenerateTrainData = _module
contourTranPoint = _module
Preprocess = _module
dataset = _module
datasets = _module
main = _module
miscc = _module
config = _module
getContour = _module
logger = _module
render3D = _module
utils = _module
vutils = _module
model = _module
model_LSTM = _module
trainer = _module
trainer_evaluator = _module
trainer_generator = _module
datasets = _module
functional = _module
main = _module
utils = _module
model_graph = _module
regionProcessing = _module
trainer = _module
transforms = _module
vutils = _module
config = _module
main = _module
train_utils = _module
maskgradientloss = _module
network = _module
prepareTemplates = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchvision, types, typing, uuid, warnings
import operator as op
from dataclasses import dataclass
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'
xrange = range
wraps = functools.wraps


import time


import math


import torch


import functools


import numpy as np


from torchvision import transforms


import torch.utils.data as data


import random


import torch.nn as nn


import torch.optim as optim


import torch.backends.cudnn as cudnn


import matplotlib.pyplot as plt


import scipy.sparse as sp


import numbers


import collections


import warnings


import torchvision.transforms as transforms


from torch.nn.parameter import Parameter


from torch.nn.modules.module import Module


import torch.nn.functional as F


import types


import torch.utils.data


import torchvision.utils as vutils


import torch.utils.hooks as hooks


from torchvision import models


from torch.utils.data import Dataset


class LayoutEvaluator(nn.Module):

    def __init__(self, room_dim, room_hiddern_dim, score_hiddern_dim, bidirectional=True):
        super(LayoutEvaluator, self).__init__()
        self.LayoutSeq = nn.LSTM(input_size=room_dim, hidden_size=room_hiddern_dim, bidirectional=bidirectional)
        if bidirectional:
            score_dim = [2 * room_hiddern_dim, score_hiddern_dim, 1]
        else:
            score_dim = [room_hiddern_dim, score_hiddern_dim, 1]
        self.mlp = self.build_mlp(dim_list=score_dim)

    def build_mlp(self, dim_list):
        layers = []
        for i in range(len(dim_list) - 1):
            dim_in, dim_out = dim_list[i], dim_list[i + 1]
            layers.append(nn.Linear(dim_in, dim_out))
            if i + 1 == len(dim_list) - 1:
                layers.append(nn.BatchNorm1d(dim_out))
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.BatchNorm1d(dim_out))
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, input):
        layout_feature, _ = self.LayoutSeq(input)
        layout_feature = layout_feature[-1]
        Score = self.mlp(layout_feature)
        return Score


class LayoutGenerator(nn.Module):

    def __init__(self, room_dim, room_hiddern_dim, room_gen_hiddern_dim, init_hidden_dim, max_len, logger=None, bidirectional=False):
        super(LayoutGenerator, self).__init__()
        self.encoder = nn.LSTM(input_size=room_dim, hidden_size=room_hiddern_dim, bidirectional=bidirectional)
        self.decoder = nn.LSTM(input_size=room_hiddern_dim, hidden_size=room_hiddern_dim, bidirectional=False)
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.01)
            elif 'weight' in name:
                nn.init.orthogonal(param)
        gen_dim = [room_hiddern_dim, room_gen_hiddern_dim, 4]
        self.mlp = self.build_mlp(dim_list=gen_dim)
        self.logger = logger

    def build_process(self, dim_list):
        layers = []
        for i in range(len(dim_list) - 1):
            dim_in, dim_out = dim_list[i], dim_list[i + 1]
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def build_mlp(self, dim_list):
        layers = []
        for i in range(len(dim_list) - 1):
            dim_in, dim_out = dim_list[i], dim_list[i + 1]
            layers.append(nn.Linear(dim_in, dim_out))
            if i + 1 == len(dim_list) - 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def forward(self, room):
        encoder_outputs, encoder_hidden = self.encoder(room)
        layout_feature = encoder_outputs.transpose(0, 1)
        room_point = self.mlp(layout_feature)
        room_point = room_point.transpose(0, 1)
        return room_point


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if cfg.CUDA:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            if cfg.CUDA:
                self.bias = Parameter(torch.FloatTensor(out_features))
            else:
                self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


class MaskedGradient(nn.Module):
    """docstring for MaskedGradient"""

    def __init__(self, opt):
        super(MaskedGradient, self).__init__()
        self.opt = opt
        self.dx = nn.Conv2d(in_channels=opt.nc, out_channels=opt.nc, kernel_size=(1, 3), stride=1, padding=0, bias=False, groups=3)
        self.dy = nn.Conv2d(in_channels=opt.nc, out_channels=opt.nc, kernel_size=(3, 1), stride=1, padding=0, bias=False, groups=3)
        self.dx.weight.requires_grad = False
        self.dy.weight.requires_grad = False
        self._init_weights()
        self.criterion = nn.L1Loss()
        self.gamma = 1

    def _init_weights(self):
        weights_dx = torch.FloatTensor([1, 0, -1])
        weights_dy = torch.FloatTensor([[1], [0], [-1]])
        for i in range(self.dx.weight.size(0)):
            for j in range(self.dx.weight.size(1)):
                self.dx.weight.data[i][j].copy_(weights_dx)
        for i in range(self.dy.weight.size(0)):
            for j in range(self.dy.weight.size(1)):
                self.dy.weight.data[i][j].copy_(weights_dy)

    def _normalize(self, inputs):
        eps = 1e-05
        inputs_view = inputs.view(inputs.size(0), -1)
        min_element, _ = torch.min(inputs_view, 1)
        min_element = min_element.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        max_element, _ = torch.max(inputs_view, 1)
        max_element = max_element.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        outputs = (inputs - min_element) / (max_element - min_element + eps)
        return outputs

    def _abs_normalize(self, inputs):
        eps = 1e-05
        inputs_view = inputs.view(inputs.size(0), -1)
        f_norm = torch.norm(inputs_view, 2, 1)
        f_norm = f_norm.view(inputs.size(0), 1, 1, 1).expand_as(inputs)
        outputs = inputs / (f_norm + eps)
        return outputs

    def _combine_gradient_xy(self, gradient_x, gradient_y):
        eps = 0.0001
        return torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2) + eps)

    def _padding_gradient(self, gradient_x, gradient_y):
        output_x = F.pad(gradient_x, (1, 1, 0, 0), 'constant', 0)
        output_y = F.pad(gradient_y, (0, 0, 1, 1), 'constant', 0)
        return output_x, output_y

    def forward(self, inputs, targets):
        inputs_grad_x = self.dx(inputs)
        inputs_grad_y = self.dy(inputs)
        targets_grad_x = self.dx(targets.detach())
        targets_grad_y = self.dy(targets.detach())
        inputs_grad_x, inputs_grad_y = self._padding_gradient(inputs_grad_x, inputs_grad_y)
        targets_grad_x, targets_grad_y = self._padding_gradient(targets_grad_x, targets_grad_y)
        inputs_grad = self._combine_gradient_xy(inputs_grad_x, inputs_grad_y)
        targets_grad = self._combine_gradient_xy(targets_grad_x, targets_grad_y)
        targets_mask = targets_grad
        grad_loss = self.criterion(inputs_grad, targets_mask.detach())
        loss = grad_loss * 1
        return loss, inputs_grad, targets_mask


KER = 5


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.gamma = nn.Parameter(torch.zeros(1))

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(KER // 2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(KER // 2)]
        elif padding_type == 'zero':
            p = KER // 2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=KER, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(KER // 2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(KER // 2)]
        elif padding_type == 'zero':
            p = KER // 2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=KER, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.gamma * self.conv_block(x)
        return out


parser = argparse.ArgumentParser()


opt = parser.parse_args()


norma = nn.BatchNorm2d


class NetED(nn.Module):

    def __init__(self, ngf, nDep, nz=0, Ubottleneck=-1, nc=3, ncIn=None, bTanh=True, lessD=0, bCopyIn=False):
        super(NetED, self).__init__()
        self.nDep = nDep
        self.eblocks = nn.ModuleList()
        self.dblocks = nn.ModuleList()
        self.bCopyIn = bCopyIn
        if Ubottleneck <= 0:
            Ubottleneck = ngf * 2 ** (nDep - 1)
        if ncIn is None:
            of = nc
        else:
            of = ncIn
        of += bfirstNoise * nz
        for i in range(self.nDep):
            layers = []
            if i == self.nDep - 1:
                nf = Ubottleneck
            else:
                nf = ngf * 2 ** i
            layers += [nn.Conv2d(of, nf, 5, 2, 2)]
            if i != 0:
                layers += [norma(nf)]
            if i < self.nDep - 1:
                layers += [nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers += [nn.Tanh()]
            of = nf
            block = nn.Sequential(*layers)
            self.eblocks += [block]
        of = nz
        for i in range(nDep + lessD):
            layers = []
            if i == nDep - 1:
                nf = nc
            else:
                nf = ngf * 2 ** (nDep - 2 - i)
            for j in range(opt.nBlocks):
                layers += [ResnetBlock(of, padding_type='zero', norm_layer=norma, use_dropout=False, use_bias=True)]
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            layers += [nn.Conv2d(of, nf, 4 + 1, 1, 2)]
            if i == nDep - 1:
                if bTanh:
                    layers += [nn.Tanh()]
            else:
                layers += [norma(nf)]
                layers += [nn.ReLU(True)]
            of = nf
            block = nn.Sequential(*layers)
            self.dblocks += [block]

    def e(self, x):
        for i in range(self.nDep):
            x = self.eblocks[i].forward(x)
        return x

    def d(self, x):
        for i in range(len(self.dblocks)):
            x = self.dblocks[i].forward(x)
        return x

    def forward(self, input1, input2=None):
        raise Exception


class NetUskip(nn.Module):

    def __init__(self, ngf, nDep, nz=0, Ubottleneck=-1, nc=3, ncIn=None, bSkip=True, bTanh=True, bCopyIn=False):
        super(NetUskip, self).__init__()
        self.nDep = nDep
        self.eblocks = nn.ModuleList()
        self.dblocks = nn.ModuleList()
        self.bSkip = bSkip
        self.bCopyIn = bCopyIn
        if Ubottleneck <= 0:
            Ubottleneck = ngf * 2 ** (nDep - 1)
        if ncIn is None:
            of = nc
        else:
            of = ncIn
        of += bfirstNoise * nz
        for i in range(self.nDep):
            layers = []
            if i == self.nDep - 1:
                nf = Ubottleneck
            else:
                nf = ngf * 2 ** i
            if i > 0 and self.bCopyIn:
                of += 2
            layers += [nn.Conv2d(of, nf, 5, 2, 2)]
            if i != 0:
                layers += [norma(nf)]
            if i < self.nDep - 1:
                layers += [nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers += [nn.Tanh()]
            of = nf
            block = nn.Sequential(*layers)
            self.eblocks += [block]
        of = nz + Ubottleneck - bfirstNoise * nz
        for i in range(nDep):
            layers = []
            if i == nDep - 1:
                nf = nc
            else:
                nf = ngf * 2 ** (nDep - 2 - i)
            if i > 0 and self.bSkip:
                of *= 2
                if self.bCopyIn:
                    of += 2
            for j in range(opt.nBlocks):
                layers += [ResnetBlock(of, padding_type='zero', norm_layer=norma, use_dropout=False, use_bias=True)]
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            layers += [nn.Conv2d(of, nf, 5, 1, 2)]
            if i == nDep - 1:
                if bTanh:
                    layers += [nn.Tanh()]
            else:
                layers += [norma(nf)]
                layers += [nn.ReLU(True)]
            of = nf
            block = nn.Sequential(*layers)
            self.dblocks += [block]

    def forward(self, input1, input2=None):
        if bfirstNoise and input2 is not None:
            x = torch.cat([input1, nn.functional.upsample(input2, scale_factor=2 ** self.nDep, mode='bilinear')], 1)
            input2 = None
        else:
            x = input1
        skips = []
        input1 = input1[:, 3:5]
        for i in range(self.nDep):
            if i > 0 and self.bCopyIn:
                input1 = nn.functional.avg_pool2d(input1, int(2))
                x = torch.cat([x, input1], 1)
            x = self.eblocks[i].forward(x)
            if i != self.nDep - 1:
                if self.bCopyIn:
                    skips += [torch.cat([x, nn.functional.avg_pool2d(input1, int(2))], 1)]
                else:
                    skips += [x]
        bottle = x
        if input2 is not None:
            bottle = torch.cat((x, input2), 1)
        x = bottle
        for i in range(len(self.dblocks)):
            x = self.dblocks[i].forward(x)
            if i < self.nDep - 1 and self.bSkip:
                x = torch.cat((x, skips[-1 - i]), 1)
        return x


def getTemplateMixImage(mix, templates, mode='bilinear'):
    if type(mix) is list:
        out = []
        for xx in mix:
            out.append(getTemplateMixImage(xx, templates, mode))
        return out
    nFT = templates.shape[4] // mix.shape[3]
    if nFT > 1:
        mix = F.upsample(mix, scale_factor=nFT, mode=mode)
    N = mix.shape[1]
    B = mix.shape[0]
    H = mix.shape[2]
    W = mix.shape[3]
    C = templates.shape[2]
    mix = mix.permute(0, 2, 3, 1).contiguous().view(-1, 1, N)
    templates = templates.permute(0, 3, 4, 1, 2).contiguous().view(-1, N, C)
    prod = torch.bmm(mix, templates)
    return prod.view(B, H, W, C).permute(0, 3, 1, 2)


class NetU_MultiScale(nn.Module):

    def __init__(self, ngf, nDep, nz=0, Ubottleneck=-1, nc=3, ncIn=None, bSkip=True, bTanh=True, bCopyIn=False):
        super(NetU_MultiScale, self).__init__()
        self.nDep = nDep
        self.eblocks = nn.ModuleList()
        self.dblocks = nn.ModuleList()
        self.bSkip = bSkip
        self.bCopyIn = bCopyIn
        assert ngf > opt.N + 10
        if Ubottleneck <= 0:
            Ubottleneck = ngf * 2 ** (nDep - 1)
        if ncIn is None:
            of = nc
        else:
            of = ncIn
        of += bfirstNoise * nz
        for i in range(self.nDep):
            layers = []
            if i == self.nDep - 1:
                nf = Ubottleneck
            else:
                nf = ngf * 2 ** i
            if i > 0 and self.bCopyIn:
                of += 2
            layers += [nn.Conv2d(of, nf, 5, 2, 2)]
            if i != 0:
                layers += [norma(nf)]
            if i < self.nDep - 1:
                layers += [nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers += [nn.Tanh()]
            of = nf
            block = nn.Sequential(*layers)
            self.eblocks += [block]
        of = nz + Ubottleneck - bfirstNoise * nz
        for i in range(nDep):
            layers = []
            if i == nDep - 1:
                nf = nc
            else:
                nf = ngf * 2 ** (nDep - 2 - i)
            if i > 0 and self.bSkip:
                of *= 2
                if self.bCopyIn:
                    of += 2
            if i > 0:
                of += 3
            for j in range(opt.nBlocks):
                layers += [ResnetBlock(of, padding_type='zero', norm_layer=norma, use_dropout=False, use_bias=True)]
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            layers += [nn.Conv2d(of, nf, 4 + 1, 1, 2)]
            if i == nDep - 1:
                if bTanh:
                    layers += [nn.Tanh()]
            else:
                layers += [norma(nf)]
                layers += [nn.ReLU(True)]
            of = nf
            block = nn.Sequential(*layers)
            self.dblocks += [block]

    def forward(self, input1, input2=None, M=None):
        if bfirstNoise and input2 is not None:
            x = torch.cat([input1, nn.functional.upsample(input2, scale_factor=2 ** self.nDep, mode='bilinear')], 1)
            input2 = None
        else:
            x = input1
        skips = []
        input1 = input1[:, 3:5]
        for i in range(self.nDep):
            if i > 0 and self.bCopyIn:
                input1 = nn.functional.avg_pool2d(input1, int(2))
                x = torch.cat([x, input1], 1)
            x = self.eblocks[i].forward(x)
            if i != self.nDep - 1:
                if self.bCopyIn:
                    skips += [torch.cat([x, nn.functional.avg_pool2d(input1, int(2))], 1)]
                else:
                    skips += [x]
        bottle = x
        if input2 is not None:
            bottle = torch.cat((x, input2), 1)
        x = bottle
        with torch.no_grad():
            MM = M.view(-1, 3, M.shape[3], M.shape[4])
            mFeat = []
            for i in range(1, self.nDep):
                sc = 2 ** i
                mFeat.append(nn.functional.avg_pool2d(MM, int(sc)).view(M.shape[0], M.shape[1], 3, M.shape[3] // sc, M.shape[4] // sc))
            mFeat = mFeat[::-1]
        for i in range(len(self.dblocks)):
            x = self.dblocks[i].forward(x)
            if i < self.nDep - 1 and self.bSkip:
                x = torch.cat((x, skips[-1 - i]), 1)
            if i < self.nDep - 1:
                blendA = 4 * nn.functional.tanh(x[:, :opt.N])
                blendA = nn.functional.softmax(1 * (blendA - blendA.detach().max()), dim=1)
                mixed = getTemplateMixImage(blendA, mFeat[i])
                x = torch.cat((x, mixed), 1)
        return x


class ColorReconstruction(nn.Module):

    def __init__(self, ndf, nDep, nc=3):
        super(ColorReconstruction, self).__init__()
        layers = []
        of = nc
        nf = of
        for i in range(nDep):
            nf = ndf * 2 ** i
            layers += [nn.Conv2d(of, nf, 1)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            of = nf
        layers += [nn.Conv2d(nf, nc, 1)]
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class GLU(nn.Module):

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


class ENCODER(nn.Module):

    def __init__(self, ndf, nDep, nz, ncIn=3, bSigm=True, condition=True):
        super(ENCODER, self).__init__()
        self.condition = condition
        layers = []
        of = ncIn
        for i in range(nDep - 1):
            nf = ndf * 2 ** i
            layers += [nn.Conv2d(of, nf, 5, 2, 2)]
            if i != 0 and i != nDep - 1:
                if True:
                    layers += [norma(nf)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            of = nf
        self.main = nn.Sequential(*layers)
        self.mu_net = nn.Sequential(nn.Conv2d(of, opt.zLoc, 5, 2, 2))
        self.logvar_net = nn.Sequential(nn.Conv2d(of, opt.zLoc, 5, 2, 2))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        output = self.main(x)
        mu = self.mu_net(output)
        logvar = self.logvar_net(output)
        z = self.reparametrize(mu, logvar)
        if opt.WGAN:
            return output.mean(3).mean(2).unsqueeze(2).unsqueeze(3)
        return z, mu, logvar


class NetG(nn.Module):

    def __init__(self, ngf, nDep, nz, nc=3, condition=True):
        super(NetG, self).__init__()
        self.condition = condition
        of = nz
        layers = []
        for i in range(nDep):
            if i == nDep - 1:
                nf = nc
            else:
                nf = ngf * 2 ** (nDep - 2 - i)
            for j in range(opt.nBlocks):
                layers += [ResnetBlock(of, padding_type='zero', norm_layer=norma, use_dropout=False, use_bias=True)]
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            layers += [nn.Conv2d(of, nf, 5, 1, 2)]
            if i == nDep - 1:
                layers += [nn.Tanh()]
            else:
                layers += [norma(nf)]
                layers += [nn.ReLU(True)]
            of = nf
        self.G = nn.Sequential(*layers)

    def forward(self, z, c=None):
        if self.condition:
            c = c.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, z.shape[2], z.shape[3])
            z = torch.cat((z, c), dim=1)
        output = self.G(z)
        return output


class Discriminator(nn.Module):

    def __init__(self, ndf, nDep, ncIn=3, bSigm=True, condition=True):
        super(Discriminator, self).__init__()
        self.condition = condition
        layers = []
        of = ncIn
        for i in range(nDep - 1):
            nf = ndf * 2 ** i
            layers += [nn.Conv2d(of, nf, 5, 2, 2)]
            if i != 0 and i != nDep - 1:
                if True:
                    layers += [norma(nf)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            of = nf
        self.main = nn.Sequential(*layers)
        self.adv = nn.Sequential(nn.Conv2d(of, 1, 5, 2, 2), nn.Sigmoid())
        self.material = nn.Sequential(nn.Conv2d(of, opt.z_material, 5, 2, 2))
        self.color = nn.Sequential(nn.Conv2d(of, opt.z_color, 5, 2, 2))

    def forward(self, x):
        output = self.main(x)
        output_adv = self.adv(output)
        output_material = self.material(output)
        output_color = self.color(output)
        if opt.WGAN:
            return output.mean(3).mean(2).unsqueeze(2).unsqueeze(3)
        return output_adv, output_material, output_color


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ColorReconstruction,
     lambda: ([], {'ndf': 4, 'nDep': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (GLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayoutEvaluator,
     lambda: ([], {'room_dim': 4, 'room_hiddern_dim': 4, 'score_hiddern_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LayoutGenerator,
     lambda: ([], {'room_dim': 4, 'room_hiddern_dim': 4, 'room_gen_hiddern_dim': 4, 'init_hidden_dim': 4, 'max_len': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (MaskedGradient,
     lambda: ([], {'opt': _mock_config(nc=18)}),
     lambda: ([torch.rand([4, 18, 64, 64]), torch.rand([4, 18, 64, 64])], {}),
     False),
]

class Test_chenqi008_HPGM(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

