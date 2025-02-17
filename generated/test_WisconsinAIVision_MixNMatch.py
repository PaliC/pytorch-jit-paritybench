import sys
_module = sys.modules[__name__]
del sys
config = _module
datasets = _module
eval = _module
model_eval = _module
model_train = _module
train_first_stage = _module
train_second_stage = _module
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


import torch.utils.data as data


import torchvision.transforms as transforms


import random


import numpy as np


import pandas as pd


import torch


from copy import deepcopy


import time


import torch.backends.cudnn as cudnn


import torch.nn as nn


import torch.optim as optim


import torchvision.utils as vutils


import torch.nn.functional as F


from sklearn.cluster import KMeans


import matplotlib.pyplot as plt


from random import sample


import torch.nn.parallel


from torch.autograd import Variable


from torch.nn import Upsample


from collections import deque


from itertools import chain


class GLU(nn.Module):

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


class ResBlock(nn.Module):

    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(conv3x3(channel_num, channel_num * 2), nn.BatchNorm2d(channel_num * 2), GLU(), conv3x3(channel_num, channel_num), nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class GET_IMAGE(nn.Module):

    def __init__(self, ngf):
        super().__init__()
        self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        return self.img(h_code)


class GET_MASK(nn.Module):

    def __init__(self, ngf):
        super().__init__()
        self.img = nn.Sequential(conv3x3(ngf, 1), nn.Sigmoid())

    def forward(self, h_code):
        return self.img(h_code)


def upBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), conv3x3(in_planes, out_planes * 2), nn.BatchNorm2d(out_planes * 2), GLU())
    return block


class BACKGROUND_STAGE(nn.Module):

    def __init__(self, ngf):
        super().__init__()
        self.ngf = ngf
        in_dim = cfg.GAN.Z_DIM + cfg.FINE_GRAINED_CATEGORIES
        self.fc = nn.Sequential(nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False), nn.BatchNorm1d(ngf * 4 * 4 * 2), GLU())
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 8)
        self.upsample5 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_input, input):
        in_code = torch.cat([z_input, input], dim=1)
        out_code = self.fc(in_code).view(-1, self.ngf, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        out_code = self.upsample5(out_code)
        return out_code


def sameBlock(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes * 2), nn.BatchNorm2d(out_planes * 2), GLU())
    return block


class PARENT_STAGE(nn.Module):

    def __init__(self, ngf):
        super().__init__()
        self.ngf = ngf
        in_dim = cfg.GAN.Z_DIM + cfg.SUPER_CATEGORIES
        self.code_len = cfg.SUPER_CATEGORIES
        self.fc = nn.Sequential(nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False), nn.BatchNorm1d(ngf * 4 * 4 * 2), GLU())
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 32)
        self.upsample5 = upBlock(ngf // 32, ngf // 32)
        self.jointConv = sameBlock(cfg.SUPER_CATEGORIES + ngf // 32, ngf // 32)
        self.residual = self._make_layer(3, ngf // 32)

    def _make_layer(self, num_residual, ngf):
        layers = []
        for _ in range(num_residual):
            layers.append(ResBlock(ngf))
        return nn.Sequential(*layers)

    def forward(self, z_input, input, which):
        if which == 'code':
            in_code = torch.cat([z_input, input], dim=1)
            out_code = self.fc(in_code).view(-1, self.ngf, 4, 4)
            out_code = self.upsample1(out_code)
            out_code = self.upsample2(out_code)
            out_code = self.upsample3(out_code)
            out_code = self.upsample4(out_code)
            out_code = self.upsample5(out_code)
            h, w = out_code.size(2), out_code.size(3)
            input = input.view(-1, self.code_len, 1, 1).repeat(1, 1, h, w)
            out_code = torch.cat((out_code, input), dim=1)
            out_code = self.jointConv(out_code)
            out_code = self.residual(out_code)
            return out_code
        elif which == 'feature':
            return input
        else:
            raise ValueError('either code or feature')


class CHILD_STAGE(nn.Module):

    def __init__(self, ngf, num_residual=2):
        super().__init__()
        self.ngf = ngf
        self.code_len = cfg.FINE_GRAINED_CATEGORIES
        self.num_residual = num_residual
        self.jointConv = sameBlock(self.code_len + self.ngf, ngf * 2)
        self.residual = self._make_layer()
        self.samesample = sameBlock(ngf * 2, ngf)

    def _make_layer(self):
        layers = []
        for _ in range(self.num_residual):
            layers.append(ResBlock(self.ngf * 2))
        return nn.Sequential(*layers)

    def forward(self, h_code, code):
        h, w = h_code.size(2), h_code.size(3)
        code = code.view(-1, self.code_len, 1, 1).repeat(1, 1, h, w)
        h_c_code = torch.cat((code, h_code), 1)
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.samesample(out_code)
        return out_code


class G_NET(nn.Module):

    def __init__(self):
        super(G_NET, self).__init__()
        ngf = cfg.GAN.GF_DIM
        self.scale_fimg = nn.UpsamplingBilinear2d(size=[126, 126])
        self.background_stage = BACKGROUND_STAGE(ngf * 8)
        self.background_image = GET_IMAGE(ngf // 2)
        self.parent_stage = PARENT_STAGE(ngf * 8)
        self.parent_image = GET_IMAGE(ngf // 4)
        self.parent_mask = GET_MASK(ngf // 4)
        self.child_stage = CHILD_STAGE(ngf // 4)
        self.child_image = GET_IMAGE(ngf // 4)
        self.child_mask = GET_MASK(ngf // 4)

    def forward(self, z_code, c_code, p_code, b_code, which, only=False):
        fake_imgs = []
        fg_imgs = []
        mk_imgs = []
        fg_mk = []
        temp = self.background_stage(z_code, b_code)
        fake_img1 = self.background_image(temp)
        fake_img1_126 = self.scale_fimg(fake_img1)
        fake_imgs.append(fake_img1_126)
        p_temp = self.parent_stage(z_code, p_code, which)
        fake_img2_foreground = self.parent_image(p_temp)
        fake_img2_mask = self.parent_mask(p_temp)
        fg_masked2 = fake_img2_foreground * fake_img2_mask
        fake_img2 = fg_masked2 + fake_img1 * (1 - fake_img2_mask)
        fg_mk.append(fg_masked2)
        fake_imgs.append(fake_img2)
        fg_imgs.append(fake_img2_foreground)
        mk_imgs.append(fake_img2_mask)
        temp = self.child_stage(p_temp, c_code)
        fake_img3_foreground = self.child_image(temp)
        fake_img3_mask = self.child_mask(temp)
        fg_masked3 = torch.mul(fake_img3_foreground, fake_img3_mask)
        fake_img3 = fg_masked3 + fake_img2 * (1 - fake_img3_mask)
        fg_mk.append(fg_masked3)
        fake_imgs.append(fake_img3)
        fg_imgs.append(fake_img3_foreground)
        mk_imgs.append(fake_img3_mask)
        if only:
            return p_temp, fake_imgs[2]
        return fake_imgs, fg_imgs, mk_imgs, fg_mk


def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True))
    return block


def downBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True))
    return block


def encode_parent_and_child_img(ndf, in_c=3):
    encode_img = nn.Sequential(nn.Conv2d(in_c, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True))
    return encode_img


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.ndf = 64
        self.softmax = torch.nn.Softmax(dim=1)
        self.model_z = nn.Sequential(encode_parent_and_child_img(self.ndf), downBlock(self.ndf * 8, self.ndf * 16), Block3x3_leakRelu(self.ndf * 16, self.ndf * 8), Block3x3_leakRelu(self.ndf * 8, self.ndf * 8), nn.Conv2d(self.ndf * 8, cfg.GAN.Z_DIM, kernel_size=4, stride=4), nn.Tanh())
        self.model_b = nn.Sequential(encode_parent_and_child_img(self.ndf), downBlock(self.ndf * 8, self.ndf * 16), Block3x3_leakRelu(self.ndf * 16, self.ndf * 8), Block3x3_leakRelu(self.ndf * 8, self.ndf * 8), nn.Conv2d(self.ndf * 8, cfg.FINE_GRAINED_CATEGORIES, kernel_size=4, stride=4))
        self.model_p = nn.Sequential(encode_parent_and_child_img(self.ndf), downBlock(self.ndf * 8, self.ndf * 16), Block3x3_leakRelu(self.ndf * 16, self.ndf * 8), Block3x3_leakRelu(self.ndf * 8, self.ndf * 8), nn.Conv2d(self.ndf * 8, cfg.SUPER_CATEGORIES, kernel_size=4, stride=4))
        self.model_c = nn.Sequential(encode_parent_and_child_img(self.ndf), downBlock(self.ndf * 8, self.ndf * 16), Block3x3_leakRelu(self.ndf * 16, self.ndf * 8), Block3x3_leakRelu(self.ndf * 8, self.ndf * 8), nn.Conv2d(self.ndf * 8, cfg.FINE_GRAINED_CATEGORIES, kernel_size=4, stride=4))

    def forward(self, x_var, type_):
        code_z = self.model_z(x_var).view(-1, cfg.GAN.Z_DIM) * 4
        code_b = self.model_b(x_var).view(-1, cfg.FINE_GRAINED_CATEGORIES)
        code_p = self.model_p(x_var).view(-1, cfg.SUPER_CATEGORIES)
        code_c = self.model_c(x_var).view(-1, cfg.FINE_GRAINED_CATEGORIES)
        if type_ == 'logits':
            return code_z, code_b, code_p, code_c
        if type_ == 'softmax':
            return code_z, self.softmax(code_b), self.softmax(code_p), self.softmax(code_c)


def Down_unet(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_c * 2, 4, 2, 1), nn.BatchNorm2d(out_c * 2), GLU())


def Up_unet(in_c, out_c):
    return nn.Sequential(nn.ConvTranspose2d(in_c, out_c * 2, 4, 2, 1), nn.BatchNorm2d(out_c * 2), GLU())


class FeatureExtractor(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.first = nn.Sequential(sameBlock(in_c, 32), sameBlock(32, 32))
        self.down1 = Down_unet(32, 32)
        self.down2 = Down_unet(32, 64)
        self.down3 = Down_unet(64, 128)
        self.down4 = Down_unet(128, 256)
        self.down5 = Down_unet(256, 512)
        self.down6 = Down_unet(512, 512)
        self.up1 = Up_unet(512, 256)
        self.up2 = Up_unet(256 + 512, 512)
        self.up3 = Up_unet(512 + 256, 256)
        self.up4 = Up_unet(256 + 128, 128)
        self.up5 = Up_unet(128 + 64, 64)
        self.up6 = Up_unet(64 + 32, out_c)
        self.last = nn.Sequential(ResBlock(out_c), ResBlock(out_c))

    def forward(self, x):
        x = self.first(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(self.down6(x5))
        x = self.up2(torch.cat([x, x5], dim=1))
        x = self.up3(torch.cat([x, x4], dim=1))
        x = self.up4(torch.cat([x, x3], dim=1))
        x = self.up5(torch.cat([x, x2], dim=1))
        x = self.up6(torch.cat([x, x1], dim=1))
        return self.last(x)


class BACKGROUND_D(nn.Module):

    def __init__(self, ndf=64):
        super().__init__()
        self.encode_img = nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 0, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 0, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 1, 0, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.class_logits = nn.Sequential(nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1), nn.Sigmoid())
        self.rf_logits = nn.Sequential(nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1), nn.Sigmoid())

    def forward(self, x):
        x = self.encode_img(x)
        return self.class_logits(x), self.rf_logits(x)


class PARENT_D(nn.Module):

    def __init__(self, ndf=32):
        super().__init__()
        self.code_len = cfg.SUPER_CATEGORIES
        self.encode_mask = nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True))
        self.code_logits = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, self.code_len, kernel_size=4, stride=4))
        self.rf_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x):
        x = self.encode_mask(x)
        return self.code_logits(x).view(-1, self.code_len), self.rf_logits(x)


class CHILD_D(nn.Module):

    def __init__(self, ndf=64):
        super().__init__()
        self.code_len = cfg.FINE_GRAINED_CATEGORIES
        self.encode_img = nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True))
        self.code_logits = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, self.code_len, kernel_size=4, stride=4))
        self.rf_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, x):
        x = self.encode_img(x)
        return self.code_logits(x).view(-1, self.code_len), self.rf_logits(x)


class Gaussian(nn.Module):

    def __init__(self, std):
        super(Gaussian, self).__init__()
        self.std = std

    def forward(self, x):
        n = torch.randn_like(x) * self.std
        return x + n


class ConvT_Block(nn.Module):

    def __init__(self, in_c, out_c, k, s, p, bn=True, activation=None, noise=False, std=None):
        super(ConvT_Block, self).__init__()
        model = [nn.ConvTranspose2d(in_c, out_c, k, s, p)]
        if bn:
            model.append(nn.BatchNorm2d(out_c))
        if activation == 'relu':
            model.append(nn.ReLU())
        elif activation == 'elu':
            model.append(nn.ELU())
        elif activation == 'leaky':
            model.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            model.append(nn.Tanh())
        elif activation == 'sigmoid':
            model.append(nn.Sigmoid())
        elif activation == 'softmax':
            model.append(nn.Softmax(dim=1))
        if noise:
            model.append(Gaussian(std))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Conv_Block(nn.Module):

    def __init__(self, in_c, out_c, k, s, p=0, bn=True, activation=None, noise=False, std=None):
        super(Conv_Block, self).__init__()
        model = [nn.Conv2d(in_c, out_c, k, s, p)]
        if bn:
            model.append(nn.BatchNorm2d(out_c))
        if activation == 'relu':
            model.append(nn.ReLU())
        if activation == 'elu':
            model.append(nn.ELU())
        if activation == 'leaky':
            model.append(nn.LeakyReLU(0.2))
        if activation == 'tanh':
            model.append(nn.Tanh())
        if activation == 'sigmoid':
            model.append(nn.Sigmoid())
        if activation == 'softmax':
            model.append(nn.Softmax(dim=1))
        if noise:
            model.append(Gaussian(std))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Linear_Block(nn.Module):

    def __init__(self, in_c, out_c, bn=True, activation=None, noise=False, std=None):
        super(Linear_Block, self).__init__()
        model = [nn.Linear(in_c, out_c)]
        if bn:
            model.append(nn.BatchNorm1d(out_c))
        if activation == 'relu':
            model.append(nn.ReLU())
        if activation == 'elu':
            model.append(nn.ELU())
        if activation == 'leaky':
            model.append(nn.LeakyReLU(0.2))
        if activation == 'tanh':
            model.append(nn.Tanh())
        if activation == 'sigmoid':
            model.append(nn.Sigmoid())
        if activation == 'softmax':
            model.append(nn.Softmax(dim=1))
        if noise:
            model.append(Gaussian(std))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Viewer(nn.Module):

    def __init__(self, shape):
        super(Viewer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Bi_Dis_base(nn.Module):

    def __init__(self, code_len, ngf=16):
        super(Bi_Dis_base, self).__init__()
        self.image_layer = nn.Sequential(Conv_Block(3, ngf, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.3), Conv_Block(ngf, ngf * 2, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5), Conv_Block(ngf * 2, ngf * 4, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5), Conv_Block(ngf * 4, ngf * 8, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5), Conv_Block(ngf * 8, ngf * 16, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5), Conv_Block(ngf * 16, 512, 4, 1, 0, bn=False, activation='leaky', noise=False, std=0.5), Viewer([-1, 512]))
        self.code_layer = nn.Sequential(Linear_Block(code_len, 512, bn=False, activation='leaky', noise=True, std=0.5), Linear_Block(512, 512, bn=False, activation='leaky', noise=True, std=0.3), Linear_Block(512, 512, bn=False, activation='leaky', noise=True, std=0.3))
        self.joint = nn.Sequential(Linear_Block(1024, 1024, bn=False, activation='leaky', noise=False, std=0.5), Linear_Block(1024, 1, bn=False, activation='None'), Viewer([-1]))

    def forward(self, img, code):
        t1 = self.image_layer(img)
        t2 = self.code_layer(code)
        return self.joint(torch.cat([t1, t2], dim=1))


class Bi_Dis(nn.Module):

    def __init__(self):
        super(Bi_Dis, self).__init__()
        self.BD_z = Bi_Dis_base(cfg.GAN.Z_DIM)
        self.BD_b = Bi_Dis_base(cfg.FINE_GRAINED_CATEGORIES)
        self.BD_p = Bi_Dis_base(cfg.SUPER_CATEGORIES)
        self.BD_c = Bi_Dis_base(cfg.FINE_GRAINED_CATEGORIES)

    def forward(self, img, z_code, b_code, p_code, c_code):
        which_pair_z = self.BD_z(img, z_code)
        which_pair_b = self.BD_b(img, b_code)
        which_pair_p = self.BD_p(img, p_code)
        which_pair_c = self.BD_c(img, c_code)
        return which_pair_z, which_pair_b, which_pair_p, which_pair_c


class Dis_Dis(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.encode_img = nn.Sequential(nn.Conv2d(in_c, 32, 4, 2, 0, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(32, 64, 4, 2, 0, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, 4, 1, 0, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.rf_logits = nn.Sequential(nn.Conv2d(128, 1, kernel_size=4, stride=1), nn.Sigmoid())

    def forward(self, x):
        x = F.interpolate(x, [126, 126], mode='bilinear', align_corners=True)
        x = self.encode_img(x)
        return self.rf_logits(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BACKGROUND_D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ConvT_Block,
     lambda: ([], {'in_c': 4, 'out_c': 4, 'k': 4, 's': 4, 'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv_Block,
     lambda: ([], {'in_c': 4, 'out_c': 4, 'k': 4, 's': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dis_Dis,
     lambda: ([], {'in_c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureExtractor,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (GET_IMAGE,
     lambda: ([], {'ngf': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GET_MASK,
     lambda: ([], {'ngf': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Gaussian,
     lambda: ([], {'std': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear_Block,
     lambda: ([], {'in_c': 4, 'out_c': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'channel_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_WisconsinAIVision_MixNMatch(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

