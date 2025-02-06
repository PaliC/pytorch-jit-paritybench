import sys
_module = sys.modules[__name__]
del sys
case_brain = _module
case_cifar = _module
case_covir = _module
case_dog_cat = _module
case_face_detect = _module
case_lung_mask = _module
case_mnist = _module
case_fft = _module
OpticalNet = _module
fast_conv = _module
BinaryDNet = _module
D2NN_tf = _module
D2NNet = _module
DiffractiveLayer = _module
DropOutLayer = _module
FFT_layer = _module
Loss = _module
NET_config = _module
Net_Instance = _module
OpticalFormer = _module
OpticalFormer_util = _module
PoolForCls = _module
RGBO_CNN = _module
SparseSupport = _module
ToExcel = _module
Visualizing = _module
Z_utils = _module
onnet = _module
__version__ = _module
optical_trans = _module
some_utils = _module

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


import torch


from torch.utils.data import Dataset


from torchvision.transforms import ToPILImage


import math


from enum import Enum


import re


from torchvision.transforms import transforms


import numpy as np


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


import torchvision


import torchvision.transforms as transforms


import time


import torch.nn.init as init


import pandas as pd


import random


from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.nn import CrossEntropyLoss


import logging


from torch.optim import Adam


from torchvision import transforms


from sklearn.metrics import f1_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from sklearn.metrics import accuracy_score


from sklearn.metrics import classification_report


import matplotlib.pyplot as plt


from torchvision import datasets


from torch.utils.data.sampler import SubsetRandomSampler


from typing import Callable


from typing import Any


from typing import NamedTuple


from typing import List


from torch.optim.lr_scheduler import StepLR


import torchvision.transforms.functional as F


import matplotlib


from torch import nn


from torchvision.transforms.functional import to_grayscale


from torch.autograd import Variable


from copy import deepcopy


from torch.nn import ReflectionPad2d


from torch.nn.functional import relu


from torch.nn.functional import max_pool2d


from torch.nn.functional import dropout


from torch.nn.functional import dropout2d


class MyModel(torch.nn.Module):

    def __init__(self, in_feature):
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=in_feature, out_features=500)
        self.fc2 = torch.nn.Linear(in_features=500, out_features=100)
        self.fc3 = torch.nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 53 * 53, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x


class Fasion_Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


IMG_size = 112, 112


class Mnist_Net(nn.Module):

    def __init__(self, config, nCls=10):
        super(Mnist_Net, self).__init__()
        self.title = 'Mnist_Net'
        self.config = config
        self.config.learning_rate = 0.01
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.isDropOut = False
        self.nFC = 1
        if self.isDropOut:
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
        if IMG_size[0] == 56:
            nFC1 = 43264
        else:
            nFC1 = 9216
        if self.nFC == 1:
            self.fc1 = nn.Linear(nFC1, 10)
        else:
            self.fc1 = nn.Linear(nFC1, 128)
            self.fc2 = nn.Linear(128, 10)
        self.loss = F.cross_entropy
        self.nClass = nCls

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        if self.isDropOut:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.isDropOut:
            x = self.dropout2(x)
        if self.nFC == 2:
            x = self.fc2(x)
        output = x
        return output

    def predict(self, output):
        pred = output.max(1, keepdim=True)[1]
        return pred


class View(nn.Module):

    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1, *self.shape)


def split__sections(dim_0, nClass):
    split_dim = range(dim_0)
    sections = []
    for arr in np.array_split(np.array(split_dim), nClass):
        sections.append(len(arr))
    assert len(sections) > 0
    return sections


class DiffractiveLayer(torch.nn.Module):

    def SomeInit(self, M_in, N_in, HZ=400000000000.0):
        assert M_in == N_in
        self.M = M_in
        self.N = N_in
        self.z_modulus = Z.modulus
        self.size = M_in
        self.delta = 0.03
        self.dL = 0.02
        self.c = 300000000.0
        self.Hz = HZ
        self.H_z = self.Init_H()

    def __repr__(self):
        main_str = f'DiffractiveLayer_[{int(self.Hz / 1000000000.0)}G]_[{self.M},{self.N}]'
        return main_str

    def __init__(self, M_in, N_in, config, HZ=400000000000.0):
        super(DiffractiveLayer, self).__init__()
        self.SomeInit(M_in, N_in, HZ)
        assert config is not None
        self.config = config
        if not hasattr(self.config, 'wavelet') or self.config.wavelet is None:
            if self.config.modulation == 'phase':
                self.transmission = torch.nn.Parameter(data=torch.Tensor(self.size, self.size), requires_grad=True)
            else:
                self.transmission = torch.nn.Parameter(data=torch.Tensor(self.size, self.size, 2), requires_grad=True)
            init_param = self.transmission.data
            if self.config.init_value == 'reverse':
                half = self.transmission.data.shape[-2] // 2
                init_param[..., :half, :] = 0
                init_param[..., half:, :] = np.pi
            elif self.config.init_value == 'random':
                init_param.uniform_(0, np.pi * 2)
            elif self.config.init_value == 'random_reverse':
                init_param = torch.randint_like(init_param, 0, 2) * np.pi
            elif self.config.init_value == 'chunk':
                sections = split__sections()
                for xx in init_param.split(sections, -1):
                    xx = random.random(0, np.pi * 2)

    def visualize(self, visual, suffix, params):
        param = self.transmission.data
        name = f'{suffix}_{self.config.modulation}_'
        return visual.image(name, param, params)

    def share_weight(self, layer_1):
        tp = type(self)
        assert type(layer_1) == tp

    def Init_H(self):
        N = self.size
        df = 1.0 / self.dL
        d = self.delta
        lmb = self.c / self.Hz
        k = np.pi * 2.0 / lmb
        D = self.dL * self.dL / (N * lmb)

        def phase(i, j):
            i -= N // 2
            j -= N // 2
            return i * df * (i * df) + j * df * (j * df)
        ph = np.fromfunction(phase, shape=(N, N), dtype=np.float32)
        H = np.exp(1.0j * k * d) * np.exp(-1.0j * lmb * np.pi * d * ph)
        H_f = np.fft.fftshift(H) * self.dL * self.dL / (N * N)
        H_z = np.zeros(H_f.shape + (2,))
        H_z[..., 0] = H_f.real
        H_z[..., 1] = H_f.imag
        H_z = torch.from_numpy(H_z)
        return H_z

    def Diffractive_(self, u0, theta=0.0):
        if Z.isComplex(u0):
            z0 = u0
        else:
            z0 = u0.new_zeros(u0.shape + (2,))
            z0[..., 0] = u0
        N = self.size
        df = 1.0 / self.dL
        z0 = Z.fft(z0)
        u1 = Z.Hadamard(z0, self.H_z.float())
        u2 = Z.fft(u1, 'C2C', inverse=True)
        return u2 * N * N * df * df

    def GetTransCoefficient(self):
        """
            eps = 1e-5; momentum = 0.1; affine = True

            mean = torch.mean(self.transmission, 1)
            vari = torch.var(self.transmission, 1)
            amp_bn = torch.batch_norm(self.transmission,mean,vari)
        :return:
        """
        amp_s = Z.exp_euler(self.transmission)
        return amp_s

    def forward(self, x):
        diffrac = self.Diffractive_(x)
        amp_s = self.GetTransCoefficient()
        x = Z.Hadamard(diffrac, amp_s.float())
        if self.config.rDrop > 0:
            drop = Z.rDrop2D(1 - self.rDrop, (self.M, self.N), isComlex=True)
            x = Z.Hadamard(x, drop)
        return x


class OpticalBlock(nn.Module):
    expansion = 1

    def __init__(self, config, in_planes, planes, stride=1):
        super(OpticalBlock, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        M, N = self.config.IMG_size[0], self.config.IMG_size[1]
        self.diffrac = DiffractiveLayer(M, N, config)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class OpticalNet(nn.Module):

    def __init__(self, config, block, num_blocks):
        super(OpticalNet, self).__init__()
        num_classes = config.nClass
        self.config = config
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.config, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def shrink(x0, x1, max_sz=2):
    if x1 - x0 > max_sz:
        center = (x1 + x0) // 2
        x1 = center + max_sz // 2
        x0 = center - max_sz // 2
    return x0, x1


def split_regions_2d(shape, nClass):
    dim_1, dim_2 = shape[-1], shape[-2]
    n1 = int(math.sqrt(nClass))
    n2 = int(math.ceil(nClass / n1))
    assert n1 * n2 >= nClass
    section_1 = split__sections(dim_1, n1)
    section_2 = split__sections(dim_2, n2)
    regions = []
    x1, x2 = 0, 0
    for sec_1 in section_1:
        for sec_2 in section_2:
            box = shrink(x1, x1 + sec_1) + shrink(x2, x2 + sec_2)
            regions.append(box)
            if len(regions) >= nClass:
                break
            x2 = x2 + sec_2
        x1 = x1 + sec_1
        x2 = 0
    return regions


class ChunkPool(torch.nn.Module):

    def __init__(self, nCls, config, pooling='max', chunk_dim=-1):
        super(ChunkPool, self).__init__()
        self.nClass = nCls
        self.pooling = pooling
        self.chunk_dim = chunk_dim
        self.config = config

    def __repr__(self):
        main_str = super(ChunkPool, self).__repr__()
        main_str += f'_cls[{self.nClass}]_pool[{self.pooling}]'
        return main_str

    def forward(self, x):
        nSamp = x.shape[0]
        if False:
            x1 = torch.zeros((nSamp, self.nClass)).double()
            step = self.M // self.nClass
            for samp in range(nSamp):
                for i in range(self.nClass):
                    x1[samp, i] = torch.max(x[samp, :, :, i * step:(i + 1) * step])
            x_np = x1.detach().cpu().numpy()
            x = x1
        else:
            x_max = []
            if self.config.output_chunk == '1D':
                sections = split__sections(x.shape[self.chunk_dim], self.nClass)
                for xx in x.split(sections, self.chunk_dim):
                    x2 = xx.contiguous().view(nSamp, -1)
                    if self.pooling == 'max':
                        x3 = torch.max(x2, 1)
                        x_max.append(x3.values)
                    else:
                        x3 = torch.mean(x2, 1)
                        x_max.append(x3)
            else:
                regions = split_regions_2d(x.shape, self.nClass)
                for box in regions:
                    x2 = x[..., box[0]:box[1], box[2]:box[3]]
                    x2 = x2.contiguous().view(nSamp, -1)
                    if self.pooling == 'max':
                        x3 = torch.max(x2, 1)
                        x_max.append(x3.values)
                    else:
                        x3 = torch.mean(x2, 1)
                        x_max.append(x3)
            assert len(x_max) == self.nClass
            x = torch.stack(x_max, 1)
        return x


class GatePipe(torch.nn.Module):

    def __init__(self, M, N, nHidden, config, pooling='max'):
        super(GatePipe, self).__init__()
        self.config = config
        self.M = M
        self.N = N
        self.nHidden = nHidden
        self.pooling = pooling
        self.layers = nn.ModuleList([DiffractiveLayer(self.M, self.N, self.config, HZ=300000000000.0) for j in range(self.nHidden)])
        if True:
            chunk_dim = -1 if random.choice([True, False]) else -2
            self.pool = ChunkPool(2, self.config, pooling=self.pooling, chunk_dim=chunk_dim)
        else:
            self.pt1 = random.randint(0, self.M - 1), random.randint(0, self.N - 1)
            self.pt2 = random.randint(0, self.M - 1), random.randint(0, self.N - 1)

    def __repr__(self):
        main_str = super(GatePipe, self).__repr__()
        main_str = f'GatePipe_[{len(self.layers)}]_pool[{self.pooling}]'
        return main_str

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        x1 = Z.modulus(x)
        if True:
            x1 = self.pool(x1)
        else:
            x_pt1 = x1[:, 0, self.pt1[0], self.pt1[1]]
            x_pt2 = x1[:, 0, self.pt2[0], self.pt2[1]]
            x1 = torch.stack([x_pt1, x_pt2], 1)
        x2 = F.log_softmax(x1, dim=1)
        return x2


class DiffractiveWavelet(DiffractiveLayer):

    def __init__(self, M_in, N_in, config, HZ=400000000000.0):
        super(DiffractiveWavelet, self).__init__(M_in, N_in, config, HZ)
        self.Init_DisTrans()

    def __repr__(self):
        main_str = f'Diffrac_Wavelet_[{int(self.Hz / 1000000000.0)}G]_[{self.M},{self.N}]'
        return main_str

    def share_weight(self, layer_1):
        tp = type(self)
        assert type(layer_1) == tp
        del self.wavelet
        self.wavelet = layer_1.wavelet
        del self.dis_map
        self.dis_map = layer_1.dis_map
        del self.wav_indices
        self.wav_indices = layer_1.wav_indices

    def Init_DisTrans(self):
        origin_r, origin_c = (self.M - 1) / 2, (self.N - 1) / 2
        origin_r = random.uniform(0, self.M - 1)
        origin_c = random.uniform(0, self.N - 1)
        self.dis_map = {}
        self.wav_indices = torch.LongTensor(self.size * self.size)
        nz = 0
        for r in range(self.M):
            for c in range(self.N):
                off = np.sqrt((r - origin_r) * (r - origin_r) + (c - origin_c) * (c - origin_c))
                i_off = int(off + 0.5)
                if i_off not in self.dis_map:
                    self.dis_map[i_off] = len(self.dis_map)
                id = self.dis_map[i_off]
                self.wav_indices[nz] = id
                nz = nz + 1
        nD = len(self.dis_map)
        if False:
            plt.imshow(self.dis_trans.numpy())
            plt.show()
        self.wavelet = torch.nn.Parameter(data=torch.Tensor(nD), requires_grad=True)
        self.wavelet.data.uniform_(0, np.pi * 2)

    def GetXita(self):
        if False:
            xita = torch.zeros((self.size, self.size))
            for r in range(self.M):
                for c in range(self.N):
                    pos = self.dis_trans[r, c]
                    xita[r, c] = self.wavelet[pos]
            origin_r, origin_c = self.M / 2, self.N / 2
        else:
            xita = torch.index_select(self.wavelet, 0, self.wav_indices)
            xita = xita.view(self.size, self.size)
        return xita

    def GetTransCoefficient(self):
        xita = self.GetXita()
        amp_s = Z.exp_euler(xita)
        return amp_s

    def visualize(self, visual, suffix, params):
        xita = self.GetXita()
        name = f'{suffix}'
        return visual.image(name, torch.sin(xita.detach()), params)


class FFT_Layer(torch.nn.Module):

    def SomeInit(self, M_in, N_in, isInv=False):
        assert M_in == N_in
        self.M = M_in
        self.N = N_in
        self.isInv = isInv

    def __repr__(self):
        i_ = '_i' if self.isInv else ''
        main_str = f'FFT_Layer{i_}_[{self.M},{self.N}]'
        return main_str

    def __init__(self, M_in, N_in, config, isInv=False):
        super(FFT_Layer, self).__init__()
        self.SomeInit(M_in, N_in, isInv)
        assert config is not None
        self.config = config

    def visualize(self, visual, suffix, params):
        param = self.transmission.data
        name = f'{suffix}_{self.config.modulation}_'
        return visual.image(name, param, params)

    def Diffractive_(self, u0, theta=0.0):
        if Z.isComplex(u0):
            z0 = u0
        else:
            z0 = u0.new_zeros(u0.shape + (2,))
            z0[..., 0] = u0
        N = self.size
        df = 1.0 / self.dL
        z0 = Z.fft(z0)
        u1 = Z.Hadamard(z0, self.H_z.float())
        u2 = Z.fft(u1, 'C2C', inverse=True)
        return u2 * N * N * df * df

    def forward(self, x):
        if Z.isComplex(x):
            z0 = x
        else:
            z0 = x.new_zeros(x.shape + (2,))
            z0[..., 0] = x
        if self.isInv:
            x = Z.fft(z0, 'C2C', inverse=self.isInv)
        else:
            x = Z.fft(z0, 'C2C', inverse=self.isInv)
        x_0, x_1 = torch.min(x), torch.max(x)
        return x

    def trans(img):
        plt.figure(figsize=(10, 8))
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        f = abs(np.fft.fftshift(fftn(img))) ** 0.25 * 255 ** 3
        plt.subplot(122), plt.imshow(f, cmap='gray')
        plt.title('Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()


class SuppLayer(torch.nn.Module):


    class SUPP(Enum):
        exp, sparse, expW, diff = 'exp', 'sparse', 'expW', 'differentia'

    def __init__(self, config, nClass, nSupp=10):
        super(SuppLayer, self).__init__()
        self.nClass = nClass
        self.nSupp = nSupp
        self.nChunk = self.nClass * 2
        self.config = config
        self.w_11 = False
        if self.config.support == self.SUPP.sparse:
            if self.w_11:
                tSupp = torch.ones(self.nClass, self.nSupp)
            else:
                tSupp = torch.Tensor(self.nClass, self.nSupp).uniform_(-1, 1)
            self.wSupp = torch.nn.Parameter(tSupp)
            self.nChunk = self.nSupp * self.nSupp
            self.chunk_map = np.random.randint(self.nChunk, size=(self.nClass, self.nSupp))

    def __repr__(self):
        w_init = '1' if self.w_11 else 'random'
        main_str = f'SupportLayer supp=({self.nSupp},{w_init}) type="{self.config.support}" nChunk={self.nChunk}'
        return main_str

    def sparse_support(self, x):
        feats = []
        for i in range(self.nClass):
            feat = 0
            for j in range(self.nSupp):
                col = int(self.chunk_map[i, j])
                feat += x[:, col] * self.wSupp[i, j]
            feats.append(torch.exp(feat))
        output = torch.stack(feats, 1)
        return output

    def forward(self, x):
        if self.config.support == self.SUPP.sparse:
            output = self.sparse_support(x)
            return output
        assert x.shape[1] == self.nClass * 2
        if self.config.support == self.SUPP.diff:
            for i in range(self.nClass):
                x[:, i] = (x[:, 2 * i] - x[:, 2 * i + 1]) / (x[:, 2 * i] + x[:, 2 * i + 1])
            output = x[..., 0:self.nClass]
        elif self.config.support == self.SUPP.exp:
            for i in range(self.nClass):
                x[:, i] = torch.exp(x[:, 2 * i] - x[:, 2 * i + 1])
            output = x[..., 0:self.nClass]
        elif self.config.support == self.SUPP.expW:
            output = torch.zeros_like(x)
            for i in range(self.nClass):
                output[:, i] = torch.exp(x[:, 2 * i] * self.w2[0] - x[:, 2 * i + 1] * self.w2[1])
            output = output[..., 0:self.nClass]
        return output


class UserLoss(object):

    @staticmethod
    def cys_loss(output, target, reduction='mean'):
        loss = F.cross_entropy(output, target, reduction=reduction)
        return loss


class D2NNet(nn.Module):

    @staticmethod
    def binary_loss(output, target, reduction='mean'):
        nSamp = target.shape[0]
        nGate = output.shape[1] // 2
        loss = 0
        for i in range(nGate):
            target_i = target % 2
            val_2 = torch.stack([output[:, 2 * i], output[:, 2 * i + 1]], 1)
            loss_i = F.cross_entropy(val_2, target_i, reduction=reduction)
            loss += loss_i
            target = (target - target_i) / 2
        return loss

    @staticmethod
    def logit_loss(output, target, reduction='mean'):
        nSamp = target.shape[0]
        nGate = output.shape[1]
        loss = 0
        loss_BCE = nn.BCEWithLogitsLoss()
        for i in range(nGate):
            target_i = target % 2
            out_i = output[:, i]
            loss_i = loss_BCE(out_i, target_i.double())
            loss += loss_i
            target = (target - target_i) / 2
        return loss

    def predict(self, output):
        if self.config.support == 'binary':
            nGate = output.shape[1] // 2
            pred = 0
            for i in range(nGate):
                no = 2 * (nGate - 1 - i)
                val_2 = torch.stack([output[:, no], output[:, no + 1]], 1)
                pred_i = val_2.max(1, keepdim=True)[1]
                pred = pred * 2 + pred_i
        elif self.config.support == 'logit':
            nGate = output.shape[1]
            pred = 0
            for i in range(nGate):
                no = nGate - 1 - i
                val_2 = F.sigmoid(output[:, no])
                pred_i = (val_2 + 0.5).long()
                pred = pred * 2 + pred_i
        else:
            pred = output.max(1, keepdim=True)[1]
        return pred

    def GetLayer_(self):
        if self.config.wavelet is None:
            layer = DiffractiveLayer
        else:
            layer = DiffractiveWavelet
        return layer

    def __init__(self, IMG_size, nCls, nDifrac, config):
        super(D2NNet, self).__init__()
        self.M, self.N = IMG_size
        self.z_modulus = Z.modulus
        self.nDifrac = nDifrac
        self.nClass = nCls
        self.config = config
        self.title = f'DNNet'
        self.highWay = 1
        if self.config.input_plane == 'fourier':
            self.highWay = 0
        if hasattr(self.config, 'feat_extractor'):
            if self.config.feat_extractor != 'last_layer':
                self.feat_extractor = []
        if self.config.output_chunk == '2D':
            assert self.M * self.N >= self.nClass
        else:
            assert self.M >= self.nClass and self.N >= self.nClass
        None
        layer = self.GetLayer_()
        self.DD = nn.ModuleList([layer(self.M, self.N, config) for i in range(self.nDifrac)])
        if self.config.input_plane == 'fourier':
            self.DD.insert(0, FFT_Layer(self.M, self.N, config, isInv=False))
            self.DD.append(FFT_Layer(self.M, self.N, config, isInv=True))
        self.nD = len(self.DD)
        self.laySupp = None
        if self.highWay > 0:
            self.wLayer = torch.nn.Parameter(torch.ones(len(self.DD)))
            if self.highWay == 2:
                self.wLayer.data.uniform_(-1, 1)
            elif self.highWay == 1:
                self.wLayer = torch.nn.Parameter(torch.ones(len(self.DD)))
        if self.config.isFC:
            self.fc1 = nn.Linear(self.M * self.N, self.nClass)
            self.loss = UserLoss.cys_loss
            self.title = f'DNNet_FC'
        elif self.config.support != None:
            self.laySupp = SuppLayer(config, self.nClass)
            self.last_chunk = ChunkPool(self.laySupp.nChunk, config, pooling=config.output_pooling)
            self.loss = UserLoss.cys_loss
            a = self.config.support.value
            self.title = f'DNNet_{self.config.support.value}'
        else:
            self.last_chunk = ChunkPool(self.nClass, config, pooling=config.output_pooling)
            self.loss = UserLoss.cys_loss
        if self.config.wavelet is not None:
            self.title = self.title + f'_W'
        if self.highWay > 0:
            self.title = self.title + f'_H'
        if self.config.custom_legend is not None:
            self.title = self.title + f'_{self.config.custom_legend}'
        """ 
        BinaryChunk is pool
        elif self.config.support=="binary":
            self.last_chunk = BinaryChunk(self.nClass, pooling="max")
            self.loss = D2NNet.binary_loss
            self.title = f"DNNet_binary"
        elif self.config.support == "logit":
            self.last_chunk = BinaryChunk(self.nClass, isLogit=True, pooling="max")
            self.loss = D2NNet.logit_loss
        """

    def visualize(self, visual, suffix):
        no = 0
        for plot in visual.plots:
            images, path = [], ''
            if plot['object'] == 'layer pattern':
                path = f'{visual.img_dir}/{suffix}.jpg'
                for no, layer in enumerate(self.DD):
                    info = f'{suffix},{no}]'
                    title = f'layer_{no + 1}'
                    if self.highWay == 2:
                        a = self.wLayer[no]
                        a = torch.sigmoid(a)
                        info = info + f'_{a:.2g}'
                    elif self.highWay == 1:
                        a = self.wLayer[no]
                        info = info + f'_{a:.2g}'
                        title = title + f' w={a:.2g}'
                    image = layer.visualize(visual, info, {'save': False, 'title': title})
                    images.append(image)
                    no = no + 1
            if len(images) > 0:
                image_all = np.concatenate(images, axis=1)
                cv2.imwrite(path, image_all)

    def legend(self):
        if self.config.custom_legend is not None:
            leg_ = self.config.custom_legend
        else:
            leg_ = self.title
        return leg_

    def __repr__(self):
        main_str = super(D2NNet, self).__repr__()
        main_str += f'\n========init={self.config.init_value}'
        return main_str

    def input_trans(self, x):
        if True:
            x = x * self.config.input_scale
            x_0, x_1 = torch.min(x).item(), torch.max(x).item()
            assert x_0 >= 0
            x = torch.sqrt(x)
        else:
            x = Z.exp_euler(x * 2 * math.pi).float()
            x_0, x_1 = torch.min(x).item(), torch.max(x).item()
        return x

    def do_classify(self, x):
        if self.config.isFC:
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            return x
        x = self.last_chunk(x)
        if self.laySupp != None:
            x = self.laySupp(x)
        return x

    def OnLayerFeats(self):
        pass

    def forward(self, x):
        if hasattr(self, 'feat_extractor'):
            self.feat_extractor.clear()
        nSamp, nChannel = x.shape[0], x.shape[1]
        assert nChannel == 1
        if nChannel > 1:
            no = random.randint(0, nChannel - 1)
            x = x[:, 0:1, ...]
        x = self.input_trans(x)
        if hasattr(self, 'visual'):
            self.visual.onX(x.cpu(), f'X@input')
        summary = 0
        for no, layD in enumerate(self.DD):
            info = layD.__repr__()
            x = layD(x)
            if hasattr(self, 'feat_extractor'):
                self.feat_extractor.append((self.z_modulus(x), self.wLayer[no]))
            if hasattr(self, 'visual'):
                self.visual.onX(x, f'X@{no + 1}')
            if self.highWay == 2:
                s = torch.sigmoid(self.wLayer[no])
                summary += x * s
                x = x * (1 - s)
            elif self.highWay == 1:
                summary += x * self.wLayer[no]
            elif self.highWay == 3:
                summary += self.z_modulus(x) * self.wLayer[no]
        if self.highWay == 2:
            x = x + summary
            x = self.z_modulus(x)
        elif self.highWay == 1:
            x = summary
            x = self.z_modulus(x)
        elif self.highWay == 3:
            x = summary
        elif self.highWay == 0:
            x = self.z_modulus(x)
        if hasattr(self, 'visual'):
            self.visual.onX(x, f'X@output')
        if hasattr(self, 'feat_extractor'):
            return
        elif hasattr(self.config, 'feat_extractor') and self.config.feat_extractor == 'last_layer':
            return x
        else:
            output = self.do_classify(x)
            return output


class MultiDNet(D2NNet):

    def __init__(self, IMG_size, nCls, nInterDifrac, freq_list, config, shareWeight=True):
        super(MultiDNet, self).__init__(IMG_size, nCls, nInterDifrac, config)
        self.isShareWeight = shareWeight
        self.freq_list = freq_list
        nFreq = len(self.freq_list)
        del self.DD
        self.DD = None
        self.wFreq = torch.nn.Parameter(torch.ones(nFreq))
        layer = self.GetLayer_()
        self.freq_nets = nn.ModuleList([nn.ModuleList([layer(self.M, self.N, self.config, HZ=freq) for i in range(self.nDifrac)]) for freq in freq_list])
        if self.isShareWeight:
            nSubNet = len(self.freq_nets)
            net_0 = self.freq_nets[0]
            for i in range(1, nSubNet):
                net_1 = self.freq_nets[i]
                for j in range(self.nDifrac):
                    net_1[j].share_weight(net_0[j])

    def legend(self):
        if self.config.custom_legend is not None:
            leg_ = self.config.custom_legend
        else:
            title = f'MF_DNet({len(self.freq_list)} channels)'
        return title

    def __repr__(self):
        main_str = super(MultiDNet, self).__repr__()
        main_str += f'\nfreq_list={self.freq_list}_'
        return main_str

    def forward(self, x0):
        nSamp = x0.shape[0]
        x_sum = 0
        for id, fNet in enumerate(self.freq_nets):
            x = self.input_trans(x0)
            for layD in fNet:
                x = layD(x)
            x_sum += self.z_modulus(x) * self.wFreq[id]
        x = x_sum
        output = self.do_classify(x)
        return output


class DiffractiveAMP(DiffractiveLayer):

    def __init__(self, M_in, N_in, rDrop=0.0):
        super(DiffractiveAMP, self).__init__(M_in, N_in, rDrop, params='amp')
        self.transmission.data.uniform_(0, 1)

    def GetTransCoefficient(self):
        amp_s = self.transmission
        return amp_s


class DropOutLayer(torch.nn.Module):

    def __init__(self, M_in, N_in, drop=0.5):
        super(DropOutLayer, self).__init__()
        assert M_in == N_in
        self.M = M_in
        self.N = N_in
        self.rDrop = drop

    def forward(self, x):
        assert Z.isComplex(x)
        nX = x.numel() // 2
        d_shape = x.shape[:-1]
        drop = np.random.binomial(1, self.rDrop, size=d_shape).astype(np.float)
        drop = torch.from_numpy(drop)
        x[..., 0] *= drop
        x[..., 1] *= drop
        return x


class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.fn is None:
            x = self.norm(x)
        else:
            x = self.fn(self.norm(x), **kwargs)
        return x


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class QK_Attention(nn.Module):

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_project = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = QK_Attention()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        if self.attention is None:
            x = self.dropout(x)
        else:
            if self.h == 1:
                query, key, value = x, x, x
            else:
                query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linear_project, (x, x, x))]
            x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
            if self.h > 1:
                x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = GELU()

    def forward(self, x):
        if self.dropout is None:
            return self.w_2(self.activation(self.w_1(x)))
        else:
            return self.w_2(self.dropout(self.activation(self.w_1(x))))


class BTransformer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, clip_grad=''):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super().__init__()
        None
        self.clip_grad = clip_grad
        if self.clip_grad == 'agc':
            self.attn = Residual(MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout))
            self.ff = Residual(PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout))
        else:
            self.attn = Residual(PreNorm(hidden, MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)))
            self.ff = Residual(PreNorm(hidden, PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(self, x, mask):
        x = self.attn(x, mask=mask)
        x = self.ff(x)
        if self.dropout is not None:
            return self.dropout(x)
        else:
            return x


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, clip_grad=''):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.isV0 = False
        for _ in range(depth):
            if self.isV0:
                self.layers.append(nn.ModuleList([Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))), Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))
            else:
                self.layers.append(BTransformer(dim, heads, dim * 4, dropout, clip_grad=clip_grad))

    def forward(self, x, mask=None):
        if self.isV0:
            for attn, ff in self.layers:
                x = attn(x, mask=mask)
                x = ff(x)
        else:
            for BTrans in self.layers:
                x = BTrans(x, mask)
        return x


MIN_NUM_PATCHES = 16


class OpticalFormer(nn.Module):

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, ff_hidden, pool='cls', channels=3, dim_head=64, dropout=0.0, emb_dropout=0.0, clip_grad=''):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.patch_size = patch_size
        self.clip_grad = clip_grad
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, ff_hidden, dropout, clip_grad=self.clip_grad)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.Identity() if self.clip_grad == 'agc' else nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def name_(self):
        return 'ViT_'

    def forward(self, img, mask=None):
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.transformer(x, mask)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

    def predict(self, output):
        if self.config.support == 'binary':
            nGate = output.shape[1] // 2
            pred = 0
            for i in range(nGate):
                no = 2 * (nGate - 1 - i)
                val_2 = torch.stack([output[:, no], output[:, no + 1]], 1)
                pred_i = val_2.max(1, keepdim=True)[1]
                pred = pred * 2 + pred_i
        elif self.config.support == 'logit':
            nGate = output.shape[1]
            pred = 0
            for i in range(nGate):
                no = nGate - 1 - i
                val_2 = F.sigmoid(output[:, no])
                pred_i = (val_2 + 0.5).long()
                pred = pred * 2 + pred_i
        else:
            pred = output.max(1, keepdim=True)[1]
        return pred


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class AttentionQKV(nn.Module):

    def __init__(self, hidden, attn_heads, dropout):
        super(AttentionQKV, self).__init__()
        self.attn = Residual(PreNorm(hidden, MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)))

    def forward(self, x, mask=None):
        shape = list(x.shape)
        if len(shape) == 2:
            x = x.unsqueeze(1)
        x = self.attn(x, mask=mask)
        if len(shape) == 2:
            x = x.squeeze(1)
        return x


class BinaryChunk(torch.nn.Module):

    def __init__(self, nCls, isLogit=False, pooling='max', chunk_dim=-1):
        super(BinaryChunk, self).__init__()
        self.nClass = nCls
        self.nChunk = int(math.ceil(math.log2(self.nClass)))
        self.pooling = pooling
        self.isLogit = isLogit

    def __repr__(self):
        main_str = super(BinaryChunk, self).__repr__()
        if self.isLogit:
            main_str += '_logit'
        main_str += f'_nChunk{self.nChunk}_cls[{self.nClass}]_pool[{self.pooling}]'
        return main_str

    def chunk_poll(self, ck, nSamp):
        x2 = ck.contiguous().view(nSamp, -1)
        if self.pooling == 'max':
            x3 = torch.max(x2, 1)
            return x3.values
        else:
            x3 = torch.mean(x2, 1)
            return x3

    def forward(self, x):
        nSamp = x.shape[0]
        x_max = []
        for ck in x.chunk(self.nChunk, -1):
            if self.isLogit:
                x_max.append(self.chunk_poll(ck, nSamp))
            else:
                for xx in ck.chunk(2, -2):
                    x2 = xx.contiguous().view(nSamp, -1)
                    if self.pooling == 'max':
                        x3 = torch.max(x2, 1)
                        x_max.append(x3.values)
                    else:
                        x3 = torch.mean(x2, 1)
                        x_max.append(x3)
        x = torch.stack(x_max, 1)
        return x


class D_input(nn.Module):

    def __init__(self, config, DNet):
        super(D_input, self).__init__()
        self.config = config
        self.DNet = DNet
        self.inplanes = 64
        self.nLayD = DNet.nDifrac
        self.c_input = nn.Conv2d(3 + self.nLayD, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        nChan = x.shape[1]
        assert nChan == 3 or nChan == 1
        if nChan == 3:
            gray = x[:, 0:1] * 0.3 + 0.59 * x[:, 1:2] + 0.11 * x[:, 2:3]
        else:
            gray = x
        return self.DNet.forward(gray)
        listT = []
        for i in range(nChan):
            listT.append(x[:, i:i + 1])
        if self.nLayD >= 1:
            self.DNet.forward(gray)
            assert len(self.DNet.feat_extractor) == self.nLayD
            for opti, w in self.DNet.feat_extractor:
                listT.append(opti)
        elif self.nLayD == 0:
            pass
        else:
            listT.append(gray)
        x = torch.stack(listT, dim=1).squeeze()
        if hasattr(self, 'visual'):
            self.visual.onX(x, f'D_input')
        x = self.c_input(x)
        return x

    def forward_000(self, x):
        if False:
            gray = x[:, 0:1]
            self.DNet.forward(gray)
            for opti, w in self.DNet.feat_extractor:
                opti = torch.stack([opti, opti, opti], 1).squeeze()
                out_opti = self.resNet.forward(opti)
                out_sum = out_sum + out_opti * w
        pass


def seed_everything(seed=0):
    None
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RGBO_CNN(torch.nn.Module):
    """
        resnet  https://missinglink.ai/guides/pytorch/pytorch-resnet-building-training-scaling-residual-networks-pytorch/
    """

    def pick_models(self):
        if False:
            model_names = sorted(name for name in cnn_models.__dict__ if name.islower() and not name.startswith('__') and callable(models.__dict__[name]))
            None
            model_names = ['alexnet', 'bninception', 'cafferesnet101', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'dpn107', 'dpn131', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'fbresnet152', 'inceptionresnetv2', 'inceptionv3', 'inceptionv4', 'nasnetalarge', 'nasnetamobile', 'pnasnet5large', 'polynet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x4d', 'resnext101_64x4d', 'se_resnet101', 'se_resnet152', 'se_resnet50', 'se_resnext101_32x4d', 'se_resnext50_32x4d', 'senet154', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'xception']
        self.back_bone = 'resnet18_x'
        cnn_model = ResNet34()
        return cnn_model

    def __init__(self, config, DNet):
        super(RGBO_CNN, self).__init__()
        seed_everything(42)
        self.config = config
        backbone = self.pick_models()
        if self.config.dnet_type == 'stack_feature':
            self.DInput = D_input(config, DNet)
        elif self.config.dnet_type == 'stack_input':
            self.CNet = nn.Sequential(*list(backbone.children())[1:])
        else:
            self.CNet = nn.Sequential(*list(backbone.children()))
        if False:
            if config.gpu_device is not None:
                self
                None
                self.thickness_criterion = self.thickness_criterion
                self.metal_criterion = self.metal_criterion
            elif config.distributed:
                self
                self = torch.nn.parallel.DistributedDataParallel(self)
            else:
                self = torch.nn.DataParallel(self)

    def save_acti(self, x, name):
        acti = x.cpu().data.numpy()
        self.activations.append({'name': name, 'shape': acti.shape, 'activation': acti})

    def forward_0(self, x):
        if hasattr(self, 'DInput'):
            x = self.DInput(x)
        for no, lay in enumerate(self.CNet):
            if isinstance(lay, nn.Linear):
                x = F.avg_pool2d(x, 4)
                x = x.reshape(x.size(0), -1)
            x = lay(x)
            if isinstance(lay, nn.AdaptiveAvgPool2d):
                x = x.reshape(x.size(0), -1)
        out_sum = x
        return out_sum

    def forward(self, x):
        out_sum = 0
        if self.config.dnet_type == 'stack_feature':
            out_sum = self.DInput(x)
        for no, lay in enumerate(self.CNet):
            if isinstance(lay, nn.Linear):
                x = F.avg_pool2d(x, 4)
                x = x.reshape(x.size(0), -1)
            x = lay(x)
            if isinstance(lay, nn.AdaptiveAvgPool2d):
                x = x.reshape(x.size(0), -1)
        out_sum += x
        return out_sum


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BinaryChunk,
     lambda: ([], {'nCls': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChunkPool,
     lambda: ([], {'nCls': 4, 'config': _mock_config(output_chunk=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeedForward,
     lambda: ([], {'dim': 4, 'hidden_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PreNorm,
     lambda: ([], {'dim': 4, 'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Residual,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (View,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_closest_git_ONNet(_paritybench_base):
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

