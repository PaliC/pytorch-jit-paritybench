import sys
_module = sys.modules[__name__]
del sys
config = _module
data_loader = _module
losses = _module
main = _module
CalcPSNR = _module
CalcSSIM = _module
CalcNIMA = _module
mobile_net_v2 = _module
cli = _module
common = _module
app = _module
inference_model = _module
utils = _module
mobile_net_v2 = _module
model = _module
clean_dataset = _module
datasets = _module
emd_loss = _module
main = _module
test = _module
models = _module
tester = _module
trainer = _module
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


from itertools import chain


import torch


from torch.utils import data


from torchvision import transforms


import torch.nn.functional as F


from math import exp


import torch.nn as nn


import torchvision.models as models


from math import pi


import torchvision.models


import torchvision.transforms as transforms


import numpy as np


import math


from torchvision.datasets.folder import default_loader


import pandas as pd


from torch.utils.data import Dataset


from torch.autograd import Variable


from torch.utils.data import DataLoader


import functools


import time


from torchvision.utils import save_image


import numbers


import random


import tensorflow as tf


from torch.utils.tensorboard import SummaryWriter


import torchvision


import scipy.misc


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


class VGG19_relu(torch.nn.Module):

    def __init__(self):
        super(VGG19_relu, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = models.vgg19(pretrained=True)
        cnn = cnn
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()
        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()
        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()
        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()
        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()
        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])
        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])
        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])
        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])
        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])
        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])
        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])
        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])
        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])
        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])
        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])
        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])
        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])
        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])
        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)
        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)
        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)
        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)
        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)
        out = {'relu1_1': relu1_1, 'relu1_2': relu1_2, 'relu2_1': relu2_1, 'relu2_2': relu2_2, 'relu3_1': relu3_1, 'relu3_2': relu3_2, 'relu3_3': relu3_3, 'relu3_4': relu3_4, 'relu4_1': relu4_1, 'relu4_2': relu4_2, 'relu4_3': relu4_3, 'relu4_4': relu4_4, 'relu5_1': relu5_1, 'relu5_2': relu5_2, 'relu5_3': relu5_3, 'relu5_4': relu5_4}
        return out


class PerceptualLoss(nn.Module):

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19_relu())
        self.criterion = torch.nn.MSELoss()
        self.weights = [1.0 / 64, 1.0 / 64, 1.0 / 32, 1.0 / 32, 1.0 / 1]
        self.IN = nn.InstanceNorm2d(512, affine=False, track_running_stats=False)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

    def __call__(self, x, y):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = self.weights[0] * self.criterion(self.IN(x_vgg['relu1_1']), self.IN(y_vgg['relu1_1']))
        loss += self.weights[1] * self.criterion(self.IN(x_vgg['relu2_1']), self.IN(y_vgg['relu2_1']))
        loss += self.weights[2] * self.criterion(self.IN(x_vgg['relu3_1']), self.IN(y_vgg['relu3_1']))
        loss += self.weights[3] * self.criterion(self.IN(x_vgg['relu4_1']), self.IN(y_vgg['relu4_1']))
        loss += self.weights[4] * self.criterion(self.IN(x_vgg['relu5_1']), self.IN(y_vgg['relu5_1']))
        return loss


class TVLoss(nn.Module):

    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class AngularLoss(torch.nn.Module):

    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, feature1, feature2):
        cos_criterion = torch.nn.CosineSimilarity(dim=1)
        cos = cos_criterion(feature1, feature2)
        clip_bound = 0.999999
        cos = torch.clamp(cos, -clip_bound, clip_bound)
        if False:
            return 1 - torch.mean(cos)
        else:
            return torch.mean(torch.acos(cos)) * 180 / pi


class MultiscaleRecLoss(nn.Module):

    def __init__(self, scale=3, rec_loss_type='l1', multiscale=True):
        super(MultiscaleRecLoss, self).__init__()
        self.multiscale = multiscale
        if rec_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif rec_loss_type == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        elif rec_loss_type == 'l2':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError('Loss [{}] is not implemented'.format(rec_loss_type))
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        if self.multiscale:
            self.weights = [1.0, 1.0 / 2, 1.0 / 4]
            self.weights = self.weights[:scale]

    def forward(self, input, target):
        loss = 0
        pred = input.clone()
        gt = target.clone()
        if self.multiscale:
            for i in range(len(self.weights)):
                loss += self.weights[i] * self.criterion(pred, gt)
                if i != len(self.weights) - 1:
                    pred = self.downsample(pred)
                    gt = self.downsample(gt)
        else:
            loss = self.criterion(pred, gt)
        return loss


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        elif gan_mode == 'rahinge':
            pass
        elif gan_mode == 'rals':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        if self.gan_mode == 'original':
            if for_real:
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(real_preds, target_tensor)
                return loss
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                loss = F.binary_cross_entropy_with_logits(fake_preds, target_tensor)
                return loss
            else:
                raise NotImplementedError('nither for real_preds nor for fake_preds')
        elif self.gan_mode == 'ls':
            if for_real:
                target_tensor = self.get_target_tensor(real_preds, target_is_real)
                return F.mse_loss(real_preds, target_tensor)
            elif for_fake:
                target_tensor = self.get_target_tensor(fake_preds, target_is_real)
                return F.mse_loss(fake_preds, target_tensor)
            else:
                raise NotImplementedError('nither for real_preds nor for fake_preds')
        elif self.gan_mode == 'hinge':
            if for_real:
                if for_discriminator:
                    if target_is_real:
                        minval = torch.min(real_preds - 1, self.get_zero_tensor(real_preds))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-real_preds - 1, self.get_zero_tensor(real_preds))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(real_preds)
                return loss
            elif for_fake:
                if for_discriminator:
                    if target_is_real:
                        minval = torch.min(fake_preds - 1, self.get_zero_tensor(fake_preds))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-fake_preds - 1, self.get_zero_tensor(fake_preds))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real, "The generator's hinge loss must be aiming for real"
                    loss = -torch.mean(fake_preds)
                return loss
            else:
                raise NotImplementedError('nither for real_preds nor for fake_preds')
        elif self.gan_mode == 'rahinge':
            if for_discriminator:
                r_f_diff = real_preds - torch.mean(fake_preds)
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean(torch.nn.ReLU()(1 - r_f_diff)) + torch.mean(torch.nn.ReLU()(1 + f_r_diff))
                return loss / 2
            else:
                r_f_diff = real_preds - torch.mean(fake_preds)
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean(torch.nn.ReLU()(1 + r_f_diff)) + torch.mean(torch.nn.ReLU()(1 - f_r_diff))
                return loss / 2
        elif self.gan_mode == 'rals':
            if for_discriminator:
                r_f_diff = real_preds - torch.mean(fake_preds)
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean((r_f_diff - 1) ** 2) + torch.mean((f_r_diff + 1) ** 2)
                return loss / 2
            else:
                r_f_diff = real_preds - torch.mean(fake_preds)
                f_r_diff = fake_preds - torch.mean(real_preds)
                loss = torch.mean((r_f_diff + 1) ** 2) + torch.mean((f_r_diff - 1) ** 2)
                return loss / 2
        elif for_real:
            if target_is_real:
                return -real_preds.mean()
            else:
                return real_preds.mean()
        elif for_fake:
            if target_is_real:
                return -fake_preds.mean()
            else:
                return fake_preds.mean()
        else:
            raise NotImplementedError('nither for real_preds nor for fake_preds')

    def __call__(self, real_preds, fake_preds, target_is_real, for_real=None, for_fake=None, for_discriminator=True):
        if isinstance(real_preds, list):
            loss = 0
            for pred_real_i, pred_fake_i in zip(real_preds, fake_preds):
                if isinstance(pred_real_i, list):
                    pred_real_i = pred_real_i[-1]
                if isinstance(pred_fake_i, list):
                    pred_fake_i = pred_fake_i[-1]
                loss_tensor = self.loss(pred_real_i, pred_fake_i, target_is_real, for_real, for_fake, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss
        else:
            return self.loss(real_preds, target_is_real, for_discriminator)


MOBILE_NET_V2_UTR = 'https://s3-us-west-1.amazonaws.com/models-nima/mobilenetv2.pth.tar'


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(inplace=True), nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU6(inplace=True), nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


def conv_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        self.interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size // 32))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(self.last_channel, n_class))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename


def mobile_net_v2(pretrained=True):
    model = MobileNetV2()
    if pretrained:
        path_to_model = '/tmp/mobilenetv2.pth.tar'
        if not os.path.exists(path_to_model):
            path_to_model = download_file(MOBILE_NET_V2_UTR, path_to_model)
        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
    return model


class NIMA(nn.Module):

    def __init__(self, pretrained_base_model=False):
        super(NIMA, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.base_model = base_model
        self.head = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(p=0.75), nn.Linear(1280, 10), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class EDMLoss(nn.Module):

    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target: 'Variable', p_estimate: 'Variable'):
        assert p_target.shape == p_estimate.shape
        cdf_target = torch.cumsum(p_target, dim=1)
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()


def SpectralNorm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def get_act_fun(act_fun_type='LeakyReLU'):
    if isinstance(act_fun_type, str):
        if act_fun_type == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun_type == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_fun_type == 'Swish':
            return Swish()
        elif act_fun_type == 'SELU':
            return nn.SELU(inplace=True)
        elif act_fun_type == 'none':
            return nn.Sequential()
        else:
            raise NotImplementedError('activation function [%s] is not found' % act_fun_type)
    else:
        return act_fun_type()


class Identity(nn.Module):

    def forward(self, x):
        return x


def get_norm_fun(norm_fun_type='none'):
    if norm_fun_type == 'BatchNorm':
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'InstanceNorm':
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'none':
        norm_fun = lambda x: Identity()
    else:
        raise NotImplementedError('normalization function [%s] is not found' % norm_fun_type)
    return norm_fun


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun, use_sn):
        super(ConvBlock, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        main = []
        main.append(nn.ReflectionPad2d(self.padding))
        main.append(SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn))
        norm_fun = get_norm_fun(norm_fun)
        main.append(norm_fun(out_channels))
        main.append(get_act_fun(act_fun))
        self.main = nn.Sequential(*main)

    def forward(self, x):
        return self.main(x)


def calc_mean_std(feat, eps=1e-05):
    size = feat.data.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class GAM(nn.Module):
    """Global attention module"""

    def __init__(self, in_nc, out_nc, reduction=8, bias=False, use_sn=False, norm=False):
        super(GAM, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_nc * 2, out_channels=in_nc // reduction, kernel_size=1, stride=1, bias=bias, padding=0, dilation=1), nn.ReLU(inplace=True), nn.Conv2d(in_channels=in_nc // reduction, out_channels=out_nc, kernel_size=1, stride=1, bias=bias, padding=0, dilation=1))
        self.fuse = nn.Sequential(SpectralNorm(nn.Conv2d(in_channels=in_nc * 2, out_channels=out_nc, kernel_size=1, stride=1, bias=True, padding=0, dilation=1), use_sn))
        self.in_norm = nn.InstanceNorm2d(out_nc)
        self.norm = norm

    def forward(self, x):
        x_mean, x_std = calc_mean_std(x)
        out = self.conv(torch.cat([x_mean, x_std], dim=1))
        out = self.fuse(torch.cat([x, out.expand_as(x)], dim=1))
        if self.norm:
            out = self.in_norm(out)
        return out


class Interpolate(nn.Module):

    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return out


class SNConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, use_sn):
        super(SNConv, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.main = nn.Sequential(nn.ReflectionPad2d(self.padding), SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn))

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    """Generator network"""

    def __init__(self, conv_dim, norm_fun, act_fun, use_sn):
        super(Generator, self).__init__()
        self.enc1 = ConvBlock(in_channels=3, out_channels=conv_dim * 1, kernel_size=7, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)
        self.enc2 = ConvBlock(in_channels=conv_dim * 1, out_channels=conv_dim * 2, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)
        self.enc3 = ConvBlock(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)
        self.enc4 = ConvBlock(in_channels=conv_dim * 4, out_channels=conv_dim * 8, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)
        self.enc5 = ConvBlock(in_channels=conv_dim * 8, out_channels=conv_dim * 16, kernel_size=3, stride=2, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)
        self.upsample1 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim * 16, conv_dim * 8, 1, 1, 0, 1, True, use_sn))
        self.upsample2 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim * 8, conv_dim * 4, 1, 1, 0, 1, True, use_sn))
        self.upsample3 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim * 4, conv_dim * 2, 1, 1, 0, 1, True, use_sn))
        self.upsample4 = nn.Sequential(Interpolate(2, 'bilinear', True), SNConv(conv_dim * 2, conv_dim * 1, 1, 1, 0, 1, True, use_sn))
        self.dec1 = ConvBlock(in_channels=conv_dim * 16, out_channels=conv_dim * 8, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)
        self.dec2 = ConvBlock(in_channels=conv_dim * 8, out_channels=conv_dim * 4, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)
        self.dec3 = ConvBlock(in_channels=conv_dim * 4, out_channels=conv_dim * 2, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)
        self.dec4 = ConvBlock(in_channels=conv_dim * 2, out_channels=conv_dim * 1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)
        self.dec5 = nn.Sequential(SNConv(in_channels=conv_dim * 1, out_channels=conv_dim * 1, kernel_size=3, stride=1, padding=0, dilation=1, use_bias=True, use_sn=False), SNConv(in_channels=conv_dim * 1, out_channels=3, kernel_size=7, stride=1, padding=0, dilation=1, use_bias=True, use_sn=False), nn.Tanh())
        self.ga5 = GAM(conv_dim * 16, conv_dim * 16, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga4 = GAM(conv_dim * 8, conv_dim * 8, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga3 = GAM(conv_dim * 4, conv_dim * 4, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga2 = GAM(conv_dim * 2, conv_dim * 2, reduction=8, bias=False, use_sn=use_sn, norm=True)
        self.ga1 = GAM(conv_dim * 1, conv_dim * 1, reduction=8, bias=False, use_sn=use_sn, norm=True)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x5 = self.ga5(x5)
        y1 = self.upsample1(x5)
        y1 = torch.cat([y1, self.ga4(x4)], dim=1)
        y1 = self.dec1(y1)
        y2 = self.upsample2(y1)
        y2 = torch.cat([y2, self.ga3(x3)], dim=1)
        y2 = self.dec2(y2)
        y3 = self.upsample3(y2)
        y3 = torch.cat([y3, self.ga2(x2)], dim=1)
        y3 = self.dec3(y3)
        y4 = self.upsample4(y3)
        y4 = torch.cat([y4, self.ga1(x1)], dim=1)
        y4 = self.dec4(y4)
        res = self.dec5(y4.mul(x1))
        out = torch.clamp(res + x, min=-1.0, max=1.0)
        return out


def dis_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun, use_sn):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []
    main.append(nn.ReflectionPad2d(padding))
    main.append(SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias), use_sn))
    norm_fun = get_norm_fun(norm_fun)
    main.append(norm_fun(out_channels))
    main.append(get_act_fun(act_fun))
    main = nn.Sequential(*main)
    return main


def dis_pred_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, type):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []
    main.append(nn.ReflectionPad2d(padding))
    main.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=use_bias))
    if type in ['ls', 'rals']:
        main.append(nn.Sigmoid())
    elif type in ['hinge', 'rahinge']:
        main.append(nn.Tanh())
    else:
        raise NotImplementedError('Adversarial loss [{}] is not found'.format(type))
    main = nn.Sequential(*main)
    return main


class Discriminator(nn.Module):

    def __init__(self, conv_dim, norm_fun, act_fun, use_sn, adv_loss_type):
        super(Discriminator, self).__init__()
        d_1 = [dis_conv_block(in_channels=3, out_channels=conv_dim, kernel_size=7, stride=2, padding=3, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_1_pred = [dis_pred_conv_block(in_channels=conv_dim, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]
        d_2 = [dis_conv_block(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=7, stride=2, padding=3, dilation=1, norm_fun=norm_fun, use_bias=True, act_fun=act_fun, use_sn=use_sn)]
        d_2_pred = [dis_pred_conv_block(in_channels=conv_dim * 2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]
        d_3 = [dis_conv_block(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=7, stride=2, padding=3, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_3_pred = [dis_pred_conv_block(in_channels=conv_dim * 4, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, use_bias=False, type=adv_loss_type)]
        d_4 = [dis_conv_block(in_channels=conv_dim * 4, out_channels=conv_dim * 8, kernel_size=5, stride=2, padding=2, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_4_pred = [dis_pred_conv_block(in_channels=conv_dim * 8, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, use_bias=False, type=adv_loss_type)]
        d_5 = [dis_conv_block(in_channels=conv_dim * 8, out_channels=conv_dim * 16, kernel_size=5, stride=2, padding=2, dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_5_pred = [dis_pred_conv_block(in_channels=conv_dim * 16, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1, use_bias=False, type=adv_loss_type)]
        self.d1 = nn.Sequential(*d_1)
        self.d1_pred = nn.Sequential(*d_1_pred)
        self.d2 = nn.Sequential(*d_2)
        self.d2_pred = nn.Sequential(*d_2_pred)
        self.d3 = nn.Sequential(*d_3)
        self.d3_pred = nn.Sequential(*d_3_pred)
        self.d4 = nn.Sequential(*d_4)
        self.d4_pred = nn.Sequential(*d_4_pred)
        self.d5 = nn.Sequential(*d_5)
        self.d5_pred = nn.Sequential(*d_5_pred)

    def forward(self, x):
        ds1 = self.d1(x)
        ds1_pred = self.d1_pred(ds1)
        ds2 = self.d2(ds1)
        ds2_pred = self.d2_pred(ds2)
        ds3 = self.d3(ds2)
        ds3_pred = self.d3_pred(ds3)
        ds4 = self.d4(ds3)
        ds4_pred = self.d4_pred(ds4)
        ds5 = self.d5(ds4)
        ds5_pred = self.d5_pred(ds5)
        return [ds1_pred, ds2_pred, ds3_pred, ds4_pred, ds5_pred]


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data. Default value is 2 (spatial).
    """

    def __init__(self, channels=3, kernel_size=21, sigma=3, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *([1] * (kernel.dim() - 1)))
        self.register_buffer('weight', kernel)
        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def forward(self, input):
        input = self.padding(input)
        out = self.conv(input, weight=self.weight, groups=self.groups)
        return out


class GaussianNoise(nn.Module):
    """A gaussian noise module.

    Args:
        stddev (float): The standard deviation of the normal distribution.
                        Default: 0.1.

    Shape:
        - Input: (batch, *)
        - Output: (batch, *) (same shape as input)
    """

    def __init__(self, mean=0.0, stddev=0.1):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, x):
        noise = torch.empty_like(x)
        noise.normal_(0, self.stddev)
        return x + noise


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AngularLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (EDMLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GaussianNoise,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GaussianSmoothing,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (MultiscaleRecLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NIMA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (SNConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1, 'use_bias': 4, 'use_sn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TVLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGG19_relu,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_eezkni_UEGAN(_paritybench_base):
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

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

