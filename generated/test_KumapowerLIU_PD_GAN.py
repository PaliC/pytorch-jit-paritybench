import sys
_module = sys.modules[__name__]
del sys
config = _module
base_config = _module
test_config = _module
train_config = _module
data = _module
dataprocess = _module
PConv = _module
PDGAN = _module
models = _module
base_model = _module
SPDNorm = _module
blocks = _module
loss = _module
pconvblocks = _module
Discriminator = _module
network = _module
networks = _module
pconv = _module
pdgan = _module
test = _module
train = _module
util = _module
util = _module

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


from torchvision import transforms


import torchvision.transforms.functional as transFunc


import random


import numpy as np


import torch.utils.data as data


from collections import OrderedDict


from torchvision import utils


from functools import reduce


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.utils.spectral_norm as spectral_norm


import torchvision


import torchvision.models as models


from torch.nn import functional as F


from torch.nn import init


import functools


from torch.optim import lr_scheduler


import time


from torch.utils import data


import inspect


import re


import collections


import math


from torch.autograd import Variable


class MaskGet(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

    def forward(self, input):
        with torch.no_grad():
            output_mask = self.mask_conv(input)
        no_update_holes = output_mask == 0
        new_mask = torch.ones_like(output_mask)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        return new_mask


def PositionalNorm2d(x, epsilon=1e-05):
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output


class SPDNorm(nn.Module):

    def __init__(self, norm_channel, norm_type='batch'):
        super().__init__()
        label_nc = 3
        param_free_norm_type = norm_type
        ks = 3
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_channel, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_channel, affine=False)
        elif param_free_norm_type == 'position':
            self.param_free_norm = PositionalNorm2d
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE' % param_free_norm_type)
        pw = ks // 2
        nhidden = 128
        self.mlp_activate = nn.Sequential(nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_channel, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_channel, kernel_size=ks, padding=pw)

    def forward(self, x, prior_f, weight):
        normalized = self.param_free_norm(x)
        actv = self.mlp_activate(prior_f)
        gamma = self.mlp_gamma(actv) * weight
        beta = self.mlp_beta(actv) * weight
        out = normalized * (1 + gamma) + beta
        return out


class SPDNormResnetBlock(nn.Module):

    def __init__(self, fin, fout, mask_number, mask_ks, cfg):
        super().__init__()
        nhidden = 128
        fmiddle = min(fin, fout)
        lable_nc = 3
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        self.learned_shortcut = True
        if 'spectral' in cfg.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        self.norm_0 = SPDNorm(fin, norm_type='position')
        self.norm_1 = SPDNorm(fmiddle, norm_type='position')
        self.norm_s = SPDNorm(fin, norm_type='position')
        self.v = nn.Parameter(torch.zeros(1))
        self.activeweight = nn.Sigmoid()
        self.mask_number = mask_number
        self.mask_ks = mask_ks
        pw_mask = int(np.ceil((self.mask_ks - 1.0) / 2))
        self.mask_conv = MaskGet(1, 1, kernel_size=self.mask_ks, padding=pw_mask)
        self.conv_to_f = nn.Sequential(nn.Conv2d(lable_nc, nhidden, kernel_size=3, padding=1), nn.InstanceNorm2d(nhidden), nn.ReLU(), nn.Conv2d(nhidden, fin, kernel_size=3, padding=1))
        self.attention = nn.Sequential(nn.Conv2d(fin * 2, fin, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, x, prior_image, mask):
        """

        Args:
            x: input feature
            prior_image: the output of PCConv
            mask: mask


        """
        b, c, h, w = x.size()
        prior_image_resize = F.interpolate(prior_image, size=x.size()[2:], mode='nearest')
        mask_resize = F.interpolate(mask, size=x.size()[2:], mode='nearest')
        prior_feature = self.conv_to_f(prior_image_resize)
        soft_map = self.attention(torch.cat([prior_feature, x], 1))
        soft_map = (1 - mask_resize) * soft_map + mask_resize
        mask_pre = mask_resize
        hard_map = 0.0
        for i in range(self.mask_number):
            mask_out = self.mask_conv(mask_pre)
            mask_generate = (mask_out - mask_pre) * (1 / torch.exp(torch.tensor(i + 1)))
            mask_pre = mask_out
            hard_map = hard_map + mask_generate
        hard_map_inner = (1 - mask_out) * (1 / torch.exp(torch.tensor(i + 1)))
        hard_map = hard_map + mask_resize + hard_map_inner
        soft_out = self.conv_s(self.norm_s(x, prior_image_resize, soft_map))
        hard_out = self.conv_0(self.actvn(self.norm_0(x, prior_image_resize, hard_map)))
        hard_out = self.conv_1(self.actvn(self.norm_1(hard_out, prior_image_resize, hard_map)))
        out = soft_out + hard_out
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 0.2)


class VGG16(torch.nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()
        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()
        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.max3 = torch.nn.Sequential()
        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
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
        for x in range(16, 17):
            self.max3.add_module(str(x), features[x])
        for x in range(17, 19):
            self.relu4_1.add_module(str(x), features[x])
        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])
        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])
        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])
        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])
        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])
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
        max_3 = self.max3(relu3_3)
        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {'relu1_1': relu1_1, 'relu1_2': relu1_2, 'relu2_1': relu2_1, 'relu2_2': relu2_2, 'relu3_1': relu3_1, 'relu3_2': relu3_2, 'relu3_3': relu3_3, 'max_3': max_3, 'relu4_1': relu4_1, 'relu4_2': relu4_2, 'relu4_3': relu4_3, 'relu5_1': relu5_1, 'relu5_2': relu5_2, 'relu5_3': relu5_3}
        return out


class StyleLoss(nn.Module):
    """
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG16())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_3']), self.compute_gram(y_vgg['relu3_3']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_3']), self.compute_gram(y_vgg['relu4_3']))
        return style_loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG16())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])
        return content_loss


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor, cfg=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.cfg = cfg
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
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

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        elif target_is_real:
            return -input.mean()
        else:
            return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class Diversityloss(nn.Module):

    def __init__(self):
        super(Diversityloss, self).__init__()
        self.vgg = VGG16()
        self.criterion = nn.L1Loss()
        self.weights = [1.0, 1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        diversity_loss = 0.0
        diversity_loss += self.weights[4] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        return diversity_loss


class PartialConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        input = x[0]
        mask = x[1].float()
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)
        with torch.no_grad():
            output_mask = self.mask_conv(mask)
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.bool(), 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes.bool(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        out = [output, new_mask]
        return out


class PCBActiv(nn.Module):

    def __init__(self, in_ch, out_ch, norm_layer='instance', sample='down-4', activ='leaky', conv_bias=False, inner=False, outer=False):
        super().__init__()
        if sample == 'same-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == 'same-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == 'down-4':
            self.conv = PartialConv(in_ch, out_ch, 4, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)
        if norm_layer == 'instance':
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm_layer == 'batch':
            self.norm = nn.BatchNorm2d(out_ch, affine=True)
        else:
            pass
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            pass
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.activation(out[0])
            out = self.conv(out)
        elif self.outer:
            out = self.conv(out)
        else:
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.norm(out[0])
        return out


class ResnetBlock(nn.Module):

    def __init__(self, dim, norm='instance'):
        super(ResnetBlock, self).__init__()
        self.conv_1 = PartialConv(dim, dim, 3, 1, 1, 1)
        if norm == 'instance':
            self.norm_1 = nn.InstanceNorm2d(dim, track_running_stats=False)
            self.norm_2 = nn.InstanceNorm2d(dim, track_running_stats=False)
        elif norm == 'batch':
            self.norm_1 = nn.BatchNorm2d(dim, track_running_stats=False)
            self.norm_2 = nn.BatchNorm2d(dim, track_running_stats=False)
        self.active = nn.ReLU(True)
        self.conv_2 = PartialConv(dim, dim, 3, 1, 1, 1)

    def forward(self, x):
        out = self.conv_1(x)
        out[0] = self.norm_1(out[0])
        out[0] = self.active(out[0])
        out = self.conv_2(out)
        out[0] = self.norm_2(out[0])
        out[0] = x[0] + out[0]
        return out


class UnetSkipConnectionDBlock(nn.Module):

    def __init__(self, inner_nc, outer_nc, outermost=False, innermost=False, norm_layer='instance'):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU()
        upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
        if norm_layer == 'instance':
            upnorm = nn.InstanceNorm2d(outer_nc, affine=True)
        elif norm_layer == 'batch':
            upnorm = nn.BatchNorm2d(outer_nc, affine=True)
        else:
            pass
        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def get_nonspade_norm_layer(cfg, norm_type='instance'):

    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)
        return nn.Sequential(layer, norm_layer)
    return add_norm_layer


class NLayerDiscriminator(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = cfg.ndf
        input_nc = cfg.input_nc_D
        norm_layer = get_nonspade_norm_layer(cfg, cfg.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]]
        for n in range(1, cfg.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == cfg.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)), nn.LeakyReLU(0.2, False)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
        get_intermediate_features = not self.cfg.no_ganFeat_loss
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]


class MultiscaleDiscriminator(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_D = cfg.num_D
        for i in range(cfg.num_D):
            subnetD = self.create_single_discriminator(cfg)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, cfg):
        netD = NLayerDiscriminator(cfg)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        get_intermediate_features = not self.cfg.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)
        return result


class Encoder(nn.Module):

    def __init__(self, input_nc, ngf=64, res_num=4, norm_layer='instance'):
        super(Encoder, self).__init__()
        Encoder_1 = PCBActiv(input_nc, ngf, norm_layer=None, activ=None, outer=True)
        Encoder_2 = PCBActiv(ngf, ngf * 2, norm_layer=norm_layer)
        Encoder_3 = PCBActiv(ngf * 2, ngf * 4, norm_layer=norm_layer)
        Encoder_4 = PCBActiv(ngf * 4, ngf * 8, norm_layer=norm_layer)
        Encoder_5 = PCBActiv(ngf * 8, ngf * 8, norm_layer=norm_layer)
        Encoder_6 = PCBActiv(ngf * 8, ngf * 8, norm_layer=None, inner=True)
        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf * 8)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)
        self.Encoder_1 = Encoder_1
        self.Encoder_2 = Encoder_2
        self.Encoder_3 = Encoder_3
        self.Encoder_4 = Encoder_4
        self.Encoder_5 = Encoder_5
        self.Encoder_6 = Encoder_6

    def forward(self, x):
        out_1 = self.Encoder_1(x)
        out_2 = self.Encoder_2(out_1)
        out_3 = self.Encoder_3(out_2)
        out_4 = self.Encoder_4(out_3)
        out_5 = self.Encoder_5(out_4)
        out_6 = self.Encoder_6(out_5)
        out_7 = self.middle(out_6)
        return out_7, out_5, out_4, out_3, out_2, out_1


class Decoder(nn.Module):

    def __init__(self, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()
        Decoder_1 = UnetSkipConnectionDBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        Decoder_2 = UnetSkipConnectionDBlock(ngf * 16, ngf * 8, norm_layer=norm_layer)
        Decoder_3 = UnetSkipConnectionDBlock(ngf * 16, ngf * 4, norm_layer=norm_layer)
        Decoder_4 = UnetSkipConnectionDBlock(ngf * 8, ngf * 2, norm_layer=norm_layer)
        Decoder_5 = UnetSkipConnectionDBlock(ngf * 4, ngf, norm_layer=norm_layer)
        Decoder_6 = UnetSkipConnectionDBlock(ngf * 2, output_nc, norm_layer=norm_layer, outermost=True)
        self.Decoder_1 = Decoder_1
        self.Decoder_2 = Decoder_2
        self.Decoder_3 = Decoder_3
        self.Decoder_4 = Decoder_4
        self.Decoder_5 = Decoder_5
        self.Decoder_6 = Decoder_6

    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6):
        y_1 = self.Decoder_1(input_6[0])
        y_2 = self.Decoder_2(torch.cat([y_1, input_5[0]], 1))
        y_3 = self.Decoder_3(torch.cat([y_2, input_4[0]], 1))
        y_4 = self.Decoder_4(torch.cat([y_3, input_3[0]], 1))
        y_5 = self.Decoder_5(torch.cat([y_4, input_2[0]], 1))
        y_6 = self.Decoder_6(torch.cat([y_5, input_1[0]], 1))
        out = y_6
        return out


class SPDNormGeneratorUnit(nn.Module):

    def __init__(self, in_channels, out_channels, mask_number, mask_ks, cfg):
        super().__init__()
        self.block = SPDNormResnetBlock(in_channels, out_channels, mask_number, mask_ks, cfg)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, prior_f, mask):
        out = self.block(x, prior_f, mask)
        out = self.up(out)
        return out


class SPDNormGenerator(nn.Module):
    """
    First, transfer the random vector z with an fc layer.
    Then,
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        nf = self.cfg.ngf
        self.sw = self.cfg.latent_sw
        self.sh = self.cfg.latent_sh
        self.fc = nn.Linear(cfg.z_dim, 16 * nf * self.sw * self.sh)
        self.generated = nn.ModuleList([SPDNormGeneratorUnit(16 * nf, 16 * nf, 2, 3, cfg), SPDNormGeneratorUnit(16 * nf, 16 * nf, 3, 3, cfg), SPDNormGeneratorUnit(16 * nf, 8 * nf, 4, 3, cfg), SPDNormGeneratorUnit(8 * nf, 4 * nf, 5, 3, cfg), SPDNormGeneratorUnit(4 * nf, 2 * nf, 6, 5, cfg), SPDNormGeneratorUnit(2 * nf, 1 * nf, 7, 5, cfg)])
        self.conv_img = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, z, pre_image, mask):
        latent_v = self.fc(z)
        latent_v = latent_v.view(-1, 16 * self.cfg.ngf, self.sh, self.sw)
        input_mask = mask[:, 0, :, :].unsqueeze(1)
        out = latent_v
        for i, conv in enumerate(self.generated):
            out = conv(out, pre_image, input_mask)
        out = self.conv_img(F.leaky_relu(out, 0.2))
        out = F.tanh(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Diversityloss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     True),
    (MaskGet,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiscaleDiscriminator,
     lambda: ([], {'cfg': _mock_config(num_D=4, ndf=4, input_nc_D=4, norm_D=4, n_layers_D=1, no_ganFeat_loss=MSELoss())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NLayerDiscriminator,
     lambda: ([], {'cfg': _mock_config(ndf=4, input_nc_D=4, norm_D=4, n_layers_D=1, no_ganFeat_loss=MSELoss())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PCBActiv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PartialConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResnetBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SPDNorm,
     lambda: ([], {'norm_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 3, 4, 4]), torch.rand([4, 1, 4, 4])], {}),
     True),
    (UnetSkipConnectionDBlock,
     lambda: ([], {'inner_nc': 4, 'outer_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGG16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_KumapowerLIU_PD_GAN(_paritybench_base):
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

