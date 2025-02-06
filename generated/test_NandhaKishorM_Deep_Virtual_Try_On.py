import sys
_module = sys.modules[__name__]
del sys
application = _module
application_holistic = _module
config = _module
data = _module
base_dataset = _module
demo_dataset = _module
regular_dataset = _module
demo = _module
detect_edges_image = _module
kaffe = _module
caffe = _module
caffepb = _module
resolver = _module
errors = _module
graph = _module
layers = _module
shapes = _module
tensorflow = _module
network = _module
transformer = _module
transformers = _module
geometric_matching_multi_gpu = _module
make_dataset = _module
models = _module
base_model = _module
generation_model = _module
models = _module
network_d = _module
network_g = _module
networks = _module
parsing = _module
test_new = _module
test_pgn = _module
train = _module
train_pgn = _module
utils = _module
datasample_pool = _module
image_reader = _module
image_reader_pgn = _module
loss = _module
model_pgn = _module
ops = _module
pose_transform = _module
pose_utils = _module
transforms = _module
warp_image = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.optim as optim


from torch.utils.data import DataLoader


from torchvision import transforms


from time import time


import torch.backends.cudnn as cudnn


from torchvision import utils


import torch.nn.functional as F


import time


import scipy.misc


import scipy.io as sio


import tensorflow as tf


import torch.utils.data


import random


import torch.utils.data as data


import torchvision.transforms as transforms


from abc import ABC


from abc import abstractmethod


from torch.utils.data.dataset import Dataset


from torch.nn import init


from torchvision import models


from torch.optim import lr_scheduler


import functools


from torch.nn import BCEWithLogitsLoss


from torch.nn import CrossEntropyLoss


from torch.autograd import Variable


import itertools


def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    None
    net.apply(init_func)


class FeatureExtraction(nn.Module):

    def __init__(self, input_nc, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FeatureExtraction, self).__init__()
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2 ** i * ngf if 2 ** i * ngf < 512 else 512
            out_ngf = 2 ** (i + 1) * ngf if 2 ** i * ngf < 512 else 512
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, nn.ReLU(True)]
            model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        model += [norm_layer(512)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        self.model = nn.Sequential(*model)
        init_weights(self.model, init_type='normal')

    def forward(self, x):
        return self.model(x)


class FeatureL2Norm(torch.nn.Module):

    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-06
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FeatureCorrelation(nn.Module):

    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class FeatureRegression(nn.Module):

    def __init__(self, input_nc=512, output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear = nn.Linear(64 * 4 * 3, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        return x


class AffineGridGen(nn.Module):

    def __init__(self, out_h=256, out_w=192, out_ch=3):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size)


class TpsGridGen(nn.Module):

    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X
            self.grid_Y = self.grid_Y
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))
            P_Y = np.reshape(P_Y, (-1, 1))
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X
                self.P_Y = self.P_Y
                self.P_X_base = self.P_X_base
                self.P_Y_base = self.P_Y_base

    def forward(self, theta):
        gpu_id = theta.get_device()
        self.grid_X = self.grid_X
        self.grid_Y = self.grid_Y
        self.P_X = self.P_X
        self.P_Y = self.P_Y
        self.P_X_base = self.P_X_base
        self.P_Y_base = self.P_Y_base
        self.Li = self.Li
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))
        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        self.Li = torch.inverse(L)
        if self.use_cuda:
            self.Li = self.Li
        return self.Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        batch_size = theta.size()[0]
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 1].size() + (1, self.N))
        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)
        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])
        points_X_prime = A_X[:, :, :, :, 0] + torch.mul(A_X[:, :, :, :, 1], points_X_batch) + torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)
        points_Y_prime = A_Y[:, :, :, :, 0] + torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)
        return torch.cat((points_X_prime, points_Y_prime), 3)


class UnetBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False, with_tanh=True):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            if with_tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, with_tanh=True):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, with_tanh=with_tanh)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, with_tanh=with_tanh)
        unet_block = UnetBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, with_tanh=with_tanh)
        unet_block = UnetBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, with_tanh=with_tanh)
        unet_block = UnetBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, with_tanh=with_tanh)
        self.model = UnetBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, with_tanh=with_tanh)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class Vgg19(nn.Module):

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class NNLoss(nn.Module):

    def __init__(self):
        super(NNLoss, self).__init__()

    def forward(self, predicted, ground_truth, nh=5, nw=5):
        v_pad = nh // 2
        h_pad = nw // 2
        val_pad = nn.ConstantPad2d((v_pad, v_pad, h_pad, h_pad), -10000)(ground_truth)
        reference_tensors = []
        for i_begin in range(0, nh):
            i_end = i_begin - nh + 1
            i_end = None if i_end == 0 else i_end
            for j_begin in range(0, nw):
                j_end = j_begin - nw + 1
                j_end = None if j_end == 0 else j_end
                sub_tensor = val_pad[:, :, i_begin:i_end, j_begin:j_end]
                reference_tensors.append(sub_tensor.unsqueeze(-1))
        reference = torch.cat(reference_tensors, dim=-1)
        ground_truth = ground_truth.unsqueeze(dim=-1)
        predicted = predicted.unsqueeze(-1)
        abs = torch.abs(reference - predicted)
        norms = torch.sum(abs, dim=1)
        loss, _ = torch.min(norms, dim=-1)
        loss = torch.mean(loss)
        return loss


def create_part(source_img, source_parse, part, for_L1):
    if part == 'cloth':
        filter_part = source_parse[:, 5, :, :] + source_parse[:, 6, :, :] + source_parse[:, 7, :, :]
    elif part == 'image_without_cloth':
        filter_part = 1 - (source_parse[:, 5, :, :] + source_parse[:, 6, :, :] + source_parse[:, 7, :, :])
    elif part == 'face':
        filter_part = source_parse[:, 1, :, :] + source_parse[:, 2, :, :] + source_parse[:, 4, :, :] + source_parse[:, 13, :, :]
    elif part == 'foreground':
        filter_part = torch.sum(source_parse[:, 1:, :, :], dim=1)
    elif part == 'downcloth':
        filter_part = source_parse[:, 9, :, :] + source_parse[:, 12, :, :] + source_parse[:, 16, :, :] + source_parse[:, 17, :, :] + source_parse[:, 18, :, :] + source_parse[:, 19, :, :]
    elif part == 'shoe':
        filter_part = source_parse[:, 14, :, :] + source_parse[:, 15, :, :] + source_parse[:, 18, :, :] + source_parse[:, 19, :, :]
    elif part == 'hand':
        pass
    elif part == 'post_process':
        filter_part = torch.sum(source_parse[:, 1:, :, :], dim=1)
        filter_part = torch.unsqueeze(filter_part, 1).float()
        source_img = source_img.float()
        source_img = source_img * filter_part + (1 - filter_part) * 0.8
        return source_img.float()
    if for_L1:
        filter_part = torch.unsqueeze(filter_part, 1).float()
        source_img = source_img.float()
        source_img = source_img * filter_part + (1 - filter_part)
    else:
        filter_part = torch.unsqueeze(filter_part, 1).float()
        source_img = source_img.float()
        source_img = source_img * filter_part
    return source_img.float()


class VGGLoss(torch.nn.Module):

    def __init__(self, model_path, requires_grad=False):
        super(VGGLoss, self).__init__()
        self.model = models.vgg19()
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)
        self.model = torch.nn.DataParallel(self.model)
        vgg_pretrained_features = self.model.module.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.loss = nn.L1Loss()
        self.lossmse = nn.MSELoss()
        self.norm = FeatureL2Norm()
        self.nnloss = NNLoss()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        self.slice = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]
        for i in range(len(self.slice)):
            self.slice[i] = torch.nn.DataParallel(self.slice[i])
        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pred, target, target_parse, masksampled, gram, nearest, use_l1=True):
        weight = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        weight.reverse()
        loss = 0
        if gram:
            loss_conv12 = self.lossmse(self.gram(self.slice[0](pred)), self.gram(self.slice[0](target)))
        elif nearest:
            loss_conv12 = self.nnloss(self.slice[0](pred), self.slice[0](target))
        else:
            loss_conv12 = self.loss(self.slice[0](pred), self.slice[0](target))
        for i in range(5):
            if not masksampled:
                if gram:
                    gram_pred = self.gram(self.slice[i](pred))
                    gram_target = self.gram(self.slice[i](target))
                else:
                    gram_pred = self.slice[i](pred)
                    gram_target = self.slice[i](target)
                if use_l1:
                    loss = loss + weight[i] * self.loss(gram_pred, gram_target)
                else:
                    loss = loss + weight[i] * self.lossmse(gram_pred, gram_target)
            else:
                pred = create_part(pred, target_parse, 'cloth')
                target = create_part(pred, target_parse, 'cloth')
                if gram:
                    gram_pred = self.gram(self.slice[i](pred))
                    gram_target = self.gram(self.slice[i](target))
                else:
                    gram_pred = self.slice[i](pred)
                    gram_target = self.slice[i](target)
                if use_l1:
                    loss = loss + weight[i] * self.loss(gram_pred, gram_target)
                else:
                    loss = loss + weight[i] * self.lossmse(gram_pred, gram_target)
        return loss, loss_conv12

    def gram(self, x):
        bs, ch, h, w = x.size()
        f = x.view(bs, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (ch * h * w)
        return G


class GMM(nn.Module):
    """ Geometric Matching Module
    """

    def __init__(self, opt):
        super(GMM, self).__init__()
        self.extractionA = FeatureExtraction(22, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d)
        self.extractionB = FeatureExtraction(3, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=192, output_dim=2 * opt.grid_size ** 2, use_cuda=True)
        self.gridGen = TpsGridGen(opt.fine_height, opt.fine_width, use_cuda=True, grid_size=opt.grid_size)

    def forward(self, inputA, inputB):
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA, featureB)
        theta = self.regression(correlation)
        grid = self.gridGen(theta)
        return grid, theta


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.InstanceNorm2d):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.net = nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias), norm_layer(ndf * 2), nn.LeakyReLU(0.2, True), nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias), nn.Sigmoid())

    def forward(self, input):
        return self.net(input)


class PatchDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(PatchDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetDiscriminator(nn.Module):

    def __init__(self, input_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', use_sigmoid=True, n_downsampling=2):
        assert n_blocks >= 0
        super(ResnetDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        if n_downsampling <= 2:
            for i in range(n_downsampling):
                mult = 2 ** i
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        elif n_downsampling == 3:
            mult = 2 ** 0
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]
            mult = 2 ** 1
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]
            mult = 2 ** 2
            model += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult), nn.ReLU(True)]
        if n_downsampling <= 2:
            mult = 2 ** n_downsampling
        else:
            mult = 4
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class DilatedResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(DilatedResnetBlock, self).__init__()
        self.conv_block1 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout=use_dropout, use_bias=use_bias, num_dilation=1, num_padding=1)
        self.conv_block2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout=use_dropout, use_bias=use_bias, num_dilation=2, num_padding=2)
        self.conv_block3 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout=use_dropout, use_bias=use_bias, num_dilation=3, num_padding=3)
        self.conv_block4 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout=use_dropout, use_bias=use_bias, num_dilation=4, num_padding=4)
        self.joint1 = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, kernel_size=1, padding=0, bias=use_bias))
        self.joint2 = nn.Sequential(nn.Conv2d(dim * 4, dim, kernel_size=1, padding=0, bias=use_bias))

    def build_conv_block(self, dim, padding_type, norm_layer, num_dilation, num_padding, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(num_padding)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(num_padding)]
        elif padding_type == 'zero':
            p = num_padding
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=num_dilation), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(num_padding)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(num_padding)]
        elif padding_type == 'zero':
            p = num_padding
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias, dilation=num_dilation), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        child1 = self.conv_block1(x)
        child2 = self.conv_block2(x)
        child3 = self.conv_block3(x)
        child4 = self.conv_block4(x)
        node1 = self.joint1(torch.cat((child1, child2), dim=1))
        node2 = self.joint1(torch.cat((child3, child4), dim=1))
        node = self.joint2(torch.cat((node1, node2), dim=1))
        out = x + node
        return out


class TreeResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', with_tanh=True):
        assert n_blocks >= 0
        super(TreeResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [DilatedResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if with_tanh:
            model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class Identity(nn.Module):

    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_layer == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('norm layer [%s] is not implemented' % norm_type)
    return norm_layer


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[0, 1, 2, 3]):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids)
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net = torch.nn.DataParallel(net)
    init_weights(net, init_type, init_gain=init_gain)
    return net


class Define_G(nn.Module):

    def __init__(self, input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=4, with_tanh=True):
        super(Define_G, self).__init__()
        net = None
        norm_layer = get_norm_layer(norm_type=norm)
        self.netG = netG
        if netG == 'treeresnet':
            net = TreeResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, with_tanh=with_tanh)
        elif netG == 'unet_128':
            net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        elif netG == 'unet_256':
            net = UnetGenerator(input_nc, output_nc, 6, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
        self.model = init_net(net, init_type, init_gain, gpu_ids)

    def forward(self, x):
        return self.model(x)


class Define_D(nn.Module):

    def __init__(self, input_nc, ndf, netD, n_layers_D=3, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=4, n_blocks=3):
        super(Define_D, self).__init__()
        net = None
        norm_layer = get_norm_layer(norm_type=norm)
        if netD == 'basic':
            net = PatchDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
        elif netD == 'n_layers':
            net = PatchDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
        elif netD == 'pixel':
            net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
        elif netD == 'resnet_blocks':
            net = ResnetDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=True, n_blocks=n_blocks)
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
        self.model = init_net(net, init_type, init_gain, gpu_ids)

    def forward(self, x):
        return self.model(x)


class NewL1Loss(nn.Module):

    def __init__(self):
        super(NewL1Loss, self).__init__()

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        max_diff = torch.max(diff)
        weight = diff / max_diff
        loss = weight * diff
        return loss.mean()


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        """
        If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer, you should register them as buffers.
        Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        buffer will be saved while loading buffer ===
        class Test(nn.Module):
            def __init__(self, module):
                super(Test, self).__init__()
                self.module = module
                self.register_param()
            
            def register_param():
                exist_w = hasattr(self.module, 'w')
                if not exist_w:
                    w = nn.Parameter(torch.ones(1))
                    self.module.register_parameter(w) # register 'w' to module

            def forward(self, x)
                return x
        """
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            if isinstance(prediction[0], list):
                loss = 0
                for pred in prediction:
                    pred_ = pred[-1]
                    target_tensor = self.get_target_tensor(pred_, target_is_real)
                    loss += self.loss(pred_, target_tensor)
                return loss
            else:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class PixelWiseBCELoss(nn.Module):

    def __init__(self, weight):
        super(PixelWiseBCELoss, self).__init__()
        self.weight = weight
        self.loss = BCEWithLogitsLoss()

    def forward(self, pred, target):
        loss = 0
        for index in range(pred.size(1)):
            loss += self.weight[index] * self.loss(pred[:, index, :, :], target[:, index, :, :])
        return loss


class PixelSoftmaxLoss(nn.Module):

    def __init__(self, weight):
        super(PixelSoftmaxLoss, self).__init__()
        self.loss = CrossEntropyLoss(weight=weight)

    def forward(self, pred, target):
        pred = pred.reshape(pred.size(0), pred.size(1), -1)
        _, pos = torch.topk(target, 1, 1, True)
        pos = pos.reshape(pos.size(0), -1)
        loss = self.loss(pred, pos)
        return loss


class TVLoss(nn.Module):

    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class AffineLayer(nn.Module):

    def __init__(self):
        super(AffineLayer, self).__init__()

    def forward(self, input, transforms):
        num_transforms = transforms.shape[1]
        N, C, H, W = input.shape
        input = input.unsqueeze(-1)
        input = input.repeat(1, num_transforms, 1, 1, 1)
        input = input.view(N * num_transforms, C, H, W)
        transforms = transforms[:, :, :6].view(-1, 2, 3)
        transforms = self.normalize_transforms(transforms, H, W)
        grid = F.affine_grid(transforms, input.shape)
        warped_map = F.grid_sample(input, grid)
        warped_map = warped_map.view(-1, num_transforms, C, H, W)
        return warped_map

    def normalize_transforms(self, transforms, H, W):
        transforms[:, 0, 0] = transforms[:, 0, 0]
        transforms[:, 0, 1] = transforms[:, 0, 1] * W / H
        transforms[:, 0, 2] = transforms[:, 0, 2] * 2 / H + transforms[:, 0, 0] + transforms[:, 0, 1] - 1
        transforms[:, 1, 0] = transforms[:, 1, 0] * H / W
        transforms[:, 1, 1] = transforms[:, 1, 1]
        transforms[:, 1, 2] = transforms[:, 1, 2] * 2 / W + transforms[:, 1, 0] + transforms[:, 1, 1] - 1
        return transforms


class AffineTransformLayer(nn.Module):

    def __init__(self, number_of_transforms, init_image_size, warp_skip):
        super(AffineTransformLayer, self).__init__()
        self.number_of_transforms = number_of_transforms
        self.init_image_size = init_image_size
        self.affine_layer = AffineLayer()
        self.warp_skip = warp_skip

    def forward(self, input, warps, masks):
        self.image_size = input.shape[2:]
        self.affine_mul = torch.Tensor([1, 1, self.init_image_size[0] / self.image_size[0], 1, 1, self.init_image_size[1] / self.image_size[1], 1, 1])
        warps = warps / self.affine_mul
        affine_transform = self.affine_layer(input, warps)
        if self.warp_skip == 'mask':
            masks = torch.from_numpy(np.array([cv2.resize(np.transpose(mask, [1, 2, 0]), (self.image_size[1], self.image_size[0])) for mask in masks.data.cpu().numpy()]))
            masks = masks.permute(0, 3, 1, 2)
            masks = torch.unsqueeze(masks, dim=2).float()
            affine_transform = affine_transform * masks
        res, _ = torch.max(affine_transform, dim=1)
        return res


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeatureCorrelation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureExtraction,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (FeatureL2Norm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NNLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (NewL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelDiscriminator,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TVLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Vgg19,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_NandhaKishorM_Deep_Virtual_Try_On(_paritybench_base):
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

