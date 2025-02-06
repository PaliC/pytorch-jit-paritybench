import sys
_module = sys.modules[__name__]
del sys
data_augmentation = _module
demo = _module
main = _module
deeplab_v3_plus = _module
hed_res = _module
hed_vgg16 = _module
hf_fcn_res = _module
hf_fcn_vgg16 = _module
models = _module
pspnet = _module
spp = _module
unet = _module
aug_GDAL = _module
aug_PIL = _module
dataset = _module
metrics = _module
plots = _module
trainval = _module
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


import random


import numpy as np


import torchvision.transforms.functional as tf


import math


from torch import nn


from torch import optim


from torch.utils.tensorboard import SummaryWriter


import torchvision


import torch.nn.functional as F


from torchvision.transforms import transforms


from torchvision import transforms


from torch.utils import data


from torch.backends import cudnn


from torch.optim import lr_scheduler


class ASPP(nn.Module):

    def __init__(self, in_channel=512, depth=256):
        super().__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1), nn.ReLU(inplace=True))
        self.atrous_block1 = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1), nn.ReLU(inplace=True))
        self.atrous_block6 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6), nn.ReLU(inplace=True))
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12), nn.ReLU(inplace=True))
        self.atrous_block18 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18), nn.ReLU(inplace=True))
        self.conv_1x1_output = nn.Sequential(nn.Conv2d(depth * 5, depth, 1, 1), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear', align_corners=True)
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6, atrous_block12, atrous_block18], dim=1))
        return net


class Deeplab_v3_plus(nn.Module):

    def __init__(self, class_number=5, fine_tune=True, backbone='resnet50'):
        super().__init__()
        encoder = getattr(torchvision.models, backbone)(pretrained=fine_tune)
        self.start = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.low_feature = nn.Sequential(nn.Conv2d(64, 48, 1, 1), nn.ReLU(inplace=True))
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.aspp = ASPP(in_channel=self.layer4[-1].conv1.in_channels, depth=256)
        self.conv_cat = nn.Sequential(nn.Conv2d(256 + 48, 256, 3, 1, padding=1), nn.ReLU(inplace=True))
        self.conv_cat1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU(inplace=True))
        self.conv_cat2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, padding=1), nn.ReLU(inplace=True))
        self.score = nn.Conv2d(256, class_number, 1, 1)

    def forward(self, x):
        size1 = x.shape[2:]
        x = self.start(x)
        xm = self.maxpool(x)
        x = self.layer1(xm)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        low_feature = self.low_feature(xm)
        size2 = low_feature.shape[2:]
        decoder_feature = F.upsample(x, size=size2, mode='bilinear', align_corners=True)
        conv_cat = self.conv_cat(torch.cat([low_feature, decoder_feature], dim=1))
        conv_cat1 = self.conv_cat1(conv_cat)
        conv_cat2 = self.conv_cat2(conv_cat1)
        score_small = self.score(conv_cat2)
        score = F.upsample(score_small, size=size1, mode='bilinear', align_corners=True)
        return score


class HED_res34(nn.Module):

    def __init__(self, num_filters=32, pretrained=False, class_number=2):
        super().__init__()
        encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.start = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.d_convs = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.scores = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer1 = encoder.layer1
        self.d_conv1 = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.score1 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.layer2 = encoder.layer2
        self.d_conv2 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))
        self.score2 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.layer3 = encoder.layer3
        self.d_conv3 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.score3 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.layer4 = encoder.layer4
        self.d_conv4 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.score4 = nn.UpsamplingBilinear2d(scale_factor=32)
        self.score = nn.Conv2d(5, class_number, 1, 1)

    def forward(self, x):
        x = self.start(x)
        s_x = self.d_convs(x)
        ss = self.scores(s_x)
        x = self.pool(x)
        x = self.layer1(x)
        s_x = self.d_conv1(x)
        s1 = self.score1(s_x)
        x = self.layer2(x)
        s_x = self.d_conv2(x)
        s2 = self.score2(s_x)
        x = self.layer3(x)
        s_x = self.d_conv3(x)
        s3 = self.score3(s_x)
        x = self.layer4(x)
        s_x = self.d_conv4(x)
        s4 = self.score4(s_x)
        score = self.score(torch.cat([s1, s2, s3, s4, ss], axis=1))
        return score


class HED_vgg16(nn.Module):

    def __init__(self, num_filters=32, pretrained=False, class_number=2):
        super().__init__()
        encoder = torchvision.models.vgg16(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = encoder[0:4]
        self.score1 = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv2 = encoder[5:9]
        self.d_conv2 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))
        self.score2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3 = encoder[10:16]
        self.d_conv3 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.score3 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv4 = encoder[17:23]
        self.d_conv4 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.score4 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.conv5 = encoder[24:30]
        self.d_conv5 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.score5 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.score = nn.Conv2d(5, class_number, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        s1 = self.score1(x)
        x = self.pool(x)
        x = self.conv2(x)
        s_x = self.d_conv2(x)
        s2 = self.score2(s_x)
        x = self.pool(x)
        x = self.conv3(x)
        s_x = self.d_conv3(x)
        s3 = self.score3(s_x)
        x = self.pool(x)
        x = self.conv3(x)
        s_x = self.d_conv4(x)
        s4 = self.score4(s_x)
        x = self.pool(x)
        x = self.conv5(x)
        s_x = self.d_conv5(x)
        s5 = self.score5(s_x)
        score = self.score(torch.cat([s1, s2, s3, s4, s5], axis=1))
        return score


class HF_res34(nn.Module):

    def __init__(self, class_number=2, pretrained=True, num_filters=32):
        super().__init__()
        encoder = torchvision.models.resnet34(pretrained=pretrained)
        self.start = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.d_convs = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.layer10 = encoder.layer1[0]
        self.layer11 = encoder.layer1[1]
        self.layer12 = encoder.layer1[2]
        self.d_conv10 = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv11 = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv12 = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.layer20 = encoder.layer2[0]
        self.layer21 = encoder.layer2[1]
        self.layer22 = encoder.layer2[2]
        self.layer23 = encoder.layer2[3]
        self.d_conv20 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv21 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv22 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv23 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))
        self.layer30 = encoder.layer3[0]
        self.layer31 = encoder.layer3[1]
        self.layer32 = encoder.layer3[2]
        self.layer33 = encoder.layer3[3]
        self.layer34 = encoder.layer3[4]
        self.layer35 = encoder.layer3[5]
        self.d_conv30 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv31 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv32 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv33 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv34 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv35 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.layer40 = encoder.layer4[0]
        self.layer41 = encoder.layer4[1]
        self.layer42 = encoder.layer4[2]
        self.d_conv40 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv41 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.d_conv42 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.score = nn.Conv2d(17, class_number, 1, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.start(x)
        s_x = self.d_convs(x)
        ss = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        """ why no relu after upsample ? because before it , d_conv has relu """
        x = self.pool(x)
        x = self.layer10(x)
        s_x = self.d_conv10(x)
        s10 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer11(x)
        s_x = self.d_conv11(x)
        s11 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer12(x)
        s_x = self.d_conv12(x)
        s12 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer20(x)
        s_x = self.d_conv20(x)
        s20 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer21(x)
        s_x = self.d_conv21(x)
        s21 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer22(x)
        s_x = self.d_conv22(x)
        s22 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer23(x)
        s_x = self.d_conv23(x)
        s23 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer30(x)
        s_x = self.d_conv30(x)
        s30 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer31(x)
        s_x = self.d_conv31(x)
        s31 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer32(x)
        s_x = self.d_conv32(x)
        s32 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer33(x)
        s_x = self.d_conv33(x)
        s33 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer34(x)
        s_x = self.d_conv34(x)
        s34 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer35(x)
        s_x = self.d_conv35(x)
        s35 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer40(x)
        s_x = self.d_conv40(x)
        s40 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer41(x)
        s_x = self.d_conv41(x)
        s41 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        x = self.layer42(x)
        s_x = self.d_conv42(x)
        s42 = F.upsample(s_x, size=input_size, mode='bilinear', align_corners=True)
        cat = [ss, s10, s11, s12, s20, s21, s22, s23, s30, s31, s32, s33, s34, s35, s40, s41, s42]
        score = self.score(torch.cat(cat, dim=1))
        return score


class HF_FCN(nn.Module):

    def __init__(self, class_number=2, pretrained=True, num_filters=32):
        super().__init__()
        encoder = torchvision.models.vgg16(pretrained=pretrained).features
        self.maxpool = encoder[4]
        self.conv1_1 = encoder[0:2]
        self.dconv1_1 = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv1_2 = encoder[2:4]
        self.dconv1_2 = nn.Sequential(nn.Conv2d(num_filters * 2, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv2_1 = encoder[5:7]
        self.dconv2_1 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv2_2 = encoder[7:9]
        self.dconv2_2 = nn.Sequential(nn.Conv2d(num_filters * 4, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv3_1 = encoder[10:12]
        self.dconv3_1 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv3_2 = encoder[12:14]
        self.dconv3_2 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv3_3 = encoder[14:16]
        self.dconv3_3 = nn.Sequential(nn.Conv2d(num_filters * 8, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv4_1 = encoder[17:19]
        self.dconv4_1 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv4_2 = encoder[19:21]
        self.dconv4_2 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv4_3 = encoder[21:23]
        self.dconv4_3 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv5_1 = encoder[24:26]
        self.dconv5_1 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv5_2 = encoder[26:28]
        self.dconv5_2 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.conv5_3 = encoder[28:30]
        self.dconv5_3 = nn.Sequential(nn.Conv2d(num_filters * 16, 1, 1, 1), nn.ReLU(inplace=True))
        self.score = nn.Conv2d(13, class_number, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        x = self.conv1_1(x)
        s1_1 = self.dconv1_1(x)
        x = self.conv1_2(x)
        s1_2 = self.dconv1_2(x)
        x = self.maxpool(x)
        x = self.conv2_1(x)
        s = self.dconv2_1(x)
        s2_1 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv2_2(x)
        s = self.dconv2_2(x)
        s2_2 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.maxpool(x)
        x = self.conv3_1(x)
        s = self.dconv3_1(x)
        s3_1 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv3_2(x)
        s = self.dconv3_2(x)
        s3_2 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv3_3(x)
        s = self.dconv3_3(x)
        s3_3 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.maxpool(x)
        x = self.conv4_1(x)
        s = self.dconv4_1(x)
        s4_1 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv4_2(x)
        s = self.dconv4_2(x)
        s4_2 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv4_3(x)
        s = self.dconv4_3(x)
        s4_3 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.maxpool(x)
        x = self.conv5_1(x)
        s = self.dconv5_1(x)
        s5_1 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv5_2(x)
        s = self.dconv5_2(x)
        s5_2 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        x = self.conv5_3(x)
        s = self.dconv5_3(x)
        s5_3 = F.upsample(s, size=size, mode='bilinear', align_corners=True)
        score = self.score(torch.cat([s1_1, s1_2, s2_1, s2_2, s3_1, s3_2, s3_3, s4_1, s4_2, s4_3, s5_1, s5_2, s5_3], dim=1))
        return score


def atrous_conv3x3(in_planes, out_planes, rate=1, padding=1, stride=1):
    """3x3  atrous convolution and no bias"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=rate, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution and no bias; downsample 1/stride"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, first_inplanes, inplanes, planes, rate=1, padding=1, stride=1, downsample=None):
        """
        pspnet conv1_3's num_output=128 not 64 so we modify some code
        first_inplanes: only layer1 not same (conv1_3)128 != (layer1-block1-conv1k_1s)64
        """
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = atrous_conv3x3(planes, planes, rate, padding)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if first_inplanes != inplanes and downsample is not None:
            self.conv1 = conv1x1(first_inplanes, planes, stride)
            self.downsample = nn.Sequential(conv1x1(first_inplanes, planes * self.expansion, stride), nn.BatchNorm2d(planes * self.expansion))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SppBlock(nn.Module):

    def __init__(self, level, in_channel=2048, out_numput=512):
        super().__init__()
        self.level = level
        self.convblock = nn.Sequential(conv1x1(in_channel, out_numput), nn.BatchNorm2d(out_numput), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        x = F.adaptive_avg_pool2d(x, output_size=(self.level, self.level))
        x = self.convblock(x)
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)
        return x


class SppBlock1(nn.Module):

    def __init__(self, level, k, s, in_channel=2048, out_numput=512):
        super().__init__()
        self.level = level
        self.avgpool = nn.AvgPool2d(k, s)
        self.convblock = nn.Sequential(conv1x1(in_channel, out_numput), nn.BatchNorm2d(out_numput), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[2:]
        x = self.avgpool(x)
        x = self.convblock(x)
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)
        return x


class SPP(nn.Module):

    def __init__(self, in_channel=2048):
        super().__init__()
        self.spp1 = SppBlock(level=1, in_channel=in_channel)
        self.spp2 = SppBlock(level=2, in_channel=in_channel)
        self.spp3 = SppBlock(level=3, in_channel=in_channel)
        self.spp6 = SppBlock(level=6, in_channel=in_channel)

    def forward(self, x):
        x1 = self.spp1(x)
        x2 = self.spp2(x)
        x3 = self.spp3(x)
        x6 = self.spp6(x)
        out = torch.cat([x, x1, x2, x3, x6], dim=1)
        return out


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))


class PSPNet(nn.Module):

    def __init__(self, block, layers, class_number, dropout_rate=0.2, in_channel=3):
        super().__init__()
        self.inplanes = 64
        self.conv1_1 = conv3x3_bn_relu(in_channel, 64, stride=2)
        self.conv1_2 = conv3x3_bn_relu(64, 64)
        self.conv1_3 = conv3x3_bn_relu(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, 64, layers[0])
        self.layer2 = self._make_layer(block, 256, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, 256, layers[2], rate=2, padding=2)
        self.layer4 = self._make_layer(block, 1024, 512, layers[2], rate=4, padding=4)
        self.spp = SPP(in_channel=2048)
        self.conv5_4 = conv3x3_bn_relu(2048 + 512 * 4, 512)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv6 = nn.Conv2d(512, class_number, 1, 1)
        """ init weight """
        None
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        size = x.shape[2:]
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.spp(x)
        x = self.conv5_4(x)
        x = self.dropout(x)
        x = self.conv6(x)
        x = F.upsample(x, size, mode='bilinear', align_corners=True)
        return x
    """first_inplanes, inplanes, planes, rate=1, padding=1, stride=1, downsample=None"""

    def _make_layer(self, block, first_inplanes, planes, blocks, rate=1, padding=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(first_inplanes, self.inplanes, planes, rate, padding, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, planes, rate, padding))
        return nn.Sequential(*layers)


class SppNet(nn.Module):

    def __init__(self, batch_size=1, out_pool_size=[1, 2, 4], class_number=2):
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=False).features[:-1]
        self.out_pool_size = out_pool_size
        self.batch_size = batch_size
        self.encoder = vgg
        self.spp = self.make_spp(batch_size=batch_size, out_pool_size=out_pool_size)
        sum0 = 0
        for i in out_pool_size:
            sum0 += i ** 2
        self.fc = nn.Sequential(nn.Linear(512 * sum0, 1024), nn.ReLU(inplace=True))
        self.score = nn.Linear(1024, class_number)

    def make_spp(self, batch_size=1, out_pool_size=[1, 2, 4]):
        func = []
        for i in range(len(out_pool_size)):
            func.append(nn.AdaptiveAvgPool2d(output_size=(out_pool_size[i], out_pool_size[i])))
        return func

    def forward(self, x):
        assert x.shape[0] == 1, 'batch size need to set to be 1'
        encoder = self.encoder(x)
        spp = []
        for i in range(len(self.out_pool_size)):
            spp.append(self.spp[i](encoder).view(self.batch_size, -1))
        fc = self.fc(torch.cat(spp, dim=1))
        score = self.score(fc)
        return score


class SppNet1(nn.Module):

    def __init__(self, batch_size=1, out_pool_size=[1, 2, 4], class_number=2):
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=False).features[:-1]
        self.out_pool_size = out_pool_size
        self.batch_size = batch_size
        self.encoder = vgg
        sum0 = 0
        for i in out_pool_size:
            sum0 += i ** 2
        self.fc = nn.Sequential(nn.Linear(512 * sum0, 1024), nn.ReLU(inplace=True))
        self.score = nn.Linear(1024, class_number)

    def forward(self, x):
        assert x.shape[0] == 1, 'batch size need to set to be 1'
        encoder = self.encoder(x)
        spp = []
        for i in range(len(self.out_pool_size)):
            spp.append(F.adaptive_avg_pool2d(encoder, output_size=(self.out_pool_size[i], self.out_pool_size[i])).view(self.batch_size, -1))
        fc = self.fc(torch.cat(spp, dim=1))
        score = self.score(fc)
        return score


def concat(in_features1, in_features2):
    return torch.cat([in_features1, in_features2], dim=1)


def upsample(in_features, out_features):
    shape = out_features.shape[2:]
    return F.upsample(in_features, size=shape, mode='bilinear', align_corners=True)


class U_Net(nn.Module):

    def __init__(self, class_number=5, in_channels=3):
        super().__init__()
        self.conv1_1 = conv3x3_bn_relu(in_channels, 64)
        self.conv1_2 = conv3x3_bn_relu(64, 64)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2_1 = conv3x3_bn_relu(64, 128)
        self.conv2_2 = conv3x3_bn_relu(128, 128)
        self.conv3_1 = conv3x3_bn_relu(128, 256)
        self.conv3_2 = conv3x3_bn_relu(256, 256)
        self.conv4_1 = conv3x3_bn_relu(256, 512)
        self.conv4_2 = conv3x3_bn_relu(512, 512)
        self.conv5_1 = conv3x3_bn_relu(512, 1024)
        self.conv5_2 = conv3x3_bn_relu(1024, 1024)
        self.conv6 = conv3x3_bn_relu(1024, 512)
        self.conv6_1 = conv3x3_bn_relu(1024, 512)
        self.conv6_2 = conv3x3_bn_relu(512, 512)
        self.conv7 = conv3x3_bn_relu(512, 256)
        self.conv7_1 = conv3x3_bn_relu(512, 256)
        self.conv7_2 = conv3x3_bn_relu(256, 256)
        self.conv8 = conv3x3_bn_relu(256, 128)
        self.conv8_1 = conv3x3_bn_relu(256, 128)
        self.conv8_2 = conv3x3_bn_relu(128, 128)
        self.conv9 = conv3x3_bn_relu(128, 64)
        self.conv9_1 = conv3x3_bn_relu(128, 64)
        self.conv9_2 = conv3x3_bn_relu(64, 64)
        self.score = nn.Conv2d(64, class_number, 1, 1)

    def forward(self, x):
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.maxpool(conv1_2)
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.maxpool(conv2_2)
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.maxpool(conv3_2)
        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.maxpool(conv4_2)
        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        up6 = upsample(conv5_2, conv4_2)
        conv6 = self.conv6(up6)
        merge6 = concat(conv6, conv4_2)
        conv6_1 = self.conv6_1(merge6)
        conv6_2 = self.conv6_2(conv6_1)
        up7 = upsample(conv6_2, conv3_2)
        conv7 = self.conv7(up7)
        merge7 = concat(conv7, conv3_2)
        conv7_1 = self.conv7_1(merge7)
        conv7_2 = self.conv7_2(conv7_1)
        up8 = upsample(conv7_2, conv2_2)
        conv8 = self.conv8(up8)
        merge8 = concat(conv8, conv2_2)
        conv8_1 = self.conv8_1(merge8)
        conv8_2 = self.conv8_2(conv8_1)
        up9 = upsample(conv8_2, conv1_2)
        conv9 = self.conv9(up9)
        merge9 = concat(conv9, conv1_2)
        conv9_1 = self.conv9_1(merge9)
        conv9_2 = self.conv9_2(conv9_1)
        score = self.score(conv9_2)
        return score


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 512, 4, 4])], {}),
     True),
    (Deeplab_v3_plus,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (HF_FCN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (HF_res34,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SPP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048, 4, 4])], {}),
     True),
    (SppBlock,
     lambda: ([], {'level': 4}),
     lambda: ([torch.rand([4, 2048, 64, 64])], {}),
     True),
    (SppBlock1,
     lambda: ([], {'level': 4, 'k': 4, 's': 4}),
     lambda: ([torch.rand([4, 2048, 4, 4])], {}),
     True),
    (U_Net,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_gengyanlei_segmentation_pytorch(_paritybench_base):
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

