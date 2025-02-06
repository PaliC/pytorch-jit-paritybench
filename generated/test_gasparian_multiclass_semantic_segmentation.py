import sys
_module = sys.modules[__name__]
del sys
eval = _module
model_summary = _module
train = _module
FPN = _module
TTA = _module
Unet = _module
utils = _module
cityscapes_utils = _module
dataset = _module
kitti_lane_utils = _module
predictor = _module
trainer = _module

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


import re


import time


import random


import warnings


import numpy as np


import torch


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


import torchvision


from torch import nn


from torch import cat


from torch import squeeze


from torch.utils.data import Dataset


import copy


import logging


import torch.backends.cudnn as cudnn


from torch.optim.lr_scheduler import ReduceLROnPlateau


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, activation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)
        x0 = self.encoder.relu(x0)
        x1 = self.encoder.maxpool(x0)
        x1 = self.encoder.layer1(x1)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)
        x = self.decoder([x4, x3, x2, x1, x0])
        return x

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function (if activation is not `None`) with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)
        return x


class Conv3x3GNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False), nn.GroupNorm(32, out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPNBlock(nn.Module):

    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):

    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()
        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class FPNDecoder(Model):

    def __init__(self, encoder_channels, pyramid_channels=256, segmentation_channels=128, final_upsampling=4, final_channels=1, dropout=0.2, merge_policy='add'):
        super().__init__()
        if merge_policy not in ['add', 'cat']:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(merge_policy))
        self.merge_policy = merge_policy
        self.final_upsampling = final_upsampling
        self.conv1 = nn.Conv2d(encoder_channels[1], pyramid_channels, kernel_size=(1, 1))
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[3])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[4])
        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        if self.merge_policy == 'cat':
            segmentation_channels *= 4
        self.final_conv = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)
        self.initialize()

    def forward(self, x):
        c5, c4, c3, c2, _ = x
        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])
        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)
        if self.merge_policy == 'add':
            x = s5 + s4 + s3 + s2
        elif self.merge_policy == 'cat':
            x = torch.cat([s5, s4, s3, s2], dim=1)
        x = self.dropout(x)
        x = self.final_conv(x)
        if self.final_upsampling is not None and self.final_upsampling > 1:
            x = F.interpolate(x, scale_factor=self.final_upsampling, mode='bilinear', align_corners=True)
        return x


def get_encoder(model, pretrained=True):
    if model == 'resnet18':
        encoder = torchvision.models.resnet18(pretrained=pretrained)
    elif model == 'resnet34':
        encoder = torchvision.models.resnet34(pretrained=pretrained)
    elif model == 'resnet50':
        encoder = torchvision.models.resnet50(pretrained=pretrained)
    elif model == 'resnext50':
        encoder = torchvision.models.resnext50_32x4d(pretrained=pretrained)
    elif model == 'resnext101':
        encoder = torchvision.models.resnext101_32x8d(pretrained=pretrained)
    if model in ['resnet18', 'resnet34']:
        model = 'resnet18-34'
    else:
        model = 'resnet50-101'
    filters_dict = {'resnet18-34': [512, 512, 256, 128, 64], 'resnet50-101': [2048, 2048, 1024, 512, 256]}
    return encoder, filters_dict[model]


class FPN(EncoderDecoder):
    """FPN_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_pyramid_channels: a number of convolution filters in Feature Pyramid of FPN_.
        decoder_segmentation_channels: a number of convolution filters in segmentation head of FPN_.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        dropout: spatial dropout rate in range (0, 1).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        final_upsampling: optional, final upsampling factor
            (default is 4 to preserve input -> output spatial shape identity)
        decoder_merge_policy: determines how to merge outputs inside FPN.
            One of [``add``, ``cat``]
    Returns:
        ``torch.nn.Module``: **FPN**
    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
    """

    def __init__(self, encoder_name='resnet34', pretrained=True, decoder_pyramid_channels=256, decoder_segmentation_channels=128, classes=1, dropout=0.2, activation='sigmoid', final_upsampling=4, decoder_merge_policy='add'):
        encoder, filters_dict = get_encoder(encoder_name, pretrained)
        encoder.out_shapes = filters_dict
        decoder = FPNDecoder(encoder_channels=encoder.out_shapes, pyramid_channels=decoder_pyramid_channels, segmentation_channels=decoder_segmentation_channels, final_channels=classes, dropout=dropout, final_upsampling=final_upsampling, merge_policy=decoder_merge_policy)
        super().__init__(encoder, decoder, activation)
        self.name = 'fpn-{}'.format(encoder_name)


def vflip(x):
    """flip batch of images vertically"""
    return x.flip(2)


def inv_transform(x):
    """
    input: 4D tensor
    output: 4D tensor
    inverse transform: [orig, vflip->orig]
    """
    x[1] = vflip(x[1])
    return x


def transform(x):
    """
    input: 3D tensor
    output: 4D tensor: [orig, vflip]
    """
    output = [x.unsqueeze(0), vflip(x).unsqueeze(0)]
    return torch.cat(output)


class TTAWrapper(nn.Module):

    def __init__(self, model, merge_mode='mean', activate=False, temperature=0.5):
        super().__init__()
        self.model = model
        self.activate = activate
        self.temperature = temperature
        self.merge_mode = merge_mode
        if self.merge_mode not in ['mean', 'tsharpen']:
            raise ValueError('Merge type is not correct: `{}`.'.format(self.merge_mode))

    def forward(self, images):
        result = []
        batch_size = images.size(0)
        for image in images:
            augmented = transform(image)
            aug_prediction = self.model(augmented)
            aug_prediction = inv_transform(aug_prediction)
            if self.merge_mode == 'mean':
                result.append(aug_prediction.sum(0))
            elif self.merge_mode == 'tsharpen':
                aug_prediction[aug_prediction < 0] = 0
                result.append(torch.pow(aug_prediction[0], self.temperature))
                for pred in aug_prediction[1:]:
                    result[-1] += torch.pow(pred, self.temperature)
            result[-1] = (result[-1] / batch_size).unsqueeze(0)
        return torch.cat(result)


class ConvRelu(nn.Module):

    def __init__(self, in_: 'int', out: 'int', activate=True, batch_norm=False):
        super(ConvRelu, self).__init__()
        self.activate = activate
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(out)
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.activate:
            x = self.activation(x)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: 'int', num_filters: 'int', batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(num_filters)
        self.activation = nn.ReLU(inplace=True)
        self.conv_block = ConvRelu(in_channels, num_filters, activate=True, batch_norm=True)
        self.conv_block_na = ConvRelu(in_channels, num_filters, activate=False, batch_norm=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inp):
        x = self.conv_block(inp)
        x = self.conv_block_na(x)
        if self.batch_norm:
            x = self.bn(x)
        x = x.add(inp)
        x = self.activation(x)
        return x


class DecoderBlockResNet(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    https://distill.pub/2016/deconv-checkerboard/

    About residual blocks:  
    http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, in_channels, middle_channels, out_channels, res_blocks_dec=False):
        super(DecoderBlockResNet, self).__init__()
        self.in_channels = in_channels
        self.res_blocks_dec = res_blocks_dec
        layers_list = [ConvRelu(in_channels, middle_channels, activate=True, batch_norm=False)]
        if self.res_blocks_dec:
            layers_list.append(ResidualBlock(middle_channels, middle_channels, batch_norm=True))
        layers_list.append(nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1))
        if not self.res_blocks_dec:
            layers_list.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.block(x)


class UnetResNet(nn.Module):

    def __init__(self, input_channels=3, num_classes=1, num_filters=32, res_blocks_dec=False, Dropout=0.2, encoder_name='resnet50', pretrained=True):
        super().__init__()
        self.encoder, self.filters_dict = get_encoder(encoder_name, pretrained)
        self.num_classes = num_classes
        self.Dropout = Dropout
        self.res_blocks_dec = res_blocks_dec
        self.input_channels = input_channels
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        if self.input_channels != 3:
            self.channel_tuner = nn.Conv2d(input_channels, 3, kernel_size=1)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4
        self.center = DecoderBlockResNet(self.filters_dict[0], num_filters * 8 * 2, num_filters * 8, res_blocks_dec=False)
        self.dec5 = DecoderBlockResNet(self.filters_dict[1] + num_filters * 8, num_filters * 8 * 2, num_filters * 8, res_blocks_dec=self.res_blocks_dec)
        self.dec4 = DecoderBlockResNet(self.filters_dict[2] + num_filters * 8, num_filters * 8 * 2, num_filters * 8, res_blocks_dec=self.res_blocks_dec)
        self.dec3 = DecoderBlockResNet(self.filters_dict[3] + num_filters * 8, num_filters * 4 * 2, num_filters * 2, res_blocks_dec=self.res_blocks_dec)
        self.dec2 = DecoderBlockResNet(self.filters_dict[4] + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, res_blocks_dec=self.res_blocks_dec)
        self.dec1 = DecoderBlockResNet(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, res_blocks_dec=False)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        self.dropout_2d = nn.Dropout2d(p=self.Dropout)

    def forward(self, x, z=None):
        if self.input_channels != 3:
            x = self.channel_tuner(x)
        conv1 = self.conv1(x)
        conv2 = self.dropout_2d(self.conv2(conv1))
        conv3 = self.dropout_2d(self.conv3(conv2))
        conv4 = self.dropout_2d(self.conv4(conv3))
        conv5 = self.dropout_2d(self.conv5(conv4))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(cat([center, conv5], 1))
        dec4 = self.dec4(cat([dec5, conv4], 1))
        dec3 = self.dec3(cat([dec4, conv3], 1))
        dec2 = self.dec2(cat([dec3, conv2], 1))
        dec2 = self.dropout_2d(dec2)
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        return self.final(dec0)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvRelu,
     lambda: ([], {'in_': 4, 'out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderBlockResNet,
     lambda: ([], {'in_channels': 4, 'middle_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FPN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (FPNBlock,
     lambda: ([], {'pyramid_channels': 4, 'skip_channels': 4}),
     lambda: ([(torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 16, 16]))], {}),
     False),
    (FPNDecoder,
     lambda: ([], {'encoder_channels': [4, 4, 4, 4, 4]}),
     lambda: ([(torch.rand([4, 4, 8, 8]), torch.rand([4, 4, 16, 16]), torch.rand([4, 4, 32, 32]), torch.rand([4, 4, 64, 64]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (ResidualBlock,
     lambda: ([], {'in_channels': 4, 'num_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TTAWrapper,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnetResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_gasparian_multiclass_semantic_segmentation(_paritybench_base):
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

