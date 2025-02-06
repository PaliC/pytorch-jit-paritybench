import sys
_module = sys.modules[__name__]
del sys
TestAccuracy = _module
TrainModel = _module
augmentations = _module
croppingDataset = _module
croppingModel = _module
demo_eval = _module
mobilenetv2 = _module
rod_align = _module
build = _module
rod_align = _module
rod_align = _module
roi_align = _module
build = _module
roi_align = _module
roi_align = _module
thop = _module
count_hooks = _module
profile = _module
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


from torch.autograd import Variable


import torch.backends.cudnn as cudnn


import torch.utils.data as data


from scipy.stats import spearmanr


from scipy.stats import pearsonr


import torch.optim as optim


import numpy as np


import random


from torchvision import transforms


import types


from numpy import random


import torch.nn as nn


import torchvision.models as models


import torch.nn.init as init


from torch.autograd import Function


from torch.nn.modules.module import Module


from torch.nn.functional import avg_pool2d


from torch.nn.functional import max_pool2d


import logging


from torch.nn.modules.conv import _ConvNd


class vgg_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4):
        super(vgg_base, self).__init__()
        vgg = models.vgg16(pretrained=True)
        if downsample == 4:
            self.feature = nn.Sequential(vgg.features[:-1])
        elif downsample == 5:
            self.feature = nn.Sequential(vgg.features)
        self.feature3 = nn.Sequential(vgg.features[:23])
        self.feature4 = nn.Sequential(vgg.features[23:30])
        self.feature5 = nn.Sequential(vgg.features[30:])

    def forward(self, x):
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class resnet50_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4):
        super(resnet50_base, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.feature3 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1, resnet50.layer2)
        self.feature4 = nn.Sequential(resnet50.layer3)
        self.feature5 = nn.Sequential(resnet50.layer4)

    def forward(self, x):
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.identity = stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_1x1_bn(inp, oup):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, input_size=224, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        self.cfgs = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, output_channel, s, t))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        self.classifier = nn.Linear(output_channel, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
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


class mobilenetv2_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4, model_path='pretrained_model/mobilenetv2_1.0-0c6065bc.pth'):
        super(mobilenetv2_base, self).__init__()
        model = MobileNetV2(width_mult=1.0)
        if loadweights:
            model.load_state_dict(torch.load(model_path))
        self.feature3 = nn.Sequential(model.features[:7])
        self.feature4 = nn.Sequential(model.features[7:14])
        self.feature5 = nn.Sequential(model.features[14:])

    def forward(self, x):
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class shufflenetv2_base(nn.Module):

    def __init__(self, loadweights=True, downsample=4, model_path='pretrained_model/shufflenetv2_x1_69.402_88.374.pth.tar'):
        super(shufflenetv2_base, self).__init__()
        model = shufflenetv2(width_mult=1.0)
        if loadweights:
            model.load_state_dict(torch.load(model_path))
        self.feature3 = nn.Sequential(model.conv1, model.maxpool, model.features[:4])
        self.feature4 = nn.Sequential(model.features[4:12])
        self.feature5 = nn.Sequential(model.features[12:])

    def forward(self, x):
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class RoDAlignFunction(Function):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        if features.is_cuda:
            rod_align.rod_align_forward_cuda(self.aligned_height, self.aligned_width, self.spatial_scale, features, rois, output)
        else:
            rod_align.rod_align_forward(self.aligned_height, self.aligned_width, self.spatial_scale, features, rois, output)
        return output

    def backward(self, grad_output):
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = self.rois.new(batch_size, num_channels, data_height, data_width).zero_()
        rod_align.rod_align_backward_cuda(self.aligned_height, self.aligned_width, self.spatial_scale, grad_output, self.rois, grad_input)
        return grad_input, None


class RoDAlignAvg(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoDAlignAvg, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoDAlignFunction(self.aligned_height + 1, self.aligned_width + 1, self.spatial_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


class RoIAlignFunction(Function):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        self.rois = rois
        self.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, self.aligned_height, self.aligned_width).zero_()
        if features.is_cuda:
            roi_align.roi_align_forward_cuda(self.aligned_height, self.aligned_width, self.spatial_scale, features, rois, output)
        else:
            roi_align.roi_align_forward(self.aligned_height, self.aligned_width, self.spatial_scale, features, rois, output)
        return output

    def backward(self, grad_output):
        batch_size, num_channels, data_height, data_width = self.feature_size
        grad_input = self.rois.new(batch_size, num_channels, data_height, data_width).zero_()
        roi_align.roi_align_backward_cuda(self.aligned_height, self.aligned_width, self.spatial_scale, grad_output, self.rois, grad_input)
        return grad_input, None


class RoIAlignAvg(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignAvg, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 1, self.spatial_scale)(features, rois)
        return avg_pool2d(x, kernel_size=2, stride=1)


def fc_layers(reddim=32, alignsize=8):
    conv1 = nn.Sequential(nn.Conv2d(reddim, 768, kernel_size=alignsize, padding=0), nn.ReLU(inplace=True))
    conv2 = nn.Sequential(nn.Conv2d(768, 128, kernel_size=1), nn.ReLU(inplace=True))
    conv3 = nn.Conv2d(128, 1, kernel_size=1)
    layers = nn.Sequential(conv1, conv2, dropout, conv3)
    return layers


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


class crop_model_single_scale(nn.Module):

    def __init__(self, alignsize=8, reddim=8, loadweight=True, model=None, downsample=4):
        super(crop_model_single_scale, self).__init__()
        if model == 'shufflenetv2':
            self.Feat_ext = shufflenetv2_base(loadweight, downsample)
            if downsample == 4:
                self.DimRed = nn.Conv2d(232, reddim, kernel_size=1, padding=0)
            else:
                self.DimRed = nn.Conv2d(464, reddim, kernel_size=1, padding=0)
        elif model == 'mobilenetv2':
            self.Feat_ext = mobilenetv2_base(loadweight, downsample)
            if downsample == 4:
                self.DimRed = nn.Conv2d(96, reddim, kernel_size=1, padding=0)
            else:
                self.DimRed = nn.Conv2d(320, reddim, kernel_size=1, padding=0)
        elif model == 'vgg16':
            self.Feat_ext = vgg_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(512, reddim, kernel_size=1, padding=0)
        elif model == 'resnet50':
            self.Feat_ext = resnet50_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(1024, reddim, kernel_size=1, padding=0)
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
        self.FC_layers = fc_layers(reddim * 2, alignsize)

    def forward(self, im_data, boxes):
        f3, base_feat, f5 = self.Feat_ext(im_data)
        red_feat = self.DimRed(base_feat)
        RoI_feat = self.RoIAlign(red_feat, boxes)
        RoD_feat = self.RoDAlign(red_feat, boxes)
        final_feat = torch.cat((RoI_feat, RoD_feat), 1)
        prediction = self.FC_layers(final_feat)
        return prediction

    def _init_weights(self):
        None
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)


class crop_model_multi_scale_individual(nn.Module):

    def __init__(self, alignsize=8, reddim=32, loadweight=True, model=None, downsample=4):
        super(crop_model_multi_scale_individual, self).__init__()
        if model == 'shufflenetv2':
            self.Feat_ext1 = shufflenetv2_base(loadweight, downsample)
            self.Feat_ext2 = shufflenetv2_base(loadweight, downsample)
            self.Feat_ext3 = shufflenetv2_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(232, reddim, kernel_size=1, padding=0)
        elif model == 'mobilenetv2':
            self.Feat_ext1 = mobilenetv2_base(loadweight, downsample)
            self.Feat_ext2 = mobilenetv2_base(loadweight, downsample)
            self.Feat_ext3 = mobilenetv2_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(96, reddim, kernel_size=1, padding=0)
        elif model == 'vgg16':
            self.Feat_ext1 = vgg_base(loadweight, downsample)
            self.Feat_ext2 = vgg_base(loadweight, downsample)
            self.Feat_ext3 = vgg_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(512, reddim, kernel_size=1, padding=0)
        self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=1.0 / 2.0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
        self.FC_layers = fc_layers(reddim * 2, alignsize)

    def forward(self, im_data, boxes):
        base_feat = self.Feat_ext1(im_data)
        up_im = self.upsample2(im_data)
        up_feat = self.Feat_ext2(up_im)
        up_feat = self.downsample2(up_feat)
        down_im = self.downsample2(im_data)
        down_feat = self.Feat_ext3(down_im)
        down_feat = self.upsample2(down_feat)
        cat_feat = 0.5 * base_feat + 0.35 * up_feat + 0.15 * down_feat
        red_feat = self.DimRed(cat_feat)
        RoI_feat = self.RoIAlign(red_feat, boxes)
        RoD_feat = self.RoDAlign(red_feat, boxes)
        final_feat = torch.cat((RoI_feat, RoD_feat), 1)
        prediction = self.FC_layers(final_feat)
        return prediction

    def _init_weights(self):
        None
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)


class crop_model_multi_scale_shared(nn.Module):

    def __init__(self, alignsize=8, reddim=32, loadweight=True, model=None, downsample=4):
        super(crop_model_multi_scale_shared, self).__init__()
        if model == 'shufflenetv2':
            self.Feat_ext = shufflenetv2_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(812, reddim, kernel_size=1, padding=0)
        elif model == 'mobilenetv2':
            self.Feat_ext = mobilenetv2_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(448, reddim, kernel_size=1, padding=0)
        elif model == 'vgg16':
            self.Feat_ext = vgg_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(1536, reddim, kernel_size=1, padding=0)
        elif model == 'resnet50':
            self.Feat_ext = resnet50_base(loadweight, downsample)
            self.DimRed = nn.Conv2d(3584, reddim, kernel_size=1, padding=0)
        self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=1.0 / 2.0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0 / 2 ** downsample)
        self.FC_layers = fc_layers(reddim * 2, alignsize)

    def forward(self, im_data, boxes):
        f3, f4, f5 = self.Feat_ext(im_data)
        cat_feat = torch.cat((self.downsample2(f3), f4, 0.5 * self.upsample2(f5)), 1)
        red_feat = self.DimRed(cat_feat)
        RoI_feat = self.RoIAlign(red_feat, boxes)
        RoD_feat = self.RoDAlign(red_feat, boxes)
        final_feat = torch.cat((RoI_feat, RoD_feat), 1)
        prediction = self.FC_layers(final_feat)
        return prediction

    def _init_weights(self):
        None
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)


class RoDAlign(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoDAlign, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoDAlignFunction(self.aligned_height, self.aligned_width, self.spatial_scale)(features, rois)


class RoDAlignMax(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoDAlignMax, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoDAlignFunction(self.aligned_height + 1, self.aligned_width + 1, self.spatial_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)


class RoIAlign(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlign, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return RoIAlignFunction(self.aligned_height, self.aligned_width, self.spatial_scale)(features, rois)


class RoIAlignMax(Module):

    def __init__(self, aligned_height, aligned_width, spatial_scale):
        super(RoIAlignMax, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        x = RoIAlignFunction(self.aligned_height + 1, self.aligned_width + 1, self.spatial_scale)(features, rois)
        return max_pool2d(x, kernel_size=2, stride=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (resnet50_base,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (vgg_base,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_HuiZeng_Grid_Anchor_based_Image_Cropping_Pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

