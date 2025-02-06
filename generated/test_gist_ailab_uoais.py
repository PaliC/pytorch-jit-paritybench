import sys
_module = sys.modules[__name__]
del sys
adet = _module
checkpoint = _module
adet_checkpoint = _module
config = _module
defaults = _module
data = _module
pycocotools = _module
coco = _module
cocoeval = _module
mask = _module
augmentation = _module
builtin = _module
builtin_meta = _module
dataset_mapper = _module
detection_utils = _module
register_uoais = _module
register_wisdom = _module
uoais = _module
wisdom = _module
evaluation = _module
amodal_evaluation = _module
amodalvisible_evaluation = _module
evaluator = _module
rrc_evaluation_funcs = _module
text_eval_script = _module
text_evaluation = _module
visible_evaluation = _module
layers = _module
bezier_align = _module
conv_with_kaiming_uniform = _module
def_roi_align = _module
deform_conv = _module
gcn = _module
iou_loss = _module
ml_nms = _module
naive_group_norm = _module
modeling = _module
backbone = _module
bifpn = _module
cbam = _module
dla = _module
fpn = _module
lpf = _module
mobilenet = _module
resnet_depth = _module
resnet_interval = _module
resnet_lpf = _module
resnet_r2d_guide = _module
rgbdfpn = _module
vovnet = _module
poolers = _module
rcnn = _module
bbox_pooler = _module
box_head = _module
faster_rcnn = _module
mask_heads = _module
pooler = _module
rcnn_heads = _module
roi_heads = _module
attn_predictor = _module
text_head = _module
structures = _module
beziers = _module
utils = _module
comm = _module
measures = _module
post_process = _module
visualizer = _module
compute_PRF = _module
eval_on_OCID = _module
eval_on_OSD = _module
eval_utils = _module
munkres = _module
model = _module
uoais_node = _module
setup = _module
tools = _module
k4a_demo = _module
rs_demo = _module
run_on_OSD = _module
train_net = _module
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


import random


import numpy as np


import copy


import logging


import torch


import torch.nn.functional as F


import itertools


from collections import OrderedDict


from math import pi


import time


from collections import abc


from typing import List


from typing import Union


from torch import nn


import re


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.nn.modules.utils import _pair


import torch.nn as nn


from torch.nn import Module


from torch.nn import Parameter


from torch.nn import init


import math


import torch.utils.model_zoo as model_zoo


import torch.nn.parallel


from torch.nn import BatchNorm2d


from torchvision.ops import RoIPool


from torch.nn import functional as F


from typing import Callable


from typing import Any


from typing import TypeVar


from typing import Tuple


from abc import abstractmethod


from typing import Dict


from typing import Optional


from torch.autograd import Variable


import torch.distributed as dist


from enum import Enum


from enum import unique


import matplotlib as mpl


import matplotlib.colors as mplc


import matplotlib.figure as mplfigure


from matplotlib.backends.backend_agg import FigureCanvasAgg


from matplotlib import cm


from torch.utils.cpp_extension import CUDA_HOME


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


class _BezierAlign(Function):

    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio, aligned):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        ctx.aligned = aligned
        output = _C.bezier_align_forward(input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.bezier_align_backward(grad_output, rois, spatial_scale, output_size[0], output_size[1], bs, ch, h, w, sampling_ratio, ctx.aligned)
        return grad_input, None, None, None, None, None


bezier_align = _BezierAlign.apply


class BezierAlign(nn.Module):

    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling bezier_align. This produces the correct neighbors; see
            adet/tests/test_bezier_align.py for verification.

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
        """
        super(BezierAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx17 boxes. First column is the index into N. The other 16 columns are [xy]x8.
        """
        assert rois.dim() == 2 and rois.size(1) == 17
        return bezier_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ', aligned=' + str(self.aligned)
        tmpstr += ')'
        return tmpstr


class _DefROIAlign(Function):

    @staticmethod
    def forward(ctx, input, roi, offsets, output_size, spatial_scale, sampling_ratio, trans_std, aligned):
        ctx.save_for_backward(input, roi, offsets)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.trans_std = trans_std
        ctx.input_shape = input.size()
        ctx.aligned = aligned
        output = _C.def_roi_align_forward(input, roi, offsets, spatial_scale, output_size[0], output_size[1], sampling_ratio, trans_std, aligned)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        data, rois, offsets = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        trans_std = ctx.trans_std
        bs, ch, h, w = ctx.input_shape
        grad_offsets = torch.zeros_like(offsets)
        grad_input = _C.def_roi_align_backward(data, grad_output, rois, offsets, grad_offsets, spatial_scale, output_size[0], output_size[1], bs, ch, h, w, sampling_ratio, trans_std, ctx.aligned)
        return grad_input, None, grad_offsets, None, None, None, None, None


def_roi_align = _DefROIAlign.apply


class DefROIAlign(nn.Module):

    def __init__(self, output_size, spatial_scale, sampling_ratio, trans_std, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            trans_std (float): offset scale according to the normalized roi size
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.
        """
        super(DefROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.trans_std = trans_std
        self.aligned = aligned

    def forward(self, input, rois, offsets):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        return def_roi_align(input, rois, offsets, self.output_size, self.spatial_scale, self.sampling_ratio, self.trans_std, self.aligned)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ', sampling_ratio=' + str(self.sampling_ratio)
        tmpstr += ', trans_std=' + str(self.trans_std)
        tmpstr += ', aligned=' + str(self.aligned)
        tmpstr += ')'
        return tmpstr


class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class DFConv2d(nn.Module):
    """
    Deformable convolutional layer with configurable
    deformable groups, dilations and groups.

    Code is from:
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/misc.py


    """

    def __init__(self, in_channels, out_channels, with_modulated_dcn=True, kernel_size=3, stride=1, groups=1, dilation=1, deformable_groups=1, bias=False, padding=None):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert isinstance(stride, (list, tuple))
            assert isinstance(dilation, (list, tuple))
            assert len(kernel_size) == 2
            assert len(stride) == 2
            assert len(dilation) == 2
            padding = dilation[0] * (kernel_size[0] - 1) // 2, dilation[1] * (kernel_size[1] - 1) // 2
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            padding = dilation * (kernel_size - 1) // 2
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            offset_channels = offset_base_channels * 3
            conv_block = ModulatedDeformConv
        else:
            offset_channels = offset_base_channels * 2
            conv_block = DeformConv
        self.offset = Conv2d(in_channels, deformable_groups * offset_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=1, dilation=dilation)
        for l in [self.offset]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            torch.nn.init.constant_(l.bias, 0.0)
        self.conv = conv_block(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_split = offset_base_channels * deformable_groups * 2

    def forward(self, x, return_offset=False):
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset_mask = self.offset(x)
                x = self.conv(x, offset_mask)
            else:
                offset_mask = self.offset(x)
                offset = offset_mask[:, :self.offset_split, :, :]
                mask = offset_mask[:, self.offset_split:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            if return_offset:
                return x, offset_mask
            return x
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // d + 1) for i, p, di, k, d in zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class Conv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding='same', stride=1, dilation=1, groups=1):
        super(Conv2D, self).__init__()
        assert type(kernel_size) in [int, tuple], 'Allowed kernel type [int or tuple], not {}'.format(type(kernel_size))
        assert padding == 'same', 'Allowed padding type {}, not {}'.format('same', padding)
        self.kernel_size = kernel_size
        if isinstance(kernel_size, tuple):
            self.h_kernel = kernel_size[0]
            self.w_kernel = kernel_size[1]
        else:
            self.h_kernel = kernel_size
            self.w_kernel = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=self.stride, dilation=self.dilation, groups=self.groups)

    def forward(self, x):
        if self.padding == 'same':
            height, width = x.shape[2:]
            h_pad_need = max(0, (height - 1) * self.stride + self.h_kernel - height)
            w_pad_need = max(0, (width - 1) * self.stride + self.w_kernel - width)
            pad_left = w_pad_need // 2
            pad_right = w_pad_need - pad_left
            pad_top = h_pad_need // 2
            pad_bottom = h_pad_need - pad_top
            padding = pad_left, pad_right, pad_top, pad_bottom
            x = F.pad(x, padding, 'constant', 0)
        x = self.conv(x)
        return x


class GCN(nn.Module):
    """
        Large Kernel Matters -- https://arxiv.org/abs/1703.02719
    """

    def __init__(self, in_channels, out_channels, k=3):
        super(GCN, self).__init__()
        self.conv_l1 = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, 1), padding='same')
        self.conv_l2 = Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, k), padding='same')
        self.conv_r1 = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), padding='same')
        self.conv_r2 = Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=(k, 1), padding='same')

    def forward(self, x):
        x1 = self.conv_l1(x)
        x1 = self.conv_l2(x1)
        x2 = self.conv_r1(x)
        x2 = self.conv_r2(x2)
        out = x1 + x2
        return out


class IOULoss(nn.Module):
    """
    Intersetion Over Union (IoU) loss which supports three
    different IoU computations:

    * IoU
    * Linear IoU
    * gIoU
    """

    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, ious, gious=None, weight=None):
        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            assert gious is not None
            losses = 1 - gious
        else:
            raise NotImplementedError
        if weight is not None:
            return (losses * weight).sum()
        else:
            return losses.sum()


class NaiveGroupNorm(Module):
    """NaiveGroupNorm implements Group Normalization with the high-level matrix operations in PyTorch.
    It is a temporary solution to export GN by ONNX before the official GN can be exported by ONNX.
    The usage of NaiveGroupNorm is exactly the same as the official :class:`torch.nn.GroupNorm`.
    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\\text{num\\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = NaiveGroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = NaiveGroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = NaiveGroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine', 'weight', 'bias']

    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True):
        super(NaiveGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_channels))
            self.bias = Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        N, C, H, W = input.size()
        assert C % self.num_groups == 0
        input = input.reshape(N, self.num_groups, -1)
        mean = input.mean(dim=-1, keepdim=True)
        var = (input ** 2).mean(dim=-1, keepdim=True) - mean ** 2
        std = torch.sqrt(var + self.eps)
        input = (input - mean) / std
        input = input.reshape(N, C, H, W)
        if self.affine:
            input = input * self.weight.reshape(1, C, 1, 1) + self.bias.reshape(1, C, 1, 1)
        return input

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, affine={affine}'.format(**self.__dict__)


class FeatureMapResampler(nn.Module):

    def __init__(self, in_channels, out_channels, stride, norm=''):
        super(FeatureMapResampler, self).__init__()
        if in_channels != out_channels:
            self.reduction = Conv2d(in_channels, out_channels, kernel_size=1, bias=norm == '', norm=get_norm(norm, out_channels), activation=None)
        else:
            self.reduction = None
        assert stride <= 2
        self.stride = stride

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)
        if self.stride == 2:
            x = F.max_pool2d(x, kernel_size=self.stride + 1, stride=self.stride, padding=1)
        elif self.stride == 1:
            pass
        else:
            raise NotImplementedError()
        return x


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(Flatten(), nn.Linear(gate_channels, gate_channels // reduction_ratio), nn.ReLU(), nn.Linear(gate_channels // reduction_ratio, gate_channels))
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale, scale


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):

    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale, scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out, cha_att = self.ChannelGate(x)
        if not self.no_spatial:
            x_out, spa_att = self.SpatialGate(x_out)
        return x_out, cha_att, spa_att


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad2d
    else:
        None
    return PadLayer


class Downsample(nn.Module):

    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2)), int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))]
        self.pad_sizes = [(pad_size + pad_off) for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        if self.filt_size == 1:
            a = np.array([1.0])
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


_NORM = False


def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [(f'{module_name}_{postfix}/conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)), (f'{module_name}_{postfix}/norm', get_norm(_NORM, out_channels)), (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = nn.Sequential(Downsample(filt_size=filter_size, stride=stride, channels=planes), conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution with padding"""
    return [(f'{module_name}_{postfix}/conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)), (f'{module_name}_{postfix}/norm', get_norm(_NORM, out_channels)), (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, groups)
        self.bn2 = norm_layer(planes)
        if stride == 1:
            self.conv3 = conv1x1(planes, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(Downsample(filt_size=filter_size, stride=stride, channels=planes), conv1x1(planes, planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, cfg, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes, kernel_size=1, bias=False)
        self.bn1 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation, groups=cardinality)
        self.bn2 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes, kernel_size=1, bias=False)
        self.bn3 = get_norm(cfg.MODEL.DLA.NORM, planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out


class Root(nn.Module):

    def __init__(self, cfg, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = get_norm(cfg.MODEL.DLA.NORM, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Module):

    def __init__(self, cfg, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_kernel_size=1, dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(cfg, in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(cfg, out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(cfg, levels - 1, block, in_channels, out_channels, stride, root_dim=0, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(cfg, levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels, root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(cfg, root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False), get_norm(cfg.MODEL.DLA.NORM, out_channels))

    def forward(self, x, residual=None, children=None):
        if self.training and residual is not None:
            x = x + residual.sum() * 0.0
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature='res5'):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            weight_init.c2_xavier_fill(module)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class LastLevelP6(nn.Module):
    """
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, in_channels, out_channels, in_features='res5'):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_features
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            weight_init.c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        return [p6]


def get_pad_layer_1d(pad_type):
    if pad_type in ['refl', 'reflect']:
        PadLayer = nn.ReflectionPad1d
    elif pad_type in ['repl', 'replicate']:
        PadLayer = nn.ReplicationPad1d
    elif pad_type == 'zero':
        PadLayer = nn.ZeroPad1d
    else:
        None
    return PadLayer


class Downsample1D(nn.Module):

    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.0 * (filt_size - 1) / 2), int(np.ceil(1.0 * (filt_size - 1) / 2))]
        self.pad_sizes = [(pad_size + pad_off) for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels
        if self.filt_size == 1:
            a = np.array([1.0])
        elif self.filt_size == 2:
            a = np.array([1.0, 1.0])
        elif self.filt_size == 3:
            a = np.array([1.0, 2.0, 1.0])
        elif self.filt_size == 4:
            a = np.array([1.0, 3.0, 3.0, 1.0])
        elif self.filt_size == 5:
            a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        elif self.filt_size == 6:
            a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
        elif self.filt_size == 7:
            a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))
        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = 'p5'

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):

    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class _OSA_module(nn.Module):

    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False):
        super(_OSA_module, self).__init__()
        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))
        self.ese = eSEModule(concat_ch)

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.ese(xt)
        if self.identity:
            xt = xt + identity_feat
        return xt


class _OSA_stage(nn.Sequential):

    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=False):
        super(_OSA_stage, self).__init__()
        if not stage_num == 2:
            self.add_module('Pooling', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        if block_per_stage != 1:
            SE = False
        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE))
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:
                SE = False
            module_name = f'OSA{stage_num}_{i + 2}'
            self.add_module(module_name, _OSA_module(concat_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, identity=True))


def assign_boxes_to_levels(box_lists, min_level, max_level, canonical_box_size, canonical_level):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.
    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (sqrt(box area)).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.
    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    eps = sys.float_info.epsilon
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    level_assignments = torch.floor(canonical_level + torch.log2(box_sizes / canonical_box_size + eps))
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments - min_level


def convert_boxes_to_pooler_format(box_lists):
    """
    Convert all boxes in `box_lists` to the low-level format used by ROI pooling ops
    (see description under Returns).
    Args:
        box_lists (list[Boxes] | list[RotatedBoxes]):
            A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
    Returns:
        When input is list[Boxes]:
            A tensor of shape (M, 5), where M is the total number of boxes aggregated over all
            N batch images.
            The 5 columns are (batch index, x0, y0, x1, y1), where batch index
            is the index in [0, N) identifying which batch image the box with corners at
            (x0, y0, x1, y1) comes from.
        When input is list[RotatedBoxes]:
            A tensor of shape (M, 6), where M is the total number of boxes aggregated over all
            N batch images.
            The 6 columns are (batch index, x_ctr, y_ctr, width, height, angle_degrees),
            where batch index is the index in [0, N) identifying which batch image the
            rotated box (x_ctr, y_ctr, width, height, angle_degrees) comes from.
    """

    def fmt_box_list(box_tensor, batch_index):
        repeated_index = torch.full((len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device)
        return cat((repeated_index, box_tensor), dim=1)
    pooler_fmt_boxes = cat([fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0)
    return pooler_fmt_boxes


class BBOXROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(self, output_size, scales, sampling_ratio, pooler_type, canonical_box_size=224, canonical_level=4):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.
                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()
        if isinstance(output_size, int):
            output_size = output_size, output_size
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size
        if pooler_type == 'ROIAlign':
            self.level_poolers = nn.ModuleList(ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False) for scale in scales)
        elif pooler_type == 'ROIAlignV2':
            self.level_poolers = nn.ModuleList(ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True) for scale in scales)
        elif pooler_type == 'ROIPool':
            self.level_poolers = nn.ModuleList(RoIPool(output_size, spatial_scale=scale) for scale in scales)
        elif pooler_type == 'ROIAlignRotated':
            self.level_poolers = nn.ModuleList(ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio) for scale in scales)
        else:
            raise ValueError('Unknown pooler type: {}'.format(pooler_type))
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level)), 'Featuremap stride is not power of 2!'
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert len(scales) == self.max_level - self.min_level + 1, '[ROIPooler] Sizes of input featuremaps do not form a pyramid!'
        assert 0 < self.min_level and self.min_level <= self.max_level
        if len(scales) > 1:
            assert self.min_level <= canonical_level and canonical_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(self, x, box_lists):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.
        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        num_level_assignments = len(self.level_poolers)
        assert isinstance(x, list) and isinstance(box_lists, list), 'Arguments to pooler must be lists'
        assert len(x) == num_level_assignments, 'unequal value, num_level_assignments={}, but x is list of {} Tensors'.format(num_level_assignments, len(x))
        assert len(box_lists) == x[0].size(0), 'unequal value, x[0] batch dim 0 is {}, but box_list has length {}'.format(x[0].size(0), len(box_lists))
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        level_assignments = assign_boxes_to_levels(box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level)
        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]
        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros((num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device)
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = torch.nonzero(level_assignments == level).squeeze(1)
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)
        return output


class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(self, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        Args:
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()
        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas


class MaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: 'ShapeSpec'):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(MaskRCNNConvUpsampleHead, self).__init__()
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(input_channels if k == 0 else conv_dims, conv_dims, kernel_size=3, stride=1, padding=1, bias=not self.norm, norm=get_norm(self.norm, conv_dims), activation=F.relu)
            self.add_module('mask_fcn{}'.format(k + 1), conv)
            self.conv_norm_relus.append(conv)
        self.deconv = ConvTranspose2d(conv_dims if num_conv > 0 else input_channels, conv_dims, kernel_size=2, stride=2, padding=0)
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        for layer in (self.conv_norm_relus + [self.deconv]):
            weight_init.c2_msra_fill(layer)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)


class VisibleMaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: 'ShapeSpec'):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(VisibleMaskRCNNConvUpsampleHead, self).__init__()
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        self.input_channels = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        self.MLC = cfg.MODEL.MULTI_LEVEL_CODING
        self.occ_cls_at_mask = cfg.MODEL.OCC_CLS_AT_MASK
        self.hom = cfg.MODEL.HIERARCHCIAL_OCCLUSION_MODELING
        self.guidance_type = cfg.MODEL.GUIDANCE_TYPE
        self.prediction_order = cfg.MODEL.PREDICTION_ORDER
        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(self.input_channels if k == 0 else conv_dims, conv_dims, kernel_size=3, stride=1, padding=1, bias=not self.norm, norm=get_norm(self.norm, conv_dims), activation=F.relu)
            self.add_module('visible_mask_fcn{}'.format(k + 1), conv)
            self.conv_norm_relus.append(conv)
        self.deconv = ConvTranspose2d(conv_dims if num_conv > 0 else self.input_channels, conv_dims, kernel_size=2, stride=2, padding=0)
        self.mlc_layers = []
        self.mlc_layers.append(Conv2d(2 * conv_dims, 2 * conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2 * conv_dims, 2 * conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2 * conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
        for i, layer in enumerate(self.mlc_layers):
            self.add_module('visible_mlc_layer{}'.format(i), layer)
        self.guide_conv_layers = []
        n_features = self.prediction_order.index('V') + 1 if not cfg.MODEL.NO_DENSE_GUIDANCE else 1
        if self.hom and self.guidance_type == 'concat':
            self.guide_conv_layers.append(Conv2d(n_features * conv_dims, n_features * conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features * conv_dims, n_features * conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features * conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
            for i, layer in enumerate(self.guide_conv_layers):
                self.add_module('visible_guidance_layer{}'.format(i), layer)
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        for layer in (self.conv_norm_relus + [self.deconv, self.predictor] + self.mlc_layers + self.guide_conv_layers):
            weight_init.c2_msra_fill(layer)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x, instances, extracted_features):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances: contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
            vis_mask_features: visible mask features from visible mask head
        Returns:
            mask_logits: predicted mask logits from given region features
        """
        if self.hom and self.guidance_type == 'concat':
            for layer in self.guide_conv_layers:
                x = layer(x)
        for i, layer in enumerate(self.conv_norm_relus):
            if i == 0 and self.MLC:
                x = layer(x)
                x = torch.cat([x, extracted_features], 1)
                for mlc_layer in self.mlc_layers:
                    x = mlc_layer(x)
            else:
                x = layer(x)
        mask_logits = self.predictor(F.relu(self.deconv(x)))
        return mask_logits, x


class AmodalMaskRCNNConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: 'ShapeSpec'):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(AmodalMaskRCNNConvUpsampleHead, self).__init__()
        num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        self.input_channels = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        self.MLC = cfg.MODEL.MULTI_LEVEL_CODING
        self.occ_cls_at_mask = cfg.MODEL.OCC_CLS_AT_MASK
        self.hom = cfg.MODEL.HIERARCHCIAL_OCCLUSION_MODELING
        self.guidance_type = cfg.MODEL.GUIDANCE_TYPE
        self.prediction_order = cfg.MODEL.PREDICTION_ORDER
        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(self.input_channels if k == 0 else conv_dims, conv_dims, kernel_size=3, stride=1, padding=1, bias=not self.norm, norm=get_norm(self.norm, conv_dims), activation=F.relu)
            self.add_module('amodal_mask_fcn{}'.format(k + 1), conv)
            self.conv_norm_relus.append(conv)
        self.deconv = ConvTranspose2d(conv_dims if num_conv > 0 else self.input_channels, conv_dims, kernel_size=2, stride=2, padding=0)
        self.mlc_layers = []
        self.mlc_layers.append(Conv2d(2 * conv_dims, 2 * conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2 * conv_dims, 2 * conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2 * conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
        for i, layer in enumerate(self.mlc_layers):
            self.add_module('amodal_mlc_layer{}'.format(i), layer)
        self.guide_conv_layers = []
        n_features = self.prediction_order.index('A') + 1 if not cfg.MODEL.NO_DENSE_GUIDANCE else 2
        if self.hom and self.guidance_type == 'concat':
            self.guide_conv_layers.append(Conv2d(n_features * conv_dims, n_features * conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features * conv_dims, n_features * conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features * conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
            for i, layer in enumerate(self.guide_conv_layers):
                self.add_module('amodal_guidance_layer{}'.format(i), layer)
        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)
        for layer in (self.conv_norm_relus + [self.deconv, self.predictor] + self.mlc_layers + self.guide_conv_layers):
            weight_init.c2_msra_fill(layer)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, x, instances, extracted_features):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances: contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
            vis_mask_features: visible mask features from visible mask head
        Returns:
            mask_logits: predicted mask logits from given region features
        """
        if self.hom and self.guidance_type == 'concat':
            for layer in self.guide_conv_layers:
                x = layer(x)
        for i, layer in enumerate(self.conv_norm_relus):
            if i == 0 and self.MLC:
                x = layer(x)
                x = torch.cat([x, extracted_features], 1)
                for mlc_layer in self.mlc_layers:
                    x = mlc_layer(x)
            else:
                x = layer(x)
        mask_logits = self.predictor(F.relu(self.deconv(x)))
        return mask_logits, x


class OCCCLSMaskHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: 'ShapeSpec', name=''):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(OCCCLSMaskHead, self).__init__()
        conv_dims = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels = input_shape.channels
        self.MLC = cfg.MODEL.MULTI_LEVEL_CODING
        self.occ_cls_at_mask = cfg.MODEL.OCC_CLS_AT_MASK
        self.hom = cfg.MODEL.HIERARCHCIAL_OCCLUSION_MODELING
        self.guidance_type = cfg.MODEL.GUIDANCE_TYPE
        self.prediction_order = cfg.MODEL.PREDICTION_ORDER
        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(input_channels if k == 0 else conv_dims, conv_dims, kernel_size=3, stride=2 if k == 1 else 1, padding=1, bias=not self.norm, norm=get_norm(self.norm, conv_dims), activation=F.relu)
            self.add_module('{}_occ_cls_fcn{}'.format(name, k + 1), conv)
            self.conv_norm_relus.append(conv)
        self.mlc_layers = []
        self.mlc_layers.append(Conv2d(2 * conv_dims, 2 * conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2 * conv_dims, 2 * conv_dims, 3, 1, 1, activation=F.relu))
        self.mlc_layers.append(Conv2d(2 * conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
        for i, layer in enumerate(self.mlc_layers):
            self.add_module('occ_cls_{}_mlc_layer{}'.format(name, i), layer)
        self.guide_conv_layers = []
        n_features = self.prediction_order.index('O') + 1 if not cfg.MODEL.NO_DENSE_GUIDANCE else 2
        if self.hom and self.guidance_type == 'concat':
            self.guide_conv_layers.append(Conv2d(n_features * conv_dims, n_features * conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features * conv_dims, n_features * conv_dims, 3, 1, 1, activation=F.relu))
            self.guide_conv_layers.append(Conv2d(n_features * conv_dims, conv_dims, 3, 1, 1, activation=F.relu))
            for i, layer in enumerate(self.guide_conv_layers):
                self.add_module('occlusion_guidance_layer{}'.format(i), layer)
        self.deconv = ConvTranspose2d(conv_dims if num_conv > 0 else input_channels, conv_dims, kernel_size=2, stride=2, padding=0)
        input_size = input_shape.channels * (input_shape.width // 2 or 1) * (input_shape.height // 2 or 1)
        self.predictor = nn.Linear(input_size, 2)
        weight_init.c2_xavier_fill(self.predictor)
        for layer in (self.conv_norm_relus + self.mlc_layers + [self.predictor, self.deconv] + self.guide_conv_layers):
            weight_init.c2_msra_fill(layer)

    def forward(self, x, instances, extracted_features):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances: contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            mask_logits: predicted mask logits from given region features
        """
        if self.hom and self.guidance_type == 'concat':
            for layer in self.guide_conv_layers:
                x = layer(x)
        for i, layer in enumerate(self.conv_norm_relus):
            x = layer(x)
            if i == 0 and self.MLC:
                x = torch.cat([x, extracted_features], 1)
                for mlc_layer in self.mlc_layers:
                    x = mlc_layer(x)
        if x.dim() > 2:
            x_flatten = torch.flatten(x, start_dim=1)
        class_logits = self.predictor(x_flatten)
        return class_logits, self.deconv(x)


def _img_area(instance):
    device = instance.pred_classes.device
    image_size = instance.image_size
    area = torch.as_tensor(image_size[0] * image_size[1], dtype=torch.float, device=device)
    tmp = torch.zeros((len(instance.pred_classes), 1), dtype=torch.float, device=device)
    return (area + tmp).squeeze(1)


def assign_boxes_to_levels_by_ratio(instances, min_level, max_level, is_train=False):
    """
    Map each box in `instances` to a feature map level index by adaptive ROI mapping function 
    in CenterMask paper and return the assignment
    vector.
    Args:
        instances (list[Instances]): the per-image instances to train/predict masks.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    eps = sys.float_info.epsilon
    if is_train:
        box_lists = [x.proposal_boxes for x in instances]
    else:
        box_lists = [x.pred_boxes for x in instances]
    box_areas = cat([boxes.area() for boxes in box_lists])
    img_areas = cat([_img_area(instance_i) for instance_i in instances])
    level_assignments = torch.ceil(max_level - torch.log2(img_areas / box_areas + eps))
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments - min_level


class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(self, output_size, scales, sampling_ratio, pooler_type, canonical_box_size=224, canonical_level=4, assign_crit='area'):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as a 1 / s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.
                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()
        if isinstance(output_size, int):
            output_size = output_size, output_size
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size
        if pooler_type == 'ROIAlign':
            self.level_poolers = nn.ModuleList(ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False) for scale in scales)
        elif pooler_type == 'ROIAlignV2':
            self.level_poolers = nn.ModuleList(ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True) for scale in scales)
        elif pooler_type == 'ROIPool':
            self.level_poolers = nn.ModuleList(RoIPool(output_size, spatial_scale=scale) for scale in scales)
        elif pooler_type == 'ROIAlignRotated':
            self.level_poolers = nn.ModuleList(ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio) for scale in scales)
        else:
            raise ValueError('Unknown pooler type: {}'.format(pooler_type))
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(max_level, int(max_level)), 'Featuremap stride is not power of 2!'
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert len(scales) == self.max_level - self.min_level + 1, '[ROIPooler] Sizes of input featuremaps do not form a pyramid!'
        assert 0 < self.min_level and self.min_level <= self.max_level
        if len(scales) > 1:
            assert self.min_level <= canonical_level and canonical_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size
        self.assign_crit = assign_crit

    def forward(self, x, instances, is_train=False):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
            is_train (True/False)
        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """
        if is_train:
            box_lists = [x.proposal_boxes for x in instances]
        else:
            box_lists = [x.pred_boxes for x in instances]
        num_level_assignments = len(self.level_poolers)
        assert isinstance(x, list) and isinstance(box_lists, list), 'Arguments to pooler must be lists'
        assert len(x) == num_level_assignments, 'unequal value, num_level_assignments={}, but x is list of {} Tensors'.format(num_level_assignments, len(x))
        assert len(box_lists) == x[0].size(0), 'unequal value, x[0] batch dim 0 is {}, but box_list has length {}'.format(x[0].size(0), len(box_lists))
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        if self.assign_crit == 'ratio':
            level_assignments = assign_boxes_to_levels_by_ratio(instances, self.min_level, self.max_level, is_train)
        else:
            level_assignments = assign_boxes_to_levels(box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level)
        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]
        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros((num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device)
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = torch.nonzero(level_assignments == level).squeeze(1)
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)
        return output


class ROIHeads(nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.
    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.
    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: 'Dict[str, ShapeSpec]'):
        super(ROIHeads, self).__init__()
        self.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.proposal_matcher = Matcher(cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS, cfg.MODEL.ROI_HEADS.IOU_LABELS, allow_low_quality_matches=False)

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.
        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_classes[matched_labels == 0] = self.num_classes
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes)
        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)
        proposals_with_gt = []
        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes, proposals_per_image.proposal_boxes)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(matched_idxs, matched_labels, targets_per_image.gt_classes)
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for trg_name, trg_value in targets_per_image.get_fields().items():
                    if trg_name.startswith('gt_') and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4)))
                proposals_per_image.gt_boxes = gt_boxes
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
        storage = get_event_storage()
        storage.put_scalar('roi_head/num_fg_samples', np.mean(num_fg_samples))
        storage.put_scalar('roi_head/num_bg_samples', np.mean(num_bg_samples))
        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.
        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
            detected instances. Returned during inference only; may be [] during training.
            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


def fast_rcnn_inference_single_image(boxes, scores, occ_scores, image_shape, score_thresh, nms_thresh, topk_per_image):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).
    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.
    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        if occ_scores is not None:
            occ_scores = occ_scores[valid_mask]
    scores = scores[:, :-1]
    if occ_scores is not None:
        occ_scores, occ_preds = occ_scores.max(-1)
        occ_scores = torch.unsqueeze(occ_scores, -1)
        occ_preds = torch.unsqueeze(occ_preds, -1)
    num_bbox_reg_classes = boxes.shape[1] // 4
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)
    filter_mask = scores > score_thresh
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    if occ_scores is not None:
        result.pred_occlusion_scores = occ_scores[keep]
        result.pred_occlusions = occ_preds[keep]
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def fast_rcnn_inference(boxes, scores, occ_scores, image_shapes, score_thresh, nms_thresh, topk_per_image):
    """
    Call `fast_rcnn_inference_single_image` for all images.
    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.
    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [fast_rcnn_inference_single_image(boxes_per_image, scores_per_image, occ_scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image) for scores_per_image, occ_scores_per_image, boxes_per_image, image_shape in zip(scores, occ_scores, boxes, image_shapes)]
    return tuple(list(x) for x in zip(*result_per_image))


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(self, box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta, pred_occlusion_class_logits):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.pred_occlusion_class_logits = pred_occlusion_class_logits
        box_type = type(proposals[0].proposal_boxes)
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert not self.proposals.tensor.requires_grad, 'Proposals should not require gradients!'
        self.image_shapes = [x.image_size for x in proposals]
        if proposals[0].has('gt_boxes'):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has('gt_classes')
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            gt_occludeds = []
            for i, p in enumerate(proposals):
                if p.has('gt_occludeds'):
                    gt_occludeds.append(p.gt_occludeds)
                else:
                    gt_occludeds.append(torch.zeros_like(proposals[i].gt_classes))
            self.gt_occludeds = cat(gt_occludeds, dim=0)

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]
        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()
        storage = get_event_storage()
        storage.put_scalar('fast_rcnn/cls_accuracy', num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar('fast_rcnn/fg_cls_accuracy', fg_num_accurate / num_fg)
            storage.put_scalar('fast_rcnn/false_negative', num_false_negative / num_fg)

    def _log_occ_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_occludeds.numel()
        pred_gt_occludeds = self.pred_occlusion_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_occlusion_class_logits.shape[1] - 1
        fg_inds = (self.gt_occludeds >= 0) & (self.gt_occludeds < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_occludeds = self.gt_occludeds[fg_inds]
        fg_pred_gt_occludeds = pred_gt_occludeds[fg_inds]
        num_false_negative = (fg_pred_gt_occludeds == bg_class_ind).nonzero().numel()
        num_accurate = (pred_gt_occludeds == self.gt_occludeds).nonzero().numel()
        fg_num_accurate = (fg_pred_gt_occludeds == fg_gt_occludeds).nonzero().numel()
        storage = get_event_storage()
        storage.put_scalar('fast_rcnn/occ_cls_accuracy', num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar('fast_rcnn/fg_occ_cls_accuracy', fg_num_accurate / num_fg)
            storage.put_scalar('fast_rcnn/occ_false_negative', num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction='mean')

    def occ_softmax_cross_entropy_loss(self):
        self._log_occ_accuracy()
        n_occ = torch.sum(self.gt_occludeds)
        n_gt = self.gt_occludeds.shape[0]
        n_noocc = n_gt - n_occ
        return F.cross_entropy(self.pred_occlusion_class_logits, self.gt_occludeds, weight=torch.Tensor([1, n_noocc / n_occ]), reduction='mean')

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.
        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(self.proposals.tensor, self.gt_boxes.tensor)
        box_dim = gt_proposal_deltas.size(1)
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device
        bg_class_ind = self.pred_class_logits.shape[1] - 1
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(1)
        if cls_agnostic_bbox_reg:
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)
        loss_box_reg = smooth_l1_loss(self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols], gt_proposal_deltas[fg_inds], self.smooth_l1_beta, reduction='sum')
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.
        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        losses = {'loss_cls': self.softmax_cross_entropy_loss(), 'loss_box_reg': self.smooth_l1_loss()}
        if self.pred_occlusion_class_logits is not None:
            losses['loss_occ_cls'] = self.occ_softmax_cross_entropy_loss()
        return losses

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(self.pred_proposal_deltas.view(num_pred * K, B), self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B))
        return boxes.view(num_pred, K * B)

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_boxes_for_gt_classes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        predicted_boxes = self._predict_boxes()
        B = self.proposals.tensor.shape[1]
        if predicted_boxes.shape[1] > B:
            num_pred = len(self.proposals)
            num_classes = predicted_boxes.shape[1] // B
            gt_classes = torch.clamp(self.gt_classes, 0, num_classes - 1)
            predicted_boxes = predicted_boxes.view(num_pred, num_classes, B)[torch.arange(num_pred, dtype=torch.long, device=predicted_boxes.device), gt_classes]
        return predicted_boxes.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def predict_occ_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_occlusion_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        if self.pred_occlusion_class_logits is not None:
            occ_scores = self.predict_occ_probs()
        else:
            occ_scores = [None] * len(scores)
        image_shapes = self.image_shapes
        return fast_rcnn_inference(boxes, scores, occ_scores, image_shapes, score_thresh, nms_thresh, topk_per_image)


def build_amodal_mask_head(cfg, input_shape):
    """
    Build a amodal mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)


def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)


def build_visible_mask_head(cfg, input_shape):
    name = cfg.MODEL.ROI_VISIBLE_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)


def mask_rcnn_inference(pred_mask_logits, pred_instances, target='pred_masks'):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.set(target, prob)


def compute_dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = 2 * a / (b + c)
    return torch.mean(1 - d)


@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: 'torch.Tensor', instances: 'List[Instances]', target='gt_mask', dice_loss=False):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), 'Mask prediction must be square!'
    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes
            gt_classes.append(gt_classes_per_image)
        gt_masks_per_image = instances_per_image.get(target).crop_and_resize(instances_per_image.proposal_boxes.tensor, mask_side_len)
        gt_masks.append(gt_masks_per_image)
    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0
    gt_masks = cat(gt_masks, dim=0)
    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks
    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction='mean')
    if dice_loss:
        mask_loss += compute_dice_loss(pred_mask_logits.sigmoid(), gt_masks)
    return mask_loss


def occlusion_mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)
    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_occlusion_masks = prob


@torch.jit.unused
def occlusion_mask_rcnn_loss(pred_mask_logits: 'torch.Tensor', instances: 'List[Instances]', vis_period: 'int'=0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), 'Mask prediction must be square!'
    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes
            gt_classes.append(gt_classes_per_image)
        gt_masks_per_image = instances_per_image.gt_occluded_masks.crop_and_resize(instances_per_image.proposal_boxes.tensor, mask_side_len)
        gt_masks.append(gt_masks_per_image)
    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0
    gt_masks = cat(gt_masks, dim=0)
    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]
    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0)
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(gt_masks_bool.numel() - num_positive, 1.0)
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)
    storage = get_event_storage()
    storage.put_scalar('occlusion_mask_rcnn/accuracy', mask_accuracy)
    storage.put_scalar('occlusion_mask_rcnn/false_positive', false_positive)
    storage.put_scalar('occlusion_mask_rcnn/false_negative', false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = 'Left: mask prediction;   Right: mask GT'
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f' ({idx})', vis_mask)
    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction='mean')
    return mask_loss


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.
    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.
    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has('gt_classes')
    fg_proposals = []
    fg_selection_masks = []
    fg_indexes = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
        fg_indexes.append(fg_idxs)
    return fg_proposals, fg_selection_masks, fg_indexes


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


def conv_with_kaiming_uniform(norm=None, activation=None, use_deformable=False, use_sep=False):

    def make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1):
        if use_deformable:
            conv_func = DFConv2d
        else:
            conv_func = Conv2d
        if use_sep:
            assert in_channels == out_channels
            groups = in_channels
        else:
            groups = 1
        conv = conv_func(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation * (kernel_size - 1) // 2, dilation=dilation, groups=groups, bias=norm is None)
        if not use_deformable:
            nn.init.kaiming_uniform_(conv.weight, a=1)
            if norm is None:
                nn.init.constant_(conv.bias, 0)
        module = [conv]
        if norm is not None and len(norm) > 0:
            if norm == 'GN':
                norm_module = nn.GroupNorm(32, out_channels)
            else:
                norm_module = get_norm(norm, out_channels)
            module.append(norm_module)
        if activation is not None:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv
    return make_conv


class CRNN(nn.Module):

    def __init__(self, cfg, in_channels):
        super(CRNN, self).__init__()
        conv_func = conv_with_kaiming_uniform(norm='GN', activation=True)
        convs = []
        for i in range(2):
            convs.append(conv_func(in_channels, in_channels, 3, stride=(2, 1)))
        self.convs = nn.Sequential(*convs)
        self.rnn = BidirectionalLSTM(in_channels, in_channels, in_channels)

    def forward(self, x):
        x = self.convs(x)
        x = x.mean(dim=2)
        x = x.permute(2, 0, 1)
        x = self.rnn(x)
        return x


class Attention(nn.Module):

    def __init__(self, cfg, in_channels):
        super(Attention, self).__init__()
        self.hidden_size = in_channels
        self.output_size = cfg.MODEL.BATEXT.VOC_SIZE + 1
        self.dropout_p = 0.1
        self.max_len = cfg.MODEL.BATEXT.NUM_CHARS
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.vat = nn.Linear(self.hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):
        """
        hidden: 1 x n x self.hidden_size
        encoder_outputs: time_step x n x self.hidden_size (T,N,C)
        """
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        batch_size = encoder_outputs.shape[1]
        alpha = hidden + encoder_outputs
        alpha = alpha.view(-1, alpha.shape[-1])
        attn_weights = self.vat(torch.tanh(alpha))
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2, 1, 0))
        attn_weights = F.softmax(attn_weights, dim=2)
        attn_applied = torch.matmul(attn_weights, encoder_outputs.permute((1, 0, 2)))
        if embedded.dim() == 1:
            embedded = embedded.unsqueeze(0)
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result

    def prepare_targets(self, targets):
        target_lengths = (targets != self.output_size - 1).long().sum(dim=-1)
        sum_targets = [t[:l] for t, l in zip(targets, target_lengths)]
        return target_lengths, sum_targets


class ATTPredictor(nn.Module):

    def __init__(self, cfg):
        super(ATTPredictor, self).__init__()
        in_channels = cfg.MODEL.BATEXT.CONV_DIM
        self.CRNN = CRNN(cfg, in_channels)
        self.criterion = torch.nn.NLLLoss()
        self.attention = Attention(cfg, in_channels)
        self.teach_prob = 0.5

    def forward(self, rois, targets=None):
        rois = self.CRNN(rois)
        if self.training:
            target_variable = targets
            _init = torch.zeros((rois.size()[1], 1)).long()
            _init = torch.LongTensor(_init)
            target_variable = torch.cat((_init, target_variable.long()), 1)
            target_variable = target_variable
            decoder_input = target_variable[:, 0]
            decoder_hidden = self.attention.initHidden(rois.size()[1])
            loss = 0.0
            for di in range(1, target_variable.shape[1]):
                decoder_output, decoder_hidden, decoder_attention = self.attention(decoder_input, decoder_hidden, rois)
                loss += self.criterion(decoder_output, target_variable[:, di])
                teach_forcing = True if random.random() > self.teach_prob else False
                if teach_forcing:
                    decoder_input = target_variable[:, di]
                else:
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.squeeze()
                    decoder_input = ni
            return None, loss
        else:
            n = rois.size()[1]
            decodes = torch.zeros((n, self.attention.max_len))
            prob = 1.0
            decoder_input = torch.zeros(n).long()
            decoder_hidden = self.attention.initHidden(n)
            for di in range(self.attention.max_len):
                decoder_output, decoder_hidden, decoder_attention = self.attention(decoder_input, decoder_hidden, rois)
                probs = torch.exp(decoder_output)
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze()
                decoder_input = ni
                prob *= probs[:, ni]
                decodes[:, di] = decoder_input
            return decodes, None


class SeqConvs(nn.Module):

    def __init__(self, conv_dim, roi_size):
        super().__init__()
        height = roi_size[0]
        downsample_level = math.log2(height) - 2
        assert math.isclose(downsample_level, int(downsample_level))
        downsample_level = int(downsample_level)
        conv_block = conv_with_kaiming_uniform(norm='BN', activation=True)
        convs = []
        for i in range(downsample_level):
            convs.append(conv_block(conv_dim, conv_dim, 3, stride=(2, 1)))
        convs.append(nn.Conv2d(conv_dim, conv_dim, kernel_size=(4, 1), bias=False))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


def ctc_loss(preds, targets, voc_size):
    target_lengths = (targets != voc_size).long().sum(dim=-1)
    trimmed_targets = [t[:l] for t, l in zip(targets, target_lengths)]
    targets = torch.cat(trimmed_targets)
    x = F.log_softmax(preds, dim=-1)
    input_lengths = torch.full((x.size(1),), x.size(0), dtype=torch.long)
    return F.ctc_loss(x, targets, input_lengths, target_lengths, blank=voc_size, zero_infinity=True)


def build_recognition_loss_fn(rec_type='ctc'):
    if rec_type == 'ctc':
        return ctc_loss
    else:
        raise NotImplementedError('{} is not a valid recognition loss'.format(rec_type))


class RNNPredictor(nn.Module):

    def __init__(self, cfg):
        super(RNNPredictor, self).__init__()
        self.voc_size = cfg.MODEL.BATEXT.VOC_SIZE
        conv_dim = cfg.MODEL.BATEXT.CONV_DIM
        roi_size = cfg.MODEL.BATEXT.POOLER_RESOLUTION
        self.convs = SeqConvs(conv_dim, roi_size)
        self.rnn = nn.LSTM(conv_dim, conv_dim, num_layers=1, bidirectional=True)
        self.clf = nn.Linear(conv_dim * 2, self.voc_size + 1)
        self.recognition_loss_fn = build_recognition_loss_fn()

    def forward(self, x, targets=None):
        if x.size(0) == 0:
            return x.new_zeros((x.size(2), 0, self.voc_size))
        x = self.convs(x).squeeze(dim=2)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        preds = self.clf(x)
        if self.training:
            rec_loss = self.recognition_loss_fn(preds, targets, self.voc_size)
            return preds, rec_loss
        else:
            _, preds = preds.permute(1, 0, 2).max(dim=-1)
            return preds, None


class Beziers:
    """
    This structure stores a list of bezier curves as a Nx16 torch.Tensor.
    It will support some common methods about bezier shapes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: 'torch.Tensor'):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, 16))
        assert tensor.dim() == 2 and tensor.size(-1) == 16, tensor.size()
        self.tensor = tensor

    def to(self, device: 'str') ->'Beziers':
        return Beziers(self.tensor)

    def __getitem__(self, item: 'Union[int, slice, torch.BoolTensor]') ->'Beziers':
        """
        Returns:
            Beziers: Create a new :class:`Beziers` by indexing.
        """
        if isinstance(item, int):
            return Beziers(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, 'Indexing on Boxes with {} failed to return a matrix!'.format(item)
        return Beziers(b)


def _bezier_height(beziers):
    beziers = beziers.tensor
    p1 = beziers[:, :2]
    p2 = beziers[:, 14:]
    height = ((p1 - p2) ** 2).sum(dim=1).sqrt()
    return height


def _box_max_size(boxes):
    box = boxes.tensor
    max_size = torch.max(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])
    return max_size


def assign_boxes_to_levels_by_metric(box_lists, min_level, max_level, canonical_box_size, canonical_level, metric_fn=_box_max_size):
    """
    Map each box in `box_lists` to a feature map level index and return the assignment
    vector.

    Args:
        box_lists (list[detectron2.structures.Boxes]): A list of N Boxes or N RotatedBoxes,
            where N is the number of images in the batch.
        min_level (int): Smallest feature map level index. The input is considered index 0,
            the output of stage 1 is index 1, and so.
        max_level (int): Largest feature map level index.
        canonical_box_size (int): A canonical box size in pixels (shorter side).
        canonical_level (int): The feature map level index on which a canonically-sized box
            should be placed.

    Returns:
        A tensor of length M, where M is the total number of boxes aggregated over all
            N batch images. The memory layout corresponds to the concatenation of boxes
            from all images. Each element is the feature map index, as an offset from
            `self.min_level`, for the corresponding box (so value i means the box is at
            `self.min_level + i`).
    """
    eps = sys.float_info.epsilon
    box_sizes = cat([metric_fn(boxes) for boxes in box_lists])
    level_assignments = torch.floor(canonical_level + torch.log2(box_sizes / canonical_box_size + eps))
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments - min_level


def assign_boxes_to_levels_bezier(box_lists, min_level, max_level, canonical_box_size, canonical_level):
    return assign_boxes_to_levels_by_metric(box_lists, min_level, max_level, canonical_box_size, canonical_level, metric_fn=_bezier_height)


def assign_boxes_to_levels_max(box_lists, min_level, max_level, canonical_box_size, canonical_level):
    return assign_boxes_to_levels_by_metric(box_lists, min_level, max_level, canonical_box_size, canonical_level, metric_fn=_box_max_size)


class TopPooler(ROIPooler):
    """
    ROIPooler with option to assign level by max length. Used by top modules.
    """

    def __init__(self, output_size, scales, sampling_ratio, pooler_type, canonical_box_size=224, canonical_level=4, assign_crit='area'):
        parent_pooler_type = 'ROIAlign' if pooler_type == 'BezierAlign' else pooler_type
        super().__init__(output_size, scales, sampling_ratio, parent_pooler_type, canonical_box_size=canonical_box_size, canonical_level=canonical_level)
        if parent_pooler_type != pooler_type:
            self.level_poolers = nn.ModuleList(BezierAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio) for scale in scales)
        self.assign_crit = assign_crit

    def forward(self, x, box_lists):
        """
        see 
        """
        num_level_assignments = len(self.level_poolers)
        assert isinstance(x, list) and isinstance(box_lists, list), 'Arguments to pooler must be lists'
        assert len(x) == num_level_assignments, 'unequal value, num_level_assignments={}, but x is list of {} Tensors'.format(num_level_assignments, len(x))
        assert len(box_lists) == x[0].size(0), 'unequal value, x[0] batch dim 0 is {}, but box_list has length {}'.format(x[0].size(0), len(box_lists))
        if isinstance(box_lists[0], torch.Tensor):
            box_lists = [Beziers(x) for x in box_lists]
        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)
        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)
        if self.assign_crit == 'max':
            assign_method = assign_boxes_to_levels_max
        elif self.assign_crit == 'bezier':
            assign_method = assign_boxes_to_levels_bezier
        else:
            assign_method = assign_boxes_to_levels
        level_assignments = assign_method(box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level)
        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size
        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros((num_boxes, num_channels, output_size[0], output_size[1]), dtype=dtype, device=device)
        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = torch.nonzero(level_assignments == level).squeeze(1)
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)
        return output


def build_recognizer(cfg, type):
    if type == 'rnn':
        return RNNPredictor(cfg)
    if type == 'attn':
        return ATTPredictor(cfg)
    else:
        raise NotImplementedError('{} is not a valid recognizer'.format(type))


class TextHead(nn.Module):
    """
    TextHead performs text region alignment and recognition.
    
    It is a simplified ROIHeads, only ground truth RoIs are
    used during training.
    """

    def __init__(self, cfg, input_shape: 'Dict[str, ShapeSpec]'):
        """
        Args:
            in_channels (int): number of channels of the input feature
        """
        super(TextHead, self).__init__()
        pooler_resolution = cfg.MODEL.BATEXT.POOLER_RESOLUTION
        pooler_scales = cfg.MODEL.BATEXT.POOLER_SCALES
        sampling_ratio = cfg.MODEL.BATEXT.SAMPLING_RATIO
        conv_dim = cfg.MODEL.BATEXT.CONV_DIM
        num_conv = cfg.MODEL.BATEXT.NUM_CONV
        canonical_size = cfg.MODEL.BATEXT.CANONICAL_SIZE
        self.in_features = cfg.MODEL.BATEXT.IN_FEATURES
        self.voc_size = cfg.MODEL.BATEXT.VOC_SIZE
        recognizer = cfg.MODEL.BATEXT.RECOGNIZER
        self.top_size = cfg.MODEL.TOP_MODULE.DIM
        self.pooler = TopPooler(output_size=pooler_resolution, scales=pooler_scales, sampling_ratio=sampling_ratio, pooler_type='BezierAlign', canonical_box_size=canonical_size, canonical_level=3, assign_crit='bezier')
        conv_block = conv_with_kaiming_uniform(norm='BN', activation=True)
        tower = []
        for i in range(num_conv):
            tower.append(conv_block(conv_dim, conv_dim, 3, 1))
        self.tower = nn.Sequential(*tower)
        self.recognizer = build_recognizer(cfg, recognizer)

    def forward(self, images, features, proposals, targets=None):
        """
        see detectron2.modeling.ROIHeads
        """
        del images
        features = [features[f] for f in self.in_features]
        if self.training:
            beziers = [p.beziers for p in targets]
            targets = torch.cat([x.text for x in targets], dim=0)
        else:
            beziers = [p.top_feat for p in proposals]
        bezier_features = self.pooler(features, beziers)
        bezier_features = self.tower(bezier_features)
        if self.training:
            preds, rec_loss = self.recognizer(bezier_features, targets)
            rec_loss *= 0.05
            losses = {'rec_loss': rec_loss}
            return None, losses
        else:
            if bezier_features.size(0) == 0:
                for box in proposals:
                    box.beziers = box.top_feat
                    box.recs = box.top_feat
                return proposals, {}
            preds, _ = self.recognizer(bezier_features, targets)
            start_ind = 0
            for proposals_per_im in proposals:
                end_ind = start_ind + len(proposals_per_im)
                proposals_per_im.recs = preds[start_ind:end_ind]
                proposals_per_im.beziers = proposals_per_im.top_feat
                start_ind = end_ind
            return proposals, {}


class ConvBNPReLU(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BNPReLU(nn.Module):

    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class ConvBN(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=0.001)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class Conv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseConv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class DilatedConv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseDilatedConv(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = ConvBNPReLU(nIn, nOut, 3, 2)
        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)
        self.bn = nn.BatchNorm2d(2 * nOut, eps=0.001)
        self.act = nn.PReLU(2 * nOut)
        self.reduce = Conv(2 * nOut, nOut, 1, 1)
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)
        output = self.F_glo(joi_feat)
        return output


class ContextGuidedBlock(nn.Module):

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)
        self.F_loc = ChannelWiseConv(n, n, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate)
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn_prelu(joi_feat)
        output = self.F_glo(joi_feat)
        if self.add:
            output = input + output
        return output


class InputInjection(nn.Module):

    def __init__(self, downsamplingRatio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, downsamplingRatio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)
        return input


class Context_Guided_Network(nn.Module):
    """
    This class defines the proposed Context Guided Network (CGNet) in this work.
    """

    def __init__(self, classes=19, in_channel=1, M=3, N=21, dropout_flag=False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()
        self.level1_0 = ConvBNPReLU(in_channel, 32, 3, 2)
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)
        self.sample1 = InputInjection(1)
        self.sample2 = InputInjection(2)
        self.b1 = BNPReLU(32 + in_channel)
        self.level2_0 = ContextGuidedBlock_Down(32 + in_channel, 64, dilation_rate=2, reduction=8)
        self.level2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.level2.append(ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8))
        self.bn_prelu_2 = BNPReLU(128 + in_channel)
        self.level3_0 = ContextGuidedBlock_Down(128 + in_channel, 128, dilation_rate=4, reduction=16)
        self.level3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.level3.append(ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16))
        self.bn_prelu_3 = BNPReLU(256)
        if dropout_flag:
            None
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False), Conv(256, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(256, classes, 1, 1))
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find('ConvTranspose2d') != -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
        classifier = self.classifier(output2_cat)
        out = F.upsample(classifier, input.size()[2:], mode='bilinear', align_corners=False)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BNPReLU,
     lambda: ([], {'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BidirectionalLSTM,
     lambda: ([], {'nIn': 4, 'nHidden': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CBAM,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelGate,
     lambda: ([], {'gate_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChannelPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelWiseConv,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kSize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ChannelWiseDilatedConv,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kSize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContextGuidedBlock,
     lambda: ([], {'nIn': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContextGuidedBlock_Down,
     lambda: ([], {'nIn': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Context_Guided_Network,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (Conv,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kSize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBN,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kSize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNPReLU,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kSize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DilatedConv,
     lambda: ([], {'nIn': 4, 'nOut': 4, 'kSize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FGlo,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FastRCNNOutputLayers,
     lambda: ([], {'input_size': 4, 'num_classes': 4, 'cls_agnostic_bbox_reg': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (FeatureMapResampler,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GCN,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Hsigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IOULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InputInjection,
     lambda: ([], {'downsamplingRatio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LastLevelMaxPool,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NaiveGroupNorm,
     lambda: ([], {'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SeqConvs,
     lambda: ([], {'conv_dim': 4, 'roi_size': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpatialGate,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (eSEModule,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_gist_ailab_uoais(_paritybench_base):
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

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

