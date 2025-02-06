import sys
_module = sys.modules[__name__]
del sys
faster_fpn_x101_32x4d = _module
faster_fpn_x50_32x4d = _module
fbhit = _module
faster_rcnn_1333x800_r50_fpn_1x = _module
faster_rcnn_640x640_r50_fpn_1x = _module
run_faster_detnas_1G_nasfpn_4conv1fc = _module
run_faster_nasfpn = _module
faster_rcnn_r50_panet_syncbn = _module
faster_rcnn_r50_fpn_voc = _module
train_hit_paper_voc = _module
retinanet_1333x800_r50_fpn_1x = _module
mmcv = _module
arraymisc = _module
quantization = _module
cnn = _module
alexnet = _module
resnet = _module
vgg = _module
weight_init = _module
fileio = _module
handlers = _module
base = _module
json_handler = _module
pickle_handler = _module
yaml_handler = _module
io = _module
parse = _module
image = _module
transforms = _module
colorspace = _module
geometry = _module
normalize = _module
resize = _module
opencv_info = _module
parallel = _module
_functions = _module
collate = _module
data_container = _module
data_parallel = _module
distributed = _module
scatter_gather = _module
runner = _module
checkpoint = _module
dist_utils = _module
hooks = _module
closure = _module
hook = _module
iter_timer = _module
logger = _module
pavi = _module
tensorboard = _module
text = _module
lr_updater = _module
memory = _module
optimizer = _module
sampler_seed = _module
log_buffer = _module
parallel_test = _module
priority = _module
runner = _module
utils = _module
config = _module
misc = _module
path = _module
progressbar = _module
timer = _module
version = _module
video = _module
optflow = _module
optflow_warp = _module
processing = _module
visualization = _module
color = _module
mmdet = _module
apis = _module
env = _module
inference = _module
train = _module
core = _module
anchor = _module
anchor_generator = _module
anchor_target = _module
guided_anchor_target = _module
bbox = _module
assign_sampling = _module
assigners = _module
approx_max_iou_assigner = _module
assign_result = _module
base_assigner = _module
max_iou_assigner = _module
bbox_target = _module
geometry = _module
samplers = _module
base_sampler = _module
combined_sampler = _module
instance_balanced_pos_sampler = _module
iou_balanced_neg_sampler = _module
ohem_sampler = _module
pseudo_sampler = _module
random_sampler = _module
sampling_result = _module
transforms = _module
evaluation = _module
bbox_overlaps = _module
class_names = _module
coco_utils = _module
eval_hooks = _module
mean_ap = _module
recall = _module
fp16 = _module
decorators = _module
hooks = _module
utils = _module
mask = _module
mask_target = _module
post_processing = _module
bbox_nms = _module
merge_augs = _module
dist_utils = _module
datasets = _module
builder = _module
cityscapes = _module
coco = _module
custom = _module
dataset_wrappers = _module
loader = _module
build_loader = _module
sampler = _module
pipelines = _module
compose = _module
formating = _module
loading = _module
test_aug = _module
registry = _module
transforms = _module
utils = _module
voc = _module
wider_face = _module
xml_style = _module
models = _module
anchor_heads = _module
anchor_head = _module
fcos_head = _module
ga_retina_head = _module
ga_rpn_head = _module
guided_anchor_head = _module
retina_head = _module
rpn_head = _module
ssd_head = _module
backbones = _module
detnas = _module
dropblock = _module
fbnet = _module
fbnet_arch = _module
fbnet_blocks = _module
hrnet = _module
mnasnet = _module
mobilenetv2 = _module
resnet = _module
resnext = _module
ssd_vgg = _module
utils = _module
bbox_heads = _module
auto_head = _module
build_head = _module
mbblock_head_search = _module
mbblock_ops = _module
bbox_head = _module
convfc_bbox_head = _module
double_bbox_head = _module
builder = _module
detectors = _module
base = _module
cascade_rcnn = _module
double_head_rcnn = _module
fast_rcnn = _module
faster_rcnn = _module
fcos = _module
grid_rcnn = _module
htc = _module
mask_rcnn = _module
mask_scoring_rcnn = _module
retinanet = _module
rpn = _module
single_stage = _module
test_mixins = _module
two_stage = _module
losses = _module
accuracy = _module
balanced_l1_loss = _module
cross_entropy_loss = _module
focal_loss = _module
ghm_loss = _module
iou_loss = _module
mse_loss = _module
smooth_l1_loss = _module
utils = _module
mask_heads = _module
fcn_mask_head = _module
fused_semantic_head = _module
grid_head = _module
htc_mask_head = _module
maskiou_head = _module
necks = _module
auto_neck = _module
build_neck = _module
hit_neck_search = _module
hit_ops = _module
bfp = _module
fpn = _module
fpn_panet = _module
hrfpn = _module
nas_fpn = _module
search_pafpn = _module
plugins = _module
generalized_attention = _module
non_local = _module
roi_extractors = _module
single_level = _module
shared_heads = _module
res_layer = _module
conv_module = _module
conv_ws = _module
norm = _module
quant_conv = _module
scale = _module
weight_init = _module
ops = _module
dcn = _module
functions = _module
deform_conv = _module
deform_pool = _module
modules = _module
deform_conv = _module
deform_pool = _module
setup = _module
gcb = _module
context_block = _module
masked_conv = _module
masked_conv = _module
masked_conv = _module
setup = _module
nms = _module
nms_wrapper = _module
setup = _module
roi_align = _module
roi_align = _module
gradcheck = _module
roi_align = _module
roi_align = _module
setup = _module
roi_pool = _module
roi_pool = _module
gradcheck = _module
roi_pool = _module
setup = _module
sigmoid_focal_loss = _module
sigmoid_focal_loss = _module
sigmoid_focal_loss = _module
setup = _module
collect_env = _module
contextmanagers = _module
flops_counter = _module
profiling = _module
registry = _module
util_mixins = _module
test = _module
analyze_logs = _module
coco_eval = _module
pascal_voc = _module
detectron2pytorch = _module
get_flops = _module
publish_model = _module
test = _module
upgrade_model_version = _module
voc_eval = _module
train = _module

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


import logging


import torch.nn as nn


import torch.utils.checkpoint as cp


import torch


from torch.nn.parallel._functions import _get_stream


import collections


import torch.nn.functional as F


from torch.utils.data.dataloader import default_collate


import functools


from torch.nn.parallel import DataParallel


import torch.distributed as dist


from torch._utils import _flatten_dense_tensors


from torch._utils import _take_tensors


from torch._utils import _unflatten_dense_tensors


from torch.nn.parallel._functions import Scatter as OrigScatter


import time


import warnings


from collections import OrderedDict


import torchvision


from torch.utils import model_zoo


import torch.multiprocessing as mp


import numpy as np


from torch.nn.utils import clip_grad


import math


import random


import matplotlib.pyplot as plt


import re


from abc import ABCMeta


from abc import abstractmethod


from torch.utils.data import Dataset


from inspect import getfullargspec


import copy


from collections import abc


from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


from functools import partial


import torch.utils.data.sampler as _sampler


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler as _DistributedSampler


from torch.utils.data import Sampler


from collections.abc import Sequence


from torch.nn.modules.batchnorm import _BatchNorm


from torch import nn


from torch.autograd import Variable


from torch.nn import functional as F


from torch.nn.modules.utils import _pair


from torch.utils.checkpoint import checkpoint


from torch.autograd import Function


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd import gradcheck


from torch.nn.modules.module import Module


from torch.autograd.function import once_differentiable


from collections import defaultdict


from typing import List


from torch.nn.modules.conv import _ConvNd


from torch.nn.modules.conv import _ConvTransposeMixin


from torch.nn.modules.pooling import _AdaptiveAvgPoolNd


from torch.nn.modules.pooling import _AdaptiveMaxPoolNd


from torch.nn.modules.pooling import _AvgPoolNd


from torch.nn.modules.pooling import _MaxPoolNd


import inspect


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.
    Args:
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    own_state = module.state_dict()
    state_dict_modify = state_dict.copy()
    for name, param in state_dict.items():
        """ for mobilenet v2
        if 'features' in name:
            name = name.replace('features.','features')
        """
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if 'conv2' in name and 'layer4.0.conv2_d2.weight' in own_state.keys():
            d1 = name.replace('conv2', 'conv2_d1')
            d1_c = own_state[d1].size(0)
            own_state[d1].copy_(param[:d1_c, :, :, :])
            state_dict_modify[d1] = param[:d1_c, :, :, :]
            d2 = name.replace('conv2', 'conv2_d2')
            d2_c = own_state[d2].size(0)
            own_state[d2].copy_(param[d1_c:d1_c + d2_c, :, :, :])
            state_dict_modify[d2] = param[d1_c:d1_c + d2_c, :, :, :]
            d3 = name.replace('conv2', 'conv2_d3')
            own_state[d3].copy_(param[d1_c + d2_c:, :, :, :])
            state_dict_modify[d3] = param[d1_c + d2_c:, :, :, :]
        else:
            if name not in own_state:
                unexpected_keys.append(name)
                continue
            try:
                own_state[name].copy_(param)
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
    missing_keys = set(own_state.keys()) - set(state_dict_modify.keys())
    """
    if 'layer4.0.conv2_d2.weight' in own_state.keys():
        missing_keys = set(own_state.keys()) - set(state_dict_modify.keys())
    else:
        # for mobilenetv2
        own_state_set = []
        for name in set(own_state.keys()):
            own_state_set.append(name.replace('features','features.'))
        missing_keys = set(own_state_set) - set(state_dict.keys())
    """
    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warn(err_msg)
        else:
            None


def load_checkpoint(model, filename, strict=False, logger=None):
    checkpoint = torch.load(filename)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError('No state_dict found in checkpoint file {}'.format(filename))
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class AlexNet(nn.Module):
    """AlexNet backbone.

    Args:
        num_classes (int): number of classes for classification.
    """

    def __init__(self, num_classes=-1):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2))
        if self.num_classes > 0:
            self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, num_classes))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.features(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)
        return x


def conv_ws_2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, eps=1e-05):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, eps=1e-05):
        super(ConvWS2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.eps)


class BirealActivation(Function):
    """
    take a real value x
    output sign(x)
    """

    @staticmethod
    def forward(ctx, input, nbit_a=1):
        ctx.save_for_backward(input)
        return input.clamp(-1, 1).sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = (2 + 2 * input) * input.lt(0).float() + (2 - 2 * input) * input.ge(0).float()
        grad_input = torch.clamp(grad_input, 0)
        grad_input *= grad_output
        return grad_input, None


def bireal_a(input, nbit_a=1, *args, **kwargs):
    return BirealActivation.apply(input)


class Signer(Function):
    """
    take a real value x
    output sign(x)
    """

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def sign(input):
    return Signer.apply(input)


def bireal_w(w, nbit_w=1, *args, **kwargs):
    if nbit_w != 1:
        raise ValueError('nbit_w must be 1 in Bi-Real-Net.')
    return sign(w) * torch.mean(torch.abs(w.clone().detach()))


class Quantizer(Function):
    """
    take a real value x in alpha*[0,1] or alpha*[-1,1]
    output a discrete-valued x in alpha*{0, 1/(2^k-1), ..., (2^k-1)/(2^k-1)} or likeness
    where k is nbit
    """

    @staticmethod
    def forward(ctx, input, nbit, alpha=None, offset=None):
        ctx.alpha = alpha
        ctx.offset = offset
        scale = 2 ** nbit - 1 if alpha is None else (2 ** nbit - 1) / alpha
        ctx.scale = scale
        return torch.round(input * scale) / scale if offset is None else (torch.round(input * scale) + torch.round(offset)) / scale

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.offset is None:
            return grad_output, None, None, None
        else:
            return grad_output, None, None, torch.sum(grad_output) / ctx.scale


def quantize(input, nbit, alpha=None, offset=None):
    return Quantizer.apply(input, nbit, alpha, offset)


def dorefa_a(input, nbit_a, *args, **kwargs):
    return quantize(torch.clamp(input, 0, 1.0), nbit_a, *args, **kwargs)


class ScaleSigner(Function):
    """
    take a real value x
    output sign(x) * E(|x|)
    """

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def scale_sign(input):
    return ScaleSigner.apply(input)


def dorefa_w(w, nbit_w, *args, **kwargs):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        w = w / (2 * torch.max(torch.abs(w))) + 0.5
        w = 2 * quantize(w, nbit_w) - 1
    return w


def pact_a(input, nbit_a, alpha, *args, **kwargs):
    x = 0.5 * (torch.abs(input) - torch.abs(input - alpha) + alpha)
    return quantize(x, nbit_a, alpha, *args, **kwargs)


def wrpn_w(w, nbit_w, *args, **kwargs):
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = quantize(torch.clamp(w, -1, 1), nbit_w - 1)
    return w


class Xnor(Function):
    """
    take a real value x
    output sign(x_c) * E(|x_c|)
    """

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input) * torch.mean(torch.abs(input), dim=[1, 2, 3], keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def xnor(input):
    return Xnor.apply(input)


def xnor_w(w, nbit_w=1, *args, **kwargs):
    if nbit_w != 1:
        raise ValueError('nbit_w must be 1 in XNOR-Net.')
    return xnor(w)


class QuantConv(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QuantConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_custome_parameters()
        self.quant_config()

    def quant_config(self, quant_name_w='dorefa', quant_name_a='dorefa', nbit_w=1, nbit_a=1, has_offset=False):
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w, 'pact': dorefa_w, 'wrpn': wrpn_w, 'xnor': xnor_w, 'bireal': bireal_w}
        name_a_dict = {'dorefa': dorefa_a, 'pact': pact_a, 'wrpn': dorefa_a, 'xnor': dorefa_a, 'bireal': bireal_a}
        self.quant_w = name_w_dict[quant_name_w]
        self.quant_a = name_a_dict[quant_name_a]
        if quant_name_a == 'pact':
            self.alpha_a = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_a', None)
        if quant_name_w == 'pact':
            self.alpha_w = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_w', None)
        if has_offset:
            self.offset = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('offset', None)
        if self.alpha_a is not None:
            nn.init.constant_(self.alpha_a, 10)
        if self.alpha_w is not None:
            nn.init.constant_(self.alpha_w, 10)
        if self.offset is not None:
            nn.init.constant_(self.offset, 0)

    def reset_custome_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        if self.nbit_w == 0 or self.nbit_a == 0:
            diff_channels = self.out_channels - self.in_channels
            if self.stride == 2 or self.stride == (2, 2):
                x = F.pad(input[:, :, ::2, ::2], (0, 0, 0, 0, diff_channels // 2, diff_channels - diff_channels // 2), 'constant', 0)
                return x
            else:
                x = F.pad(input, (0, 0, 0, 0, diff_channels // 2, diff_channels - diff_channels // 2), 'constant', 0)
                return x
        if self.nbit_w < 32:
            w = self.quant_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        else:
            w = self.weight
        if self.nbit_a < 32:
            x = self.quant_a(input, self.nbit_a, self.alpha_a)
        else:
            x = F.relu(input)
        x = F.conv2d(x, w, None, self.stride, self.padding, self.dilation, self.groups)
        return x


conv_cfg = {'Conv': nn.Conv2d, 'ConvWS': ConvWS2d, 'QuantConv': QuantConv}


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer

    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.

    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]
    layer = conv_layer(*args, **kwargs, **cfg_)
    return layer


norm_cfg = {'BN': ('bn', nn.BatchNorm2d), 'SyncBN': ('bn', nn.SyncBatchNorm), 'GN': ('gn', nn.GroupNorm)}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-05)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=3, dilation=1, downsample=None, style='pytorch', with_cp=False, conv2_split=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, gcb=None, gen_attention=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert gen_attention is None, 'Not implemented yet.'
        assert gcb is None, 'Not implemented yet.'
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.conv1 = build_conv_layer(conv_cfg, inplanes, planes, 3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, weight, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError('Expected 4D tensor as input, got {}D tensor instead.'.format(input.dim()))
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        ctx.save_for_backward(input, offset, weight)
        output = input.new_empty(DeformConvFunction._output_size(input, weight, ctx.padding, ctx.dilation, ctx.stride))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]
        if not input.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            deform_conv_cuda.deform_conv_forward_cuda(input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors
        grad_input = grad_offset = grad_weight = None
        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            cur_im2col_step = min(ctx.im2col_step, input.shape[0])
            assert input.shape[0] % cur_im2col_step == 0, 'im2col step must divide batchsize'
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                grad_input = torch.zeros_like(input)
                grad_offset = torch.zeros_like(offset)
                deform_conv_cuda.deform_conv_backward_input_cuda(input, offset, grad_output, grad_input, grad_offset, weight, ctx.bufs_[0], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, cur_im2col_step)
            if ctx.needs_input_grad[2]:
                grad_weight = torch.zeros_like(weight)
                deform_conv_cuda.deform_conv_backward_parameters_cuda(input, offset, grad_output, grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0], ctx.padding[1], ctx.padding[0], ctx.dilation[1], ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1, cur_im2col_step)
        return grad_input, grad_offset, grad_weight, None, None, None, None, None

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = input.size(0), channels
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError('convolution input is too small (output would be {})'.format('x'.join(map(str, output_size))))
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        assert not bias
        assert in_channels % groups == 0, 'in_channels {} cannot be divisible by groups {}'.format(in_channels, groups)
        assert out_channels % groups == 0, 'out_channels {} cannot be divisible by groups {}'.format(out_channels, groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        return deform_conv(x, offset, self.weight, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class ModulatedDeformConvFunction(Function):

    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(1)
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(ModulatedDeformConvFunction._infer_shape(ctx, input, weight))
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        deform_conv_cuda.modulated_deform_conv_cuda_forward(input, weight, bias, ctx._bufs[0], offset, mask, output, ctx._bufs[1], weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        deform_conv_cuda.modulated_deform_conv_cuda_backward(input, weight, bias, ctx._bufs[0], offset, mask, ctx._bufs[1], grad_input, grad_weight, grad_bias, grad_offset, grad_mask, grad_output, weight.shape[2], weight.shape[3], ctx.stride, ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation, ctx.groups, ctx.deformable_groups, ctx.with_bias)
        if not ctx.with_bias:
            grad_bias = None
        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, None, None, None, None, None

    @staticmethod
    def _infer_shape(ctx, input, weight):
        n = input.size(0)
        channels_out = weight.size(0)
        height, width = input.shape[2:4]
        kernel_h, kernel_w = weight.shape[2:4]
        height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) + 1)) // ctx.stride + 1
        width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)) // ctx.stride + 1
        return n, channels_out, height_out, width_out


modulated_deform_conv = ModulatedDeformConvFunction.apply


class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(ModulatedDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x, offset, mask):
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def make_res_layer(block, inplanes, planes, blocks, stride=1, dilation=1, groups=1, base_width=4, style='pytorch', with_cp=False, conv_cfg=None, norm_cfg=dict(type='BN'), dcn=None, gcb=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(build_conv_layer(conv_cfg, inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), build_norm_layer(norm_cfg, planes * block.expansion)[1])
    layers = []
    layers.append(block(inplanes=inplanes, planes=planes, stride=stride, dilation=dilation, downsample=downsample, groups=groups, base_width=base_width, style=style, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, gcb=gcb))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes=inplanes, planes=planes, stride=1, dilation=dilation, groups=groups, base_width=base_width, style=style, with_cp=with_cp, conv_cfg=conv_cfg, norm_cfg=norm_cfg, dcn=dcn, gcb=gcb))
    return nn.Sequential(*layers)


def conv3x3(in_planes, out_planes, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation)


def make_vgg_layer(inplanes, planes, num_blocks, dilation=1, with_bn=False, ceil_mode=False):
    layers = []
    for _ in range(num_blocks):
        layers.append(conv3x3(inplanes, planes, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))
        inplanes = planes
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))
    return layers


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


class VGG(nn.Module):
    """VGG backbone.

    Args:
        depth (int): Depth of vgg, from {11, 13, 16, 19}.
        with_bn (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
    """
    arch_settings = {(11): (1, 1, 2, 2, 2), (13): (2, 2, 2, 2, 2), (16): (2, 2, 3, 3, 3), (19): (2, 2, 4, 4, 4)}

    def __init__(self, depth, with_bn=False, num_classes=-1, num_stages=5, dilations=(1, 1, 1, 1, 1), out_indices=(0, 1, 2, 3, 4), frozen_stages=-1, bn_eval=True, bn_frozen=False, ceil_mode=False, with_last_pool=True):
        super(VGG, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for vgg'.format(depth))
        assert num_stages >= 1 and num_stages <= 5
        stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        assert len(dilations) == num_stages
        assert max(out_indices) <= num_stages
        self.num_classes = num_classes
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.inplanes = 3
        start_idx = 0
        vgg_layers = []
        self.range_sub_modules = []
        for i, num_blocks in enumerate(self.stage_blocks):
            num_modules = num_blocks * (2 + with_bn) + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            planes = 64 * 2 ** i if i < 4 else 512
            vgg_layer = make_vgg_layer(self.inplanes, planes, num_blocks, dilation=dilation, with_bn=with_bn, ceil_mode=ceil_mode)
            vgg_layers.extend(vgg_layer)
            self.inplanes = planes
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx
        if not with_last_pool:
            vgg_layers.pop(-1)
            self.range_sub_modules[-1][1] -= 1
        self.module_name = 'features'
        self.add_module(self.module_name, nn.Sequential(*vgg_layers))
        if self.num_classes > 0:
            self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        outs = []
        vgg_layers = getattr(self, self.module_name)
        for i, num_blocks in enumerate(self.stage_blocks):
            for j in range(*self.range_sub_modules[i]):
                vgg_layer = vgg_layers[j]
                x = vgg_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            outs.append(x)
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def train(self, mode=True):
        super(VGG, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.bn_frozen:
                        for params in m.parameters():
                            params.requires_grad = False
        vgg_layers = getattr(self, self.module_name)
        if mode and self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                for j in range(*self.range_sub_modules[i]):
                    mod = vgg_layers[j]
                    mod.eval()
                    for param in mod.parameters():
                        param.requires_grad = False


def assert_tensor_type(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError('{} has no attribute {} for type {}'.format(args[0].__class__.__name__, func.__name__, args[0].datatype))
        return func(*args, **kwargs)
    return wrapper


class DataContainer(object):
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self, data, stack=False, padding_value=0, cpu_only=False, pad_dims=2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self.data))

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self._cpu_only

    @property
    def stack(self):
        return self._stack

    @property
    def padding_value(self):
        return self._padding_value

    @property
    def pad_dims(self):
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self):
        return self.data.dim()


def get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception('Unknown type {}.'.format(type(input)))

