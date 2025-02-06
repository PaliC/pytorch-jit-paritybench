import sys
_module = sys.modules[__name__]
del sys
DiRA_models = _module
data_loader = _module
main_DiRA_moco = _module
segmentation_models_pytorch = _module
__version__ = _module
base = _module
heads = _module
initialization = _module
model = _module
modules = _module
deeplabv3 = _module
decoder = _module
model = _module
encoders = _module
_base = _module
_preprocessing = _module
_utils = _module
densenet = _module
dpn = _module
efficientnet = _module
inceptionresnetv2 = _module
inceptionv4 = _module
mobilenet = _module
resnet = _module
senet = _module
timm_efficientnet = _module
timm_regnet = _module
timm_res2net = _module
timm_resnest = _module
timm_sknet = _module
vgg = _module
xception = _module
fpn = _module
decoder = _module
model = _module
linknet = _module
decoder = _module
model = _module
losses = _module
_functional = _module
constants = _module
dice = _module
focal = _module
jaccard = _module
lovasz = _module
soft_bce = _module
soft_ce = _module
manet = _module
decoder = _module
model = _module
pan = _module
decoder = _module
model = _module
pspnet = _module
decoder = _module
model = _module
transskipunet = _module
decoder = _module
model = _module
transformer = _module
unet = _module
decoder = _module
model = _module
unetplusplus = _module
decoder = _module
model = _module
utils = _module
base = _module
functional = _module
losses = _module
meter = _module
metrics = _module
train = _module
trainer = _module
transformation = _module

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


import torch.nn as nn


import random


import copy


import numpy as np


from scipy import ndimage


from torch.utils.data import Dataset


import math


import time


import warnings


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


from torch.autograd import Variable


from typing import Optional


from torch import nn


from torch.nn import functional as F


import functools


import torch.utils.model_zoo as model_zoo


from typing import List


from collections import OrderedDict


import re


from torchvision.models.densenet import DenseNet


import torch.nn.functional as F


import torchvision


from copy import deepcopy


from torchvision.models.resnet import ResNet


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import Bottleneck


from torchvision.models.vgg import VGG


from torchvision.models.vgg import make_layers


from typing import Union


from torch.nn.modules.loss import _Loss


from functools import partial


from torch import Tensor


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all)
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        self._dequeue_and_enqueue(k)
        return logits, labels


class DiRA_MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, backbone='resnet50', encoder_weights=None, activation=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(DiRA_MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.encoder_q = base_encoder(backbone, encoder_weights, activation, num_classes=dim)
        self.encoder_k = base_encoder(backbone, encoder_weights, activation, num_classes=dim)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all)
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        q, rec_output = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k, _ = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        self._dequeue_and_enqueue(k)
        return logits, labels, rec_output


class DiRA_UNet(nn.Module):

    def __init__(self, backbone, encoder_weights=None, activation=None, num_classes=1000):
        super(DiRA_UNet, self).__init__()
        self.backbone = smp.Unet(backbone, encoder_weights=encoder_weights, activation=activation)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        latent = 2048
        self.fc = nn.Linear(latent, num_classes)

    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_output = self.backbone.decoder(*features)
        masks = self.backbone.segmentation_head(decoder_output)
        f = self.avgpool(features[-1])
        f = torch.flatten(f, 1)
        f = self.fc(f)
        return f, masks


class Discriminator(nn.Module):

    def __init__(self, channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):

    def __init__(self, name, **params):
        super().__init__()
        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling='avg', dropout=0.2, activation=None):
        if pooling not in ('max', 'avg'):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x)
        return x


class Conv2dReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        if use_batchnorm == 'inplace' and InPlaceABN is None:
            raise RuntimeError("In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. " + 'To install see: https://github.com/mapillary/inplace_abn')
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        if use_batchnorm == 'inplace':
            bn = InPlaceABN(out_channels, activation='leaky_relu', activation_param=0.0)
            relu = nn.Identity()
        elif use_batchnorm and use_batchnorm != 'inplace':
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels // reduction, 1), nn.ReLU(inplace=True), nn.Conv2d(in_channels // reduction, in_channels, 1), nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError('Attention {} is not implemented'.format(name))

    def forward(self, x):
        return self.attention(x)


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super().__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class SeparableConv2d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        dephtwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        super().__init__(dephtwise_conv, pointwise_conv)


class ASPPSeparableConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(SeparableConv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())


class ASPP(nn.Module):

    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv
        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Decoder(nn.Sequential):

    def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
        super().__init__(ASPP(in_channels, out_channels, atrous_rates), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.out_channels = out_channels

    def forward(self, *features):
        return super().forward(features[-1])


class DeepLabV3PlusDecoder(nn.Module):

    def __init__(self, encoder_channels, out_channels=256, atrous_rates=(12, 24, 36), output_stride=16):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError('Output stride should be 8 or 16, got {}.'.format(output_stride))
        self.out_channels = out_channels
        self.output_stride = output_stride
        self.aspp = nn.Sequential(ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True), SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48
        self.block1 = nn.Sequential(nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(highres_out_channels), nn.ReLU())
        self.block2 = nn.Sequential(SeparableConv2d(highres_out_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features


encoders = {}


def get_encoder(name, in_channels=3, depth=5, weights=None):
    try:
        Encoder = encoders[name]['encoder']
    except KeyError:
        raise KeyError('Wrong encoder name `{}`, supported encoders: {}'.format(name, list(encoders.keys())))
    params = encoders[name]['params']
    params.update(depth=depth)
    encoder = Encoder(**params)
    if weights is not None:
        if weights.lower() != 'imagenet':
            checkpoint = torch.load(weights, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state_dict = {k.replace('encoder_q.', ''): v for k, v in state_dict.items()}
            for k in list(state_dict.keys()):
                if k.startswith('fc') or k.startswith('classifier') or k.startswith('projection_head') or k.startswith('prototypes') or k.startswith('encoder_k') or k.startswith('queue'):
                    del state_dict[k]
            encoder.load_state_dict(state_dict)
            None
        else:
            try:
                settings = encoders[name]['pretrained_settings'][weights.lower()]
            except KeyError:
                raise KeyError('Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}'.format(weights, name, list(encoders[name]['pretrained_settings'].keys())))
            None
            if settings['url'].startswith('http'):
                encoder.load_state_dict(model_zoo.load_url(settings['url']))
            else:
                encoder.load_state_dict(torch.load(settings['url'], map_location='cpu'))
            None
    encoder.set_in_channels(in_channels)
    return encoder


class DeepLabV3(SegmentationModel):
    """DeepLabV3_ implementation from "Rethinking Atrous Convolution for Semantic Image Segmentation"

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3**

    .. _DeeplabV3:
        https://arxiv.org/abs/1706.05587

    """

    def __init__(self, encoder_name: 'str'='resnet34', encoder_depth: 'int'=5, encoder_weights: 'Optional[str]'='imagenet', decoder_channels: 'int'=256, in_channels: 'int'=3, classes: 'int'=1, activation: 'Optional[str]'=None, upsampling: 'int'=8, aux_params: 'Optional[dict]'=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.encoder.make_dilated(stage_list=[4, 5], dilation_list=[2, 4])
        self.decoder = DeepLabV3Decoder(in_channels=self.encoder.out_channels[-1], out_channels=decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=self.decoder.out_channels, out_channels=classes, activation=activation, kernel_size=1, upsampling=upsampling)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None


class DeepLabV3Plus(SegmentationModel):
    """DeepLabV3+ implementation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"
    
    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**
    
    Reference:
        https://arxiv.org/abs/1802.02611v3

    """

    def __init__(self, encoder_name: 'str'='resnet34', encoder_depth: 'int'=5, encoder_weights: 'Optional[str]'='imagenet', encoder_output_stride: 'int'=16, decoder_channels: 'int'=256, decoder_atrous_rates: 'tuple'=(12, 24, 36), in_channels: 'int'=3, classes: 'int'=1, activation: 'Optional[str]'=None, upsampling: 'int'=4, aux_params: 'Optional[dict]'=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        if encoder_output_stride == 8:
            self.encoder.make_dilated(stage_list=[4, 5], dilation_list=[2, 4])
        elif encoder_output_stride == 16:
            self.encoder.make_dilated(stage_list=[5], dilation_list=[2])
        else:
            raise ValueError('Encoder output stride should be 8 or 16, got {}'.format(encoder_output_stride))
        self.decoder = DeepLabV3PlusDecoder(encoder_channels=self.encoder.out_channels, out_channels=decoder_channels, atrous_rates=decoder_atrous_rates, output_stride=encoder_output_stride)
        self.segmentation_head = SegmentationHead(in_channels=self.decoder.out_channels, out_channels=classes, activation=activation, kernel_size=1, upsampling=upsampling)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None


class TransitionWithSkip(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        for module in self.module:
            x = module(x)
            if isinstance(module, nn.ReLU):
                skip = x
        return x, skip


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

    def forward(self, x, skip=None):
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


class MergeBlock(nn.Module):

    def __init__(self, policy):
        super().__init__()
        if policy not in ['add', 'cat']:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == 'add':
            return sum(x)
        elif self.policy == 'cat':
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))


class FPNDecoder(nn.Module):

    def __init__(self, encoder_channels, encoder_depth=5, pyramid_channels=256, segmentation_channels=128, dropout=0.2, merge_policy='add'):
        super().__init__()
        self.out_channels = segmentation_channels if merge_policy == 'add' else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError('Encoder depth for FPN decoder cannot be less than 3, got {}.'.format(encoder_depth))
        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[:encoder_depth + 1]
        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])
        self.seg_blocks = nn.ModuleList([SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples) for n_upsamples in [3, 2, 1, 0]])
        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]
        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)
        return x


class FPN(SegmentationModel):
    """FPN_ is a fully convolution neural network for image semantic segmentation.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_pyramid_channels: A number of convolution filters in Feature Pyramid of FPN_
        decoder_segmentation_channels: A number of convolution filters in segmentation blocks of FPN_
        decoder_merge_policy: Determines how to merge pyramid features inside FPN. Available options are **add** and **cat**
        decoder_dropout: Spatial dropout rate in range (0, 1) for feature pyramid in FPN_
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    def __init__(self, encoder_name: 'str'='resnet34', encoder_depth: 'int'=5, encoder_weights: 'Optional[str]'='imagenet', decoder_pyramid_channels: 'int'=256, decoder_segmentation_channels: 'int'=128, decoder_merge_policy: 'str'='add', decoder_dropout: 'float'=0.2, in_channels: 'int'=3, classes: 'int'=1, activation: 'Optional[str]'=None, upsampling: 'int'=4, aux_params: 'Optional[dict]'=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.decoder = FPNDecoder(encoder_channels=self.encoder.out_channels, encoder_depth=encoder_depth, pyramid_channels=decoder_pyramid_channels, segmentation_channels=decoder_segmentation_channels, dropout=decoder_dropout, merge_policy=decoder_merge_policy)
        self.segmentation_head = SegmentationHead(in_channels=self.decoder.out_channels, out_channels=classes, activation=activation, kernel_size=1, upsampling=upsampling)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'fpn-{}'.format(encoder_name)
        self.initialize()


class TransposeX2(nn.Sequential):

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True)]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        super().__init__(*layers)


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        self.conv1 = md.Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class LinknetDecoder(nn.Module):

    def __init__(self, encoder_channels, prefinal_channels=32, n_blocks=5, use_batchnorm=True):
        super().__init__()
        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        channels = list(encoder_channels) + [prefinal_channels]
        self.blocks = nn.ModuleList([DecoderBlock(channels[i], channels[i + 1], use_batchnorm=use_batchnorm) for i in range(n_blocks)])

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]
        x = features[0]
        skips = features[1:]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


class Linknet(SegmentationModel):
    """Linknet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *sum*
    for fusing decoder blocks with skip connections.

    Note:
        This implementation by default has 4 skip connections (original - 3).

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **Linknet**

    .. _Linknet:
        https://arxiv.org/abs/1707.03718
    """

    def __init__(self, encoder_name: 'str'='resnet34', encoder_depth: 'int'=5, encoder_weights: 'Optional[str]'='imagenet', decoder_use_batchnorm: 'bool'=True, in_channels: 'int'=3, classes: 'int'=1, activation: 'Optional[Union[str, callable]]'=None, aux_params: 'Optional[dict]'=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.decoder = LinknetDecoder(encoder_channels=self.encoder.out_channels, n_blocks=encoder_depth, prefinal_channels=32, use_batchnorm=decoder_use_batchnorm)
        self.segmentation_head = SegmentationHead(in_channels=32, out_channels=classes, activation=activation, kernel_size=1)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'link-{}'.format(encoder_name)
        self.initialize()


def focal_loss_with_logits(output: 'torch.Tensor', target: 'torch.Tensor', gamma: 'float'=2.0, alpha: 'Optional[float]'=0.25, reduction: 'str'='mean', normalized: 'bool'=False, reduced_threshold: 'Optional[float]'=None, eps: 'float'=1e-06) ->torch.Tensor:
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(output.type())
    logpt = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    pt = torch.exp(-logpt)
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1
    loss = focal_term * logpt
    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)
    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor
    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)
    return loss


class FocalLoss(_Loss):

    def __init__(self, mode: 'str', alpha: 'Optional[float]'=None, gamma: 'Optional[float]'=2.0, ignore_index: 'Optional[int]'=None, reduction: 'Optional[str]'='mean', normalized: 'bool'=False, reduced_threshold: 'Optional[float]'=None):
        """Compute Focal loss

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you should use `reduction="sum"`.
        
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(focal_loss_with_logits, alpha=alpha, gamma=gamma, reduced_threshold=reduced_threshold, reduction=reduction, normalized=normalized)

    def forward(self, y_pred: 'torch.Tensor', y_true: 'torch.Tensor') ->torch.Tensor:
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]
            loss = self.focal_loss_fn(y_pred, y_true)
        elif self.mode == MULTICLASS_MODE:
            num_classes = y_pred.size(1)
            loss = 0
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index
            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]
                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]
                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)
        return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Variable, logits at each prediction (between -infinity and +infinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
    """
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nanmean compatible with generators.
    """
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def _lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Variable, logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(_lovasz_hinge_flat(*_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)) for log, lab in zip(logits, labels))
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch
    """
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    C = probas.size(1)
    probas = torch.movedim(probas, 0, -1)
    probas = probas.contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


def _lovasz_softmax_flat(probas, labels, classes='present'):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).type_as(probas)
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore: void class labels
    """
    if per_image:
        loss = mean(_lovasz_softmax_flat(*_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes) for prob, lab in zip(probas, labels))
    else:
        loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore), classes=classes)
    return loss


class LovaszLoss(_Loss):

    def __init__(self, mode: 'str', per_image: 'bool'=False, ignore_index: 'Optional[int]'=None, from_logits: 'bool'=True):
        """Implementation of Lovasz loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            per_image: If True loss computed per each image and then averaged, else computed per whole batch
        
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.per_image = per_image

    def forward(self, y_pred, y_true):
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            loss = _lovasz_hinge(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        elif self.mode == MULTICLASS_MODE:
            y_pred = y_pred.softmax(dim=1)
            loss = _lovasz_softmax(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        else:
            raise ValueError('Wrong mode {}.'.format(self.mode))
        return loss


class SoftBCEWithLogitsLoss(nn.Module):
    __constants__ = ['weight', 'pos_weight', 'reduction', 'ignore_index', 'smooth_factor']

    def __init__(self, weight: 'Optional[torch.Tensor]'=None, ignore_index: 'Optional[int]'=-100, reduction: 'str'='mean', smooth_factor: 'Optional[float]'=None, pos_weight: 'Optional[torch.Tensor]'=None):
        """Drop-in replacement for torch.nn.BCEWithLogitsLoss with few additions: ignore_index and label_smoothing
        
        Args:
            ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient. 
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 1] -> [0.9, 0.1, 0.9])
        
        Shape
             - **y_pred** - torch.Tensor of shape NxCxHxW
             - **y_true** - torch.Tensor of shape NxHxW or Nx1xHxW

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, y_pred: 'torch.Tensor', y_true: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)
        
        Returns:
            loss: torch.Tensor
        """
        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (1 - self.smooth_factor)
        else:
            soft_targets = y_true
        loss = F.binary_cross_entropy_with_logits(y_pred, soft_targets, self.weight, pos_weight=self.pos_weight, reduction='none')
        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def label_smoothed_nll_loss(lprobs: 'torch.Tensor', target: 'torch.Tensor', epsilon: 'float', ignore_index=None, reduction='mean', dim=-1) ->torch.Tensor:
    """
    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py
    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)
        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)
    if reduction == 'sum':
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == 'mean':
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss


class SoftCrossEntropyLoss(nn.Module):
    __constants__ = ['reduction', 'ignore_index', 'smooth_factor']

    def __init__(self, reduction: 'str'='mean', smooth_factor: 'Optional[float]'=None, ignore_index: 'Optional[int]'=-100, dim: 'int'=1):
        """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing
        
        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])
        
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, y_pred: 'torch.Tensor', y_true: 'torch.Tensor') ->torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return label_smoothed_nll_loss(log_prob, y_true, epsilon=self.smooth_factor, ignore_index=self.ignore_index, reduction=self.reduction, dim=self.dim)


class PAB(nn.Module):

    def __init__(self, in_channels, out_channels, pab_channels=64):
        super(PAB, self).__init__()
        self.pab_channels = pab_channels
        self.in_channels = in_channels
        self.top_conv = nn.Conv2d(in_channels, pab_channels, kernel_size=1)
        self.center_conv = nn.Conv2d(in_channels, pab_channels, kernel_size=1)
        self.bottom_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.map_softmax = nn.Softmax(dim=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        bsize = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        x_top = self.top_conv(x)
        x_center = self.center_conv(x)
        x_bottom = self.bottom_conv(x)
        x_top = x_top.flatten(2)
        x_center = x_center.flatten(2).transpose(1, 2)
        x_bottom = x_bottom.flatten(2).transpose(1, 2)
        sp_map = torch.matmul(x_center, x_top)
        sp_map = self.map_softmax(sp_map.view(bsize, -1)).view(bsize, h * w, h * w)
        sp_map = torch.matmul(sp_map, x_bottom)
        sp_map = sp_map.reshape(bsize, self.in_channels, h, w)
        x = x + sp_map
        x = self.out_conv(x)
        return x


class MFAB(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, reduction=16):
        super(MFAB, self).__init__()
        self.hl_conv = nn.Sequential(md.Conv2dReLU(in_channels, in_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm), md.Conv2dReLU(in_channels, skip_channels, kernel_size=1, use_batchnorm=use_batchnorm))
        self.SE_ll = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(skip_channels, skip_channels // reduction, 1), nn.ReLU(inplace=True), nn.Conv2d(skip_channels // reduction, skip_channels, 1), nn.Sigmoid())
        self.SE_hl = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(skip_channels, skip_channels // reduction, 1), nn.ReLU(inplace=True), nn.Conv2d(skip_channels // reduction, skip_channels, 1), nn.Sigmoid())
        self.conv1 = md.Conv2dReLU(skip_channels + skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = md.Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)

    def forward(self, x, skip=None):
        x = self.hl_conv(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        attention_hl = self.SE_hl(x)
        if skip is not None:
            attention_ll = self.SE_ll(skip)
            attention_hl = attention_hl + attention_ll
            x = x * attention_hl
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MAnetDecoder(nn.Module):

    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, reduction=16, use_batchnorm=True, pab_channels=64):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError('Model depth is {}, but you provide `decoder_channels` for {} blocks.'.format(n_blocks, len(decoder_channels)))
        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        self.center = PAB(head_channels, head_channels, pab_channels=pab_channels)
        kwargs = dict(use_batchnorm=use_batchnorm)
        blocks = [(MFAB(in_ch, skip_ch, out_ch, reduction=reduction, **kwargs) if skip_ch > 0 else DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)) for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]
        head = features[0]
        skips = features[1:]
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


class MAnet(SegmentationModel):
    """MAnet_ :  Multi-scale Attention Net. The MA-Net can capture rich contextual dependencies based on the attention mechanism, 
    using two blocks:
     - Position-wise Attention Block (PAB), which captures the spatial dependencies between pixels in a global view
     - Multi-scale Fusion Attention Block (MFAB), which  captures the channel dependencies between any feature map by
       multi-scale semantic feature fusion

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_pab_channels: A number of channels for PAB module in decoder. 
            Default is 64.
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **MAnet**

    .. _MAnet:
        https://ieeexplore.ieee.org/abstract/document/9201310

    """

    def __init__(self, encoder_name: 'str'='resnet34', encoder_depth: 'int'=5, encoder_weights: 'Optional[str]'='imagenet', decoder_use_batchnorm: 'bool'=True, decoder_channels: 'List[int]'=(256, 128, 64, 32, 16), decoder_pab_channels: 'int'=64, in_channels: 'int'=3, classes: 'int'=1, activation: 'Optional[Union[str, callable]]'=None, aux_params: 'Optional[dict]'=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.decoder = MAnetDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels, n_blocks=encoder_depth, use_batchnorm=decoder_use_batchnorm, pab_channels=decoder_pab_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=classes, activation=activation, kernel_size=3)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'manet-{}'.format(encoder_name)
        self.initialize()


class ConvBnRelu(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int', stride: 'int'=1, padding: 'int'=0, dilation: 'int'=1, groups: 'int'=1, bias: 'bool'=True, add_relu: 'bool'=True, interpolate: 'bool'=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
        self.add_relu = add_relu
        self.interpolate = interpolate
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x


class FPABlock(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_mode='bilinear'):
        super(FPABlock, self).__init__()
        self.upscale_mode = upscale_mode
        if self.upscale_mode == 'bilinear':
            self.align_corners = True
        else:
            self.align_corners = False
        self.branch1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
        self.mid = nn.Sequential(ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0))
        self.down1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), ConvBnRelu(in_channels=in_channels, out_channels=1, kernel_size=7, stride=1, padding=3))
        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2))
        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1), ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1))
        self.conv2 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv1 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        upscale_parameters = dict(mode=self.upscale_mode, align_corners=self.align_corners)
        b1 = F.interpolate(b1, size=(h, w), **upscale_parameters)
        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), **upscale_parameters)
        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(x, size=(h // 2, w // 2), **upscale_parameters)
        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(x, size=(h, w), **upscale_parameters)
        x = torch.mul(x, mid)
        x = x + b1
        return x


class GAUBlock(nn.Module):

    def __init__(self, in_channels: 'int', out_channels: 'int', upscale_mode: 'str'='bilinear'):
        super(GAUBlock, self).__init__()
        self.upscale_mode = upscale_mode
        self.align_corners = True if upscale_mode == 'bilinear' else None
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), ConvBnRelu(in_channels=out_channels, out_channels=out_channels, kernel_size=1, add_relu=False), nn.Sigmoid())
        self.conv2 = ConvBnRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x, y):
        """
        Args:
            x: low level feature
            y: high level feature
        """
        h, w = x.size(2), x.size(3)
        y_up = F.interpolate(y, size=(h, w), mode=self.upscale_mode, align_corners=self.align_corners)
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)
        return y_up + z


class PANDecoder(nn.Module):

    def __init__(self, encoder_channels, decoder_channels, upscale_mode: 'str'='bilinear'):
        super().__init__()
        self.fpa = FPABlock(in_channels=encoder_channels[-1], out_channels=decoder_channels)
        self.gau3 = GAUBlock(in_channels=encoder_channels[-2], out_channels=decoder_channels, upscale_mode=upscale_mode)
        self.gau2 = GAUBlock(in_channels=encoder_channels[-3], out_channels=decoder_channels, upscale_mode=upscale_mode)
        self.gau1 = GAUBlock(in_channels=encoder_channels[-4], out_channels=decoder_channels, upscale_mode=upscale_mode)

    def forward(self, *features):
        bottleneck = features[-1]
        x5 = self.fpa(bottleneck)
        x4 = self.gau3(features[-2], x5)
        x3 = self.gau2(features[-3], x4)
        x2 = self.gau1(features[-4], x3)
        return x2


class PAN(SegmentationModel):
    """ Implementation of PAN_ (Pyramid Attention Network).

    Note:
        Currently works with shape of input tensor >= [B x C x 128 x 128] for pytorch <= 1.1.0
        and with shape of input tensor >= [B x C x 256 x 256] for pytorch == 1.3.1

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_dilation: Flag to use dilation in encoder last layer. Doesn't work with ***ception***, **vgg***, 
            **densenet*`** backbones, default is **True**
        decoder_channels: A number of convolution layer filters in decoder blocks
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **PAN**

    .. _PAN:
        https://arxiv.org/abs/1805.10180

    """

    def __init__(self, encoder_name: 'str'='resnet34', encoder_weights: 'Optional[str]'='imagenet', encoder_dilation: 'bool'=True, decoder_channels: 'int'=32, in_channels: 'int'=3, classes: 'int'=1, activation: 'Optional[Union[str, callable]]'=None, upsampling: 'int'=4, aux_params: 'Optional[dict]'=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights)
        if encoder_dilation:
            self.encoder.make_dilated(stage_list=[5], dilation_list=[2])
        self.decoder = PANDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels, out_channels=classes, activation=activation, kernel_size=3, upsampling=upsampling)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'pan-{}'.format(encoder_name)
        self.initialize()


class PSPBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)), modules.Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm))

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


class PSPModule(nn.Module):

    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()
        self.blocks = nn.ModuleList([PSPBlock(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes])

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module):

    def __init__(self, encoder_channels, use_batchnorm=True, out_channels=512, dropout=0.2):
        super().__init__()
        self.psp = PSPModule(in_channels=encoder_channels[-1], sizes=(1, 2, 3, 6), use_bathcnorm=use_batchnorm)
        self.conv = modules.Conv2dReLU(in_channels=encoder_channels[-1] * 2, out_channels=out_channels, kernel_size=1, use_batchnorm=use_batchnorm)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, *features):
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class PSPNet(SegmentationModel):
    """PSPNet_ is a fully convolution neural network for image semantic segmentation. Consist of 
    *encoder* and *Spatial Pyramid* (decoder). Spatial Pyramid build on top of encoder and does not 
    use "fine-features" (features of high spatial resolution). PSPNet can be used for multiclass segmentation
    of high resolution images, however it is not good for detecting small objects and producing accurate, pixel-level mask. 

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        psp_out_channels: A number of filters in Spatial Pyramid
        psp_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        psp_dropout: Spatial dropout rate in [0, 1) used in Spatial Pyramid
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 8 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **PSPNet**

    .. _PSPNet:
        https://arxiv.org/abs/1612.01105
    """

    def __init__(self, encoder_name: 'str'='resnet34', encoder_weights: 'Optional[str]'='imagenet', encoder_depth: 'int'=3, psp_out_channels: 'int'=512, psp_use_batchnorm: 'bool'=True, psp_dropout: 'float'=0.2, in_channels: 'int'=3, classes: 'int'=1, activation: 'Optional[Union[str, callable]]'=None, upsampling: 'int'=8, aux_params: 'Optional[dict]'=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.decoder = PSPDecoder(encoder_channels=self.encoder.out_channels, use_batchnorm=psp_use_batchnorm, out_channels=psp_out_channels, dropout=psp_dropout)
        self.segmentation_head = SegmentationHead(in_channels=psp_out_channels, out_channels=classes, kernel_size=3, activation=activation, upsampling=upsampling)
        if aux_params:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'psp-{}'.format(encoder_name)
        self.initialize()


class CenterBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        conv2 = md.Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):

    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, use_batchnorm=True, attention_type=None, center=False):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError('Model depth is {}, but you provide `decoder_channels` for {} blocks.'.format(n_blocks, len(decoder_channels)))
        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [DecoderBlock(in_ch, skip_ch, out_ch, **kwargs) for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]
        head = features[0]
        skips = features[1:]
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask: 'Optional[Tensor]'=None, memory_mask: 'Optional[Tensor]'=None, tgt_key_padding_mask: 'Optional[Tensor]'=None, memory_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None, query_pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: 'Optional[Tensor]'):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask: 'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None, pos: 'Optional[Tensor]'=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


def build_transformer(d_model=256, dropout=0.01, nhead=8, dim_feedforward=2048, num_encoder_layers=6, num_decoder_layers=6, normalize_before=False, return_intermediate_dec=True):
    return Transformer(d_model=d_model, dropout=dropout, nhead=nhead, dim_feedforward=dim_feedforward, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, normalize_before=normalize_before, return_intermediate_dec=return_intermediate_dec)


class TrSkipUnet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(self, encoder_name: 'str'='resnet34', encoder_depth: 'int'=5, encoder_weights: 'Optional[str]'='imagenet', decoder_use_batchnorm: 'bool'=True, decoder_channels: 'List[int]'=(256, 128, 64, 32, 16), decoder_attention_type: 'Optional[str]'=None, in_channels: 'int'=3, classes: 'int'=1, activation: 'Optional[Union[str, callable]]'=None, aux_params: 'Optional[dict]'=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.transformer = build_transformer(args)
        self.decoder = UnetDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels, n_blocks=encoder_depth, use_batchnorm=decoder_use_batchnorm, center=True if encoder_name.startswith('vgg') else False, attention_type=decoder_attention_type)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=classes, activation=activation, kernel_size=3)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'u-{}'.format(encoder_name)
        self.initialize()


class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(self, encoder_name: 'str'='resnet34', encoder_depth: 'int'=5, encoder_weights: 'Optional[str]'='imagenet', decoder_use_batchnorm: 'bool'=True, decoder_channels: 'List[int]'=(256, 128, 64, 32, 16), decoder_attention_type: 'Optional[str]'=None, in_channels: 'int'=3, classes: 'int'=1, activation: 'Optional[Union[str, callable]]'=None, aux_params: 'Optional[dict]'=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.decoder = UnetDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels, n_blocks=encoder_depth, use_batchnorm=decoder_use_batchnorm, center=True if encoder_name.startswith('vgg') else False, attention_type=decoder_attention_type)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=classes, activation=activation, kernel_size=3)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'u-{}'.format(encoder_name)
        self.initialize()


class UnetPlusPlusDecoder(nn.Module):

    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, use_batchnorm=True, attention_type=None, center=False):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError('Model depth is {}, but you provide `decoder_channels` for {} blocks.'.format(n_blocks, len(decoder_channels)))
        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f'x_{depth_idx}_{layer_idx}'] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f'x_{0}_{len(self.in_channels) - 1}'] = DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1], **kwargs)
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f'x_{depth_idx}_{depth_idx}'](features[depth_idx], features[depth_idx + 1])
                    dense_x[f'x_{depth_idx}_{depth_idx}'] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f'x_{idx}_{dense_l_i}'] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f'x_{depth_idx}_{dense_l_i}'] = self.blocks[f'x_{depth_idx}_{dense_l_i}'](dense_x[f'x_{depth_idx}_{dense_l_i - 1}'], cat_features)
        dense_x[f'x_{0}_{self.depth}'] = self.blocks[f'x_{0}_{self.depth}'](dense_x[f'x_{0}_{self.depth - 1}'])
        return dense_x[f'x_{0}_{self.depth}']


class UnetPlusPlus(SegmentationModel):
    """Unet++ is a fully convolution neural network for image semantic segmentation. Consist of *encoder* 
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial 
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Decoder of
    Unet++ is more complex than in usual Unet.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: **Unet++**

    Reference:
        https://arxiv.org/abs/1807.10165

    """

    def __init__(self, encoder_name: 'str'='resnet34', encoder_depth: 'int'=5, encoder_weights: 'Optional[str]'='imagenet', decoder_use_batchnorm: 'bool'=True, decoder_channels: 'List[int]'=(256, 128, 64, 32, 16), decoder_attention_type: 'Optional[str]'=None, in_channels: 'int'=3, classes: 'int'=1, activation: 'Optional[Union[str, callable]]'=None, aux_params: 'Optional[dict]'=None):
        super().__init__()
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights)
        self.decoder = UnetPlusPlusDecoder(encoder_channels=self.encoder.out_channels, decoder_channels=decoder_channels, n_blocks=encoder_depth, use_batchnorm=decoder_use_batchnorm, center=True if encoder_name.startswith('vgg') else False, attention_type=decoder_attention_type)
        self.segmentation_head = SegmentationHead(in_channels=decoder_channels[-1], out_channels=classes, activation=activation, kernel_size=3)
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = 'unetplusplus-{}'.format(encoder_name)
        self.initialize()


class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', name)
            return re.sub('([a-z0-9])([A-Z])', '\\1_\\2', s1).lower()
        else:
            return self._name


class Metric(BaseObject):
    pass

