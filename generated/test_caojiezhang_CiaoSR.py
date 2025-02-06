import sys
_module = sys.modules[__name__]
del sys
metrics = _module
datamodule = _module
crop = _module
generate_assistant = _module
random_bicubic_sampling = _module
random_degradations = _module
ciaosr_net = _module
swinir_net = _module
arch_csnln = _module
arch_util = _module
basicblock = _module
vgg_arch = _module
unet_disc = _module
mlp_refiner = _module
gan_loss = _module
perceptual_loss = _module
basic_restorer = _module
ciaosr = _module
real_ciaosr = _module
utils_image = _module
test = _module
train = _module
train_pl = _module

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


from typing import Any


from typing import Optional


from typing import Union


from typing import Dict


from typing import List


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import default_collate


import numpy as np


from torch.nn.modules.utils import _pair


import torch


import math


import logging


import random


import torch.nn.functional as F


from scipy.linalg import orth


import torch.nn as nn


import time


import torch.utils.checkpoint as checkpoint


from collections import OrderedDict


import torchvision.models.vgg as vgg


import torch.nn.init as init


import torch.nn.utils.spectral_norm as spectral_norm


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.utils import spectral_norm


from torch import Tensor


import torch.autograd as autograd


from torch.nn.functional import conv2d


from torch.nn import functional as F


import numbers


from copy import deepcopy


from torchvision.utils import make_grid


import copy


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True, bn=False, act=nn.PReLU()):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride, bias=bias)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    padding_top = int(padding_rows / 2.0)
    padding_left = int(padding_cols / 2.0)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = padding_left, padding_right, padding_top, padding_bottom
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.                Only "same" or "valid" are supported.'.format(padding))
    unfold = torch.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0, stride=strides)
    patches = unfold(images)
    return patches


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


class CrossScaleAttention(nn.Module):

    def __init__(self, channel=64, reduction=2, ksize=3, scale=2, stride=1, softmax_scale=10, average=True, conv=default_conv):
        super(CrossScaleAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.softmax_scale = softmax_scale
        self.scale = scale
        self.average = average
        escape_NaN = torch.FloatTensor([0.0001])
        self.register_buffer('escape_NaN', escape_NaN)
        self.conv_match_1 = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match_2 = BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
        if 3 in scale:
            self.downx3 = nn.Conv2d(channel, channel, ksize, 3, 1)
        if 4 in scale:
            self.downx4 = nn.Conv2d(channel, channel, ksize, 4, 1)
        self.down = nn.Conv2d(channel, channel, ksize, 2, 1)

    def forward(self, input):
        _, _, H, W = input.shape
        if not isinstance(self.scale, list):
            self.scale = [self.scale]
        res_y = []
        for s in self.scale:
            mod_pad_h, mod_pad_w = 0, 0
            if H % s != 0:
                mod_pad_h = s - H % s
            if W % s != 0:
                mod_pad_w = s - W % s
            input_pad = F.pad(input, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            embed_w = self.conv_assembly(input_pad)
            match_input = self.conv_match_1(input_pad)
            shape_input = list(embed_w.size())
            input_groups = torch.split(match_input, 1, dim=0)
            kernel = s * self.ksize
            raw_w = extract_image_patches(embed_w, ksizes=[kernel, kernel], strides=[self.stride * s, self.stride * s], rates=[1, 1], padding='same')
            raw_w = raw_w.view(shape_input[0], shape_input[1], kernel, kernel, -1)
            raw_w = raw_w.permute(0, 4, 1, 2, 3).contiguous()
            raw_w_groups = torch.split(raw_w, 1, dim=0)
            ref = F.interpolate(input_pad, scale_factor=1.0 / s, mode='bilinear')
            ref = self.conv_match_2(ref)
            w = extract_image_patches(ref, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
            shape_ref = ref.shape
            w = w.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, -1)
            w = w.permute(0, 4, 1, 2, 3).contiguous()
            w_groups = torch.split(w, 1, dim=0)
            y = []
            for xi, wi, raw_wi in zip(input_groups, w_groups, raw_w_groups):
                wi = wi[0]
                max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2), axis=[1, 2, 3], keepdim=True)), self.escape_NaN)
                wi_normed = wi / max_wi
                xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])
                yi = F.conv2d(xi, wi_normed, stride=1)
                yi = yi.view(1, shape_ref[2] * shape_ref[3], shape_input[2], shape_input[3])
                yi = F.softmax(yi * self.softmax_scale, dim=1)
                if self.average == False:
                    yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()
                wi_center = raw_wi[0]
                yi = F.conv_transpose2d(yi, wi_center, stride=self.stride * s, padding=s)
                if s == 2:
                    yi = self.down(yi)
                elif s == 3:
                    yi = self.downx3(yi)
                elif s == 4:
                    yi = self.downx4(yi)
                yi = yi / 6.0
                y.append(yi)
            y = torch.cat(y, dim=0)
            y = y[:, :, :H, :W]
            res_y.append(y)
        res_y = torch.cat(res_y, dim=1)
        return res_y


class LocalImplicitSRNet(nn.Module):
    """
    The subclasses should define `generator` with `encoder` and `imnet`,
        and overwrite the function `gen_feature`.
    If `encoder` does not contain `mid_channels`, `__init__` should be
        overwrite.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self, encoder, imnet_q, imnet_k, imnet_v, query_mlp, key_mlp, value_mlp, local_size=2, feat_unfold=True, eval_bsize=None, non_local_attn=True, multi_scale=[2], softmax_scale=1):
        super().__init__()
        self.feat_unfold = feat_unfold
        self.eval_bsize = eval_bsize
        self.local_size = local_size
        self.non_local_attn = non_local_attn
        self.multi_scale = multi_scale
        self.softmax_scale = softmax_scale
        self.encoder = build_backbone(encoder)
        if hasattr(self.encoder, 'mid_channels'):
            imnet_dim = self.encoder.mid_channels
        else:
            imnet_dim = self.encoder.embed_dim
        if self.feat_unfold:
            imnet_q['in_dim'] = imnet_dim * 9
            imnet_k['in_dim'] = imnet_k['out_dim'] = imnet_dim * 9
            imnet_v['in_dim'] = imnet_v['out_dim'] = imnet_dim * 9
        else:
            imnet_q['in_dim'] = imnet_dim
            imnet_k['in_dim'] = imnet_k['out_dim'] = imnet_dim
            imnet_v['in_dim'] = imnet_v['out_dim'] = imnet_dim
        imnet_k['in_dim'] += 4
        imnet_v['in_dim'] += 4
        if self.non_local_attn:
            imnet_q['in_dim'] += imnet_dim * len(multi_scale)
            imnet_v['in_dim'] += imnet_dim * len(multi_scale)
            imnet_v['out_dim'] += imnet_dim * len(multi_scale)
        self.imnet_q = build_component(imnet_q)
        self.imnet_k = build_component(imnet_k)
        self.imnet_v = build_component(imnet_v)
        if self.non_local_attn:
            self.cs_attn = CrossScaleAttention(channel=imnet_dim, scale=multi_scale)

    def forward(self, x, coord, cell, test_mode=False):
        """Forward function.

        Args:
            x: input tensor.
            coord (Tensor): coordinates tensor.
            cell (Tensor): cell tensor.
            test_mode (bool): Whether in test mode or not. Default: False.

        Returns:
            pred (Tensor): output of model.
        """
        feature = self.gen_feature(x)
        if self.eval_bsize is None or not test_mode:
            pred = self.query_rgb(feature, coord, cell)
        else:
            pred = self.batched_predict(feature, coord, cell)
        pred += F.grid_sample(x, coord.flip(-1).unsqueeze(1), mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        return pred

    def query_rgb(self, features, coord, scale=None):
        """Query RGB value of GT.

        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.

        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).

        Returns:
            result (Tensor): (part of) output.
        """
        res_features = []
        for feature in features:
            B, C, H, W = feature.shape
            if self.feat_unfold:
                feat_q = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)
                feat_k = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)
                if self.non_local_attn:
                    non_local_feat_v = self.cs_attn(feature)
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)
                    feat_v = torch.cat([feat_v, non_local_feat_v], dim=1)
                else:
                    feat_v = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)
            else:
                feat_q = feat_k = feat_v = feature
            query = F.grid_sample(feat_q, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False).permute(0, 3, 2, 1).contiguous()
            feat_coord = make_coord(feature.shape[-2:], flatten=False).permute(2, 0, 1).unsqueeze(0).expand(B, 2, *feature.shape[-2:])
            feat_coord = feat_coord
            if self.local_size == 1:
                v_lst = [(0, 0)]
            else:
                v_lst = [(i, j) for i in range(-1, 2, 4 - self.local_size) for j in range(-1, 2, 4 - self.local_size)]
            eps_shift = 1e-06
            preds_k, preds_v = [], []
            for v in v_lst:
                vx, vy = v[0], v[1]
                tx = ((H - 1) / (1 - scale[:, 0, 0])).view(B, 1)
                ty = ((W - 1) / (1 - scale[:, 0, 1])).view(B, 1)
                rx = (2 * abs(vx) - 1) / tx if vx != 0 else 0
                ry = (2 * abs(vy) - 1) / ty if vy != 0 else 0
                bs, q = coord.shape[:2]
                coord_ = coord.clone()
                if vx != 0:
                    coord_[:, :, 0] += vx / abs(vx) * rx + eps_shift
                if vy != 0:
                    coord_[:, :, 1] += vy / abs(vy) * ry + eps_shift
                coord_.clamp_(-1 + 1e-06, 1 - 1e-06)
                key = F.grid_sample(feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()
                value = F.grid_sample(feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()
                coord_k = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                Q, K = coord, coord_k
                rel = Q - K
                rel[:, :, 0] *= feature.shape[-2]
                rel[:, :, 1] *= feature.shape[-1]
                inp = rel
                scale_ = scale.clone()
                scale_[:, :, 0] *= feature.shape[-2]
                scale_[:, :, 1] *= feature.shape[-1]
                inp_v = torch.cat([value, inp, scale_], dim=-1)
                inp_k = torch.cat([key, inp, scale_], dim=-1)
                inp_k = inp_k.contiguous().view(bs * q, -1)
                inp_v = inp_v.contiguous().view(bs * q, -1)
                weight_k = self.imnet_k(inp_k).view(bs, q, -1).contiguous()
                pred_k = (key * weight_k).view(bs, q, -1)
                weight_v = self.imnet_v(inp_v).view(bs, q, -1).contiguous()
                pred_v = (value * weight_v).view(bs, q, -1)
                preds_v.append(pred_v)
                preds_k.append(pred_k)
            preds_k = torch.stack(preds_k, dim=-1)
            preds_v = torch.stack(preds_v, dim=-2)
            attn = query @ preds_k
            x = (attn / self.softmax_scale).softmax(dim=-1) @ preds_v
            x = x.view(bs * q, -1)
            res_features.append(x)
        result = torch.cat(res_features, dim=-1)
        result = self.imnet_q(result)
        result = result.view(bs, q, -1)
        return result

    def batched_predict(self, x, coord, cell):
        """Batched predict.

        Args:
            x (Tensor): Input tensor.
            coord (Tensor): coord tensor.
            cell (Tensor): cell tensor.

        Returns:
            pred (Tensor): output of model.
        """
        with torch.no_grad():
            n = coord.shape[1]
            left = 0
            preds = []
            while left < n:
                right = min(left + self.eval_bsize, n)
                pred = self.query_rgb(x, coord[:, left:right, :], cell[:, left:right, :])
                preds.append(pred)
                left = right
            pred = torch.cat(preds, dim=1)
        return pred

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')


class LocalImplicitSRRDN(LocalImplicitSRNet):
    """ITSRN net based on RDN.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feat unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self, encoder, imnet_q, imnet_k, imnet_v, query_mlp=None, key_mlp=None, value_mlp=None, local_size=2, feat_unfold=True, eval_bsize=None, non_local_attn=True, multi_scale=[2], softmax_scale=1):
        super().__init__(encoder=encoder, imnet_q=imnet_q, imnet_k=imnet_k, imnet_v=imnet_v, query_mlp=query_mlp, key_mlp=key_mlp, value_mlp=value_mlp, local_size=local_size, feat_unfold=feat_unfold, eval_bsize=eval_bsize, non_local_attn=non_local_attn, multi_scale=multi_scale, softmax_scale=softmax_scale)
        self.sfe1 = self.encoder.sfe1
        self.sfe2 = self.encoder.sfe2
        self.rdbs = self.encoder.rdbs
        self.gff = self.encoder.gff
        self.num_blocks = self.encoder.num_blocks
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        x = sfe2
        local_features = []
        for i in range(self.num_blocks):
            x = self.rdbs[i](x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, 1)) + sfe1
        return [x]


class LocalImplicitSREDSR(LocalImplicitSRNet):
    """LocalImplicitSR based on EDSR.

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self, encoder, imnet_q, imnet_k, imnet_v, query_mlp=None, key_mlp=None, value_mlp=None, local_size=2, feat_unfold=True, eval_bsize=None, non_local_attn=True, multi_scale=[2], softmax_scale=1):
        super().__init__(encoder=encoder, imnet_q=imnet_q, imnet_k=imnet_k, imnet_v=imnet_v, query_mlp=query_mlp, key_mlp=key_mlp, value_mlp=value_mlp, local_size=local_size, feat_unfold=feat_unfold, eval_bsize=eval_bsize, non_local_attn=non_local_attn, multi_scale=multi_scale, softmax_scale=softmax_scale)
        self.conv_first = self.encoder.conv_first
        self.body = self.encoder.body
        self.conv_after_body = self.encoder.conv_after_body
        del self.encoder

    def gen_feature(self, x):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.conv_first(x)
        res = self.body(x)
        res = self.conv_after_body(res)
        res += x
        return [res]


class LocalImplicitSRSWINIR(LocalImplicitSRNet):
    """ITSRN net based on EDSR.

    Paper: Learning Continuous Image Representation with
           Local Implicit Image Function

    Args:
        encoder (dict): Config for the generator.
        imnet (dict): Config for the imnet.
        local_ensemble (bool): Whether to use local ensemble. Default: True.
        feat_unfold (bool): Whether to use feature unfold. Default: True.
        cell_decode (bool): Whether to use cell decode. Default: True.
        eval_bsize (int): Size of batched predict. Default: None.
    """

    def __init__(self, window_size, encoder, imnet_q, imnet_k, imnet_v, query_mlp=None, key_mlp=None, value_mlp=None, local_size=2, feat_unfold=True, eval_bsize=None, non_local_attn=True, multi_scale=[2], softmax_scale=1):
        super().__init__(encoder=encoder, imnet_q=imnet_q, imnet_k=imnet_k, imnet_v=imnet_v, query_mlp=query_mlp, key_mlp=key_mlp, value_mlp=value_mlp, local_size=local_size, feat_unfold=feat_unfold, eval_bsize=eval_bsize, non_local_attn=non_local_attn, multi_scale=multi_scale, softmax_scale=softmax_scale)
        self.window_size = window_size
        self.conv_first = self.encoder.conv_first
        self.patch_embed = self.encoder.patch_embed
        self.pos_drop = self.encoder.pos_drop
        self.layers = self.encoder.layers
        self.norm = self.encoder.norm
        self.patch_unembed = self.encoder.patch_unembed
        self.conv_after_body = self.encoder.conv_after_body
        del self.encoder

    def forward_features(self, x):
        x_size = x.shape[2], x.shape[3]
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def gen_feature(self, img):
        """Generate feature.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = img.size()
        if h % self.window_size != 0:
            mod_pad_h = self.window_size - h % self.window_size
        if w % self.window_size != 0:
            mod_pad_w = self.window_size - w % self.window_size
        img_pad = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = self.conv_first(img_pad)
        res = self.forward_features(x)
        res = self.conv_after_body(res)
        res += x
        _, _, h, w = res.size()
        res = res[:, :, 0:h - mod_pad_h, 0:w - mod_pad_w]
        return [res]


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size))
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) ->str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += H // 2 * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) ->str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    """ Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.residual_group = BasicLayer(dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, downsample=downsample, use_checkpoint=use_checkpoint)
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(dim // 4, dim, 3, 1, 1))
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        nf (int): Channel number of intermediate features.
    """

    def __init__(self, scale, nf):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(nf, 4 * nf, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(nf, 9 * nf, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, scale ** 2 * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class SwinIR(nn.Module):
    """ SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, upscale=2, img_range=1.0, upsampler='', resi_connection='1conv', **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = 0.4488, 0.4371, 0.404
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint, img_size=img_size, patch_size=patch_size, resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = x.shape[2], x.shape[3]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)
        x = x / self.img_range + self.mean
        return x[:, :, :H * self.upscale, :W * self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class ContrasExtractorLayer(nn.Module):

    def __init__(self):
        super(ContrasExtractorLayer, self).__init__()
        vgg16_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5']
        conv3_1_idx = vgg16_layers.index('conv3_1')
        features = getattr(vgg, 'vgg16')(pretrained=True).features[:conv3_1_idx + 1]
        modified_net = OrderedDict()
        for k, v in zip(vgg16_layers, features):
            modified_net[k] = v
        self.model = nn.Sequential(modified_net)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, batch):
        batch = (batch - self.mean) / self.std
        output = self.model(batch)
        return output


class ContrasExtractorSep(nn.Module):

    def __init__(self):
        super(ContrasExtractorSep, self).__init__()
        self.feature_extraction_image1 = ContrasExtractorLayer()
        self.feature_extraction_image2 = ContrasExtractorLayer()

    def forward(self, image1, image2):
        dense_features1 = self.feature_extraction_image1(image1)
        dense_features2 = self.feature_extraction_image2(image2)
        return {'dense_features1': dense_features1, 'dense_features2': dense_features2}


NAMES = {'vgg11': ['conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'], 'vgg13': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'], 'vgg16': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'], 'vgg19': ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']}


def insert_bn(names: 'list'):
    """Inserts bn layer after each conv.

    Args:
        names (list): The list of layer names.

    Returns:
        list: The list of layer names with bn layers.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            pos = name.replace('conv', '')
            names_bn.append('bn' + pos)
    return names_bn


class VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.

    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.

    Args:
        layer_name_list (list[str]): According to the name in this list, forward
            function will return the corresponding features. Hear is an example:
            {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed.  Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """

    def __init__(self, layer_name_list, vgg_type='vgg19', use_input_norm=True, requires_grad=False, remove_pooling=False, pooling_stride=2):
        super(VGGFeatureExtractor, self).__init__()
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.names = NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            self.names = insert_bn(self.names)
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx
        features = getattr(vgg, vgg_type)(pretrained=True).features[:max_idx + 1]
        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                if remove_pooling:
                    continue
                else:
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v
        self.vgg_net = nn.Sequential(modified_net)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()
        return output


def sample_patches(inputs, patch_size=3, stride=1):
    """Extract sliding local patches from an input feature tensor.
    The sampled pathes are row-major.
    Args:
        inputs (Tensor): the input feature maps, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.
    Returns:
        patches (Tensor): extracted patches, shape: (c, patch_size,
            patch_size, n_patches).
    """
    c, h, w = inputs.shape
    patches = inputs.unfold(1, patch_size, stride).unfold(2, patch_size, stride).reshape(c, -1, patch_size, patch_size).permute(0, 2, 3, 1)
    return patches


def feature_match_index(feat_input, feat_ref, patch_size=3, input_stride=1, ref_stride=1, is_norm=True, norm_input=False):
    """Patch matching between input and reference features.
    Args:
        feat_input (Tensor): the feature of input, shape: (c, h, w).
        feat_ref (Tensor): the feature of reference, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.
        is_norm (bool): determine to normalize the ref feature or not.
            Default:True.
    Returns:
        max_idx (Tensor): The indices of the most similar patches.
        max_val (Tensor): The correlation values of the most similar patches.
    """
    patches_ref = sample_patches(feat_ref, patch_size, ref_stride)
    _, h, w = feat_input.shape
    batch_size = int(1024.0 ** 2 * 512 / (h * w))
    n_patches = patches_ref.shape[-1]
    max_idx, max_val = None, None
    for idx in range(0, n_patches, batch_size):
        batch = patches_ref[..., idx:idx + batch_size]
        if is_norm:
            batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-05)
        corr = F.conv2d(feat_input.unsqueeze(0), batch.permute(3, 0, 1, 2), stride=input_stride)
        max_val_tmp, max_idx_tmp = corr.squeeze(0).max(dim=0)
        if max_idx is None:
            max_idx, max_val = max_idx_tmp, max_val_tmp
        else:
            indices = max_val_tmp > max_val
            max_val[indices] = max_val_tmp[indices]
            max_idx[indices] = max_idx_tmp[indices] + idx
    if norm_input:
        patches_input = sample_patches(feat_input, patch_size, input_stride)
        norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-05
        norm = norm.view(int((h - patch_size) / input_stride + 1), int((w - patch_size) / input_stride + 1))
        max_val = max_val / norm
    return max_idx, max_val


def tensor_shift(x, shift=(2, 2), fill_val=0):
    """ Tensor shift.

    Args:
        x (Tensor): the input tensor. The shape is [b, h, w, c].
        shift (tuple): shift pixel.
        fill_val (float): fill value

    Returns:
        Tensor: the shifted tensor.
    """
    _, h, w, _ = x.size()
    shift_h, shift_w = shift
    new = torch.ones_like(x) * fill_val
    if shift_h >= 0 and shift_w >= 0:
        len_h = h - shift_h
        len_w = w - shift_w
        new[:, shift_h:shift_h + len_h, shift_w:shift_w + len_w, :] = x.narrow(1, 0, len_h).narrow(2, 0, len_w)
    else:
        raise NotImplementedError
    return new


class CorrespondenceFeatGenerationArch(nn.Module):

    def __init__(self, patch_size=3, stride=1, vgg_layer_list=['relu3_1'], vgg_type='vgg19'):
        super(CorrespondenceFeatGenerationArch, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.vgg_layer_list = vgg_layer_list
        self.vgg = VGGFeatureExtractor(layer_name_list=vgg_layer_list, vgg_type=vgg_type)

    def index_to_flow(self, max_idx):
        device = max_idx.device
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float()
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h), dim=2).unsqueeze(0).float()
        flow = flow - grid
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))
        return flow

    def forward(self, dense_features, img_ref_hr):
        batch_offset_relu = []
        for ind in range(img_ref_hr.size(0)):
            feat_in = dense_features['dense_features1'][ind]
            feat_ref = dense_features['dense_features2'][ind]
            c, h, w = feat_in.size()
            feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w)
            feat_ref = F.normalize(feat_ref.reshape(c, -1), dim=0).view(c, h, w)
            _max_idx, _max_val = feature_match_index(feat_in, feat_ref, patch_size=self.patch_size, input_stride=self.stride, ref_stride=self.stride, is_norm=True, norm_input=True)
            offset_relu3 = self.index_to_flow(_max_idx)
            shifted_offset_relu3 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu3, (i, j))
                    shifted_offset_relu3.append(flow_shift)
            shifted_offset_relu3 = torch.cat(shifted_offset_relu3, dim=0)
            batch_offset_relu.append(shifted_offset_relu3)
        batch_offset_relu = torch.stack(batch_offset_relu, dim=0)
        img_ref_feat = self.vgg(img_ref_hr)
        return batch_offset_relu, img_ref_feat


class CorrespondenceGenerationArch(nn.Module):

    def __init__(self, patch_size=3, stride=1):
        super(CorrespondenceGenerationArch, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

    def index_to_flow(self, max_idx):
        device = max_idx.device
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float()
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h), dim=2).unsqueeze(0).float()
        flow = flow - grid
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))
        return flow

    def forward(self, feats_in, feats_ref):
        batch_offset_relu = []
        for ind in range(feats_in.size(0)):
            feat_in = feats_in[ind]
            feat_ref = feats_ref[ind]
            c, h, w = feat_in.size()
            feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w)
            feat_ref = F.normalize(feat_ref.reshape(c, -1), dim=0).view(c, h // 2, w // 2)
            _max_idx, _max_val = feature_match_index(feat_in, feat_ref, patch_size=self.patch_size, input_stride=self.stride, ref_stride=self.stride, is_norm=True, norm_input=True)
            offset = self.index_to_flow(_max_idx)
            shifted_offset = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset, (i, j))
                    shifted_offset.append(flow_shift)
            shifted_offset = torch.cat(shifted_offset, dim=0)
            batch_offset_relu.append(shifted_offset)
        batch_offset_relu = torch.stack(batch_offset_relu, dim=0)
        return batch_offset_relu


class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(arch_util.ResidualBlockNoBN, n_blocks, nf=nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)
        return feat


def default_init_weights(module_list, scale=1):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        nf (int): Number of features. Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
        sn (bool): Whether to use spectral norm. Default: False.
        n_power_iterations (int): Used in spectral norm. Default: 1.
        sn_bias (bool): Whether to apply spectral norm to bias. Default: True.

    """

    def __init__(self, nf=64, res_scale=1, pytorch_init=False, sn=False, n_power_iterations=1, sn_bias=True):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if sn:
            self.conv1 = spectral_norm(self.conv1, name='weight', n_power_iterations=n_power_iterations)
            self.conv2 = spectral_norm(self.conv2, name='weight', n_power_iterations=n_power_iterations)
            if sn_bias:
                self.conv1 = spectral_norm(self.conv1, name='bias', n_power_iterations=n_power_iterations)
                self.conv2 = spectral_norm(self.conv2, name='bias', n_power_iterations=n_power_iterations)
        self.relu = nn.ReLU(inplace=True)
        if not sn and not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ResidualBlockwithBN(nn.Module):
    """Residual block with BN.

    It has a style of:
        ---Conv-BN-ReLU-Conv-BN-+-
         |______________________|

    Args:
        nf (int): Number of features. Channel number of intermediate features.
            Default: 64.
        bn_affine (bool): Whether to use affine in BN layers. Default: True.
    """

    def __init__(self, nf=64, bn_affine=True):
        super(ResidualBlockwithBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(nf, affine=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(nf, affine=True)
        self.relu = nn.ReLU(inplace=True)
        default_init_weights([self.conv1, self.conv2], 1)

    def forward(self, x):
        identity = x
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        return identity + out


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class FFTBlock(nn.Module):

    def __init__(self, channel=64):
        super(FFTBlock, self).__init__()
        self.conv_fc = nn.Sequential(nn.Conv2d(1, channel, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel, 1, 1, padding=0, bias=True), nn.Softplus())

    def forward(self, x, u, d, sigma):
        rho = self.conv_fc(sigma)
        x = torch.irfft(self.divcomplex(u + rho.unsqueeze(-1) * torch.rfft(x, 2, onesided=False), d + self.real2complex(rho)), 2, onesided=False)
        return x

    def divcomplex(self, x, y):
        a = x[..., 0]
        b = x[..., 1]
        c = y[..., 0]
        d = y[..., 1]
        cd2 = c ** 2 + d ** 2
        return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)

    def real2complex(self, x):
        return torch.stack([x, torch.zeros(x.shape).type_as(x)], -1)


class ConcatBlock(nn.Module):

    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return self.sub.__repr__() + 'concat'


class ShortcutBlock(nn.Module):

    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class BasicConv(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True), BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False))

    def forward(self, x):
        return self.main(x) + x


class CALayer(nn.Module):

    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_fc(y)
        return x * y


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.0001, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


class RCABlock(nn.Module):

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)
        return res + x


class RCAGroup(nn.Module):

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', reduction=16, nb=12):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]
        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding, bias, mode, reduction) for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))
        self.rg = nn.Sequential(*RG)

    def forward(self, x):
        res = self.rg(x)
        return res + x


class ResidualDenseBlock_5C(nn.Module):

    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR'):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode)
        self.conv2 = conv(nc + gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv3 = conv(nc + 2 * gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv4 = conv(nc + 3 * gc, gc, kernel_size, stride, padding, bias, mode)
        self.conv5 = conv(nc + 4 * gc, nc, kernel_size, stride, padding, bias, mode[:-1])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x


class RRDB(nn.Module):

    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1, bias=True, mode='CR'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride, padding, bias, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul_(0.2) + x


def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R'):
    assert len(mode) < 4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:])
    return sequential(pool, pool_tail)


def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R'):
    assert len(mode) < 4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:])
    return sequential(pool, pool_tail)


def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R'):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode)
    return down1


class NonLocalBlock2D(nn.Module):

    def __init__(self, nc=64, kernel_size=1, stride=1, padding=0, bias=True, act_mode='B', downsample=False, downsample_mode='maxpool'):
        super(NonLocalBlock2D, self).__init__()
        inter_nc = nc // 2
        self.inter_nc = inter_nc
        self.W = conv(inter_nc, nc, kernel_size, stride, padding, bias, mode='C' + act_mode)
        self.theta = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')
        if downsample:
            if downsample_mode == 'avgpool':
                downsample_block = downsample_avgpool
            elif downsample_mode == 'maxpool':
                downsample_block = downsample_maxpool
            elif downsample_mode == 'strideconv':
                downsample_block = downsample_strideconv
            else:
                raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))
            self.phi = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
            self.g = downsample_block(nc, inter_nc, kernel_size, stride, padding, bias, mode='2')
        else:
            self.phi = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')
            self.g = conv(nc, inter_nc, kernel_size, stride, padding, bias, mode='C')

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_nc, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_nc, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class BasicConv2(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False, channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv2, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BasicConv_do(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True, transpose=False, relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BasicConv_do_eval(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False, relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do_eval, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(DOConv2d_eval(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock2(nn.Module):

    def __init__(self, out_channel):
        super(ResBlock2, self).__init__()
        self.main = nn.Sequential(BasicConv2(out_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=False), BasicConv2(out_channel, out_channel, kernel_size=3, stride=1, relu=False, norm=False))

    def forward(self, x):
        return self.main(x) + x


class ResBlock_do(nn.Module):

    def __init__(self, out_channel):
        super(ResBlock_do, self).__init__()
        self.main = nn.Sequential(BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True), BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False))

    def forward(self, x):
        return self.main(x) + x


class ResBlock_do_eval(nn.Module):

    def __init__(self, out_channel):
        super(ResBlock_do_eval, self).__init__()
        self.main = nn.Sequential(BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=True), BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=False))

    def forward(self, x):
        return self.main(x) + x


class ResBlock_do_fft_bench(nn.Module):

    def __init__(self, out_channel):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main = nn.Sequential(BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True), BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False))
        self.main_fft = nn.Sequential(BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True), BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False))
        self.dim = out_channel

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm='backward')
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        return self.main(x) + x + y


class ResBlock_fft_bench(nn.Module):

    def __init__(self, n_feat):
        super(ResBlock_fft_bench, self).__init__()
        self.main = nn.Sequential(BasicConv2(n_feat, n_feat, kernel_size=3, stride=1, relu=True), BasicConv2(n_feat, n_feat, kernel_size=3, stride=1, relu=False))
        self.main_fft = nn.Sequential(BasicConv2(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=True), BasicConv2(n_feat * 2, n_feat * 2, kernel_size=1, stride=1, relu=False))
        self.dim = n_feat

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm='backward')
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        return self.main(x) + x + y


class ResBlock_do_fft_bench_eval(nn.Module):

    def __init__(self, out_channel):
        super(ResBlock_do_fft_bench_eval, self).__init__()
        self.main = nn.Sequential(BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=True), BasicConv_do_eval(out_channel, out_channel, kernel_size=3, stride=1, relu=False))
        self.main_fft = nn.Sequential(BasicConv_do_eval(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True), BasicConv_do_eval(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False))
        self.dim = out_channel

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm='backward')
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        return self.main(x) + x + y


class UNetDiscriminatorWithSpectralNorm(nn.Module):
    """A U-Net discriminator with spectral normalization.

    Args:
        in_channels (int): Channel number of the input.
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        skip_connection (bool, optional): Whether to use skip connection.
            Default: True.
    """

    def __init__(self, in_channels, mid_channels=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        self.conv_0 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_1 = spectral_norm(nn.Conv2d(mid_channels, mid_channels * 2, 4, 2, 1, bias=False))
        self.conv_2 = spectral_norm(nn.Conv2d(mid_channels * 2, mid_channels * 4, 4, 2, 1, bias=False))
        self.conv_3 = spectral_norm(nn.Conv2d(mid_channels * 4, mid_channels * 8, 4, 2, 1, bias=False))
        self.conv_4 = spectral_norm(nn.Conv2d(mid_channels * 8, mid_channels * 4, 3, 1, 1, bias=False))
        self.conv_5 = spectral_norm(nn.Conv2d(mid_channels * 4, mid_channels * 2, 3, 1, 1, bias=False))
        self.conv_6 = spectral_norm(nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1, bias=False))
        self.conv_7 = spectral_norm(nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False))
        self.conv_8 = spectral_norm(nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False))
        self.conv_9 = nn.Conv2d(mid_channels, 1, 3, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img):
        """Forward function.

        Args:
            img (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        feat_0 = self.lrelu(self.conv_0(img))
        feat_1 = self.lrelu(self.conv_1(feat_0))
        feat_2 = self.lrelu(self.conv_2(feat_1))
        feat_3 = self.lrelu(self.conv_3(feat_2))
        feat_3 = self.upsample(feat_3.float() if feat_3.dtype == torch.bfloat16 else feat_3)
        feat_4 = self.lrelu(self.conv_4(feat_3))
        if self.skip_connection:
            feat_4 = feat_4 + feat_2
        feat_4 = self.upsample(feat_4.float() if feat_4.dtype == torch.bfloat16 else feat_4)
        feat_5 = self.lrelu(self.conv_5(feat_4))
        if self.skip_connection:
            feat_5 = feat_5 + feat_1
        feat_5 = self.upsample(feat_5.float() if feat_5.dtype == torch.bfloat16 else feat_5)
        feat_6 = self.lrelu(self.conv_6(feat_5))
        if self.skip_connection:
            feat_6 = feat_6 + feat_0
        out = self.lrelu(self.conv_7(feat_6))
        out = self.lrelu(self.conv_8(out))
        return self.conv_9(out)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained, strict=strict, logger=None)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')


class SmallUNetDiscriminatorWithSpectralNorm(nn.Module):
    """A U-Net discriminator with spectral normalization.

    Args:
        in_channels (int): Channel number of the input.
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        skip_connection (bool, optional): Whether to use skip connection.
            Default: True.
    """

    def __init__(self, in_channels, mid_channels=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        self.conv_0 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv_1 = spectral_norm(nn.Conv2d(mid_channels, mid_channels * 1, 4, 2, 1, bias=False))
        self.conv_2 = spectral_norm(nn.Conv2d(mid_channels * 1, mid_channels * 1, 4, 2, 1, bias=False))
        self.conv_5 = spectral_norm(nn.Conv2d(mid_channels * 1, mid_channels * 1, 3, 1, 1, bias=False))
        self.conv_6 = spectral_norm(nn.Conv2d(mid_channels * 1, mid_channels, 3, 1, 1, bias=False))
        self.conv_7 = spectral_norm(nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False))
        self.conv_8 = spectral_norm(nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False))
        self.conv_9 = nn.Conv2d(mid_channels, 1, 3, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img):
        """Forward function.

        Args:
            img (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        feat_0 = self.lrelu(self.conv_0(img))
        feat_1 = self.lrelu(self.conv_1(feat_0))
        feat_2 = self.lrelu(self.conv_2(feat_1))
        feat_4 = self.upsample(feat_2)
        feat_5 = self.lrelu(self.conv_5(feat_4))
        if self.skip_connection:
            feat_5 = feat_5 + feat_1
        feat_5 = self.upsample(feat_5)
        feat_6 = self.lrelu(self.conv_6(feat_5))
        if self.skip_connection:
            feat_6 = feat_6 + feat_0
        out = self.lrelu(self.conv_7(feat_6))
        out = self.lrelu(self.conv_8(out))
        return self.conv_9(out)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        pass


class PositionalEncoding1D(nn.Module):

    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / 10000 ** (torch.arange(0, channels, 2).float() / channels)
        self.register_buffer('inv_freq', inv_freq)
        self.cached_penc = None

    def get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError('The input tensor has to be 3d!')
        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc
        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum('i,j->ij', pos_x, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, :self.channels] = emb_x
        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class Cos(nn.Module):

    def forward(self, input: 'Tensor') ->Tensor:
        return torch.cos(input)


class Sin(nn.Module):

    def forward(self, input: 'Tensor') ->Tensor:
        return torch.sin(input)


class MLPRefiner(nn.Module):
    """Multilayer perceptrons (MLPs), refiner used in LIIF.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_list (list[int]): List of hidden dimensions.
    """

    def __init__(self, in_dim, out_dim, hidden_list=None, act=None):
        super().__init__()
        layers = []
        lastv = in_dim
        if hidden_list:
            for hidden in hidden_list:
                layers.append(nn.Linear(lastv, hidden))
                if act == 'cos':
                    layers.append(Cos())
                elif act == 'sin':
                    layers.append(Sin())
                else:
                    layers.append(nn.ReLU())
                lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): The input of MLP.

        Returns:
            Tensor: The output of MLP.
        """
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')


class PosMLPRefiner(nn.Module):
    """Multilayer perceptrons (MLPs), refiner used in LIIF.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_list (list[int]): List of hidden dimensions.
    """

    def __init__(self, in_dim, out_dim, hidden_list=None, is_pos=True):
        super().__init__()
        self.layers = []
        lastv = in_dim
        if hidden_list:
            for hidden in hidden_list:
                self.layers.append(nn.Linear(lastv, hidden))
                self.layers.append(nn.ReLU())
                if is_pos:
                    self.layers.append(PositionalEncoding1D(hidden))
                lastv = hidden
        self.layers.append(nn.Linear(lastv, out_dim))

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): The input of MLP.

        Returns:
            Tensor: The output of MLP.
        """
        B, C, D = x.shape
        shape = x.shape[:-1]
        for layer in self.layers:
            if 'Pos' in str(layer):
                x += layer(x)
            else:
                x = layer(x.view(B * C, -1))
                x = x.view(*shape, -1)
        return x

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError(f'"pretrained" must be a str or None. But received {type(pretrained)}.')


class GaussianBlur(nn.Module):
    """A Gaussian filter which blurs a given tensor with a two-dimensional
    gaussian kernel by convolving it along each channel. Batch operation
    is supported.

    This function is modified from kornia.filters.gaussian:
    `<https://kornia.readthedocs.io/en/latest/_modules/kornia/filters/gaussian.html>`.

    Args:
        kernel_size (tuple[int]): The size of the kernel. Default: (71, 71).
        sigma (tuple[float]): The standard deviation of the kernel.
        Default (10.0, 10.0)

    Returns:
        Tensor: The Gaussian-blurred tensor.

    Shape:
        - input: Tensor with shape of (n, c, h, w)
        - output: Tensor with shape of (n, c, h, w)
    """

    def __init__(self, kernel_size=(71, 71), sigma=(10.0, 10.0)):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = self.compute_zero_padding(kernel_size)
        self.kernel = self.get_2d_gaussian_kernel(kernel_size, sigma)

    @staticmethod
    def compute_zero_padding(kernel_size):
        """Compute zero padding tuple."""
        padding = [((ks - 1) // 2) for ks in kernel_size]
        return padding[0], padding[1]

    def get_2d_gaussian_kernel(self, kernel_size, sigma):
        """Get the two-dimensional Gaussian filter matrix coefficients.

        Args:
            kernel_size (tuple[int]): Kernel filter size in the x and y
                                      direction. The kernel sizes
                                      should be odd and positive.
            sigma (tuple[int]): Gaussian standard deviation in
                                the x and y direction.

        Returns:
            kernel_2d (Tensor): A 2D torch tensor with gaussian filter
                                matrix coefficients.
        """
        if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
            raise TypeError('kernel_size must be a tuple of length two. Got {}'.format(kernel_size))
        if not isinstance(sigma, tuple) or len(sigma) != 2:
            raise TypeError('sigma must be a tuple of length two. Got {}'.format(sigma))
        kernel_size_x, kernel_size_y = kernel_size
        sigma_x, sigma_y = sigma
        kernel_x = self.get_1d_gaussian_kernel(kernel_size_x, sigma_x)
        kernel_y = self.get_1d_gaussian_kernel(kernel_size_y, sigma_y)
        kernel_2d = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
        return kernel_2d

    def get_1d_gaussian_kernel(self, kernel_size, sigma):
        """Get the Gaussian filter coefficients in one dimension (x or y direction).

        Args:
            kernel_size (int): Kernel filter size in x or y direction.
                               Should be odd and positive.
            sigma (float): Gaussian standard deviation in x or y direction.

        Returns:
            kernel_1d (Tensor): A 1D torch tensor with gaussian filter
                                coefficients in x or y direction.
        """
        if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
            raise TypeError('kernel_size must be an odd positive integer. Got {}'.format(kernel_size))
        kernel_1d = self.gaussian(kernel_size, sigma)
        return kernel_1d

    def gaussian(self, kernel_size, sigma):

        def gauss_arg(x):
            return -(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)
        gauss = torch.stack([torch.exp(torch.tensor(gauss_arg(x))) for x in range(kernel_size)])
        return gauss / gauss.sum()

    def forward(self, x):
        if not torch.is_tensor(x):
            raise TypeError('Input x type is not a torch.Tensor. Got {}'.format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError('Invalid input shape, we expect BxCxHxW. Got: {}'.format(x.shape))
        _, c, _, _ = x.shape
        tmp_kernel = self.kernel.to(x.device)
        kernel = tmp_kernel.repeat(c, 1, 1, 1)
        return conv2d(x, kernel, padding=self.padding, stride=1, groups=c)


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super().__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight
        if self.gan_type == 'smgan':
            self.gaussian_blur = GaussianBlur()
        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan' or self.gan_type == 'smgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """
        if self.gan_type == 'wgan':
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False, mask=None):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the target is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:
                loss = -input.mean()
        elif self.gan_type == 'smgan':
            input_height, input_width = input.shape[2:]
            mask_height, mask_width = mask.shape[2:]
            if input_height != mask_height or input_width != mask_width:
                input = F.interpolate(input, size=(mask_height, mask_width), mode='bilinear', align_corners=True)
                target_label = self.get_target_label(input, target_is_real)
            if is_disc:
                if target_is_real:
                    target_label = target_label
                else:
                    target_label = self.gaussian_blur(mask).detach() if mask.is_cuda else self.gaussian_blur(mask).detach().cpu()
                loss = self.loss(input, target_label)
            else:
                loss = self.loss(input, target_label) * mask / mask.mean()
                loss = loss.mean()
        else:
            loss = self.loss(input, target_label)
        return loss if is_disc else loss * self.loss_weight


class PerceptualVGG(nn.Module):
    """VGG network used in calculating perceptual loss.
    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.
    Args:
        layer_name_list (list[str]): According to the name in this list,
            forward function will return the corresponding features. This
            list contains the name each layer in `vgg.feature`. An example
            of this list is ['4', '10'].
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image.
            Importantly, the input feature must in the range [0, 1].
            Default: True.
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'
    """

    def __init__(self, layer_name_list, vgg_type='vgg19', use_input_norm=True, pretrained='torchvision://vgg19'):
        super().__init__()
        if pretrained.startswith('torchvision://'):
            assert vgg_type in pretrained
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        _vgg = getattr(vgg, vgg_type)()
        self.init_weights(_vgg, pretrained)
        num_layers = max(map(int, layer_name_list)) + 1
        assert len(_vgg.features) >= num_layers
        self.vgg_layers = _vgg.features[:num_layers]
        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for v in self.vgg_layers.parameters():
            v.requires_grad = False

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = {}
        for name, module in self.vgg_layers.named_children():
            x = module(x)
            if name in self.layer_name_list:
                output[name] = x.clone()
        return output

    def init_weights(self, model, pretrained):
        """Init weights.
        Args:
            model (nn.Module): Models to be inited.
            pretrained (str): Path for pretrained weights.
        """
        logger = get_root_logger()
        load_checkpoint(model, pretrained, logger=logger)


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.
    Args:
        layers_weights (dict): The weight for each layer of vgg feature for
            perceptual loss. Here is an example: {'4': 1., '9': 1., '18': 1.},
            which means the 5th, 10th and 18th feature layer will be
            extracted with weight 1.0 in calculating losses.
        layers_weights_style (dict): The weight for each layer of vgg feature
            for style loss. If set to 'None', the weights are set equal to
            the weights for perceptual loss. Default: None.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 1.0.
        norm_img (bool): If True, the image will be normed to [0, 1]. Note that
            this is different from the `use_input_norm` which norm the input in
            in forward function of vgg according to the statistics of dataset.
            Importantly, the input image must be in range [-1, 1].
        pretrained (str): Path for pretrained weights. Default:
            'torchvision://vgg19'.
        criterion (str): Criterion type. Options are 'l1' and 'mse'.
            Default: 'l1'.
    """

    def __init__(self, layer_weights, layer_weights_style=None, vgg_type='vgg19', use_input_norm=True, perceptual_weight=1.0, style_weight=1.0, norm_img=True, pretrained='torchvision://vgg19', criterion='l1'):
        super().__init__()
        self.norm_img = norm_img
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.layer_weights_style = layer_weights_style
        self.vgg = PerceptualVGG(layer_name_list=list(self.layer_weights.keys()), vgg_type=vgg_type, use_input_norm=use_input_norm, pretrained=pretrained)
        if self.layer_weights_style is not None and self.layer_weights_style != self.layer_weights:
            self.vgg_style = PerceptualVGG(layer_name_list=list(self.layer_weights_style.keys()), vgg_type=vgg_type, use_input_norm=use_input_norm, pretrained=pretrained)
        else:
            self.layer_weights_style = self.layer_weights
            self.vgg_style = None
        criterion = criterion.lower()
        if criterion == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif criterion == 'mse':
            self.criterion = torch.nn.MSELoss()
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported in this version.')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        if self.norm_img:
            x = (x + 1.0) * 0.5
            gt = (gt + 1.0) * 0.5
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None
        if self.style_weight > 0:
            if self.vgg_style is not None:
                x_features = self.vgg_style(x)
                gt_features = self.vgg_style(gt.detach())
            style_loss = 0
            for k in x_features.keys():
                style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(gt_features[k])) * self.layer_weights_style[k]
            style_loss *= self.style_weight
        else:
            style_loss = None
        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'conv': _mock_layer, 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CALayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 4, 4])], {}),
     True),
    (ConcatBlock,
     lambda: ([], {'submodule': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ContrasExtractorLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (ContrasExtractorSep,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
    (Cos,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GaussianBlur,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLPRefiner,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NonLocalBlock2D,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     False),
    (PosMLPRefiner,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (PositionalEncoding1D,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (RCABlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (RCAGroup,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (RRDB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ResBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock2,
     lambda: ([], {'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock_fft_bench,
     lambda: ([], {'n_feat': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResidualBlockNoBN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ResidualBlockwithBN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ResidualDenseBlock_5C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (ShortcutBlock,
     lambda: ([], {'submodule': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sin,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SmallUNetDiscriminatorWithSpectralNorm,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UNetDiscriminatorWithSpectralNorm,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
]

class Test_caojiezhang_CiaoSR(_paritybench_base):
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

