import sys
_module = sys.modules[__name__]
del sys
callbacks = _module
evaluation = _module
utils = _module
vis = _module
coco = _module
dataset = _module
pascal = _module
main = _module
asnet = _module
asnethm = _module
aslayer = _module
conv4d = _module
correlation = _module
feature = _module
hsnet = _module
ifsl = _module
asnet = _module
hsnet = _module
panet = _module
pfenet = _module
panet = _module
pfenet = _module
evaluation = _module
utils = _module
coco = _module
dataset = _module
pascal = _module
main = _module
asnet = _module
aslayer = _module
correlation = _module
feature = _module
learner = _module

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


import math


import torchvision.transforms as transforms


import torch.nn.functional as F


import matplotlib.pyplot as plt


from torch.utils.data import Dataset


from torchvision import transforms


from torch.utils.data import DataLoader


from functools import reduce


from torchvision.models import resnet


import torch.nn as nn


from torchvision.models import vgg


class Attention(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, heads=8, groups=4, pool_kv=False):
        super(Attention, self).__init__()
        self.heads = heads
        """
        Size of conv output = floor((input  + 2 * pad - kernel) / stride) + 1
        The second condition of `retain_dim` checks the spatial size consistency by setting input=output=0;
        Use this term with caution to check the size consistency for generic cases!
        """
        retain_dim = in_channels == out_channels and math.floor((2 * padding - kernel_size) / stride) == -1
        hidden_channels = out_channels // 2
        assert hidden_channels % self.heads == 0, 'out_channels should be divided by heads. (example: out_channels: 40, heads: 4)'
        ksz_q = 1, kernel_size, kernel_size
        str_q = 1, stride, stride
        pad_q = 0, padding, padding
        self.short_cut = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=ksz_q, stride=str_q, padding=pad_q, bias=False), nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True)) if not retain_dim else nn.Identity()
        self.qhead = nn.Conv3d(in_channels, hidden_channels, kernel_size=ksz_q, stride=str_q, padding=pad_q, bias=bias)
        ksz = (1, kernel_size, kernel_size) if pool_kv else (1, 1, 1)
        str = (1, stride, stride) if pool_kv else (1, 1, 1)
        pad = (0, padding, padding) if pool_kv else (0, 0, 0)
        self.khead = nn.Conv3d(in_channels, hidden_channels, kernel_size=ksz, stride=str, padding=pad, bias=bias)
        self.vhead = nn.Conv3d(in_channels, hidden_channels, kernel_size=ksz, stride=str, padding=pad, bias=bias)
        self.agg = nn.Sequential(nn.GroupNorm(groups, hidden_channels), nn.ReLU(inplace=True), nn.Conv3d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False), nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True))
        self.out_norm = nn.GroupNorm(groups, out_channels)

    def forward(self, input):
        x, support_mask = input
        x_ = self.short_cut(x)
        q_out = self.qhead(x)
        k_out = self.khead(x)
        v_out = self.vhead(x)
        q_h, q_w = q_out.shape[-2:]
        k_h, k_w = k_out.shape[-2:]
        q_out = rearrange(q_out, 'b (g c) t h w -> b g c t (h w)', g=self.heads)
        k_out = rearrange(k_out, 'b (g c) t h w -> b g c t (h w)', g=self.heads)
        v_out = rearrange(v_out, 'b (g c) t h w -> b g c t (h w)', g=self.heads)
        out = torch.einsum('b g c t l, b g c t m -> b g t l m', q_out, k_out)
        out = self.attn_mask(out, support_mask, spatial_size=(k_h, k_w))
        out = F.softmax(out, dim=-1)
        out = torch.einsum('b g t l m, b g c t m -> b g c t l', out, v_out)
        out = rearrange(out, 'b g c t (h w) -> b (g c) t h w', h=q_h, w=q_w)
        out = self.agg(out)
        return self.out_norm(out + x_)

    def attn_mask(self, x, mask, spatial_size):
        mask = F.interpolate(mask.float().unsqueeze(1), spatial_size, mode='bilinear', align_corners=True)
        mask = rearrange(mask, 'b 1 h w -> b 1 1 1 (h w)')
        out = x.masked_fill_(mask == 0, -1000000000.0)
        return out


class FeedForward(nn.Module):

    def __init__(self, out_channels, groups=4, size=2):
        super(FeedForward, self).__init__()
        hidden_channels = out_channels // size
        self.ff = nn.Sequential(nn.Conv3d(out_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.GroupNorm(groups, hidden_channels), nn.ReLU(inplace=True), nn.Conv3d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.out_norm = nn.GroupNorm(groups, out_channels)

    def forward(self, x):
        x_ = x
        out = self.ff(x)
        return self.out_norm(out + x_)


class AttentiveSqueezeLayer(nn.Module):
    """
    Attentive squeeze layer consisting of a global self-attention layer followed by a feed-forward MLP
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, heads=8, groups=4, pool_kv=False):
        super(AttentiveSqueezeLayer, self).__init__()
        self.attn = Attention(in_channels, out_channels, kernel_size, stride, padding, bias, heads, groups, pool_kv)
        self.ff = FeedForward(out_channels, groups)

    def forward(self, input):
        x, support_mask = input
        batch, c, qh, qw, sh, sw = x.shape
        x = rearrange(x, 'b c d t h w -> b c (d t) h w')
        out = self.attn((x, support_mask))
        out = self.ff(out)
        out = rearrange(out, 'b c (d t) h w -> b c d t h w', d=qh, t=qw)
        return out, support_mask


class CenterPivotConv4d(nn.Module):
    """ CenterPivot 4D conv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2], bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:], bias=bias, padding=padding[2:])
        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)
        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()
        bsz, inch, ha, wa, hb, wb = x.size()
        out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()
        y = out1 + out2
        return y


class AttentionLearner(nn.Module):

    def __init__(self, inch):
        super(AttentionLearner, self).__init__()

        def make_building_attentive_block(in_channel, out_channels, kernel_sizes, spt_strides, pool_kv=False):
            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                padding = ksz // 2 if ksz > 2 else 0
                building_block_layers.append(AttentiveSqueezeLayer(inch, outch, ksz, stride, padding, pool_kv=pool_kv))
            return nn.Sequential(*building_block_layers)
        self.feat_ids = list(range(4, 17))
        self.encoder_layer4 = make_building_attentive_block(inch[0], [32, 128], [5, 3], [4, 2])
        self.encoder_layer3 = make_building_attentive_block(inch[1], [32, 128], [5, 5], [4, 4], pool_kv=True)
        self.encoder_layer2 = make_building_attentive_block(inch[2], [32, 128], [5, 5], [4, 4], pool_kv=True)
        self.encoder_layer4to3 = make_building_attentive_block(128, [128, 128], [1, 2], [1, 1])
        self.encoder_layer3to2 = make_building_attentive_block(128, [128, 128], [1, 2], [1, 1])
        self.decoder1 = nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=True), nn.ReLU(), nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True), nn.ReLU())
        self.decoder2 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=True), nn.ReLU(), nn.Conv2d(64, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_query_dims(self, hypercorr, spatial_size):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = rearrange(hypercorr, 'b c d t h w -> (b h w) c d t')
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        return rearrange(hypercorr, '(b h w) c d t -> b c d t h w', b=bsz, h=hb, w=wb)

    def forward(self, hypercorr_pyramid, support_mask):
        hypercorr_sqz4 = self.encoder_layer4((hypercorr_pyramid[0], support_mask))[0]
        hypercorr_sqz3 = self.encoder_layer3((hypercorr_pyramid[1], support_mask))[0]
        hypercorr_sqz2 = self.encoder_layer2((hypercorr_pyramid[2], support_mask))[0]
        hypercorr_sqz4 = hypercorr_sqz4.mean(dim=[-1, -2], keepdim=True)
        hypercorr_sqz4 = self.interpolate_query_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3((hypercorr_mix43, support_mask))[0]
        hypercorr_mix43 = self.interpolate_query_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2((hypercorr_mix432, support_mask))[0]
        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).squeeze(-1)
        hypercorr_decoded = self.decoder1(hypercorr_encoded)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.decoder2(hypercorr_decoded)
        return logit_mask


class HPNLearner(nn.Module):

    def __init__(self, inch, way):
        super(HPNLearner, self).__init__()
        self.way = way

        def make_building_conv_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)
            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4
                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*building_block_layers)
        outch1, outch2, outch3 = 16, 64, 128
        self.encoder_layer4 = make_building_conv_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_conv_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_conv_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])
        self.encoder_layer4to3 = make_building_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(), nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU())
        self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU(), nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid):
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)
        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)
        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)
        hypercorr_decoded = self.decoder1(hypercorr_encoded)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        logit_mask = self.decoder2(hypercorr_decoded)
        logit_mask = logit_mask.view(-1, self.way, *logit_mask.shape[1:])
        return logit_mask


class PrototypeAlignmentLearner(nn.Module):

    def __init__(self, way, shot, lazy_merge=False, ignore_label=255, temperature=20):
        super(PrototypeAlignmentLearner, self).__init__()
        self.way = way
        self.shot = shot
        self.ignore_label = ignore_label
        self.temperature = temperature
        self.eps = 1e-06
        self.lazy_merge = lazy_merge

    def forward(self, qry_feat, spt_feat, spt_mask, spt_ignore_idx):
        spt_mask = spt_mask.unsqueeze(1)
        spt_mask = F.interpolate(spt_mask.float(), spt_feat.size()[-2:], mode='bilinear', align_corners=True)
        qry_feat = rearrange(qry_feat, 'b c h w -> b 1 c h w')
        spt_feat = rearrange(spt_feat, '(b n s) c h w -> b n s c h w', n=self.way, s=self.shot)
        spt_mask = rearrange(spt_mask, '(b n s) 1 h w -> b n s 1 h w', n=self.way, s=self.shot)
        if spt_ignore_idx is not None:
            spt_ignore_idx = spt_ignore_idx.unsqueeze(1)
            spt_ignore_idx = F.interpolate(spt_ignore_idx.float(), spt_feat.size()[-2:], mode='bilinear', align_corners=True)
            spt_ignore_idx = rearrange(spt_ignore_idx, '(b n s) 1 h w -> b n s 1 h w', n=self.way, s=self.shot)
            spt_mask_fg_count = torch.logical_and(spt_mask > 0, spt_ignore_idx != self.ignore_label).sum(dim=[2, -1, -2], keepdim=True).float()
            spt_mask_bg_count = torch.logical_and(spt_mask == 0, spt_ignore_idx != self.ignore_label).sum(dim=[2, -1, -2], keepdim=True).float()
            spt_mask_fg_binary = torch.logical_and(spt_mask > 0, spt_ignore_idx != self.ignore_label).float()
            spt_mask_bg_binary = torch.logical_and(spt_mask == 0, spt_ignore_idx != self.ignore_label).float()
        else:
            spt_mask_fg_count = (spt_mask > 0).sum(dim=[2, -1, -2], keepdim=True).float()
            spt_mask_bg_count = (spt_mask == 0).sum(dim=[2, -1, -2], keepdim=True).float()
            spt_mask_fg_binary = (spt_mask > 0).float()
            spt_mask_bg_binary = (spt_mask == 0).float()
        proto_fg = torch.sum(spt_feat * spt_mask_fg_binary, dim=[2, -1, -2], keepdim=True) / (spt_mask_fg_count + self.eps)
        proto_fg = proto_fg.squeeze(2)
        if self.lazy_merge:
            """
            This option enables the PANet to be trained/evaluated under the iFSL framework.
            But we leave it as an option since the authors originally propose to use the eager merge
            """
            proto_bg = torch.sum(spt_feat * spt_mask_bg_binary, dim=[2, -1, -2], keepdim=True) / (spt_mask_bg_count + self.eps)
            proto_bg = proto_bg.squeeze(2)
            """ The episodic mask scheme proposed for iFSL """
            logit_mask_fg = F.cosine_similarity(qry_feat, proto_fg, dim=2) * self.temperature
            logit_mask_bg = F.cosine_similarity(qry_feat, proto_bg, dim=2) * self.temperature
            logit_mask = torch.cat((logit_mask_bg.mean(dim=1, keepdim=True), logit_mask_fg), dim=1)
        else:
            proto_bg = torch.sum(spt_feat * spt_mask_bg_binary, dim=[1, 2, -1, -2], keepdim=True) / (spt_mask_bg_count.sum(dim=1, keepdim=True) + self.eps)
            proto_bg = proto_bg.squeeze(2)
            proto = torch.cat((proto_bg, proto_fg), dim=1)
            logit_mask = F.cosine_similarity(qry_feat, proto, dim=2) * self.temperature
        return logit_mask


class PFENetLearner(nn.Module):

    def __init__(self, way, shot, ppm_scales=[50, 25, 13, 7]):
        super(PFENetLearner, self).__init__()
        self.way = way
        self.shot = shot
        self.eps = 1e-06
        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(nn.AdaptiveAvgPool2d(bin))
        reduce_dim = 256
        fea_dim = 1024 + 512
        factor = 1
        mask_add_num = 1
        classes = 2
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False), nn.ReLU(inplace=True)))
            self.beta_conv.append(nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True)))
            self.inner_cls.append(nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Dropout2d(p=0.1), nn.Conv2d(reduce_dim, classes, kernel_size=1)))
        self.init_merge = nn.ModuleList(self.init_merge)
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)
        self.res1 = nn.Sequential(nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False), nn.ReLU(inplace=True))
        self.res2 = nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True))
        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins) - 1):
            self.alpha_conv.append(nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False), nn.ReLU()))
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
        self.cls = nn.Sequential(nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True), nn.Dropout2d(p=0.1), nn.Conv2d(reduce_dim, classes, kernel_size=1))

    def forward(self, qry_feat, spt_feat, corr_query_mask):
        out_list = []
        pyramid_feat_list = []
        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(qry_feat.shape[2] * tmp_bin)
                qry_feat_bin = nn.AdaptiveAvgPool2d(bin)(qry_feat)
            else:
                bin = tmp_bin
                qry_feat_bin = self.avgpool_list[idx](qry_feat)
            spt_feat_bin = spt_feat.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            if self.way > 1:
                qry_feat_bin = qry_feat_bin.repeat_interleave(self.way, dim=0)
                corr_mask_bin = corr_mask_bin.repeat_interleave(self.way, dim=0)
            merge_feat_bin = torch.cat([qry_feat_bin, spt_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)
            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin
            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(qry_feat.size(2), qry_feat.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)
        qry_feat = torch.cat(pyramid_feat_list, 1)
        qry_feat = self.res1(qry_feat)
        qry_feat = self.res2(qry_feat) + qry_feat
        logit_mask = self.cls(qry_feat)
        logit_mask = logit_mask.view(-1, self.way, *logit_mask.shape[1:])
        return logit_mask

