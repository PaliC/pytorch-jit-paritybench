import sys
_module = sys.modules[__name__]
del sys
data = _module
helper = _module
main = _module
models = _module
base = _module
pointmixer = _module
pointmlp = _module
pointtransformer = _module
utils = _module
logger = _module
misc = _module
progress = _module
bar = _module
counter = _module
helpers = _module
spinner = _module
setup = _module
test_progress = _module
voting = _module
pointnet2_ops = _module
_version = _module
pointnet2_modules = _module
pointnet2_utils = _module
pointnet2_modules = _module
pointnet2_utils = _module
setup = _module
pointops2 = _module
functions = _module
pointops = _module
pointops2 = _module
pointops_ablation = _module
setup = _module
src = _module
dataset = _module
loader_s3dis = _module
loader_scannet = _module
loader_scannet_js = _module
data_util = _module
transform = _module
transform_scannet = _module
voxelize = _module
lib = _module
pointops = _module
setup = _module
knnquery = _module
knnquery_heap = _module
pointops = _module
pointops2 = _module
pointops_ablation = _module
setup = _module
model = _module
net_pointmixer = _module
get_network = _module
hier = _module
inter = _module
pointmixer = _module
pointtransformer = _module
test_pl = _module
test_split_save = _module
train_pl = _module
common_util = _module
logger = _module
my_args = _module

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


from torch.utils.data import Dataset


import torch


import torch.nn.functional as F


import logging


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import CosineAnnealingLR


import sklearn.metrics as metrics


import torch.nn as nn


import matplotlib.pyplot as plt


import time


import math


import random


import torch.nn.init as init


from torch.autograd import Variable


from typing import List


from typing import Optional


from typing import Tuple


import warnings


from torch.autograd import Function


from typing import *


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from copy import deepcopy


import scipy


import scipy.ndimage


import scipy.interpolate


import torch.distributed as dist


from torch import nn


from torch.nn.modules.conv import _ConvNd


from torch.nn.modules.batchnorm import _BatchNorm


import torch.nn.init as initer


from scipy.spatial.distance import cdist


import matplotlib


from matplotlib.image import imread


from matplotlib.lines import Line2D


import scipy.stats


from collections import OrderedDict


class IdentityInterSetLayer(nn.Module):

    def __init__(self, in_planes, share_planes, nsample=16, use_xyz=False):
        super().__init__()
        self.in_planes = in_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.use_xyz = use_xyz

    def forward(self, input):
        x = input[0]
        return x


class BilinearFeedForward(nn.Module):

    def __init__(self, in_planes1, in_planes2, out_planes):
        super().__init__()
        self.bilinear = nn.Bilinear(in_planes1, in_planes2, out_planes)

    def forward(self, x):
        x = x.contiguous()
        x = self.bilinear(x, x)
        return x


class PointMixerIntraSetLayerPaper(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.channelMixMLPs01 = nn.Sequential(nn.Linear(3 + in_planes, nsample), nn.ReLU(inplace=True), BilinearFeedForward(nsample, nsample, nsample))
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.Sequential(Rearrange('n k c -> n c k'), nn.BatchNorm1d(3), Rearrange('n c k -> n k c')), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.shrink_p = nn.Sequential(Rearrange('n k (a b) -> n k a b', b=nsample), Reduce('n k a b -> n k b', 'sum', b=nsample))
        self.channelMixMLPs02 = nn.Sequential(Rearrange('n k c -> n c k'), nn.Conv1d(nsample + nsample, mid_planes, kernel_size=1, bias=False), nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True), nn.Conv1d(mid_planes, mid_planes // share_planes, kernel_size=1, bias=False), nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True), nn.Conv1d(mid_planes // share_planes, out_planes // share_planes, kernel_size=1), Rearrange('n c k -> n k c'))
        self.channelMixMLPs03 = nn.Linear(in_planes, out_planes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        p, x, o = pxo
        x_knn, knn_idx = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)
        p_r = x_knn[:, :, 0:3]
        energy = self.channelMixMLPs01(x_knn)
        p_embed = self.linear_p(p_r)
        p_embed_shrink = self.shrink_p(p_embed)
        energy = torch.cat([energy, p_embed_shrink], dim=-1)
        energy = self.channelMixMLPs02(energy)
        w = self.softmax(energy)
        x_v = self.channelMixMLPs03(x)
        n = knn_idx.shape[0]
        knn_idx_flatten = knn_idx.flatten()
        x_v = x_v[knn_idx_flatten, :].view(n, self.nsample, -1)
        n, nsample, out_planes = x_v.shape
        x_knn = (x_v + p_embed).view(n, nsample, self.share_planes, out_planes // self.share_planes)
        x_knn = x_knn * w.unsqueeze(2)
        x_knn = x_knn.reshape(n, nsample, out_planes)
        x = x_knn.sum(1)
        return x, x_knn, knn_idx, p_r


class PointMixerIntraSetLayerPaperv3(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.channelMixMLPs01 = nn.Sequential(nn.Linear(3 + in_planes, nsample), nn.ReLU(inplace=True), BilinearFeedForward(nsample, nsample, nsample))
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.shrink_p = nn.Sequential(Rearrange('n k (a b) -> n k a b', b=nsample), Reduce('n k a b -> (n k) b', 'sum', b=nsample))
        self.channelMixMLPs02 = nn.Sequential(nn.Linear(nsample + nsample, mid_planes, bias=False), nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True), nn.Linear(mid_planes, mid_planes // share_planes, bias=False), nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True), nn.Linear(mid_planes // share_planes, out_planes // share_planes, bias=True), Rearrange('(n k) c -> n k c', k=nsample))
        self.channelMixMLPs03 = nn.Linear(in_planes, out_planes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) ->torch.Tensor:
        p, x, o = pxo
        x_knn, knn_idx = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)
        p_r = x_knn[:, :, 0:3]
        x_knn_flatten = rearrange(x_knn, 'n k c -> (n k) c')
        energy_flatten = self.channelMixMLPs01(x_knn_flatten)
        n = p_r.shape[0]
        p_embed = self.linear_p(p_r.view(-1, 3))
        p_embed = p_embed.view(n, self.nsample, -1)
        p_embed_shrink_flatten = self.shrink_p(p_embed)
        energy_flatten = torch.cat([energy_flatten, p_embed_shrink_flatten], dim=-1)
        energy = self.channelMixMLPs02(energy_flatten)
        w = self.softmax(energy)
        x_v = self.channelMixMLPs03(x)
        n = knn_idx.shape[0]
        knn_idx_flatten = knn_idx.flatten()
        x_v = x_v[knn_idx_flatten, :].view(n, self.nsample, -1)
        n, nsample, out_planes = x_v.shape
        x_knn = (x_v + p_embed).view(n, nsample, self.share_planes, out_planes // self.share_planes)
        x_knn = x_knn * w.unsqueeze(2)
        x_knn = x_knn.reshape(n, nsample, out_planes)
        x = x_knn.sum(1)
        return x, x_knn, knn_idx, p_r


class PointMixerInterSetLayerGroupMLPv3(nn.Module):

    def __init__(self, in_planes, share_planes, nsample=16, use_xyz=False):
        super().__init__()
        self.share_planes = share_planes
        self.linear = nn.Linear(in_planes, in_planes // share_planes)
        self.linear_x = nn.Linear(in_planes, in_planes // share_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3, bias=False), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, in_planes))

    def forward(self, input):
        x, x_knn, knn_idx, p_r = input
        N = x_knn.shape[0]
        with torch.no_grad():
            knn_idx_flatten = rearrange(knn_idx, 'n k -> (n k) 1')
        p_r_flatten = rearrange(p_r, 'n k c -> (n k) c')
        p_embed_flatten = self.linear_p(p_r_flatten)
        x_knn_flatten = rearrange(x_knn, 'n k c -> (n k) c')
        x_knn_flatten_shrink = self.linear(x_knn_flatten + p_embed_flatten)
        x_knn_prob_flatten_shrink = scatter_softmax(x_knn_flatten_shrink, knn_idx_flatten, dim=0)
        x_v_knn_flatten = self.linear_x(x_knn_flatten)
        x_knn_weighted_flatten = x_v_knn_flatten * x_knn_prob_flatten_shrink
        residual = scatter_sum(x_knn_weighted_flatten, knn_idx_flatten, dim=0, dim_size=N)
        residual = repeat(residual, 'n c -> n (repeat c)', repeat=self.share_planes)
        return x + residual


class PointMixerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, use_xyz=False, intraLayer='PointMixerIntraSetLayer', interLayer='PointMixerInterSetLayer'):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = nn.Sequential(globals()[intraLayer](planes, planes, share_planes, nsample), globals()[interLayer](in_planes, share_planes, nsample))
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x = x + identity
        x = self.relu(x)
        return [p, x, o]


class PointMixerBlockPaperInterSetLayerGroupMLPv3(PointMixerBlock):
    expansion = 1
    intraLayer = PointMixerIntraSetLayerPaper
    interLayer = PointMixerInterSetLayerGroupMLPv3


class SymmetricTransitionUpBlock(nn.Module):

    def __init__(self, in_planes, out_planes, nsample):
        super().__init__()
        self.nsample = nsample
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes, bias=False), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes, bias=False), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes, bias=False), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.channel_shrinker = nn.Sequential(nn.Linear(in_planes + 3, in_planes, bias=False), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True), nn.Linear(in_planes, 1))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            y = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            knn_idx = pointops.knnquery(self.nsample, p1, p2, o1, o2)[0].long()
            with torch.no_grad():
                knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
            p_r = p1[knn_idx_flatten, :].view(len(p2), self.nsample, 3) - p2.unsqueeze(1)
            x2_knn = x2.view(len(p2), 1, -1).repeat(1, self.nsample, 1)
            x2_knn = torch.cat([p_r, x2_knn], dim=-1)
            with torch.no_grad():
                knn_idx_flatten = knn_idx_flatten.unsqueeze(-1)
            x2_knn_flatten = rearrange(x2_knn, 'm k c -> (m k) c')
            x2_knn_flatten_shrink = self.channel_shrinker(x2_knn_flatten)
            x2_knn_prob_flatten_shrink = scatter_softmax(x2_knn_flatten_shrink, knn_idx_flatten, dim=0)
            x2_knn_prob_shrink = rearrange(x2_knn_prob_flatten_shrink, '(m k) 1 -> m k 1', k=self.nsample)
            up_x2_weighted = self.linear2(x2).unsqueeze(1) * x2_knn_prob_shrink
            up_x2_weighted_flatten = rearrange(up_x2_weighted, 'm k c -> (m k) c')
            up_x2 = scatter_sum(up_x2_weighted_flatten, knn_idx_flatten, dim=0, dim_size=len(p1))
            y = self.linear1(x1) + up_x2
        return y


class SymmetricTransitionDownBlockPaperv3(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes, bias=False), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.channel_shrinker = nn.Sequential(nn.Linear(3 + in_planes, in_planes, bias=False), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True), nn.Linear(in_planes, 1))
        else:
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes, bias=False), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))

    def forward(self, pxo):
        p, x, o = pxo
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)
            n_p = p[idx.long(), :]
            x_knn, knn_idx = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True, return_idx=True)
            m, k, c = x_knn.shape
            x_knn_flatten = rearrange(x_knn, 'm k c -> (m k) c')
            x_knn_flatten_shrink = self.channel_shrinker(x_knn_flatten)
            x_knn_shrink = rearrange(x_knn_flatten_shrink, '(m k) c -> m k c', m=m, k=k)
            x_knn_prob_shrink = F.softmax(x_knn_shrink, dim=1)
            y = self.linear2(x)
            with torch.no_grad():
                knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
            y_knn_flatten = y[knn_idx_flatten, :]
            y_knn = rearrange(y_knn_flatten, '(m k) c -> m k c', m=m, k=k)
            x_knn_weighted = y_knn * x_knn_prob_shrink
            y = torch.sum(x_knn_weighted, dim=1).contiguous()
            p, o = n_p, n_o
        else:
            y = self.linear2(x)
        return [p, y, o]


class PointMixerClsNet(nn.Module):

    def __init__(self, blocks, c=3, k=40, nsample=[16, 16, 16, 16, 16], stride=[1, 2, 2, 2, 2], planes=[32, 64, 128, 256, 512], share_planes=8, mixerblock='PointMixerBlockPaperInterSetLayerGroupMLPv3', transup='SymmetricTransitionUpBlock', transdown='SymmetricTransitionDownBlockPaperv3', use_avgmax=False):
        super().__init__()
        self.c = c
        self.mixerblock = mixerblock
        self.transup = transup
        self.transdown = transdown
        self.in_planes = c
        self.use_avgmax = use_avgmax
        assert stride[0] == 1, 'or you will meet errors.'
        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])
        self.emb_dim = planes[4]
        cls_in_planes = int(2 * planes[4]) if use_avgmax else planes[4]
        self.cls = nn.Sequential(nn.Linear(cls_in_planes, planes[4]), nn.BatchNorm1d(planes[4]), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(planes[4], planes[4] // 2), nn.BatchNorm1d(planes[4] // 2), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(planes[4] // 2, k))

    def _make_enc(self, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(globals()[self.transdown](self.in_planes, planes, stride, nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(globals()[self.mixerblock](self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def po_from_batched_pcd(self, pcd):
        B, C, N = pcd.shape
        assert C == 3
        p = pcd.transpose(1, 2).contiguous().view(-1, 3)
        o = torch.IntTensor([(N * i) for i in range(1, B + 1)])
        return p, o

    def forward(self, pcd):
        B = pcd.shape[0]
        p0, o0 = self.po_from_batched_pcd(pcd)
        x0 = p0
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = x5.view(B, -1, self.emb_dim).transpose(1, 2).contiguous()
        if self.use_avgmax:
            x5_avg = F.adaptive_avg_pool1d(x5, 1).squeeze(dim=-1)
            x5_max = F.adaptive_max_pool1d(x5, 1).squeeze(dim=-1)
            x5 = torch.cat([x5_avg, x5_max], dim=-1)
        else:
            x5 = F.adaptive_max_pool1d(x5, 1).squeeze(dim=-1)
        x5 = self.cls(x5).view(B, -1)
        return x5


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):

    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize='center', **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ['center', 'anchor']:
            None
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)
        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        if self.normalize is not None:
            if self.normalize == 'center':
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == 'anchor':
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-05)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


class ConvBNReLU1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias), nn.BatchNorm1d(out_channels), self.act)

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):

    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion), kernel_size=kernel_size, groups=groups, bias=bias), nn.BatchNorm1d(int(channel * res_expansion)), self.act)
        if groups > 1:
            self.net2 = nn.Sequential(nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel, kernel_size=kernel_size, groups=groups, bias=bias), nn.BatchNorm1d(channel), self.act, nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, bias=bias), nn.BatchNorm1d(channel))
        else:
            self.net2 = nn.Sequential(nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel, kernel_size=kernel_size, bias=bias), nn.BatchNorm1d(channel))

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):

    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):

    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        return self.operation(x)


class Model(nn.Module):

    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0, activation='relu', bias=True, use_xyz=True, normalize='center', dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2], k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), 'Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers.'
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)
            self.local_grouper_list.append(local_grouper)
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)
            last_channel = out_channel
        self.act = get_activation(activation)
        self.classifier = nn.Sequential(nn.Linear(last_channel, 512), nn.BatchNorm1d(512), self.act, nn.Dropout(0.5), nn.Linear(512, 256), nn.BatchNorm1d(256), self.act, nn.Dropout(0.5), nn.Linear(256, self.class_num))

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        x = self.embedding(x)
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            x = self.pre_blocks_list[i](x)
            x = self.pos_blocks_list[i](x)
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        x = self.classifier(x)
        return x


class PointNet2Layer(nn.Module):

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.mlp = nn.Sequential(nn.Linear(in_planes + 3, out_planes // 2, bias=False), Rearrange('n k c -> n c k'), nn.BatchNorm1d(out_planes // 2), Rearrange('n c k -> n k c'), nn.ReLU(inplace=True), nn.Linear(out_planes // 2, out_planes))

    def forward(self, pxo) ->torch.Tensor:
        p, x, o = pxo
        x_knn = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True)
        x_knn = self.mlp(x_knn)
        x = x_knn.max(dim=1)[0]
        return x


class PointMixerIntraSetLayerPaperLinear(PointMixerIntraSetLayerPaper):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super(PointMixerIntraSetLayerPaperLinear, self).__init__(in_planes, out_planes, share_planes, nsample)
        self.channelMixMLPs01 = nn.Sequential(nn.Linear(3 + in_planes, nsample), nn.ReLU(inplace=True), nn.Linear(nsample, nsample))


class PointMixerBlockPaperIdentityInterSetLayerGroupMLPv3(PointMixerBlockPaperInterSetLayerGroupMLPv3):
    interLayer = IdentityInterSetLayer


class PointNet2Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointNet2Block, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.layer2 = PointNet2Layer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.layer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class Transformer(nn.Module):

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True), nn.Linear(mid_planes, mid_planes // share_planes), nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True), nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) ->torch.Tensor:
        p, x, o = pxo
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        for i, layer in enumerate(self.linear_p):
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)
        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
        return x


class TransitionDown(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)
            n_p = p[idx.long(), :]
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))
            x = self.pool(x).squeeze(-1)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))
        return [p, x, o]


class TransitionUp(nn.Module):

    def __init__(self, in_planes, out_planes, nsample):
        super().__init__()
        self.nsample = nsample
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))

    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = Transformer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointTransformerClsNet(nn.Module):

    def __init__(self, blocks, c=3, k=40, nsample=[16, 16, 16, 16, 16], stride=[1, 2, 2, 2, 2], planes=[32, 64, 128, 256, 512], share_planes=8, transformerblock='Bottleneck', transup='TransitionUp', transdown='TransitionDown', use_avgmax=False):
        super().__init__()
        self.c = c
        self.transformerblock = transformerblock
        self.transup = transup
        self.transdown = transdown
        self.in_planes = c
        self.use_avgmax = use_avgmax
        assert stride[0] == 1, 'or you will meet errors.'
        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])
        self.emb_dim = planes[4]
        cls_in_planes = int(2 * planes[4]) if use_avgmax else planes[4]
        self.cls = nn.Sequential(nn.Linear(cls_in_planes, planes[4]), nn.BatchNorm1d(planes[4]), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(planes[4], planes[4] // 2), nn.BatchNorm1d(planes[4] // 2), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(planes[4] // 2, k))

    def _make_enc(self, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(globals()[self.transdown](self.in_planes, planes, stride, nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(globals()[self.transformerblock](self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def po_from_batched_pcd(self, pcd):
        B, C, N = pcd.shape
        assert C == 3
        p = pcd.transpose(1, 2).contiguous().view(-1, 3)
        o = torch.IntTensor([(N * i) for i in range(1, B + 1)])
        return p, o

    def forward(self, pcd):
        B = pcd.shape[0]
        p0, o0 = self.po_from_batched_pcd(pcd)
        x0 = p0
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = x5.view(B, -1, self.emb_dim).transpose(1, 2).contiguous()
        if self.use_avgmax:
            x5_avg = F.adaptive_avg_pool1d(x5, 1).squeeze(dim=-1)
            x5_max = F.adaptive_max_pool1d(x5, 1).squeeze(dim=-1)
            x5 = torch.cat([x5_avg, x5_max], dim=-1)
        else:
            x5 = F.adaptive_max_pool1d(x5, 1).squeeze(dim=-1)
        x5 = self.cls(x5).view(B, -1)
        return x5


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz: 'torch.Tensor', features: 'Optional[torch.Tensor]') ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \\sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)).transpose(1, 2).contiguous() if self.npoint is not None else None
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


def build_shared_mlp(mlp_spec: 'List[int]', bn: 'bool'=True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        super(PointnetSAModuleMSG, self).__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(build_shared_mlp(mlp_spec, bn))


class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True):
        super(PointnetSAModule, self).__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz)


class PointnetFPModule(nn.Module):
    """Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        """
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-08)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*(known_feats.size()[0:2] + [unknown.size(1)]))
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return new_features.squeeze(-1)


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius: 'float', nsample: 'int', xyz: 'torch.Tensor', new_xyz: 'torch.Tensor') ->torch.Tensor:
        """
        input: radius: float, radius of the balls
               nsample: int, maximum number of features in the balls
               xyz: torch.Tensor, (b, n, 3) xyz coordinates of the features
               new_xyz: torch.Tensor, (b, m, 3) centers of the ball query
        output: (b, m, nsample) tensor with the indicies of the features that form the query balls
        """
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        b, n, _ = xyz.size()
        m = new_xyz.size(1)
        idx = torch.IntTensor(b, m, nsample).zero_()
        pointops_cuda.ballquery_cuda(b, n, m, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ballquery = BallQuery.apply


class Grouping(Function):

    @staticmethod
    def forward(ctx, input, idx):
        """
        input: input: (n, c), idx : (m, nsample)
        output: (m, nsample, c)
        """
        assert input.is_contiguous() and idx.is_contiguous()
        m, nsample, n, c = idx.shape[0], idx.shape[1], input.shape[0], input.shape[1]
        output = torch.FloatTensor(m, nsample, c)
        pointops_cuda.grouping_forward_cuda(m, nsample, c, input, idx, output)
        ctx.n = n
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (m, c, nsample)
        output: (n, c), None
        """
        n = ctx.n
        idx, = ctx.saved_tensors
        m, nsample, c = grad_output.shape
        grad_input = torch.FloatTensor(n, c).zero_()
        pointops_cuda.grouping_backward_cuda(m, nsample, c, grad_output, idx, grad_input)
        return grad_input, None


grouping = Grouping.apply


class KNNQuery_Heap(Function):

    @staticmethod
    def forward(ctx, nsample: 'int', xyz: 'torch.Tensor', new_xyz: 'torch.Tensor'=None) ->Tuple[torch.Tensor]:
        """
        KNN Indexing
        input: nsample: int32, Number of neighbor
               xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
            output: idx: (b, m, nsample)
                   ( dist2: (b, m, nsample) )
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous()
        assert new_xyz.is_contiguous()
        b, m, _ = new_xyz.size()
        n = xyz.size(1)
        idx = torch.IntTensor(b, m, nsample).zero_()
        dist2 = torch.FloatTensor(b, m, nsample).zero_()
        pointops_cuda.knnquery_heap_cuda(b, n, m, nsample, xyz, new_xyz, idx, dist2)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None


knnquery_heap = KNNQuery_Heap.apply


class QueryAndGroup(nn.Module):
    """
    Groups with a ball query of radius
    parameters:
        radius: float32, Radius of ball
        nsample: int32, Maximum number of features to gather in the ball
    """

    def __init__(self, radius=None, nsample=32, use_xyz=True):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: 'torch.Tensor', new_xyz: 'torch.Tensor'=None, features: 'torch.Tensor'=None, idx: 'torch.Tensor'=None) ->torch.Tensor:
        """
        input: xyz: (b, n, 3) coordinates of the features
               new_xyz: (b, m, 3) centriods
               features: (b, c, n)
               idx: idx of neighbors
               # idxs: (b, n)
        output: new_features: (b, c+3, m, nsample)
              #  grouped_idxs: (b, m, nsample)
        """
        if new_xyz is None:
            new_xyz = xyz
        if idx is None:
            if self.radius is not None:
                idx = ballquery(self.radius, self.nsample, xyz, new_xyz)
            else:
                idx = knnquery_heap(self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping(xyz_trans, idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if features is not None:
            grouped_features = grouping(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz
        return new_features


class GroupAll(nn.Module):
    """
    Groups all features
    """

    def __init__(self, use_xyz: 'bool'=True):
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: 'torch.Tensor', new_xyz: 'torch.Tensor', features: 'torch.Tensor'=None) ->Tuple[torch.Tensor]:
        """
        input: xyz: (b, n, 3) coordinates of the features
               new_xyz: ignored torch
               features: (b, c, n) descriptors of the features
        output: new_features: (b, c+3, 1, N) tensor
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        return new_features


class TransitionDownBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)
            n_p = p[idx.long(), :]
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))
            x = self.pool(x).squeeze(-1)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))
        return [p, x, o]


class SymmetricTransitionDownBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, nsample):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes, bias=False), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.channel_shrinker = nn.Sequential(nn.Linear(3 + in_planes, in_planes, bias=False), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True), nn.Linear(in_planes, 1))
        else:
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes, bias=False), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))

    def forward(self, pxo):
        p, x, o = pxo
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)
            n_p = p[idx.long(), :]
            x_knn, knn_idx = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True, return_idx=True)
            m, k, c = x_knn.shape
            x_knn_flatten = rearrange(x_knn, 'm k c -> (m k) c')
            x_knn_flatten_shrink = self.channel_shrinker(x_knn_flatten)
            x_knn_shrink = rearrange(x_knn_flatten_shrink, '(m k) c -> m k c', m=m, k=k)
            x_knn_prob_shrink = F.softmax(x_knn_shrink, dim=1)
            y = self.linear2(x)
            with torch.no_grad():
                knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
            y_knn_flatten = y[knn_idx_flatten, :]
            y_knn = rearrange(y_knn_flatten, '(m k) c -> m k c', m=m, k=k)
            x_knn_weighted = y_knn * x_knn_prob_shrink
            y = torch.sum(x_knn_weighted, dim=1).contiguous()
            p, o = n_p, n_o
        else:
            y = self.linear2(x)
        return [p, y, o]


class NoInterSetLayer(nn.Module):

    def __init__(self, in_planes, nsample=16, use_xyz=False):
        super().__init__()

    def forward(self, input):
        x, x_knn, knn_idx, p_r = input
        return x


class PointMixerInterSetLayer(nn.Module):

    def __init__(self, in_planes, share_planes, nsample):
        super().__init__()
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear = nn.Sequential(nn.Linear(in_planes + in_planes, in_planes // share_planes), nn.ReLU(inplace=True))
        self.linear_x = nn.Sequential(nn.Linear(in_planes, in_planes // share_planes), nn.ReLU(inplace=True))
        self.linear_p = nn.Sequential(nn.Linear(3, 3, bias=False), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, in_planes))

    def forward(self, input):
        x, x_knn, knn_idx, p_r = input
        N = x_knn.shape[0]
        with torch.no_grad():
            knn_idx_flatten = rearrange(knn_idx, 'n k -> (n k) 1')
        p_r_flatten = rearrange(p_r, 'n k c -> (n k) c')
        p_embed_flatten = self.linear_p(p_r_flatten)
        x_knn_flatten = rearrange(x_knn, 'n k c -> (n k) c')
        x_knn_flatten_shrink = self.linear(torch.cat([p_embed_flatten, x_knn_flatten], dim=1))
        x_knn_prob_flatten_shrink = scatter_softmax(x_knn_flatten_shrink, knn_idx_flatten, dim=0)
        x_v_knn_flatten = self.linear_x(x_knn_flatten)
        x_knn_weighted_flatten = x_v_knn_flatten * x_knn_prob_flatten_shrink
        residual = scatter_sum(x_knn_weighted_flatten, knn_idx_flatten, dim=0, dim_size=N)
        residual = repeat(residual, 'n c -> n (repeat c)', repeat=self.share_planes)
        return x + residual


class NoIntraSetLayer(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.out_planes = out_planes
        self.nsample = nsample

    def forward(self, pxo) ->torch.Tensor:
        p, x, o = pxo
        x_knn, knn_idx = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)
        p_r = x_knn[:, :, 0:3]
        x_knn = x_knn[:, :, 3:]
        return x, x_knn, knn_idx, p_r


class PointMixerIntraSetLayer(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.channelMixMLPs01 = nn.Sequential(nn.Linear(3 + in_planes, nsample), nn.ReLU(inplace=True), BilinearFeedForward(nsample, nsample, nsample))
        self.linear_p = nn.Sequential(nn.Linear(3, 3, bias=False), nn.Sequential(Rearrange('n k c -> n c k'), nn.BatchNorm1d(3), Rearrange('n c k -> n k c')), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.shrink_p = nn.Sequential(Rearrange('n k (a b) -> n k a b', b=nsample), Reduce('n k a b -> n k b', 'sum', b=nsample))
        self.channelMixMLPs02 = nn.Sequential(Rearrange('n k c -> n c k'), nn.Conv1d(nsample + nsample, mid_planes, kernel_size=1, bias=False), nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True), nn.Conv1d(mid_planes, mid_planes // share_planes, kernel_size=1, bias=False), nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True), nn.Conv1d(mid_planes // share_planes, out_planes // share_planes, kernel_size=1), Rearrange('n c k -> n k c'))
        self.channelMixMLPs03 = nn.Linear(in_planes, out_planes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) ->torch.Tensor:
        p, x, o = pxo
        x_knn, knn_idx = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)
        p_r = x_knn[:, :, 0:3]
        energy = self.channelMixMLPs01(x_knn)
        p_embed = self.linear_p(p_r)
        p_embed_shrink = self.shrink_p(p_embed)
        energy = torch.cat([energy, p_embed_shrink], dim=-1)
        energy = self.channelMixMLPs02(energy)
        w = self.softmax(energy)
        x_v = self.channelMixMLPs03(x)
        n = knn_idx.shape[0]
        knn_idx_flatten = knn_idx.flatten()
        x_v = x_v[knn_idx_flatten, :].view(n, self.nsample, -1)
        n, nsample, out_planes = x_v.shape
        x_knn = (x_v + p_embed).view(n, nsample, self.share_planes, out_planes // self.share_planes)
        x_knn = x_knn * w.unsqueeze(2)
        x_knn = x_knn.reshape(n, nsample, out_planes)
        x = x_knn.sum(1)
        return x, x_knn, knn_idx, p_r


class PointMixerSegNet(nn.Module):
    mixerblock = PointMixerBlock

    def __init__(self, block, blocks, c=6, k=13, nsample=[8, 16, 16, 16, 16], stride=[1, 4, 4, 4, 4], intraLayer='PointMixerIntraSetLayer', interLayer='PointMixerInterSetLayer', transup='SymmetricTransitionUpBlock', transdown='TransitionDownBlock'):
        super().__init__()
        self.c = c
        self.intraLayer = intraLayer
        self.interLayer = interLayer
        self.transup = transup
        self.transdown = transdown
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        assert stride[0] == 1, 'or you will meet errors.'
        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])
        self.dec5 = self._make_dec(planes[4], 2, share_planes, nsample=nsample[4], is_head=True)
        self.dec4 = self._make_dec(planes[3], 2, share_planes, nsample=nsample[3])
        self.dec3 = self._make_dec(planes[2], 2, share_planes, nsample=nsample[2])
        self.dec2 = self._make_dec(planes[1], 2, share_planes, nsample=nsample[1])
        self.dec1 = self._make_dec(planes[0], 2, share_planes, nsample=nsample[0])
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = []
        layers.append(globals()[self.transdown](in_planes=self.in_planes, out_planes=planes, stride=stride, nsample=nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(self.mixerblock(in_planes=self.in_planes, planes=self.in_planes, share_planes=share_planes, nsample=nsample, intraLayer=self.intraLayer, interLayer=self.interLayer))
        return nn.Sequential(*layers)

    def _make_dec(self, planes, blocks, share_planes, nsample, is_head=False):
        layers = []
        layers.append(globals()[self.transup](in_planes=self.in_planes, out_planes=None if is_head else planes, nsample=nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(self.mixerblock(in_planes=self.in_planes, planes=self.in_planes, share_planes=share_planes, nsample=nsample, intraLayer=self.intraLayer, interLayer=self.interLayer))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x


class PointTransformerIntraSetLayer(nn.Module):

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.Sequential(Rearrange('n k c -> n c k'), nn.BatchNorm1d(3), Rearrange('n c k -> n k c')), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_w = nn.Sequential(nn.Sequential(Rearrange('n k c -> n c k'), nn.BatchNorm1d(mid_planes), Rearrange('n c k -> n k c')), nn.ReLU(inplace=True), nn.Linear(mid_planes, mid_planes // share_planes), nn.Sequential(Rearrange('n k c -> n c k'), nn.BatchNorm1d(mid_planes // share_planes), Rearrange('n c k -> n k c')), nn.ReLU(inplace=True), nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) ->torch.Tensor:
        p, x, o = pxo
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        channel_length = x_k.shape[1]
        x_kv = torch.cat([x_k, x_v], dim=1)
        x_kv, knn_idx = pointops.queryandgroup(self.nsample, p, p, x_kv, None, o, o, use_xyz=True, return_idx=True)
        p_r = x_kv[:, :, 0:3]
        x_k = x_kv[:, :, 3:3 + channel_length]
        x_v = x_kv[:, :, -channel_length:]
        p_embed = self.linear_p(p_r)
        w = x_k - x_q.unsqueeze(1) + p_embed.view(p_embed.shape[0], p_embed.shape[1], self.out_planes // self.mid_planes, self.mid_planes).sum(2)
        w = self.linear_w(w)
        w = self.softmax(w)
        n, nsample, c = x_v.shape
        s = self.share_planes
        x_knn = (x_v + p_embed).view(n, self.nsample, s, c // s)
        x_knn = x_knn * w.unsqueeze(2)
        x_knn = x_knn.reshape(n, self.nsample, c)
        x = x_knn.sum(1)
        return x, x_knn, knn_idx, p_r


class PointTransformerBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16, use_xyz=False, intraLayer='PointTransformerIntraSetLayer', interLayer='NoInterSetLayer'):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = nn.Sequential(globals()[intraLayer](planes, planes, share_planes, nsample), globals()[interLayer](in_planes, share_planes, nsample))
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x = x + identity
        x = self.relu(x)
        return [p, x, o]


class PointTransformerSegNet(nn.Module):
    mixerblock = PointTransformerBlock

    def __init__(self, block, blocks, c=6, k=13, nsample=[8, 16, 16, 16, 16], stride=[1, 4, 4, 4, 4], intraLayer='PointTransformerIntraSetLayer', interLayer='NoInterSetLayer', transup='TransitionUp', transdown='TransitionDownBlock'):
        super().__init__()
        self.c = c
        self.intraLayer = intraLayer
        self.interLayer = interLayer
        self.transup = transup
        self.transdown = transdown
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        assert stride[0] == 1, 'or you will meet errors.'
        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])
        self.dec5 = self._make_dec(planes[4], 2, share_planes, nsample=nsample[4], is_head=True)
        self.dec4 = self._make_dec(planes[3], 2, share_planes, nsample=nsample[3])
        self.dec3 = self._make_dec(planes[2], 2, share_planes, nsample=nsample[2])
        self.dec2 = self._make_dec(planes[1], 2, share_planes, nsample=nsample[1])
        self.dec1 = self._make_dec(planes[0], 2, share_planes, nsample=nsample[0])
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))

    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = []
        layers.append(globals()[self.transdown](in_planes=self.in_planes, out_planes=planes, stride=stride, nsample=nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(self.mixerblock(in_planes=self.in_planes, planes=self.in_planes, share_planes=share_planes, nsample=nsample, intraLayer=self.intraLayer, interLayer=self.interLayer))
        return nn.Sequential(*layers)

    def _make_dec(self, planes, blocks, share_planes, nsample, is_head=False):
        layers = []
        layers.append(globals()[self.transup](in_planes=self.in_planes, out_planes=None if is_head else planes, nsample=nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(self.mixerblock(in_planes=self.in_planes, planes=self.in_planes, share_planes=share_planes, nsample=nsample, intraLayer=self.intraLayer, interLayer=self.interLayer))
        return nn.Sequential(*layers)

    def forward(self, pxo):
        p0, x0, o0 = pxo
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BilinearFeedForward,
     lambda: ([], {'in_planes1': 4, 'in_planes2': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBNReLU1D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ConvBNReLURes1D,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (GroupAll,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (IdentityInterSetLayer,
     lambda: ([], {'in_planes': 4, 'share_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NoInterSetLayer,
     lambda: ([], {'in_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PosExtraction,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (PreExtraction,
     lambda: ([], {'channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 11, 11, 11])], {}),
     True),
    (SymmetricTransitionDownBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'stride': 1, 'nsample': 4}),
     lambda: ([(torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (SymmetricTransitionDownBlockPaperv3,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([(torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (TransitionDown,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([(torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (TransitionDownBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([(torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
]

class Test_LifeBeyondExpectations_ECCV22_PointMixer(_paritybench_base):
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

