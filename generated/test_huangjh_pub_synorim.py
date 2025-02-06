import sys
_module = sys.modules[__name__]
del sys
pointconv = _module
pointnet2 = _module
dataset = _module
evaluate = _module
metric = _module
base_model = _module
basis_net = _module
desc_net = _module
full_sync = _module
train = _module
exp = _module
misc = _module
base = _module
flow_dataset = _module
evaluate = _module
metric = _module
base_model = _module
basis_net = _module
basis_net_self = _module
desc_net = _module
desc_net_self = _module
full_sync = _module
spconv = _module
train = _module
exp = _module
point = _module

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


from typing import List


from typing import Optional


from typing import Tuple


import math


import functools


import collections


import torch


from numpy.random import RandomState


from torch.utils.data import Dataset


from enum import Enum


from typing import Mapping


from typing import Any


from typing import Callable


from typing import Union


from torch import nn


from torch.optim.lr_scheduler import LambdaLR


import random


from collections import defaultdict


import torch.nn.functional as F


from torch.nn import Parameter


from torch.utils.data import DataLoader


import torch.linalg


import torch.nn as nn


from torch.utils.tensorboard import SummaryWriter


from collections import OrderedDict


class DensityNet(nn.Module):

    def __init__(self, hidden_unit=[8, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_convs.append(nn.Conv1d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.InstanceNorm1d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv1d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.InstanceNorm1d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv1d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.InstanceNorm1d(1))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def execute(self, xyz_density):
        B, N = xyz_density.shape
        density_scale = xyz_density.unsqueeze(1)
        for i in range(len(self.mlp_convs)):
            bn = self.mlp_bns[i]
            conv = self.mlp_convs[i]
            density_scale = bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = self.sigmoid(density_scale) + 0.5
            else:
                density_scale = self.relu(density_scale)
        return density_scale


class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.relu = nn.ReLU()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.InstanceNorm(out_channel))
        else:
            self.mlp_convs.append(nn.Conv(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.InstanceNorm(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.InstanceNorm(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.InstanceNorm(out_channel))

    def execute(self, localized_xyz):
        weights = localized_xyz
        for i in range(len(self.mlp_convs)):
            conv = self.mlp_convs[i]
            bn = self.mlp_bns[i]
            weights = self.relu(bn(conv(weights)))
        return weights


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
    dist = -2 * jt.matmul(src, dst.permute(0, 2, 1))
    dist += jt.sum(src ** 2, -1).view(B, N, 1)
    dist += jt.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def compute_density(xyz, bandwidth):
    """
    xyz: input points position data, [B, N, C]
    """
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = jt.exp(-sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim=-1)
    return xyz_density


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    centroids = jt.zeros((B, npoint))
    distance = jt.ones((B, N)) * 10000000000.0
    farthest = np.random.randint(0, N, B, dtype='l')
    batch_indices = np.arange(B, dtype='l')
    farthest = jt.array(farthest)
    batch_indices = jt.array(batch_indices)
    for i in range(npoint):
        None
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :]
        centroid = centroid.view(B, 1, 3)
        dist = jt.sum((xyz - centroid.repeat(1, N, 1)) ** 2, 2)
        mask = dist < distance
        if mask.sum().data[0] > 0:
            distance[mask] = dist[mask]
        farthest = jt.argmax(distance, 1)[0]
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = np.arange(B, dtype='l')
    batch_indices = jt.array(batch_indices).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def topk(input, k, dim=None, largest=True, sorted=True):
    if dim is None:
        dim = -1
    if dim < 0:
        dim += input.ndim
    transpose_dims = [i for i in range(input.ndim)]
    transpose_dims[0] = dim
    transpose_dims[dim] = 0
    input = input.transpose(transpose_dims)
    index, values = jt.argsort(input, dim=0, descending=largest)
    indices = index[:k]
    values = values[:k]
    indices = indices.transpose(transpose_dims)
    values = values.transpose(transpose_dims)
    return [values, indices]


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
    _, group_idx = topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def sample_and_group(npoint, nsample, xyz, points, density_scale=None):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    None
    fps_idx = farthest_point_sample(xyz, npoint)
    None
    new_xyz = index_points(xyz, fps_idx)
    None
    idx = knn_point(nsample, xyz, new_xyz)
    None
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    None
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = concat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    None
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density


class PointConvDensitySetAbstraction(nn.Module):

    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super(PointConvDensitySetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.InstanceNorm(out_channel))
            last_channel = out_channel
        self.weightnet = WeightNet(3, 16)
        self.densitynet = DensityNet()
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.InstanceNorm1d(mlp[-1])
        self.group_all = group_all
        self.bandwidth = bandwidth
        self.relu = nn.ReLU()

    def execute(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        xyz_density = compute_density(xyz, self.bandwidth)
        density_scale = self.densitynet(xyz_density)
        if self.group_all:
            raise NotImplementedError
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group(self.npoint, self.nsample, xyz, points, density_scale.reshape(B, N, 1))
        new_points = new_points.permute(0, 3, 2, 1)
        for i in range(len(self.mlp_convs)):
            conv = self.mlp_convs[i]
            bn = self.mlp_bns[i]
            new_points = self.relu(bn(conv(new_points)))
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = new_points * grouped_density.permute(0, 3, 2, 1)
        new_points = jt.matmul(new_points.permute(0, 3, 1, 2), weights.permute(0, 3, 2, 1)).reshape(B, self.npoint, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = self.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointConvDensitySetInterpolation(nn.Module):

    def __init__(self, nsample, in_channel, mlp, bandwidth):
        super(PointConvDensitySetInterpolation, self).__init__()
        self.bandwidth = bandwidth
        self.nsample = nsample
        self.in_channel = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.relu = nn.ReLU()
        last_channel = in_channel
        self.weightnet = WeightNet(3, 16)
        self.densitynet = DensityNet()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.InstanceNorm2d(out_channel))
            last_channel = out_channel
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.InstanceNorm1d(mlp[-1])

    def execute(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        dists = square_distance(xyz1, xyz2)
        idx, dists = jt.argsort(dists, dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]
        dist_recip = 1.0 / (dists + 1e-08)
        norm = jt.sum(dist_recip, dim=2, keepdims=True)
        weight = dist_recip / norm
        interpolated_points = jt.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        xyz_density = compute_density(xyz1, self.bandwidth)
        density_scale = self.densitynet(xyz_density)
        None
        interpolated_points = jt.concat([points1, interpolated_points], dim=-1)
        None
        new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group(N, self.nsample, xyz1, interpolated_points, density_scale.reshape(B, N, 1))
        None
        new_points = new_points.permute(0, 3, 2, 1)
        None
        for i in range(len(self.mlp_convs)):
            None
            conv = self.mlp_convs[i]
            None
            bn = self.mlp_bns[i]
            None
            new_points = self.relu(bn(conv(new_points)))
            None
        None
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = new_points * grouped_density.permute(0, 3, 2, 1)
        new_points = jt.matmul(new_points.permute(0, 3, 1, 2), weights.permute(0, 3, 2, 1)).reshape(B, N, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = self.relu(new_points)
        return new_points


class PointConv(nn.Module):

    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        n_points = config.n_points
        self.depth = len(n_points)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sas = nn.ModuleList()
        self.ins = nn.ModuleList()
        channels = [in_channels, config.n_c]
        for layer_idx in range(len(n_points)):
            self.sas.append(PointConvDensitySetAbstraction(npoint=n_points[layer_idx], nsample=32, in_channel=3 + channels[layer_idx], mlp=[channels[layer_idx + 1] // 2, channels[layer_idx + 1] // 2, channels[layer_idx + 1]], bandwidth=0.1 * 2 ** layer_idx, group_all=False))
            channels.append(channels[-1] * 2)
        mlp_channel = 0
        for layer_idx in range(len(n_points)):
            p2_channel = channels[min(layer_idx + 2, len(n_points))]
            mlp_channel = channels[max(2, layer_idx + 1)]
            self.ins.append(PointConvDensitySetInterpolation(nsample=16, in_channel=3 + channels[layer_idx] + p2_channel, mlp=[mlp_channel] * (2 if layer_idx > 0 else 3), bandwidth=0.1 * 2 ** layer_idx))
        self.fc1 = nn.Conv1d(mlp_channel, mlp_channel, 1)
        self.bn1 = nn.InstanceNorm1d(mlp_channel)
        self.fc3 = nn.Conv1d(mlp_channel, self.out_channels, 1)
        self.relu = nn.ReLU()

    def execute(self, xyz, feat):
        xyz = xyz.permute(0, 2, 1)
        feat = feat.permute(0, 2, 1)
        B, _, _ = xyz.shape
        xyzs, feats = [xyz], [feat]
        for d in range(self.depth):
            l_xyz, l_points = self.sas[d](xyzs[d], feats[d])
            xyzs.append(l_xyz)
            feats.append(l_points)
        for d in range(self.depth - 1, -1, -1):
            feats[d] = self.ins[d](xyzs[d], xyzs[d + 1], feats[d], feats[d + 1])
        x = self.relu(self.bn1(self.fc1(feats[-1])))
        x = self.fc3(x)
        x = x.permute(0, 2, 1)
        return x


class PointNetFeaturePropagation(nn.Module):

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        self.relu = nn.ReLU()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.InstanceNorm1d(out_channel))
            last_channel = out_channel

    def execute(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            idx, dists = jt.argsort(dists, dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-08)
            norm = jt.sum(dist_recip, dim=2, keepdims=True)
            weight = dist_recip / norm
            interpolated_points = jt.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        if points1 is not None:
            new_points = concat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in self.mlp_convs.layers.items():
            bn = self.mlp_bns[i]
            new_points = self.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)


def optimal_block(batch_size):
    return 2 ** int(math.log(batch_size))


class FurthestPointSampler(nn.Module):
    cuda_src = """
        __device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                                int idx1, int idx2) {
            const float v1 = dists[idx1], v2 = dists[idx2];
            const int i1 = dists_i[idx1], i2 = dists_i[idx2];
            dists[idx1] = max(v1, v2);
            dists_i[idx1] = v2 > v1 ? i2 : i1;
        }

        __global__ void furthest_point_sampling_kernel (
            int b, int n, int m, int block_size,
            const float *__restrict__ dataset,
            float *__restrict__ temp, 
            int *__restrict__ idxs) {

            if (m <= 0) return;

            extern __shared__ int dists_i[];
            float *dists =  (float *) &dists_i[block_size];

            int batch_index = blockIdx.x;
            dataset += batch_index * n * 3;
            temp += batch_index * n;
            idxs += batch_index * m;

            int tid = threadIdx.x;
            const int stride = block_size;

            int old = 0;
            if (threadIdx.x == 0) idxs[0] = old;

            // initialize temp with INF
            for (int k = tid; k < n; k += stride)
                temp[k] = 1e10;

            __syncthreads();
            for (int j = 1; j < m; j++) {
                int besti = 0;
                float best = -1;
                float x1 = dataset[old * 3 + 0];
                float y1 = dataset[old * 3 + 1];
                float z1 = dataset[old * 3 + 2];
                for (int k = tid; k < n; k += stride) {
                    float x2, y2, z2;
                    x2 = dataset[k * 3 + 0];
                    y2 = dataset[k * 3 + 1];
                    z2 = dataset[k * 3 + 2];
                    float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
                    if (mag <= 1e-3) continue;

                    float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

                    float d2 = min(d, temp[k]);
                    temp[k] = d2;
                    besti = d2 > best ? k : besti;
                    best = d2 > best ? d2 : best;
                }
                dists[tid] = best;
                dists_i[tid] = besti;
                __syncthreads();

                if (block_size >= 512) {
                    if (tid < 256) {
                        __update(dists, dists_i, tid, tid + 256);
                    }
                    __syncthreads();
                }
                if (block_size >= 256) {
                    if (tid < 128) {
                        __update(dists, dists_i, tid, tid + 128);
                    }
                    __syncthreads();
                }
                if (block_size >= 128) {
                    if (tid < 64) {
                        __update(dists, dists_i, tid, tid + 64);
                    }
                    __syncthreads();
                }
                if (block_size >= 64) {
                    if (tid < 32) {
                        __update(dists, dists_i, tid, tid + 32);
                    }
                    __syncthreads();
                }
                if (block_size >= 32) {
                    if (tid < 16) {
                        __update(dists, dists_i, tid, tid + 16);
                    }
                    __syncthreads();
                }
                if (block_size >= 16) {
                    if (tid < 8) {
                        __update(dists, dists_i, tid, tid + 8);
                    }
                    __syncthreads();
                }
                if (block_size >= 8) {
                    if (tid < 4) {
                        __update(dists, dists_i, tid, tid + 4);
                    }
                    __syncthreads();
                }
                if (block_size >= 4) {
                    if (tid < 2) {
                        __update(dists, dists_i, tid, tid + 2);
                    }
                    __syncthreads();
                }
                if (block_size >= 2) {
                    if (tid < 1) {
                        __update(dists, dists_i, tid, tid + 1);
                    }
                    __syncthreads();
                }

                old = dists_i[0];
                if (tid == 0) idxs[j] = old;
            }
        }

        int block_size = #block_size;

        float *temp;
        cudaMallocManaged(&temp, in0_shape0 * in0_shape1 * sizeof(float));

        furthest_point_sampling_kernel<<<in0_shape0, block_size, 2*block_size*sizeof(int)>>>(
            in0_shape0,
            in0_shape1,
            out_shape1,
            block_size,
            in0_p,
            temp,
            out_p
        );
        cudaDeviceSynchronize();
        cudaFree(temp);
    """

    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def execute(self, x):
        """
        Parameters
        ----------
        x: jt.Var, (B, N, 3)

        Returns
        -------
        y: jt.Var, (B, n_samples, 3)
        """
        batch_size, n_points, n_coords = x.shape
        assert self.n_samples <= n_points
        assert n_coords == 3
        assert x.dtype == 'float32'
        block_size = optimal_block(batch_size)
        cuda_src = self.cuda_src.replace('#block_size', str(block_size))
        idxs_shape = [batch_size, self.n_samples]
        idxs = jt.code(idxs_shape, 'int32', [x], cuda_src=cuda_src)
        y = x.reindex([batch_size, self.n_samples, 3], ['i0', '@e0(i0, i1)', 'i2'], extras=[idxs])
        return y


class BallQueryGrouper(nn.Module):
    cuda_src = """
        __global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                                int nsample,
                                                const float *__restrict__ new_xyz,
                                                const float *__restrict__ xyz,
                                                int *__restrict__ idx,
                                                int *__restrict__ cnt) {
            int batch_index = blockIdx.x;
            xyz += batch_index * n * 3;
            new_xyz += batch_index * m * 3;
            idx += m * nsample * batch_index;
            cnt += batch_index * m;

            int index = threadIdx.x;
            int stride = blockDim.x;

            float radius2 = radius * radius;
            for (int j = index; j < m; j += stride) {
                float new_x = new_xyz[j * 3 + 0];
                float new_y = new_xyz[j * 3 + 1];
                float new_z = new_xyz[j * 3 + 2];
                cnt[j] = 0;

                for (int k = 0; k < n && cnt[j] < nsample; ++k) {
                    float x = xyz[k * 3 + 0];
                    float y = xyz[k * 3 + 1];
                    float z = xyz[k * 3 + 2];
                    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                                (new_z - z) * (new_z - z);

                    if (d2 < radius2) {
                        if (cnt[j] == 0) {
                            for (int l = 0; l < nsample; ++l)
                                idx[j * nsample + l] = k;
                        }
                        idx[j * nsample + cnt[j]] = k;
                        ++cnt[j];
                    }
                }
            }
        }

        int block_size = #block_size;

        query_ball_point_kernel<<<in0_shape0, block_size>>>(
            in0_shape0, in1_shape1, in0_shape1, #radius, #nsample,
            in0_p, in1_p, out0_p, out1_p
        );
    """

    def __init__(self, radius, n_samples, use_xyz):
        super().__init__()
        self.radius = radius
        self.n_samples = n_samples
        self.use_xyz = use_xyz

    def execute(self, new_xyz, pointset, feature):
        """
        Parameters
        ----------
        xyz: jt.Var, (B, N, 3)
        features: jt.Var, (B, N, C)

        Returns
        -------
        new_feature: jt.Var, (B, N, n_samples, C)
        """
        batch_size_x, n_input, n_coords = new_xyz.shape
        assert n_coords == 3
        batch_size_p, n_points, n_coords = pointset.shape
        assert n_coords == 3
        assert batch_size_x == batch_size_p
        if feature is not None:
            batch_size_f, n_points_f, n_feature = feature.shape
            assert batch_size_x == batch_size_f
            assert n_points == n_points_f
        block_size = optimal_block(batch_size_x)
        cuda_src = self.cuda_src.replace('#block_size', str(block_size)).replace('#radius', str(self.radius)).replace('#nsample', str(self.n_samples))
        idxs_shape = [batch_size_x, n_input, self.n_samples]
        cnts_shape = [batch_size_x, n_input]
        idxs, cnts = jt.code([idxs_shape, cnts_shape], ['int32', 'int32'], [new_xyz, pointset], cuda_src=cuda_src)
        pc_shape = [batch_size_x, n_input, self.n_samples, 3]
        new_pointset = pointset.reindex(pc_shape, ['i0', '@e0(i0, i1, i2)', 'i3'], extras=[idxs])
        if feature is not None:
            feature_shape = [batch_size_x, n_input, self.n_samples, n_feature]
            new_feature = feature.reindex(feature_shape, ['i0', '@e0(i0, i1, i2)', 'i3'], extras=[idxs])
        else:
            new_feature = None
        if self.use_xyz:
            local_xyz = new_pointset - new_xyz.unsqueeze(dim=2)
            if new_feature is not None:
                new_feature = jt.contrib.concat([local_xyz, new_feature], dim=-1)
            else:
                new_feature = local_xyz
        return new_feature


class GroupAll(nn.Module):

    def __init__(self, use_xyz):
        super().__init__()
        self.use_xyz = use_xyz

    def execute(self, new_xyz, pointset, feature):
        if self.use_xyz:
            new_feature = jt.contrib.concat([pointset, feature], dim=-1)
        new_feature = new_feature.unsqueeze(dim=1)
        return new_feature


class KNN(nn.Module):

    def __init__(self, k):
        self.k = k
        self.cuda_inc = """
        #undef out
        #include "helper_cuda.h" 

        __global__ void compute_distances(float * ref,
                                        int     ref_width,
                                        int     ref_pitch,
                                        float * query,
                                        int     query_width,
                                        int     query_pitch,
                                        int     height,
                                        float * dist) {

            // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
            const int BLOCK_DIM = 16;
            __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
            __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

            // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
            __shared__ int begin_A;
            __shared__ int begin_B;
            __shared__ int step_A;
            __shared__ int step_B;
            __shared__ int end_A;

            // Thread index
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int batch_id = blockIdx.z;

            // Initializarion of the SSD for the current thread
            float ssd = 0.f;

            // Loop parameters
            begin_A = BLOCK_DIM * blockIdx.y;
            begin_B = BLOCK_DIM * blockIdx.x;
            step_A  = BLOCK_DIM * ref_pitch;
            step_B  = BLOCK_DIM * query_pitch;
            end_A   = begin_A + (height-1) * ref_pitch;

            // Conditions
            int cond0 = (begin_A + tx < ref_width); // used to write in shared memory
            int cond1 = (begin_B + tx < query_width); // used to write in shared memory & to computations and to write in output array 
            int cond2 = (begin_A + ty < ref_width); // used to computations and to write in output matrix

            // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
            for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

                // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
                if (a/ref_pitch + ty < height) {
                    shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx + batch_id * height * ref_pitch] : 0;
                    shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx + batch_id * height * query_pitch] : 0;
                }
                else {
                    shared_A[ty][tx] = 0;
                    shared_B[ty][tx] = 0;
                }

                // Synchronize to make sure the matrices are loaded
                __syncthreads();

                // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
                if (cond2 && cond1) {
                    for (int k = 0; k < BLOCK_DIM; ++k){
                        float tmp = shared_A[k][ty] - shared_B[k][tx];
                        ssd += tmp*tmp;
                    }
                }

                // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
                __syncthreads();
            }

            // Write the block sub-matrix to device memory; each thread writes one element
            if (cond2 && cond1) {
                dist[ (begin_A + ty) * query_pitch + begin_B + tx + batch_id * ref_pitch * query_pitch ] = ssd;
            }
        }

        __global__ void modified_insertion_sort(float * dist,
                                                int     ref_pitch,
                                                int *   index,
                                                int     index_pitch,
                                                int     width,
                                                int     height,
                                                int     k){

            // Column position
            unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
            int batch_id = blockIdx.z ;


            // Do nothing if we are out of bounds
            if (xIndex < width) {

                // Pointer shift
                float * p_dist  = dist  + xIndex + batch_id * ref_pitch * index_pitch;
                int *   p_index = index + xIndex + batch_id * index_pitch * k;

                // Initialise the first index
                p_index[0] = 0;

                // Go through all points
                for (int i=1; i<height; ++i) {

                    // Store current distance and associated index
                    float curr_dist = p_dist[i*index_pitch];
                    int   curr_index  = i;

                    // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
                    if (i >= k && curr_dist >= p_dist[(k-1)*index_pitch]) {
                        continue;
                    }

                    // Shift values (and indexes) higher that the current distance to the right
                    int j = min(i, k-1);
                    while (j > 0 && p_dist[(j-1)*index_pitch] > curr_dist) {
                        p_dist[j*index_pitch]   = p_dist[(j-1)*index_pitch];
                        p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                        --j;
                    }

                    // Write the current distance and index at their position
                    p_dist[j*index_pitch]   = curr_dist;
                    p_index[j*index_pitch] = curr_index; 
                }
            }
        }

            __global__ void compute_sqrt(float * dist, int width, int pitch, int k){
                unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
                unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
                int batch_id = blockIdx.z;
                if (xIndex<width && yIndex<k)
                    dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
            }

           inline static bool knn_cuda_global(
               int batch_size, 
               float * ref,
                    int           ref_nb,
               float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     int *         knn_index, 
                     float *  tmp_dist ){

            // Constants
            const int BLOCK_DIM = 16;

            const unsigned int size_of_float = sizeof(float);
            const unsigned int size_of_int   = sizeof(int);

            // Return variables
            cudaError_t err0, err1, err2, err3;

            // Allocate global memory
            float * ref_dev   = ref;
            float * query_dev = query;
            float * dist_dev  = tmp_dist;
            int   * index_dev = knn_index;

            // Deduce pitch values
            size_t ref_pitch   = ref_nb; 
            size_t query_pitch = query_nb;
            size_t dist_pitch  = query_nb; 
            size_t index_pitch = query_nb; 

            // Compute the squared Euclidean distances
            dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
            dim3 grid0(query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, batch_size);
            if (query_nb % BLOCK_DIM != 0) grid0.x += 1;
            if (ref_nb   % BLOCK_DIM != 0) grid0.y += 1;


            // printf("%d", cudaDeviceSynchronize()); 
            // checkCudaErrors(cudaDeviceSynchronize());
            // printf(" before compute_distances \\n");

            compute_distances<<<grid0, block0>>>(ref_dev, ref_nb, ref_pitch, query_dev, query_nb, query_pitch, dim, dist_dev);
            // checkCudaErrors(cudaDeviceSynchronize());

            // printf("%d", cudaDeviceSynchronize()); 
            // printf(" after compute_distances \\n");

            // Sort the distances with their respective indexes
            dim3 block1(256, 1, 1);
            dim3 grid1(query_nb / 256, 1, batch_size);
            if (query_nb % 256 != 0) grid1.x += 1;
            // printf("%d", cudaDeviceSynchronize()); 
            // printf(" before modified_insertion_sort \\n");
            // checkCudaErrors(cudaDeviceSynchronize());

            modified_insertion_sort<<<grid1, block1>>>(dist_dev, ref_pitch, index_dev, index_pitch, query_nb, ref_nb, k);

            // checkCudaErrors(cudaDeviceSynchronize());
            // printf("%d", cudaDeviceSynchronize()); 
            // printf(" after modified_insertion_sort \\n");

            // Compute the square root of the k smallest distances
            //dim3 block2(16, 16, 1);
            //dim3 grid2(query_nb / 16, k / 16, batch_size);
            //if (query_nb % 16 != 0) grid2.x += 1;
            //if (k % 16 != 0)        grid2.y += 1;
            //compute_sqrt<<<grid2, block2>>>(dist_dev, query_nb, query_pitch, k);	


            // Copy k smallest distances / indexes from the device to the host
            // TODO: batch 2d copy dist
            // cudaMemcpy2DAsync(knn_dist,  query_nb * size_of_float, dist_dev,  dist_pitch*size_of_float,  query_nb * size_of_float, k, cudaMemcpyDefault);

            return true;
        }


        """
        self.cuda_src = """
            const int k = out0_shape1;
            const int query_nb = in1_shape2; 
            const int ref_nb = in0_shape2;
            const int dim = in0_shape1;
            const int batch_size = in0_shape0;
            knn_cuda_global(batch_size, in0_p, ref_nb, in1_p, query_nb, dim, k, out0_p, in2_p);
        """

    def execute(self, x_q, x_r):
        batch_size, c_dim, q_points = x_q.shape
        batch_size, c_dim, r_points = x_r.shape
        out_idx_shapes = [batch_size, self.k, q_points]
        tmp_dist = jt.empty((batch_size, r_points, q_points), 'float32')
        idxs, = jt.code([out_idx_shapes], ['int32'], [x_r, x_q, tmp_dist], cuda_src=self.cuda_src, cuda_header=self.cuda_inc)
        return idxs


class PN2BackboneLarge(nn.Module):

    def __init__(self, out_channels=50, use_xyz=True):
        super().__init__()
        self.out_channels = out_channels
        self.use_xyz = use_xyz
        self.build_model()

    def build_model(self):
        self.pointnet_modules = nn.ModuleList()
        self.pointnet_modules.append(PointnetModule(n_points=512, radius=0.2, n_samples=64, mlp=[3, 64, 64, 128], use_xyz=self.use_xyz))
        self.pointnet_modules.append(PointnetModule(n_points=128, radius=0.4, n_samples=64, mlp=[128, 128, 128, 256], use_xyz=self.use_xyz))
        self.pointnet_modules.append(PointnetModule(mlp=[256, 256, 512, 1024], use_xyz=self.use_xyz))
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 6, mlp=[128, 128, 128])
        self.fc_layer = nn.Sequential(nn.Conv1d(128, 128, 1), nn.InstanceNorm1d(128), nn.Dropout(0.5), nn.Conv1d(128, self.out_channels, 1))

    def execute(self, xyz, feature):
        B, N, _ = xyz.shape
        l1_xyz, l1_feature = self.pointnet_modules[0](xyz, feature)
        l2_xyz, l2_feature = self.pointnet_modules[1](l1_xyz, l1_feature)
        l3_xyz, l3_feature = self.pointnet_modules[2](l2_xyz, l2_feature)
        l2_feature = self.fp3(l2_xyz, l3_xyz, l2_feature, l3_feature)
        l1_feature = self.fp2(l1_xyz, l2_xyz, l1_feature, l2_feature)
        feature = self.fp1(xyz, l1_xyz, concat([xyz, feature], 2), l1_feature)
        feature = feature.permute(0, 2, 1)
        return self.fc_layer(feature)


class PN2BackboneSmall(nn.Module):

    def __init__(self, out_channels=50, use_xyz=True):
        super().__init__()
        self.out_channels = out_channels
        self.use_xyz = use_xyz
        self.build_model()

    def build_model(self):
        self.pointnet_modules = nn.ModuleList()
        self.pointnet_modules.append(PointnetModule(n_points=512, radius=0.2, n_samples=64, mlp=[9, 64, 64, 128], use_xyz=self.use_xyz))
        self.pointnet_modules.append(PointnetModule(n_points=128, radius=0.4, n_samples=64, mlp=[128, 128, 128, 256], use_xyz=self.use_xyz))
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=140, mlp=[128, 128, 128])
        self.fc_layer = nn.Sequential(nn.Conv1d(128, 128, 1), nn.InstanceNorm1d(128), nn.Dropout(0.5), nn.Conv1d(128, self.out_channels, 1))

    def execute(self, xyz, feature):
        B, N, _ = xyz.shape
        l1_xyz, l1_feature = self.pointnet_modules[0](xyz, feature)
        l2_xyz, l2_feature = self.pointnet_modules[1](l1_xyz, l1_feature)
        l1_feature = self.fp2(l1_xyz, l2_xyz, l1_feature, l2_feature)
        feature = self.fp1(xyz, l1_xyz, concat([xyz, feature], 2), l1_feature)
        feature = feature.permute(0, 2, 1)
        return self.fc_layer(feature)


class AverageMeter:
    """
    Maintain named lists of numbers. Compute their average to evaluate dataset statistics.
    This can not only used for loss, but also for progressive training logging, supporting import/export data.
    """

    def __init__(self):
        self.loss_dict = OrderedDict()

    def clear(self):
        self.loss_dict.clear()

    def export(self, f):
        if isinstance(f, str):
            f = open(f, 'wb')
        pickle.dump(self.loss_dict, f)

    def load(self, f):
        if isinstance(f, str):
            f = open(f, 'rb')
        self.loss_dict = pickle.load(f)
        return self

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: [loss_val]})
            else:
                self.loss_dict[loss_name].append(loss_val)

    def get_mean_loss_dict(self):
        loss_dict = {}
        for loss_name, loss_arr in self.loss_dict.items():
            loss_dict[loss_name] = sum(loss_arr) / len(loss_arr)
        return loss_dict

    def get_mean_loss(self):
        mean_loss_dict = self.get_mean_loss_dict()
        if len(mean_loss_dict) == 0:
            return 0.0
        else:
            return sum(mean_loss_dict.values()) / len(mean_loss_dict)

    def get_printable_mean(self):
        text = ''
        all_loss_sum = 0.0
        for loss_name, loss_mean in self.get_mean_loss_dict().items():
            all_loss_sum += loss_mean
            text += '(%s:%.4f) ' % (loss_name, loss_mean)
        text += ' sum = %.4f' % all_loss_sum
        return text

    def get_newest_loss_dict(self, return_count=False):
        loss_dict = {}
        loss_count_dict = {}
        for loss_name, loss_arr in self.loss_dict.items():
            if len(loss_arr) > 0:
                loss_dict[loss_name] = loss_arr[-1]
                loss_count_dict[loss_name] = len(loss_arr)
        if return_count:
            return loss_dict, loss_count_dict
        else:
            return loss_dict

    def get_printable_newest(self):
        nloss_val, nloss_count = self.get_newest_loss_dict(return_count=True)
        return ', '.join([f'{loss_name}[{nloss_count[loss_name] - 1}]: {nloss_val[loss_name]}' for loss_name in nloss_val.keys()])

    def print_format_loss(self, color=None):
        if hasattr(sys.stdout, 'terminal'):
            color_device = sys.stdout.terminal
        else:
            color_device = sys.stdout
        if color == 'y':
            color_device.write('\x1b[93m')
        elif color == 'g':
            color_device.write('\x1b[92m')
        elif color == 'b':
            color_device.write('\x1b[94m')
        None
        if color is not None:
            color_device.write('\x1b[0m')


def lambda_lr_wrapper(it, lr_config, batch_size):
    return max(lr_config['decay_mult'] ** int(it * batch_size / lr_config['decay_step']), lr_config['clip'] / lr_config['init'])


class BaseModel(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.log_cache = AverageMeter()

    @staticmethod
    def load_module(spec_path):
        """
        Load a module given spec_path
        :param spec_path: Path to a model ckpt.
        :return: the module class, possibly with weight loaded.
        """
        spec_path = Path(spec_path)
        config_args = parse_config_yaml(spec_path.parent / 'config.yaml')
        net_module = importlib.import_module('models.' + config_args.model).Model
        net_model = net_module(config_args)
        if 'none.pth' not in spec_path.name:
            ckpt_data = torch.load(spec_path)
            net_model.load_state_dict(ckpt_data['state_dict'])
            None
        return net_model

    def configure_optimizers(self):
        lr_config = self.hparams.learning_rate
        if self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=lr_config['init'], momentum=0.9, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr_config['init'], weight_decay=self.hparams.weight_decay, amsgrad=True)
        else:
            raise NotImplementedError
        scheduler = LambdaLR(optimizer, lr_lambda=functools.partial(lambda_lr_wrapper, lr_config=lr_config, batch_size=self.hparams.batch_size))
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def on_after_backward(self):
        grad_clip_val = self.hparams.get('grad_clip', 1000.0)
        torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=grad_clip_val)
        has_nan_value = False
        for p in filter(lambda p: p.grad is not None, self.parameters()):
            pdata = p.grad.data
            grad_is_nan = pdata != pdata
            if torch.any(grad_is_nan):
                has_nan_value = True
                pdata[grad_is_nan] = 0.0
        if has_nan_value:
            None

    def log(self, key, value):
        if self.hparams.is_training:
            assert key not in self.log_cache.loss_dict
        self.log_cache.append_loss({key: value.item() if isinstance(value, torch.Tensor) else value})

    def log_dict(self, dictionary: 'Mapping[str, Any]'):
        for k, v in dictionary.items():
            self.log(str(k), v)

    def write_log(self, writer, it):
        logs_written = {}
        if not self.hparams.is_training or it % 10 == 0:
            for k, v in self.log_cache.get_mean_loss_dict().items():
                writer.add_scalar(k, v, it)
                logs_written[k] = v
        self.log_cache.clear()
        return logs_written


class DatasetSpec(Enum):
    FILENAME = 100
    PC = 200
    FULL_FLOW = 300
    FULL_MASK = 400
    QUANTIZED_COORDS = 500


class PairwiseFlowMetric:

    def __init__(self, batch_mean: 'bool'=False, compute_epe3d: 'bool'=True, compute_acc3d_outlier: 'bool'=False, scene_level: 'bool'=False):
        """
        :param batch_mean: Whether to return an array with size (B, ) or a single scalar (mean)
        :param compute_epe3d: compute EPE3D metric
        :param compute_acc3d_outlier: compute Acc3d-strict, Acc3d-relax and outlier metric
        :param scene_level: whether use the scene threshold as proposed in FlowNet3D.
        """
        self.batch_mean = batch_mean
        self.compute_epe3d = compute_epe3d
        self.compute_acc3d_outlier = compute_acc3d_outlier
        self.scene_level = scene_level

    def evaluate(self, gt_flow: 'torch.Tensor', pd_flow: 'torch.Tensor', valid_mask: 'torch.Tensor'=None):
        """
        Compute the pairwise flow metric; batch dimension will not be reduced. (Unit will be the same as input)
        :param gt_flow: (..., N, 3)
        :param pd_flow: (..., N, 3)
        :param valid_mask: (..., N)
        :return: metrics dict.
        """
        result_dict = {}
        assert gt_flow.size(-1) == pd_flow.size(-1) == 3
        assert gt_flow.size(-2) == pd_flow.size(-2)
        n_point = gt_flow.size(-2)
        gt_flow = gt_flow.reshape(-1, n_point, 3)
        pd_flow = pd_flow.reshape(-1, n_point, 3)
        if valid_mask is None:
            valid_mask = torch.ones((gt_flow.size(0), n_point), dtype=bool, device=gt_flow.device)
        else:
            valid_mask = valid_mask.reshape(-1, n_point)
        l2_norm = torch.norm(pd_flow - gt_flow, dim=-1)
        if self.compute_epe3d:
            result_dict['epe3d'] = (l2_norm * valid_mask).sum(-1) / (valid_mask.sum(-1) + 1e-06)
        if self.compute_acc3d_outlier:
            sf_norm = torch.norm(gt_flow, dim=-1)
            rel_err = l2_norm / (sf_norm + 0.0001)
            if self.scene_level:
                acc3d_strict_mask = torch.logical_or(l2_norm < 0.05, rel_err < 0.05).float()
                acc3d_relax_mask = torch.logical_or(l2_norm < 0.1, rel_err < 0.1).float()
                outlier_mask = torch.logical_or(l2_norm > 0.3, rel_err > 0.1).float()
            else:
                acc3d_strict_mask = torch.logical_or(l2_norm < 0.02, rel_err < 0.05).float()
                acc3d_relax_mask = torch.logical_or(l2_norm < 0.05, rel_err < 0.1).float()
                outlier_mask = (rel_err > 0.3).float()
            result_dict['acc3d_strict'] = (acc3d_strict_mask * valid_mask).sum(-1) / (valid_mask.sum(-1) + 1e-06)
            result_dict['acc3d_relax'] = (acc3d_relax_mask * valid_mask).sum(-1) / (valid_mask.sum(-1) + 1e-06)
            result_dict['outlier'] = (outlier_mask * valid_mask).sum(-1) / (valid_mask.sum(-1) + 1e-06)
        if self.batch_mean:
            for ckey in list(result_dict.keys()):
                result_dict[ckey] = torch.mean(result_dict[ckey])
        return result_dict


def list_collate(batch):
    """
    This collation does not stack batch dimension, but instead output only lists.
    """
    elem = None
    for e in batch:
        if e is not None:
            elem = e
            break
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            return list_collate([(torch.as_tensor(b) if b is not None else None) for b in batch])
        elif elem.shape == ():
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: list_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [list_collate(samples) for samples in transposed]
    elif elem is None:
        return batch
    raise NotImplementedError


def index_points_group(points, knn_idx, t=False):
    """
    Input:
        points: input points data, [B, N', C], or [B, C, N'](transposed)
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C] or [B, C, N, K](transposed)
    """
    B, Np, C = points.size()
    if t:
        Np, C = C, Np
    _, N, K = knn_idx.size()
    knn_idx = knn_idx.reshape(B, -1)
    if not t:
        new_points = torch.gather(points, dim=1, index=knn_idx.unsqueeze(-1).expand(-1, -1, points.size(-1)))
        new_points = new_points.reshape(B, N, K, C)
    else:
        new_points = torch.gather(points, dim=-1, index=knn_idx.unsqueeze(1).expand(-1, points.size(1), -1))
        new_points = new_points.reshape(B, C, N, K)
    return new_points


def propagate_features(source_pc: 'torch.Tensor', target_pc: 'torch.Tensor', source_feat: 'torch.Tensor', nk: 'int'=3, batched: 'bool'=True):
    """
    Propagate features from the domain of source to the domain of target.
    :param source_pc: (B, N, 3) point coordinates
    :param target_pc: (B, M, 3) point coordinates
    :param source_feat: (B, N, F) source features
    :param nk: propagate k number
    :param batched: whether dimension B is present or not.
    :return: (B, M, F) target feature
    """
    if not batched:
        source_pc = source_pc.unsqueeze(0)
        target_pc = target_pc.unsqueeze(0)
        source_feat = source_feat.unsqueeze(0)
    dist = torch.cdist(target_pc, source_pc)
    dist, group_idx = torch.topk(dist, nk, dim=-1, largest=False, sorted=False)
    w_func = 1 / (dist + 1e-06)
    weight = (w_func / torch.sum(w_func, dim=-1, keepdim=True)).unsqueeze(-1)
    sparse_feature = index_points_group(source_feat, group_idx)
    full_flow = (sparse_feature * weight).sum(-2)
    if not batched:
        full_flow = full_flow[0]
    return full_flow


class Model(BaseModel):
    """
    This model runs the full test of our model, taking multiple point clouds as input.
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        self.basis_net = self.load_module(self.hparams.basis_checkpoint)
        self.desc_net = self.basis_net.desc_net
        self.hparams.voxel_size = self.basis_net.hparams.voxel_size

    def update_device(self):
        self.basis_net.device = self.device
        self.desc_net.device = self.device

    def forward(self, batch):
        s_basis_desc, s_sels = self.basis_net(batch)
        return s_basis_desc[0], s_basis_desc[1], s_sels

    def optimize_map(self, old_c_dict, ca_dict, cb_dict):
        """
        Optimize for the best C that satisfies C* = argmin sum |ca @ c - cb| + |c @ hl - hk|
        :param old_c_dict: (k, l) --> (MxM) Fmap dict
        :param ca_dict: data dictionary for each (k, l) pair
        :param cb_dict: data dictionary for each (k, l) pair
        :return: optimized Fmap dict.
        """
        all_keys = sorted(list(old_c_dict.keys()))
        num_frames = max(*[max(t) for t in all_keys]) + 1
        view_sizes = {}
        for fid in range(num_frames - 1):
            view_sizes[fid] = old_c_dict[fid, fid + 1].size(0)
            view_sizes[fid + 1] = old_c_dict[fid, fid + 1].size(1)
        var_sizes_T = np.cumsum([0] + [view_sizes[i] for i in range(num_frames)])
        cur_device = old_c_dict[all_keys[0]].device
        num_universe = max(view_sizes.values()) - self.hparams.num_v_sub
        consistent_weight = self.hparams.cycle_weight
        C_star = {k: v for k, v in old_c_dict.items()}
        C_init_scale = {k: torch.linalg.norm(v.flatten()) for k, v in C_star.items()}
        robust_kernel, _ = self.basis_net.get_robust_kernel()
        for iter_i in range(self.hparams.sync_iter):
            """
            1. Solve for {H} matrices, fixing {C}
            """
            h_rows = []
            for fid_i in range(num_frames):
                h_cols = []
                for fid_j in range(num_frames):
                    sum_matrices = [torch.zeros((view_sizes[fid_i], view_sizes[fid_j]), device=cur_device)]
                    if fid_i < fid_j:
                        if (fid_i, fid_j) in all_keys:
                            sum_matrices.append(-C_star[fid_i, fid_j])
                        if (fid_j, fid_i) in all_keys:
                            sum_matrices.append(-C_star[fid_j, fid_i].transpose(-1, -2))
                    elif fid_i == fid_j:
                        for fid_k in range(num_frames):
                            if (fid_i, fid_k) in all_keys:
                                sum_matrices.append(torch.eye(view_sizes[fid_i], device=cur_device))
                            if (fid_k, fid_i) in all_keys:
                                X_ji = C_star[fid_k, fid_i]
                                sum_matrices.append(X_ji.transpose(-1, -2) @ X_ji)
                    h_cols.append(sum(sum_matrices))
                h_rows.append(torch.cat(h_cols, dim=-1))
            full_h_matrix = torch.cat(h_rows, dim=0)
            _, h_star = torch.linalg.eigh(full_h_matrix, UPLO='U')
            h_star = h_star[..., :num_universe]
            """
            2. Solve for {C} matrices, fixing {H}
            """
            change_scales = []
            for mid, nid in all_keys:
                C_ij_star = self.basis_net.solve_optimal_maps(ca_dict[mid, nid], cb_dict[mid, nid], robust_kernel=robust_kernel, k_i=h_star[var_sizes_T[mid]:var_sizes_T[mid + 1]], k_j=h_star[var_sizes_T[nid]:var_sizes_T[nid + 1]], sqrt_mu=np.sqrt(consistent_weight), c_init=C_star[mid, nid])
                change_scale = torch.linalg.norm((C_ij_star - C_star[mid, nid]).flatten()) / C_init_scale[mid, nid]
                change_scales.append(change_scale.item())
                C_star[mid, nid] = C_ij_star
            rel_change = np.mean(change_scales)
            if rel_change < self.hparams.sync_converge_rel:
                break
        return C_star

    def test_step(self, batch, batch_idx):
        basis_output, desc_output, all_sels = self(batch)
        num_views = len(batch[DS.QUANTIZED_COORDS])
        iters_ij, iters_upper_ij = [], []
        for view_i in range(num_views):
            for view_j in range(num_views):
                if view_i == view_j:
                    continue
                if view_i < view_j:
                    iters_upper_ij.append((view_i, view_j))
                iters_ij.append((view_i, view_j))
        full_pc, sub_pc = {}, {}
        for view_i in range(num_views):
            full_pc[view_i] = batch[DS.PC][view_i][0]
            sub_pc[view_i] = full_pc[view_i][all_sels[view_i]]
        normalized_basis, basis_multiplier = {}, {}
        for view_i in range(num_views):
            basis_origin = basis_output.features_at(view_i)
            svd_res = torch.svd(basis_origin)
            right_multiplier = torch.diag_embed(svd_res.S) @ svd_res.V.transpose(-1, -2)
            normalized_basis[view_i], basis_multiplier[view_i] = svd_res.U, right_multiplier
        sub_desc = {}
        for view_i in range(num_views):
            sub_desc[view_i] = desc_output.features_at(view_i)
        phi_i_all, phi_j_all = {}, {}
        for view_i, view_j in iters_upper_ij:
            phi_i_all[view_i, view_j], phi_j_all[view_i, view_j] = self.basis_net.align_basis_via_pd_test(normalized_basis[view_i], normalized_basis[view_j], sub_desc[view_i], sub_desc[view_j], thres=self.hparams.gpd_thres)
            phi_i_all[view_j, view_i], phi_j_all[view_j, view_i] = phi_j_all[view_i, view_j], phi_i_all[view_i, view_j]
        maps_init = {}
        robust_kernel, robust_iter = self.basis_net.get_robust_kernel()
        for view_i, view_j in iters_ij:
            maps_init[view_i, view_j] = self.basis_net.solve_optimal_maps(phi_i_all[view_i, view_j], phi_j_all[view_i, view_j], robust_kernel=robust_kernel, num_iter=robust_iter)
        maps_optimized = self.optimize_map(maps_init, phi_i_all, phi_j_all)
        final_flows = {}
        for view_i, view_j in iters_ij:
            final_flows[view_i, view_j] = self.basis_net.compute_flow_from_maps(normalized_basis[view_i], normalized_basis[view_j], maps_optimized[view_i, view_j], sub_pc[view_i], sub_pc[view_j], pcond_multiplier=basis_multiplier[view_j], sparse_coords=batch[DS.QUANTIZED_COORDS][view_i][0][0])['final']
        full_final_flows = self.propagte_to_full_flow(batch, final_flows)
        error = self.evaluate_flow_error(batch, full_final_flows)
        return full_final_flows, error

    def propagte_to_full_flow(self, batch, sub_flows):
        """
        Propagate from flows at the sub-sampled positions to the input resolution.
        """
        eval_pairs = list(sub_flows.keys())
        output_full_flow = {}
        for view_i, view_j in eval_pairs:
            full_pc_i = batch[DS.PC][view_i][0]
            voxelized_coords = batch[DS.QUANTIZED_COORDS][view_i][0][0] * self.hparams.voxel_size
            pd_flow_ij = propagate_features(voxelized_coords, full_pc_i, sub_flows[view_i, view_j], batched=False, nk=self.hparams.flow_k)
            output_full_flow[view_i, view_j] = pd_flow_ij
        return output_full_flow

    @staticmethod
    def evaluate_flow_error(batch, pd_flows):
        eval_pairs = list(batch[DS.FULL_FLOW].keys())
        err_acc_dict = defaultdict(list)
        for view_i, view_j in eval_pairs:
            gt_flow_ij = batch[DS.FULL_FLOW][view_i, view_j][0]
            gt_mask_ij = batch[DS.FULL_MASK][view_i, view_j][0]
            if gt_flow_ij is None:
                continue
            err_dict = PairwiseFlowMetric(compute_epe3d=True, compute_acc3d_outlier=True).evaluate(gt_flow_ij, pd_flows[view_i, view_j], valid_mask=gt_mask_ij)
            err_full_dict = PairwiseFlowMetric(compute_epe3d=True, compute_acc3d_outlier=True).evaluate(gt_flow_ij, pd_flows[view_i, view_j])
            err_dict = {k: v.item() for k, v in err_dict.items()}
            err_full_dict = {k: v.item() for k, v in err_full_dict.items()}
            for err_name in err_dict.keys():
                err_acc_dict[err_name].append(err_dict[err_name])
                err_acc_dict[err_name + '-full'].append(err_full_dict[err_name])
        err_acc_final_dict = {}
        for mkey, marray in err_acc_dict.items():
            err_acc_final_dict[f'{mkey}-avg'] = np.mean(marray)
            err_acc_final_dict[f'{mkey}-std'] = np.std(marray)
        return err_acc_final_dict

    def test_dataloader(self):
        test_set = FlowDataset(**self.hparams.test_kwargs, spec=[DS.FILENAME, DS.QUANTIZED_COORDS, DS.PC, DS.FULL_FLOW, DS.FULL_MASK], hparams=self.hparams)
        return DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_collate)


class BasicBlockBase(nn.Module):
    """
    A double-conv ResBlock with relu activation, with residual connection.
    """

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, D=3):
        super(BasicBlockBase, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=3, stride=stride, dimension=D)
        self.norm1 = ME.MinkowskiInstanceNorm(planes)
        self.conv2 = ME.MinkowskiConvolution(planes, planes, kernel_size=3, stride=1, dilation=dilation, bias=False, dimension=D)
        self.norm2 = ME.MinkowskiInstanceNorm(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = MEF.relu(out)
        return out

