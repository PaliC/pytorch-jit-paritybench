import sys
_module = sys.modules[__name__]
del sys
lib = _module
config = _module
dataset = _module
enet = _module
loss = _module
pc_util = _module
projection = _module
scene_util = _module
solver = _module
utils = _module
_version = _module
pointnet2_modules = _module
pointnet2_semseg = _module
pointnet2_test = _module
pointnet2_utils = _module
pytorch_utils = _module
setup = _module
collect_scannet_scenes = _module
scannet_util = _module
visualize_prep_scene = _module
compute_multiview_features = _module
compute_multiview_projection = _module
eval = _module
project_multiview_features = _module
train = _module
visualize = _module

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


import torch


import numpy as np


import torch.nn as nn


from functools import reduce


from torch.autograd import Variable


import torch._utils


import torch.nn.functional as F


from torch.autograd import Function


from torch.optim.lr_scheduler import StepLR


from typing import List


from torch.autograd import gradcheck


from typing import Tuple


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import math


import torchvision.transforms as transforms


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.optim as optim


class LambdaBase(nn.Sequential):

    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):

    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


class LambdaMap(LambdaBase):

    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))


class LambdaReduce(LambdaBase):

    def forward(self, input):
        return reduce(self.lambda_func, self.forward_prepare(input))


class Padding(nn.Module):

    def __init__(self, dim, pad, value, index, nInputDim):
        super(Padding, self).__init__()
        self.value = value
        self.dim = dim
        self.pad = pad
        self.nInputDim = nInputDim
        if index != 0:
            raise NotImplementedError('Padding: index != 0 not implemented')

    def forward(self, input):
        dim = self.dim
        if self.nInputDim != 0:
            dim += input.dim() - self.nInputDim
        pad_size = list(input.size())
        pad_size[dim] = self.pad
        padder = Variable(input.data.new(*pad_size).fill_(self.value))
        if self.pad < 0:
            padded = torch.cat((padder, input), dim)
        else:
            padded = torch.cat((input, padder), dim)
        return padded


class Dropout(nn.Dropout):
    """
    Cancel out PyTorch rescaling by 1/(1-p)
    """

    def forward(self, input):
        input = input * (1 - self.p)
        return super(Dropout, self).forward(input)


class Dropout2d(nn.Dropout2d):
    """
    Cancel out PyTorch rescaling by 1/(1-p)
    """

    def forward(self, input):
        input = input * (1 - self.p)
        return super(Dropout2d, self).forward(input)


class StatefulMaxPool2d(nn.MaxPool2d):

    def __init__(self, *args, **kwargs):
        super(StatefulMaxPool2d, self).__init__(*args, **kwargs)
        self.indices = None
        self.input_size = None

    def forward(self, x):
        return_indices, self.return_indices = self.return_indices, True
        output, indices = super(StatefulMaxPool2d, self).forward(x)
        self.return_indices = return_indices
        self.indices = indices
        self.input_size = x.size()
        if return_indices:
            return output, indices
        return output


class StatefulMaxUnpool2d(nn.Module):

    def __init__(self, pooling):
        super(StatefulMaxUnpool2d, self).__init__()
        self.pooling = pooling
        self.unpooling = nn.MaxUnpool2d(pooling.kernel_size, pooling.stride, pooling.padding)

    def forward(self, x):
        return self.unpooling.forward(x, self.pooling.indices, self.pooling.input_size)


class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=-100):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, weights=None):
        assert inputs.size(0) == targets.size(0) == weights.size(0)
        loss = F.cross_entropy(input=inputs, target=targets, reduction='none', ignore_index=self.ignore_index)
        if weights is not None:
            loss = torch.mean(loss * weights)
        else:
            loss = torch.mean(loss)
        return loss


class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz: 'torch.Tensor', features: 'torch.Tensor'=None) ->(torch.Tensor, torch.Tensor):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, npoint, \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
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

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool=True, use_xyz: bool=True, sample_uniformly: bool=False):
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


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

    def __init__(self, *, mlp: List[int], npoint: int=None, radius: float=None, nsample: int=None, bn: bool=True, use_xyz: bool=True):
        super().__init__(mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz)


class PointnetSAModuleVotes(nn.Module):
    """ Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes """

    def __init__(self, *, mlp: List[int], npoint: int=None, radius: float=None, nsample: int=None, bn: bool=True, use_xyz: bool=True, pooling: str='max', sigma: float=None, normalize_xyz: bool=False, sample_uniformly: bool=False, ret_unique_cnt: bool=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        if npoint is not None:
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz, sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)
        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)

    def forward(self, xyz: 'torch.Tensor', features: 'torch.Tensor'=None, inds: 'torch.Tensor'=None) ->(torch.Tensor, torch.Tensor):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \\sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        else:
            assert inds.shape[1] == self.npoint
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None
        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(xyz, new_xyz, features)
        new_features = self.mlp_module(grouped_features)
        if self.pooling == 'max':
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        elif self.pooling == 'avg':
            new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])
        elif self.pooling == 'rbf':
            rbf = torch.exp(-1 * grouped_xyz.pow(2).sum(1, keepdim=False) / self.sigma ** 2 / 2)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(self.nsample)
        new_features = new_features.squeeze(-1)
        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return new_xyz, new_features, inds, unique_cnt


class PointnetSAModuleMSGVotes(nn.Module):
    """ Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes """

    def __init__(self, *, mlps: List[List[int]], npoint: int, radii: List[float], nsamples: List[int], bn: bool=True, use_xyz: bool=True, sample_uniformly: bool=False):
        super().__init__()
        assert len(mlps) == len(nsamples) == len(radii)
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly) if npoint is not None else pointnet2_utils.GroupAll(use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

    def forward(self, xyz: 'torch.Tensor', features: 'torch.Tensor'=None, inds: 'torch.Tensor'=None) ->(torch.Tensor, torch.Tensor):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, C) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \\sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1), inds


class PointnetFPModule(nn.Module):
    """Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, *, mlp: List[int], bn: bool=True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: 'torch.Tensor', known: 'torch.Tensor', unknow_feats: 'torch.Tensor', known_feats: 'torch.Tensor') ->torch.Tensor:
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
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))
        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return new_features.squeeze(-1)


class PointnetLFPModuleMSG(nn.Module):
    """ Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    learnable feature propagation layer."""

    def __init__(self, *, mlps: List[List[int]], radii: List[float], nsamples: List[int], post_mlp: List[int], bn: bool=True, use_xyz: bool=True, sample_uniformly: bool=False):
        super().__init__()
        assert len(mlps) == len(nsamples) == len(radii)
        self.post_mlp = pt_utils.SharedMLP(post_mlp, bn=bn)
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, sample_uniformly=sample_uniformly))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

    def forward(self, xyz2: 'torch.Tensor', xyz1: 'torch.Tensor', features2: 'torch.Tensor', features1: 'torch.Tensor') ->torch.Tensor:
        """ Propagate features from xyz1 to xyz2.
        Parameters
        ----------
        xyz2 : torch.Tensor
            (B, N2, 3) tensor of the xyz coordinates of the features
        xyz1 : torch.Tensor
            (B, N1, 3) tensor of the xyz coordinates of the features
        features2 : torch.Tensor
            (B, C2, N2) tensor of the descriptors of the the features
        features1 : torch.Tensor
            (B, C1, N1) tensor of the descriptors of the the features

        Returns
        -------
        new_features1 : torch.Tensor
            (B, \\sum_k(mlps[k][-1]), N1) tensor of the new_features descriptors
        """
        new_features_list = []
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz1, xyz2, features1)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
            if features2 is not None:
                new_features = torch.cat([new_features, features2], dim=1)
            new_features = new_features.unsqueeze(-1)
            new_features = self.post_mlp(new_features)
            new_features_list.append(new_features)
        return torch.cat(new_features_list, dim=1).squeeze(-1)


class Pointnet2MSG(nn.Module):

    def __init__(self, num_classes, input_channels=3, use_xyz=True, bn=True):
        super().__init__()
        NPOINTS = [1024, 256, 64, 16]
        RADIUS = [[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]]
        NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
        MLPS = [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]], [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]
        FP_MLPS = [[128, 128], [256, 256], [512, 512], [512, 512]]
        CLS_FC = [128]
        DP_RATIO = 0.5
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels
        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            self.SA_modules.append(PointnetSAModuleMSG(npoint=NPOINTS[k], radii=RADIUS[k], nsamples=NSAMPLE[k], mlps=mlps, use_xyz=use_xyz, bn=bn))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k], bn=bn))
        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=bn))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, num_classes, activation=None, bn=bn))
        cls_layers.insert(1, nn.Dropout(DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud: 'torch.cuda.FloatTensor'):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()
        return pred_cls


class Pointnet2SSG(nn.Module):

    def __init__(self, num_classes, input_channels=3, use_xyz=True, bn=True):
        super().__init__()
        NPOINTS = [1024, 256, 64, 16]
        RADIUS = [0.1, 0.2, 0.4, 0.8]
        NSAMPLE = [32, 32, 32, 32]
        MLPS = [[32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512]]
        FP_MLPS = [[128, 128], [256, 128], [256, 256], [256, 256]]
        CLS_FC = [128]
        DP_RATIO = 0.5
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels
        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            mlps = [channel_in] + mlps
            channel_out += mlps[-1]
            self.SA_modules.append(PointnetSAModule(npoint=NPOINTS[k], radius=RADIUS[k], nsample=NSAMPLE[k], mlp=mlps, use_xyz=use_xyz, bn=bn))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k], bn=bn))
        cls_layers = []
        pre_channel = FP_MLPS[0][-1]
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, CLS_FC[k], bn=bn))
            pre_channel = CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, num_classes, activation=None, bn=bn))
        cls_layers.insert(1, nn.Dropout(DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud: 'torch.cuda.FloatTensor'):
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()
        return pred_cls


class RandomDropout(nn.Module):

    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class BallQuery(Function):

    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        """

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        inds = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features, idx):
        """

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        ctx.for_backwards = idx, N
        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        """

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class QueryAndGroup(nn.Module):
    """
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def forward(self, xyz, new_xyz, features=None):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz
        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class GroupAll(nn.Module):
    """
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        """
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
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
        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features


class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=''):
        super().__init__()
        self.add_module(name + 'bn', batch_norm(in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: 'int', name: 'str'=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class _ConvBase(nn.Sequential):

    def __init__(self, in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=None, batch_norm=None, bias=True, preact=False, name=''):
        super().__init__()
        bias = bias and not bn
        conv_unit = conv(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)
        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
        self.add_module(name + 'conv', conv_unit)
        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv2d(_ConvBase):

    def __init__(self, in_size: 'int', out_size: 'int', *, kernel_size: Tuple[int, int]=(1, 1), stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0), activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str=''):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv2d, batch_norm=BatchNorm2d, bias=bias, preact=preact, name=name)


class SharedMLP(nn.Sequential):

    def __init__(self, args: 'List[int]', *, bn: bool=False, activation=nn.ReLU(inplace=True), preact: bool=False, first: bool=False, name: str=''):
        super().__init__()
        for i in range(len(args) - 1):
            self.add_module(name + 'layer{}'.format(i), Conv2d(args[i], args[i + 1], bn=(not first or not preact or i != 0) and bn, activation=activation if not first or not preact or i != 0 else None, preact=preact))


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: 'int', *, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm3d(_BNBase):

    def __init__(self, in_size: 'int', name: 'str'=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class Conv1d(_ConvBase):

    def __init__(self, in_size: 'int', out_size: 'int', *, kernel_size: int=1, stride: int=1, padding: int=0, activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str=''):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv1d, batch_norm=BatchNorm1d, bias=bias, preact=preact, name=name)


class Conv3d(_ConvBase):

    def __init__(self, in_size: 'int', out_size: 'int', *, kernel_size: Tuple[int, int, int]=(1, 1, 1), stride: Tuple[int, int, int]=(1, 1, 1), padding: Tuple[int, int, int]=(0, 0, 0), activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str=''):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv3d, batch_norm=BatchNorm3d, bias=bias, preact=preact, name=name)


class FC(nn.Sequential):

    def __init__(self, in_size: 'int', out_size: 'int', *, activation=nn.ReLU(inplace=True), bn: bool=False, init=None, preact: bool=False, name: str=''):
        super().__init__()
        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant_(fc.bias, 0)
        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)
        self.add_module(name + 'fc', fc)
        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))
            if activation is not None:
                self.add_module(name + 'activation', activation)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm1d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BatchNorm2d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm3d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (Conv1d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Dropout2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FC,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LambdaBase,
     lambda: ([], {'fn': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StatefulMaxPool2d,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_daveredrum_Pointnet2_ScanNet(_paritybench_base):
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

