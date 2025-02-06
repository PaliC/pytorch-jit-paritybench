import sys
_module = sys.modules[__name__]
del sys
DatasetLidarCamera = _module
evaluate_calib = _module
losses = _module
LCCNet = _module
correlation = _module
setup = _module
quaternion_distances = _module
train_with_sacred = _module
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


from math import radians


import numpy as np


import pandas as pd


import torch


import torchvision.transforms.functional as TTF


from torch.utils.data import Dataset


from torchvision import transforms


import random


import matplotlib.pyplot as plt


import torch.nn.functional as F


import torch.nn.parallel


import torch.utils.data


import time


from torch import nn as nn


import torchvision


import torchvision.transforms as transforms


from torch.autograd import Variable


import torchvision.models as models


import torch.utils.model_zoo as model_zoo


import torch.nn as nn


import torch.optim as optim


import math


import matplotlib.image as mpimg


from torch.autograd import Function


from torch.nn.modules.module import Module


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from matplotlib import cm


from torch.utils.data.dataloader import default_collate


def quatinv(q):
    """
    Batch quaternion inversion
    Args:
        q (torch.Tensor/np.ndarray): shape=[Nx4]

    Returns:
        torch.Tensor/np.ndarray: shape=[Nx4]
    """
    if isinstance(q, torch.Tensor):
        t = q.clone()
    elif isinstance(q, np.ndarray):
        t = q.copy()
    else:
        raise TypeError('Type not supported')
    t *= -1
    t[:, 0] *= -1
    return t


def quatmultiply(q, r):
    """
    Multiply two quaternions
    Args:
        q (torch.Tensor/nd.ndarray): shape=[4], first quaternion
        r (torch.Tensor/nd.ndarray): shape=[4], second quaternion

    Returns:
        torch.Tensor: shape=[4], normalized quaternion q*r
    """
    t = torch.zeros(4, device=q.device)
    t[0] = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t[1] = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t[2] = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t[3] = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    return t / t.norm()


def quaternion_distance(q, r, device):
    """
    Batch quaternion distances, used as loss
    Args:
        q (torch.Tensor): shape=[Nx4]
        r (torch.Tensor): shape=[Nx4]
        device (str): 'cuda' or 'cpu'

    Returns:
        torch.Tensor: shape=[N]
    """
    t = quatmultiply(q, quatinv(r), device)
    return 2 * torch.atan2(torch.norm(t[:, 1:], dim=1), torch.abs(t[:, 0]))


class GeometricLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.sx = torch.nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.sq = torch.nn.Parameter(torch.Tensor([-3.0]), requires_grad=True)
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = torch.exp(-self.sx) * loss_transl + self.sx
        total_loss += torch.exp(-self.sq) * loss_rot + self.sq
        return total_loss


class ProposedLoss(nn.Module):

    def __init__(self, rescale_trans, rescale_rot):
        super(ProposedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.losses = {}

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = 0.0
        if self.rescale_trans != 0.0:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean() * 100
        loss_rot = 0.0
        if self.rescale_rot != 0.0:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        total_loss = self.rescale_trans * loss_transl + self.rescale_rot * loss_rot
        self.losses['total_loss'] = total_loss
        self.losses['transl_loss'] = loss_transl
        self.losses['rot_loss'] = loss_rot
        return self.losses


class L1Loss(nn.Module):

    def __init__(self, rescale_trans, rescale_rot):
        super(L1Loss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, target_transl, target_rot, transl_err, rot_err):
        loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = self.transl_loss(rot_err, target_rot).sum(1).mean()
        total_loss = self.rescale_trans * loss_transl + self.rescale_rot * loss_rot
        return total_loss


def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), 'Not a valid quaternion'
    if q.norm() != 1.0:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2 * q[2] ** 2 - 2 * q[3] ** 2
    mat[0, 1] = 2 * q[1] * q[2] - 2 * q[3] * q[0]
    mat[0, 2] = 2 * q[1] * q[3] + 2 * q[2] * q[0]
    mat[1, 0] = 2 * q[1] * q[2] + 2 * q[3] * q[0]
    mat[1, 1] = 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2
    mat[1, 2] = 2 * q[2] * q[3] - 2 * q[1] * q[0]
    mat[2, 0] = 2 * q[1] * q[3] - 2 * q[2] * q[0]
    mat[2, 1] = 2 * q[2] * q[3] + 2 * q[1] * q[0]
    mat[2, 2] = 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2
    mat[3, 3] = 1.0
    return mat


def rotate_points(PC, R, T=None, inverse=True):
    if T is not None:
        R = R.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(T)
        RT = T * R
    else:
        RT = R.copy()
    if inverse:
        RT.invert_safe()
    RT = torch.tensor(RT, device=PC.device, dtype=torch.float)
    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError('Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)')
    return PC


def tvector2mat(t):
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), 'Not a valid translation'
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat


def rotate_points_torch(PC, R, T=None, inverse=True):
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()
    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError('Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)')
    return PC


def rotate_forward(PC, R, T=None):
    """
    Transform the point cloud PC, so to have the points 'as seen from' the new
    pose T*R
    Args:
        PC (torch.Tensor): Point Cloud to be transformed, shape [4xN] or [Nx4]
        R (torch.Tensor/mathutils.Euler): can be either:
            * (mathutils.Euler) euler angles of the rotation part, in this case T cannot be None
            * (torch.Tensor shape [4]) quaternion representation of the rotation part, in this case T cannot be None
            * (mathutils.Matrix shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
            * (torch.Tensor shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
        T (torch.Tensor/mathutils.Vector): Translation of the new pose, shape [3], or None (depending on R)

    Returns:
        torch.Tensor: Transformed Point Cloud 'as seen from' pose T*R
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC, R, T, inverse=True)
    else:
        return rotate_points(PC, R, T, inverse=True)


class DistancePoints3D(nn.Module):

    def __init__(self):
        super(DistancePoints3D, self).__init__()

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The mean distance between 3D points
        """
        total_loss = torch.tensor([0.0])
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i]
            point_cloud_out = point_clouds[i].clone()
            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)
            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)
            RT_total = torch.mm(RT_target.inverse(), RT_predicted)
            point_cloud_out = rotate_forward(point_cloud_out, RT_total)
            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.0)
            total_loss += error.mean()
        return total_loss / target_transl.shape[0]


class CombinedLoss(nn.Module):

    def __init__(self, rescale_trans, rescale_rot, weight_point_cloud):
        super(CombinedLoss, self).__init__()
        self.rescale_trans = rescale_trans
        self.rescale_rot = rescale_rot
        self.transl_loss = nn.SmoothL1Loss(reduction='none')
        self.weight_point_cloud = weight_point_cloud
        self.loss = {}

    def forward(self, point_clouds, target_transl, target_rot, transl_err, rot_err):
        """
        The Combination of Pose Error and Points Distance Error
        Args:
            point_cloud: list of B Point Clouds, each in the relative GT frame
            target_transl: groundtruth of the translations
            target_rot: groundtruth of the rotations
            transl_err: network estimate of the translations
            rot_err: network estimate of the rotations

        Returns:
            The combination loss of Pose error and the mean distance between 3D points
        """
        loss_transl = 0.0
        if self.rescale_trans != 0.0:
            loss_transl = self.transl_loss(transl_err, target_transl).sum(1).mean()
        loss_rot = 0.0
        if self.rescale_rot != 0.0:
            loss_rot = quaternion_distance(rot_err, target_rot, rot_err.device).mean()
        pose_loss = self.rescale_trans * loss_transl + self.rescale_rot * loss_rot
        point_clouds_loss = torch.tensor([0.0])
        for i in range(len(point_clouds)):
            point_cloud_gt = point_clouds[i]
            point_cloud_out = point_clouds[i].clone()
            R_target = quat2mat(target_rot[i])
            T_target = tvector2mat(target_transl[i])
            RT_target = torch.mm(T_target, R_target)
            R_predicted = quat2mat(rot_err[i])
            T_predicted = tvector2mat(transl_err[i])
            RT_predicted = torch.mm(T_predicted, R_predicted)
            RT_total = torch.mm(RT_target.inverse(), RT_predicted)
            point_cloud_out = rotate_forward(point_cloud_out, RT_total)
            error = (point_cloud_out - point_cloud_gt).norm(dim=0)
            error.clamp(100.0)
            point_clouds_loss += error.mean()
        total_loss = (1 - self.weight_point_cloud) * pose_loss + self.weight_point_cloud * (point_clouds_loss / target_transl.shape[0])
        self.loss['total_loss'] = total_loss
        self.loss['transl_loss'] = loss_transl
        self.loss['rot_loss'] = loss_rot
        self.loss['point_clouds_loss'] = point_clouds_loss / target_transl.shape[0]
        return self.loss


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.elu = nn.ELU()
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.leakyRELU(out)
        return out


class SEBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.leakyRELU(out)
        return out


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {(18): models.resnet18, (34): models.resnet34, (50): models.resnet50, (101): models.resnet101, (152): models.resnet152}
        if num_layers not in resnets:
            raise ValueError('{} is not a valid number of resnet layers'.format(num_layers))
        self.encoder = resnets[num_layers](pretrained)
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.maxpool(self.features[-1]))
        self.features.append(self.encoder.layer1(self.features[-1]))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features


class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()
            correlation_cuda.forward(input1, input2, rbot1, rbot2, output, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)
        return output

    def backward(self, grad_output):
        input1, input2 = self.saved_tensors
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            grad_input1 = input1.new()
            grad_input2 = input2.new()
            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)
        return grad_input1, grad_input2


class Correlation(Module):

    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)(input1, input2)
        return result


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=True), nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


class LCCNet(nn.Module):
    """
    Based on the PWC-DC net. add resnet encoder, dilation convolution and densenet connections
    """

    def __init__(self, image_size, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0, Action_Func='leakyrelu', attention=False, res_num=18):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(LCCNet, self).__init__()
        input_lidar = 1
        self.res_num = res_num
        self.use_feat_from = use_feat_from
        if use_reflectance:
            input_lidar = 2
        self.pretrained_encoder = True
        self.net_encoder = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)
        self.Action_Func = Action_Func
        self.attention = attention
        self.inplanes = 64
        if self.res_num == 50:
            layers = [3, 4, 6, 3]
            add_list = [1024, 512, 256, 64]
        elif self.res_num == 18:
            layers = [2, 2, 2, 2]
            add_list = [256, 128, 64, 64]
        if self.attention:
            block = SEBottleneck
        elif self.res_num == 50:
            block = Bottleneck
        elif self.res_num == 18:
            block = BasicBlock
        self.conv1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.elu_rgb = nn.ELU()
        self.leakyRELU_rgb = nn.LeakyReLU(0.1)
        self.maxpool_rgb = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_rgb = self._make_layer(block, 64, layers[0])
        self.layer2_rgb = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_rgb = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_rgb = self._make_layer(block, 512, layers[3], stride=2)
        self.inplanes = 64
        self.conv1_lidar = nn.Conv2d(input_lidar, 64, kernel_size=7, stride=2, padding=3)
        self.elu_lidar = nn.ELU()
        self.leakyRELU_lidar = nn.LeakyReLU(0.1)
        self.maxpool_lidar = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar = self._make_layer(block, 64, layers[0])
        self.layer2_lidar = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_lidar = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_lidar = self._make_layer(block, 512, layers[3], stride=2)
        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])
        od = nd
        self.conv6_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)
        if use_feat_from > 1:
            self.predict_flow6 = predict_flow(od + dd[4])
            self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
            od = nd + add_list[0] + 4
            self.conv5_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv5_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv5_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv5_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv5_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)
        if use_feat_from > 2:
            self.predict_flow5 = predict_flow(od + dd[4])
            self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
            od = nd + add_list[1] + 4
            self.conv4_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv4_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv4_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv4_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv4_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)
        if use_feat_from > 3:
            self.predict_flow4 = predict_flow(od + dd[4])
            self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
            od = nd + add_list[2] + 4
            self.conv3_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv3_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv3_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv3_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv3_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)
        if use_feat_from > 4:
            self.predict_flow3 = predict_flow(od + dd[4])
            self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
            od = nd + add_list[3] + 4
            self.conv2_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv2_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv2_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv2_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv2_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)
        if use_feat_from > 5:
            self.predict_flow2 = predict_flow(od + dd[4])
            self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.dc_conv1 = myconv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
            self.dc_conv2 = myconv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
            self.dc_conv3 = myconv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
            self.dc_conv4 = myconv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
            self.dc_conv5 = myconv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
            self.dc_conv6 = myconv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
            self.dc_conv7 = predict_flow(32)
        fc_size = od + dd[4]
        downsample = 128 // 2 ** use_feat_from
        if image_size[0] % downsample == 0:
            fc_size *= image_size[0] // downsample
        else:
            fc_size *= image_size[0] // downsample + 1
        if image_size[1] % downsample == 0:
            fc_size *= image_size[1] // downsample
        else:
            fc_size *= image_size[1] // downsample + 1
        self.fc1 = nn.Linear(fc_size * 4, 512)
        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)
        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)
        self.dropout = nn.Dropout(dropout)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            grid = grid
        vgrid = Variable(grid) + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size()))
        mask = nn.functional.grid_sample(mask, vgrid)
        mask = torch.floor(torch.clamp(mask, 0, 1))
        return output * mask

    def forward(self, rgb, lidar):
        H, W = rgb.shape[2:4]
        if self.pretrained_encoder:
            features1 = self.net_encoder(rgb)
            c12 = features1[0]
            c13 = features1[2]
            c14 = features1[3]
            c15 = features1[4]
            c16 = features1[5]
            x2 = self.conv1_lidar(lidar)
            if self.Action_Func == 'leakyrelu':
                c22 = self.leakyRELU_lidar(x2)
            elif self.Action_Func == 'elu':
                c22 = self.elu_lidar(x2)
            c23 = self.layer1_lidar(self.maxpool_lidar(c22))
            c24 = self.layer2_lidar(c23)
            c25 = self.layer3_lidar(c24)
            c26 = self.layer4_lidar(c25)
        else:
            x1 = self.conv1_rgb(rgb)
            x2 = self.conv1_lidar(lidar)
            if self.Action_Func == 'leakyrelu':
                c12 = self.leakyRELU_rgb(x1)
                c22 = self.leakyRELU_lidar(x2)
            elif self.Action_Func == 'elu':
                c12 = self.elu_rgb(x1)
                c22 = self.elu_lidar(x2)
            c13 = self.layer1_rgb(self.maxpool_rgb(c12))
            c23 = self.layer1_lidar(self.maxpool_lidar(c22))
            c14 = self.layer2_rgb(c13)
            c24 = self.layer2_lidar(c23)
            c15 = self.layer3_rgb(c14)
            c25 = self.layer3_lidar(c24)
            c16 = self.layer4_rgb(c15)
            c26 = self.layer4_lidar(c25)
        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        if self.use_feat_from > 1:
            flow6 = self.predict_flow6(x)
            up_flow6 = self.deconv6(flow6)
            up_feat6 = self.upfeat6(x)
            warp5 = self.warp(c25, up_flow6 * 0.625)
            corr5 = self.corr(c15, warp5)
            corr5 = self.leakyRELU(corr5)
            x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
            x = torch.cat((self.conv5_0(x), x), 1)
            x = torch.cat((self.conv5_1(x), x), 1)
            x = torch.cat((self.conv5_2(x), x), 1)
            x = torch.cat((self.conv5_3(x), x), 1)
            x = torch.cat((self.conv5_4(x), x), 1)
        if self.use_feat_from > 2:
            flow5 = self.predict_flow5(x)
            up_flow5 = self.deconv5(flow5)
            up_feat5 = self.upfeat5(x)
            warp4 = self.warp(c24, up_flow5 * 1.25)
            corr4 = self.corr(c14, warp4)
            corr4 = self.leakyRELU(corr4)
            x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
            x = torch.cat((self.conv4_0(x), x), 1)
            x = torch.cat((self.conv4_1(x), x), 1)
            x = torch.cat((self.conv4_2(x), x), 1)
            x = torch.cat((self.conv4_3(x), x), 1)
            x = torch.cat((self.conv4_4(x), x), 1)
        if self.use_feat_from > 3:
            flow4 = self.predict_flow4(x)
            up_flow4 = self.deconv4(flow4)
            up_feat4 = self.upfeat4(x)
            warp3 = self.warp(c23, up_flow4 * 2.5)
            corr3 = self.corr(c13, warp3)
            corr3 = self.leakyRELU(corr3)
            x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
            x = torch.cat((self.conv3_0(x), x), 1)
            x = torch.cat((self.conv3_1(x), x), 1)
            x = torch.cat((self.conv3_2(x), x), 1)
            x = torch.cat((self.conv3_3(x), x), 1)
            x = torch.cat((self.conv3_4(x), x), 1)
        if self.use_feat_from > 4:
            flow3 = self.predict_flow3(x)
            up_flow3 = self.deconv3(flow3)
            up_feat3 = self.upfeat3(x)
            warp2 = self.warp(c22, up_flow3 * 5.0)
            corr2 = self.corr(c12, warp2)
            corr2 = self.leakyRELU(corr2)
            x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
            x = torch.cat((self.conv2_0(x), x), 1)
            x = torch.cat((self.conv2_1(x), x), 1)
            x = torch.cat((self.conv2_2(x), x), 1)
            x = torch.cat((self.conv2_3(x), x), 1)
            x = torch.cat((self.conv2_4(x), x), 1)
        if self.use_feat_from > 5:
            flow2 = self.predict_flow2(x)
            x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.leakyRELU(self.fc1(x))
        transl = self.leakyRELU(self.fc1_trasl(x))
        rot = self.leakyRELU(self.fc1_rot(x))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        rot = F.normalize(rot, dim=1)
        return transl, rot


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L1Loss,
     lambda: ([], {'rescale_trans': 1.0, 'rescale_rot': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_IIPCVLAB_LCCNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

