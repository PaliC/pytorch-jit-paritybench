import sys
_module = sys.modules[__name__]
del sys
Dataset = _module
pcdet = _module
config = _module
datasets = _module
augmentor = _module
augmentor_utils = _module
data_augmentor = _module
database_sampler = _module
dataset = _module
kitti = _module
kitti_dataset = _module
kitti_object_eval_python = _module
eval = _module
kitti_common = _module
rotate_iou = _module
kitti_utils = _module
lyft = _module
lyft_dataset = _module
lyft_mAP_eval = _module
lyft_eval = _module
lyft_utils = _module
nuscenes = _module
nuscenes_dataset = _module
nuscenes_utils = _module
pandaset = _module
pandaset_dataset = _module
processor = _module
data_processor = _module
point_feature_encoder = _module
waymo = _module
waymo_dataset = _module
waymo_eval = _module
waymo_utils = _module
models = _module
backbones_2d = _module
base_bev_backbone = _module
map_to_bev = _module
conv2d_collapse = _module
height_compression = _module
pointpillar_scatter = _module
backbones_3d = _module
pfe = _module
voxel_set_abstraction = _module
pointnet2_backbone = _module
spconv_backbone = _module
spconv_unet = _module
vfe = _module
dynamic_mean_vfe = _module
dynamic_pillar_vfe = _module
image_vfe = _module
image_vfe_modules = _module
f2v = _module
frustum_grid_generator = _module
frustum_to_voxel = _module
sampler = _module
ffn = _module
ddn = _module
ddn_deeplabv3 = _module
ddn_template = _module
ddn_loss = _module
balancer = _module
ddn_loss = _module
depth_ffn = _module
mean_vfe = _module
pillar_vfe = _module
vfe_template = _module
dense_heads = _module
anchor_head_multi = _module
anchor_head_single = _module
anchor_head_template = _module
center_head = _module
point_head_box = _module
point_head_simple = _module
point_head_template = _module
point_intra_part_head = _module
target_assigner = _module
anchor_generator = _module
atss_target_assigner = _module
axis_aligned_target_assigner = _module
PartA2_net = _module
detectors = _module
caddn = _module
centerpoint = _module
detector3d_template = _module
point_rcnn = _module
pointpillar = _module
pv_rcnn = _module
pv_rcnn_plusplus = _module
second_net = _module
second_net_iou = _module
voxel_rcnn = _module
model_utils = _module
basic_block_2d = _module
centernet_utils = _module
model_nms_utils = _module
roi_heads = _module
partA2_head = _module
pointrcnn_head = _module
pvrcnn_head = _module
roi_head_template = _module
second_head = _module
proposal_target_layer = _module
voxelrcnn_head = _module
ops = _module
iou3d_nms = _module
iou3d_nms_utils = _module
pointnet2 = _module
pointnet2_batch = _module
pointnet2_modules = _module
pointnet2_utils = _module
pointnet2_stack = _module
pointnet2_modules = _module
pointnet2_utils = _module
voxel_pool_modules = _module
voxel_query_utils = _module
roiaware_pool3d = _module
roiaware_pool3d_utils = _module
roipoint_pool3d = _module
roipoint_pool3d_utils = _module
utils = _module
box_coder_utils = _module
box_utils = _module
calibration_kitti = _module
common_utils = _module
commu_utils = _module
loss_utils = _module
object3d_kitti = _module
spconv_utils = _module
transform_utils = _module
setup = _module
_init_path = _module
demo = _module
eval_utils = _module
test = _module
train = _module
optimization = _module
fastai_optim = _module
learning_schedules_fastai = _module
train_utils = _module
open3d_vis_utils = _module
visualize_utils = _module
data_classes = _module
kitty_utils = _module
metrics = _module
pointnet2_modules = _module
pointnet2_utils = _module
pytorch_utils = _module
setup = _module
multihead_attention = _module
pointnet_transformer = _module
transformer = _module
linalg_utils = _module
pointnet2_modules = _module
pointnet2_utils = _module
searchspace = _module
sequence_dataset = _module
setup = _module
test_tracking = _module
test_tracking_waymo = _module
train_tracking = _module
train_tracking_waymo = _module
waymo_dataset = _module

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


from torch.utils.data import Dataset


import numpy as np


import pandas as pd


import torch


import logging


from functools import partial


import copy


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler as _DistributedSampler


import torch.distributed as dist


from collections import defaultdict


import torch.utils.data as torch_data


from collections import namedtuple


import torch.nn as nn


import math


import torch.nn.functional as F


from collections import OrderedDict


from torch import hub


from torch.nn.init import kaiming_normal_


from typing import List


from typing import Tuple


from torch.autograd import Function


from torch.autograd import Variable


import scipy


from scipy.spatial import Delaunay


import random


import torch.multiprocessing as mp


import time


from typing import Set


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import re


import torch.optim as optim


import torch.optim.lr_scheduler as lr_sched


from torch import nn


from torch._utils import _unflatten_dense_tensors


from torch.nn.utils import parameters_to_vector


from torch.nn.utils import clip_grad_norm_


import matplotlib


from scipy.optimize import leastsq


from typing import Optional


from torch import Tensor


from enum import Enum


import scipy.io as sio


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim.lr_scheduler as lr_scheduler


import torch.utils.data


class BaseBEVBackbone(nn.Module):

    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []
        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [nn.ZeroPad2d(1), nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3, stride=layer_strides[idx], padding=0, bias=False), nn.BatchNorm2d(num_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()]
            for k in range(layer_nums[idx]):
                cur_layers.extend([nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(num_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False), nn.BatchNorm2d(num_upsample_filters[idx], eps=0.001, momentum=0.01), nn.ReLU()))
        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False), nn.BatchNorm2d(c_in, eps=0.001, momentum=0.01), nn.ReLU()))
        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        data_dict['spatial_features_2d'] = x
        return data_dict


class BasicBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initializes convolutional block
        Args:
            in_channels: int, Number of input channels
            out_channels: int, Number of output channels
            **kwargs: Dict, Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features: (B, C_in, H, W), Input features
        Returns:
            x: (B, C_out, H, W), Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DCollapse(nn.Module):

    def __init__(self, model_cfg, grid_size):
        """
        Initializes 2D convolution collapse module
        Args:
            model_cfg: EasyDict, Model configuration
            grid_size: (X, Y, Z) Voxel grid size
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.block = BasicBlock2D(in_channels=self.num_bev_features * self.num_heights, out_channels=self.num_bev_features, **self.model_cfg.ARGS)

    def forward(self, batch_dict):
        """
        Collapses voxel features to BEV via concatenation and channel reduction
        Args:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Voxel feature representation
        Returns:
            batch_dict:
                spatial_features: (B, C, Y, X), BEV feature representation
        """
        voxel_features = batch_dict['voxel_features']
        bev_features = voxel_features.flatten(start_dim=1, end_dim=2)
        bev_features = self.block(bev_features)
        batch_dict['spatial_features'] = bev_features
        return batch_dict


class HeightCompression(nn.Module):

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict


class PointPillarScatter(nn.Module):

    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(self.num_bev_features, self.nz * self.nx * self.ny, dtype=pillar_features.dtype, device=pillar_features.device)
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)
    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]
    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t(torch.t(Ia) * wa) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


def sample_points_with_roi(rois, points, sample_radius_with_roi, num_max_points_of_part=200000):
    """
    Args:
        rois: (M, 7 + C)
        points: (N, 3)
        sample_radius_with_roi:
        num_max_points_of_part:

    Returns:
        sampled_points: (N_out, 3)
    """
    if points.shape[0] < num_max_points_of_part:
        distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
        min_dis, min_dis_roi_idx = distance.min(dim=-1)
        roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
        point_mask = min_dis < roi_max_dim + sample_radius_with_roi
    else:
        start_idx = 0
        point_mask_list = []
        while start_idx < points.shape[0]:
            distance = (points[start_idx:start_idx + num_max_points_of_part, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            cur_point_mask = min_dis < roi_max_dim + sample_radius_with_roi
            point_mask_list.append(cur_point_mask)
            start_idx += num_max_points_of_part
        point_mask = torch.cat(point_mask_list, dim=0)
    sampled_points = points[:1] if point_mask.sum() == 0 else points[point_mask, :]
    return sampled_points, point_mask


def sector_fps(points, num_sampled_points, num_sectors):
    """
    Args:
        points: (N, 3)
        num_sampled_points: int
        num_sectors: int

    Returns:
        sampled_points: (N_out, 3)
    """
    sector_size = np.pi * 2 / num_sectors
    point_angles = torch.atan2(points[:, 1], points[:, 0]) + np.pi
    sector_idx = (point_angles / sector_size).floor().clamp(min=0, max=num_sectors)
    xyz_points_list = []
    xyz_batch_cnt = []
    num_sampled_points_list = []
    for k in range(num_sectors):
        mask = sector_idx == k
        cur_num_points = mask.sum().item()
        if cur_num_points > 0:
            xyz_points_list.append(points[mask])
            xyz_batch_cnt.append(cur_num_points)
            ratio = cur_num_points / points.shape[0]
            num_sampled_points_list.append(min(cur_num_points, math.ceil(ratio * num_sampled_points)))
    if len(xyz_batch_cnt) == 0:
        xyz_points_list.append(points)
        xyz_batch_cnt.append(len(points))
        num_sampled_points_list.append(num_sampled_points)
        None
    xyz = torch.cat(xyz_points_list, dim=0)
    xyz_batch_cnt = torch.tensor(xyz_batch_cnt, device=points.device).int()
    sampled_points_batch_cnt = torch.tensor(num_sampled_points_list, device=points.device).int()
    sampled_pt_idxs = pointnet2_stack_utils.stack_farthest_point_sample(xyz.contiguous(), xyz_batch_cnt, sampled_points_batch_cnt).long()
    sampled_points = xyz[sampled_pt_idxs]
    return sampled_points


class VoxelSetAbstraction(nn.Module):

    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None, num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        SA_cfg = self.model_cfg.SA_LAYER
        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            if SA_cfg[src_name].get('INPUT_CHANNELS', None) is None:
                input_channels = SA_cfg[src_name].MLPS[0][0] if isinstance(SA_cfg[src_name].MLPS[0], list) else SA_cfg[src_name].MLPS[0]
            else:
                input_channels = SA_cfg[src_name]['INPUT_CHANNELS']
            cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(input_channels=input_channels, config=SA_cfg[src_name])
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)
            c_in += cur_num_c_out
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev
        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            self.SA_rawpoints, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(input_channels=num_rawpoint_features - 3, config=SA_cfg['raw_points'])
            c_in += cur_num_c_out
        self.vsa_point_feature_fusion = nn.Sequential(nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False), nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES), nn.ReLU())
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        """
        Args:
            keypoints: (N1 + N2 + ..., 4)
            bev_features: (B, C, H, W)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride
        point_bev_features_list = []
        for k in range(batch_size):
            bs_mask = keypoints[:, 0] == k
            cur_x_idxs = x_idxs[bs_mask]
            cur_y_idxs = y_idxs[bs_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)
        point_bev_features = torch.cat(point_bev_features_list, dim=0)
        return point_bev_features

    def sectorized_proposal_centric_sampling(self, roi_boxes, points):
        """
        Args:
            roi_boxes: (M, 7 + C)
            points: (N, 3)

        Returns:
            sampled_points: (N_out, 3)
        """
        sampled_points, _ = sample_points_with_roi(rois=roi_boxes, points=points, sample_radius_with_roi=self.model_cfg.SPC_SAMPLING.SAMPLE_RADIUS_WITH_ROI, num_max_points_of_part=self.model_cfg.SPC_SAMPLING.get('NUM_POINTS_OF_EACH_SAMPLE_PART', 200000))
        sampled_points = sector_fps(points=sampled_points, num_sampled_points=self.model_cfg.NUM_KEYPOINTS, num_sectors=self.model_cfg.SPC_SAMPLING.NUM_SECTORS)
        return sampled_points

    def get_sampled_points(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:
            keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
        """
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(batch_dict['voxel_coords'][:, 1:4], downsample_times=1, voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = batch_indices == bs_idx
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS).long()
                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]
                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            elif self.model_cfg.SAMPLE_METHOD == 'SPC':
                cur_keypoints = self.sectorized_proposal_centric_sampling(roi_boxes=batch_dict['rois'][bs_idx], points=sampled_points[0])
                bs_idxs = cur_keypoints.new_ones(cur_keypoints.shape[0]) * bs_idx
                keypoints = torch.cat((bs_idxs[:, None], cur_keypoints), dim=1)
            else:
                raise NotImplementedError
            keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)
        if len(keypoints.shape) == 3:
            batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1, 1)
            keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 3)), dim=1)
        return keypoints

    @staticmethod
    def aggregate_keypoint_features_from_one_source(batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt, filter_neighbors_with_roi=False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None):
        """

        Args:
            aggregate_func:
            xyz: (N, 3)
            xyz_features: (N, C)
            xyz_bs_idxs: (N)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

            filter_neighbors_with_roi: True/False
            radius_of_neighbor: float
            num_max_points_of_part: int
            rois: (batch_size, num_rois, 7 + C)
        Returns:

        """
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        if filter_neighbors_with_roi:
            point_features = torch.cat((xyz, xyz_features), dim=-1) if xyz_features is not None else xyz
            point_features_list = []
            for bs_idx in range(batch_size):
                bs_mask = xyz_bs_idxs == bs_idx
                _, valid_mask = sample_points_with_roi(rois=rois[bs_idx], points=xyz[bs_mask], sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part)
                point_features_list.append(point_features[bs_mask][valid_mask])
                xyz_batch_cnt[bs_idx] = valid_mask.sum()
            valid_point_features = torch.cat(point_features_list, dim=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:, 3:] if xyz_features is not None else None
        else:
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum()
        pooled_points, pooled_features = aggregate_func(xyz=xyz.contiguous(), xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, features=xyz_features.contiguous())
        return pooled_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints = self.get_sampled_points(batch_dict)
        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(keypoints, batch_dict['spatial_features'], batch_dict['batch_size'], bev_stride=batch_dict['spatial_features_stride'])
            point_features_list.append(point_bev_features)
        batch_size = batch_dict['batch_size']
        new_xyz = keypoints[:, 1:4].contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum()
        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            pooled_features = self.aggregate_keypoint_features_from_one_source(batch_size=batch_size, aggregate_func=self.SA_rawpoints, xyz=raw_points[:, 1:4], xyz_features=raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None, xyz_bs_idxs=raw_points[:, 0], new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, filter_neighbors_with_roi=self.model_cfg.SA_LAYER['raw_points'].get('FILTER_NEIGHBOR_WITH_ROI', False), radius_of_neighbor=self.model_cfg.SA_LAYER['raw_points'].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None), rois=batch_dict.get('rois', None))
            point_features_list.append(pooled_features)
        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            cur_features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()
            xyz = common_utils.get_voxel_centers(cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name], voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
            pooled_features = self.aggregate_keypoint_features_from_one_source(batch_size=batch_size, aggregate_func=self.SA_layers[k], xyz=xyz.contiguous(), xyz_features=cur_features, xyz_bs_idxs=cur_coords[:, 0], new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt, filter_neighbors_with_roi=self.model_cfg.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False), radius_of_neighbor=self.model_cfg.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None), rois=batch_dict.get('rois', None))
            point_features_list.append(pooled_features)
        point_features = torch.cat(point_features_list, dim=-1)
        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))
        batch_dict['point_features'] = point_features
        batch_dict['point_coords'] = keypoints
        return batch_dict


class PointNet2MSG(nn.Module):

    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            self.SA_modules.append(pointnet2_modules.PointnetSAModuleMSG(npoint=self.model_cfg.SA_CONFIG.NPOINTS[k], radii=self.model_cfg.SA_CONFIG.RADIUS[k], nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k], mlps=mlps, use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True)))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(pointnet2_modules.PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]))
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i])
        point_features = l_features[0].permute(0, 2, 1).contiguous()
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        return batch_dict


class PointNet2Backbone(nn.Module):
    """
    DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
    """

    def __init__(self, model_cfg, input_channels, **kwargs):
        assert False, 'DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723'
        super().__init__()
        self.model_cfg = model_cfg
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3
        self.num_points_each_layer = []
        skip_channel_list = [input_channels]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            self.SA_modules.append(pointnet2_modules_stack.StackSAModuleMSG(radii=self.model_cfg.SA_CONFIG.RADIUS[k], nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k], mlps=mlps, use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True)))
            skip_channel_list.append(channel_out)
            channel_in = channel_out
        self.FP_modules = nn.ModuleList()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(pointnet2_modules_stack.StackPointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]))
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = pc[:, 4:].contiguous() if pc.size(-1) > 4 else None
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
        for i in range(len(self.SA_modules)):
            new_xyz_list = []
            for k in range(batch_size):
                if len(l_xyz) == 1:
                    cur_xyz = l_xyz[0][batch_idx == k]
                else:
                    last_num_points = self.num_points_each_layer[i - 1]
                    cur_xyz = l_xyz[-1][k * last_num_points:(k + 1) * last_num_points]
                cur_pt_idxs = pointnet2_utils_stack.farthest_point_sample(cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]).long()[0]
                if cur_xyz.shape[0] < self.num_points_each_layer[i]:
                    empty_num = self.num_points_each_layer[i] - cur_xyz.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
                new_xyz_list.append(cur_xyz[cur_pt_idxs])
            new_xyz = torch.cat(new_xyz_list, dim=0)
            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
            li_xyz, li_features = self.SA_modules[i](xyz=l_xyz[i], features=l_features[i], xyz_batch_cnt=l_batch_cnt[i], new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_batch_cnt.append(new_xyz_batch_cnt)
        l_features[0] = points[:, 1:]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](unknown=l_xyz[i - 1], unknown_batch_cnt=l_batch_cnt[i - 1], known=l_xyz[i], known_batch_cnt=l_batch_cnt[i], unknown_feats=l_features[i - 1], known_feats=l_features[i])
        batch_dict['point_features'] = l_features[0]
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0]), dim=1)
        return batch_dict


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError
    m = spconv.SparseSequential(conv, norm_fn(out_channels), nn.ReLU())
    return m


class VoxelBackBone8x(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = post_act_block
        self.conv1 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'))
        self.conv4 = spconv.SparseSequential(block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'))
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU())
        self.num_point_features = 128
        self.backbone_channels = {'x_conv1': 16, 'x_conv2': 32, 'x_conv3': 64, 'x_conv4': 64}

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        batch_dict.update({'encoded_spconv_tensor': out, 'encoded_spconv_tensor_stride': 8})
        batch_dict.update({'multi_scale_3d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4}})
        batch_dict.update({'multi_scale_3d_strides': {'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8}})
        return batch_dict


def replace_feature(out, new_features):
    if 'replace_feature' in out.__dir__():
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class VoxelResBackBone8x(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = post_act_block
        self.conv1 = spconv.SparseSequential(SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'), SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'), SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'), SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'))
        self.conv4 = spconv.SparseSequential(block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'), SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'))
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU())
        self.num_point_features = 128
        self.backbone_channels = {'x_conv1': 16, 'x_conv2': 32, 'x_conv3': 64, 'x_conv4': 128}

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        batch_dict.update({'encoded_spconv_tensor': out, 'encoded_spconv_tensor_stride': 8})
        batch_dict.update({'multi_scale_3d_features': {'x_conv1': x_conv1, 'x_conv2': x_conv2, 'x_conv3': x_conv3, 'x_conv4': x_conv4}})
        batch_dict.update({'multi_scale_3d_strides': {'x_conv1': 1, 'x_conv2': 2, 'x_conv3': 4, 'x_conv4': 8}})
        return batch_dict


class UNetV2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        norm_fn = partial(nn.BatchNorm1d, eps=0.001, momentum=0.01)
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'), norm_fn(16), nn.ReLU())
        block = post_act_block
        self.conv1 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.conv2 = spconv.SparseSequential(block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'), block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'))
        self.conv3 = spconv.SparseSequential(block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'))
        self.conv4 = spconv.SparseSequential(block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'), block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'))
        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
            last_pad = self.model_cfg.get('last_pad', 0)
            self.conv_out = spconv.SparseSequential(spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'), norm_fn(128), nn.ReLU())
        else:
            self.conv_out = None
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')
        self.conv5 = spconv.SparseSequential(block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'))
        self.num_point_features = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert in_channels % out_channels == 0 and in_channels >= out_channels
        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords.int(), spatial_shape=self.sparse_shape, batch_size=batch_size)
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        if self.conv_out is not None:
            out = self.conv_out(x_conv4)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)
        batch_dict['point_features'] = x_up1.features
        point_coords = common_utils.get_voxel_centers(x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
        batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
        return batch_dict


class PFNLayerV2(nn.Module):

    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):
        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class Sampler(nn.Module):

    def __init__(self, mode='bilinear', padding_mode='zeros'):
        """
        Initializes module
        Args:
            mode: string, Sampling mode [bilinear/nearest]
            padding_mode: string, Padding mode for outside grid values [zeros/border/reflection]
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, input_features, grid):
        """
        Samples input using sampling grid
        Args:
            input_features: (B, C, D, H, W), Input frustum features
            grid: (B, X, Y, Z, 3), Sampling grids for input features
        Returns
            output_features: (B, C, X, Y, Z) Output voxel features
        """
        output = F.grid_sample(input=input_features, grid=grid, mode=self.mode, padding_mode=self.padding_mode)
        return output


class FrustumToVoxel(nn.Module):

    def __init__(self, model_cfg, grid_size, pc_range, disc_cfg):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            model_cfg: EasyDict, Module configuration
            grid_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
            disc_cfg: EasyDict, Depth discretiziation configuration
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.disc_cfg = disc_cfg
        self.grid_generator = FrustumGridGenerator(grid_size=grid_size, pc_range=pc_range, disc_cfg=disc_cfg)
        self.sampler = Sampler(**model_cfg.SAMPLER)

    def forward(self, batch_dict):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                frustum_features: (B, C, D, H_image, W_image), Image frustum features
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
        """
        grid = self.grid_generator(lidar_to_cam=batch_dict['trans_lidar_to_cam'], cam_to_img=batch_dict['trans_cam_to_img'], image_shape=batch_dict['image_shape'])
        voxel_features = self.sampler(input_features=batch_dict['frustum_features'], grid=grid)
        voxel_features = voxel_features.permute(0, 1, 4, 3, 2)
        batch_dict['voxel_features'] = voxel_features
        return batch_dict


class DDNTemplate(nn.Module):

    def __init__(self, constructor, feat_extract_layer, num_classes, pretrained_path=None, aux_loss=None):
        """
        Initializes depth distribution network.
        Args:
            constructor: function, Model constructor
            feat_extract_layer: string, Layer to extract features from
            num_classes: int, Number of classes
            pretrained_path: string, (Optional) Path of the model to load weights from
            aux_loss: bool, Flag to include auxillary loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss
        if self.pretrained:
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])
        self.model = self.get_model(constructor=constructor)
        self.feat_extract_layer = feat_extract_layer
        self.model.backbone.return_layers = {feat_extract_layer: 'features', **self.model.backbone.return_layers}

    def get_model(self, constructor):
        """
        Get model
        Args:
            constructor: function, Model constructor
        Returns:
            model: nn.Module, Model
        """
        model = constructor(pretrained=False, pretrained_backbone=False, num_classes=self.num_classes, aux_loss=self.aux_loss)
        if self.pretrained_path is not None:
            model_dict = model.state_dict()
            checkpoint_path = Path(self.pretrained_path)
            if not checkpoint_path.exists():
                checkpoint = checkpoint_path.name
                save_dir = checkpoint_path.parent
                save_dir.mkdir(parents=True)
                url = f'https://download.pytorch.org/models/{checkpoint}'
                hub.load_state_dict_from_url(url, save_dir)
            pretrained_dict = torch.load(self.pretrained_path)
            pretrained_dict = self.filter_pretrained_dict(model_dict=model_dict, pretrained_dict=pretrained_dict)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        return model

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        """
        Removes layers from pretrained state dict that are not used or changed in model
        Args:
            model_dict: dict, Default model state dictionary
            pretrained_dict: dict, Pretrained model state dictionary
        Returns:
            pretrained_dict: dict, Pretrained model state dictionary with removed weights
        """
        if 'aux_classifier.0.weight' in pretrained_dict and 'aux_classifier.0.weight' not in model_dict:
            pretrained_dict = {key: value for key, value in pretrained_dict.items() if 'aux_classifier' not in key}
        model_num_classes = model_dict['classifier.4.weight'].shape[0]
        pretrained_num_classes = pretrained_dict['classifier.4.weight'].shape[0]
        if model_num_classes != pretrained_num_classes:
            pretrained_dict.pop('classifier.4.weight')
            pretrained_dict.pop('classifier.4.bias')
        return pretrained_dict

    def forward(self, images):
        """
        Forward pass
        Args:
            images: (N, 3, H_in, W_in), Input images
        Returns
            result: dict[torch.Tensor], Depth distribution result
                features: (N, C, H_out, W_out), Image features
                logits: (N, num_classes, H_out, W_out), Classification logits
                aux: (N, num_classes, H_out, W_out), Auxillary classification logits
        """
        x = self.preprocess(images)
        result = OrderedDict()
        features = self.model.backbone(x)
        result['features'] = features['features']
        feat_shape = features['features'].shape[-2:]
        x = features['out']
        x = self.model.classifier(x)
        x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
        result['logits'] = x
        if self.model.aux_classifier is not None:
            x = features['aux']
            x = self.model.aux_classifier(x)
            x = F.interpolate(x, size=feat_shape, mode='bilinear', align_corners=False)
            result['aux'] = x
        return result

    def preprocess(self, images):
        """
        Preprocess images
        Args:
            images: (N, 3, H, W), Input images
        Return
            x: (N, 3, H, W), Preprocessed images
        """
        x = images
        if self.pretrained:
            mask = torch.isnan(x)
            x = normalize(x, mean=self.norm_mean, std=self.norm_std)
            x[mask] = 0
        return x


class Balancer(nn.Module):

    def __init__(self, fg_weight, bg_weight, downsample_factor=1):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def forward(self, loss, gt_boxes2d):
        """
        Forward pass
        Args:
            loss: (B, H, W), Pixel-wise loss
            gt_boxes2d: (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Total loss after foreground/background balancing
            tb_dict: dict[float], All losses to log in tensorboard
        """
        fg_mask = loss_utils.compute_fg_mask(gt_boxes2d=gt_boxes2d, shape=loss.shape, downsample_factor=self.downsample_factor, device=loss.device)
        bg_mask = ~fg_mask
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()
        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels
        loss = fg_loss + bg_loss
        tb_dict = {'balancer_loss': loss.item(), 'fg_loss': fg_loss.item(), 'bg_loss': bg_loss.item()}
        return loss, tb_dict


class DDNLoss(nn.Module):

    def __init__(self, weight, alpha, gamma, disc_cfg, fg_weight, bg_weight, downsample_factor):
        """
        Initializes DDNLoss module
        Args:
            weight: float, Loss function weight
            alpha: float, Alpha value for Focal Loss
            gamma: float, Gamma value for Focal Loss
            disc_cfg: dict, Depth discretiziation configuration
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        self.disc_cfg = disc_cfg
        self.balancer = Balancer(downsample_factor=downsample_factor, fg_weight=fg_weight, bg_weight=bg_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction='none')
        self.weight = weight

    def forward(self, depth_logits, depth_maps, gt_boxes2d):
        """
        Gets DDN loss
        Args:
            depth_logits: (B, D+1, H, W), Predicted depth logits
            depth_maps: (B, H, W), Depth map [m]
            gt_boxes2d: torch.Tensor (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        tb_dict = {}
        depth_target = transform_utils.bin_depths(depth_maps, **self.disc_cfg, target=True)
        loss = self.loss_func(depth_logits, depth_target)
        loss, tb_dict = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d)
        loss *= self.weight
        tb_dict.update({'ddn_loss': loss.item()})
        return loss, tb_dict


class DepthFFN(nn.Module):

    def __init__(self, model_cfg, downsample_factor):
        """
        Initialize frustum feature network via depth distribution estimation
        Args:
            model_cfg: EasyDict, Depth classification network config
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.disc_cfg = model_cfg.DISCRETIZE
        self.downsample_factor = downsample_factor
        self.ddn = ddn.__all__[model_cfg.DDN.NAME](num_classes=self.disc_cfg['num_bins'] + 1, backbone_name=model_cfg.DDN.BACKBONE_NAME, **model_cfg.DDN.ARGS)
        self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)
        self.ddn_loss = ddn_loss.__all__[model_cfg.LOSS.NAME](disc_cfg=self.disc_cfg, downsample_factor=downsample_factor, **model_cfg.LOSS.ARGS)
        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
        return self.channel_reduce.out_channels

    def forward(self, batch_dict):
        """
        Predicts depths and creates image depth feature volume using depth distributions
        Args:
            batch_dict:
                images: (N, 3, H_in, W_in), Input images
        Returns:
            batch_dict:
                frustum_features: (N, C, D, H_out, W_out), Image depth features
        """
        images = batch_dict['images']
        ddn_result = self.ddn(images)
        image_features = ddn_result['features']
        depth_logits = ddn_result['logits']
        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features)
        frustum_features = self.create_frustum_features(image_features=image_features, depth_logits=depth_logits)
        batch_dict['frustum_features'] = frustum_features
        if self.training:
            self.forward_ret_dict['depth_maps'] = batch_dict['depth_maps']
            self.forward_ret_dict['gt_boxes2d'] = batch_dict['gt_boxes2d']
            self.forward_ret_dict['depth_logits'] = depth_logits
        return batch_dict

    def create_frustum_features(self, image_features, depth_logits):
        """
        Create image depth feature volume by multiplying image features with depth distributions
        Args:
            image_features: (N, C, H, W), Image features
            depth_logits: (N, D+1, H, W), Depth classification logits
        Returns:
            frustum_features: (N, C, D, H, W), Image features
        """
        channel_dim = 1
        depth_dim = 2
        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)
        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]
        frustum_features = depth_probs * image_features
        return frustum_features

    def get_loss(self):
        """
        Gets DDN loss
        Args:
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        loss, tb_dict = self.ddn_loss(**self.forward_ret_dict)
        return loss, tb_dict


class PFNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part * self.part:(num_part + 1) * self.part]) for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class VFETemplate(nn.Module):

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError


class SingleHead(BaseBEVBackbone):

    def __init__(self, model_cfg, input_channels, num_class, num_anchors_per_location, code_size, rpn_head_cfg=None, head_label_indices=None, separate_reg_config=None):
        super().__init__(rpn_head_cfg, input_channels)
        self.num_anchors_per_location = num_anchors_per_location
        self.num_class = num_class
        self.code_size = code_size
        self.model_cfg = model_cfg
        self.separate_reg_config = separate_reg_config
        self.register_buffer('head_label_indices', head_label_indices)
        if self.separate_reg_config is not None:
            code_size_cnt = 0
            self.conv_box = nn.ModuleDict()
            self.conv_box_names = []
            num_middle_conv = self.separate_reg_config.NUM_MIDDLE_CONV
            num_middle_filter = self.separate_reg_config.NUM_MIDDLE_FILTER
            conv_cls_list = []
            c_in = input_channels
            for k in range(num_middle_conv):
                conv_cls_list.extend([nn.Conv2d(c_in, num_middle_filter, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(num_middle_filter), nn.ReLU()])
                c_in = num_middle_filter
            conv_cls_list.append(nn.Conv2d(c_in, self.num_anchors_per_location * self.num_class, kernel_size=3, stride=1, padding=1))
            self.conv_cls = nn.Sequential(*conv_cls_list)
            for reg_config in self.separate_reg_config.REG_LIST:
                reg_name, reg_channel = reg_config.split(':')
                reg_channel = int(reg_channel)
                cur_conv_list = []
                c_in = input_channels
                for k in range(num_middle_conv):
                    cur_conv_list.extend([nn.Conv2d(c_in, num_middle_filter, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(num_middle_filter), nn.ReLU()])
                    c_in = num_middle_filter
                cur_conv_list.append(nn.Conv2d(c_in, self.num_anchors_per_location * int(reg_channel), kernel_size=3, stride=1, padding=1, bias=True))
                code_size_cnt += reg_channel
                self.conv_box[f'conv_{reg_name}'] = nn.Sequential(*cur_conv_list)
                self.conv_box_names.append(f'conv_{reg_name}')
            for m in self.conv_box.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            assert code_size_cnt == code_size, f'Code size does not match: {code_size_cnt}:{code_size}'
        else:
            self.conv_cls = nn.Conv2d(input_channels, self.num_anchors_per_location * self.num_class, kernel_size=1)
            self.conv_box = nn.Conv2d(input_channels, self.num_anchors_per_location * self.code_size, kernel_size=1)
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(input_channels, self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS, kernel_size=1)
        else:
            self.conv_dir_cls = None
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        if isinstance(self.conv_cls, nn.Conv2d):
            nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        else:
            nn.init.constant_(self.conv_cls[-1].bias, -np.log((1 - pi) / pi))

    def forward(self, spatial_features_2d):
        ret_dict = {}
        spatial_features_2d = super().forward({'spatial_features': spatial_features_2d})['spatial_features_2d']
        cls_preds = self.conv_cls(spatial_features_2d)
        if self.separate_reg_config is None:
            box_preds = self.conv_box(spatial_features_2d)
        else:
            box_preds_list = []
            for reg_name in self.conv_box_names:
                box_preds_list.append(self.conv_box[reg_name](spatial_features_2d))
            box_preds = torch.cat(box_preds_list, dim=1)
        if not self.use_multihead:
            box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
            cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            H, W = box_preds.shape[2:]
            batch_size = box_preds.shape[0]
            box_preds = box_preds.view(-1, self.num_anchors_per_location, self.code_size, H, W).permute(0, 1, 3, 4, 2).contiguous()
            cls_preds = cls_preds.view(-1, self.num_anchors_per_location, self.num_class, H, W).permute(0, 1, 3, 4, 2).contiguous()
            box_preds = box_preds.view(batch_size, -1, self.code_size)
            cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            if self.use_multihead:
                dir_cls_preds = dir_cls_preds.view(-1, self.num_anchors_per_location, self.model_cfg.NUM_DIR_BINS, H, W).permute(0, 1, 3, 4, 2).contiguous()
                dir_cls_preds = dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            else:
                dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        else:
            dir_cls_preds = None
        ret_dict['cls_preds'] = cls_preds
        ret_dict['box_preds'] = box_preds
        ret_dict['dir_cls_preds'] = dir_cls_preds
        return ret_dict


class ATSSTargetAssigner(object):
    """
    Reference: https://arxiv.org/abs/1912.02424
    """

    def __init__(self, topk, box_coder, match_height=False):
        self.topk = topk
        self.box_coder = box_coder
        self.match_height = match_height

    def assign_targets(self, anchors_list, gt_boxes_with_classes, use_multihead=False):
        """
        Args:
            anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8)
        Returns:

        """
        if not isinstance(anchors_list, list):
            anchors_list = [anchors_list]
            single_set_of_anchor = True
        else:
            single_set_of_anchor = len(anchors_list) == 1
        cls_labels_list, reg_targets_list, reg_weights_list = [], [], []
        for anchors in anchors_list:
            batch_size = gt_boxes_with_classes.shape[0]
            gt_classes = gt_boxes_with_classes[:, :, -1]
            gt_boxes = gt_boxes_with_classes[:, :, :-1]
            if use_multihead:
                anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
            else:
                anchors = anchors.view(-1, anchors.shape[-1])
            cls_labels, reg_targets, reg_weights = [], [], []
            for k in range(batch_size):
                cur_gt = gt_boxes[k]
                cnt = cur_gt.__len__() - 1
                while cnt > 0 and cur_gt[cnt].sum() == 0:
                    cnt -= 1
                cur_gt = cur_gt[:cnt + 1]
                cur_gt_classes = gt_classes[k][:cnt + 1]
                cur_cls_labels, cur_reg_targets, cur_reg_weights = self.assign_targets_single(anchors, cur_gt, cur_gt_classes)
                cls_labels.append(cur_cls_labels)
                reg_targets.append(cur_reg_targets)
                reg_weights.append(cur_reg_weights)
            cls_labels = torch.stack(cls_labels, dim=0)
            reg_targets = torch.stack(reg_targets, dim=0)
            reg_weights = torch.stack(reg_weights, dim=0)
            cls_labels_list.append(cls_labels)
            reg_targets_list.append(reg_targets)
            reg_weights_list.append(reg_weights)
        if single_set_of_anchor:
            ret_dict = {'box_cls_labels': cls_labels_list[0], 'box_reg_targets': reg_targets_list[0], 'reg_weights': reg_weights_list[0]}
        else:
            ret_dict = {'box_cls_labels': torch.cat(cls_labels_list, dim=1), 'box_reg_targets': torch.cat(reg_targets_list, dim=1), 'reg_weights': torch.cat(reg_weights_list, dim=1)}
        return ret_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes):
        """
        Args:
            anchors: (N, 7) [x, y, z, dx, dy, dz, heading]
            gt_boxes: (M, 7) [x, y, z, dx, dy, dz, heading]
            gt_classes: (M)
        Returns:

        """
        num_anchor = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        if self.match_height:
            ious = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7])
        else:
            ious = iou3d_nms_utils.boxes_iou_bev(anchors[:, 0:7], gt_boxes[:, 0:7])
        distance = (anchors[:, None, 0:3] - gt_boxes[None, :, 0:3]).norm(dim=-1)
        _, topk_idxs = distance.topk(self.topk, dim=0, largest=False)
        candidate_ious = ious[topk_idxs, torch.arange(num_gt)]
        iou_mean_per_gt = candidate_ious.mean(dim=0)
        iou_std_per_gt = candidate_ious.std(dim=0)
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt + 1e-06
        is_pos = candidate_ious >= iou_thresh_per_gt[None, :]
        candidate_anchors = anchors[topk_idxs.view(-1)]
        gt_boxes_of_each_anchor = gt_boxes[:, :].repeat(self.topk, 1)
        xyz_local = candidate_anchors[:, 0:3] - gt_boxes_of_each_anchor[:, 0:3]
        xyz_local = common_utils.rotate_points_along_z(xyz_local[:, None, :], -gt_boxes_of_each_anchor[:, 6]).squeeze(dim=1)
        xy_local = xyz_local[:, 0:2]
        lw = gt_boxes_of_each_anchor[:, 3:5][:, [1, 0]]
        is_in_gt = ((xy_local <= lw / 2) & (xy_local >= -lw / 2)).all(dim=-1).view(-1, num_gt)
        is_pos = is_pos & is_in_gt
        for ng in range(num_gt):
            topk_idxs[:, ng] += ng * num_anchor
        INF = -2147483647
        ious_inf = torch.full_like(ious, INF).t().contiguous().view(-1)
        index = topk_idxs.view(-1)[is_pos.view(-1)]
        ious_inf[index] = ious.t().contiguous().view(-1)[index]
        ious_inf = ious_inf.view(num_gt, -1).t()
        anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
        max_iou_of_each_gt, argmax_iou_of_each_gt = ious.max(dim=0)
        anchors_to_gt_indexs[argmax_iou_of_each_gt] = torch.arange(0, num_gt, device=ious.device)
        anchors_to_gt_values[argmax_iou_of_each_gt] = max_iou_of_each_gt
        cls_labels = gt_classes[anchors_to_gt_indexs]
        cls_labels[anchors_to_gt_values == INF] = 0
        matched_gts = gt_boxes[anchors_to_gt_indexs]
        pos_mask = cls_labels > 0
        reg_targets = matched_gts.new_zeros((num_anchor, self.box_coder.code_size))
        reg_weights = matched_gts.new_zeros(num_anchor)
        if pos_mask.sum() > 0:
            reg_targets[pos_mask > 0] = self.box_coder.encode_torch(matched_gts[pos_mask > 0], anchors[pos_mask > 0])
            reg_weights[pos_mask] = 1.0
        return cls_labels, reg_targets, reg_weights


class AnchorGenerator(object):

    def __init__(self, anchor_range, anchor_generator_config):
        super().__init__()
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range
        self.anchor_sizes = [config['anchor_sizes'] for config in anchor_generator_config]
        self.anchor_rotations = [config['anchor_rotations'] for config in anchor_generator_config]
        self.anchor_heights = [config['anchor_bottom_heights'] for config in anchor_generator_config]
        self.align_center = [config.get('align_center', False) for config in anchor_generator_config]
        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(self.anchor_heights)
        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes):
        assert len(grid_sizes) == self.num_of_anchor_sets
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(grid_sizes, self.anchor_sizes, self.anchor_rotations, self.anchor_heights, self.align_center):
            num_anchors_per_location.append(len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0
            x_shifts = torch.arange(self.anchor_range[0] + x_offset, self.anchor_range[3] + 1e-05, step=x_stride, dtype=torch.float32)
            y_shifts = torch.arange(self.anchor_range[1] + y_offset, self.anchor_range[4] + 1e-05, step=y_stride, dtype=torch.float32)
            z_shifts = x_shifts.new_tensor(anchor_height)
            num_anchor_size, num_anchor_rotation = anchor_size.__len__(), anchor_rotation.__len__()
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            x_shifts, y_shifts, z_shifts = torch.meshgrid([x_shifts, y_shifts, z_shifts])
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat([*anchors.shape[0:3], num_anchor_size, 1, 1])
            anchors = torch.cat((anchors, anchor_rotation), dim=-1)
            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            anchors[..., 2] += anchors[..., 5] / 2
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location


class AxisAlignedTargetAssigner(object):

    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()
        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']
        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)

    def assign_targets(self, all_anchors, gt_boxes_with_classes):
        """
        Args:
            all_anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8)
        Returns:

        """
        bbox_targets = []
        cls_labels = []
        reg_weights = []
        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]
        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()
            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([(self.class_names[c - 1] == anchor_class_name) for c in cur_gt_classes], dtype=torch.bool)
                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    selected_classes = cur_gt_classes[mask]
                else:
                    feature_map_size = anchors.shape[:3]
                    anchors = anchors.view(-1, anchors.shape[-1])
                    selected_classes = cur_gt_classes[mask]
                single_target = self.assign_targets_single(anchors, cur_gt[mask], gt_classes=selected_classes, matched_threshold=self.matched_thresholds[anchor_class_name], unmatched_threshold=self.unmatched_thresholds[anchor_class_name])
                target_list.append(single_target)
            if self.use_multihead:
                target_dict = {'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list], 'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list], 'reg_weights': [t['reg_weights'].view(-1) for t in target_list]}
                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list], 'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size) for t in target_list], 'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list]}
                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=-2).view(-1, self.box_coder.code_size)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)
            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])
        bbox_targets = torch.stack(bbox_targets, dim=0)
        cls_labels = torch.stack(cls_labels, dim=0)
        reg_weights = torch.stack(reg_weights, dim=0)
        all_targets_dict = {'box_cls_labels': cls_labels, 'box_reg_targets': bbox_targets, 'reg_weights': reg_weights}
        return all_targets_dict

    def assign_targets_single(self, anchors, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45):
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0)
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1
            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()
            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)
        fg_inds = (labels > 0).nonzero()[:, 0]
        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]
            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
        elif len(gt_boxes) == 0 or anchors.shape[0] == 0:
            labels[:] = 0
        else:
            labels[bg_inds] = 0
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)
        reg_weights = anchors.new_zeros((num_anchors,))
        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0
        ret_dict = {'box_cls_labels': labels, 'box_reg_targets': bbox_targets, 'reg_weights': reg_weights}
        return ret_dict


class AnchorHeadTemplate(nn.Module):

    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)
        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6), **anchor_target_cfg.get('BOX_CODER_CONFIG', {}))
        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range, anchor_ndim=self.box_coder.code_size)
        self.anchors = [x for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)
        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(anchor_range=point_cloud_range, anchor_generator_config=anchor_generator_cfg)
        feature_map_size = [(grid_size[:2] // config['feature_map_stride']) for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)
        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors
        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(topk=anchor_target_cfg.TOPK, box_coder=self.box_coder, use_multihead=self.use_multihead, match_height=anchor_target_cfg.MATCH_HEIGHT)
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(model_cfg=self.model_cfg, class_names=self.class_names, box_coder=self.box_coder, match_height=anchor_target_cfg.MATCH_HEIGHT)
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module('cls_loss_func', loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0))
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None else losses_cfg.REG_LOSS_TYPE
        self.add_module('reg_loss_func', getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights']))
        self.add_module('dir_loss_func', loss_utils.WeightedCrossEntropyLoss())

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(self.anchors, gt_boxes)
        return targets_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            box_cls_labels[positives] = 1
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)
        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(*list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device)
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)
        cls_loss = cls_loss_src.sum() / batch_size
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {'rpn_loss_cls': cls_loss.item()}
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype, device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])
        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1, box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else box_preds.shape[-1])
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {'rpn_loss_loc': loc_loss.item()}
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(anchors, box_reg_targets, dir_offset=self.model_cfg.DIR_OFFSET, num_bins=self.model_cfg.NUM_DIR_BINS)
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()
        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)
        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]
            period = 2 * np.pi / self.model_cfg.NUM_DIR_BINS
            dir_rot = common_utils.limit_period(batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period)
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels
        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(-(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2)
        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError


class SeparateHead(nn.Module):

    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict
        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']
            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias), nn.BatchNorm2d(input_channels), nn.ReLU()))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, 'bias') and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)
        return ret_dict


class CenterHead(nn.Module):

    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)
        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array([self.class_names.index(x) for x in cur_class_names if x in class_names]))
            self.class_id_mapping_each_head.append(cur_class_id_mapping)
        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'
        self.shared_conv = nn.Sequential(nn.Conv2d(input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1, bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)), nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL), nn.ReLU())
        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(SeparateHead(input_channels=self.model_cfg.SHARED_CONV_CHANNEL, sep_head_dict=cur_head_dict, init_bias=-2.19, use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)))
        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500, gaussian_overlap=0.1, min_radius=2):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()
        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()
        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride
        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)
        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue
            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue
            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())
            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1
            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]
        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        batch_size = gt_boxes.shape[0]
        ret_dict = {'heatmaps': [], 'target_boxes': [], 'inds': [], 'masks': [], 'heatmap_masks': []}
        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]
                gt_boxes_single_head = []
                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])
                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)
                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(), feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE, num_max_objs=target_assigner_cfg.NUM_MAX_OBJS, gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP, min_radius=target_assigner_cfg.MIN_RADIUS)
                heatmap_list.append(heatmap)
                target_boxes_list.append(ret_boxes)
                inds_list.append(inds)
                masks_list.append(mask)
            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=0.0001, max=1 - 0.0001)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        tb_dict = {}
        loss = 0
        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)
            reg_loss = self.reg_loss_func(pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes)
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()
        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).float()
        ret_dict = [{'pred_boxes': [], 'pred_scores': [], 'pred_labels': []} for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None
            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin, center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel, point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size, feature_map_stride=self.feature_map_stride, K=post_process_cfg.MAX_OBJ_PER_SAMPLE, circle_nms=post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms', score_thresh=post_process_cfg.SCORE_THRESH, post_center_limit_range=post_center_limit_range)
            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'], nms_config=post_process_cfg.NMS_CONFIG, score_thresh=None)
                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]
                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])
        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1
        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)
        pred_boxes = pred_dicts[0]['pred_boxes']
        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()
        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])
            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)
        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))
        if self.training:
            target_dict = self.assign_targets(data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:], feature_map_stride=data_dict.get('spatial_features_2d_strides', None))
            self.forward_ret_dict['target_dicts'] = target_dict
        self.forward_ret_dict['pred_dicts'] = pred_dicts
        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(data_dict['batch_size'], pred_dicts)
            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts
        return data_dict


class PointHeadTemplate(nn.Module):

    def __init__(self, model_cfg, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module('cls_loss_func', loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0))
        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None))
        else:
            self.reg_loss_func = F.smooth_l1_loss

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([nn.Linear(c_in, fc_cfg[k], bias=False), nn.BatchNorm1d(fc_cfg[k]), nn.ReLU()])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None, ret_box_labels=False, ret_part_labels=False, set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, 'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        for k in range(batch_size):
            bs_mask = bs_idx == k
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()).long().squeeze(dim=0)
            box_fg_flag = box_idxs_of_pts >= 0
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(points_single.unsqueeze(dim=0), extend_gt_boxes[k:k + 1, :, 0:7].contiguous()).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = (box_centers - points_single).norm(dim=1) < central_radius
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag], gt_classes=gt_box_of_fg_points[:, -1].long())
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single
            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = transformed_points / gt_box_of_fg_points[:, 3:6] + offset
                point_part_labels[bs_mask] = point_part_labels_single
        targets_dict = {'point_cls_labels': point_cls_labels, 'point_box_labels': point_box_labels, 'point_part_labels': point_part_labels}
        return targets_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)
        positives = point_cls_labels > 0
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_cls': point_loss_cls.item(), 'point_pos_num': pos_normalizer.item()})
        return point_loss_cls, tb_dict

    def get_part_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['point_part_labels']
        point_part_preds = self.forward_ret_dict['point_part_preds']
        point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_part': point_loss_part.item()})
        return point_loss_part, tb_dict

    def get_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['point_box_labels']
        point_box_preds = self.forward_ret_dict['point_box_preds']
        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        point_loss_box_src = self.reg_loss_func(point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...])
        point_loss_box = point_loss_box_src.sum()
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict

    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Args:
            points: (N, 3)
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)
        Returns:
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)

        """
        _, pred_classes = point_cls_preds.max(dim=-1)
        point_box_preds = self.box_coder.decode_torch(point_box_preds, points, pred_classes + 1)
        return point_cls_preds, point_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError


class PointIntraPartOffsetHead(PointHeadTemplate):
    """
    Point-based head for predicting the intra-object part locations.
    Reference Paper: https://arxiv.org/abs/1907.03670
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """

    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(fc_cfg=self.model_cfg.CLS_FC, input_channels=input_channels, output_channels=num_class)
        self.part_reg_layers = self.make_fc_layers(fc_cfg=self.model_cfg.PART_FC, input_channels=input_channels, output_channels=3)
        target_cfg = self.model_cfg.TARGET_CONFIG
        if target_cfg.get('BOX_CODER', None) is not None:
            self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(**target_cfg.BOX_CODER_CONFIG)
            self.box_layers = self.make_fc_layers(fc_cfg=self.model_cfg.REG_FC, input_channels=input_channels, output_channels=self.box_coder.code_size)
        else:
            self.box_layers = None

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes, set_ignore_flag=True, use_ball_constraint=False, ret_part_labels=True, ret_box_labels=self.box_layers is not None)
        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict = self.get_cls_layer_loss(tb_dict)
        point_loss_part, tb_dict = self.get_part_layer_loss(tb_dict)
        point_loss = point_loss_cls + point_loss_part
        if self.box_layers is not None:
            point_loss_box, tb_dict = self.get_box_layer_loss(tb_dict)
            point_loss += point_loss_box
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)
        point_part_preds = self.part_reg_layers(point_features)
        ret_dict = {'point_cls_preds': point_cls_preds, 'point_part_preds': point_part_preds}
        if self.box_layers is not None:
            point_box_preds = self.box_layers(point_features)
            ret_dict['point_box_preds'] = point_box_preds
        point_cls_scores = torch.sigmoid(point_cls_preds)
        point_part_offset = torch.sigmoid(point_part_preds)
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)
        batch_dict['point_part_offset'] = point_part_offset
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_part_labels'] = targets_dict.get('point_part_labels')
            ret_dict['point_box_labels'] = targets_dict.get('point_box_labels')
        if self.box_layers is not None and (not self.training or self.predict_boxes_when_training):
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(points=batch_dict['point_coords'][:, 1:4], point_cls_preds=point_cls_preds, point_box_preds=ret_dict['point_box_preds'])
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False
        self.forward_ret_dict = ret_dict
        return batch_dict


def find_all_spconv_keys(model: 'nn.Module', prefix='') ->Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: 'Set[str]' = set()
    for name, child in model.named_children():
        new_prefix = f'{prefix}.{name}' if prefix != '' else name
        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f'{new_prefix}.weight'
            found_keys.add(new_prefix)
        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))
    return found_keys


class Detector3DTemplate(nn.Module):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.module_topology = ['vfe', 'backbone_3d', 'map_to_bev_module', 'pfe', 'backbone_2d', 'dense_head', 'point_head', 'roi_head']

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {'module_list': [], 'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features, 'num_point_features': self.dataset.point_feature_encoder.num_point_features, 'grid_size': self.dataset.grid_size, 'point_cloud_range': self.dataset.point_cloud_range, 'voxel_size': self.dataset.voxel_size, 'depth_downsample_factor': self.dataset.depth_downsample_factor}
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(model_info_dict=model_info_dict)
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict
        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](model_cfg=self.model_cfg.VFE, num_point_features=model_info_dict['num_rawpoint_features'], point_cloud_range=model_info_dict['point_cloud_range'], voxel_size=model_info_dict['voxel_size'], grid_size=model_info_dict['grid_size'], depth_downsample_factor=model_info_dict['depth_downsample_factor'])
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict
        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](model_cfg=self.model_cfg.BACKBONE_3D, input_channels=model_info_dict['num_point_features'], grid_size=model_info_dict['grid_size'], voxel_size=model_info_dict['voxel_size'], point_cloud_range=model_info_dict['point_cloud_range'])
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict
        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](model_cfg=self.model_cfg.MAP_TO_BEV, grid_size=model_info_dict['grid_size'])
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict
        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](model_cfg=self.model_cfg.BACKBONE_2D, input_channels=model_info_dict['num_bev_features'])
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict
        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](model_cfg=self.model_cfg.PFE, voxel_size=model_info_dict['voxel_size'], point_cloud_range=model_info_dict['point_cloud_range'], num_bev_features=model_info_dict['num_bev_features'], num_rawpoint_features=model_info_dict['num_rawpoint_features'])
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](model_cfg=self.model_cfg.DENSE_HEAD, input_channels=model_info_dict['num_bev_features'], num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1, class_names=self.class_names, grid_size=model_info_dict['grid_size'], point_cloud_range=model_info_dict['point_cloud_range'], predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False), voxel_size=model_info_dict.get('voxel_size', False))
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict
        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']
        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](model_cfg=self.model_cfg.POINT_HEAD, input_channels=num_point_features, num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1, predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False))
        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](model_cfg=self.model_cfg.ROI_HEAD, input_channels=model_info_dict['num_point_features'], backbone_channels=model_info_dict['backbone_channels'], point_cloud_range=model_info_dict['point_cloud_range'], voxel_size=model_info_dict['voxel_size'], num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1)
        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = batch_dict['batch_index'] == index
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]
                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']
                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx:cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(cls_scores=cur_cls_preds, box_preds=cur_box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]
                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(box_scores=cls_preds, box_preds=box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]
                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
            recall_dict = self.generate_recall_record(box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds, recall_dict=recall_dict, batch_index=index, data_dict=batch_dict, thresh_list=post_process_cfg.RECALL_THRESH_LIST)
            record_dict = {'pred_boxes': final_boxes, 'pred_scores': final_scores, 'pred_labels': final_labels}
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict
        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]
        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % str(cur_thresh)] = 0
                recall_dict['rcnn_%s' % str(cur_thresh)] = 0
        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))
            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])
            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled
            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()
        spconv_keys = find_all_spconv_keys(self)
        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                val_native = val.transpose(-1, -2)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()
            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        version = checkpoint.get('version', None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)
        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)
        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))
        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)
        self._load_state_dict(checkpoint['model_state'], strict=True)
        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])
        if 'version' in checkpoint:
            None
        logger.info('==> Done')
        return it, epoch


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = box_scores >= score_thresh
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]
    selected = []
    if box_scores.shape[0] > 0:
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config)
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]
    if score_thresh is not None:
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected]
    return selected, src_box_scores[selected]


class SECONDNetIoU(Detector3DTemplate):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    @staticmethod
    def cal_scores_by_npoints(cls_scores, iou_scores, num_points_in_gt, cls_thresh=10, iou_thresh=100):
        """
        Args:
            cls_scores: (N)
            iou_scores: (N)
            num_points_in_gt: (N, 7+c)
            cls_thresh: scalar
            iou_thresh: scalar
        """
        assert iou_thresh >= cls_thresh
        alpha = torch.zeros(cls_scores.shape, dtype=torch.float32)
        alpha[num_points_in_gt <= cls_thresh] = 0
        alpha[num_points_in_gt >= iou_thresh] = 1
        mask = (num_points_in_gt > cls_thresh) & (num_points_in_gt < iou_thresh)
        alpha[mask] = (num_points_in_gt[mask] - 10) / (iou_thresh - cls_thresh)
        scores = (1 - alpha) * cls_scores + alpha * iou_scores
        return scores

    def set_nms_score_by_class(self, iou_preds, cls_preds, label_preds, score_by_class):
        n_classes = torch.unique(label_preds).shape[0]
        nms_scores = torch.zeros(iou_preds.shape, dtype=torch.float32)
        for i in range(n_classes):
            mask = label_preds == i + 1
            class_name = self.class_names[i]
            score_type = score_by_class[class_name]
            if score_type == 'iou':
                nms_scores[mask] = iou_preds[mask]
            elif score_type == 'cls':
                nms_scores[mask] = cls_preds[mask]
            else:
                raise NotImplementedError
        return nms_scores

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = batch_dict['batch_index'] == index
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            iou_preds = batch_dict['batch_cls_preds'][batch_mask]
            cls_preds = batch_dict['roi_scores'][batch_mask]
            src_iou_preds = iou_preds
            src_box_preds = box_preds
            src_cls_preds = cls_preds
            assert iou_preds.shape[1] in [1, self.num_class]
            if not batch_dict['cls_preds_normalized']:
                iou_preds = torch.sigmoid(iou_preds)
                cls_preds = torch.sigmoid(cls_preds)
            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                iou_preds, label_preds = torch.max(iou_preds, dim=-1)
                label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels', False) else label_preds + 1
                if post_process_cfg.NMS_CONFIG.get('SCORE_BY_CLASS', None) and post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'score_by_class':
                    nms_scores = self.set_nms_score_by_class(iou_preds, cls_preds, label_preds, post_process_cfg.NMS_CONFIG.SCORE_BY_CLASS)
                elif post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) == 'iou' or post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) is None:
                    nms_scores = iou_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'cls':
                    nms_scores = cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'weighted_iou_cls':
                    nms_scores = post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.iou * iou_preds + post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.cls * cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'num_pts_iou_cls':
                    point_mask = batch_dict['points'][:, 0] == batch_mask
                    batch_points = batch_dict['points'][point_mask][:, 1:4]
                    num_pts_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(batch_points.cpu(), box_preds[:, 0:7].cpu()).sum(dim=1).float()
                    score_thresh_cfg = post_process_cfg.NMS_CONFIG.SCORE_THRESH
                    nms_scores = self.cal_scores_by_npoints(cls_preds, iou_preds, num_pts_in_gt, score_thresh_cfg.cls, score_thresh_cfg.iou)
                else:
                    raise NotImplementedError
                selected, selected_scores = class_agnostic_nms(box_scores=nms_scores, box_preds=box_preds, nms_config=post_process_cfg.NMS_CONFIG, score_thresh=post_process_cfg.SCORE_THRESH)
                if post_process_cfg.OUTPUT_RAW_SCORE:
                    raise NotImplementedError
                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
            recall_dict = self.generate_recall_record(box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds, recall_dict=recall_dict, batch_index=index, data_dict=batch_dict, thresh_list=post_process_cfg.RECALL_THRESH_LIST)
            record_dict = {'pred_boxes': final_boxes, 'pred_scores': final_scores, 'pred_labels': final_labels, 'pred_cls_scores': cls_preds[selected], 'pred_iou_scores': iou_preds[selected]}
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict


class ProposalTargetLayer(nn.Module):

    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(batch_dict=batch_dict)
        reg_valid_mask = (batch_roi_ious > self.roi_sampler_cfg.REG_FG_THRESH).long()
        if self.roi_sampler_cfg.CLS_SCORE_TYPE == 'cls':
            batch_cls_labels = (batch_roi_ious > self.roi_sampler_cfg.CLS_FG_THRESH).long()
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg.CLS_BG_THRESH) & (batch_roi_ious < self.roi_sampler_cfg.CLS_FG_THRESH)
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg.CLS_SCORE_TYPE == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg.CLS_BG_THRESH
            iou_fg_thresh = self.roi_sampler_cfg.CLS_FG_THRESH
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)
            batch_cls_labels = (fg_mask > 0).float()
            batch_cls_labels[interval_mask] = (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        else:
            raise NotImplementedError
        targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois, 'gt_iou_of_rois': batch_roi_ious, 'roi_scores': batch_roi_scores, 'roi_labels': batch_roi_labels, 'reg_valid_mask': reg_valid_mask, 'rcnn_cls_labels': batch_cls_labels}
        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes']
        code_size = rois.shape[-1]
        batch_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size)
        batch_gt_of_rois = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE, code_size + 1)
        batch_roi_ious = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_scores = rois.new_zeros(batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE)
        batch_roi_labels = rois.new_zeros((batch_size, self.roi_sampler_cfg.ROI_PER_IMAGE), dtype=torch.long)
        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt
            if self.roi_sampler_cfg.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(rois=cur_roi, roi_labels=cur_roi_labels, gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long())
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
            sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)
            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]
        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps):
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg.FG_RATIO * self.roi_sampler_cfg.ROI_PER_IMAGE))
        fg_thresh = min(self.roi_sampler_cfg.REG_FG_THRESH, self.roi_sampler_cfg.CLS_FG_THRESH)
        fg_inds = (max_overlaps >= fg_thresh).nonzero().view(-1)
        easy_bg_inds = (max_overlaps < self.roi_sampler_cfg.CLS_BG_THRESH_LO).nonzero().view(-1)
        hard_bg_inds = ((max_overlaps < self.roi_sampler_cfg.REG_FG_THRESH) & (max_overlaps >= self.roi_sampler_cfg.CLS_BG_THRESH_LO)).nonzero().view(-1)
        fg_num_rois = fg_inds.numel()
        bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()
        if fg_num_rois > 0 and bg_num_rois > 0:
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO)
        elif fg_num_rois > 0 and bg_num_rois == 0:
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg.ROI_PER_IMAGE) * fg_num_rois)
            rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
            fg_inds = fg_inds[rand_num]
            bg_inds = []
        elif bg_num_rois > 0 and fg_num_rois == 0:
            bg_rois_per_this_image = self.roi_sampler_cfg.ROI_PER_IMAGE
            bg_inds = self.sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg.HARD_BG_RATIO)
        else:
            None
            None
            raise NotImplementedError
        sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
        return sampled_inds

    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            hard_bg_inds = hard_bg_inds[rand_idx]
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            easy_bg_inds = easy_bg_inds[rand_idx]
            bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_rois_num = bg_rois_per_this_image
            rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_rois_num = bg_rois_per_this_image
            rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError
        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = rois.new_zeros(rois.shape[0])
        gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])
        for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
            roi_mask = roi_labels == k
            gt_mask = gt_labels == k
            if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = gt_mask.nonzero().view(-1)
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt)
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                max_overlaps[roi_mask] = cur_max_overlaps
                gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]
        return max_overlaps, gt_assignment


class RoIHeadTemplate(nn.Module):

    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(**self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {}))
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module('reg_loss_func', loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights']))

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False), nn.BatchNorm1d(fc_list[k]), nn.ReLU()])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = batch_dict['batch_index'] == index
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]
            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)
            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config)
            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]
        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)
        rois = targets_dict['rois']
        gt_of_rois = targets_dict['gt_of_rois']
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry
        gt_of_rois = common_utils.rotate_points_along_z(points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)).view(batch_size, -1, gt_of_rois.shape[-1])
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)
        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]
        fg_mask = reg_valid_mask > 0
        fg_sum = fg_mask.long().sum().item()
        tb_dict = {}
        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor)
            rcnn_loss_reg = self.reg_loss_func(rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0), reg_targets.unsqueeze(dim=0))
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]
                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors).view(-1, code_size)
                rcnn_boxes3d = common_utils.rotate_points_along_z(rcnn_boxes3d.unsqueeze(dim=1), roi_ry).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz
                loss_corner = loss_utils.get_corner_loss_lidar(rcnn_boxes3d[:, 0:7], gt_of_rois_src[fg_mask][:, 0:7])
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']
                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError
        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError
        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)
        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)
        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)
        batch_box_preds = common_utils.rotate_points_along_z(batch_box_preds.unsqueeze(dim=1), roi_ry).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds


class SECONDHead(RoIHeadTemplate):

    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = self.model_cfg.ROI_GRID_POOL.IN_CHANNEL * GRID_SIZE * GRID_SIZE
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False), nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]), nn.ReLU()])
            pre_channel = self.model_cfg.SHARED_FC[k]
            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        self.iou_layers = self.make_fc_layers(input_channels=pre_channel, output_channels=1, fc_list=self.model_cfg.IOU_FC)
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                spatial_features_2d: (B, C, H, W)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois'].detach()
        spatial_features_2d = batch_dict['spatial_features_2d'].detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)
        dataset_cfg = batch_dict['dataset_cfg']
        min_x = dataset_cfg.POINT_CLOUD_RANGE[0]
        min_y = dataset_cfg.POINT_CLOUD_RANGE[1]
        voxel_size_x = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[0]
        voxel_size_y = dataset_cfg.DATA_PROCESSOR[-1].VOXEL_SIZE[1]
        down_sample_ratio = self.model_cfg.ROI_GRID_POOL.DOWNSAMPLE_RATIO
        pooled_features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            x1 = (rois[b_id, :, 0] - rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            x2 = (rois[b_id, :, 0] + rois[b_id, :, 3] / 2 - min_x) / (voxel_size_x * down_sample_ratio)
            y1 = (rois[b_id, :, 1] - rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            y2 = (rois[b_id, :, 1] + rois[b_id, :, 4] / 2 - min_y) / (voxel_size_y * down_sample_ratio)
            angle, _ = common_utils.check_numpy_to_torch(rois[b_id, :, 6])
            cosa = torch.cos(angle)
            sina = torch.sin(angle)
            theta = torch.stack(((x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * -sina, (x1 + x2 - width + 1) / (width - 1), (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)), dim=1).view(-1, 2, 3).float()
            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            grid = nn.functional.affine_grid(theta, torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size)))
            pooled_features = nn.functional.grid_sample(spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width), grid)
            pooled_features_list.append(pooled_features)
        torch.backends.cudnn.enabled = True
        pooled_features = torch.cat(pooled_features_list, dim=0)
        return pooled_features

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
        pooled_features = self.roi_grid_pool(batch_dict)
        batch_size_rcnn = pooled_features.shape[0]
        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_iou = self.iou_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)
        if not self.training:
            batch_dict['batch_cls_preds'] = rcnn_iou.view(batch_dict['batch_size'], -1, rcnn_iou.shape[-1])
            batch_dict['batch_box_preds'] = batch_dict['rois']
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_iou'] = rcnn_iou
            self.forward_ret_dict = targets_dict
        return batch_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_iou_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def get_box_iou_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_iou = forward_ret_dict['rcnn_iou']
        rcnn_iou_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        rcnn_iou_flat = rcnn_iou.view(-1)
        if loss_cfgs.IOU_LOSS == 'BinaryCrossEntropy':
            batch_loss_iou = nn.functional.binary_cross_entropy_with_logits(rcnn_iou_flat, rcnn_iou_labels.float(), reduction='none')
        elif loss_cfgs.IOU_LOSS == 'L2':
            batch_loss_iou = nn.functional.mse_loss(rcnn_iou_flat, rcnn_iou_labels, reduction='none')
        elif loss_cfgs.IOU_LOSS == 'smoothL1':
            diff = rcnn_iou_flat - rcnn_iou_labels
            batch_loss_iou = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(diff, 1.0 / 9.0)
        elif loss_cfgs.IOU_LOSS == 'focalbce':
            batch_loss_iou = loss_utils.sigmoid_focal_cls_loss(rcnn_iou_flat, rcnn_iou_labels)
        else:
            raise NotImplementedError
        iou_valid_mask = (rcnn_iou_labels >= 0).float()
        rcnn_loss_iou = (batch_loss_iou * iou_valid_mask).sum() / torch.clamp(iou_valid_mask.sum(), min=1.0)
        rcnn_loss_iou = rcnn_loss_iou * loss_cfgs.LOSS_WEIGHTS['rcnn_iou_weight']
        tb_dict = {'rcnn_loss_iou': rcnn_loss_iou.item()}
        return rcnn_loss_iou, tb_dict


class VoxelRCNNHead(RoIHeadTemplate):

    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(query_ranges=LAYER_cfg[src_name].QUERY_RANGES, nsamples=LAYER_cfg[src_name].NSAMPLE, radii=LAYER_cfg[src_name].POOL_RADIUS, mlps=mlps, pool_method=LAYER_cfg[src_name].POOL_METHOD)
            self.roi_grid_pool_layers.append(pool_layer)
            c_out += sum([x[-1] for x in mlps])
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False), nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]), nn.ReLU(inplace=True)])
            pre_channel = self.model_cfg.SHARED_FC[k]
            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False), nn.BatchNorm1d(self.model_cfg.CLS_FC[k]), nn.ReLU()])
            pre_channel = self.model_cfg.CLS_FC[k]
            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)
        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False), nn.BatchNorm1d(self.model_cfg.REG_FC[k]), nn.ReLU()])
            pre_channel = self.model_cfg.REG_FC[k]
            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)
        self.init_weights()

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(rois, grid_size=self.pool_cfg.GRID_SIZE)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)
        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])
        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]
            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(cur_coords[:, 1:4], downsample_times=cur_stride, voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range)
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            cur_roi_grid_coords = roi_grid_coords // cur_stride
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            pooled_features = pool_layer(xyz=cur_voxel_xyz.contiguous(), xyz_batch_cnt=cur_voxel_xyz_batch_cnt, new_xyz=roi_grid_xyz.contiguous().view(-1, 3), new_xyz_batch_cnt=roi_grid_batch_cnt, new_coords=cur_roi_grid_coords.contiguous().view(-1, 4), features=cur_sp_tensors.features.contiguous(), voxel2point_indices=v2p_ind_tensor)
            pooled_features = pooled_features.view(-1, self.pool_cfg.GRID_SIZE ** 3, pooled_features.shape[-1])
            pooled_features_list.append(pooled_features)
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]
        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)
        global_roi_grid_points = common_utils.rotate_points_along_z(local_roi_grid_points.clone(), rois[:, 6]).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()
        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) - local_roi_size.unsqueeze(dim=1) / 2
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        targets_dict = self.proposal_layer(batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST'])
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
        pooled_features = self.roi_grid_pool(batch_dict)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))
        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg)
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            self.forward_ret_dict = targets_dict
        return batch_dict


def calc_square_dist(a, b, return_cos=False):
    """
    Calculating square distance between a and b
    a: [bs, c, n]
    b: [bs, c, m]
    """
    a = a.transpose(1, 2)
    b = b.transpose(1, 2)
    n = a.shape[1]
    m = b.shape[1]
    num_channel = a.shape[-1]
    a_square = a.unsqueeze(dim=2)
    b_square = b.unsqueeze(dim=1)
    a_square = torch.sum(a_square * a_square, dim=-1)
    b_square = torch.sum(b_square * b_square, dim=-1)
    a_square = a_square.repeat((1, 1, m))
    b_square = b_square.repeat((1, n, 1))
    coor = torch.matmul(a, b.transpose(1, 2))
    if not return_cos:
        dist = a_square + b_square - 2.0 * coor
    else:
        dist = coor / torch.sqrt(a_square) / torch.sqrt(b_square)
    return dist


class _PointnetSAModuleBase(nn.Module):

    def __init__(self, sample_method=None):
        super(_PointnetSAModuleBase, self).__init__()
        self.groupers = None
        self.mlps = None
        self.sample_method = sample_method

    def forward(self, xyz, features, npoint, target_feature=None, keep_first_half=False):
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
            (B,  \\sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """
        self.npoint = npoint
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if target_feature is not None:
            assert features is not None
            with torch.no_grad():
                assert target_feature.shape[1] == features.shape[1]
                _, _, m = target_feature.shape
                sample_ratio = 2
                npoint = self.npoint - self.npoint // sample_ratio
                features_for_fps_distance = calc_square_dist(features[..., npoint:], target_feature)
                min_val, target_idx = torch.min(features_for_fps_distance, dim=2)
                min_val, min_top_k = torch.topk(min_val, self.npoint // sample_ratio, largest=False, dim=1)
                target_idx = npoint + min_top_k.int()
        else:
            npoint = self.npoint
        if self.sample_method is None:
            if keep_first_half:
                idx0 = torch.arange(npoint // 2).repeat(xyz.size(0), 1).int()
                idx1 = torch.arange(npoint - npoint // 2).repeat(xyz.size(0), 1).int() + xyz.shape[1] // 2
                idx = torch.cat([idx0, idx1], dim=1)
            else:
                idx = torch.arange(npoint).repeat(xyz.size(0), 1).int()
        elif self.sample_method == 'fps':
            idx = pointnet2_utils.furthest_point_sample(xyz, npoint)
        else:
            raise ValueError()
        if target_feature is not None:
            idx = torch.cat([idx, target_idx], dim=-1)
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, idx).transpose(1, 2).contiguous()
        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1), idx


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

    def __init__(self, radii, nsamples, mlps, bn=True, use_xyz=True, vote=False, sample_method=None):
        super(PointnetSAModuleMSG, self).__init__(sample_method=sample_method)
        assert len(radii) == len(nsamples) == len(mlps)
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if vote is False:
                self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            else:
                self.groupers.append(pointnet2_utils.QueryAndGroup_score(radius, nsample, use_xyz=use_xyz))
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

    def __init__(self, mlp, radius=None, nsample=None, bn=True, use_xyz=True, sample_method=None):
        super(PointnetSAModule, self).__init__(mlps=[mlp], radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, sample_method=sample_method)


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
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

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
        idxs = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(idxs)
        return idxs

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
        ctx.mark_non_differentiable(idx)
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

    def __init__(self, radius, nsample, use_xyz=True):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

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
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if features is not None:
            grouped_features = grouping_operation(features, idx)
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

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
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
        return new_features


class StackSAModuleMSG(nn.Module):

    def __init__(self, *, radii: List[float], nsamples: List[int], mlps: List[List[int]], use_xyz: bool=True, pool_method='max_pool'):
        """
        Args:
            radii: list of float, list of radii to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()
        assert len(radii) == len(nsamples) == len(mlps)
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz))
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[k + 1]), nn.ReLU()])
            self.mlps.append(nn.Sequential(*shared_mlps))
        self.pool_method = pool_method
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features=None, empty_voxel_set_zeros=True):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []
        for k in range(len(self.groupers)):
            new_features, ball_idxs = self.groupers[k](xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)
            new_features = new_features.permute(1, 0, 2).unsqueeze(dim=0)
            new_features = self.mlps[k](new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            else:
                raise NotImplementedError
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)
        new_features = torch.cat(new_features_list, dim=1)
        return new_xyz, new_features


class StackPointnetFPModule(nn.Module):

    def __init__(self, *, mlp: List[int]):
        """
        Args:
            mlp: list of int
        """
        super().__init__()
        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp[k + 1]), nn.ReLU()])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, unknown, unknown_batch_cnt, known, known_batch_cnt, unknown_feats=None, known_feats=None):
        """
        Args:
            unknown: (N1 + N2 ..., 3)
            known: (M1 + M2 ..., 3)
            unknow_feats: (N1 + N2 ..., C1)
            known_feats: (M1 + M2 ..., C2)

        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        dist, idx = pointnet2_utils.three_nn(unknown, unknown_batch_cnt, known, known_batch_cnt)
        dist_recip = 1.0 / (dist + 1e-08)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)
        else:
            new_features = interpolated_feats
        new_features = new_features.permute(1, 0)[None, :, :, None]
        new_features = self.mlp(new_features)
        new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)
        return new_features


class VectorPoolLocalInterpolateModule(nn.Module):

    def __init__(self, mlp, num_voxels, max_neighbour_distance, nsample, neighbor_type, use_xyz=True, neighbour_distance_multiplier=1.0, xyz_encoding_type='concat'):
        """
        Args:
            mlp:
            num_voxels:
            max_neighbour_distance:
            neighbor_type: 1: ball, others: cube
            nsample: find all (-1), find limited number(>0)
            use_xyz:
            neighbour_distance_multiplier:
            xyz_encoding_type:
        """
        super().__init__()
        self.num_voxels = num_voxels
        self.num_total_grids = self.num_voxels[0] * self.num_voxels[1] * self.num_voxels[2]
        self.max_neighbour_distance = max_neighbour_distance
        self.neighbor_distance_multiplier = neighbour_distance_multiplier
        self.nsample = nsample
        self.neighbor_type = neighbor_type
        self.use_xyz = use_xyz
        self.xyz_encoding_type = xyz_encoding_type
        if mlp is not None:
            if self.use_xyz:
                mlp[0] += 9 if self.xyz_encoding_type == 'concat' else 0
            shared_mlps = []
            for k in range(len(mlp) - 1):
                shared_mlps.extend([nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp[k + 1]), nn.ReLU()])
            self.mlp = nn.Sequential(*shared_mlps)
        else:
            self.mlp = None
        self.num_avg_length_of_neighbor_idxs = 1000

    def forward(self, support_xyz, support_features, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt):
        """
        Args:
            support_xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            support_features: (N1 + N2 ..., C) point-wise features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_grid_centers: (M1 + M2 ..., num_total_grids, 3) grids centers of each grid
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        with torch.no_grad():
            dist, idx, num_avg_length_of_neighbor_idxs = pointnet2_utils.three_nn_for_vector_pool_by_two_step(support_xyz, xyz_batch_cnt, new_xyz, new_xyz_grid_centers, new_xyz_batch_cnt, self.max_neighbour_distance, self.nsample, self.neighbor_type, self.num_avg_length_of_neighbor_idxs, self.num_total_grids, self.neighbor_distance_multiplier)
        self.num_avg_length_of_neighbor_idxs = max(self.num_avg_length_of_neighbor_idxs, num_avg_length_of_neighbor_idxs.item())
        dist_recip = 1.0 / (dist + 1e-08)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / torch.clamp_min(norm, min=1e-08)
        empty_mask = idx.view(-1, 3)[:, 0] == -1
        idx.view(-1, 3)[empty_mask] = 0
        interpolated_feats = pointnet2_utils.three_interpolate(support_features, idx.view(-1, 3), weight.view(-1, 3))
        interpolated_feats = interpolated_feats.view(idx.shape[0], idx.shape[1], -1)
        if self.use_xyz:
            near_known_xyz = support_xyz[idx.view(-1, 3).long()].view(-1, 3, 3)
            local_xyz = (new_xyz_grid_centers.view(-1, 1, 3) - near_known_xyz).view(-1, idx.shape[1], 9)
            if self.xyz_encoding_type == 'concat':
                interpolated_feats = torch.cat((interpolated_feats, local_xyz), dim=-1)
            else:
                raise NotImplementedError
        new_features = interpolated_feats.view(-1, interpolated_feats.shape[-1])
        new_features[empty_mask, :] = 0
        if self.mlp is not None:
            new_features = new_features.permute(1, 0)[None, :, :, None]
            new_features = self.mlp(new_features)
            new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)
        return new_features


class VectorPoolAggregationModule(nn.Module):

    def __init__(self, input_channels, num_local_voxel=(3, 3, 3), local_aggregation_type='local_interpolation', num_reduced_channels=30, num_channels_of_local_aggregation=32, post_mlps=(128,), max_neighbor_distance=None, neighbor_nsample=-1, neighbor_type=0, neighbor_distance_multiplier=2.0):
        super().__init__()
        self.num_local_voxel = num_local_voxel
        self.total_voxels = self.num_local_voxel[0] * self.num_local_voxel[1] * self.num_local_voxel[2]
        self.local_aggregation_type = local_aggregation_type
        assert self.local_aggregation_type in ['local_interpolation', 'voxel_avg_pool', 'voxel_random_choice']
        self.input_channels = input_channels
        self.num_reduced_channels = input_channels if num_reduced_channels is None else num_reduced_channels
        self.num_channels_of_local_aggregation = num_channels_of_local_aggregation
        self.max_neighbour_distance = max_neighbor_distance
        self.neighbor_nsample = neighbor_nsample
        self.neighbor_type = neighbor_type
        if self.local_aggregation_type == 'local_interpolation':
            self.local_interpolate_module = VectorPoolLocalInterpolateModule(mlp=None, num_voxels=self.num_local_voxel, max_neighbour_distance=self.max_neighbour_distance, nsample=self.neighbor_nsample, neighbor_type=self.neighbor_type, neighbour_distance_multiplier=neighbor_distance_multiplier)
            num_c_in = (self.num_reduced_channels + 9) * self.total_voxels
        else:
            self.local_interpolate_module = None
            num_c_in = (self.num_reduced_channels + 3) * self.total_voxels
        num_c_out = self.total_voxels * self.num_channels_of_local_aggregation
        self.separate_local_aggregation_layer = nn.Sequential(nn.Conv1d(num_c_in, num_c_out, kernel_size=1, groups=self.total_voxels, bias=False), nn.BatchNorm1d(num_c_out), nn.ReLU())
        post_mlp_list = []
        c_in = num_c_out
        for cur_num_c in post_mlps:
            post_mlp_list.extend([nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False), nn.BatchNorm1d(cur_num_c), nn.ReLU()])
            c_in = cur_num_c
        self.post_mlps = nn.Sequential(*post_mlp_list)
        self.num_mean_points_per_grid = 20
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def extra_repr(self) ->str:
        ret = f'radius={self.max_neighbour_distance}, local_voxels=({self.num_local_voxel}, local_aggregation_type={self.local_aggregation_type}, num_c_reduction={self.input_channels}->{self.num_reduced_channels}, num_c_local_aggregation={self.num_channels_of_local_aggregation}'
        return ret

    def vector_pool_with_voxel_query(self, xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt):
        use_xyz = 1
        pooling_type = 0 if self.local_aggregation_type == 'voxel_avg_pool' else 1
        new_features, new_local_xyz, num_mean_points_per_grid, point_cnt_of_grid = pointnet2_utils.vector_pool_with_voxel_query_op(xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt, self.num_local_voxel[0], self.num_local_voxel[1], self.num_local_voxel[2], self.max_neighbour_distance, self.num_reduced_channels, use_xyz, self.num_mean_points_per_grid, self.neighbor_nsample, self.neighbor_type, pooling_type)
        self.num_mean_points_per_grid = max(self.num_mean_points_per_grid, num_mean_points_per_grid.item())
        num_new_pts = new_features.shape[0]
        new_local_xyz = new_local_xyz.view(num_new_pts, -1, 3)
        new_features = new_features.view(num_new_pts, -1, self.num_reduced_channels)
        new_features = torch.cat((new_local_xyz, new_features), dim=-1).view(num_new_pts, -1)
        return new_features, point_cnt_of_grid

    @staticmethod
    def get_dense_voxels_by_center(point_centers, max_neighbour_distance, num_voxels):
        """
        Args:
            point_centers: (N, 3)
            max_neighbour_distance: float
            num_voxels: [num_x, num_y, num_z]

        Returns:
            voxel_centers: (N, total_voxels, 3)
        """
        R = max_neighbour_distance
        device = point_centers.device
        x_grids = torch.arange(-R + R / num_voxels[0], R - R / num_voxels[0] + 1e-05, 2 * R / num_voxels[0], device=device)
        y_grids = torch.arange(-R + R / num_voxels[1], R - R / num_voxels[1] + 1e-05, 2 * R / num_voxels[1], device=device)
        z_grids = torch.arange(-R + R / num_voxels[2], R - R / num_voxels[2] + 1e-05, 2 * R / num_voxels[2], device=device)
        x_offset, y_offset, z_offset = torch.meshgrid(x_grids, y_grids, z_grids)
        xyz_offset = torch.cat((x_offset.contiguous().view(-1, 1), y_offset.contiguous().view(-1, 1), z_offset.contiguous().view(-1, 1)), dim=-1)
        voxel_centers = point_centers[:, None, :] + xyz_offset[None, :, :]
        return voxel_centers

    def vector_pool_with_local_interpolate(self, xyz, xyz_batch_cnt, features, new_xyz, new_xyz_batch_cnt):
        """
        Args:
            xyz: (N, 3)
            xyz_batch_cnt: (batch_size)
            features: (N, C)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size)
        Returns:
            new_features: (M, total_voxels * C)
        """
        voxel_centers = self.get_dense_voxels_by_center(point_centers=new_xyz, max_neighbour_distance=self.max_neighbour_distance, num_voxels=self.num_local_voxel)
        voxel_features = self.local_interpolate_module.forward(support_xyz=xyz, support_features=features, xyz_batch_cnt=xyz_batch_cnt, new_xyz=new_xyz, new_xyz_grid_centers=voxel_centers, new_xyz_batch_cnt=new_xyz_batch_cnt)
        voxel_features = voxel_features.contiguous().view(-1, self.total_voxels * voxel_features.shape[-1])
        return voxel_features

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features, **kwargs):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        N, C = features.shape
        assert C % self.num_reduced_channels == 0, f'the input channels ({C}) should be an integral multiple of num_reduced_channels({self.num_reduced_channels})'
        features = features.view(N, -1, self.num_reduced_channels).sum(dim=1)
        if self.local_aggregation_type in ['voxel_avg_pool', 'voxel_random_choice']:
            vector_features, point_cnt_of_grid = self.vector_pool_with_voxel_query(xyz=xyz, xyz_batch_cnt=xyz_batch_cnt, features=features, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt)
        elif self.local_aggregation_type == 'local_interpolation':
            vector_features = self.vector_pool_with_local_interpolate(xyz=xyz, xyz_batch_cnt=xyz_batch_cnt, features=features, new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt)
        else:
            raise NotImplementedError
        vector_features = vector_features.permute(1, 0)[None, :, :]
        new_features = self.separate_local_aggregation_layer(vector_features)
        new_features = self.post_mlps(new_features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)
        return new_xyz, new_features


class VectorPoolAggregationModuleMSG(nn.Module):

    def __init__(self, input_channels, config):
        super().__init__()
        self.model_cfg = config
        self.num_groups = self.model_cfg.NUM_GROUPS
        self.layers = []
        c_in = 0
        for k in range(self.num_groups):
            cur_config = self.model_cfg[f'GROUP_CFG_{k}']
            cur_vector_pool_module = VectorPoolAggregationModule(input_channels=input_channels, num_local_voxel=cur_config.NUM_LOCAL_VOXEL, post_mlps=cur_config.POST_MLPS, max_neighbor_distance=cur_config.MAX_NEIGHBOR_DISTANCE, neighbor_nsample=cur_config.NEIGHBOR_NSAMPLE, local_aggregation_type=self.model_cfg.LOCAL_AGGREGATION_TYPE, num_reduced_channels=self.model_cfg.get('NUM_REDUCED_CHANNELS', None), num_channels_of_local_aggregation=self.model_cfg.NUM_CHANNELS_OF_LOCAL_AGGREGATION, neighbor_distance_multiplier=2.0)
            self.__setattr__(f'layer_{k}', cur_vector_pool_module)
            c_in += cur_config.POST_MLPS[-1]
        c_in += 3
        shared_mlps = []
        for cur_num_c in self.model_cfg.MSG_POST_MLPS:
            shared_mlps.extend([nn.Conv1d(c_in, cur_num_c, kernel_size=1, bias=False), nn.BatchNorm1d(cur_num_c), nn.ReLU()])
            c_in = cur_num_c
        self.msg_post_mlps = nn.Sequential(*shared_mlps)

    def forward(self, **kwargs):
        features_list = []
        for k in range(self.num_groups):
            cur_xyz, cur_features = self.__getattr__(f'layer_{k}')(**kwargs)
            features_list.append(cur_features)
        features = torch.cat(features_list, dim=-1)
        features = torch.cat((cur_xyz, features), dim=-1)
        features = features.permute(1, 0)[None, :, :]
        new_features = self.msg_post_mlps(features)
        new_features = new_features.squeeze(dim=0).permute(1, 0)
        return cur_xyz, new_features


class NeighborVoxelSAModuleMSG(nn.Module):

    def __init__(self, *, query_ranges: List[List[int]], radii: List[float], nsamples: List[int], mlps: List[List[int]], use_xyz: bool=True, pool_method='max_pool'):
        """
        Args:
            query_ranges: list of int, list of neighbor ranges to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()
        assert len(query_ranges) == len(nsamples) == len(mlps)
        self.groupers = nn.ModuleList()
        self.mlps_in = nn.ModuleList()
        self.mlps_pos = nn.ModuleList()
        self.mlps_out = nn.ModuleList()
        for i in range(len(query_ranges)):
            max_range = query_ranges[i]
            nsample = nsamples[i]
            radius = radii[i]
            self.groupers.append(voxel_query_utils.VoxelQueryAndGrouping(max_range, radius, nsample))
            mlp_spec = mlps[i]
            cur_mlp_in = nn.Sequential(nn.Conv1d(mlp_spec[0], mlp_spec[1], kernel_size=1, bias=False), nn.BatchNorm1d(mlp_spec[1]))
            cur_mlp_pos = nn.Sequential(nn.Conv2d(3, mlp_spec[1], kernel_size=1, bias=False), nn.BatchNorm2d(mlp_spec[1]))
            cur_mlp_out = nn.Sequential(nn.Conv1d(mlp_spec[1], mlp_spec[2], kernel_size=1, bias=False), nn.BatchNorm1d(mlp_spec[2]), nn.ReLU())
            self.mlps_in.append(cur_mlp_in)
            self.mlps_pos.append(cur_mlp_pos)
            self.mlps_out.append(cur_mlp_out)
        self.relu = nn.ReLU()
        self.pool_method = pool_method
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, new_coords, features, voxel2point_indices):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :param point_indices: (B, Z, Y, X) tensor of point indices
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \\sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_coords = new_coords[:, [0, 3, 2, 1]].contiguous()
        new_features_list = []
        for k in range(len(self.groupers)):
            features_in = features.permute(1, 0).unsqueeze(0)
            features_in = self.mlps_in[k](features_in)
            features_in = features_in.permute(0, 2, 1).contiguous()
            features_in = features_in.view(-1, features_in.shape[-1])
            grouped_features, grouped_xyz, empty_ball_mask = self.groupers[k](new_coords, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features_in, voxel2point_indices)
            grouped_features[empty_ball_mask] = 0
            grouped_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(-1)
            grouped_xyz[empty_ball_mask] = 0
            grouped_xyz = grouped_xyz.permute(1, 0, 2).unsqueeze(0)
            position_features = self.mlps_pos[k](grouped_xyz)
            new_features = grouped_features + position_features
            new_features = self.relu(new_features)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)]).squeeze(dim=-1)
            else:
                raise NotImplementedError
            new_features = self.mlps_out[k](new_features)
            new_features = new_features.squeeze(dim=0).permute(1, 0)
            new_features_list.append(new_features)
        new_features = torch.cat(new_features_list, dim=1)
        return new_features


class VoxelQuery(Function):

    @staticmethod
    def forward(ctx, max_range: 'int', radius: 'float', nsample: 'int', xyz: 'torch.Tensor', new_xyz: 'torch.Tensor', new_coords: 'torch.Tensor', point_indices: 'torch.Tensor'):
        """
        Args:
            ctx:
            max_range: int, max range of voxels to be grouped
            nsample: int, maximum number of features in the balls
            new_coords: (M1 + M2, 4), [batch_id, z, y, x] cooridnates of keypoints
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            point_indices: (batch_size, Z, Y, X) 4-D tensor recording the point indices of voxels
        Returns:
            idx: (M1 + M2, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()
        assert new_coords.is_contiguous()
        assert point_indices.is_contiguous()
        M = new_coords.shape[0]
        B, Z, Y, X = point_indices.shape
        idx = torch.IntTensor(M, nsample).zero_()
        z_range, y_range, x_range = max_range
        pointnet2.voxel_query_wrapper(M, Z, Y, X, nsample, radius, z_range, y_range, x_range, new_xyz, xyz, new_coords, point_indices, idx)
        empty_ball_mask = idx[:, 0] == -1
        idx[empty_ball_mask] = 0
        return idx, empty_ball_mask

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


voxel_query = VoxelQuery.apply


class VoxelQueryAndGrouping(nn.Module):

    def __init__(self, max_range: 'int', radius: 'float', nsample: 'int'):
        """
        Args:
            radius: float, radius of ball
            nsample: int, maximum number of features to gather in the ball
        """
        super().__init__()
        self.max_range, self.radius, self.nsample = max_range, radius, nsample

    def forward(self, new_coords: 'torch.Tensor', xyz: 'torch.Tensor', xyz_batch_cnt: 'torch.Tensor', new_xyz: 'torch.Tensor', new_xyz_batch_cnt: 'torch.Tensor', features: 'torch.Tensor', voxel2point_indices: 'torch.Tensor'):
        """
        Args:
            new_coords: (M1 + M2 ..., 3) centers voxel indices of the ball query
            xyz: (N1 + N2 ..., 3) xyz coordinates of the features
            xyz_batch_cnt: (batch_size), [N1, N2, ...]
            new_xyz: (M1 + M2 ..., 3) centers of the ball query
            new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
            features: (N1 + N2 ..., C) tensor of features to group
            voxel2point_indices: (B, Z, Y, X) tensor of points indices of voxels

        Returns:
            new_features: (M1 + M2, C, nsample) tensor
        """
        assert xyz.shape[0] == xyz_batch_cnt.sum(), 'xyz: %s, xyz_batch_cnt: %s' % (str(xyz.shape), str(new_xyz_batch_cnt))
        assert new_coords.shape[0] == new_xyz_batch_cnt.sum(), 'new_coords: %s, new_xyz_batch_cnt: %s' % (str(new_coords.shape), str(new_xyz_batch_cnt))
        batch_size = xyz_batch_cnt.shape[0]
        idx1, empty_ball_mask1 = voxel_query(self.max_range, self.radius, self.nsample, xyz, new_xyz, new_coords, voxel2point_indices)
        idx1 = idx1.view(batch_size, -1, self.nsample)
        count = 0
        for bs_idx in range(batch_size):
            idx1[bs_idx] -= count
            count += xyz_batch_cnt[bs_idx]
        idx1 = idx1.view(-1, self.nsample)
        idx1[empty_ball_mask1] = 0
        idx = idx1
        empty_ball_mask = empty_ball_mask1
        grouped_xyz = pointnet2_utils.grouping_operation(xyz, xyz_batch_cnt, idx, new_xyz_batch_cnt)
        grouped_features = pointnet2_utils.grouping_operation(features, xyz_batch_cnt, idx, new_xyz_batch_cnt)
        return grouped_features, grouped_xyz, empty_ball_mask


class RoIAwarePool3dFunction(Function):

    @staticmethod
    def forward(ctx, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """
        Args:
            ctx:
            rois: (N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
            pts: (npoints, 3)
            pts_feature: (npoints, C)
            out_size: int or tuple, like 7 or (7, 7, 7)
            max_pts_each_voxel:
            pool_method: 'max' or 'avg'

        Returns:
            pooled_features: (N, out_x, out_y, out_z, C)
        """
        assert rois.shape[1] == 7 and pts.shape[1] == 3
        if isinstance(out_size, int):
            out_x = out_y = out_z = out_size
        else:
            assert len(out_size) == 3
            for k in range(3):
                assert isinstance(out_size[k], int)
            out_x, out_y, out_z = out_size
        num_rois = rois.shape[0]
        num_channels = pts_feature.shape[-1]
        num_pts = pts.shape[0]
        pooled_features = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels))
        argmax = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, num_channels), dtype=torch.int)
        pts_idx_of_voxels = pts_feature.new_zeros((num_rois, out_x, out_y, out_z, max_pts_each_voxel), dtype=torch.int)
        pool_method_map = {'max': 0, 'avg': 1}
        pool_method = pool_method_map[pool_method]
        roiaware_pool3d_cuda.forward(rois, pts, pts_feature, argmax, pts_idx_of_voxels, pooled_features, pool_method)
        ctx.roiaware_pool3d_for_backward = pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels
        return pooled_features

    @staticmethod
    def backward(ctx, grad_out):
        """
        :param grad_out: (N, out_x, out_y, out_z, C)
        :return:
            grad_in: (npoints, C)
        """
        pts_idx_of_voxels, argmax, pool_method, num_pts, num_channels = ctx.roiaware_pool3d_for_backward
        grad_in = grad_out.new_zeros((num_pts, num_channels))
        roiaware_pool3d_cuda.backward(pts_idx_of_voxels, argmax, grad_out.contiguous(), grad_in, pool_method)
        return None, None, grad_in, None, None, None


class RoIAwarePool3d(nn.Module):

    def __init__(self, out_size, max_pts_each_voxel=128):
        super().__init__()
        self.out_size = out_size
        self.max_pts_each_voxel = max_pts_each_voxel

    def forward(self, rois, pts, pts_feature, pool_method='max'):
        assert pool_method in ['max', 'avg']
        return RoIAwarePool3dFunction.apply(rois, pts, pts_feature, self.out_size, self.max_pts_each_voxel, pool_method)


class RoIPointPool3dFunction(Function):

    @staticmethod
    def forward(ctx, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        """
        Args:
            ctx:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, num_boxes, 7), [x, y, z, dx, dy, dz, heading]
            pool_extra_width:
            num_sampled_points:

        Returns:
            pooled_features: (B, num_boxes, 512, 3 + C)
            pooled_empty_flag: (B, num_boxes)
        """
        assert points.shape.__len__() == 3 and points.shape[2] == 3
        batch_size, boxes_num, feature_len = points.shape[0], boxes3d.shape[1], point_features.shape[2]
        pooled_boxes3d = box_utils.enlarge_box3d(boxes3d.view(-1, 7), pool_extra_width).view(batch_size, -1, 7)
        pooled_features = point_features.new_zeros((batch_size, boxes_num, num_sampled_points, 3 + feature_len))
        pooled_empty_flag = point_features.new_zeros((batch_size, boxes_num)).int()
        roipoint_pool3d_cuda.forward(points.contiguous(), pooled_boxes3d.contiguous(), point_features.contiguous(), pooled_features, pooled_empty_flag)
        return pooled_features, pooled_empty_flag

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class RoIPointPool3d(nn.Module):

    def __init__(self, num_sampled_points=512, pool_extra_width=1.0):
        super().__init__()
        self.num_sampled_points = num_sampled_points
        self.pool_extra_width = pool_extra_width

    def forward(self, points, point_features, boxes3d):
        """
        Args:
            points: (B, N, 3)
            point_features: (B, N, C)
            boxes3d: (B, M, 7), [x, y, z, dx, dy, dz, heading]

        Returns:
            pooled_features: (B, M, 512, 3 + C)
            pooled_empty_flag: (B, M)
        """
        return RoIPointPool3dFunction.apply(points, point_features, boxes3d, self.pool_extra_width, self.num_sampled_points)


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: 'float'=2.0, alpha: 'float'=0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: 'torch.Tensor', target: 'torch.Tensor'):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor', weights: 'torch.Tensor'):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)
        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)
        loss = focal_weight * bce_loss
        if weights.shape.__len__() == 2 or weights.shape.__len__() == 1 and target.shape.__len__() == 2:
            weights = weights.unsqueeze(-1)
        assert weights.shape.__len__() == loss.shape.__len__()
        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """

    def __init__(self, beta: 'float'=1.0 / 9.0, code_weights: 'list'=None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights)

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-05:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor', weights: 'torch.Tensor'=None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)
        diff = input - target
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)
        loss = self.smooth_l1_loss(diff, self.beta)
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)
        return loss


class WeightedL1Loss(nn.Module):

    def __init__(self, code_weights: 'list'=None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights)

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor', weights: 'torch.Tensor'=None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)
        diff = input - target
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)
        loss = torch.abs(diff)
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor', weights: 'torch.Tensor'):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask
    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)
    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    loss = loss / torch.clamp_min(num, min=1.0)
    return loss


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


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

    def __init__(self, in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=None, batch_norm=None, bias=True, preact=False, name='', instance_norm=False, instance_norm_func=None):
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
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size, affine=False, track_running_stats=False)
            else:
                in_unit = instance_norm_func(in_size, affine=False, track_running_stats=False)
        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)
        self.add_module(name + 'conv', conv_unit)
        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)
            if activation is not None:
                self.add_module(name + 'activation', activation)
            if not bn and instance_norm:
                self.add_module(name + 'in', in_unit)


class Conv2d(_ConvBase):

    def __init__(self, in_size: 'int', out_size: 'int', *, kernel_size: Tuple[int, int]=(1, 1), stride: Tuple[int, int]=(1, 1), padding: Tuple[int, int]=(0, 0), activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv2d, batch_norm=BatchNorm2d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm2d)


class SharedMLP(nn.Sequential):

    def __init__(self, args: 'List[int]', *, bn: bool=False, activation=nn.ReLU(inplace=True), preact: bool=False, first: bool=False, name: str='', instance_norm: bool=False):
        super().__init__()
        for i in range(len(args) - 1):
            self.add_module(name + 'layer{}'.format(i), Conv2d(args[i], args[i + 1], bn=(not first or not preact or i != 0) and bn, activation=activation if not first or not preact or i != 0 else None, preact=preact, instance_norm=instance_norm))


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: 'int', *, name: str=''):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class Conv1d(_ConvBase):

    def __init__(self, in_size: 'int', out_size: 'int', *, kernel_size: int=1, stride: int=1, padding: int=0, activation=nn.ReLU(inplace=True), bn: bool=False, init=nn.init.kaiming_normal_, bias: bool=True, preact: bool=False, name: str='', instance_norm=False):
        super().__init__(in_size, out_size, kernel_size, stride, padding, activation, bn, init, conv=nn.Conv1d, batch_norm=BatchNorm1d, bias=bias, preact=preact, name=name, instance_norm=instance_norm, instance_norm_func=nn.InstanceNorm1d)


class FC(nn.Sequential):

    def __init__(self, in_size: 'int', out_size: 'int', *, activation=nn.ReLU(inplace=True), bn: bool=False, init=None, preact: bool=False, name: str=''):
        super().__init__()
        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant(fc.bias, 0)
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


class TransNonlinear(nn.Module):

    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class RelationUnit(nn.Module):

    def __init__(self, feature_dim=512, key_feature_dim=64):
        super(RelationUnit, self).__init__()
        self.temp = 1
        self.WK = nn.Linear(feature_dim, key_feature_dim, bias=False)
        self.WQ = nn.Linear(feature_dim, key_feature_dim, bias=False)
        self.WV = nn.Linear(feature_dim, feature_dim, bias=False)
        self.after_norm = nn.BatchNorm1d(feature_dim)
        self.trans_conv = nn.Linear(feature_dim, feature_dim, bias=False)
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2.0 / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2.0 / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2.0 / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query=None, key=None, value=None, mask=None):
        w_k = self.WK(key)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(1, 2, 0)
        w_q = self.WQ(query)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(1, 0, 2)
        dot_prod = torch.bmm(w_q, w_k)
        if mask is not None:
            dot_prod = dot_prod.masked_fill(mask == 0, -1000000000.0)
        affinity = F.softmax(dot_prod * self.temp, dim=-1)
        affinity = affinity / (1e-09 + affinity.sum(dim=1, keepdim=True))
        w_v = self.WV(value)
        w_v = w_v.permute(1, 0, 2)
        output = torch.bmm(affinity, w_v)
        output = output.permute(1, 0, 2)
        output = self.trans_conv(query - output)
        return F.relu(output)


class MultiheadAttention(nn.Module):

    def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64, extra_nonlinear=True):
        super(MultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        self.extra_nonlinear = nn.ModuleList()
        for N in range(self.Nh):
            self.head.append(RelationUnit(feature_dim, key_feature_dim))
            if extra_nonlinear:
                self.extra_nonlinear.append(TransNonlinear(feature_dim, key_feature_dim))
            else:
                self.extra_nonlinear = None

    def forward(self, query=None, key=None, value=None):
        """
        query : #pixel x batch x dim

        """
        isFirst = True
        for N in range(self.Nh):
            if isFirst:
                concat = self.head[N](query, key, value)
                if self.extra_nonlinear:
                    concat = self.extra_nonlinear[N](concat)
                isFirst = False
            else:
                tmp = self.head[N](query, key, value)
                if self.extra_nonlinear:
                    tmp = self.extra_nonlinear[N](tmp)
                concat = torch.cat((concat, tmp), -1)
        output = concat
        return output


class Pointnet_Backbone(nn.Module):

    def __init__(self, input_channels=3, use_xyz=True, sample_method=None, first_sample_method=None):
        super(Pointnet_Backbone, self).__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(PointnetSAModule(radius=0.3, nsample=32, mlp=[input_channels, 64, 64, 128], use_xyz=use_xyz, sample_method=first_sample_method))
        self.SA_modules.append(PointnetSAModule(radius=0.5, nsample=32, mlp=[128, 128, 128, 256], use_xyz=use_xyz, sample_method=sample_method))
        self.SA_modules.append(PointnetSAModule(radius=0.7, nsample=32, mlp=[256, 256, 256, 256], use_xyz=use_xyz, sample_method=sample_method))
        self.cov_final = nn.Conv1d(256, 256, kernel_size=1)
        self.sample_method = sample_method

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, numpoints, target_features=None, keep_first_half=False):
        """
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        l_xyz, l_features = [xyz], [features]
        idxs = []
        for i in range(len(self.SA_modules)):
            if i > 0 and target_features is not None:
                assert not keep_first_half
                li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i], target_feature=target_features[i])
            else:
                li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i], keep_first_half=keep_first_half)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            idxs.append(idx)
        output_dict = {'xyz': l_xyz[-1], 'feature': self.cov_final(l_features[-1]), 'idxs': idxs, 'xyzs': l_xyz, 'features': l_features}
        return output_dict


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel=3, num_pos_feats=256):
        super().__init__()
        self.position_embedding_head = nn.Sequential(nn.Conv1d(input_channel, num_pos_feats, kernel_size=1), nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True), nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class TransformerDecoderLayer(nn.Module):

    def __init__(self, multihead_attn, FFN, d_model, self_posembed=None):
        super().__init__()
        self.self_attn = multihead_attn
        self.cross_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)
        self.FFN = FFN
        self.norm1 = nn.InstanceNorm1d(d_model)
        self.norm2 = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def with_pos_embed(self, tensor, pos_embed: 'Optional[Tensor]'):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, tgt, memory, query_pos=None):
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        query = key = value = self.with_pos_embed(tgt, query_pos_embed)
        tgt2 = self.self_attn(query=query, key=key, value=value)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt = F.relu(tgt)
        mask = self.cross_attn(query=tgt, key=memory, value=memory)
        tgt2 = tgt + mask
        tgt2 = self.norm2(tgt2.permute(1, 2, 0)).permute(2, 0, 1)
        tgt2 = F.relu(tgt2)
        return tgt2


def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])


class TransformerDecoder(nn.Module):

    def __init__(self, multihead_attn, FFN, d_model=512, num_decoder_layers=6, activation='relu', self_posembed=None):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(multihead_attn, FFN, d_model, self_posembed=self_posembed)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)

    def forward(self, tgt, memory, query_pos=None):
        assert tgt.dim() == 3, 'Expect 3 dimensional inputs'
        tgt_shape = tgt.shape
        num_imgs, batch, dim = tgt.shape
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, query_pos=query_pos)
        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, multihead_attn, FFN, d_model, self_posembed=None):
        super().__init__()
        self.self_attn = multihead_attn
        self.FFN = FFN
        self.norm = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed
        self.dropout = nn.Dropout(0.1)

    def with_pos_embed(self, tensor, pos_embed: 'Optional[Tensor]'):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, query_pos=None):
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        query = key = value = self.with_pos_embed(src, query_pos_embed)
        src2 = self.self_attn(query=query, key=key, value=value)
        src = src + src2
        src = self.norm(src.permute(1, 2, 0)).permute(2, 0, 1)
        return F.relu(src)


class TransformerEncoder(nn.Module):

    def __init__(self, multihead_attn, FFN, d_model=512, num_encoder_layers=6, activation='relu', self_posembed=None):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(multihead_attn, FFN, d_model, self_posembed=self_posembed)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

    def forward(self, src, query_pos=None):
        num_imgs, batch, dim = src.shape
        output = src
        for layer in self.layers:
            output = layer(output, query_pos=query_pos)
        return output


class PointnetTransformerSiamese(nn.Module):

    def __init__(self, input_channels=3, use_xyz=True, objective=False, input_size=1024, sample_method=None):
        super(PointnetTransformerSiamese, self).__init__()
        self.input_size = input_size
        sample_method = None
        vote_sample_method = None
        first_sample_method = None
        self.sample_method = sample_method
        d_model = 256
        num_layers = 1
        self.with_pos_embed = True
        self.backbone_net = Pointnet_Backbone(input_channels, use_xyz, sample_method=sample_method, first_sample_method=first_sample_method)
        self.cosine = nn.CosineSimilarity(dim=1)
        self.mlp = pt_utils.SharedMLP([4 + 256, 256, 256, 256], bn=True)
        self.FC_layer_cla = pt_utils.Seq(256).conv1d(256, bn=True).conv1d(256, bn=True).conv1d(1, activation=None)
        self.fea_layer = pt_utils.Seq(256).conv1d(256, bn=True).conv1d(256, activation=None)
        self.vote_layer = pt_utils.Seq(3 + 256).conv1d(256, bn=True).conv1d(256, bn=True).conv1d(3 + 256, activation=None)
        self.group5 = pointnet2_utils.QueryAndGroup(1.0, 8, use_xyz=use_xyz)
        self.group3 = pointnet2_utils.QueryAndGroup(0.3, 8, use_xyz=use_xyz)
        self.group1 = pointnet2_utils.QueryAndGroup(0.1, 16, use_xyz=use_xyz)
        self.vote_aggregation = PointnetSAModule(radius=0.3, nsample=16, mlp=[1 + 256, 256, 256, 256], use_xyz=use_xyz, sample_method=vote_sample_method)
        self.num_proposal = input_size // 16
        self.FC_proposal = pt_utils.Seq(256 + 1 + 256 + 3 + 256 + 3).conv1d(256, bn=True).conv1d(256, bn=True).conv1d(256, bn=True).conv1d(256, bn=True).conv1d(3 + 1 + 1, activation=None)
        multihead_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)
        if self.with_pos_embed:
            encoder_pos_embed = PositionEmbeddingLearned(3, d_model)
            decoder_pos_embed = PositionEmbeddingLearned(3, d_model)
        else:
            encoder_pos_embed = None
            decoder_pos_embed = None
        self.encoder = TransformerEncoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_encoder_layers=num_layers, self_posembed=encoder_pos_embed)
        self.decoder = TransformerDecoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_decoder_layers=num_layers, self_posembed=decoder_pos_embed)

    def xcorr(self, x_label, x_object, template_xyz):
        B = x_object.size(0)
        f = x_object.size(1)
        n1 = x_object.size(2)
        n2 = x_label.size(2)
        final_out_cla = self.cosine(x_object.unsqueeze(-1).expand(B, f, n1, n2), x_label.unsqueeze(2).expand(B, f, n1, n2))
        fusion_feature = torch.cat((final_out_cla.unsqueeze(1), template_xyz.transpose(1, 2).contiguous().unsqueeze(-1).expand(B, 3, n1, n2)), dim=1)
        fusion_feature = torch.cat((fusion_feature, x_object.unsqueeze(-1).expand(B, f, n1, n2)), dim=1)
        fusion_feature = self.mlp(fusion_feature)
        fusion_feature = F.max_pool2d(fusion_feature, kernel_size=[fusion_feature.size(2), 1])
        fusion_feature = fusion_feature.squeeze(2)
        fusion_feature = self.fea_layer(fusion_feature)
        return fusion_feature

    def transform_fuse(self, search_feature, search_xyz, template_feature, template_xyz):
        """Use transformer to fuse feature.

        template_feature : BxCxN
        template_xyz : BxNx3
        """
        search_feature = search_feature.permute(2, 0, 1)
        template_feature = template_feature.permute(2, 0, 1)
        num_img_train = search_feature.shape[0]
        num_img_template = template_feature.shape[0]
        encoded_memory = self.encoder(template_feature, query_pos=template_xyz if self.with_pos_embed else None)
        encoded_feat = self.decoder(search_feature, memory=encoded_memory, query_pos=search_xyz)
        encoded_feat = encoded_feat.permute(1, 2, 0)
        encoded_feat = self.fea_layer(encoded_feat)
        return encoded_feat

    def forward(self, input_dict):
        """
            template: B*512*3 or B*512*6
            search: B*1024*3 or B*1024*6
        """
        template = input_dict['template']
        search = input_dict['search']
        output_dict = {}
        template_output_dict = self.backbone_net(template, [self.input_size // 4, self.input_size // 8, self.input_size // 16], keep_first_half=True)
        template_features = template_output_dict['features']
        search_output_dict = self.backbone_net(search, [self.input_size // 2, self.input_size // 4, self.input_size // 8], target_features=template_features, keep_first_half=False)
        if 'cls_label' in input_dict:
            idxs = search_output_dict['idxs']
            b = idxs[0].shape[0]
            idxs_list = defaultdict(list)
            for i in range(len(idxs)):
                for bi in range(b):
                    if i == 0:
                        idxs_list[i].append(idxs[0][bi])
                    else:
                        idxs_list[i].append(idxs_list[i - 1][bi][idxs[i][bi].long()])
            final_idx = torch.stack(idxs_list[len(idxs) - 1], dim=0)
            output_dict['cls_label'] = torch.gather(input_dict['cls_label'], 1, final_idx.long())
            output_dict['reg_label'] = torch.gather(input_dict['reg_label'], 1, final_idx.long().unsqueeze(-1).repeat(1, 1, 4))
        template_xyz = template_output_dict['xyz']
        template_feature = template_output_dict['feature']
        search_xyz = search_output_dict['xyz']
        search_feature = search_output_dict['feature']
        fusion_feature = self.transform_fuse(search_feature, search_xyz, template_feature, template_xyz)
        estimation_cla = self.FC_layer_cla(fusion_feature)
        score = estimation_cla.sigmoid()
        fusion_xyz_feature = torch.cat((search_xyz.transpose(1, 2).contiguous(), fusion_feature), dim=1)
        offset_feature = self.vote_layer(fusion_xyz_feature)
        offset = offset_feature[:, :3, :].transpose(1, 2).contiguous()
        fusion_feature = fusion_feature + offset_feature[:, 3:, :]
        proposals = search_output_dict['xyz'] - offset
        temp_pooling_feature1 = self.group5(template_output_dict['xyzs'][-1], proposals, template_output_dict['features'][-1])
        temp_pooling_feature1, _ = torch.max(temp_pooling_feature1, dim=-1, keepdim=False)
        pooling_feature = temp_pooling_feature1
        search_pooling_feature1 = self.group5(search_output_dict['xyzs'][-1], search_output_dict['xyz'], search_output_dict['features'][-1])
        search_pooling_feature1, _ = torch.max(search_pooling_feature1, dim=-1, keepdim=False)
        search_pooling_feature = search_pooling_feature1
        pooling_feature = torch.cat([pooling_feature, search_pooling_feature], dim=1)
        proposal_features = torch.cat([score, pooling_feature, fusion_feature], dim=1)
        proposal_offsets = self.FC_proposal(proposal_features)
        estimation_boxs = torch.cat((proposal_offsets[:, 0:3, :] + offset.detach().transpose(1, 2).contiguous(), proposal_offsets[:, 3:5, :]), dim=1)
        output_dict.update({'estimation_cla': estimation_cla.squeeze(1), 'estimation_reg': offset, 'estimation_box': estimation_boxs.transpose(1, 2).contiguous(), 'center_xyz': offset.detach()})
        return output_dict


class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """

    def __init__(self, size_average=True, eps=1e-05, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (self.scale * (input.shape[1] * input.shape[2] * input.shape[3] / (torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps)).sqrt())
        else:
            return input * (self.scale / (torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps).sqrt())


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=1, num_layers=1, dim_feedforward=2048, activation='relu'):
        super().__init__()
        multihead_attn = MultiheadAttention(feature_dim=d_model, n_head=1, key_feature_dim=128)
        self.encoder = TransformerEncoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_encoder_layers=num_layers)
        self.decoder = TransformerDecoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_decoder_layers=num_layers)

    def forward(self, feature):
        num_img_train = feature.shape[0]
        encoded_memory, _ = self.encoder(feature)
        for i in range(num_img_train):
            _, cur_encoded_feat = self.decoder(feature[i, ...].unsqueeze(0), memory=encoded_memory)
            if i == 0:
                encoded_feat = cur_encoded_feat
            else:
                encoded_feat = torch.cat((encoded_feat, cur_encoded_feat), 0)
        return encoded_feat, decoded_feat


class TransformerSiamese(nn.Module):

    def __init__(self, d_model=512, nhead=1, activation='relu'):
        super().__init__()
        multihead_attn = MultiheadAttention(feature_dim=d_model, n_head=nhead, key_feature_dim=128)
        self.encoder = TransformerEncoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_encoder_layers=num_layers)
        self.decoder = TransformerDecoder(multihead_attn=multihead_attn, FFN=None, d_model=d_model, num_decoder_layers=num_layers)

    def forward(self, search_feature, template_feature):
        num_img_train = search_feature.shape[0]
        num_img_template = template_feature.shape[0]
        encoded_memory, _ = self.encoder(search_feature)
        for i in range(num_img_template):
            _, cur_encoded_feat = self.decoder(template_feature[i, ...].unsqueeze(0), memory=encoded_memory)
            if i == 0:
                encoded_feat = cur_encoded_feat
            else:
                encoded_feat = torch.cat((encoded_feat, cur_encoded_feat), 0)
        return encoded_feat


class PointnetProposalModule(PointnetSAModuleMSG):
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

    def __init__(self, mlp, radius=None, nsample=None, bn=True, use_xyz=True, vote=True, sample_method=None):
        super(PointnetProposalModule, self).__init__(mlps=[mlp], radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz, vote=vote, sample_method=sample_method)

    def forward(self, xyz, features, npoint, score):
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
            (B,  \\sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """
        self.npoint = npoint
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if self.sample_method is None:
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, torch.arange(self.npoint).repeat(xyz.size(0), 1).int()).transpose(1, 2).contiguous()
        else:
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)).transpose(1, 2).contiguous()
        for i in range(len(self.groupers)):
            new_features, score_id = self.groupers[i](xyz, new_xyz, score, features)
            idx = torch.arange(new_features.size(0))
            new_features = new_features[idx, :, score_id, :]
            new_features = new_features.unsqueeze(2)
            new_xyz = new_xyz[idx, score_id, :]
            new_features = self.mlps[i](new_features)
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])
            new_features = new_features.squeeze(-1).squeeze(-1)
            new_features_list.append(new_features)
        return new_xyz, torch.cat(new_features_list, dim=1)


class RandomDropout(nn.Module):

    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class BallQuery_score(Function):

    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz, score):
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
        inds = _ext.ball_query_score(new_xyz, xyz, score, radius, nsample)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


ball_query_score = BallQuery_score.apply


class QueryAndGroup_score(nn.Module):
    """
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        super(QueryAndGroup_score, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, score, features=None):
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
        unique_score = ball_query_score(self.radius, self.nsample, xyz, new_xyz, score)
        score_id = unique_score.argmax(dim=1)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz
        return new_features, score_id


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock2D,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNorm1d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (BatchNorm2d,
     lambda: ([], {'in_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv1d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Conv2d,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FC,
     lambda: ([], {'in_size': 4, 'out_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLossCenterNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GroupAll,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (InstanceL2Norm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionEmbeddingLearned,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 3])], {}),
     True),
    (SigmoidFocalClassificationLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransNonlinear,
     lambda: ([], {'d_model': 4, 'dim_feedforward': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WeightedCrossEntropyLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
]

class Test_Jasonkks_PTTR(_paritybench_base):
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

