import sys
_module = sys.modules[__name__]
del sys
config = _module
evaluate = _module
function = _module
inference = _module
loss = _module
dataset = _module
_init_paths = _module
h36m = _module
heatmap_dataset = _module
heatmap_dataset_h36m = _module
joints_dataset = _module
mixed_dataset = _module
mixed_tc_dataset = _module
mpii = _module
multiview_h36m = _module
multiview_mpii = _module
totalcapture = _module
totalcapture_collate = _module
models = _module
multiview_pose_net = _module
orn = _module
pose_mobilenetv2 = _module
pose_resnet = _module
multiviews = _module
body = _module
cameras = _module
cameras_cuda = _module
cameras_cuda_col = _module
h36m_body = _module
pictorial = _module
totalcapture_body = _module
triangulate = _module
utils = _module
pose_utils = _module
transforms = _module
utils = _module
vis = _module
zipreader = _module
train = _module
valid = _module
estimate = _module
estimate_triangulate = _module

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


import logging


import numpy as np


import pandas as pd


import torch


import torch.nn as nn


from torch.utils.data import Dataset


import copy


import random


import torch.nn.functional as F


import functools


import math


import torch.optim as optim


import torchvision


import matplotlib.pyplot as plt


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torch.multiprocessing


import pandas


class JointsMSELoss(nn.Module):

    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
        return loss


BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def apply_bone_offset(coords_global, bone_vectors):
    """

    :param coords_global: (nview*batch, 3, ndepth, h*w, 2*nbones)
    :param bone_vectors: (nview*batch, 3, 1, 2*nbones)
    :return:
    """
    nview_batch, coords_dim, ndepth, hw, nbones2 = coords_global.shape
    coords_global = coords_global.view(nview_batch, coords_dim, ndepth * hw, nbones2)
    res = coords_global + bone_vectors
    return res.view(nview_batch, coords_dim, ndepth, hw, nbones2)


def batch_global_to_uv(coords_global, affine_t, cam_intrimat, cam_extri_R, cam_extri_T):
    nview_batch, coords_dim, ndepth, hw, nbones2 = coords_global.shape
    coords_global_flat = coords_global.view(nview_batch, coords_dim, -1)
    cam_extri_R = cam_extri_R.view(-1, 3, 3)
    cam_extri_T = cam_extri_T.view(-1, 3, 1)
    coords_cam = torch.bmm(cam_extri_R, coords_global_flat - cam_extri_T)
    coords_cam_norm = coords_cam / coords_cam[:, 2:3]
    cam_intrimat = cam_intrimat.view(nview_batch, 3, 3)
    affine_t = affine_t.view(nview_batch, 3, 3)
    synth_trans = torch.bmm(affine_t, cam_intrimat)
    coords_uv_hm = torch.bmm(synth_trans, coords_cam_norm)
    return coords_uv_hm.view(nview_batch, 3, ndepth, hw, nbones2)


def batch_uv_to_global_with_multi_depth(uv1, inv_affine_t, inv_cam_intrimat, inv_cam_extri_R, inv_cam_extri_T, depths, nbones):
    """

    :param uv1:
    :param inv_affine_t: hm -> uv
    :param inv_cam_intrimat: uv -> norm image frame
    :param inv_cam_extri_R: transpose of cam_extri_R
    :param inv_cam_extri_T: same as cam_extri_T
    :param depths:
    :param nbones:
    :return:
    """
    dev = uv1.device
    nview_batch = inv_affine_t.shape[0]
    h = int(torch.max(uv1[1]).item()) + 1
    w = int(torch.max(uv1[0]).item()) + 1
    depths = torch.as_tensor(depths, device=dev).view(-1, 1, 1)
    ndepth = depths.shape[0]
    coords_hm_frame = uv1.view(1, 3, h * w, 1).expand(nview_batch, -1, -1, nbones * 2).contiguous().view(nview_batch, 3, -1)
    inv_cam_intrimat = inv_cam_intrimat.view(nview_batch, 3, 3)
    inv_affine_t = inv_affine_t.view(nview_batch, 3, 3)
    synth_trans = torch.bmm(inv_cam_intrimat, inv_affine_t)
    coords_img_frame = torch.bmm(synth_trans, coords_hm_frame)
    coords_img_frame = coords_img_frame.permute(1, 0, 2).contiguous().view(1, 3, -1)
    coords_img_frame_all_depth = coords_img_frame.expand(ndepth, -1, -1)
    coords_img_frame_all_depth = torch.mul(coords_img_frame_all_depth, depths)
    coords_img_frame_all_depth = coords_img_frame_all_depth.view(ndepth, 3, nview_batch, -1).permute(2, 1, 0, 3).contiguous().view(nview_batch, 3, -1)
    inv_cam_extri_R = inv_cam_extri_R.view(-1, 3, 3)
    inv_cam_extri_T = inv_cam_extri_T.view(-1, 3, 1)
    coords_global = torch.bmm(inv_cam_extri_R, coords_img_frame_all_depth) + inv_cam_extri_T
    coords_global = coords_global.view(nview_batch, 3, ndepth, h * w, 2 * nbones)
    return coords_global


def gen_hm_grid_coords(h, w, dev=None):
    """

    :param h:
    :param w:
    :param dev:
    :return: (3, h*w) each col is (u, v, 1)^T
    """
    if not dev:
        dev = torch.device('cpu')
    h = int(h)
    w = int(w)
    h_s = torch.linspace(0, h - 1, h)
    w_s = torch.linspace(0, w - 1, w)
    hm_cords = torch.meshgrid(h_s, w_s)
    flat_cords = torch.stack(hm_cords, dim=0).view(2, -1)
    out_grid = torch.ones(3, h * w, device=dev)
    out_grid[0] = flat_cords[1]
    out_grid[1] = flat_cords[0]
    return out_grid


def get_bone_vector_meta(selected_bones, body, dataset_joint_mapping):
    reverse_joint_mapping = dict()
    for k in dataset_joint_mapping.keys():
        v = dataset_joint_mapping[k]
        if v != '*':
            reverse_joint_mapping[v] = k
    bone_vec_meta_out_ref = []
    bone_vec_meta_out_cur = []
    bone_joint_map = body.get_imu_edges_reverse()
    for bone in selected_bones:
        par, child = bone_joint_map[bone]
        bone_vec_meta_out_ref.append(reverse_joint_mapping[par])
        bone_vec_meta_out_ref.append(reverse_joint_mapping[child])
        bone_vec_meta_out_cur.append(reverse_joint_mapping[child])
        bone_vec_meta_out_cur.append(reverse_joint_mapping[par])
    return bone_vec_meta_out_cur, bone_vec_meta_out_ref


def get_inv_cam(intri_mat, extri_R, extri_T):
    """
    all should be in (nview*batch, x, x)
    :param intri_mat:
    :param extri_R:
    :param extri_T:
    :return:
    """
    return torch.inverse(intri_mat), extri_R.permute(0, 2, 1).contiguous(), extri_T


class CamFusionModule(nn.Module):

    def __init__(self, nview, njoint, h, w, body, depth, joint_hm_mapping, selected_bones, config):
        super().__init__()
        self.nview = nview
        self.njoint = njoint
        self.h = h
        self.w = w
        self.selected_bones = selected_bones
        self.body = body
        self.nbones = len(self.selected_bones)
        self.depth = depth
        self.ndepth = depth.shape[0]
        self.joint_hm_mapping = joint_hm_mapping
        self.bone_vectors_meta = get_bone_vector_meta(selected_bones, body, joint_hm_mapping)
        self.config = config
        self.b_inview_fusion = config.CAM_FUSION.IN_VIEW_FUSION
        self.b_xview_self_fusion = config.CAM_FUSION.XVIEW_SELF_FUSION
        self.b_xview_fusion = config.CAM_FUSION.XVIEW_FUSION
        self.onehm = gen_hm_grid_coords(h, w)
        self.grid_norm_factor = torch.tensor([h - 1, w - 1, njoint - 1], dtype=torch.float32) / 2.0
        self.imu_bone_norm_factor = torch.ones(20, 1, 1)
        if config.DATASET.TRAIN_DATASET == 'totalcapture':
            self.imu_bone_norm_factor[2, 0, 0] = 2.0
            self.imu_bone_norm_factor[5, 0, 0] = 2.0
            self.imu_bone_norm_factor[15, 0, 0] = 2.0
            self.imu_bone_norm_factor[18, 0, 0] = 2.0
        elif config.DATASET.TRAIN_DATASET == 'multiview_h36m':
            self.imu_bone_norm_factor[2, 0, 0] = 2.0
            self.imu_bone_norm_factor[5, 0, 0] = 2.0
            self.imu_bone_norm_factor[15, 0, 0] = 2.0
            self.imu_bone_norm_factor[18, 0, 0] = 2.0

    def forward(self, heatmaps, affine_trans, cam_Intri, cam_R, cam_T, inv_affine_trans, bone_vectors_tensor):
        dev = heatmaps.device
        batch = heatmaps.shape[0] // self.nview
        self.grid_norm_factor = self.grid_norm_factor
        self.onehm = self.onehm
        cam_Intri = cam_Intri
        cam_R = cam_R
        cam_T = cam_T
        affine_trans = affine_trans
        inv_affine_trans = inv_affine_trans
        bone_vectors_tensor = bone_vectors_tensor
        imu_bone_norm_factor = self.imu_bone_norm_factor
        inv_cam_Intri, inv_cam_R, inv_cam_T = get_inv_cam(cam_Intri, cam_R, cam_T)
        out_grid_g_all_joints = batch_uv_to_global_with_multi_depth(self.onehm, inv_affine_trans, inv_cam_Intri, inv_cam_R, inv_cam_T, self.depth, self.njoint // 2)
        heatmaps_5d = heatmaps.view(self.nview * batch, 1, self.njoint, self.h, self.w)
        inview_fused = None
        xview_self_fused = None
        xview_fused = None
        if self.b_inview_fusion:
            out_grid_g = out_grid_g_all_joints[:, :, :, :, :16]
            ref_coords = apply_bone_offset(out_grid_g, bone_vectors_tensor)
            ref_coords_hm = batch_global_to_uv(ref_coords, affine_trans, cam_Intri, cam_R, cam_T)
            ref_bone_meta = torch.as_tensor(self.bone_vectors_meta[1])
            ref_bone_meta_3 = torch.zeros(len(ref_bone_meta), 3)
            ref_bone_meta_3[:, 2] = ref_bone_meta
            ref_bone_meta_3 = ref_bone_meta_3
            ref_bone_meta = ref_bone_meta_3.view(1, self.nbones * 2, 1, 1, 3)
            ref_bone_meta_expand = ref_bone_meta.expand(self.nview * batch, self.nbones * 2, self.ndepth, self.h * self.w, 3)
            ref_coords_hm = ref_coords_hm.permute(0, 4, 2, 3, 1).contiguous()
            ref_coords_hm[:, :, :, :, 2] = 0.0
            ref_coords_hm = ref_coords_hm + ref_bone_meta_expand
            ref_coords_flow = ref_coords_hm / self.grid_norm_factor - 1.0
            sampled_hm = grid_sample(input=heatmaps_5d, grid=ref_coords_flow, mode='nearest')
            sum_sampled_hm_over_depth = torch.max(sampled_hm, dim=3)[0].view(self.nview * batch * self.nbones * 2, self.h, self.w)
            fusion_hm = torch.zeros(self.nview * batch, self.njoint, self.h, self.w, device=dev)
            idx_view_batch = torch.linspace(0, self.nview * batch - 1, self.nview * batch).type(torch.long)
            idx_dst_bones = torch.tensor(self.bone_vectors_meta[0], dtype=torch.long)
            idx_put_grid_h, idx_put_grid_w = torch.meshgrid(idx_view_batch, idx_dst_bones)
            idx_put_grid_flat = idx_put_grid_h.contiguous().view(-1), idx_put_grid_w.contiguous().view(-1)
            fusion_hm.index_put_(idx_put_grid_flat, sum_sampled_hm_over_depth, accumulate=True)
            inview_fused = fusion_hm / imu_bone_norm_factor
        if self.b_xview_self_fusion:
            assert 1 == 0, 'Not implemented !'
        if self.b_xview_fusion:
            out_grid_g = out_grid_g_all_joints[:, :, :, :, :16]
            ref_coords = apply_bone_offset(out_grid_g, bone_vectors_tensor)
            ref_bone_meta = torch.as_tensor(self.bone_vectors_meta[1])
            ref_bone_meta_3 = torch.zeros(len(ref_bone_meta), 3)
            ref_bone_meta_3[:, 2] = ref_bone_meta
            ref_bone_meta_3 = ref_bone_meta_3
            ref_bone_meta = ref_bone_meta_3.view(1, self.nbones * 2, 1, 1, 3)
            ref_bone_meta_expand = ref_bone_meta.expand(self.nview * batch, self.nbones * 2, self.ndepth, self.h * self.w, 3)
            affine_trans_bv = affine_trans.view(batch, self.nview, *affine_trans.shape[1:])
            cam_Intri_bv = cam_Intri.view(batch, self.nview, *cam_Intri.shape[1:])
            cam_R_bv = cam_R.view(batch, self.nview, *cam_R.shape[1:])
            cam_T_bv = cam_T.view(batch, self.nview, *cam_T.shape[1:])
            xview_hm_sumed = torch.zeros(batch * self.nview, self.njoint, self.h, self.w, device=dev)
            for offset in range(self.nview):
                affine_trans_shift = self.roll_on_dim1(affine_trans_bv, offset=offset)
                cam_Intri_shift = self.roll_on_dim1(cam_Intri_bv, offset=offset)
                cam_R_shift = self.roll_on_dim1(cam_R_bv, offset=offset)
                cam_T_shift = self.roll_on_dim1(cam_T_bv, offset=offset)
                ref_coords_hm = batch_global_to_uv(ref_coords, affine_trans_shift, cam_Intri_shift, cam_R_shift, cam_T_shift)
                heatmaps_5d_shift = heatmaps.view(batch, self.nview, 1, self.njoint, self.h, self.w)
                heatmaps_5d_shift = self.roll_on_dim1(heatmaps_5d_shift, offset=offset)
                heatmaps_5d_shift = heatmaps_5d_shift.view(batch * self.nview, 1, self.njoint, self.h, self.w)
                ref_coords_hm = ref_coords_hm.permute(0, 4, 2, 3, 1).contiguous()
                ref_coords_hm[:, :, :, :, 2] = 0.0
                ref_coords_hm = ref_coords_hm + ref_bone_meta_expand
                ref_coords_flow = ref_coords_hm / self.grid_norm_factor - 1.0
                sampled_hm = grid_sample(input=heatmaps_5d_shift, grid=ref_coords_flow, mode='nearest')
                sum_sampled_hm_over_depth = torch.max(sampled_hm, dim=3)[0].view(batch * self.nview * self.nbones * 2, self.h, self.w)
                fusion_hm = torch.zeros(batch, self.nview, self.njoint, self.h, self.w, device=dev)
                idx_batch = torch.linspace(0, batch - 1, batch).type(torch.long)
                idx_view = (torch.linspace(0, self.nview - 1, self.nview).type(torch.long) + offset) % self.nview
                idx_dst_bones = torch.tensor(self.bone_vectors_meta[0], dtype=torch.long)
                idx_put_grid_b, idx_put_grid_v, idx_put_grid_j = torch.meshgrid(idx_batch, idx_view, idx_dst_bones)
                idx_put_grid_flat = idx_put_grid_b.contiguous().view(-1), idx_put_grid_v.contiguous().view(-1), idx_put_grid_j.contiguous().view(-1)
                fusion_hm.index_put_(idx_put_grid_flat, sum_sampled_hm_over_depth, accumulate=True)
                fusion_hm = torch.zeros(batch * self.nview, self.njoint, self.h, self.w, device=dev)
                idx_view_batch = torch.linspace(0, self.nview * batch - 1, self.nview * batch).type(torch.long)
                idx_dst_bones = torch.tensor(self.bone_vectors_meta[0], dtype=torch.long)
                idx_put_grid_h, idx_put_grid_w = torch.meshgrid(idx_view_batch, idx_dst_bones)
                idx_put_grid_flat = idx_put_grid_h.contiguous().view(-1), idx_put_grid_w.contiguous().view(-1)
                fusion_hm.index_put_(idx_put_grid_flat, sum_sampled_hm_over_depth, accumulate=True)
                fusion_hm = fusion_hm / imu_bone_norm_factor
                xview_hm_sumed += fusion_hm
            xview_fused = xview_hm_sumed / self.nview
        return inview_fused, xview_self_fused, xview_fused

    def roll_on_dim1(self, tensor, offset, maxoffset=None):
        if maxoffset is None:
            maxoffset = self.nview
        offset = offset % maxoffset
        part1 = tensor[:, :offset]
        part2 = tensor[:, offset:]
        res = torch.cat((part2, part1), dim=1).contiguous()
        return res


class HumanBody(object):

    def __init__(self):
        self.skeleton = self.get_skeleton()
        self.skeleton_sorted_by_level = self.sort_skeleton_by_level(self.skeleton)

    def get_skeleton(self):
        joint_names = ['root', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 'lank', 'belly', 'neck', 'nose', 'lsho', 'lelb', 'lwri', 'rsho', 'relb', 'rwri']
        children = [[1, 4, 7], [2], [3], [], [5], [6], [], [8], [9, 10, 13], [], [11], [12], [], [14], [15], []]
        imubone = [[-1, -1, -1], [3], [4], [], [5], [6], [], [-1], [-1, -1, -1], [], [11], [12], [], [13], [14], []]
        skeleton = []
        for i in range(len(joint_names)):
            skeleton.append({'idx': i, 'name': joint_names[i], 'children': children[i], 'imubone': imubone[i]})
        return skeleton

    def sort_skeleton_by_level(self, skeleton):
        njoints = len(skeleton)
        level = np.zeros(njoints)
        queue = [skeleton[0]]
        while queue:
            cur = queue[0]
            for child in cur['children']:
                skeleton[child]['parent'] = cur['idx']
                level[child] = level[cur['idx']] + 1
                queue.append(skeleton[child])
            del queue[0]
        desc_order = np.argsort(level)[::-1]
        sorted_skeleton = []
        for i in desc_order:
            skeleton[i]['level'] = level[i]
            sorted_skeleton.append(skeleton[i])
        return sorted_skeleton


class MultiViewPose(nn.Module):

    def __init__(self, PoseResNet, CFG):
        super(MultiViewPose, self).__init__()
        self.config = CFG
        if self.config.DATASET.TRAIN_DATASET == 'multiview_h36m':
            selected_bones = [3, 4, 5, 6, 12, 13, 14, 15]
            general_joint_mapping = {(0): 0, (1): 1, (2): 2, (3): 3, (4): 4, (5): 5, (6): 6, (7): 7, (8): '*', (9): 8, (10): '*', (11): 9, (12): 10, (13): '*', (14): 11, (15): 12, (16): 13, (17): 14, (18): 15, (19): 16}
            imu_related_joint_in_hm = [1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19]
        elif self.config.DATASET.TRAIN_DATASET == 'totalcapture':
            selected_bones = [3, 4, 5, 6, 11, 12, 13, 14]
            general_joint_mapping = {(0): 0, (1): 1, (2): 2, (3): 3, (4): 4, (5): 5, (6): 6, (7): 7, (8): '*', (9): 8, (10): '*', (11): 9, (12): '*', (13): '*', (14): 10, (15): 11, (16): 12, (17): 13, (18): 14, (19): 15}
            imu_related_joint_in_hm = [1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19]
        self.resnet = PoseResNet
        self.b_in_view_fusion = self.config.CAM_FUSION.IN_VIEW_FUSION
        self.b_xview_self_fusion = self.config.CAM_FUSION.XVIEW_SELF_FUSION
        self.b_xview_fusion = self.config.CAM_FUSION.XVIEW_FUSION
        njoints = self.config.NETWORK.NUM_JOINTS
        h = int(self.config.NETWORK.HEATMAP_SIZE[0])
        w = int(self.config.NETWORK.HEATMAP_SIZE[1])
        self.selected_views = self.config.SELECTED_VIEWS
        nview = len(self.selected_views)
        body = HumanBody()
        depth = torch.logspace(2.7, 3.9, steps=100)
        ndepth = depth.shape[0]
        self.h = h
        self.w = w
        self.njoints = njoints
        self.nview = nview
        joint_channel_mask = torch.zeros(1, 20, 1, 1)
        for sj in imu_related_joint_in_hm:
            joint_channel_mask[0, sj, 0, 0] = 1.0
        self.register_buffer('joint_channel_mask', joint_channel_mask)
        if self.b_in_view_fusion or self.b_xview_fusion:
            self.cam_fusion_module = CamFusionModule(nview, njoints, h, w, body, depth, general_joint_mapping, selected_bones, self.config)

    def forward(self, inputs, **kwargs):
        dev = inputs.device
        meta = dict()
        for kk in kwargs:
            meta[kk] = self.merge_first_two_dims(kwargs[kk])
        batch = inputs.shape[0]
        nview = inputs.shape[1]
        inputs = inputs.view(batch * nview, *inputs.shape[2:])
        hms, feature_before_final = self.resnet(inputs)
        if self.b_in_view_fusion or self.b_xview_fusion or self.b_xview_self_fusion:
            cam_R = meta['camera_R']
            cam_T = meta['camera_T']
            cam_Intri = meta['camera_Intri']
            inv_cam_Intri, inv_cam_R, inv_cam_T = get_inv_cam(cam_Intri, cam_R, cam_T)
            affine_trans = meta['affine_trans']
            inv_affine_trans = meta['inv_affine_trans']
            bones_tensor = meta['bone_vectors_tensor']
            if self.b_in_view_fusion or self.b_xview_fusion:
                inview_hm, _, xview_hm = self.cam_fusion_module.forward(hms, affine_trans, cam_Intri, cam_R, cam_T, inv_affine_trans, bones_tensor)
        extra = dict()
        extra['joint_channel_mask'] = self.joint_channel_mask
        extra['origin_hms'] = hms
        b_imu_fuse = False
        if self.b_xview_fusion:
            extra['fused_hms'] = xview_hm
            b_imu_fuse = True
        if self.b_in_view_fusion:
            extra['fused_hms'] = inview_hm
            b_imu_fuse = True
        extra['imu_fuse'] = b_imu_fuse
        if b_imu_fuse:
            imu_joint_mask = extra['joint_channel_mask'][0]
            non_imu_joint_mask = -0.5 * imu_joint_mask + 1.0
            output_hms = extra['fused_hms'] * 0.5 + extra['origin_hms'] * non_imu_joint_mask
        else:
            output_hms = extra['origin_hms']
        return output_hms, extra

    def merge_first_two_dims(self, tensor):
        dim0 = tensor.shape[0]
        dim1 = tensor.shape[1]
        left = tensor.shape[2:]
        return tensor.view(dim0 * dim1, *left)


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        self.cfgs = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        input_channel = int(32 * width_mult)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            layers.append(block(input_channel, output_channel, s, t))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


logger = logging.getLogger(__name__)


class PoseMobileNetV2(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 320
        self.deconv_with_bias = cfg.POSE_RESNET.DECONV_WITH_BIAS
        super(PoseMobileNetV2, self).__init__()
        self.mobilenetv2 = MobileNetV2()
        self.deconv_layers = self._make_deconv_layer(cfg.POSE_RESNET.NUM_DECONV_LAYERS, cfg.POSE_RESNET.NUM_DECONV_FILTERS, cfg.POSE_RESNET.NUM_DECONV_KERNELS)
        self.final_layer = nn.Conv2d(in_channels=cfg.POSE_RESNET.NUM_DECONV_FILTERS[-1], out_channels=cfg.NETWORK.NUM_JOINTS, kernel_size=cfg.POSE_RESNET.FINAL_CONV_KERNEL, stride=1, padding=1 if cfg.POSE_RESNET.FINAL_CONV_KERNEL == 3 else 0)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.mobilenetv2(x)
        x = self.deconv_layers(x)
        x_final_feature = x
        x = self.final_layer(x)
        return x, x_final_feature

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = cfg.POSE_RESNET.DECONV_WITH_BIAS
        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.deconv_layers = self._make_deconv_layer(cfg.POSE_RESNET.NUM_DECONV_LAYERS, cfg.POSE_RESNET.NUM_DECONV_FILTERS, cfg.POSE_RESNET.NUM_DECONV_KERNELS)
        self.final_layer = nn.Conv2d(in_channels=cfg.POSE_RESNET.NUM_DECONV_FILTERS[-1], out_channels=cfg.NETWORK.NUM_JOINTS, kernel_size=cfg.POSE_RESNET.FINAL_CONV_KERNEL, stride=1, padding=1 if cfg.POSE_RESNET.FINAL_CONV_KERNEL == 3 else 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), 'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)
            planes = num_filters[i]
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2, padding=padding, output_padding=output_padding, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv_layers(x)
        x_final_feature = x
        x = self.final_layer(x)
        return x, x_final_feature

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


def compute_grid(boxSize, boxCenter, nBins):
    grid1D = np.linspace(-boxSize / 2, boxSize / 2, nBins)
    gridx, gridy, gridz = np.meshgrid(grid1D + boxCenter[0], grid1D + boxCenter[1], grid1D + boxCenter[2])
    dimensions = gridx.shape[0] * gridx.shape[1] * gridx.shape[2]
    gridx, gridy, gridz = np.reshape(gridx, (dimensions, -1)), np.reshape(gridy, (dimensions, -1)), np.reshape(gridz, (dimensions, -1))
    grid = np.concatenate((gridx, gridy, gridz), axis=1)
    return grid


def affine_transform_pts(pts, t):
    """

    :param pts:
    :param t:
    :return:
    """
    xyz = np.add(np.array([[1, 0], [0, 1], [0, 0]]).dot(pts.T), np.array([[0], [0], [1]]))
    return np.dot(t, xyz).T


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def compute_unary_term(heatmap, grid, bbox2D, cam, imgSize, **kwargs):
    """
    Args:
        heatmap: array of size (n * k * h * w)
                -n: number of views,  -k: number of joints
                -h: heatmap height,   -w: heatmap width
        grid: list of k ndarrays of size (nbins * 3)
                    -k: number of joints; 1 when the grid is shared in PSM
                    -nbins: number of bins in the grid
        bbox2D: bounding box on which heatmap is computed
    Returns:
        unary_of_all_joints: a list of ndarray of size nbins
    """
    n, k = heatmap.shape[0], heatmap.shape[1]
    h, w = heatmap.shape[2], heatmap.shape[3]
    nbins = grid[0].shape[0]
    current_device = torch.device('cuda:{}'.format(heatmap.get_device()))
    heatmaps = heatmap
    grid_cords = np.zeros([n, k, nbins, 2], dtype=np.float32)
    for c in range(n):
        for j in range(k):
            grid_id = 0 if len(grid) == 1 else j
            xy = cameras.project_pose(grid[grid_id], cam[c])
            trans = get_affine_transform(bbox2D[c]['center'], bbox2D[c]['scale'], 0, imgSize)
            xy = affine_transform_pts(xy, trans) * np.array([w, h]) / imgSize
            if len(grid) == 1:
                grid_cords[c, 0, :, :] = xy / np.array([h - 1, w - 1], dtype=np.float32) * 2.0 - 1.0
                for j in range(1, k):
                    grid_cords[c, j, :, :] = grid_cords[c, 0, :, :]
                break
            else:
                grid_cords[c, j, :, :] = xy / np.array([h - 1, w - 1], dtype=np.float32) * 2.0 - 1.0
    grid_cords_tensor = torch.as_tensor(grid_cords)
    unary_all_views_joints = grid_sample(heatmaps, grid_cords_tensor)
    unary_all_views = torch.zeros(n, k, nbins)
    for j in range(k):
        unary_all_views[:, j, :] = unary_all_views_joints[:, j, j, :]
    unary_tensor = torch.zeros(k, nbins)
    for una in unary_all_views:
        unary_tensor = torch.add(unary_tensor, una)
    return unary_tensor


def get_loc_from_cube_idx(grid, pose3d_as_cube_idx):
    """
    Estimate 3d joint locations from cube index.

    Args:
        grid: a list of grids 
        pose3d_as_cube_idx: a list of tuples (joint_idx, cube_idx)
    Returns:
        pose3d: 3d pose 
    """
    njoints = len(pose3d_as_cube_idx)
    pose3d = np.zeros(shape=[njoints, 3])
    is_single_grid = len(grid) == 1
    for joint_idx, cube_idx in pose3d_as_cube_idx:
        gridid = 0 if is_single_grid else joint_idx
        pose3d[joint_idx] = grid[gridid][cube_idx]
    return pose3d


def infer(unary, pairwise, body, config, **kwargs):
    """
    Args:
        unary: a list of unary terms for all JOINTS
        pairwise: a list of pairwise terms of all EDGES
        body: tree structure human body
    Returns:
        pose3d_as_cube_idx: 3d pose as cube index
    """
    current_device = kwargs['current_device']
    skeleton = body.skeleton
    skeleton_sorted_by_level = body.skeleton_sorted_by_level
    root_idx = config.DATASET.ROOTIDX
    nbins = len(unary[root_idx])
    states_of_all_joints = {}
    for node in skeleton_sorted_by_level:
        children_state = []
        unary_current = unary[node['idx']]
        if len(node['children']) == 0:
            energy = unary[node['idx']].squeeze()
            children_state = [[-1]] * len(energy)
        else:
            children = node['children']
            for child in children:
                child_energy = states_of_all_joints[child]['Energy'].squeeze()
                pairwise_mat = pairwise[node['idx'], child]
                unary_child = torch.tensor(child_energy, dtype=torch.float32).expand_as(pairwise_mat)
                unary_child_with_pairwise = torch.mul(pairwise_mat, unary_child)
                max_v, max_i = torch.max(unary_child_with_pairwise, dim=1)
                unary_current = torch.mul(unary_current, max_v)
                children_state.append(max_i.detach().cpu().numpy())
            children_state = np.array(children_state).T
        res = {'Energy': unary_current.detach().cpu().numpy(), 'State': children_state}
        states_of_all_joints[node['idx']] = res
    pose3d_as_cube_idx = []
    energy = states_of_all_joints[root_idx]['Energy']
    cube_idx = np.argmax(energy)
    pose3d_as_cube_idx.append([root_idx, cube_idx])
    queue = pose3d_as_cube_idx.copy()
    while queue:
        joint_idx, cube_idx = queue.pop(0)
        children_state = states_of_all_joints[joint_idx]['State']
        state = children_state[cube_idx]
        children_index = skeleton[joint_idx]['children']
        if -1 not in state:
            for joint_idx, cube_idx in zip(children_index, state):
                pose3d_as_cube_idx.append([joint_idx, cube_idx])
                queue.append([joint_idx, cube_idx])
    pose3d_as_cube_idx.sort()
    return pose3d_as_cube_idx


def compute_pairwise_constrain(skeleton, limb_length, grid, tolerance, **kwargs):
    do_bone_vectors = False
    if 'do_bone_vectors' in kwargs:
        if kwargs['do_bone_vectors']:
            do_bone_vectors = True
            bone_vectors = kwargs['bone_vectors']
    pairwise_constrain = {}
    for node in skeleton:
        current = node['idx']
        children = node['children']
        if do_bone_vectors:
            bone_index = node['imubone']
        for idx_child, child in enumerate(children):
            expect_length = limb_length[current, child]
            if do_bone_vectors:
                if bone_index[idx_child] >= 0:
                    expect_orient_vector = bone_vectors[bone_index[idx_child]]
                    norm_expect_orient_vector = expect_orient_vector / (np.linalg.norm(expect_orient_vector) + 1e-09)
            nbin_current = len(grid[current])
            nbin_child = len(grid[child])
            constrain_array = np.zeros((nbin_current, nbin_child), dtype=np.float32)
            for i in range(nbin_current):
                for j in range(nbin_child):
                    actual_length = np.linalg.norm(grid[current][i] - grid[child][j]) + 1e-09
                    offset = np.abs(actual_length - expect_length)
                    if offset <= tolerance:
                        constrain_array[i, j] = 1
                    if do_bone_vectors and bone_index[idx_child] >= 0:
                        acutal_orient_vector = (grid[current][i] - grid[child][j]) / actual_length
                        cos_theta = np.dot(-norm_expect_orient_vector, acutal_orient_vector)
                        constrain_array[i, j] *= cos_theta
            pairwise_constrain[current, child] = constrain_array
    return pairwise_constrain


def recursive_infer(initpose, cams, heatmaps, boxes, img_size, heatmap_size, body, limb_length, grid_size, nbins, tolerance, config, **kwargs):
    current_device = kwargs['current_device']
    k = initpose.shape[0]
    grids = []
    for i in range(k):
        point = initpose[i]
        grid = compute_grid(grid_size, point, nbins)
        grids.append(grid)
    unary = compute_unary_term(heatmaps, grids, boxes, cams, img_size)
    skeleton = body.skeleton
    pairwise_constrain = compute_pairwise_constrain(skeleton, limb_length, grids, tolerance, **kwargs)
    pairwise_tensor = dict()
    for edge in pairwise_constrain:
        edge_pairwise = torch.as_tensor(pairwise_constrain[edge], dtype=torch.float32)
        pairwise_tensor[edge] = edge_pairwise
    pairwise_constrain = pairwise_tensor
    kwargs_infer = kwargs
    pose3d_cube = infer(unary, pairwise_constrain, body, config, **kwargs_infer)
    pose3d = get_loc_from_cube_idx(grids, pose3d_cube)
    return pose3d


def rpsm(cams, heatmaps, boxes, grid_center, limb_length, pairwise_constraint, config, **kwargs):
    """
    Args:
        cams : camera parameters for each view
        heatmaps: 2d pose heatmaps (n, k, h, w)
        boxes: on which the heatmaps are computed; n dictionaries
        grid_center: 3d location of the root
        limb_length: template limb length
        pairwise_constrain: pre-computed pairwise terms (iteration 0 psm only)
    Returns:
        pose3d: 3d pose
    """
    image_size = config.NETWORK.IMAGE_SIZE
    heatmap_size = config.NETWORK.HEATMAP_SIZE
    first_nbins = config.PICT_STRUCT.FIRST_NBINS
    recur_nbins = config.PICT_STRUCT.RECUR_NBINS
    recur_depth = config.PICT_STRUCT.RECUR_DEPTH
    grid_size = config.PICT_STRUCT.GRID_SIZE
    tolerance = config.PICT_STRUCT.LIMB_LENGTH_TOLERANCE
    current_device = kwargs['current_device']
    body = kwargs['human_body']
    grid = compute_grid(grid_size, grid_center, first_nbins)
    heatmaps = torch.as_tensor(heatmaps, dtype=torch.float32)
    extra_kwargs = kwargs
    do_bone_vectors = False
    if 'do_bone_vectors' in kwargs:
        if kwargs['do_bone_vectors']:
            do_bone_vectors = True
            bone_vectors = kwargs['bone_vectors']
    if do_bone_vectors:
        orient_pairwise = kwargs['orient_pairwise']
        new_pairwise_constrain = {}
        for node in body.skeleton:
            current = node['idx']
            children = node['children']
            bone_index = node['imubone']
            for idx_child, child in enumerate(children):
                constrain_array = pairwise_constraint[current, child]
                if bone_index[idx_child] >= 0:
                    expect_orient_vector = bone_vectors[bone_index[idx_child]]
                    expect_orient_vector = torch.as_tensor(expect_orient_vector, dtype=torch.float32)
                    norm_expect_orient_vector = expect_orient_vector / (torch.norm(expect_orient_vector) + 1e-09)
                    norm_expect_orient_vector = norm_expect_orient_vector.view(-1)
                    acutal_orient_vector = orient_pairwise
                    cos_theta = torch.matmul(acutal_orient_vector, -norm_expect_orient_vector)
                    constrain_array = torch.mul(constrain_array, cos_theta)
                new_pairwise_constrain[current, child] = constrain_array
        pairwise_constraint = new_pairwise_constrain
    unary = compute_unary_term(heatmaps, [grid], boxes, cams, image_size)
    pose3d_as_cube_idx = infer(unary, pairwise_constraint, body, config, **extra_kwargs)
    pose3d = get_loc_from_cube_idx([grid], pose3d_as_cube_idx)
    cur_grid_size = grid_size / first_nbins
    for i in range(recur_depth):
        pose3d = recursive_infer(pose3d, cams, heatmaps, boxes, image_size, heatmap_size, body, limb_length, cur_grid_size, recur_nbins, tolerance, config, **extra_kwargs)
        cur_grid_size = cur_grid_size / recur_nbins
    return pose3d


class RpsmFunc(nn.Module):

    def __init__(self, pairwise_constraint, human_body, **kwargs):
        super().__init__()
        self.current_device = None
        self.pairwise_constraint = dict()
        for idx, k in enumerate(pairwise_constraint):
            buff_name = 'pairwise_constraint_{}'.format(idx)
            self.register_buffer(buff_name, pairwise_constraint[k])
            self.pairwise_constraint[k] = self.__getattr__(buff_name)
        self.human_body = human_body
        self.do_bone_vectors = kwargs['do_bone_vectors']
        if self.do_bone_vectors:
            orient_pairwise = kwargs['orient_pairwise']
            self.register_buffer('orient_pairwise', orient_pairwise)

    def __call__(self, *args, **kwargs):
        if self.current_device is None:
            self.current_device = torch.device('cuda:{}'.format(list(self.pairwise_constraint.values())[0].get_device()))
        extra_kwargs = dict()
        extra_kwargs['human_body'] = self.human_body
        extra_kwargs['current_device'] = self.current_device
        if self.do_bone_vectors:
            extra_kwargs['orient_pairwise'] = self.orient_pairwise
        return rpsm(pairwise_constraint=self.pairwise_constraint, **kwargs, **extra_kwargs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (JointsMSELoss,
     lambda: ([], {'use_target_weight': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_CHUNYUWANG_imu_human_pose_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

