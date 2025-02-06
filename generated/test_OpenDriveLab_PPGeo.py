import sys
_module = sys.modules[__name__]
del sys
data_ytb = _module
layers = _module
loss = _module
model = _module
networks = _module
depth_decoder = _module
pose_cnn = _module
pose_decoder = _module
resnet_encoder = _module
data = _module
gen_data = _module
planning_model = _module
resnet = _module
train_planning = _module
resnet = _module
train = _module
ytb_data_preprocess = _module

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


import random


import numpy as np


from torch.utils.data import Dataset


from torchvision import transforms as T


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch import nn


from collections import OrderedDict


import torchvision.models as models


import torch.utils.model_zoo as model_zoo


import torch.utils.data


import torchvision


from torch import Tensor


from torch.hub import load_state_dict_from_url


from typing import Type


from typing import Any


from typing import Callable


from typing import Union


from typing import List


from typing import Optional


import torch.optim as optim


from torch.utils.data import DataLoader


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)
        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

    def __init__(self, batch_size, height, width, eps=1e-07):
        super(Project3D, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class Ternary(nn.Module):

    def __init__(self, device):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float()

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf ** 2)
        return transf_norm

    def rgb2gray(self, rgb, normalize=False):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        if normalize:
            r = r * 0.229 + 0.485
            g = g * 0.224 + 0.456
            b = b * 0.225 + 0.406
        gray = 0.2989 * r + 0.587 * g + 0.114 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1, normalize=False):
        self.w = self.w.type_as(img0)
        img0 = self.transform(self.rgb2gray(img0, normalize))
        img1 = self.transform(self.rgb2gray(img1, normalize))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)


class EPE(nn.Module):

    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-06) ** 0.5
        return loss_map * loss_mask


class PoseDecoder(nn.Module):

    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.convs = OrderedDict()
        self.convs['squeeze'] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs['pose', 0] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs['pose', 1] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs['pose', 2] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]
        cat_features = [self.relu(self.convs['squeeze'](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
        out = cat_features
        for i in range(3):
            out = self.convs['pose', i](out)
            if i != 2:
                out = self.relu(out)
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], 'Can only run with 18 or 50 layer resnet'
    blocks = {(18): [2, 2, 2, 2], (50): [3, 4, 6, 3]}[num_layers]
    block_type = {(18): models.resnet.BasicBlock, (50): models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {(18): models.resnet18, (34): models.resnet34, (50): models.resnet50, (101): models.resnet101, (152): models.resnet152}
        if num_layers not in resnets:
            raise ValueError('{} is not a valid number of resnet layers'.format(num_layers))
        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image, normalize=False):
        self.features = []
        if normalize:
            std = torch.tensor([0.229, 0.224, 0.225]).type_as(input_image).view(1, 3, 1, 1).repeat(1, input_image.shape[1] // 3, 1, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).type_as(input_image).view(1, 3, 1, 1).repeat(1, input_image.shape[1] // 3, 1, 1)
            x = (input_image - mean) / std
        else:
            x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features


class MotionNet(nn.Module):

    def __init__(self):
        super(MotionNet, self).__init__()
        self.visual_encoder = ResnetEncoder(34, True, num_input_images=1)
        self.motion_decoder = PoseDecoder(self.visual_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)

    def forward(self, inputs):
        motion_inputs1 = inputs['color_aug', -1, 0]
        motion_inputs2 = inputs['color_aug', 0, 0]
        enc1 = self.visual_encoder(motion_inputs1, normalize=True)
        enc2 = self.visual_encoder(motion_inputs2, normalize=True)
        axisangle1, translation1 = self.motion_decoder([enc1])
        axisangle2, translation2 = self.motion_decoder([enc2])
        return axisangle1, translation1, axisangle2, translation2


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode='nearest')


class DepthDecoder(nn.Module):

    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs['upconv', i, 0] = ConvBlock(num_ch_in, num_ch_out)
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs['upconv', i, 1] = ConvBlock(num_ch_in, num_ch_out)
        for s in self.scales:
            self.convs['dispconv', s] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs['upconv', i, 0](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs['upconv', i, 1](x)
            if i in self.scales:
                self.outputs['disp', i] = self.sigmoid(self.convs['dispconv', i](x))
        return self.outputs


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4)
    t = translation_vector.contiguous().view(-1, 3, 1)
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t
    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-07)
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca
    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)
    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    rot = torch.zeros((vec.shape[0], 4, 4))
    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1
    return rot


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()
    if invert:
        R = R.transpose(1, 2)
        t *= -1
    T = get_translation_matrix(t)
    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)
    return M


class Monodepth(nn.Module):

    def __init__(self, stage=1, batch_size=1):
        super(Monodepth, self).__init__()
        self.stage = stage
        self.num_scales = len([0, 1, 2, 3])
        self.scales = [0, 1, 2, 3]
        self.frame_ids = [0, -1, 1]
        self.height = 160
        self.width = 320
        self.num_input_frames = len([0, -1, 1])
        self.num_pose_frames = 2
        self.min_depth = 0.1
        self.max_depth = 100.0
        self.depth_encoder = ResnetEncoder(18, True)
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, self.scales)
        self.pose_encoder = ResnetEncoder(18, True, num_input_images=self.num_pose_frames)
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        self.fl = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, 2), nn.Softplus())
        self.offset = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Linear(256, 2), nn.Sigmoid())
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.scales:
            h = self.height // 2 ** scale
            w = self.width // 2 ** scale
            self.backproject_depth[scale] = BackprojectDepth(batch_size, h, w)
            self.project_3d[scale] = Project3D(batch_size, h, w)
        self.ssim = SSIM()
        self.initialized = False

    def initialize(self):
        for scale in self.scales:
            self.backproject_depth[scale] = self.backproject_depth[scale]
            self.project_3d[scale] = self.project_3d[scale]
            self.initialized = True

    def forward_stage1(self, inputs):
        if not self.initialized:
            self.device = inputs['color_aug', 0, 0].device
            self.initialize()
        features = self.depth_encoder(inputs['color_aug', 0, 0], normalize=True)
        outputs = self.depth_decoder(features)
        pose_feats = {f_i: inputs['color_aug', f_i, 0] for f_i in self.frame_ids}
        poses_inputs1 = torch.cat([pose_feats[-1], pose_feats[0]], 1)
        poses_inputs2 = torch.cat([pose_feats[0], pose_feats[1]], 1)
        pose_enc1 = self.pose_encoder(poses_inputs1, normalize=True)
        pose_enc2 = self.pose_encoder(poses_inputs2, normalize=True)
        feature_pooled1 = torch.flatten(self.avg_pooling(pose_enc1[-1]), 1)
        fl1 = self.fl(feature_pooled1)
        offsets1 = self.offset(feature_pooled1)
        feature_pooled2 = torch.flatten(self.avg_pooling(pose_enc2[-1]), 1)
        fl2 = self.fl(feature_pooled2)
        offsets2 = self.offset(feature_pooled2)
        K1 = self.compute_K(fl1, offsets1)
        K2 = self.compute_K(fl2, offsets2)
        K = (K1 + K2) / 2
        inputs = self.add_K(K, inputs)
        axisangle1, translation1 = self.pose_decoder([pose_enc1])
        axisangle2, translation2 = self.pose_decoder([pose_enc2])
        outputs['axisangle', 0, -1] = axisangle1
        outputs['translation', 0, -1] = translation1
        outputs['axisangle', 0, 1] = axisangle2
        outputs['translation', 0, 1] = translation2
        outputs['cam_T_cam', 0, -1] = transformation_from_parameters(axisangle1[:, 0], translation1[:, 0], invert=True)
        outputs['cam_T_cam', 0, 1] = transformation_from_parameters(axisangle2[:, 0], translation2[:, 0], invert=False)
        for scale in self.scales:
            disp = outputs['disp', scale]
            disp = F.interpolate(disp, [self.height, self.width], mode='bilinear', align_corners=False)
            source_scale = 0
            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            outputs['depth', 0, scale] = depth
            for i, frame_id in enumerate(self.frame_ids[1:]):
                T = outputs['cam_T_cam', 0, frame_id]
                cam_points = self.backproject_depth[source_scale](depth, inputs['inv_K', source_scale])
                pix_coords = self.project_3d[source_scale](cam_points, inputs['K', source_scale], T)
                outputs['sample', frame_id, scale] = pix_coords
                outputs['color', frame_id, scale] = F.grid_sample(inputs['color', frame_id, source_scale], outputs['sample', frame_id, scale], padding_mode='border')
                outputs['color_identity', frame_id, scale] = inputs['color', frame_id, source_scale]
        losses = {}
        total_loss = 0
        for scale in self.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0
            disp = outputs['disp', scale]
            color = inputs['color', 0, scale]
            target = inputs['color', 0, source_scale]
            for frame_id in self.frame_ids[1:]:
                pred = outputs['color', frame_id, scale]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs['color', frame_id, source_scale]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_loss = identity_reprojection_losses
            reprojection_loss = reprojection_losses
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape, device=self.device) * 1e-05
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
            outputs['identity_selection/{}'.format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()
            loss += to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-07)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += 0.001 * smooth_loss / 2 ** scale
            total_loss += loss
            losses['loss/{}'.format(scale)] = loss
        total_loss /= self.num_scales
        losses['loss'] = total_loss
        return outputs, losses

    def forward_stage2(self, inputs, axisangle1, translation1, axisangle2, translation2):
        if not self.initialized:
            self.device = inputs['color_aug', 0, 0].device
            self.initialize()
        with torch.no_grad():
            features = self.depth_encoder(inputs['color', 0, 0], normalize=True)
            outputs = self.depth_decoder(features)
            pose_feats = {f_i: inputs['color', f_i, 0] for f_i in self.frame_ids}
            poses_inputs1 = torch.cat([pose_feats[-1], pose_feats[0]], 1)
            poses_inputs2 = torch.cat([pose_feats[0], pose_feats[1]], 1)
            pose_enc1 = self.pose_encoder(poses_inputs1, normalize=True)
            pose_enc2 = self.pose_encoder(poses_inputs2, normalize=True)
            feature_pooled1 = torch.flatten(self.avg_pooling(pose_enc1[-1]), 1)
            fl1 = self.fl(feature_pooled1)
            offsets1 = self.offset(feature_pooled1)
            feature_pooled2 = torch.flatten(self.avg_pooling(pose_enc2[-1]), 1)
            fl2 = self.fl(feature_pooled2)
            offsets2 = self.offset(feature_pooled2)
            K1 = self.compute_K(fl1, offsets1)
            K2 = self.compute_K(fl2, offsets2)
            K = (K1 + K2) / 2
            inputs = self.add_K(K, inputs)
        outputs['axisangle', 0, -1] = axisangle1
        outputs['translation', 0, -1] = translation1
        outputs['axisangle', 0, 1] = axisangle2
        outputs['translation', 0, 1] = translation2
        outputs['cam_T_cam', 0, -1] = transformation_from_parameters(axisangle1[:, 0], translation1[:, 0], invert=True)
        outputs['cam_T_cam', 0, 1] = transformation_from_parameters(axisangle2[:, 0], translation2[:, 0], invert=False)
        for scale in [0]:
            with torch.no_grad():
                disp = outputs['disp', scale]
                disp = F.interpolate(disp, [self.height, self.width], mode='bilinear', align_corners=False)
                source_scale = 0
                _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
                outputs['depth', 0, scale] = depth
            for i, frame_id in enumerate(self.frame_ids[1:]):
                T = outputs['cam_T_cam', 0, frame_id]
                cam_points = self.backproject_depth[source_scale](depth, inputs['inv_K', source_scale])
                pix_coords = self.project_3d[source_scale](cam_points, inputs['K', source_scale], T)
                outputs['sample', frame_id, scale] = pix_coords
                outputs['color', frame_id, scale] = F.grid_sample(inputs['color', frame_id, source_scale], outputs['sample', frame_id, scale], padding_mode='border')
                outputs['color_identity', frame_id, scale] = inputs['color', frame_id, source_scale]
        losses = {}
        total_loss = 0
        for scale in [0]:
            loss = 0
            reprojection_losses = []
            source_scale = 0
            disp = outputs['disp', scale]
            color = inputs['color', 0, scale]
            target = inputs['color', 0, source_scale]
            for frame_id in self.frame_ids[1:]:
                pred = outputs['color', frame_id, scale]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs['color', frame_id, source_scale]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_loss = identity_reprojection_losses
            reprojection_loss = reprojection_losses
            identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape, device=self.device) * 1e-05
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
            outputs['identity_selection/{}'.format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()
            loss += to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-07)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += 0.001 * smooth_loss / 2 ** scale
            total_loss += loss
            losses['loss/{}'.format(scale)] = loss
        total_loss /= self.num_scales
        losses['loss'] = total_loss
        return outputs, losses

    def forward(self, inputs, axisangle1=None, translation1=None, axisangle2=None, translation2=None):
        if self.stage == 1:
            return self.forward_stage1(inputs)
        else:
            return self.forward_stage2(inputs, axisangle1, translation1, axisangle2, translation2)

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
		"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

    def compute_K(self, fl, offsets):
        B = fl.shape[0]
        fl = torch.diag_embed(fl)
        K = torch.cat([fl, offsets.view(-1, 2, 1)], 2)
        row = torch.tensor([[0, 0, 1], [0, 0, 0]]).view(1, 2, 3).repeat(B, 1, 1).type_as(K)
        K = torch.cat([K, row], 1)
        col = torch.tensor([0, 0, 0, 1]).view(1, 4, 1).repeat(B, 1, 1).type_as(K)
        K = torch.cat([K, col], 2)
        return K

    def add_K(self, K, inputs):
        for scale in self.scales:
            K_scale = K.clone()
            K_scale[:, 0] *= self.width // 2 ** scale
            K_scale[:, 1] *= self.height // 2 ** scale
            inv_K_scale = torch.linalg.pinv(K_scale)
            inputs['K', scale] = K_scale
            inputs['inv_K', scale] = inv_K_scale
            return inputs


class PoseCNN(nn.Module):

    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()
        self.num_input_frames = num_input_frames
        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)
        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)
        self.num_convs = len(self.convs)
        self.relu = nn.ReLU(True)
        self.net = nn.ModuleList(list(self.convs.values()))
        self.fl = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, 2), nn.Softplus())
        self.offset = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True), nn.Linear(128, 2), nn.Sigmoid())
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, out):
        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)
        feature_pooled = torch.flatten(self.avg_pooling(out), 1)
        fl = self.fl(feature_pooled)
        offset = self.offset(feature_pooled) + 0.5
        out = self.pose_conv(out)
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)
        axisangle = out[..., :3]
        translation = out[..., 3:]
        return axisangle, translation, fl, offset


def conv3x3(in_planes: 'int', out_planes: 'int', stride: 'int'=1, groups: 'int'=1, dilation: 'int'=1) ->nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion: 'int' = 1

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: 'Tensor') ->Tensor:
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


def conv1x1(in_planes: 'int', out_planes: 'int', stride: 'int'=1) ->nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion: 'int' = 4

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        super(Bottleneck, self).__init__()
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
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: 'Tensor') ->Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block: 'Type[Union[BasicBlock, Bottleneck]]', layers: 'List[int]', num_classes: 'int'=1000, zero_init_residual: 'bool'=False, groups: 'int'=1, width_per_group: 'int'=64, replace_stride_with_dilation: 'Optional[List[bool]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None) ->None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: 'Type[Union[BasicBlock, Bottleneck]]', planes: 'int', blocks: 'int', stride: 'int'=1, dilate: 'bool'=False) ->nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x: 'Tensor') ->Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_layer4 = self.layer4(x)
        return x_layer4
        x = self.avgpool(x_layer4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, x_layer4

    def forward(self, x: 'Tensor') ->Tensor:
        return self._forward_impl(x)


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth', 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth', 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth', 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth', 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'}


def _resnet(arch: 'str', block: 'Type[Union[BasicBlock, Bottleneck]]', layers: 'List[int]', pretrained: 'bool', progress: 'bool', **kwargs: Any) ->ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet34(pretrained: 'bool'=False, progress: 'bool'=True, **kwargs: Any) ->ResNet:
    """ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


class Planning_Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.perception = resnet34(pretrained=False)
        self.perception.fc = nn.Sequential()
        self.join = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 256))
        self.decoder = nn.GRUCell(input_size=2, hidden_size=256)
        self.output = nn.Linear(256, 2)
        self.pred_len = 6

    def forward(self, img):
        feature_emb = self.perception(img)
        j = self.join(feature_emb)
        z = j
        output_wp = list()
        x = torch.zeros(size=(j.shape[0], 2)).type_as(j)
        for _ in range(self.pred_len):
            x_in = x
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)
        pred_wp = torch.stack(output_wp, dim=1)
        return pred_wp


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BackprojectDepth,
     lambda: ([], {'batch_size': 4, 'height': 4, 'width': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EPE,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PoseCNN,
     lambda: ([], {'num_input_frames': 4}),
     lambda: ([torch.rand([4, 12, 64, 64])], {}),
     False),
    (Project3D,
     lambda: ([], {'batch_size': 4, 'height': 4, 'width': 4}),
     lambda: ([torch.rand([4, 3, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Ternary,
     lambda: ([], {'device': 0}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_OpenDriveLab_PPGeo(_paritybench_base):
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

