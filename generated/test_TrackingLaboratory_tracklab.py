import sys
_module = sys.modules[__name__]
del sys
conf = _module
tracklab_searchpath_plugin = _module
main = _module
sn_calibration_baseline = _module
baseline_cameras = _module
camera = _module
dataloader = _module
detect_extremities = _module
evalai_camera = _module
evaluate_camera = _module
evaluate_extremities = _module
soccerpitch = _module
tv_main_behind = _module
tv_main_center = _module
tv_main_left = _module
tv_main_right = _module
tv_main_tribune = _module
cam_modules = _module
fuse_argmin = _module
fuse_stack = _module
inference = _module
module = _module
optimize = _module
sncalib_dataset = _module
data_distr = _module
io = _module
linalg = _module
objects_3d = _module
visualization_mpl = _module
visualization_mpl_min = _module
api_example = _module
posetrack21 = _module
api = _module
base_api = _module
posetrack_api = _module
posetrack_mot_api = _module
posetrack_pose_estim_api = _module
posetrackreid_api = _module
trackeval = _module
_timing = _module
datasets = _module
_base_dataset = _module
posetrack = _module
posetrack_mot = _module
posetrack_reid = _module
eval = _module
eval_mot = _module
eval_pose = _module
eval_reid = _module
metrics = _module
_base_metric = _module
count = _module
hota = _module
hota_pose = _module
hota_pose_reid = _module
map = _module
utils = _module
run_mot = _module
run_pose_estimation = _module
run_posetrack_challenge = _module
run_posetrack_reid_challenge = _module
setup = _module
posetrack21_mot = _module
pt_sequence = _module
pt_warper = _module
evaluate_mot = _module
motmetrics = _module
distances = _module
lap = _module
math_util = _module
mot = _module
preprocess = _module
tests = _module
test_distances = _module
test_io = _module
test_issue19 = _module
test_lap = _module
test_metrics = _module
test_mot = _module
test_utils = _module
bot_sort = _module
basetrack = _module
gmc = _module
kalman_filter = _module
matching = _module
bpbreid_strong_sort = _module
ecc = _module
sort = _module
detection = _module
iou_matching = _module
linear_assignment = _module
nn_matching = _module
oks_matching = _module
preprocessing = _module
track = _module
tracker = _module
strong_sort = _module
byte_track = _module
byte_tracker = _module
deep_oc_sort = _module
args = _module
association = _module
cmc = _module
embedding = _module
kalmanfilter = _module
ocsort = _module
reid_multibackend = _module
oc_sort = _module
deep = _module
models = _module
densenet = _module
hacnn = _module
inceptionresnetv2 = _module
inceptionv4 = _module
mlfn = _module
mobilenetv2 = _module
mudeep = _module
nasnet = _module
osnet = _module
osnet_ain = _module
pcb = _module
resnet = _module
resnet_ibn_a = _module
resnet_ibn_b = _module
resnetmid = _module
senet = _module
shufflenet = _module
shufflenetv2 = _module
squeezenet = _module
xception = _module
reid_model_factory = _module
reid_multibackend = _module
nn_matching = _module
strong_sort = _module
tracklab = _module
callbacks = _module
callback = _module
evaluate = _module
handle_regions = _module
progress = _module
timer = _module
configs = _module
core = _module
evaluator = _module
visualization_engine = _module
visualizer = _module
datastruct = _module
datapipe = _module
tracker_state = _module
tracking_dataset = _module
engine = _module
engine = _module
offline = _module
pipelined = _module
video = _module
loggers = _module
main = _module
pipeline = _module
datasetlevel_module = _module
detectionlevel_module = _module
imagelevel_module = _module
videolevel_module = _module
attribute_voting = _module
collate = _module
coordinates = _module
cv2 = _module
download = _module
easyocr = _module
instantiate = _module
monkeypatch_hydra = _module
notebook = _module
openmmlab = _module
wandb = _module
visualization = _module
tracking = _module
wrappers = _module
bbox_detector = _module
mmdetection_api = _module
external_video = _module
jrdb_pose = _module
jta = _module
mot_like = _module
common = _module
dancetrack = _module
mot17 = _module
mot20 = _module
sportsmot = _module
posetrack17 = _module
posetrack18 = _module
soccernet = _module
soccernet_game_state = _module
soccernet_mot = _module
detect_multiple = _module
bottomup_mmpose_api = _module
mmdetection_api = _module
openpifpaf_api = _module
yolov8_api = _module
yolov8_pose_api = _module
detect_single = _module
topdown_mmpose_api = _module
posetrack18_evaluator = _module
posetrack21_evaluator = _module
soccer_accuracy = _module
trackeval_evaluator = _module
reid = _module
bpbreid_api = _module
bpbreid_dataset = _module
bot_sort_api = _module
bpbreid_strong_sort_api = _module
byte_track_api = _module
deep_oc_sort_api = _module
oc_sort_api = _module
strong_sort_api = _module
tracklet_agg = _module
majority_vote_api = _module

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


import numpy as np


from torch.utils.data import Dataset


import copy


import random


from collections import deque


import torch


import torch.backends.cudnn


import torch.nn as nn


from torchvision.models.segmentation import deeplabv3_resnet50


from typing import Tuple


from typing import Dict


from typing import Union


import torchvision.transforms as T


from torchvision.models.segmentation import deeplabv3_resnet101


from functools import partial


from time import time


import matplotlib.pyplot as plt


import pandas as pd


import re


import torchvision


import collections


from typing import List


from typing import Optional


from abc import ABCMeta


from torchvision.transforms.functional import to_pil_image


from matplotlib.lines import Line2D


from matplotlib.patches import Polygon


from matplotlib.patches import Rectangle


from torch.nn import functional as F


from collections import OrderedDict


from itertools import islice


import torchvision.transforms as transforms


from collections import namedtuple


from torch.utils import model_zoo


from torch import nn


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


import warnings


import math


from typing import Any


from typing import TYPE_CHECKING


from torch.utils.data import DataLoader


import logging


from abc import ABC


from abc import abstractmethod


from scipy.optimize import linear_sum_assignment


from torchvision.ops import box_iou


from torch.utils.data.dataloader import default_collate


from torch.utils.data.dataloader import DataLoader


from math import ceil


from collections import defaultdict


from collections import Counter


class FeatureScalerZScore(torch.nn.Module):

    def __init__(self, loc: 'float', scale: 'float') ->None:
        super(FeatureScalerZScore, self).__init__()
        self.loc = loc
        self.scale = scale

    def forward(self, z):
        """
        Args:
            z (Tensor): tensor of size (B, *) to be denormalized.
        Returns:
            x: tensor.
        """
        return self.denormalize(z)

    def denormalize(self, z):
        x = z * self.scale + self.loc
        return x

    def normalize(self, x):
        z = (x - self.loc) / self.scale
        return z


class SNProjectiveCamera:

    def __init__(self, phi_dict: 'Dict[str, torch.tensor]', psi: 'torch.tensor', principal_point: 'Tuple[float, float]', image_width: 'int', image_height: 'int', device: 'str'='cpu', nan_check=True) ->None:
        """Projective camera defined as K @ R [I|-t] with lens distortion module and batch dimensions B,T.

        Following Euler angles convention, we use a ZXZ succession of intrinsic rotations in order to describe
        the orientation of the camera. Starting from the world reference axis system, we first apply a rotation
        around the Z axis to pan the camera. Then the obtained axis system is rotated around its x axis in order to tilt the camera.
        Then the last rotation around the z axis of the new axis system alows to roll the camera. Note that this z axis is the principal axis of the camera.

        As T is not provided for camra location and lens distortion, these parameters are assumed to be fixed accross T.
        phi_dict is a dict of parameters containing:
        {
            'aov_x, torch.Size([B, T])',
            'pan, torch.Size([B, T])',
            'tilt, torch.Size([B, T])',
            'roll, torch.Size([B, T])',
            'c_x, torch.Size([B, 1])',
            'c_y, torch.Size([B, 1])',
            'c_z, torch.Size([B, 1])',
        }

        Internally fuses B and T dimension to pseudo batch dimension.
        {
            'aov_x, torch.Size([B*T])',
            'pan, torch.Size([B*T])',
            'tilt, torch.Size([B*T])'
            'roll, torch.Size([B*T])',
            'c_x, torch.Size([B])',
            'c_y, torch.Size([B])',
            'c_z, torch.Size([B])',
            }

        aov_x, pan, tilt, roll are assumed in radian.

        Note on lens distortion:
            Lens distortion coefficients are independent from image resolution!
            We I(dist_points(K_ndc, dist_coeff, points2d_ndc)) == I(dist_points(K_raster, dist_coeff, points2d_raster))

        Args:
            phi_dict (Dict[str, torch.tensor]): See example above
            psi (Union[None, torch.Tensor]): distortion coefficients as concatinated vector according to https://kornia.readthedocs.io/en/latest/geometry.calibration.html of shape (B, T, {2, 4, 5,8,12, 14})
            principal_point (Tuple[float, float]): Principal point assumed to be fixed across all samples (B,T,)
            image_width (int): assumed to be fixed across all samples (B,T,)
            image_height (int): assumed to be fixed across all samples (B,T,)
        """
        phi_dict_flat = {}
        for k, v in phi_dict.items():
            if len(v.shape) == 2:
                phi_dict_flat[k] = v.view(v.shape[0] * v.shape[1])
            elif len(v.shape) == 3:
                phi_dict_flat[k] = v.view(v.shape[0] * v.shape[1], v.shape[-1])
        self.batch_dim, self.temporal_dim = phi_dict['pan'].shape
        self.pseudo_batch_size = phi_dict_flat['pan'].shape[0]
        self.phi_dict_flat = phi_dict_flat
        self.principal_point = principal_point
        self.image_width = image_width
        self.image_height = image_height
        self.device = device
        self.psi = psi
        if self.psi is not None:
            if self.psi.shape[-1] != 2:
                raise NotImplementedError
            if self.psi.shape[-1] == 2:
                psi_ext = torch.zeros(*list(self.psi.shape[:-1]), 4)
                psi_ext[..., :2] = self.psi
                self.psi = psi_ext
            self.lens_dist_coeff = self.psi.view(self.pseudo_batch_size, self.psi.shape[-1])
        self.intrinsics_ndc = self.construct_intrinsics_ndc()
        self.intrinsics_raster = self.construct_intrinsics_raster()
        self.rotation = self.rotation_from_euler_angles(*[phi_dict_flat[k] for k in ['pan', 'tilt', 'roll']])
        self.position = torch.stack([phi_dict_flat[k] for k in ['c_x', 'c_y', 'c_z']], dim=-1)
        self.position = self.position.repeat_interleave(int(self.pseudo_batch_size / self.batch_dim), dim=0)
        self.P_ndc = self.construct_projection_matrix(self.intrinsics_ndc)
        self.P_raster = self.construct_projection_matrix(self.intrinsics_raster)
        self.phi_dict = phi_dict
        self.nan_check = nan_check
        super().__init__()

    def construct_projection_matrix(self, intrinsics):
        It = torch.eye(4, device=self.device)[:-1].repeat(self.pseudo_batch_size, 1, 1)
        It[:, :, -1] = -self.position
        self.It = It
        return intrinsics @ self.rotation @ It

    def construct_intrinsics_ndc(self):
        K = torch.eye(3, requires_grad=False, device=self.device)
        K = K.reshape((1, 3, 3)).repeat(self.pseudo_batch_size, 1, 1)
        K[:, 0, 0] = self.get_fl_from_aov_rad(self.phi_dict_flat['aov'], d=2)
        K[:, 1, 1] = self.get_fl_from_aov_rad(self.phi_dict_flat['aov'], d=2 * self.image_width / self.image_height)
        return K

    def construct_intrinsics_raster(self):
        K = torch.eye(3, requires_grad=False, device=self.device)
        K = K.reshape((1, 3, 3)).repeat(self.pseudo_batch_size, 1, 1)
        K[:, 0, 0] = self.get_fl_from_aov_rad(self.phi_dict_flat['aov'], d=self.image_width)
        K[:, 1, 1] = self.get_fl_from_aov_rad(self.phi_dict_flat['aov'], d=self.image_width)
        K[:, 0, 2] = self.principal_point[0]
        K[:, 1, 2] = self.principal_point[1]
        return K

    def __str__(self) ->str:
        return f"aov_deg={torch.rad2deg(self.phi_dict['aov'])}, t={torch.stack([self.phi_dict[k] for k in ['c_x', 'c_y', 'c_z']], dim=-1)}, pan_deg={torch.rad2deg(self.phi_dict['pan'])} tilt_deg={torch.rad2deg(self.phi_dict['tilt'])} roll_deg={torch.rad2deg(self.phi_dict['roll'])}"

    def str_pan_tilt_roll_fl(self, b, t):
        r = f"FOV={torch.rad2deg(self.phi_dict['aov'][b, t]):.1f}°, pan={torch.rad2deg(self.phi_dict['pan'][b, t]):.1f}° tilt={torch.rad2deg(self.phi_dict['tilt'][b, t]):.1f}° roll={torch.rad2deg(self.phi_dict['roll'][b, t]):.1f}°"
        return r

    def str_lens_distortion_coeff(self, b):
        return f'lens dist coeff=' + ' '.join([f'{x:.2f}' for x in self.lens_dist_coeff[b, :2]])

    def __repr__(self) ->str:
        return f'{self.__class__}:' + self.__str__()

    def __len__(self):
        return self.pseudo_batch_size

    def project_point2pixel(self, points3d: 'torch.tensor', lens_distortion: 'bool') ->torch.tensor:
        """Project world coordinates to pixel coordinates.

        Args:
            points3d (torch.tensor): of shape (N, 3) or (1, N, 3)

        Returns:
            torch.tensor: projected points of shape (B, T, N, 2)
        """
        position = self.position.view(self.pseudo_batch_size, 1, 3)
        point = points3d - position
        rotated_point = self.rotation @ point.transpose(1, 2)
        dist_point2cam = rotated_point[:, 2]
        dist_point2cam = dist_point2cam.view(self.pseudo_batch_size, 1, rotated_point.shape[-1])
        rotated_point = rotated_point / dist_point2cam
        projected_points = self.intrinsics_raster @ rotated_point
        projected_points = projected_points.transpose(-1, -2)
        projected_points = kornia.geometry.convert_points_from_homogeneous(projected_points)
        if lens_distortion:
            if self.psi is None:
                raise RuntimeError('Lens distortion requested, but deactivated in module')
            projected_points = self.distort_points(projected_points, self.intrinsics_raster)
        projected_points = projected_points.view(self.batch_dim, self.temporal_dim, projected_points.shape[-2], 2)
        if self.nan_check:
            if torch.isnan(projected_points).any().item():
                None
                None
                raise RuntimeWarning('NaN in project_point2pixel')
        return projected_points

    def project_point2ndc(self, points3d: 'torch.tensor', lens_distortion: 'bool') ->torch.tensor:
        """Project world coordinates to pixel coordinates.

        Args:
            points3d (torch.tensor): of shape (N, 3) or (1, N, 3)

        Returns:
            torch.tensor: projected points of shape (B, T, N, 2)
        """
        position = self.position.view(self.pseudo_batch_size, 1, 3)
        point = points3d - position
        rotated_point = self.rotation @ point.transpose(1, 2)
        dist_point2cam = rotated_point[:, 2]
        dist_point2cam = dist_point2cam.view(self.pseudo_batch_size, 1, rotated_point.shape[-1])
        rotated_point = rotated_point / dist_point2cam
        projected_points = self.intrinsics_ndc @ rotated_point
        projected_points = projected_points.transpose(-1, -2)
        projected_points = kornia.geometry.convert_points_from_homogeneous(projected_points)
        if self.nan_check:
            if torch.isnan(projected_points).any().item():
                None
                None
                None
                raise RuntimeWarning('NaN in project_point2ndc before distort')
        if lens_distortion:
            if self.psi is None:
                raise RuntimeError('Lens distortion requested, but deactivated in module')
            projected_points = self.distort_points(projected_points, self.intrinsics_ndc)
        projected_points = projected_points.view(self.batch_dim, self.temporal_dim, projected_points.shape[-2], 2)
        if self.nan_check:
            if torch.isnan(projected_points).any().item():
                None
                None
                raise RuntimeWarning('NaN in project_point2ndc after distort')
        return projected_points

    def project_point2pixel_from_P(self, points3d: 'torch.tensor', lens_distortion: 'bool') ->torch.tensor:
        """Project world coordinates to pixel coordinates from the projection matrix.

        Args:
            points3d (torch.tensor): of shape (1, N, 3)

        Returns:
            torch.tensor: projected points of shape (B, T, N, 2)
        """
        points3d = kornia.geometry.conversions.convert_points_to_homogeneous(points3d).transpose(1, 2)
        projected_points = torch.bmm(self.P_raster, points3d.repeat(self.pseudo_batch_size, 1, 1))
        normalize_by = projected_points[:, -1].view(self.pseudo_batch_size, 1, projected_points.shape[-1])
        projected_points /= normalize_by
        projected_points = projected_points.transpose(-1, -2)
        projected_points = kornia.geometry.convert_points_from_homogeneous(projected_points)
        if lens_distortion:
            if self.psi is None:
                raise RuntimeError('Lens distortion requested, but deactivated in module')
            projected_points = self.distort_points(projected_points, self.intrinsics_raster)
        projected_points = projected_points.view(self.batch_dim, self.temporal_dim, projected_points.shape[-2], 2)
        return projected_points

    def project_point2ndc_from_P(self, points3d: 'torch.tensor', lens_distortion: 'bool') ->torch.tensor:
        """Project world coordinates to pixel coordinates from the projection matrix.

        Args:
            points3d (torch.tensor): of shape (1, N, 3)

        Returns:
            torch.tensor: projected points of shape (B, T, N, 2)
        """
        points3d = kornia.geometry.conversions.convert_points_to_homogeneous(points3d).transpose(1, 2)
        projected_points = torch.bmm(self.P_ndc, points3d.repeat(self.pseudo_batch_size, 1, 1))
        normalize_by = projected_points[:, -1].view(self.pseudo_batch_size, 1, projected_points.shape[-1])
        projected_points /= normalize_by
        projected_points = projected_points.transpose(-1, -2)
        projected_points = kornia.geometry.convert_points_from_homogeneous(projected_points)
        if lens_distortion:
            if self.psi is None:
                raise RuntimeError('Lens distortion requested, but deactivated in module')
            projected_points = self.distort_points(projected_points, self.intrinsics_ndc)
        projected_points = projected_points.view(self.batch_dim, self.temporal_dim, projected_points.shape[-2], 2)
        return projected_points

    def rotation_from_euler_angles(self, pan, tilt, roll):
        mask = torch.eye(3, requires_grad=False, device=self.device).reshape((1, 3, 3)).repeat(pan.shape[0], 1, 1)
        mask[:, 0, 0] = -torch.sin(pan) * torch.sin(roll) * torch.cos(tilt) + torch.cos(pan) * torch.cos(roll)
        mask[:, 0, 1] = torch.sin(pan) * torch.cos(roll) + torch.sin(roll) * torch.cos(pan) * torch.cos(tilt)
        mask[:, 0, 2] = torch.sin(roll) * torch.sin(tilt)
        mask[:, 1, 0] = -torch.sin(pan) * torch.cos(roll) * torch.cos(tilt) - torch.sin(roll) * torch.cos(pan)
        mask[:, 1, 1] = -torch.sin(pan) * torch.sin(roll) + torch.cos(pan) * torch.cos(roll) * torch.cos(tilt)
        mask[:, 1, 2] = torch.sin(tilt) * torch.cos(roll)
        mask[:, 2, 0] = torch.sin(pan) * torch.sin(tilt)
        mask[:, 2, 1] = -torch.sin(tilt) * torch.cos(pan)
        mask[:, 2, 2] = torch.cos(tilt)
        return mask

    def get_homography_raster(self):
        return self.P_raster[:, :, [0, 1, 3]].inverse()

    def get_rays_world(self, x):
        """_summary_

        Args:
            x (_type_): x of shape (B, 3, N)

        Returns:
            LineCollection: _description_
        """
        raise NotImplementedError

    @staticmethod
    def get_aov_rad(d: 'float', fl: 'torch.tensor'):
        return 2 * torch.arctan(d / (2 * fl))

    @staticmethod
    def get_fl_from_aov_rad(aov_rad: 'torch.tensor', d: 'float'):
        return 0.5 * d * (1 / torch.tan(0.5 * aov_rad))

    def undistort_points(self, points_pixel: 'torch.tensor', intrinsics, num_iters=5) ->torch.tensor:
        """Compensate for lens distortion a set of 2D image points.

        Wrapper for kornia.geometry.undistort_points()

        Args:
            points_pixel (torch.tensor): tensor of shape (B, N, 2)

        Returns:
            torch.tensor: undistorted points of shape (B, N, 2)
        """
        batch_dim, temporal_dim, N, _ = points_pixel.shape
        points_pixel = points_pixel.view(batch_dim * temporal_dim, N, 2)
        true_batch_size = batch_dim
        lens_dist_coeff = self.lens_dist_coeff
        if true_batch_size < self.batch_dim:
            intrinsics = intrinsics[:true_batch_size]
            lens_dist_coeff = lens_dist_coeff[:true_batch_size]
        return kornia.geometry.undistort_points(points_pixel, intrinsics, dist=lens_dist_coeff, num_iters=num_iters).view(batch_dim, temporal_dim, N, 2)

    def distort_points(self, points_pixel: 'torch.tensor', intrinsics) ->torch.tensor:
        """Distortion of a set of 2D points based on the lens distortion model.

        Wrapper for kornia.geometry.distort_points()

        Args:
            points_pixel (torch.tensor): tensor of shape (B, N, 2)

        Returns:
            torch.tensor: distorted points of shape (B, N, 2)
        """
        return kornia.geometry.distort_points(points_pixel, intrinsics, dist=self.lens_dist_coeff)

    def undistort_images(self, images):
        true_batch_size, T = images.shape[:2]
        images = images.view(true_batch_size * T, 3, self.image_height, self.image_width)
        intrinsics = self.intrinsics_raster
        lens_dist_coeff = self.lens_dist_coeff
        if true_batch_size < self.batch_dim:
            intrinsics = intrinsics[:true_batch_size]
            lens_dist_coeff = lens_dist_coeff[:true_batch_size]
        return kornia.geometry.calibration.undistort_image(images, intrinsics, lens_dist_coeff).view(true_batch_size, self.temporal_dim, 3, self.image_height, self.image_width)

    def get_parameters(self, true_batch_size=None):
        """
        Get dict of relevant camera parameters and homography matrix
        :return: The dictionary
        """
        out_dict = {'pan_degrees': torch.rad2deg(self.phi_dict['pan']), 'tilt_degrees': torch.rad2deg(self.phi_dict['tilt']), 'roll_degrees': torch.rad2deg(self.phi_dict['roll']), 'position_meters': torch.stack([self.phi_dict[k] for k in ['c_x', 'c_y', 'c_z']], dim=1).squeeze(-1).unsqueeze(-2).repeat(1, self.temporal_dim, 1), 'aov_radian': self.phi_dict['aov'], 'aov_degrees': torch.rad2deg(self.phi_dict['aov']), 'x_focal_length': self.get_fl_from_aov_rad(self.phi_dict['aov'], d=self.image_width), 'y_focal_length': self.get_fl_from_aov_rad(self.phi_dict['aov'], d=self.image_width), 'principal_point': torch.tensor([[self.principal_point] * self.temporal_dim] * self.batch_dim)}
        out_dict['homography'] = self.get_homography_raster().unsqueeze(1)
        out_dict['radial_distortion'] = torch.zeros(self.batch_dim, self.temporal_dim, 6)
        out_dict['tangential_distortion'] = torch.zeros(self.batch_dim, self.temporal_dim, 2)
        out_dict['thin_prism_distortion'] = torch.zeros(self.batch_dim, self.temporal_dim, 4)
        if self.psi is not None:
            out_dict['radial_distortion'][..., :2] = self.psi[..., :2]
        if true_batch_size is None or true_batch_size == self.batch_dim:
            return out_dict
        for k in out_dict.keys():
            out_dict[k] = out_dict[k][:true_batch_size]
        return out_dict

    @staticmethod
    def static_undistort_points(points, cam):
        intrinsics = cam.intrinsics_raster
        lens_dist_coeff = cam.lens_dist_coeff
        true_batch_size = points.shape[0]
        if true_batch_size < cam.batch_dim:
            intrinsics = intrinsics[:true_batch_size]
            lens_dist_coeff = lens_dist_coeff[:true_batch_size]
        batch_size, T, _, S, N = points.shape
        points = points.view(batch_size, T, 3, S * N).transpose(2, 3)
        points[..., :2] = kornia.geometry.undistort_points(points[..., :2].view(batch_size * T, S * N, 2), intrinsics, dist=lens_dist_coeff, num_iters=1).view(batch_size, T, S * N, 2)
        points = points.transpose(2, 3).view(batch_size, T, 3, S, N)
        return points


def distance_line_pointcloud_3d(e1: 'torch.Tensor', r1: 'torch.Tensor', pc: 'torch.Tensor', reduce: 'Union[None, str]'=None) ->torch.Tensor:
    """
    Line to point cloud distance with arbitrary leading dimensions.

    TODO. if cross = (0.0.0) -> distance=0 otherwise NaNs are returned

    https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
    Args:
        e1 (torch.Tensor): direction vector of shape (*, B, 1, 3)
        r1 (torch.Tensor): support vector of shape (*, B, 1, 3)
        pc (torch.Tensor): point cloud of shape (*, B, A, 3)
        reduce (Union[None, str]): reduce distance for all points to one using 'mean' or 'min'
    Returns:
        distance of an infinite line to given points, (*, B, ) using reduce='mean' or reduce='min' or (*, B, A) if reduce=False
    """
    num_points = pc.shape[-2]
    _sub = r1 - pc
    cross = torch.cross(e1.repeat_interleave(num_points, dim=-2), _sub, dim=-1)
    e1_norm = torch.linalg.norm(e1, dim=-1)
    cross_norm = torch.linalg.norm(cross, dim=-1)
    d = cross_norm / e1_norm
    if reduce == 'mean':
        return d.mean(dim=-1)
    elif reduce == 'min':
        return d.min(dim=-1)[0]
    return d


def distance_point_pointcloud(points: 'torch.Tensor', pointcloud: 'torch.Tensor') ->torch.Tensor:
    """Batched version for point-pointcloud distance calculation
    Args:
        points (torch.Tensor): N points in homogenous coordinates; shape (B, T, 3, S, N)
        pointcloud (torch.Tensor): N_star points for each pointcloud; shape (B, T, S, N_star, 2)

    Returns:
        torch.Tensor: Minimum distance for each point N to pointcloud; shape (B, T, 1, S, N)
    """
    batch_size, T, _, S, N = points.shape
    batch_size, T, S, N_star, _ = pointcloud.shape
    pointcloud = pointcloud.reshape(batch_size * T * S, N_star, 2)
    points = convert_points_from_homogeneous(points.permute(0, 1, 3, 4, 2).reshape(batch_size * T * S, N, 3))
    distances = torch.cdist(points, pointcloud, p=2)
    distances = distances.view(batch_size, T, S, N, N_star)
    distances = distances.unsqueeze(-4)
    distances = distances.min(dim=-1)[0]
    return distances


class TVCalibModule(torch.nn.Module):

    def __init__(self, model3d, cam_distr, dist_distr, image_dim: 'Tuple[int, int]', optim_steps: 'int', device='cpu', tqdm_kwqargs=None, log_per_step=False, *args, **kwargs) ->None:
        super().__init__(*args, **kwargs)
        self.image_height, self.image_width = image_dim
        self.principal_point = self.image_width / 2, self.image_height / 2
        self.model3d = model3d
        self.cam_param_dict = CameraParameterWLensDistDictZScore(cam_distr, dist_distr, device=device)
        self.lens_distortion_active = False if dist_distr is None else True
        self.optim_steps = optim_steps
        self._device = device
        self.optim = torch.optim.AdamW(self.cam_param_dict.param_dict.parameters(), lr=0.1, weight_decay=0.01)
        self.Scheduler = partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=0.05, total_steps=self.optim_steps, pct_start=0.5)
        if self.lens_distortion_active:
            self.optim_lens_distortion = torch.optim.AdamW(self.cam_param_dict.param_dict_dist.parameters(), lr=0.001, weight_decay=0.01)
            self.Scheduler_lens_distortion = partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=0.001, total_steps=self.optim_steps, pct_start=0.33, optimizer=self.optim_lens_distortion)
        self.tqdm_kwqargs = tqdm_kwqargs
        if tqdm_kwqargs is None:
            self.tqdm_kwqargs = {}
        self.hparams = {'optim': str(self.optim), 'scheduler': str(self.Scheduler)}
        self.log_per_step = log_per_step

    def forward(self, x):
        phi_hat, psi_hat = self.cam_param_dict()
        cam = SNProjectiveCamera(phi_hat, psi_hat, self.principal_point, self.image_width, self.image_height, device=self._device, nan_check=False)
        points_px_lines_true = x['lines__ndc_projected_selection_shuffled']
        batch_size, T_l, _, S_l, N_l = points_px_lines_true.shape
        points_px_circles_true = x['circles__ndc_projected_selection_shuffled']
        _, T_c, _, S_c, N_c = points_px_circles_true.shape
        assert T_c == T_l
        points3d_lines_keypoints = self.model3d.line_segments
        points3d_lines_keypoints = points3d_lines_keypoints.reshape(3, S_l * 2).transpose(0, 1)
        points_px_lines_keypoints = convert_points_to_homogeneous(cam.project_point2ndc(points3d_lines_keypoints, lens_distortion=False))
        if batch_size < cam.batch_dim:
            points_px_lines_keypoints = points_px_lines_keypoints[:batch_size]
        points_px_lines_keypoints = points_px_lines_keypoints.view(batch_size, T_l, S_l, 2, 3)
        lp1 = points_px_lines_keypoints[..., 0, :].unsqueeze(-2)
        lp2 = points_px_lines_keypoints[..., 1, :].unsqueeze(-2)
        pc = points_px_lines_true.view(batch_size, T_l, 3, S_l * N_l).transpose(2, 3).view(batch_size, T_l, S_l, N_l, 3)
        if self.lens_distortion_active:
            pc = pc.view(batch_size, T_l, S_l * N_l, 3)
            pc = pc.detach().clone()
            pc[..., :2] = cam.undistort_points(pc[..., :2], cam.intrinsics_ndc, num_iters=1)
            pc = pc.view(batch_size, T_l, S_l, N_l, 3)
        distances_px_lines_raw = distance_line_pointcloud_3d(e1=lp2 - lp1, r1=lp1, pc=pc, reduce=None)
        distances_px_lines_raw = distances_px_lines_raw.unsqueeze(-3)
        points3d_circles_pc = self.model3d.circle_segments
        _, S_c, N_c_star = points3d_circles_pc.shape
        points3d_circles_pc = points3d_circles_pc.reshape(3, S_c * N_c_star).transpose(0, 1)
        points_px_circles_pc = cam.project_point2ndc(points3d_circles_pc, lens_distortion=False)
        if batch_size < cam.batch_dim:
            points_px_circles_pc = points_px_circles_pc[:batch_size]
        if self.lens_distortion_active:
            points_px_circles_true = points_px_circles_true.view(batch_size, T_c, 3, S_c * N_c).transpose(2, 3)
            points_px_circles_true = points_px_circles_true.detach().clone()
            points_px_circles_true[..., :2] = cam.undistort_points(points_px_circles_true[..., :2], cam.intrinsics_ndc, num_iters=1)
            points_px_circles_true = points_px_circles_true.transpose(2, 3).view(batch_size, T_c, 3, S_c, N_c)
        distances_px_circles_raw = distance_point_pointcloud(points_px_circles_true, points_px_circles_pc.view(batch_size, T_c, S_c, N_c_star, 2))
        distances_dict = {'loss_ndc_lines': distances_px_lines_raw, 'loss_ndc_circles': distances_px_circles_raw}
        return distances_dict, cam

    def self_optim_batch(self, x, *args, **kwargs):
        scheduler = self.Scheduler(self.optim)
        if self.lens_distortion_active:
            scheduler_lens_distortion = self.Scheduler_lens_distortion()
        self.cam_param_dict.initialize(None)
        self.optim.zero_grad()
        if self.lens_distortion_active:
            self.optim_lens_distortion.zero_grad()
        keypoint_masks = {'loss_ndc_lines': x['lines__is_keypoint_mask'], 'loss_ndc_circles': x['circles__is_keypoint_mask']}
        num_actual_points = {'loss_ndc_circles': keypoint_masks['loss_ndc_circles'].sum(dim=(-1, -2)), 'loss_ndc_lines': keypoint_masks['loss_ndc_lines'].sum(dim=(-1, -2))}
        per_sample_loss = {}
        per_sample_loss['mask_lines'] = keypoint_masks['loss_ndc_lines']
        per_sample_loss['mask_circles'] = keypoint_masks['loss_ndc_circles']
        per_step_info = {'loss': [], 'lr': []}
        with tqdm(range(self.optim_steps), **self.tqdm_kwqargs) as pbar:
            for step in pbar:
                self.optim.zero_grad()
                if self.lens_distortion_active:
                    self.optim_lens_distortion.zero_grad()
                distances_dict, cam = self(x)
                losses = {}
                for key_dist, distances in distances_dict.items():
                    distances[~keypoint_masks[key_dist]] = 0.0
                    per_sample_loss[f'{key_dist}_distances_raw'] = distances
                    distances_reduced = distances.sum(dim=(-1, -2))
                    distances_reduced = distances_reduced / num_actual_points[key_dist]
                    distances_reduced[num_actual_points[key_dist] == 0] = 0.0
                    distances_reduced = distances_reduced.squeeze(-1)
                    per_sample_loss[key_dist] = distances_reduced
                    loss = distances_reduced.mean(dim=-1)
                    loss = loss.sum()
                    losses[key_dist] = loss
                loss_total_dist = losses['loss_ndc_lines'] + losses['loss_ndc_circles']
                loss_total = loss_total_dist
                if self.log_per_step:
                    per_step_info['lr'].append(scheduler.get_last_lr())
                    per_step_info['loss'].append(distances_reduced)
                if step % 50 == 0:
                    pbar.set_postfix(loss=f'{loss_total_dist.detach().cpu().tolist():.5f}', loss_lines=f"{losses['loss_ndc_lines'].detach().cpu().tolist():.3f}", loss_circles=f"{losses['loss_ndc_circles'].detach().cpu().tolist():.3f}")
                loss_total.backward()
                self.optim.step()
                scheduler.step()
                if self.lens_distortion_active:
                    self.optim_lens_distortion.step()
                    scheduler_lens_distortion.step()
        per_sample_loss['loss_ndc_total'] = torch.sum(torch.stack([per_sample_loss[key_dist] for key_dist in distances_dict.keys()], dim=0), dim=0)
        if self.log_per_step:
            per_step_info['loss'] = torch.stack(per_step_info['loss'], dim=-1)
            per_step_info['lr'] = torch.tensor(per_step_info['lr'])
        return per_sample_loss, cam, per_step_info


class ConvBlock(nn.Module):
    """Basic convolutional block.
    
    convolution + batch normalization + relu.

    Args:
        in_c (int): number of input channels.
        out_c (int): number of output channels.
        k (int or tuple): kernel size.
        s (int or tuple): stride.
        p (int or tuple): padding.
    """

    def __init__(self, in_c, out_c, k, s, p):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([0, -0.75, 0, -0.25, 0, 0.25, 0, 0.75], dtype=torch.float))

    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta


class ChannelAttn(nn.Module):
    """Channel Attention (Sec. 3.1.I.2)"""

    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels % reduction_rate == 0
        self.conv1 = ConvBlock(in_channels, in_channels // reduction_rate, 1)
        self.conv2 = ConvBlock(in_channels // reduction_rate, in_channels, 1)

    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[2:])
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""

    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.conv1 = ConvBlock(1, 1, 3, s=2, p=1)
        self.conv2 = ConvBlock(1, 1, 1)

    def forward(self, x):
        x = x.mean(1, keepdim=True)
        x = self.conv1(x)
        x = F.upsample(x, (x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv2(x)
        return x


class SoftAttn(nn.Module):
    """Soft Attention (Sec. 3.1.I)
    
    Aim: Spatial Attention + Channel Attention
    
    Output: attention maps with shape identical to input.
    """

    def __init__(self, in_channels):
        super(SoftAttn, self).__init__()
        self.spatial_attn = SpatialAttn()
        self.channel_attn = ChannelAttn(in_channels)
        self.conv = ConvBlock(in_channels, in_channels, 1)

    def forward(self, x):
        y_spatial = self.spatial_attn(x)
        y_channel = self.channel_attn(x)
        y = y_spatial * y_channel
        y = torch.sigmoid(self.conv(y))
        return y


class HarmAttn(nn.Module):
    """Harmonious Attention (Sec. 3.1)"""

    def __init__(self, in_channels):
        super(HarmAttn, self).__init__()
        self.soft_attn = SoftAttn(in_channels)
        self.hard_attn = HardAttn(in_channels)

    def forward(self, x):
        y_soft_attn = self.soft_attn(x)
        theta = self.hard_attn(x)
        return y_soft_attn, theta


class InceptionA(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionA, self).__init__()
        mid_channels = out_channels // 4
        self.stream1 = nn.Sequential(ConvBlock(in_channels, mid_channels, 1), ConvBlock(mid_channels, mid_channels, 3, p=1))
        self.stream2 = nn.Sequential(ConvBlock(in_channels, mid_channels, 1), ConvBlock(mid_channels, mid_channels, 3, p=1))
        self.stream3 = nn.Sequential(ConvBlock(in_channels, mid_channels, 1), ConvBlock(mid_channels, mid_channels, 3, p=1))
        self.stream4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), ConvBlock(in_channels, mid_channels, 1))

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        y = torch.cat([s1, s2, s3, s4], dim=1)
        return y


class InceptionB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(InceptionB, self).__init__()
        mid_channels = out_channels // 4
        self.stream1 = nn.Sequential(ConvBlock(in_channels, mid_channels, 1), ConvBlock(mid_channels, mid_channels, 3, s=2, p=1))
        self.stream2 = nn.Sequential(ConvBlock(in_channels, mid_channels, 1), ConvBlock(mid_channels, mid_channels, 3, p=1), ConvBlock(mid_channels, mid_channels, 3, s=2, p=1))
        self.stream3 = nn.Sequential(nn.MaxPool2d(3, stride=2, padding=1), ConvBlock(in_channels, mid_channels * 2, 1))

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        y = torch.cat([s1, s2, s3], dim=1)
        return y


class HACNN(nn.Module):
    """Harmonious Attention Convolutional Neural Network.

    Reference:
        Li et al. Harmonious Attention Network for Person Re-identification. CVPR 2018.

    Public keys:
        - ``hacnn``: HACNN.
    """

    def __init__(self, num_classes, loss='softmax', nchannels=[128, 256, 384], feat_dim=512, learn_region=True, use_gpu=True, **kwargs):
        super(HACNN, self).__init__()
        self.loss = loss
        self.learn_region = learn_region
        self.use_gpu = use_gpu
        self.conv = ConvBlock(3, 32, 3, s=2, p=1)
        self.inception1 = nn.Sequential(InceptionA(32, nchannels[0]), InceptionB(nchannels[0], nchannels[0]))
        self.ha1 = HarmAttn(nchannels[0])
        self.inception2 = nn.Sequential(InceptionA(nchannels[0], nchannels[1]), InceptionB(nchannels[1], nchannels[1]))
        self.ha2 = HarmAttn(nchannels[1])
        self.inception3 = nn.Sequential(InceptionA(nchannels[1], nchannels[2]), InceptionB(nchannels[2], nchannels[2]))
        self.ha3 = HarmAttn(nchannels[2])
        self.fc_global = nn.Sequential(nn.Linear(nchannels[2], feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU())
        self.classifier_global = nn.Linear(feat_dim, num_classes)
        if self.learn_region:
            self.init_scale_factors()
            self.local_conv1 = InceptionB(32, nchannels[0])
            self.local_conv2 = InceptionB(nchannels[0], nchannels[1])
            self.local_conv3 = InceptionB(nchannels[1], nchannels[2])
            self.fc_local = nn.Sequential(nn.Linear(nchannels[2] * 4, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU())
            self.classifier_local = nn.Linear(feat_dim, num_classes)
            self.feat_dim = feat_dim * 2
        else:
            self.feat_dim = feat_dim

    def init_scale_factors(self):
        self.scale_factors = []
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))
        self.scale_factors.append(torch.tensor([[1, 0], [0, 0.25]], dtype=torch.float))

    def stn(self, x, theta):
        """Performs spatial transform
        
        x: (batch, channel, height, width)
        theta: (batch, 2, 3)
        """
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def transform_theta(self, theta_i, region_idx):
        """Transforms theta to include (s_w, s_h), resulting in (batch, 2, 3)"""
        scale_factors = self.scale_factors[region_idx]
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:, :, :2] = scale_factors
        theta[:, :, -1] = theta_i
        if self.use_gpu:
            theta = theta
        return theta

    def forward(self, x):
        assert x.size(2) == 160 and x.size(3) == 64, 'Input size does not match, expected (160, 64) but got ({}, {})'.format(x.size(2), x.size(3))
        x = self.conv(x)
        x1 = self.inception1(x)
        x1_attn, x1_theta = self.ha1(x1)
        x1_out = x1 * x1_attn
        if self.learn_region:
            x1_local_list = []
            for region_idx in range(4):
                x1_theta_i = x1_theta[:, region_idx, :]
                x1_theta_i = self.transform_theta(x1_theta_i, region_idx)
                x1_trans_i = self.stn(x, x1_theta_i)
                x1_trans_i = F.upsample(x1_trans_i, (24, 28), mode='bilinear', align_corners=True)
                x1_local_i = self.local_conv1(x1_trans_i)
                x1_local_list.append(x1_local_i)
        x2 = self.inception2(x1_out)
        x2_attn, x2_theta = self.ha2(x2)
        x2_out = x2 * x2_attn
        if self.learn_region:
            x2_local_list = []
            for region_idx in range(4):
                x2_theta_i = x2_theta[:, region_idx, :]
                x2_theta_i = self.transform_theta(x2_theta_i, region_idx)
                x2_trans_i = self.stn(x1_out, x2_theta_i)
                x2_trans_i = F.upsample(x2_trans_i, (12, 14), mode='bilinear', align_corners=True)
                x2_local_i = x2_trans_i + x1_local_list[region_idx]
                x2_local_i = self.local_conv2(x2_local_i)
                x2_local_list.append(x2_local_i)
        x3 = self.inception3(x2_out)
        x3_attn, x3_theta = self.ha3(x3)
        x3_out = x3 * x3_attn
        if self.learn_region:
            x3_local_list = []
            for region_idx in range(4):
                x3_theta_i = x3_theta[:, region_idx, :]
                x3_theta_i = self.transform_theta(x3_theta_i, region_idx)
                x3_trans_i = self.stn(x2_out, x3_theta_i)
                x3_trans_i = F.upsample(x3_trans_i, (6, 7), mode='bilinear', align_corners=True)
                x3_local_i = x3_trans_i + x2_local_list[region_idx]
                x3_local_i = self.local_conv3(x3_local_i)
                x3_local_list.append(x3_local_i)
        x_global = F.avg_pool2d(x3_out, x3_out.size()[2:]).view(x3_out.size(0), x3_out.size(1))
        x_global = self.fc_global(x_global)
        if self.learn_region:
            x_local_list = []
            for region_idx in range(4):
                x_local_i = x3_local_list[region_idx]
                x_local_i = F.avg_pool2d(x_local_i, x_local_i.size()[2:]).view(x_local_i.size(0), -1)
                x_local_list.append(x_local_i)
            x_local = torch.cat(x_local_list, 1)
            x_local = self.fc_local(x_local)
        if not self.training:
            if self.learn_region:
                x_global = x_global / x_global.norm(p=2, dim=1, keepdim=True)
                x_local = x_local / x_local.norm(p=2, dim=1, keepdim=True)
                return torch.cat([x_global, x_local], 1)
            else:
                return x_global
        prelogits_global = self.classifier_global(x_global)
        if self.learn_region:
            prelogits_local = self.classifier_local(x_local)
        if self.loss == 'softmax':
            if self.learn_region:
                return prelogits_global, prelogits_local
            else:
                return prelogits_global
        elif self.loss == 'triplet':
            if self.learn_region:
                return (prelogits_global, prelogits_local), (x_global, x_local)
            else:
                return prelogits_global, x_global
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class ConvLayers(nn.Module):
    """Preprocessing layers."""

    def __init__(self):
        super(ConvLayers, self).__init__()
        self.conv1 = ConvBlock(3, 48, k=3, s=1, p=1)
        self.conv2 = ConvBlock(48, 96, k=3, s=1, p=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        return x


class Fusion(nn.Module):
    """Saliency-based learning fusion layer (Sec.3.2)"""

    def __init__(self):
        super(Fusion, self).__init__()
        self.a1 = nn.Parameter(torch.rand(1, 256, 1, 1))
        self.a2 = nn.Parameter(torch.rand(1, 256, 1, 1))
        self.a3 = nn.Parameter(torch.rand(1, 256, 1, 1))
        self.a4 = nn.Parameter(torch.rand(1, 256, 1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

    def forward(self, x1, x2, x3, x4):
        s1 = self.a1.expand_as(x1) * x1
        s2 = self.a2.expand_as(x2) * x2
        s3 = self.a3.expand_as(x3) * x3
        s4 = self.a4.expand_as(x4) * x4
        y = self.avgpool(s1 + s2 + s3 + s4)
        return y


class MultiScaleA(nn.Module):
    """Multi-scale stream layer A (Sec.3.1)"""

    def __init__(self):
        super(MultiScaleA, self).__init__()
        self.stream1 = nn.Sequential(ConvBlock(96, 96, k=1, s=1, p=0), ConvBlock(96, 24, k=3, s=1, p=1))
        self.stream2 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), ConvBlock(96, 24, k=1, s=1, p=0))
        self.stream3 = ConvBlock(96, 24, k=1, s=1, p=0)
        self.stream4 = nn.Sequential(ConvBlock(96, 16, k=1, s=1, p=0), ConvBlock(16, 24, k=3, s=1, p=1), ConvBlock(24, 24, k=3, s=1, p=1))

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        y = torch.cat([s1, s2, s3, s4], dim=1)
        return y


class MultiScaleB(nn.Module):
    """Multi-scale stream layer B (Sec.3.1)"""

    def __init__(self):
        super(MultiScaleB, self).__init__()
        self.stream1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), ConvBlock(256, 256, k=1, s=1, p=0))
        self.stream2 = nn.Sequential(ConvBlock(256, 64, k=1, s=1, p=0), ConvBlock(64, 128, k=(1, 3), s=1, p=(0, 1)), ConvBlock(128, 256, k=(3, 1), s=1, p=(1, 0)))
        self.stream3 = ConvBlock(256, 256, k=1, s=1, p=0)
        self.stream4 = nn.Sequential(ConvBlock(256, 64, k=1, s=1, p=0), ConvBlock(64, 64, k=(1, 3), s=1, p=(0, 1)), ConvBlock(64, 128, k=(3, 1), s=1, p=(1, 0)), ConvBlock(128, 128, k=(1, 3), s=1, p=(0, 1)), ConvBlock(128, 256, k=(3, 1), s=1, p=(1, 0)))

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        s4 = self.stream4(x)
        return s1, s2, s3, s4


class Reduction(nn.Module):
    """Reduction layer (Sec.3.1)"""

    def __init__(self):
        super(Reduction, self).__init__()
        self.stream1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stream2 = ConvBlock(96, 96, k=3, s=2, p=1)
        self.stream3 = nn.Sequential(ConvBlock(96, 48, k=1, s=1, p=0), ConvBlock(48, 56, k=3, s=1, p=1), ConvBlock(56, 64, k=3, s=2, p=1))

    def forward(self, x):
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        y = torch.cat([s1, s2, s3], dim=1)
        return y


class MuDeep(nn.Module):
    """Multiscale deep neural network.

    Reference:
        Qian et al. Multi-scale Deep Learning Architectures
        for Person Re-identification. ICCV 2017.

    Public keys:
        - ``mudeep``: Multiscale deep neural network.
    """

    def __init__(self, num_classes, loss='softmax', **kwargs):
        super(MuDeep, self).__init__()
        self.loss = loss
        self.block1 = ConvLayers()
        self.block2 = MultiScaleA()
        self.block3 = Reduction()
        self.block4 = MultiScaleB()
        self.block5 = Fusion()
        self.fc = nn.Sequential(nn.Linear(256 * 16 * 8, 4096), nn.BatchNorm1d(4096), nn.ReLU())
        self.classifier = nn.Linear(4096, num_classes)
        self.feat_dim = 4096

    def featuremaps(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(*x)
        return x

    def forward(self, x):
        x = self.featuremaps(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        y = self.classifier(x)
        if not self.training:
            return x
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, x
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densely connected network.
    
    Reference:
        Huang et al. Densely Connected Convolutional Networks. CVPR 2017.

    Public keys:
        - ``densenet121``: DenseNet121.
        - ``densenet169``: DenseNet169.
        - ``densenet201``: DenseNet201.
        - ``densenet161``: DenseNet161.
        - ``densenet121_fc512``: DenseNet121 + FC.
    """

    def __init__(self, num_classes, loss, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, fc_dims=None, dropout_p=None, **kwargs):
        super(DenseNet, self).__init__()
        self.loss = loss
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)), ('norm0', nn.BatchNorm2d(num_init_features)), ('relu0', nn.ReLU(inplace=True)), ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = num_features
        self.fc = self._construct_fc_layer(fc_dims, num_features, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer.

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initialize models with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


model_urls = {'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth', 'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth'}


def densenet121(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(num_classes=num_classes, loss=loss, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])
    return model


def densenet121_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(num_classes=num_classes, loss=loss, num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), fc_dims=[512], dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])
    return model


def densenet161(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(num_classes=num_classes, loss=loss, num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet161'])
    return model


def densenet169(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(num_classes=num_classes, loss=loss, num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet169'])
    return model


def densenet201(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(num_classes=num_classes, loss=loss, num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet201'])
    return model


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1088, 128, kernel_size=1, stride=1), BasicConv2d(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(320, 32, kernel_size=1, stride=1), BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1), BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(2080, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)), BasicConv2d(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(192, 48, kernel_size=1, stride=1), BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2))
        self.branch2 = nn.Sequential(BasicConv2d(192, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(192, 64, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(320, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(1088, 256, kernel_size=1, stride=1), BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1), BasicConv2d(288, 320, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


pretrained_settings = {'xception': {'imagenet': {'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth', 'input_space': 'RGB', 'input_size': [3, 299, 299], 'input_range': [0, 1], 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5], 'num_classes': 1000, 'scale': 0.8975}}}


class InceptionResNetV2(nn.Module):
    """Inception-ResNet-V2.

    Reference:
        Szegedy et al. Inception-v4, Inception-ResNet and the Impact of Residual
        Connections on Learning. AAAI 2017.

    Public keys:
        - ``inceptionresnetv2``: Inception-ResNet-V2.
    """

    def __init__(self, num_classes, loss='softmax', **kwargs):
        super(InceptionResNetV2, self).__init__()
        self.loss = loss
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2))
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1536, num_classes)

    def load_imagenet_weights(self):
        settings = pretrained_settings['inceptionresnetv2']['imagenet']
        pretrain_dict = model_zoo.load_url(settings['url'])
        model_dict = self.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)

    def featuremaps(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def inceptionresnetv2(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = InceptionResNetV2(num_classes=num_classes, loss=loss, **kwargs)
    if pretrained:
        model.load_imagenet_weights()
    return model


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(384, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1), BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(384, 96, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 256, kernel_size=(7, 1), stride=1, padding=(3, 0)))
        self.branch2 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(192, 224, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(224, 224, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(224, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1024, 128, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()
        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.branch3 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False), BasicConv2d(1536, 256, kernel_size=1, stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)
        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 96, kernel_size=3, stride=1))
        self.branch1 = nn.Sequential(BasicConv2d(160, 64, kernel_size=1, stride=1), BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(64, 96, kernel_size=(3, 3), stride=1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(384, 192, kernel_size=1, stride=1), BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1), BasicConv2d(224, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(1024, 192, kernel_size=1, stride=1), BasicConv2d(192, 192, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(1024, 256, kernel_size=1, stride=1), BasicConv2d(256, 256, kernel_size=(1, 7), stride=1, padding=(0, 3)), BasicConv2d(256, 320, kernel_size=(7, 1), stride=1, padding=(3, 0)), BasicConv2d(320, 320, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class InceptionV4(nn.Module):
    """Inception-v4.

    Reference:
        Szegedy et al. Inception-v4, Inception-ResNet and the Impact of Residual
        Connections on Learning. AAAI 2017.

    Public keys:
        - ``inceptionv4``: InceptionV4.
    """

    def __init__(self, num_classes, loss, **kwargs):
        super(InceptionV4, self).__init__()
        self.loss = loss
        self.features = nn.Sequential(BasicConv2d(3, 32, kernel_size=3, stride=2), BasicConv2d(32, 32, kernel_size=3, stride=1), BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1), Mixed_3a(), Mixed_4a(), Mixed_5a(), Inception_A(), Inception_A(), Inception_A(), Inception_A(), Reduction_A(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Inception_B(), Reduction_B(), Inception_C(), Inception_C(), Inception_C())
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1536, num_classes)

    def forward(self, x):
        f = self.features(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def inceptionv4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = InceptionV4(num_classes, loss, **kwargs)
    if pretrained:
        model_url = pretrained_settings['inceptionv4']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


class MLFNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, fsm_channels, groups=32):
        super(MLFNBlock, self).__init__()
        self.groups = groups
        mid_channels = out_channels // 2
        self.fm_conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.fm_bn1 = nn.BatchNorm2d(mid_channels)
        self.fm_conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False, groups=self.groups)
        self.fm_bn2 = nn.BatchNorm2d(mid_channels)
        self.fm_conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.fm_bn3 = nn.BatchNorm2d(out_channels)
        self.fsm = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, fsm_channels[0], 1), nn.BatchNorm2d(fsm_channels[0]), nn.ReLU(inplace=True), nn.Conv2d(fsm_channels[0], fsm_channels[1], 1), nn.BatchNorm2d(fsm_channels[1]), nn.ReLU(inplace=True), nn.Conv2d(fsm_channels[1], self.groups, 1), nn.BatchNorm2d(self.groups), nn.Sigmoid())
        self.downsample = None
        if in_channels != out_channels or stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x
        s = self.fsm(x)
        x = self.fm_conv1(x)
        x = self.fm_bn1(x)
        x = F.relu(x, inplace=True)
        x = self.fm_conv2(x)
        x = self.fm_bn2(x)
        x = F.relu(x, inplace=True)
        b, c = x.size(0), x.size(1)
        n = c // self.groups
        ss = s.repeat(1, n, 1, 1)
        ss = ss.view(b, n, self.groups, 1, 1)
        ss = ss.permute(0, 2, 1, 3, 4).contiguous()
        ss = ss.view(b, c, 1, 1)
        x = ss * x
        x = self.fm_conv3(x)
        x = self.fm_bn3(x)
        x = F.relu(x, inplace=True)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return F.relu(residual + x, inplace=True), s


class MLFN(nn.Module):
    """Multi-Level Factorisation Net.

    Reference:
        Chang et al. Multi-Level Factorisation Net for
        Person Re-Identification. CVPR 2018.

    Public keys:
        - ``mlfn``: MLFN (Multi-Level Factorisation Net).
    """

    def __init__(self, num_classes, loss='softmax', groups=32, channels=[64, 256, 512, 1024, 2048], embed_dim=1024, **kwargs):
        super(MLFN, self).__init__()
        self.loss = loss
        self.groups = groups
        self.conv1 = nn.Conv2d(3, channels[0], 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.feature = nn.ModuleList([MLFNBlock(channels[0], channels[1], 1, [128, 64], self.groups), MLFNBlock(channels[1], channels[1], 1, [128, 64], self.groups), MLFNBlock(channels[1], channels[1], 1, [128, 64], self.groups), MLFNBlock(channels[1], channels[2], 2, [256, 128], self.groups), MLFNBlock(channels[2], channels[2], 1, [256, 128], self.groups), MLFNBlock(channels[2], channels[2], 1, [256, 128], self.groups), MLFNBlock(channels[2], channels[2], 1, [256, 128], self.groups), MLFNBlock(channels[2], channels[3], 2, [512, 128], self.groups), MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups), MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups), MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups), MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups), MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups), MLFNBlock(channels[3], channels[4], 2, [512, 128], self.groups), MLFNBlock(channels[4], channels[4], 1, [512, 128], self.groups), MLFNBlock(channels[4], channels[4], 1, [512, 128], self.groups)])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_x = nn.Sequential(nn.Conv2d(channels[4], embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.fc_s = nn.Sequential(nn.Conv2d(self.groups * 16, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        s_hat = []
        for block in self.feature:
            x, s = block(x)
            s_hat.append(s)
        s_hat = torch.cat(s_hat, 1)
        x = self.global_avgpool(x)
        x = self.fc_x(x)
        s_hat = self.fc_s(s_hat)
        v = (x + s_hat) * 0.5
        v = v.view(v.size(0), -1)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def mlfn(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = MLFN(num_classes, loss, **kwargs)
    if pretrained:
        import warnings
        warnings.warn('The imagenet pretrained weights need to be manually downloaded from {}'.format(model_urls['imagenet']))
    return model


class ChannelShuffle(nn.Module):

    def __init__(self, num_groups):
        super(ChannelShuffle, self).__init__()
        self.g = num_groups

    def forward(self, x):
        b, c, h, w = x.size()
        n = c // self.g
        x = x.view(b, self.g, n, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, c, h, w)
        return x


class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, num_groups, group_conv1x1=True):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2], 'Warning: stride must be either 1 or 2'
        self.stride = stride
        mid_channels = out_channels // 4
        if stride == 2:
            out_channels -= in_channels
        num_groups_conv1x1 = num_groups if group_conv1x1 else 1
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, groups=num_groups_conv1x1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.shuffle1 = ChannelShuffle(num_groups)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, groups=num_groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if stride == 2:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = self.bn2(self.conv2(out))
        out = self.bn3(self.conv3(out))
        if self.stride == 2:
            res = self.shortcut(x)
            out = F.relu(torch.cat([res, out], 1))
        else:
            out = F.relu(x + out)
        return out


class MobileNetV2(nn.Module):
    """MobileNetV2.

    Reference:
        Sandler et al. MobileNetV2: Inverted Residuals and
        Linear Bottlenecks. CVPR 2018.

    Public keys:
        - ``mobilenetv2_x1_0``: MobileNetV2 x1.0.
        - ``mobilenetv2_x1_4``: MobileNetV2 x1.4.
    """

    def __init__(self, num_classes, width_mult=1, loss='softmax', fc_dims=None, dropout_p=None, **kwargs):
        super(MobileNetV2, self).__init__()
        self.loss = loss
        self.in_channels = int(32 * width_mult)
        self.feature_dim = int(1280 * width_mult) if width_mult > 1 else 1280
        self.conv1 = ConvBlock(3, self.in_channels, 3, s=2, p=1)
        self.conv2 = self._make_layer(Bottleneck, 1, int(16 * width_mult), 1, 1)
        self.conv3 = self._make_layer(Bottleneck, 6, int(24 * width_mult), 2, 2)
        self.conv4 = self._make_layer(Bottleneck, 6, int(32 * width_mult), 3, 2)
        self.conv5 = self._make_layer(Bottleneck, 6, int(64 * width_mult), 4, 2)
        self.conv6 = self._make_layer(Bottleneck, 6, int(96 * width_mult), 3, 1)
        self.conv7 = self._make_layer(Bottleneck, 6, int(160 * width_mult), 3, 2)
        self.conv8 = self._make_layer(Bottleneck, 6, int(320 * width_mult), 1, 1)
        self.conv9 = ConvBlock(self.in_channels, self.feature_dim, 1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, self.feature_dim, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, t, c, n, s):
        layers = []
        layers.append(block(self.in_channels, c, t, s))
        self.in_channels = c
        for i in range(1, n):
            layers.append(block(self.in_channels, c, t))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer.

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def mobilenetv2_x1_0(num_classes, loss, pretrained=True, **kwargs):
    model = MobileNetV2(num_classes, loss=loss, width_mult=1, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        import warnings
        warnings.warn('The imagenet pretrained weights need to be manually downloaded from {}'.format(model_urls['mobilenetv2_x1_0']))
    return model


def mobilenetv2_x1_4(num_classes, loss, pretrained=True, **kwargs):
    model = MobileNetV2(num_classes, loss=loss, width_mult=1.4, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        import warnings
        warnings.warn('The imagenet pretrained weights need to be manually downloaded from {}'.format(model_urls['mobilenetv2_x1_4']))
    return model


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class BranchSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, name=None, bias=False):
        super(BranchSeparables, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, in_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(in_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.name = name

    def forward(self, x):
        x = self.relu(x)
        if self.name == 'specific':
            x = nn.ZeroPad2d((1, 0, 1, 0))(x)
        x = self.separable_1(x)
        if self.name == 'specific':
            x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class BranchSeparablesStem(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.relu = nn.ReLU()
        self.separable_1 = SeparableConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn_sep_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.separable_2 = SeparableConv2d(out_channels, out_channels, kernel_size, 1, padding, bias=bias)
        self.bn_sep_2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.separable_1(x)
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class CellStem0(nn.Module):

    def __init__(self, stem_filters, num_filters=42):
        super(CellStem0, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 7, 2, 3, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(self.stem_filters, self.num_filters, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.num_filters, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class AvgPoolPad(nn.Module):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.AvgPool2d(3, stride=stride, padding=padding, count_include_pad=False)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class MaxPoolPad(nn.Module):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.pad(x)
        x = self.pool(x)
        x = x[:, :, 1:, 1:].contiguous()
        return x


class CellStem1(nn.Module):

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()
        self.num_filters = num_filters
        self.stem_filters = stem_filters
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(2 * self.num_filters, self.num_filters, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters // 2, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(self.stem_filters, self.num_filters // 2, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(self.num_filters, eps=0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(self.num_filters, self.num_filters, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(self.num_filters, self.num_filters, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(self.num_filters, self.num_filters, 3, 1, 1, name='specific', bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)
        x_relu = self.relu(x_conv0)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class FirstCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.relu = nn.ReLU()
        self.path_1 = nn.Sequential()
        self.path_1.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.path_2 = nn.ModuleList()
        self.path_2.add_module('pad', nn.ZeroPad2d((0, 1, 0, 1)))
        self.path_2.add_module('avgpool', nn.AvgPool2d(1, stride=2, count_include_pad=False))
        self.path_2.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.final_path_bn = nn.BatchNorm2d(out_channels_left * 2, eps=0.001, momentum=0.1, affine=True)
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_relu = self.relu(x_prev)
        x_path1 = self.path_1(x_relu)
        x_path2 = self.path_2.pad(x_relu)
        x_path2 = x_path2[:, :, 1:, 1:]
        x_path2 = self.path_2.avgpool(x_path2)
        x_path2 = self.path_2.conv(x_path2)
        x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NormalCell(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(NormalCell, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_1_left = BranchSeparables(out_channels_left, out_channels_left, 5, 1, 2, bias=False)
        self.comb_iter_1_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1, bias=False)
        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left
        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right
        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right
        x_out = torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class BranchSeparablesReduction(BranchSeparables):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, z_padding=1, bias=False):
        BranchSeparables.__init__(self, in_channels, out_channels, kernel_size, stride, padding, bias)
        self.padding = nn.ZeroPad2d((z_padding, 0, z_padding, 0))

    def forward(self, x):
        x = self.relu(x)
        x = self.padding(x)
        x = self.separable_1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x = self.bn_sep_1(x)
        x = self.relu1(x)
        x = self.separable_2(x)
        x = self.bn_sep_2(x)
        return x


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell0, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 7, 2, 3, bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class ReductionCell1(nn.Module):

    def __init__(self, in_channels_left, out_channels_left, in_channels_right, out_channels_right):
        super(ReductionCell1, self).__init__()
        self.conv_prev_1x1 = nn.Sequential()
        self.conv_prev_1x1.add_module('relu', nn.ReLU())
        self.conv_prev_1x1.add_module('conv', nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False))
        self.conv_prev_1x1.add_module('bn', nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True))
        self.conv_1x1 = nn.Sequential()
        self.conv_1x1.add_module('relu', nn.ReLU())
        self.conv_1x1.add_module('conv', nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False))
        self.conv_1x1.add_module('bn', nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True))
        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3, name='specific', bias=False)
        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2, name='specific', bias=False)
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1, name='specific', bias=False)
        self.comb_iter_4_right = MaxPoolPad()

    def forward(self, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)
        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right
        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right
        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right
        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1
        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right
        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class NASNetAMobile(nn.Module):
    """Neural Architecture Search (NAS).

    Reference:
        Zoph et al. Learning Transferable Architectures
        for Scalable Image Recognition. CVPR 2018.

    Public keys:
        - ``nasnetamobile``: NASNet-A Mobile.
    """

    def __init__(self, num_classes, loss, stem_filters=32, penultimate_filters=1056, filters_multiplier=2, **kwargs):
        super(NASNetAMobile, self).__init__()
        self.stem_filters = stem_filters
        self.penultimate_filters = penultimate_filters
        self.filters_multiplier = filters_multiplier
        self.loss = loss
        filters = self.penultimate_filters // 24
        self.conv0 = nn.Sequential()
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=self.stem_filters, kernel_size=3, padding=0, stride=2, bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = CellStem0(self.stem_filters, num_filters=filters // filters_multiplier ** 2)
        self.cell_stem_1 = CellStem1(self.stem_filters, num_filters=filters // filters_multiplier)
        self.cell_0 = FirstCell(in_channels_left=filters, out_channels_left=filters // 2, in_channels_right=2 * filters, out_channels_right=filters)
        self.cell_1 = NormalCell(in_channels_left=2 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_2 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.cell_3 = NormalCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=6 * filters, out_channels_right=filters)
        self.reduction_cell_0 = ReductionCell0(in_channels_left=6 * filters, out_channels_left=2 * filters, in_channels_right=6 * filters, out_channels_right=2 * filters)
        self.cell_6 = FirstCell(in_channels_left=6 * filters, out_channels_left=filters, in_channels_right=8 * filters, out_channels_right=2 * filters)
        self.cell_7 = NormalCell(in_channels_left=8 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_8 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.cell_9 = NormalCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=12 * filters, out_channels_right=2 * filters)
        self.reduction_cell_1 = ReductionCell1(in_channels_left=12 * filters, out_channels_left=4 * filters, in_channels_right=12 * filters, out_channels_right=4 * filters)
        self.cell_12 = FirstCell(in_channels_left=12 * filters, out_channels_left=2 * filters, in_channels_right=16 * filters, out_channels_right=4 * filters)
        self.cell_13 = NormalCell(in_channels_left=16 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_14 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.cell_15 = NormalCell(in_channels_left=24 * filters, out_channels_left=4 * filters, in_channels_right=24 * filters, out_channels_right=4 * filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(24 * filters, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def features(self, input):
        x_conv0 = self.conv0(input)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_reduction_cell_0 = self.reduction_cell_0(x_cell_3, x_cell_2)
        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_3)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_reduction_cell_1 = self.reduction_cell_1(x_cell_9, x_cell_8)
        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_9)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_15 = self.relu(x_cell_15)
        x_cell_15 = F.avg_pool2d(x_cell_15, x_cell_15.size()[2:])
        x_cell_15 = x_cell_15.view(x_cell_15.size(0), -1)
        x_cell_15 = self.dropout(x_cell_15)
        return x_cell_15

    def forward(self, input):
        v = self.features(input)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def nasnetamobile(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = NASNetAMobile(num_classes, loss, **kwargs)
    if pretrained:
        model_url = pretrained_settings['nasnetamobile']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False, gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU()
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError('Unknown gate activation: {}'.format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.relu(x)


class LightConvStream(nn.Module):
    """Lightweight convolution stream."""

    def __init__(self, in_channels, out_channels, depth):
        super(LightConvStream, self).__init__()
        assert depth >= 1, 'depth must be equal to or larger than 1, but got {}'.format(depth)
        layers = []
        layers += [LightConv3x3(in_channels, out_channels)]
        for i in range(depth - 1):
            layers += [LightConv3x3(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class OSBlockINin(nn.Module):
    """Omni-scale feature learning block with instance normalization."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlockINin, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        x3 = self.IN(x3)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False, groups=groups)
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class OSNet(nn.Module):
    """Omni-Scale Network.
    
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss='softmax', conv1_IN=False, **kwargs):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=conv1_IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1])
        self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2))
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2))
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3])
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(self.feature_dim, channels[3], dropout_p=None)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, blocks, layer, in_channels, out_channels):
        layers = []
        layers += [blocks[0](in_channels, out_channels)]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels)]
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def osnet_ain_x0_25(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]], layers=[2, 2, 2], channels=[16, 64, 96, 128], loss=loss, conv1_IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x0_25')
    return model


def osnet_ain_x0_5(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]], layers=[2, 2, 2], channels=[32, 128, 192, 256], loss=loss, conv1_IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x0_5')
    return model


def osnet_ain_x0_75(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]], layers=[2, 2, 2], channels=[48, 192, 288, 384], loss=loss, conv1_IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x0_75')
    return model


def osnet_ain_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[[OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin], [OSBlockINin, OSBlock]], layers=[2, 2, 2], channels=[64, 256, 384, 512], loss=loss, conv1_IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x1_0')
    return model


def osnet_ibn_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[64, 256, 384, 512], loss=loss, IN=True, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_ibn_x1_0')
    return model


def osnet_x0_25(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[16, 64, 96, 128], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_25')
    return model


def osnet_x0_5(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[32, 128, 192, 256], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_5')
    return model


def osnet_x0_75(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[48, 192, 288, 384], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_x0_75')
    return model


def osnet_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[64, 256, 384, 512], loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, key='osnet_x1_0')
    return model


class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PCB(nn.Module):
    """Part-based Convolutional Baseline.

    Reference:
        Sun et al. Beyond Part Models: Person Retrieval with Refined
        Part Pooling (and A Strong Convolutional Baseline). ECCV 2018.

    Public keys:
        - ``pcb_p4``: PCB with 4-part strips.
        - ``pcb_p6``: PCB with 6-part strips.
    """

    def __init__(self, num_classes, loss, block, layers, parts=6, reduced_dim=256, nonlinear='relu', **kwargs):
        self.inplanes = 64
        super(PCB, self).__init__()
        self.loss = loss
        self.parts = parts
        self.feature_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = DimReduceLayer(512 * block.expansion, reduced_dim, nonlinear=nonlinear)
        self.feature_dim = reduced_dim
        self.classifier = nn.ModuleList([nn.Linear(self.feature_dim, num_classes) for _ in range(self.parts)])
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v_g = self.parts_avgpool(f)
        if not self.training:
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)
        v_g = self.dropout(v_g)
        v_h = self.conv5(v_g)
        y = []
        for i in range(self.parts):
            v_h_i = v_h[:, :, i, :]
            v_h_i = v_h_i.view(v_h_i.size(0), -1)
            y_i = self.classifier[i](v_h_i)
            y.append(y_i)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            v_g = F.normalize(v_g, p=2, dim=1)
            return y, v_g.view(v_g.size(0), -1)
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def pcb_p4(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=1, parts=4, reduced_dim=256, nonlinear='relu', **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def pcb_p6(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = PCB(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=1, parts=6, reduced_dim=256, nonlinear='relu', **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


class ResNet(nn.Module):
    """Residual network + IBN layer.
    
    Reference:
        - He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
        - Pan et al. Two at Once: Enhancing Learning and Generalization
          Capacities via IBN-Net. ECCV 2018.
    """

    def __init__(self, block, layers, num_classes=1000, loss='softmax', fc_dims=None, dropout_p=None, **kwargs):
        scale = 64
        self.inplanes = scale
        super(ResNet, self).__init__()
        self.loss = loss
        self.feature_dim = scale * 8 * block.expansion
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.InstanceNorm2d(scale, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0], stride=1, IN=True)
        self.layer2 = self._make_layer(block, scale * 2, layers[1], stride=2, IN=True)
        self.layer3 = self._make_layer(block, scale * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._construct_fc_layer(fc_dims, scale * 8 * block.expansion, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, IN=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, IN=IN))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def resnet101(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 23, 3], last_stride=2, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])
    return model


def resnet152(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 8, 36, 3], last_stride=2, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])
    return model


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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


def resnet18(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=BasicBlock, layers=[2, 2, 2, 2], last_stride=2, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model


def resnet34(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=BasicBlock, layers=[3, 4, 6, 3], last_stride=2, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])
    return model


def resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=2, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=1, fc_dims=[512], dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_ibn_a(num_classes, loss='softmax', pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnet50_ibn_b(num_classes, loss='softmax', pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, loss=loss, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


class ResNetMid(nn.Module):
    """Residual network + mid-level features.
    
    Reference:
        Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
        Cross-Domain Instance Matching. arXiv:1711.08106.

    Public keys:
        - ``resnet50mid``: ResNet50 + mid-level feature fusion.
    """

    def __init__(self, num_classes, loss, block, layers, last_stride=2, fc_dims=None, **kwargs):
        self.inplanes = 64
        super(ResNetMid, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        assert fc_dims is not None
        self.fc_fusion = self._construct_fc_layer(fc_dims, 512 * block.expansion * 2)
        self.feature_dim += 512 * block.expansion
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x4a = self.layer4[0](x)
        x4b = self.layer4[1](x4a)
        x4c = self.layer4[2](x4b)
        return x4a, x4b, x4c

    def forward(self, x):
        x4a, x4b, x4c = self.featuremaps(x)
        v4a = self.global_avgpool(x4a)
        v4b = self.global_avgpool(x4b)
        v4c = self.global_avgpool(x4c)
        v4ab = torch.cat([v4a, v4b], 1)
        v4ab = v4ab.view(v4ab.size(0), -1)
        v4ab = self.fc_fusion(v4ab)
        v4c = v4c.view(v4c.size(0), -1)
        v = torch.cat([v4ab, v4c], 1)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def resnet50mid(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNetMid(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=2, fc_dims=[1024], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


def resnext101_32x8d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 23, 3], last_stride=2, fc_dims=None, dropout_p=None, groups=32, width_per_group=8, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnext101_32x8d'])
    return model


def resnext50_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(num_classes=num_classes, loss=loss, block=Bottleneck, layers=[3, 4, 6, 3], last_stride=2, fc_dims=None, dropout_p=None, groups=32, width_per_group=4, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnext50_32x4d'])
    return model


class SENet(nn.Module):
    """Squeeze-and-excitation network.
    
    Reference:
        Hu et al. Squeeze-and-Excitation Networks. CVPR 2018.

    Public keys:
        - ``senet154``: SENet154.
        - ``se_resnet50``: ResNet50 + SE.
        - ``se_resnet101``: ResNet101 + SE.
        - ``se_resnet152``: ResNet152 + SE.
        - ``se_resnext50_32x4d``: ResNeXt50 (groups=32, width=4) + SE.
        - ``se_resnext101_32x4d``: ResNeXt101 (groups=32, width=4) + SE.
        - ``se_resnet50_fc512``: (ResNet50 + SE) + FC.
    """

    def __init__(self, num_classes, loss, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, last_stride=2, fc_dims=None, **kwargs):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `classifier` layer.
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        self.loss = loss
        if input_3x3:
            layer0_modules = [('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)), ('bn1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)), ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)), ('bn2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)), ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)), ('bn3', nn.BatchNorm2d(inplanes)), ('relu3', nn.ReLU(inplace=True))]
        else:
            layer0_modules = [('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)), ('bn1', nn.BatchNorm2d(inplanes)), ('relu1', nn.ReLU(inplace=True))]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], groups=groups, reduction=reduction, downsample_kernel_size=1, downsample_padding=0)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=last_stride, groups=groups, reduction=reduction, downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def featuremaps(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


def se_resnet101(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEResNetBottleneck, layers=[3, 4, 23, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, last_stride=2, fc_dims=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['se_resnet101']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


def se_resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEResNetBottleneck, layers=[3, 4, 6, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, last_stride=2, fc_dims=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['se_resnet50']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


def se_resnet50_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEResNetBottleneck, layers=[3, 4, 6, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, last_stride=1, fc_dims=[512], **kwargs)
    if pretrained:
        model_url = pretrained_settings['se_resnet50']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


class SEResNeXtBottleneck(Bottleneck):
    """ResNeXt bottleneck type C with a Squeeze-and-Excitation module"""
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64.0)) * groups)
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


def se_resnext101_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEResNeXtBottleneck, layers=[3, 4, 23, 3], groups=32, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, last_stride=2, fc_dims=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['se_resnext101_32x4d']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


def se_resnext50_32x4d(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SENet(num_classes=num_classes, loss=loss, block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], groups=32, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, last_stride=2, fc_dims=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['se_resnext50_32x4d']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


class ShuffleNet(nn.Module):
    """ShuffleNet.

    Reference:
        Zhang et al. ShuffleNet: An Extremely Efficient Convolutional Neural
        Network for Mobile Devices. CVPR 2018.

    Public keys:
        - ``shufflenet``: ShuffleNet (groups=3).
    """

    def __init__(self, num_classes, loss='softmax', num_groups=3, **kwargs):
        super(ShuffleNet, self).__init__()
        self.loss = loss
        self.conv1 = nn.Sequential(nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1))
        self.stage2 = nn.Sequential(Bottleneck(24, cfg[num_groups][0], 2, num_groups, group_conv1x1=False), Bottleneck(cfg[num_groups][0], cfg[num_groups][0], 1, num_groups), Bottleneck(cfg[num_groups][0], cfg[num_groups][0], 1, num_groups), Bottleneck(cfg[num_groups][0], cfg[num_groups][0], 1, num_groups))
        self.stage3 = nn.Sequential(Bottleneck(cfg[num_groups][0], cfg[num_groups][1], 2, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups), Bottleneck(cfg[num_groups][1], cfg[num_groups][1], 1, num_groups))
        self.stage4 = nn.Sequential(Bottleneck(cfg[num_groups][1], cfg[num_groups][2], 2, num_groups), Bottleneck(cfg[num_groups][2], cfg[num_groups][2], 1, num_groups), Bottleneck(cfg[num_groups][2], cfg[num_groups][2], 1, num_groups), Bottleneck(cfg[num_groups][2], cfg[num_groups][2], 1, num_groups))
        self.classifier = nn.Linear(cfg[num_groups][2], num_classes)
        self.feat_dim = cfg[num_groups][2]

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        if not self.training:
            return x
        y = self.classifier(x)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, x
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def shufflenet(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ShuffleNet(num_classes, loss, **kwargs)
    if pretrained:
        import warnings
        warnings.warn('The imagenet pretrained weights need to be manually downloaded from {}'.format(model_urls['imagenet']))
    return model


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        if not 1 <= stride <= 3:
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert self.stride != 1 or inp == branch_features << 1
        if self.stride > 1:
            self.branch1 = nn.Sequential(self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(inp), nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True), self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(branch_features), nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2.
    
    Reference:
        Ma et al. ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design. ECCV 2018.

    Public keys:
        - ``shufflenet_v2_x0_5``: ShuffleNetV2 x0.5.
        - ``shufflenet_v2_x1_0``: ShuffleNetV2 x1.0.
        - ``shufflenet_v2_x1_5``: ShuffleNetV2 x1.5.
        - ``shufflenet_v2_x2_0``: ShuffleNetV2 x2.0.
    """

    def __init__(self, num_classes, loss, stages_repeats, stages_out_channels, **kwargs):
        super(ShuffleNetV2, self).__init__()
        self.loss = loss
        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channels, num_classes)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def shufflenet_v2_x0_5(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ShuffleNetV2(num_classes, loss, [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['shufflenetv2_x0.5'])
    return model


def shufflenet_v2_x1_0(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ShuffleNetV2(num_classes, loss, [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['shufflenetv2_x1.0'])
    return model


def shufflenet_v2_x1_5(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ShuffleNetV2(num_classes, loss, [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['shufflenetv2_x1.5'])
    return model


def shufflenet_v2_x2_0(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ShuffleNetV2(num_classes, loss, [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['shufflenetv2_x2.0'])
    return model


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)


class SqueezeNet(nn.Module):
    """SqueezeNet.

    Reference:
        Iandola et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
        and< 0.5 MB model size. arXiv:1602.07360.

    Public keys:
        - ``squeezenet1_0``: SqueezeNet (version=1.0).
        - ``squeezenet1_1``: SqueezeNet (version=1.1).
        - ``squeezenet1_0_fc512``: SqueezeNet (version=1.0) + FC.
    """

    def __init__(self, num_classes, loss, version=1.0, fc_dims=None, dropout_p=None, **kwargs):
        super(SqueezeNet, self).__init__()
        self.loss = loss
        self.feature_dim = 512
        if version not in [1.0, 1.1]:
            raise ValueError('Unsupported SqueezeNet version {version}:1.0 or 1.1 expected'.format(version=version))
        if version == 1.0:
            self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(96, 16, 64, 64), Fire(128, 16, 64, 64), Fire(128, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(256, 32, 128, 128), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(512, 64, 256, 256))
        else:
            self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(128, 32, 128, 128), Fire(256, 32, 128, 128), nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192), Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(fc_dims, 512, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.features(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def squeezenet1_0(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SqueezeNet(num_classes, loss, version=1.0, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['squeezenet1_0'])
    return model


def squeezenet1_0_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SqueezeNet(num_classes, loss, version=1.0, fc_dims=[512], dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['squeezenet1_0'])
    return model


def squeezenet1_1(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = SqueezeNet(num_classes, loss, version=1.1, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['squeezenet1_1'])
    return model


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    """Xception.
    
    Reference:
        Chollet. Xception: Deep Learning with Depthwise
        Separable Convolutions. CVPR 2017.

    Public keys:
        - ``xception``: Xception.
    """

    def __init__(self, num_classes, loss, fc_dims=None, dropout_p=None, **kwargs):
        super(Xception, self).__init__()
        self.loss = loss
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 2048
        self.fc = self._construct_fc_layer(fc_dims, 2048, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer.

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def xception(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = Xception(num_classes, loss, fc_dims=None, dropout_p=None, **kwargs)
    if pretrained:
        model_url = pretrained_settings['xception']['imagenet']['url']
        init_pretrained_weights(model, model_url)
    return model


__model_factory = {'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152, 'resnext50_32x4d': resnext50_32x4d, 'resnext101_32x8d': resnext101_32x8d, 'resnet50_fc512': resnet50_fc512, 'se_resnet50': se_resnet50, 'se_resnet50_fc512': se_resnet50_fc512, 'se_resnet101': se_resnet101, 'se_resnext50_32x4d': se_resnext50_32x4d, 'se_resnext101_32x4d': se_resnext101_32x4d, 'densenet121': densenet121, 'densenet169': densenet169, 'densenet201': densenet201, 'densenet161': densenet161, 'densenet121_fc512': densenet121_fc512, 'inceptionresnetv2': inceptionresnetv2, 'inceptionv4': inceptionv4, 'xception': xception, 'resnet50_ibn_a': resnet50_ibn_a, 'resnet50_ibn_b': resnet50_ibn_b, 'nasnsetmobile': nasnetamobile, 'mobilenetv2_x1_0': mobilenetv2_x1_0, 'mobilenetv2_x1_4': mobilenetv2_x1_4, 'shufflenet': shufflenet, 'squeezenet1_0': squeezenet1_0, 'squeezenet1_0_fc512': squeezenet1_0_fc512, 'squeezenet1_1': squeezenet1_1, 'shufflenet_v2_x0_5': shufflenet_v2_x0_5, 'shufflenet_v2_x1_0': shufflenet_v2_x1_0, 'shufflenet_v2_x1_5': shufflenet_v2_x1_5, 'shufflenet_v2_x2_0': shufflenet_v2_x2_0, 'mudeep': MuDeep, 'resnet50mid': resnet50mid, 'hacnn': HACNN, 'pcb_p6': pcb_p6, 'pcb_p4': pcb_p4, 'mlfn': mlfn, 'osnet_x1_0': osnet_x1_0, 'osnet_x0_75': osnet_x0_75, 'osnet_x0_5': osnet_x0_5, 'osnet_x0_25': osnet_x0_25, 'osnet_ibn_x1_0': osnet_ibn_x1_0, 'osnet_ain_x1_0': osnet_ain_x1_0, 'osnet_ain_x0_75': osnet_ain_x0_75, 'osnet_ain_x0_5': osnet_ain_x0_5, 'osnet_ain_x0_25': osnet_ain_x0_25}


def build_model(name, num_classes, loss='softmax', pretrained=True, use_gpu=True):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu)


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in (file if isinstance(file, (list, tuple)) else [file]):
            s = Path(f).suffix.lower()
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}'


__model_types = ['resnet50', 'mlfn', 'hacnn', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4', 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0', 'osnet_ain_x1_0']


def get_model_name(model):
    for x in __model_types:
        if x in model.name:
            return x
    return None


__trained_urls = {'resnet50_market1501.pt': 'https://drive.google.com/uc?id=1dUUZ4rHDWohmsQXCRe2C_HbYkzz94iBV', 'resnet50_dukemtmcreid.pt': 'https://drive.google.com/uc?id=17ymnLglnc64NRvGOitY3BqMRS9UWd1wg', 'resnet50_msmt17.pt': 'https://drive.google.com/uc?id=1ep7RypVDOthCRIAqDnn4_N-UhkkFHJsj', 'resnet50_fc512_market1501.pt': 'https://drive.google.com/uc?id=1kv8l5laX_YCdIGVCetjlNdzKIA3NvsSt', 'resnet50_fc512_dukemtmcreid.pt': 'https://drive.google.com/uc?id=13QN8Mp3XH81GK4BPGXobKHKyTGH50Rtx', 'resnet50_fc512_msmt17.pt': 'https://drive.google.com/uc?id=1fDJLcz4O5wxNSUvImIIjoaIF9u1Rwaud', 'mlfn_market1501.pt': 'https://drive.google.com/uc?id=1wXcvhA_b1kpDfrt9s2Pma-MHxtj9pmvS', 'mlfn_dukemtmcreid.pt': 'https://drive.google.com/uc?id=1rExgrTNb0VCIcOnXfMsbwSUW1h2L1Bum', 'mlfn_msmt17.pt': 'https://drive.google.com/uc?id=18JzsZlJb3Wm7irCbZbZ07TN4IFKvR6p-', 'hacnn_market1501.pt': 'https://drive.google.com/uc?id=1LRKIQduThwGxMDQMiVkTScBwR7WidmYF', 'hacnn_dukemtmcreid.pt': 'https://drive.google.com/uc?id=1zNm6tP4ozFUCUQ7Sv1Z98EAJWXJEhtYH', 'hacnn_msmt17.pt': 'https://drive.google.com/uc?id=1MsKRtPM5WJ3_Tk2xC0aGOO7pM3VaFDNZ', 'mobilenetv2_x1_0_market1501.pt': 'https://drive.google.com/uc?id=18DgHC2ZJkjekVoqBWszD8_Xiikz-fewp', 'mobilenetv2_x1_0_dukemtmcreid.pt': 'https://drive.google.com/uc?id=1q1WU2FETRJ3BXcpVtfJUuqq4z3psetds', 'mobilenetv2_x1_0_msmt17.pt': 'https://drive.google.com/uc?id=1j50Hv14NOUAg7ZeB3frzfX-WYLi7SrhZ', 'mobilenetv2_x1_4_market1501.pt': 'https://drive.google.com/uc?id=1t6JCqphJG-fwwPVkRLmGGyEBhGOf2GO5', 'mobilenetv2_x1_4_dukemtmcreid.pt': 'https://drive.google.com/uc?id=12uD5FeVqLg9-AFDju2L7SQxjmPb4zpBN', 'mobilenetv2_x1_4_msmt17.pt': 'https://drive.google.com/uc?id=1ZY5P2Zgm-3RbDpbXM0kIBMPvspeNIbXz', 'osnet_x1_0_market1501.pt': 'https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA', 'osnet_x1_0_dukemtmcreid.pt': 'https://drive.google.com/uc?id=1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq', 'osnet_x1_0_msmt17.pt': 'https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M', 'osnet_x0_75_market1501.pt': 'https://drive.google.com/uc?id=1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer', 'osnet_x0_75_dukemtmcreid.pt': 'https://drive.google.com/uc?id=1IE3KRaTPp4OUa6PGTFL_d5_KQSJbP0Or', 'osnet_x0_75_msmt17.pt': 'https://drive.google.com/uc?id=1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc', 'osnet_x0_5_market1501.pt': 'https://drive.google.com/uc?id=1PLB9rgqrUM7blWrg4QlprCuPT7ILYGKT', 'osnet_x0_5_dukemtmcreid.pt': 'https://drive.google.com/uc?id=1KoUVqmiST175hnkALg9XuTi1oYpqcyTu', 'osnet_x0_5_msmt17.pt': 'https://drive.google.com/uc?id=1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv', 'osnet_x0_25_market1501.pt': 'https://drive.google.com/uc?id=1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj', 'osnet_x0_25_dukemtmcreid.pt': 'https://drive.google.com/uc?id=1eumrtiXT4NOspjyEV4j8cHmlOaaCGk5l', 'osnet_x0_25_msmt17.pt': 'https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF', 'resnet50_msmt17.pt': 'https://drive.google.com/uc?id=1yiBteqgIZoOeywE8AhGmEQl7FTVwrQmf', 'osnet_x1_0_msmt17.pt': 'https://drive.google.com/uc?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x', 'osnet_x0_75_msmt17.pt': 'https://drive.google.com/uc?id=1fhjSS_7SUGCioIf2SWXaRGPqIY9j7-uw', 'osnet_x0_5_msmt17.pt': 'https://drive.google.com/uc?id=1DHgmb6XV4fwG3n-CnCM0zdL9nMsZ9_RF', 'osnet_x0_25_msmt17.pt': 'https://drive.google.com/uc?id=1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e', 'osnet_ibn_x1_0_msmt17.pt': 'https://drive.google.com/uc?id=1q3Sj2ii34NlfxA4LvmHdWO_75NDRmECJ', 'osnet_ain_x1_0_msmt17.pt': 'https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal'}


def get_model_url(model):
    if model.name in __trained_urls:
        return __trained_urls[model.name]
    else:
        None


def load_pretrained_weights(model, weight_path):
    """Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = torch.load(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    if len(matched_layers) == 0:
        warnings.warn('The pretrained weights "{}" cannot be loaded, please check the key names manually (** ignored and continue **)'.format(weight_path))
    else:
        None
        if len(discarded_layers) > 0:
            None


def show_downloadeable_models():
    None
    None


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class IBN(nn.Module):

    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class VideoReader:

    def __init__(self):
        self.filename = None
        self.cap = None

    def set_filename(self, filename):
        self.filename = filename
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.filename)
        assert self.cap.isOpened(), 'Error opening video stream or file'

    def __getitem__(self, idx):
        assert self.filename is not None, 'You should first set the filename'
        cap = cv2.VideoCapture(self.filename)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ret, image = cap.read()
        cap.release()
        assert ret, 'Read past the end of the video file'
        return image


video_reader = VideoReader()


class EngineDatapipe(Dataset):

    def __init__(self, model) ->None:
        self.model = model
        self.image_filepaths = None
        self.img_metadatas = None
        self.detections = None

    def update(self, image_filepaths: 'dict', img_metadatas, detections):
        del self.img_metadatas
        del self.detections
        self.image_filepaths = image_filepaths
        self.img_metadatas = img_metadatas
        self.detections = detections

    def __len__(self):
        if self.model.level == 'detection':
            return len(self.detections)
        elif self.model.level == 'image':
            return len(self.img_metadatas)
        else:
            raise ValueError(f"You should provide the appropriate level for you module not '{self.model.level}'")

    def __getitem__(self, idx):
        if self.model.level == 'detection':
            detection = self.detections.iloc[idx]
            metadata = self.img_metadatas.loc[detection.image_id]
            image = cv2_load_image(self.image_filepaths[metadata.name])
            sample = detection.name, self.model.preprocess(image=image, detection=detection, metadata=metadata)
            return sample
        elif self.model.level == 'image':
            metadata = self.img_metadatas.iloc[idx]
            if self.detections is not None and len(self.detections) > 0:
                detections = self.detections[self.detections.image_id == metadata.name]
            else:
                detections = self.detections
            image = cv2_load_image(self.image_filepaths[metadata.name])
            sample = self.img_metadatas.index[idx], self.model.preprocess(image=image, detections=detections, metadata=metadata)
            return sample
        else:
            raise ValueError('Please provide appropriate level.')


class DetectionLevelModule(Module):
    """Abstract class to implement a module that operates directly on detections.

    This can for example be a top-down pose estimator, or a reidentifier module.

    The functions to implement are
     - __init__, which can take any configuration needed
     - preprocess
     - process
     - datapipe (optional) : returns an object which will be used to create the pipeline.
                            (Only modify this if you know what you're doing)
     - dataloader (optional) : returns a dataloader for the datapipe

     You should also provide the following class properties :
      - input_columns : what info you need for the detections
      - output_columns : what info you will provide when called
      - collate_fn (optional) : the function that will be used for collating the inputs
                                in a batch. (Default : pytorch collate function)

     A description of the expected behavior is provided below.
    """
    collate_fn = default_collate
    input_columns = None
    output_columns = None

    @abstractmethod
    def __init__(self, batch_size: 'int'):
        """Init function

        The arguments to this function are completely free
        and will be provided by a configuration file.

        You should call the __init__ function from the super() class.
        """
        self.batch_size = batch_size
        self._datapipe = None

    @abstractmethod
    def preprocess(self, image, detection: 'pd.Series', metadata: 'pd.Series') ->Any:
        """Adapts the default input to your specific case.

        Args:
            image: a numpy array of the current image
            detection: a Series containing all the detections pertaining to a single
                        image
            metadata: additional information about the image

        Returns:
            preprocessed_sample: input for the process function
        """
        pass

    @abstractmethod
    def process(self, batch: 'Any', detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        """The main processing function. Runs on GPU.

        Args:
            batch: The batched outputs of `preprocess`
            detections: The previous detections.
            metadatas: The previous image metadatas

        Returns:
            output : Either a DataFrame containing the new/updated detections.
                    The DataFrames can be either a list of Series, a list of DataFrames
                    or a single DataFrame. The returned objects will be aggregated
                    automatically according to the `name` of the Series/`index` of
                    the DataFrame. **It is thus mandatory here to name correctly
                    your series or index your dataframes.**
                    The output will override the previous detections
                    with the same name/index.
        """
        pass

    @property
    def datapipe(self):
        if self._datapipe is None:
            self._datapipe = EngineDatapipe(self)
        return self._datapipe

    def dataloader(self, engine: "'TrackingEngine'"):
        datapipe = self.datapipe
        return DataLoader(dataset=datapipe, batch_size=self.batch_size, collate_fn=type(self).collate_fn, num_workers=engine.num_workers, persistent_workers=False)


class ImageLevelModule(Module):
    """Abstract class to implement a module that operates directly on images.

    This can for example be a bounding box detector, or a bottom-up
    pose estimator (which outputs keypoints directly).

    The functions to implement are
     - __init__, which can take any configuration needed
     - preprocess
     - process
     - datapipe (optional) : returns an object which will be used to create the pipeline.
                            (Only modify this if you know what you're doing)
     - dataloader (optional) : returns a dataloader for the datapipe

     You should also provide the following class properties :
      - input_columns : what info you need for the detections
      - output_columns : what info you will provide when called
      - collate_fn (optional) : the function that will be used for collating the inputs
                                in a batch. (Default : pytorch collate function)

     A description of the expected behavior is provided below.
    """
    collate_fn = default_collate
    input_columns = None
    output_columns = None

    @abstractmethod
    def __init__(self, batch_size: 'int'):
        """Init function

        The arguments to this function are completely free
        and will be provided by a configuration file.

        You should call the __init__ function from the super() class.
        """
        self.batch_size = batch_size
        self._datapipe = None

    @abstractmethod
    def preprocess(self, image, detections: 'pd.DataFrame', metadata: 'pd.Series') ->Any:
        """Adapts the default input to your specific case.

        Args:
            image: a numpy array of the current image
            detections: a DataFrame containing all the detections pertaining to a single
                        image
            metadata: additional information about the image

        Returns:
            preprocessed_sample: input for the process function
        """
        pass

    @abstractmethod
    def process(self, batch: 'Any', detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        """The main processing function. Runs on GPU.

        Args:
            batch: The batched outputs of `preprocess`
            detections: The previous detections.
            metadatas: The previous image metadatas

        Returns:
            output : Either a DataFrame containing the new/updated detections
                    or a tuple containing detections and metadatas (in that order)
                    The DataFrames can be either a list of Series, a list of DataFrames
                    or a single DataFrame. The returned objects will be aggregated
                    automatically according to the `name` of the Series/`index` of
                    the DataFrame. **It is thus mandatory here to name correctly
                    your series or index your dataframes.**
                    The output will override the previous detections
                    with the same name/index.
        """
        pass

    @property
    def datapipe(self):
        if self._datapipe is None:
            self._datapipe = EngineDatapipe(self)
        return self._datapipe

    def dataloader(self, engine: "'TrackingEngine'"):
        datapipe = self.datapipe
        return DataLoader(dataset=datapipe, batch_size=self.batch_size, collate_fn=type(self).collate_fn, num_workers=engine.num_workers, persistent_workers=False)


class VideoLevelModule(Module):
    """Abstract class to implement a module that operates on whole videos, image per image.

    This can for example be an offline tracker, or a video visualizer, by implementing
    process_video directly, or an online tracker, by implementing preprocess and process

    The functions to implement are
     - __init__, which can take any configuration needed
     - process

     You should also provide the following class properties :
      - input_columns : what info you need for the detections
      - output_columns : what info you will provide when called

     A description of the expected behavior is provided below.
    """
    input_columns = None
    output_columns = None

    @abstractmethod
    def __init__(self):
        """Init function

        The arguments to this function are completely free
        and will be provided by a configuration file.

        You should call the __init__ function from the super() class.
        """
        pass

    @abstractmethod
    def process(self, detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        """The main processing function. Runs on GPU.

        Args:
            detections: The previous detections.
            metadatas: The previous image metadatas

        Returns:
            output : Either a DataFrame containing the new/updated detections
                    or a tuple containing detections and metadatas (in that order)
                    The DataFrames can be either a list of Series, a list of DataFrames
                    or a single DataFrame. The returned objects will be aggregated
                    automatically according to the `name` of the Series/`index` of
                    the DataFrame. **It is thus mandatory here to name correctly
                    your series or index your dataframes.**
                    The output will override the previous detections
                    with the same name/index.
        """
        pass


log = logging.getLogger(__name__)


def get_checkpoint(path_to_checkpoint, download_url):
    os.makedirs(os.path.dirname(path_to_checkpoint), exist_ok=True)
    if not os.path.exists(path_to_checkpoint):
        log.info('Checkpoint not found at {}'.format(path_to_checkpoint))
        log.info('Downloading checkpoint from {}'.format(download_url))
        response = requests.get(download_url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True)
        with open(path_to_checkpoint, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            log.warning(f'Something went wrong while downloading or writing {download_url} to {path_to_checkpoint}')
        else:
            log.info('Checkpoint downloaded successfully')


def sanitize_bbox_ltrb(bbox, image_shape=None, rounded=False):
    """
    Sanitizes a bounding box by clipping it to the image dimensions and ensuring that its dimensions are valid.

    Args:
        bbox (np.ndarray): A numpy array of shape (4,) representing the bounding box in the format
        `[left, top, right, bottom]`.
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.
        round (bool): Whether to round the bounding box coordinates, type becomes int.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the sanitized bounding box in the format
        `[left, top, right, bottom]`.
    """
    assert isinstance(bbox, np.ndarray), f'Expected bbox to be of type np.ndarray, got {type(bbox)}'
    assert bbox.shape == (4,), f'Expected bbox to be of shape (4,), got {bbox.shape}'
    if image_shape:
        bbox[0] = max(0, min(bbox[0], image_shape[0] - 2))
        bbox[1] = max(0, min(bbox[1], image_shape[1] - 2))
        bbox[2] = max(1, min(bbox[2], image_shape[0] - 1))
        bbox[3] = max(1, min(bbox[3], image_shape[1] - 1))
    if rounded:
        bbox = bbox.round().astype(int)
    return bbox


def ltrb_to_ltwh(bbox, image_shape=None, rounded=False):
    """
    Converts coordinates `[left, top, right, bottom]` to `[left, top, w, h]`.
    If image_shape is provided, the bbox is clipped to the image dimensions and its dimensions are ensured to be valid.
    """
    if image_shape:
        bbox = sanitize_bbox_ltrb(bbox, image_shape)
    bbox = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
    if rounded:
        bbox = bbox.round().astype(int)
    return bbox


class MMDetection(ImageLevelModule):
    collate_fn = default_collate
    input_columns = []
    output_columns = ['image_id', 'video_id', 'category_id', 'bbox_ltwh', 'bbox_conf']

    def __init__(self, config_name, path_to_checkpoint, device, batch_size, min_confidence, **kwargs):
        super().__init__(batch_size)
        self.device = device
        self.min_confidence = min_confidence
        model_df = get_model_info(package='mmdet', configs=[config_name])
        if len(model_df) != 1:
            raise ValueError(f'Multiple values found for config_name: {config_name}')
        download_url = model_df.weight.item()
        package_path = Path(get_installed_path('mmdet'))
        path_to_config = package_path / '.mim' / model_df.config.item()
        get_checkpoint(path_to_checkpoint, download_url)
        self.model = init_detector(str(path_to_config), path_to_checkpoint, device=device)
        self.test_pipeline = get_test_pipeline_cfg(self.model.cfg.copy())
        self.test_pipeline[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(self.test_pipeline)
        self.current_id = 0

    @torch.no_grad()
    def preprocess(self, image, detections: 'pd.DataFrame', metadata: 'pd.Series') ->Any:
        return self.test_pipeline(dict(img=image, img_id=0))

    @torch.no_grad()
    def process(self, batch: 'Any', detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        results = self.model.test_step(batch)
        img_metas = batch['data_samples']
        shapes = [(x.ori_shape[1], x.ori_shape[0]) for x in batch['data_samples']]
        detections = []
        for preds, image_shape, (_, metadata) in zip(results, shapes, metadatas.iterrows()):
            instances = preds.pred_instances
            for score, bbox, label in zip(instances.scores, instances.bboxes, instances.labels):
                if score < self.min_confidence or label != 0:
                    continue
                detections.append(pd.Series(dict(image_id=metadata.name, video_id=metadata.video_id, bbox_ltwh=ltrb_to_ltwh(bbox.cpu().numpy(), image_shape), bbox_conf=float(score.item()), category_id=1), name=self.current_id))
                self.current_id += 1
        return pd.DataFrame(detections)


def sanitize_bbox_ltwh(bbox: 'np.array', image_shape=None, rounded=False):
    """
    Sanitizes a bounding box by clipping it to the image dimensions and ensuring that its dimensions are valid.

    Args:
        bbox (np.ndarray): A numpy array of shape (4,) representing the bounding box in the format
        `[left, top, width, height]`.
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.
        rounded (bool): Whether to round the bounding box coordinates, type becomes int.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the sanitized bounding box in the format
        `[left, top, width, height]`.
    """
    assert isinstance(bbox, np.ndarray), f'Expected bbox to be of type np.ndarray, got {type(bbox)}'
    assert bbox.shape == (4,), f'Expected bbox to be of shape (4,), got {bbox.shape}'
    if image_shape is not None:
        bbox[0] = max(0, min(bbox[0], image_shape[0] - 2))
        bbox[1] = max(0, min(bbox[1], image_shape[1] - 2))
        bbox[2] = max(1, min(bbox[2], image_shape[0] - 1 - bbox[0]))
        bbox[3] = max(1, min(bbox[3], image_shape[1] - 1 - bbox[1]))
    if rounded:
        bbox = bbox.round().astype(int)
    return bbox


def sanitize_keypoints(keypoints, image_shape=None, rounded=False):
    """
    Sanitizes keypoints by clipping them to the image dimensions and ensuring that their confidence values are valid.

    Args:
        keypoints (np.ndarray): A numpy array of shape (K, 2 or 3) representing the keypoints in the format (x, y, (c)).
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.
        rounded (bool): Whether to round the keypoints to integers.

    Returns:
        np.ndarray: A numpy array of shape (K, 3) representing the sanitized keypoints in the format (x, y, (c)).
    """
    assert isinstance(keypoints, np.ndarray), 'Keypoints must be a numpy array.'
    assert keypoints.ndim == 2 and keypoints.shape[1] in (2, 3), 'Keypoints must be a numpy array of shape (K, 2 or 3).'
    if image_shape is not None:
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, image_shape[0] - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, image_shape[1] - 1)
    if rounded:
        keypoints[:, :2] = np.round(keypoints[:, :2]).astype(int)
    return keypoints


def generate_bbox_from_keypoints(keypoints, extension_factor, image_shape=None):
    """
    Generates a bounding box from keypoints by computing the bounding box of the keypoints and extending it by a factor.

    Args:
        keypoints (np.ndarray): A numpy array of shape (K, 3) representing the keypoints in the format (x, y, c).
        extension_factor (tuple): A tuple of float [top, bottom, right&left] representing the factor by which
        the bounding box should be extended based on the keypoints.
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the bounding box in the format (left, top, w, h).
    """
    keypoints = sanitize_keypoints(keypoints, image_shape)
    keypoints = keypoints[keypoints[:, 2] > 0]
    lt, rb = np.min(keypoints[:, :2], axis=0), np.max(keypoints[:, :2], axis=0)
    w, h = rb - lt
    lt -= np.array([extension_factor[2] * w, extension_factor[0] * h])
    rb += np.array([extension_factor[2] * w, extension_factor[1] * h])
    bbox = np.concatenate([lt, rb - lt])
    bbox = sanitize_bbox_ltwh(bbox, image_shape)
    return bbox


def mmpose_collate(batch):
    return collate(batch, len(batch))


@torch.no_grad()
class BottomUpMMPose(ImageLevelModule):
    collate_fn = mmpose_collate
    output_columns = ['image_id', 'video_id', 'category_id', 'bbox_ltwh', 'bbox_conf', 'keypoints_xyc', 'keypoints_conf']

    def __init__(self, cfg, device, batch_size):
        super().__init__(batch_size)
        get_checkpoint(cfg.path_to_checkpoint, cfg.download_url)
        self.device = device if device != 'cpu' else -1
        self.model = init_pose_model(cfg.path_to_config, cfg.path_to_checkpoint, device)
        self.id = 0
        self.cfg = self.model.cfg
        self.dataset_info = DatasetInfo(self.cfg.dataset_info)
        self.test_pipeline = Compose(self.cfg.test_pipeline)

    @torch.no_grad()
    def preprocess(self, metadata: 'pd.Series'):
        image = cv2.imread(metadata.file_path)
        data = {'dataset': self.dataset_info.dataset_name, 'img': image, 'ann_info': {'image_size': np.array(self.model.cfg.data_cfg['image_size']), 'heatmap_size': self.model.cfg.data_cfg.get('heatmap_size', None), 'num_joints': self.model.cfg.data_cfg['num_joints'], 'flip_index': self.dataset_info.flip_index, 'skeleton': self.dataset_info.skeleton}}
        return self.test_pipeline(data)

    @torch.no_grad()
    def process(self, batch, metadatas: 'pd.DataFrame'):
        batch = scatter(batch, [self.device])[0]
        images = list(batch['img'].unsqueeze(0).permute(1, 0, 2, 3, 4))
        detections = []
        for image, img_metas, (_, metadata) in zip(images, batch['img_metas'], metadatas.iterrows()):
            result = self.model(img=image, img_metas=[img_metas], return_loss=False, return_heatmap=False)
            pose_results = []
            for idx, pred in enumerate(result['preds']):
                area = (np.max(pred[:, 0]) - np.min(pred[:, 0])) * (np.max(pred[:, 1]) - np.min(pred[:, 1]))
                pose_results.append({'keypoints': pred[:, :3], 'score': result['scores'][idx], 'area': area})
            score_per_joint = self.model.cfg.model.test_cfg.get('score_per_joint', False)
            keep = oks_nms(pose_results, self.cfg.nms_threshold, self.dataset_info.sigmas, score_per_joint=score_per_joint)
            pose_results = [pose_results[_keep] for _keep in keep]
            for pose in pose_results:
                if pose['score'] >= self.cfg.min_confidence:
                    image_shape = image.shape[2], image.shape[1]
                    keypoints = sanitize_keypoints(pose['keypoints'], image_shape)
                    bbox = generate_bbox_from_keypoints(keypoints, self.cfg.bbox.extension_factor, image_shape)
                    detections.append(pd.Series(dict(image_id=metadata.name, keypoints_xyc=keypoints, keypoints_conf=pose['score'], bbox_ltwh=bbox, bbox_conf=pose['score'], video_id=metadata.video_id, category_id=1), name=self.id))
                    self.id += 1
        return detections


def collate_images_anns_meta(batch):
    idxs = [b[0] for b in batch]
    batch = [b[1] for b in batch]
    anns = [b[-2] for b in batch]
    metas = [b[-1] for b in batch]
    processed_images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    idxs = torch.utils.data.dataloader.default_collate(idxs)
    return idxs, (processed_images, anns, metas)


logger = logging.getLogger(__name__)


def tracklab_cli():
    parser = argparse.ArgumentParser(prog='python3 -m openpifpaf.predict', usage='%(prog)s [options] images', description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version', action='version', version='OpenPifPaf {version}'.format(version=openpifpaf.__version__))
    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    openpifpaf.Predictor.cli(parser)
    parser.add_argument('--json-output', default=None, nargs='?', const=True, help='Whether to output a json file, with the option to specify the output path or directory')
    parser.add_argument('--disable-cuda', action='store_true', help='disable CUDA')
    args = parser.parse_args()
    logger.configure(args, log)
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    log.info('neural network device: %s (CUDA available: %s, count: %d)', args.device, torch.cuda.is_available(), torch.cuda.device_count())
    decoder.configure(args)
    network.Factory.configure(args)
    openpifpaf.Predictor.configure(args)
    return args


class OpenPifPaf(ImageLevelModule):
    collate_fn = collate_images_anns_meta
    input_columns = []
    output_columns = ['image_id', 'id', 'video_id', 'category_id', 'bbox_ltwh', 'bbox_conf', 'keypoints_xyc', 'keypoints_conf']

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.id = 0
        if cfg.predict.checkpoint:
            old_argv = sys.argv
            sys.argv = self._hydra_to_argv(cfg.predict)
            tracklab_cli()
            predictor = openpifpaf.Predictor()
            sys.argv = old_argv
            self.model = predictor.model
            self.pifpaf_preprocess = predictor.preprocess
            self.processor = predictor.processor
            log.info(f'Loaded detection model from checkpoint: {cfg.predict.checkpoint}')

    @torch.no_grad()
    def preprocess(self, image, detections: 'pd.DataFrame', metadata: 'pd.Series'):
        image = Image.fromarray(image)
        processed_image, anns, meta = self.pifpaf_preprocess(image, [], {})
        return processed_image, anns, meta

    @torch.no_grad()
    def process(self, batch, detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        processed_image_batch, _, metas = batch
        pred_batch = self.processor.batch(self.model, processed_image_batch, device=self.device)
        detections = []
        for predictions, meta, (_, metadata) in zip(pred_batch, metas, metadatas.iterrows()):
            for prediction in predictions:
                prediction = prediction.inverse_transform(meta)
                keypoints = sanitize_keypoints(prediction.data, meta['width_height'])
                bbox = generate_bbox_from_keypoints(keypoints[keypoints[:, 2] > 0], self.cfg.bbox.extension_factor, meta['width_height'])
                detections.append(pd.Series(dict(image_id=metadata.name, keypoints_xyc=keypoints, keypoints_conf=prediction.score, bbox_ltwh=bbox, bbox_conf=prediction.score, video_id=metadata.video_id, category_id=1), name=self.id))
                self.id += 1
        return detections

    def train(self):
        old_argv = sys.argv
        sys.argv = self._hydra_to_argv(self.cfg.train)
        log.info(f'Starting training of the detection model')
        self.cfg.predict.checkpoint = openpifpaf.train.main()
        sys.argv = self._hydra_to_argv(self.cfg.predict)
        openpifpaf.predict.tracklab_cli()
        predictor = openpifpaf.Predictor()
        sys.argv = old_argv
        self.model = predictor.model
        self.pifpaf_preprocess = predictor.preprocess
        self.processor = predictor.processor
        log.info(f'Loaded trained detection model from file: {self.cfg.predict.checkpoint}')

    @staticmethod
    def _hydra_to_argv(cfg):
        new_argv = ['argv_from_hydra']
        for k, v in cfg.items():
            new_arg = f'--{str(k)}'
            if isinstance(v, ListConfig):
                new_argv.append(new_arg)
                for item in v:
                    new_argv.append(f'{str(item)}')
            elif v is not None:
                new_arg += f'={str(v)}'
                new_argv.append(new_arg)
            else:
                new_argv.append(new_arg)
        return new_argv


class TopDownMMPose(DetectionLevelModule):
    collate_fn = default_collate
    input_columns = ['bbox_ltwh', 'bbox_conf']
    output_columns = ['keypoints_xyc', 'keypoints_conf']

    def __init__(self, device, batch_size, config_name, path_to_checkpoint, vis_kp_threshold=0.4, min_num_vis_kp=3, **kwargs):
        super().__init__(batch_size)
        model_df = get_model_info(package='mmpose', configs=[config_name])
        if len(model_df) != 1:
            raise ValueError('Multiple values found for the config name')
        download_url = model_df.weight.item()
        package_path = Path(get_installed_path('mmpose'))
        path_to_config = package_path / '.mim' / model_df.config.item()
        get_checkpoint(path_to_checkpoint, download_url)
        self.model = init_model(str(path_to_config), path_to_checkpoint, device)
        self.vis_kp_threshold = vis_kp_threshold
        self.min_num_vis_kp = min_num_vis_kp
        self.dataset_info = dataset_meta_from_config(self.model.cfg, 'test')
        self.test_pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)

    @torch.no_grad()
    def preprocess(self, image, detection: 'pd.Series', metadata: 'pd.Series'):
        data_info = dict(img=cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        data_info['bbox'] = detection.bbox.ltrb()[None]
        data_info['bbox_score'] = detection.bbox_conf[None]
        data_info.update(self.model.dataset_meta)
        return self.test_pipeline(data_info)

    @torch.no_grad()
    def process(self, batch, detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        results = self.model.test_step(batch)
        kps_xyc = []
        kps_conf = []
        for result in results:
            result = result.pred_instances
            keypoints = result.keypoints[0]
            visibility_scores = result.keypoints_visible[0]
            visibility_scores[visibility_scores < self.vis_kp_threshold] = 0
            keypoints_xyc = np.concatenate([keypoints, visibility_scores[:, None]], axis=-1)
            if len(np.nonzero(visibility_scores)[0]) < self.min_num_vis_kp:
                conf = 0
            else:
                conf = np.mean(visibility_scores[visibility_scores != 0])
            kps_xyc.append(keypoints_xyc)
            kps_conf.append(conf)
        detections['keypoints_conf'] = kps_conf
        detections['keypoints_xyc'] = kps_xyc
        return detections


def rescale_keypoints(rf_keypoints, size, new_size):
    """
    Rescale keypoints to new size.
    Args:
        rf_keypoints (np.ndarray): keypoints in relative coordinates, shape (K, 2)
        size (tuple): original size, (w, h)
        new_size (tuple): new size, (w, h)
    Returns:
        rf_keypoints (np.ndarray): rescaled keypoints in relative coordinates, shape (K, 2)
    """
    w, h = size
    new_w, new_h = new_size
    rf_keypoints = rf_keypoints.copy()
    rf_keypoints[..., 0] = rf_keypoints[..., 0] * new_w / w
    rf_keypoints[..., 1] = rf_keypoints[..., 1] * new_h / h
    assert ((rf_keypoints[..., 0] >= 0) & (rf_keypoints[..., 0] <= new_w)).all()
    assert ((rf_keypoints[..., 1] >= 0) & (rf_keypoints[..., 1] <= new_h)).all()
    return rf_keypoints


class Unbatchable(tuple):
    pass


def check_md5(local_filename, md5):
    with open(local_filename, 'rb') as f:
        file_hash = hashlib.md5()
        while (chunk := f.read(8192)):
            file_hash.update(chunk)
    return file_hash.hexdigest() == md5


def download_file(url, local_filename, md5=None):
    if Path(local_filename).exists():
        if md5 is not None:
            if check_md5(local_filename, md5):
                return local_filename
            else:
                raise ValueError(f'MD5 checksum mismatch for file {local_filename}, please re-download it from {url}')
    Path(local_filename).parent.mkdir(exist_ok=True, parents=True)
    file_hash = hashlib.md5()
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        chunk_size = 8192
        with open(local_filename, 'wb') as f, tqdm(desc=f'Downloading {Path(local_filename).name}', total=total_size, unit='B', unit_scale=True) as progress_bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                file_hash.update(chunk)
                f.write(chunk)
                progress_bar.update(len(chunk))
    if md5 is not None:
        if md5 != file_hash.hexdigest():
            raise ValueError(f'MD5 checksum mismatch when downloading file from {url}. Please download it manually from {url} to {local_filename}.')
    return local_filename


class BPBReId(DetectionLevelModule):
    """
    """
    collate_fn = default_collate
    input_columns = ['bbox_ltwh']
    output_columns = ['embeddings', 'visibility_scores', 'body_masks']

    def __init__(self, cfg, tracking_dataset, dataset, device, save_path, job_id, use_keypoints_visibility_scores_for_reid, training_enabled, batch_size):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        tracking_dataset.name = dataset.name
        tracking_dataset.nickname = dataset.nickname
        self.dataset_cfg = dataset
        self.use_keypoints_visibility_scores_for_reid = use_keypoints_visibility_scores_for_reid
        tracking_dataset.name = self.dataset_cfg.name
        tracking_dataset.nickname = self.dataset_cfg.nickname
        additional_args = {'tracking_dataset': tracking_dataset, 'reid_config': self.dataset_cfg, 'pose_model': None}
        torchreid.data.register_image_dataset(tracking_dataset.name, configure_dataset_class(ReidDataset, **additional_args), tracking_dataset.nickname)
        self.cfg = CN(OmegaConf.to_container(cfg, resolve=True))
        self.download_models(load_weights=self.cfg.model.load_weights, pretrained_path=self.cfg.model.bpbreid.hrnet_pretrained_path, backbone=self.cfg.model.bpbreid.backbone)
        self.cfg.data.save_dir = save_path
        self.cfg.project.job_id = job_id
        self.cfg.use_gpu = torch.cuda.is_available()
        self.cfg = build_config(config=self.cfg)
        self.test_embeddings = self.cfg.model.bpbreid.test_embeddings
        self.training_enabled = training_enabled
        self.feature_extractor = None
        self.model = None

    def download_models(self, load_weights, pretrained_path, backbone):
        if Path(load_weights).stem == 'bpbreid_market1501_hrnet32_10642':
            md5 = 'e79262f17e7486ece33eebe198c07841'
            download_file('https://zenodo.org/records/10604211/files/bpbreid_market1501_hrnet32_10642.pth?download=1', local_filename=load_weights, md5=md5)
        if backbone == 'hrnet32':
            md5 = '58ea12b0420aa3adaa2f74114c9f9721'
            path = Path(pretrained_path) / 'hrnetv2_w32_imagenet_pretrained.pth'
            download_file('https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1', local_filename=path, md5=md5)

    @torch.no_grad()
    def preprocess(self, image, detection: 'pd.Series', metadata: 'pd.Series'):
        mask_w, mask_h = 32, 64
        l, t, r, b = detection.bbox.ltrb(image_shape=(image.shape[1], image.shape[0]), rounded=True)
        crop = image[t:b, l:r]
        crop = Unbatchable([crop])
        batch = {'img': crop}
        if not self.cfg.model.bpbreid.learnable_attention_enabled:
            bbox_ltwh = detection.bbox.ltwh(image_shape=(image.shape[1], image.shape[0]), rounded=True)
            kp_xyc_bbox = detection.keypoints.in_bbox_coord(bbox_ltwh)
            kp_xyc_mask = rescale_keypoints(kp_xyc_bbox, (bbox_ltwh[2], bbox_ltwh[3]), (mask_w, mask_h))
            if self.dataset_cfg.masks_mode == 'gaussian_keypoints':
                pixels_parts_probabilities = build_gaussian_heatmaps(kp_xyc_mask, mask_w, mask_h)
            else:
                raise NotImplementedError
            batch['masks'] = pixels_parts_probabilities
        return batch

    @torch.no_grad()
    def process(self, batch, detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        im_crops = batch['img']
        im_crops = [im_crop.cpu().detach().numpy() for im_crop in im_crops]
        if 'masks' in batch:
            external_parts_masks = batch['masks']
            external_parts_masks = external_parts_masks.cpu().detach().numpy()
        else:
            external_parts_masks = None
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(self.cfg, model_path=self.cfg.model.load_weights, device=self.device, image_size=(self.cfg.data.height, self.cfg.data.width), model=self.model, verbose=False)
        reid_result = self.feature_extractor(im_crops, external_parts_masks=external_parts_masks)
        embeddings, visibility_scores, body_masks, _ = extract_test_embeddings(reid_result, self.test_embeddings)
        embeddings = embeddings.cpu().detach().numpy()
        visibility_scores = visibility_scores.cpu().detach().numpy()
        body_masks = body_masks.cpu().detach().numpy()
        if self.use_keypoints_visibility_scores_for_reid:
            kp_visibility_scores = batch['visibility_scores'].numpy()
            if visibility_scores.shape[1] > kp_visibility_scores.shape[1]:
                kp_visibility_scores = np.concatenate([np.ones((visibility_scores.shape[0], 1)), kp_visibility_scores], axis=1)
            visibility_scores = np.float32(kp_visibility_scores)
        reid_df = pd.DataFrame({'embeddings': list(embeddings), 'visibility_scores': list(visibility_scores), 'body_masks': list(body_masks)}, index=detections.index)
        return reid_df

    def train(self):
        self.engine, self.model = build_torchreid_model_engine(self.cfg)
        self.engine.run(**engine_run_kwargs(self.cfg))


class BotSORT(ImageLevelModule):
    input_columns = ['bbox_ltwh', 'bbox_conf', 'category_id']
    output_columns = ['track_id', 'track_bbox_ltwh', 'track_bbox_conf']

    def __init__(self, cfg, device, **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = bot_sort.BoTSORT(Path(self.cfg.model_weights), self.device, self.cfg.fp16, **self.cfg.hyperparams)

    @torch.no_grad()
    def preprocess(self, image, detections: 'pd.DataFrame', metadata: 'pd.Series'):
        processed_detections = []
        if len(detections) == 0:
            return {'input': []}
        for det_id, detection in detections.iterrows():
            ltrb = detection.bbox.ltrb()
            conf = detection.bbox.conf()
            cls = detection.category_id
            tracklab_id = int(detection.name)
            processed_detections.append(np.array([*ltrb, conf, cls, tracklab_id]))
        return {'input': np.stack(processed_detections)}

    @torch.no_grad()
    def process(self, batch, detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        if len(detections) == 0:
            return []
        inputs = batch['input'][0]
        inputs = inputs[inputs[:, 4] > self.cfg.min_confidence]
        image = cv2_load_image(metadatas['file_path'].values[0])
        results = self.model.update(inputs, image)
        results = np.asarray(results)
        if results.size:
            track_bbox_ltwh = [ltrb_to_ltwh(x) for x in results[:, :4]]
            track_bbox_conf = list(results[:, 6])
            track_ids = list(results[:, 4])
            idxs = list(results[:, 7].astype(int))
            assert set(idxs).issubset(detections.index), 'Mismatch of indexes during the tracking. The results should match the detections.'
            results = pd.DataFrame({'track_bbox_ltwh': track_bbox_ltwh, 'track_bbox_conf': track_bbox_conf, 'track_id': track_ids, 'idxs': idxs})
            results.set_index('idxs', inplace=True, drop=True)
            return results
        else:
            return []


class BPBReIDStrongSORT(ImageLevelModule):
    input_columns = ['bbox_ltwh', 'embeddings', 'visibility_scores']
    output_columns = ['track_id', 'track_bbox_kf_ltwh', 'track_bbox_pred_kf_ltwh', 'matched_with', 'costs', 'hits', 'age', 'time_since_update', 'state']

    def __init__(self, cfg, device, batch_size=None, **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = strong_sort.StrongSORT(ema_alpha=self.cfg.ema_alpha, mc_lambda=self.cfg.mc_lambda, max_dist=self.cfg.max_dist, motion_criterium=self.cfg.motion_criterium, max_iou_distance=self.cfg.max_iou_distance, max_oks_distance=self.cfg.max_oks_distance, max_age=self.cfg.max_age, n_init=self.cfg.n_init, nn_budget=self.cfg.nn_budget, min_bbox_confidence=self.cfg.min_bbox_confidence, only_position_for_kf_gating=self.cfg.only_position_for_kf_gating, max_kalman_prediction_without_update=self.cfg.max_kalman_prediction_without_update, matching_strategy=self.cfg.matching_strategy, gating_thres_factor=self.cfg.gating_thres_factor, w_kfgd=self.cfg.w_kfgd, w_reid=self.cfg.w_reid, w_st=self.cfg.w_st)
        self.prev_frame = None

    def prepare_next_frame(self, next_frame: 'np.ndarray'):
        self.model.tracker.predict()
        if self.cfg.ecc:
            if self.prev_frame is not None:
                self.model.tracker.camera_update(self.prev_frame, next_frame)
            self.prev_frame = next_frame

    @torch.no_grad()
    def preprocess(self, image, detections: 'pd.DataFrame', metadata: 'pd.Series'):
        if len(detections) == 0:
            return {'id': [], 'bbox_ltwh': [], 'reid_features': [], 'visibility_scores': [], 'scores': [], 'classes': [], 'frame': []}
        if hasattr(detections, 'bbox_conf'):
            score = detections.bbox.conf()
        else:
            score = detections.keypoints_conf
        input_tuple = {'id': detections.index.to_numpy(), 'bbox_ltwh': np.stack(detections.bbox_ltwh), 'reid_features': np.stack(detections.embeddings), 'visibility_scores': np.stack(detections.visibility_scores), 'scores': np.stack(score), 'classes': np.zeros(len(detections.index)), 'frame': np.ones(len(detections.index)) * metadata.frame}
        if 'keypoints_xyc' in detections:
            input_tuple['keypoints'] = np.stack(detections.keypoints_xyc)
        return input_tuple

    @torch.no_grad()
    def process(self, batch, detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        if len(detections) == 0:
            return []
        results = self.model.update(batch['id'][0], batch['bbox_ltwh'][0], batch['reid_features'][0], batch['visibility_scores'][0], batch['scores'][0], batch['classes'][0], batch['frame'][0], batch['keypoints'][0] if 'keypoints' in batch else None)
        assert set(results.index).issubset(detections.index), 'Mismatch of indexes during the tracking. The results should match the detections.'
        return results


class ByteTrack(ImageLevelModule):
    input_columns = ['bbox_ltwh', 'bbox_conf', 'category_id']
    output_columns = ['track_id', 'track_bbox_ltwh', 'track_bbox_conf']

    def __init__(self, cfg, device, **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = byte_tracker.BYTETracker(**self.cfg.hyperparams)

    @torch.no_grad()
    def preprocess(self, image, detections: 'pd.DataFrame', metadata: 'pd.Series'):
        processed_detections = []
        if len(detections) == 0:
            return {'input': []}
        for det_id, detection in detections.iterrows():
            ltrb = detection.bbox.ltrb()
            conf = detection.bbox.conf()
            cls = detection.category_id
            tracklab_id = int(detection.name)
            processed_detections.append(np.array([*ltrb, conf, cls, tracklab_id]))
        return {'input': np.stack(processed_detections)}

    @torch.no_grad()
    def process(self, batch, detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        if len(detections) == 0:
            return []
        inputs = batch['input'][0]
        inputs = inputs[inputs[:, 4] > self.cfg.min_confidence]
        results = self.model.update(inputs, None)
        results = np.asarray(results)
        if results.size:
            track_bbox_ltwh = [ltrb_to_ltwh(x) for x in results[:, :4]]
            track_bbox_conf = list(results[:, 6])
            track_ids = list(results[:, 4])
            idxs = list(results[:, 7].astype(int))
            assert set(idxs).issubset(detections.index), 'Mismatch of indexes during the tracking. The results should match the detections.'
            results = pd.DataFrame({'track_bbox_ltwh': track_bbox_ltwh, 'track_bbox_conf': track_bbox_conf, 'track_id': track_ids, 'idxs': idxs})
            results.set_index('idxs', inplace=True, drop=True)
            return results
        else:
            return []


class DeepOCSORT(ImageLevelModule):
    input_columns = ['bbox_ltwh', 'bbox_conf', 'category_id']
    output_columns = ['track_id', 'track_bbox_ltwh', 'track_bbox_conf']

    def __init__(self, cfg, device, **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = ocsort.OCSort(Path(self.cfg.model_weights), self.device, self.cfg.fp16, **self.cfg.hyperparams)

    @torch.no_grad()
    def preprocess(self, image, detections: 'pd.DataFrame', metadata: 'pd.Series'):
        processed_detections = []
        if len(detections) == 0:
            return {'input': []}
        for det_id, detection in detections.iterrows():
            ltrb = detection.bbox.ltrb()
            conf = detection.bbox.conf()
            cls = detection.category_id
            tracklab_id = int(detection.name)
            processed_detections.append(np.array([*ltrb, conf, cls, tracklab_id]))
        return {'input': np.stack(processed_detections)}

    @torch.no_grad()
    def process(self, batch, detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        if len(detections) == 0:
            return []
        inputs = batch['input'][0]
        inputs = inputs[inputs[:, 4] > self.cfg.min_confidence]
        image = cv2_load_image(metadatas['file_path'].values[0])
        results = self.model.update(inputs, image)
        results = np.asarray(results)
        if results.size:
            track_bbox_ltwh = [ltrb_to_ltwh(x) for x in results[:, :4]]
            track_bbox_conf = list(results[:, 6])
            track_ids = list(results[:, 4])
            idxs = list(results[:, 7].astype(int))
            assert set(idxs).issubset(detections.index), 'Mismatch of indexes during the tracking. The results should match the detections.'
            results = pd.DataFrame({'track_bbox_ltwh': track_bbox_ltwh, 'track_bbox_conf': track_bbox_conf, 'track_id': track_ids, 'idxs': idxs})
            results.set_index('idxs', inplace=True, drop=True)
            return results
        else:
            return []


class OCSORT(ImageLevelModule):
    input_columns = ['bbox_ltwh', 'bbox_conf', 'category_id']
    output_columns = ['track_id', 'track_bbox_ltwh', 'track_bbox_conf']

    def __init__(self, cfg, device, **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = ocsort.OCSort(**self.cfg.hyperparams)

    @torch.no_grad()
    def preprocess(self, image, detections: 'pd.DataFrame', metadata: 'pd.Series'):
        processed_detections = []
        if len(detections) == 0:
            return {'input': []}
        for det_id, detection in detections.iterrows():
            ltrb = detection.bbox.ltrb()
            conf = detection.bbox.conf()
            cls = detection.category_id
            tracklab_id = int(detection.name)
            processed_detections.append(np.array([*ltrb, conf, cls, tracklab_id]))
        return {'input': np.stack(processed_detections)}

    @torch.no_grad()
    def process(self, batch, detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        if len(detections) == 0:
            return []
        inputs = batch['input'][0]
        inputs = inputs[inputs[:, 4] > self.cfg.min_confidence]
        results = self.model.update(inputs, None)
        results = np.asarray(results)
        if results.size:
            track_bbox_ltwh = [ltrb_to_ltwh(x) for x in results[:, :4]]
            track_bbox_conf = list(results[:, 6])
            track_ids = list(results[:, 4])
            idxs = list(results[:, 7].astype(int))
            assert set(idxs).issubset(detections.index), 'Mismatch of indexes during the tracking. The results should match the detections.'
            results = pd.DataFrame({'track_bbox_ltwh': track_bbox_ltwh, 'track_bbox_conf': track_bbox_conf, 'track_id': track_ids, 'idxs': idxs})
            results.set_index('idxs', inplace=True, drop=True)
            return results
        else:
            return []


class StrongSORT(ImageLevelModule):
    input_columns = ['bbox_ltwh', 'bbox_conf', 'category_id']
    output_columns = ['track_id', 'track_bbox_ltwh', 'track_bbox_conf']

    def __init__(self, cfg, device, **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = strong_sort.StrongSORT(Path(self.cfg.model_weights), self.device, self.cfg.fp16, **self.cfg.hyperparams)
        self.prev_frame = None

    @torch.no_grad()
    def preprocess(self, image, detections: 'pd.DataFrame', metadata: 'pd.Series'):
        processed_detections = []
        if len(detections) == 0:
            return {'input': []}
        for det_id, detection in detections.iterrows():
            ltrb = detection.bbox.ltrb()
            conf = detection.bbox.conf()
            cls = detection.category_id
            tracklab_id = int(detection.name)
            processed_detections.append(np.array([*ltrb, conf, cls, tracklab_id]))
        return {'input': np.stack(processed_detections)}

    @torch.no_grad()
    def process(self, batch, detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        image = cv2_load_image(metadatas['file_path'].values[0])
        if self.cfg.ecc:
            if self.prev_frame is not None:
                self.model.tracker.camera_update(self.prev_frame, image)
            self.prev_frame = image
        if len(detections) == 0:
            return []
        inputs = batch['input'][0]
        inputs = inputs[inputs[:, 4] > self.cfg.min_confidence]
        results = self.model.update(inputs, image)
        results = np.asarray(results)
        if results.size:
            track_bbox_ltwh = [ltrb_to_ltwh(x) for x in results[:, :4]]
            track_bbox_conf = list(results[:, 6])
            track_ids = list(results[:, 4])
            idxs = list(results[:, 8].astype(int))
            results = pd.DataFrame({'track_bbox_ltwh': track_bbox_ltwh, 'track_bbox_conf': track_bbox_conf, 'track_id': track_ids, 'idxs': idxs})
            results.set_index('idxs', inplace=True, drop=True)
            return results
        else:
            return []


def select_highest_voted_att(atts, atts_confidences=None):
    confidence_sum = {}
    atts_confidences = [1] * len(atts) if atts_confidences is None else atts_confidences
    for jn, conf in zip(atts, atts_confidences):
        if jn not in confidence_sum:
            confidence_sum[jn] = 0
        confidence_sum[jn] += conf
    if len(confidence_sum) == 0:
        return None
    max_confidence_att = max(confidence_sum, key=confidence_sum.get)
    return max_confidence_att


class MajorityVoteTracklet(VideoLevelModule):
    input_columns = []
    output_columns = []

    def __init__(self, cfg, device, tracking_dataset=None):
        self.attributes = cfg.attributes
        for attribute in self.attributes:
            self.input_columns.append(f'{attribute}_detection')
            self.input_columns.append(f'{attribute}_confidence')
            self.output_columns.append(attribute)

    @torch.no_grad()
    def process(self, detections: 'pd.DataFrame', metadatas: 'pd.DataFrame'):
        detections[self.output_columns] = np.nan
        if 'track_id' not in detections.columns:
            return detections
        for track_id in detections.track_id.unique():
            tracklet = detections[detections.track_id == track_id]
            for attribute in self.attributes:
                attribute_detection = tracklet[f'{attribute}_detection']
                attribute_confidence = tracklet[f'{attribute}_confidence']
                attribute_value = [select_highest_voted_att(attribute_detection, attribute_confidence)] * len(tracklet)
                detections.loc[tracklet.index, attribute] = attribute_value
        return detections


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AvgPoolPad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_filters': 4, 'out_filters': 4, 'reps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block17,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (Block35,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Block8,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2080, 64, 64])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'stride': 1, 'num_groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BranchSeparables,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (BranchSeparablesReduction,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BranchSeparablesStem,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CellStem0,
     lambda: ([], {'stem_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CellStem1,
     lambda: ([], {'stem_filters': 4, 'num_filters': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 8, 64, 64])], {}),
     False),
    (ChannelShuffle,
     lambda: ([], {'num_groups': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv1x1,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv1x1Linear,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_c': 4, 'out_c': 4, 'k': 4, 's': 4, 'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvLayers,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (DenseNet,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DimReduceLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'nonlinear': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureScalerZScore,
     lambda: ([], {'loc': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fire,
     lambda: ([], {'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4, 'expand3x3_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardAttn,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (IBN,
     lambda: ([], {'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InceptionResNetV2,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     False),
    (InceptionV4,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     False),
    (Inception_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (Inception_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
    (Inception_C,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1536, 64, 64])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LightConv3x3,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LightConvStream,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'depth': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLFN,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (MLFNBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 64, 'stride': 64, 'fsm_channels': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPoolPad,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mixed_3a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (Mixed_4a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 160, 64, 64])], {}),
     True),
    (Mixed_5a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {}),
     True),
    (Mixed_5b,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 192, 64, 64])], {}),
     True),
    (Mixed_6a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 320, 64, 64])], {}),
     True),
    (Mixed_7a,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1088, 64, 64])], {}),
     True),
    (MultiScaleA,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 96, 64, 64])], {}),
     True),
    (MultiScaleB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 256, 64, 64])], {}),
     True),
    (NASNetAMobile,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (NormalCell,
     lambda: ([], {'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Reduction,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 96, 64, 64])], {}),
     True),
    (ReductionCell0,
     lambda: ([], {'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReductionCell1,
     lambda: ([], {'in_channels_left': 4, 'out_channels_left': 4, 'in_channels_right': 4, 'out_channels_right': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Reduction_A,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 384, 64, 64])], {}),
     True),
    (Reduction_B,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1024, 64, 64])], {}),
     True),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleNetV2,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss(), 'stages_repeats': [4, 4, 4], 'stages_out_channels': [4, 4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SqueezeNet,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Xception,
     lambda: ([], {'num_classes': 4, 'loss': MSELoss()}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'num_input_features': 4, 'bn_size': 4, 'growth_rate': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_DenseLayer,
     lambda: ([], {'num_input_features': 4, 'growth_rate': 4, 'bn_size': 4, 'drop_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_Transition,
     lambda: ([], {'num_input_features': 4, 'num_output_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_TrackingLaboratory_tracklab(_paritybench_base):
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

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

    def test_029(self):
        self._check(*TESTCASES[29])

    def test_030(self):
        self._check(*TESTCASES[30])

    def test_031(self):
        self._check(*TESTCASES[31])

    def test_032(self):
        self._check(*TESTCASES[32])

    def test_033(self):
        self._check(*TESTCASES[33])

    def test_034(self):
        self._check(*TESTCASES[34])

    def test_035(self):
        self._check(*TESTCASES[35])

    def test_036(self):
        self._check(*TESTCASES[36])

    def test_037(self):
        self._check(*TESTCASES[37])

    def test_038(self):
        self._check(*TESTCASES[38])

    def test_039(self):
        self._check(*TESTCASES[39])

    def test_040(self):
        self._check(*TESTCASES[40])

    def test_041(self):
        self._check(*TESTCASES[41])

    def test_042(self):
        self._check(*TESTCASES[42])

    def test_043(self):
        self._check(*TESTCASES[43])

    def test_044(self):
        self._check(*TESTCASES[44])

    def test_045(self):
        self._check(*TESTCASES[45])

    def test_046(self):
        self._check(*TESTCASES[46])

    def test_047(self):
        self._check(*TESTCASES[47])

    def test_048(self):
        self._check(*TESTCASES[48])

    def test_049(self):
        self._check(*TESTCASES[49])

    def test_050(self):
        self._check(*TESTCASES[50])

    def test_051(self):
        self._check(*TESTCASES[51])

    def test_052(self):
        self._check(*TESTCASES[52])

    def test_053(self):
        self._check(*TESTCASES[53])

    def test_054(self):
        self._check(*TESTCASES[54])

    def test_055(self):
        self._check(*TESTCASES[55])

    def test_056(self):
        self._check(*TESTCASES[56])

    def test_057(self):
        self._check(*TESTCASES[57])

    def test_058(self):
        self._check(*TESTCASES[58])

    def test_059(self):
        self._check(*TESTCASES[59])

