import sys
_module = sys.modules[__name__]
del sys
clean_balance_data = _module
construct_dataset_parallel = _module
convonet_setup = _module
eval_geometry_voxel = _module
generate_data_parallel = _module
render_rollouts_nvisii = _module
save_occ_data_parallel = _module
sim_grasp_multiple = _module
sim_grasp_save_scene = _module
train_giga = _module
train_giga_aff = _module
train_giga_geo = _module
train_vgn = _module
setup = _module
ConvONets = _module
checkpoints = _module
common = _module
config = _module
conv_onet = _module
config = _module
generation = _module
models = _module
decoder = _module
training = _module
data = _module
core = _module
fields = _module
transforms = _module
encoder = _module
pointnet = _module
pointnetpp = _module
unet = _module
unet3d = _module
voxels = _module
eval = _module
layers = _module
utils = _module
binvox_rw = _module
icp = _module
io = _module
libkdtree = _module
pykdtree = _module
render_template = _module
test_tree = _module
libmcubes = _module
exporter = _module
libmesh = _module
inside_mesh = _module
libmise = _module
test = _module
libsimplify = _module
libvoxelize = _module
mesh = _module
visualize = _module
vgn = _module
baselines = _module
dataset = _module
dataset_voxel = _module
dataset_voxel_occ = _module
detection = _module
detection_implicit = _module
experiments = _module
clutter_removal = _module
clutter_removal_single = _module
grasp = _module
networks = _module
perception = _module
simulation = _module
btsim = _module
implicit = _module
misc = _module
nvisii_render = _module
saver = _module
transform = _module
visual = _module

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


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CppExtension


from torch.utils.cpp_extension import CUDAExtension


import numpy


import numpy as np


import torch


from torch.utils.data.dataloader import default_collate


from torch.utils import tensorboard


import torch.nn.functional as F


from torch.utils import model_zoo


import math


from torchvision import transforms


import torch.distributions as dist


from torch import nn


import torch.optim as optim


from torch import autograd


import time


import torch.nn as nn


from torch import distributions as dist


from torch.nn import functional as F


import logging


from torch.utils import data


from time import time


from torch.autograd import Variable


from collections import OrderedDict


from torch.nn import init


from functools import partial


from scipy import ndimage


import torch.utils.data


import matplotlib.pyplot as plt


from copy import deepcopy


import random


class ConvolutionalOccupancyNetwork(nn.Module):
    """ Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    """

    def __init__(self, decoders, encoder=None, device=None, detach_tsdf=False):
        super().__init__()
        self.decoder_qual = decoders[0]
        self.decoder_rot = decoders[1]
        self.decoder_width = decoders[2]
        if len(decoders) == 4:
            self.decoder_tsdf = decoders[3]
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None
        self._device = device
        self.detach_tsdf = detach_tsdf

    def forward(self, inputs, p, p_tsdf=None, sample=True, **kwargs):
        """ Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        """
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        qual, rot, width = self.decode(p, c)
        if p_tsdf is not None:
            if self.detach_tsdf:
                for k, v in c.items():
                    c[k] = v.detach()
            tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
            return qual, rot, width, tsdf
        else:
            return qual, rot, width

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def encode_inputs(self, inputs):
        """ Encodes the input.

        Args:
            input (tensor): the input
        """
        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            c = torch.empty(inputs.size(0), 0)
        return c

    def query_feature(self, p, c):
        return self.decoder_qual.query_feature(p, c)

    def decode_feature(self, p, feature):
        qual = self.decoder_qual.compute_out(p, feature)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot.compute_out(p, feature)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width.compute_out(p, feature)
        return qual, rot, width

    def decode_occ(self, p, c, **kwargs):
        """ Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """
        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r

    def decode(self, p, c, **kwargs):
        """ Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """
        qual = self.decoder_qual(p, c, **kwargs)
        qual = torch.sigmoid(qual)
        rot = self.decoder_rot(p, c, **kwargs)
        rot = nn.functional.normalize(rot, dim=2)
        width = self.decoder_width(p, c, **kwargs)
        return qual, rot, width

    def to(self, device):
        """ Puts the model to the device.

        Args:
            device (device): pytorch device
        """
        model = super()
        model._device = device
        return model

    def grad_refine(self, x, pos, bound_value=0.0125, lr=1e-06, num_step=1):
        pos_tmp = pos.clone()
        l_bound = pos - bound_value
        u_bound = pos + bound_value
        pos_tmp.requires_grad = True
        optimizer = torch.optim.SGD([pos_tmp], lr=lr)
        self.eval()
        for p in self.parameters():
            p.requres_grad = False
        for _ in range(num_step):
            optimizer.zero_grad()
            qual_out, _, _ = self.forward(x, pos_tmp)
            loss = -qual_out.sum()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pos_tmp = torch.maximum(torch.minimum(pos_tmp, u_bound), l_bound)
            qual_out, rot_out, width_out = self.forward(x, pos_tmp)
        for p in self.parameters():
            p.requres_grad = True
        return qual_out, pos_tmp, rot_out, width_out


class ConvolutionalOccupancyNetworkGeometry(nn.Module):

    def __init__(self, decoder, encoder=None, device=None):
        super().__init__()
        self.decoder_tsdf = decoder
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = None
        self._device = device

    def forward(self, inputs, p, p_tsdf, sample=True, **kwargs):
        """ Performs a forward pass through the network.

        Args:
            p (tensor): sampled points, B*N*C
            inputs (tensor): conditioning input, B*N*3
            sample (bool): whether to sample for z
            p_tsdf (tensor): tsdf query points, B*N_P*3
        """
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def infer_geo(self, inputs, p_tsdf, **kwargs):
        c = self.encode_inputs(inputs)
        tsdf = self.decoder_tsdf(p_tsdf, c, **kwargs)
        return tsdf

    def encode_inputs(self, inputs):
        """ Encodes the input.

        Args:
            input (tensor): the input
        """
        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            c = torch.empty(inputs.size(0), 0)
        return c

    def decode_occ(self, p, c, **kwargs):
        """ Returns occupancy probabilities for the sampled points.
        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        """
        logits = self.decoder_tsdf(p, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r


def normalize_3d_coordinate(p, padding=0.1):
    """ Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """
    p_nor = p / (1 + padding + 0.001)
    p_nor = p_nor + 0.5
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 0.001
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def normalize_coordinate(p, padding=0.1, plane='xz'):
    """ Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    """
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane == 'xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]
    xy_new = xy / (1 + padding + 1e-05)
    xy_new = xy_new + 0.5
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 1e-05
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


class FCDecoder(nn.Module):
    """Decoder.
        Instead of conditioning on global features, on plane/volume local features.
    Args:
    dim (int): input dimension
    c_dim (int): dimension of latent conditioned code c
    out_dim (int): dimension of latent conditioned code c
    leaky (bool): whether to use leaky ReLUs
    sample_mode (str): sampling feature strategy, bilinear|nearest
    padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(self, dim=3, c_dim=128, out_dim=1, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.fc = nn.Linear(dim + c_dim, out_dim)
        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)
        net = self.fc(torch.cat((c, p), dim=2)).squeeze(-1)
        return net


class ResnetBlockFC(nn.Module):
    """ Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)
        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()
        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx


class LocalDecoder(nn.Module):
    """ Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    """

    def __init__(self, dim=3, c_dim=128, hidden_size=256, n_blocks=5, out_dim=1, leaky=False, sample_mode='bilinear', padding=0.1, concat_feat=False, no_xyz=False):
        super().__init__()
        self.concat_feat = concat_feat
        if concat_feat:
            c_dim *= 3
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.no_xyz = no_xyz
        self.hidden_size = hidden_size
        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])
        if not no_xyz:
            self.fc_p = nn.Linear(dim, hidden_size)
        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])
        self.fc_out = nn.Linear(hidden_size, out_dim)
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
        self.sample_mode = sample_mode
        self.padding = padding

    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            if self.concat_feat:
                c = []
                if 'grid' in plane_type:
                    c = self.sample_grid_feature(p, c_plane['grid'])
                if 'xz' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['xz'], plane='xz'))
                if 'xy' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['xy'], plane='xy'))
                if 'yz' in plane_type:
                    c.append(self.sample_plane_feature(p, c_plane['yz'], plane='yz'))
                c = torch.cat(c, dim=1)
                c = c.transpose(1, 2)
            else:
                c = 0
                if 'grid' in plane_type:
                    c += self.sample_grid_feature(p, c_plane['grid'])
                if 'xz' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
                if 'xy' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
                if 'yz' in plane_type:
                    c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
                c = c.transpose(1, 2)
        p = p.float()
        if self.no_xyz:
            net = torch.zeros(p.size(0), p.size(1), self.hidden_size)
        else:
            net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out

    def query_feature(self, p, c_plane):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)
        return c

    def compute_out(self, p, c):
        p = p.float()
        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out


class positional_encoding(object):
    """ Positional Encoding (presented in NeRF)

    Args:
        basis_function (str): basis function
    """

    def __init__(self, basis_function='sin_cos'):
        super().__init__()
        self.func = basis_function
        L = 10
        freq_bands = 2.0 ** np.linspace(0, L - 1, L)
        self.freq_bands = freq_bands * math.pi

    def __call__(self, p):
        if self.func == 'sin_cos':
            out = []
            p = 2.0 * p - 1.0
            for freq in self.freq_bands:
                out.append(torch.sin(freq * p))
                out.append(torch.cos(freq * p))
            p = torch.cat(out, dim=2)
        return p


class map2local(object):
    """ Add new keys to the given input

    Args:
        s (float): the defined voxel size
        pos_encoding (str): method for the positional encoding, linear|sin_cos
    """

    def __init__(self, s, pos_encoding='linear'):
        super().__init__()
        self.s = s
        self.pe = positional_encoding(basis_function=pos_encoding)

    def __call__(self, p):
        p = torch.remainder(p, self.s) / self.s
        p = self.pe(p)
        return p


class PatchLocalDecoder(nn.Module):
    """ Decoder adapted for crop training.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        local_coord (bool): whether to use local coordinate
        unit_size (float): defined voxel unit size for local system
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]

    """

    def __init__(self, dim=3, c_dim=128, hidden_size=256, leaky=False, n_blocks=5, sample_mode='bilinear', local_coord=False, pos_encoding='linear', unit_size=0.1, padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])
        self.fc_out = nn.Linear(hidden_size, 1)
        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
        self.sample_mode = sample_mode
        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None
        if pos_encoding == 'sin_cos':
            self.fc_p = nn.Linear(60, hidden_size)
        else:
            self.fc_p = nn.Linear(dim, hidden_size)

    def sample_feature(self, xy, c, fea_type='2d'):
        if fea_type == '2d':
            xy = xy[:, :, None].float()
            vgrid = 2.0 * xy - 1.0
            c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        else:
            xy = xy[:, :, None, None].float()
            vgrid = 2.0 * xy - 1.0
            c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_plane, **kwargs):
        p_n = p['p_n']
        p = p['p']
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_feature(p_n['grid'], c_plane['grid'], fea_type='3d')
            if 'xz' in plane_type:
                c += self.sample_feature(p_n['xz'], c_plane['xz'])
            if 'xy' in plane_type:
                c += self.sample_feature(p_n['xy'], c_plane['xy'])
            if 'yz' in plane_type:
                c += self.sample_feature(p_n['yz'], c_plane['yz'])
            c = c.transpose(1, 2)
        p = p.float()
        if self.map2local:
            p = self.map2local(p)
        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out


class LocalPointDecoder(nn.Module):
    """ Decoder for PointConv Baseline.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of blocks ResNetBlockFC layers
        sample_mode (str): sampling mode  for points
    """

    def __init__(self, dim=3, c_dim=128, hidden_size=256, leaky=False, n_blocks=5, sample_mode='gaussian', **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        if c_dim != 0:
            self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)])
        self.fc_p = nn.Linear(dim, hidden_size)
        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])
        self.fc_out = nn.Linear(hidden_size, 1)
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
        self.sample_mode = sample_mode
        if sample_mode == 'gaussian':
            self.var = kwargs['gaussian_val'] ** 2

    def sample_point_feature(self, q, p, fea):
        if self.sample_mode == 'gaussian':
            dist = -((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 1e-05) ** 2
            weight = (dist / self.var).exp()
        else:
            weight = 1 / ((p.unsqueeze(1).expand(-1, q.size(1), -1, -1) - q.unsqueeze(2)).norm(dim=3) + 1e-05)
        weight = weight / weight.sum(dim=2).unsqueeze(-1)
        c_out = weight @ fea
        return c_out

    def forward(self, p, c, **kwargs):
        n_points = p.shape[1]
        if n_points >= 30000:
            pp, fea = c
            c_list = []
            for p_split in torch.split(p, 10000, dim=1):
                if self.c_dim != 0:
                    c_list.append(self.sample_point_feature(p_split, pp, fea))
            c = torch.cat(c_list, dim=1)
        elif self.c_dim != 0:
            pp, fea = c
            c = self.sample_point_feature(p, pp, fea)
        p = p.float()
        net = self.fc_p(p)
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)
            net = self.blocks[i](net)
        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)
        return out


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv1x1(in_channels, out_channels))


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5, start_filts=64, up_mode='transpose', merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError('"{}" is not a valid mode for upsampling. Only "transpose" and "upsample" are allowed.'.format(up_mode))
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError('"{}" is not a valid mode formerging up and down paths. Only "concat" and "add" are allowed.'.format(up_mode))
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError('up_mode "upsample" is incompatible with merge_mode "add" at the moment because it doesn\'t make sense to use nearest neighbour to reduce depth channels (by half).')
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * 2 ** i
            pooling = True if i < depth - 1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.conv_final = conv1x1(outs, self.num_classes)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)
        x = self.conv_final(x)
        return x


def conv(in_channels, out_channels, kernel_size):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


class Decoder(nn.Module):

    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        self.conv1 = conv(in_channels, filters[0], kernels[0])
        self.conv2 = conv(filters[0], filters[1], kernels[1])
        self.conv3 = conv(filters[1], filters[2], kernels[2])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.interpolate(x, 10)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.interpolate(x, 20)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, 40)
        return x


def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, 'Conv layer MUST be present'
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'
    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels
            if num_channels < num_groups:
                num_groups = 1
            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")
    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8, padding=1):
        super(SingleConv, self).__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='crg', num_groups=8):
        super(DoubleConv, self).__init__()
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels
        self.add_module('SingleConv1', SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        self.add_module('SingleConv2', SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))


def conv_stride(in_channels, out_channels, kernel_size):
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2)


class Encoder(nn.Module):

    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        self.conv1 = conv_stride(in_channels, filters[0], kernels[0])
        self.conv2 = conv_stride(filters[0], filters[1], kernels[1])
        self.conv3 = conv_stride(filters[1], filters[2], kernels[2])

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x


def number_of_features_per_level(init_channel_number, num_levels):
    return [(init_channel_number * 2 ** k) for k in range(num_levels)]


class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr', num_groups=8, num_levels=4, is_segmentation=False, testing=False, **kwargs):
        super(Abstract3DUNet, self).__init__()
        self.testing = testing
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=basic_module, conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=basic_module, conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=basic_module, conv_layer_order=layer_order, num_groups=num_groups)
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        if is_segmentation:
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None

    def forward(self, x):
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            x = decoder(encoder_features, x)
        x = self.final_conv(x)
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)
        return x


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr', num_groups=8, num_levels=4, is_segmentation=True, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid, basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order, num_groups=num_groups, num_levels=num_levels, is_segmentation=is_segmentation, **kwargs)


def coordinate2index(x, reso, coord_type='2d'):
    """ Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    """
    x = (x * reso).long()
    if coord_type == '2d':
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d':
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


class LocalPoolPointnet(nn.Module):
    """ PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5):
        super().__init__()
        self.c_dim = c_dim
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim
        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None
        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None
        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding
        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

    def generate_plane_features(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        index = coordinate2index(xy, self.reso_plane)
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane)
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)
        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)
        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()
        c_out = 0
        for key in keys:
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p):
        batch_size, T, D = p.size()
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = coordinate2index(coord['yz'], self.reso_plane)
        if 'grid' in self.plane_type:
            coord['grid'] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index['grid'] = coordinate2index(coord['grid'], self.reso_grid, coord_type='3d')
        net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')
        return fea


class PatchLocalPoolPointnet(nn.Module):
    """ PointNet-based encoder network with ResNet blocks.
        First transform input points to local system based on the given voxel size.
        Support non-fixed number of point cloud, but need to precompute the index
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
        local_coord (bool): whether to use local coordinate
        pos_encoding (str): method for the positional encoding, linear|sin_cos
        unit_size (float): defined voxel unit size for local system
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5, local_coord=False, pos_encoding='linear', unit_size=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.blocks = nn.ModuleList([ResnetBlockFC(2 * hidden_dim, hidden_dim) for i in range(n_blocks)])
        self.fc_c = nn.Linear(hidden_dim, c_dim)
        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding
        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None
        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None
        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')
        if local_coord:
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None
        if pos_encoding == 'sin_cos':
            self.fc_pos = nn.Linear(60, 2 * hidden_dim)
        else:
            self.fc_pos = nn.Linear(dim, 2 * hidden_dim)

    def generate_plane_features(self, index, c):
        c = c.permute(0, 2, 1)
        if index.max() < self.reso_plane ** 2:
            fea_plane = c.new_zeros(c.size(0), self.c_dim, self.reso_plane ** 2)
            fea_plane = scatter_mean(c, index, out=fea_plane)
        else:
            fea_plane = scatter_mean(c, index)
            if fea_plane.shape[-1] > self.reso_plane ** 2:
                fea_plane = fea_plane[:, :, :-1]
        fea_plane = fea_plane.reshape(c.size(0), self.c_dim, self.reso_plane, self.reso_plane)
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
        return fea_plane

    def generate_grid_features(self, index, c):
        c = c.permute(0, 2, 1)
        if index.max() < self.reso_grid ** 3:
            fea_grid = c.new_zeros(c.size(0), self.c_dim, self.reso_grid ** 3)
            fea_grid = scatter_mean(c, index, out=fea_grid)
        else:
            fea_grid = scatter_mean(c, index)
            if fea_grid.shape[-1] > self.reso_grid ** 3:
                fea_grid = fea_grid[:, :, :-1]
        fea_grid = fea_grid.reshape(c.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)
        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)
        return fea_grid

    def pool_local(self, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = index.keys()
        c_out = 0
        for key in keys:
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key])
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key])
            if self.scatter == scatter_max:
                fea = fea[0]
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, inputs):
        p = inputs['points']
        index = inputs['index']
        batch_size, T, D = p.size()
        if self.map2local:
            pp = self.map2local(p)
            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)
        c = self.fc_c(net)
        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(index['grid'], c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(index['xz'], c)
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(index['xy'], c)
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(index['yz'], c)
        return fea


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 10000000000.0
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


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


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):

    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
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
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-08)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNetPlusPlus(nn.Module):

    def __init__(self, dim=None, c_dim=128, padding=0.1):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, c_dim])

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        return xyz.permute(0, 2, 1), l0_points.permute(0, 2, 1)


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order, num_groups=num_groups)
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.non_linearity(out)
        return out


class Upsampling(nn.Module):
    """
    Upsamples a given multi-channel 3D data using either interpolation or learned transposed convolution.

    Args:
        transposed_conv (bool): if True uses ConvTranspose3d for upsampling, otherwise uses interpolation
        concat_joining (bool): if True uses concatenation joining between encoder and decoder features, otherwise
            uses summation joining (see Residual U-Net)
        in_channels (int): number of input channels for transposed conv
        out_channels (int): number of output channels for transpose conv
        kernel_size (int or tuple): size of the convolving kernel
        scale_factor (int or tuple): stride of the convolution
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
    """

    def __init__(self, transposed_conv, in_channels=None, out_channels=None, kernel_size=3, scale_factor=(2, 2, 2), mode='nearest'):
        super(Upsampling, self).__init__()
        if transposed_conv:
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor, padding=1)
        else:
            self.upsample = partial(self._interpolate, mode=mode)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class FinalConv(nn.Sequential):
    """
    A module consisting of a convolution layer (e.g. Conv3d+ReLU+GroupNorm3d) and the final 1x1 convolution
    which reduces the number of channels to 'out_channels'.
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ReLU use order='cbr'.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8):
        super(FinalConv, self).__init__()
        self.add_module('SingleConv', SingleConv(in_channels, in_channels, kernel_size, order, num_groups))
        final_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.add_module('final_conv', final_conv)


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr', num_groups=8, num_levels=5, is_segmentation=True, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid, basic_module=ExtResNetBlock, f_maps=f_maps, layer_order=layer_order, num_groups=num_groups, num_levels=num_levels, is_segmentation=is_segmentation, **kwargs)


class LocalVoxelEncoder(nn.Module):
    """ 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent code c
        hidden_dim (int): hidden dimension of the network
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        kernel_size (int): kernel size for the first layer of CNN
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    
    """

    def __init__(self, dim=3, c_dim=128, unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, plane_resolution=512, grid_resolution=None, plane_type='xz', kernel_size=3, padding=0.1):
        super().__init__()
        self.actvn = F.relu
        if kernel_size == 1:
            self.conv_in = nn.Conv3d(1, c_dim, 1)
        else:
            self.conv_in = nn.Conv3d(1, c_dim, kernel_size, padding=1)
        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None
        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None
        self.c_dim = c_dim
        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

    def generate_plane_features(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        index = coordinate2index(xy, self.reso_plane)
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane)
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)
        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)
        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)
        return fea_grid

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        n_voxel = x.size(1) * x.size(2) * x.size(3)
        coord1 = torch.linspace(-0.5, 0.5, x.size(1))
        coord2 = torch.linspace(-0.5, 0.5, x.size(2))
        coord3 = torch.linspace(-0.5, 0.5, x.size(3))
        coord1 = coord1.view(1, -1, 1, 1).expand_as(x)
        coord2 = coord2.view(1, 1, -1, 1).expand_as(x)
        coord3 = coord3.view(1, 1, 1, -1).expand_as(x)
        p = torch.stack([coord1, coord2, coord3], dim=4)
        p = p.view(batch_size, n_voxel, -1)
        x = x.unsqueeze(1)
        c = self.actvn(self.conv_in(x)).view(batch_size, self.c_dim, -1)
        c = c.permute(0, 2, 1)
        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        else:
            if 'xz' in self.plane_type:
                fea['xz'] = self.generate_plane_features(p, c, plane='xz')
            if 'xy' in self.plane_type:
                fea['xy'] = self.generate_plane_features(p, c, plane='xy')
            if 'yz' in self.plane_type:
                fea['yz'] = self.generate_plane_features(p, c, plane='yz')
        return fea


class VoxelEncoder(nn.Module):
    """ 3D-convolutional encoder network for voxel input.

    Args:
        dim (int): input dimension
        c_dim (int): output dimension
    """

    def __init__(self, dim=3, c_dim=128):
        super().__init__()
        self.actvn = F.relu
        self.conv_in = nn.Conv3d(1, 32, 3, padding=1)
        self.conv_0 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv_3 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.fc = nn.Linear(512 * 2 * 2 * 2, c_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(1)
        net = self.conv_in(x)
        net = self.conv_0(self.actvn(net))
        net = self.conv_1(self.actvn(net))
        net = self.conv_2(self.actvn(net))
        net = self.conv_3(self.actvn(net))
        hidden = net.view(batch_size, 512 * 2 * 2 * 2)
        c = self.fc(self.actvn(hidden))
        return c


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder(1, [16, 32, 64], [5, 3, 3])
        self.decoder = Decoder(64, [64, 32, 16], [3, 3, 5])
        self.conv_qual = conv(16, 1, 5)
        self.conv_rot = conv(16, 4, 5)
        self.conv_width = conv(16, 1, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        qual_out = torch.sigmoid(self.conv_qual(x))
        rot_out = F.normalize(self.conv_rot(x), dim=1)
        width_out = self.conv_width(x)
        return qual_out, rot_out, width_out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64, 64])], {}),
     True),
    (DoubleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'encoder': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ExtResNetBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FinalConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PointNetSetAbstraction,
     lambda: ([], {'npoint': 4, 'radius': 4, 'nsample': 4, 'in_channel': 4, 'mlp': [4, 4], 'group_all': 4}),
     lambda: ([torch.rand([4, 1, 4]), torch.rand([4, 3, 4])], {}),
     False),
    (ResnetBlockFC,
     lambda: ([], {'size_in': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingleConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UNet,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_UT_Austin_RPL_GIGA(_paritybench_base):
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

