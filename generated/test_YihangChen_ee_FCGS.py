import sys
_module = sys.modules[__name__]
del sys
arguments = _module
decode_single_scene = _module
decode_single_scene_validate = _module
encode_single_scene = _module
gaussian_renderer = _module
network_gui = _module
FCGS_model = _module
model = _module
encodings_cuda = _module
entropy_models = _module
gpcc_utils = _module
grid_utils = _module
scene = _module
cameras = _module
colmap_loader = _module
dataset_readers = _module
gaussian_model = _module
setup = _module
freqencoder = _module
backend = _module
freq = _module
setup = _module
gridcreater = _module
backend = _module
grid = _module
setup = _module
gridencoder = _module
backend = _module
grid = _module
setup = _module
setup = _module
camera_utils = _module
general_utils = _module
graphics_utils = _module
image_utils = _module
loss_utils = _module
sh_utils = _module
system_utils = _module

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


import torch.nn as nn


from typing import NamedTuple


import numpy as np


import math


import torchvision.transforms as transforms


import torch.nn.functional as nnf


from typing import Union


from torch.autograd import Function


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from torch import nn


from torch.utils.cpp_extension import CUDAExtension


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import load


from torch.autograd.function import once_differentiable


import random


import torch.nn.functional as F


from torch.autograd import Variable


from math import exp


class Channel_CTX_fea(nn.Module):

    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 64]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 64]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 64]))
        self.MLP_d0 = nn.Sequential(nn.Linear(64, 64 * 3), nn.LeakyReLU(inplace=True), nn.Linear(64 * 3, 64 * 3), nn.LeakyReLU(inplace=True), nn.Linear(64 * 3, 64 * 3))
        self.MLP_d1 = nn.Sequential(nn.Linear(64 * 2, 64 * 3), nn.LeakyReLU(inplace=True), nn.Linear(64 * 3, 64 * 3), nn.LeakyReLU(inplace=True), nn.Linear(64 * 3, 64 * 3))
        self.MLP_d2 = nn.Sequential(nn.Linear(64 * 3, 64 * 3), nn.LeakyReLU(inplace=True), nn.Linear(64 * 3, 64 * 3), nn.LeakyReLU(inplace=True), nn.Linear(64 * 3, 64 * 3))

    def forward(self, fea_q, to_dec=-1):
        NN = fea_q.shape[0]
        d0, d1, d2, d3 = torch.split(fea_q, split_size_or_sections=[64, 64, 64, 64], dim=-1)
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d0(d0), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d1(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, d2], dim=-1)), chunks=3, dim=-1)
        mean = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3], dim=-1)
        scale = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3], dim=-1)
        prob = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3], dim=-1)
        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        return mean, scale, prob


class Channel_CTX_feq(nn.Module):

    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 16]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 16]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 16]))
        self.MLP_d0 = nn.Sequential(nn.Linear(16, 16 * 3), nn.LeakyReLU(inplace=True), nn.Linear(16 * 3, 16 * 3), nn.LeakyReLU(inplace=True), nn.Linear(16 * 3, 16 * 3))
        self.MLP_d1 = nn.Sequential(nn.Linear(16 * 2, 16 * 3), nn.LeakyReLU(inplace=True), nn.Linear(16 * 3, 16 * 3), nn.LeakyReLU(inplace=True), nn.Linear(16 * 3, 16 * 3))

    def forward(self, shs_q, to_dec=-1):
        NN = shs_q.shape[0]
        shs_q = shs_q.view(NN, 16, 3)
        d0, d1, d2 = shs_q[..., 0], shs_q[..., 1], shs_q[..., 2]
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d0(d0), chunks=3, dim=-1)
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d1(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)
        mean = torch.stack([mean_d0, mean_d1, mean_d2], dim=-1).view(NN, -1)
        scale = torch.stack([scale_d0, scale_d1, scale_d2], dim=-1).view(NN, -1)
        prob = torch.stack([prob_d0, prob_d1, prob_d2], dim=-1).view(NN, -1)
        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        return mean, scale, prob


class _grid_creater(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input_norm_xyz, input_feature, resolutions_list, offsets_list, determ=False):
        input_norm_xyz = input_norm_xyz.contiguous()
        input_feature = input_feature.contiguous()
        N, num_dim = input_norm_xyz.shape
        n_levels = offsets_list.shape[0] - 1
        n_features = input_feature.shape[1]
        if determ:
            outputs0 = torch.zeros(size=[offsets_list[-1].item(), n_features], device=input_feature.device, dtype=torch.int32)
            weights0 = torch.zeros(size=[offsets_list[-1].item(), 1], device=input_feature.device, dtype=torch.int32)
            outputs = torch.zeros(size=[offsets_list[-1].item(), n_features], device=input_feature.device, dtype=input_feature.dtype)
            weights = torch.zeros(size=[offsets_list[-1].item(), 1], device=input_feature.device, dtype=input_feature.dtype)
            gc.grid_creater_forward_determ(input_norm_xyz, input_feature, outputs0, weights0, offsets_list, resolutions_list, N, num_dim, n_features, n_levels)
            outputs[...] = outputs0 * 0.0001
            weights[...] = weights0 * 0.0001
            outputs = outputs.contiguous()
            weights = weights.contiguous()
        else:
            outputs = torch.zeros(size=[offsets_list[-1].item(), n_features], device=input_feature.device, dtype=input_feature.dtype)
            weights = torch.zeros(size=[offsets_list[-1].item(), 1], device=input_feature.device, dtype=input_feature.dtype)
            gc.grid_creater_forward(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, num_dim, n_features, n_levels)
        outputs_div_weights = outputs / (weights + 1e-09)
        ctx.save_for_backward(input_norm_xyz, weights, offsets_list, resolutions_list)
        ctx.dims = [N, num_dim, n_features, n_levels]
        return outputs_div_weights

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()
        input_norm_xyz, weights, offsets_list, resolutions_list = ctx.saved_tensors
        N, num_dim, n_features, n_levels = ctx.dims
        grad_feature = torch.zeros(size=[N, n_features], device=input_norm_xyz.device, dtype=grad.dtype)
        gc.grid_creater_backward(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, num_dim, n_features, n_levels)
        return None, grad_feature, None, None, None


class _grid_encoder(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets_list, resolutions_list):
        inputs = inputs.contiguous()
        N, num_dim = inputs.shape
        n_levels = offsets_list.shape[0] - 1
        n_features = embeddings.shape[1]
        outputs = torch.empty(n_levels, N, n_features, device=inputs.device, dtype=embeddings.dtype)
        ge.grid_encode_forward(inputs, embeddings, offsets_list, resolutions_list, outputs, N, num_dim, n_features, n_levels)
        outputs = outputs.permute(1, 0, 2).reshape(N, n_levels * n_features)
        ctx.save_for_backward(inputs, embeddings, offsets_list, resolutions_list)
        ctx.dims = [N, num_dim, n_features, n_levels]
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        inputs, embeddings, offsets_list, resolutions_list = ctx.saved_tensors
        N, num_dim, n_features, n_levels = ctx.dims
        grad = grad.view(N, n_levels, n_features).permute(1, 0, 2).contiguous()
        grad_embeddings = torch.zeros_like(embeddings)
        ge.grid_encode_backward(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, num_dim, n_features, n_levels)
        return None, grad_embeddings, None, None


class Spatial_CTX(nn.Module):

    def __init__(self, reso_3D, off_3D, reso_2D, off_2D):
        super().__init__()
        self.reso_3D = reso_3D
        self.off_3D = off_3D
        self.reso_2D = reso_2D
        self.off_2D = off_2D

    def forward(self, xyz_for_creater, xyz_for_interp, feature, determ=False, return_all=False):
        assert xyz_for_creater.shape[0] == feature.shape[0]
        grid_3D = _grid_creater.apply(xyz_for_creater, feature, self.reso_3D, self.off_3D, determ)
        grid_xy = _grid_creater.apply(xyz_for_creater[:, 0:2], feature, self.reso_2D, self.off_2D, determ)
        grid_xz = _grid_creater.apply(xyz_for_creater[:, 0::2], feature, self.reso_2D, self.off_2D, determ)
        grid_yz = _grid_creater.apply(xyz_for_creater[:, 1:3], feature, self.reso_2D, self.off_2D, determ)
        context_info_3D = _grid_encoder.apply(xyz_for_interp, grid_3D, self.off_3D, self.reso_3D)
        context_info_xy = _grid_encoder.apply(xyz_for_interp[:, 0:2], grid_xy, self.off_2D, self.reso_2D)
        context_info_xz = _grid_encoder.apply(xyz_for_interp[:, 0::2], grid_xz, self.off_2D, self.reso_2D)
        context_info_yz = _grid_encoder.apply(xyz_for_interp[:, 1:3], grid_yz, self.off_2D, self.reso_2D)
        context_info = torch.cat([context_info_3D, context_info_xy, context_info_xz, context_info_yz], dim=-1)
        if return_all:
            return context_info, (xyz_for_creater, xyz_for_interp, feature, grid_3D, grid_xy, grid_xz, grid_yz, context_info_3D, context_info_xy, context_info_xz, context_info_yz, self.reso_3D, self.off_3D)
        return context_info


class Low_bound(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-06)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-06] = 0
        pass_through_if = np.logical_or(x.cpu().numpy() >= 1e-06, g.cpu().numpy() < 0.0)
        t = torch.Tensor(pass_through_if + 0.0)
        return grad1 * t


class Entropy_factorized(nn.Module):

    def __init__(self, channel=32, init_scale=10, filters=(3, 3, 3), likelihood_bound=1e-06, tail_mass=1e-09, optimize_integer_offset=True, Q=1):
        super(Entropy_factorized, self).__init__()
        self.filters = tuple(int(t) for t in filters)
        self.init_scale = float(init_scale)
        self.likelihood_bound = float(likelihood_bound)
        self.tail_mass = float(tail_mass)
        self.optimize_integer_offset = bool(optimize_integer_offset)
        self.Q = Q
        if not 0 < self.tail_mass < 1:
            raise ValueError('`tail_mass` must be between 0 and 1')
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1.0 / (len(self.filters) + 1))
        self._matrices = nn.ParameterList([])
        self._bias = nn.ParameterList([])
        self._factor = nn.ParameterList([])
        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1.0 / scale / filters[i + 1]))
            self.matrix = nn.Parameter(torch.FloatTensor(channel, filters[i + 1], filters[i]))
            self.matrix.data.fill_(init)
            self._matrices.append(self.matrix)
            self.bias = nn.Parameter(torch.FloatTensor(channel, filters[i + 1], 1))
            noise = np.random.uniform(-0.5, 0.5, self.bias.size())
            noise = torch.FloatTensor(noise)
            self.bias.data.copy_(noise)
            self._bias.append(self.bias)
            if i < len(self.filters):
                self.factor = nn.Parameter(torch.FloatTensor(channel, filters[i + 1], 1))
                self.factor.data.fill_(0.0)
                self._factor.append(self.factor)

    def _logits_cumulative(self, logits, stop_gradient):
        for i in range(len(self.filters) + 1):
            matrix = nnf.softplus(self._matrices[i])
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(matrix, logits)
            bias = self._bias[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias
            if i < len(self._factor):
                factor = nnf.tanh(self._factor[i])
                if stop_gradient:
                    factor = factor.detach()
                logits += factor * nnf.tanh(logits)
        return logits

    def forward(self, x, Q=None, return_lkl=False):
        if Q is None:
            Q = self.Q
        elif isinstance(Q, torch.Tensor):
            Q = Q.permute(1, 0).contiguous()
            Q = Q.view(Q.shape[0], 1, -1)
        x = x.permute(1, 0).contiguous()
        x = x.view(x.shape[0], 1, -1)
        lower = self._logits_cumulative(x - 0.5 * Q, stop_gradient=False)
        upper = self._logits_cumulative(x + 0.5 * Q, stop_gradient=False)
        sign = -torch.sign(torch.add(lower, upper))
        sign = sign.detach()
        likelihood = torch.abs(nnf.sigmoid(sign * upper) - nnf.sigmoid(sign * lower))
        if return_lkl:
            likelihood = likelihood.view(likelihood.shape[0], -1).permute(1, 0).contiguous()
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            bits = bits.view(bits.shape[0], -1)
            bits = bits.permute(1, 0).contiguous()
            return bits


class Entropy_gaussian(nn.Module):

    def __init__(self, Q=1):
        super(Entropy_gaussian, self).__init__()
        self.Q = Q

    def forward(self, x, mean, scale, Q=None, x_mean=None, return_lkl=False):
        if Q is None:
            Q = self.Q
        scale = torch.clamp(scale, min=1e-09)
        m1 = torch.distributions.normal.Normal(mean, scale)
        lower = m1.cdf(x - 0.5 * Q)
        upper = m1.cdf(x + 0.5 * Q)
        likelihood = torch.abs(upper - lower)
        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits


class Entropy_gaussian_mix_prob_2(nn.Module):

    def __init__(self, Q=1):
        super(Entropy_gaussian_mix_prob_2, self).__init__()
        self.Q = Q

    def forward(self, x, mean1, mean2, scale1, scale2, probs1, probs2, Q=None, x_mean=None, return_lkl=False):
        if Q is None:
            Q = self.Q
        scale1 = torch.clamp(scale1, min=1e-09)
        scale2 = torch.clamp(scale2, min=1e-09)
        m1 = torch.distributions.normal.Normal(mean1, scale1)
        m2 = torch.distributions.normal.Normal(mean2, scale2)
        likelihood1 = torch.abs(m1.cdf(x + 0.5 * Q) - m1.cdf(x - 0.5 * Q))
        likelihood2 = torch.abs(m2.cdf(x + 0.5 * Q) - m2.cdf(x - 0.5 * Q))
        likelihood = Low_bound.apply(probs1 * likelihood1 + probs2 * likelihood2)
        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits


class Entropy_gaussian_mix_prob_3(nn.Module):

    def __init__(self, Q=1):
        super(Entropy_gaussian_mix_prob_3, self).__init__()
        self.Q = Q

    def forward(self, x, mean1, mean2, mean3, scale1, scale2, scale3, probs1, probs2, probs3, Q=None, x_mean=None, return_lkl=False):
        if Q is None:
            Q = self.Q
        scale1 = torch.clamp(scale1, min=1e-09)
        scale2 = torch.clamp(scale2, min=1e-09)
        scale3 = torch.clamp(scale3, min=1e-09)
        m1 = torch.distributions.normal.Normal(mean1, scale1)
        m2 = torch.distributions.normal.Normal(mean2, scale2)
        m3 = torch.distributions.normal.Normal(mean3, scale3)
        likelihood1 = torch.abs(m1.cdf(x + 0.5 * Q) - m1.cdf(x - 0.5 * Q))
        likelihood2 = torch.abs(m2.cdf(x + 0.5 * Q) - m2.cdf(x - 0.5 * Q))
        likelihood3 = torch.abs(m3.cdf(x + 0.5 * Q) - m3.cdf(x - 0.5 * Q))
        likelihood = Low_bound.apply(probs1 * likelihood1 + probs2 * likelihood2 + probs3 * likelihood3)
        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits


nvcc_flags = ['-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']


class _freq_encoder(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, degree, output_dim):
        if not inputs.is_cuda:
            inputs = inputs
        inputs = inputs.contiguous()
        B, input_dim = inputs.shape
        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)
        _backend.freq_encode_forward(inputs, B, input_dim, degree, output_dim, outputs)
        ctx.save_for_backward(inputs, outputs)
        ctx.dims = [B, input_dim, degree, output_dim]
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()
        inputs, outputs = ctx.saved_tensors
        B, input_dim, degree, output_dim = ctx.dims
        grad_inputs = torch.zeros_like(inputs)
        _backend.freq_encode_backward(grad, outputs, B, input_dim, degree, output_dim, grad_inputs)
        return grad_inputs, None, None


freq_encode = _freq_encoder.apply


class FreqEncoder(nn.Module):

    def __init__(self, input_dim=3, degree=4):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = input_dim + input_dim * 2 * degree

    def __repr__(self):
        return f'FreqEncoder: input_dim={self.input_dim} degree={self.degree} output_dim={self.output_dim}'

    def forward(self, inputs, **kwargs):
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, self.input_dim)
        outputs = freq_encode(inputs, self.degree, self.output_dim)
        outputs = outputs.reshape(prefix_shape + [self.output_dim])
        return outputs


class STE_multistep(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, Q):
        return torch.round(input / Q) * Q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


b2M = 8 * 1024 * 1024


def gpcc_encode(encoder_path: 'str', ply_path: 'str', bin_path: 'str') ->None:
    """
    Compress geometry point cloud by GPCC codec.
    """
    enc_cmd = f'{encoder_path} --mode=0 --trisoupNodeSizeLog2=0 --mergeDuplicatedPoints=0 --neighbourAvailBoundaryLog2=8 --intra_pred_max_node_size_log2=3 --positionQuantizationScale=1 --inferredDirectCodingMode=3 --maxNumQtBtBeforeOt=2 --minQtbtSizeLog2=0 --planarEnabled=0 --planarModeIdcmUse=0 --cabac_bypass_stream_enabled_flag=1 --uncompressedDataPath={ply_path} --compressedStreamPath={bin_path} '
    enc_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    exit_code = os.system(enc_cmd)
    assert exit_code == 0, f'GPCC encoder failed with exit code {exit_code}.'


def sorted_voxels(voxelized_means: 'np.ndarray', other_params=None) ->Union[np.ndarray, tuple]:
    """
    Sort voxels by their Morton code.
    """
    indices_sorted = np.argsort(voxelized_means @ np.power(voxelized_means.max() + 1, np.arange(voxelized_means.shape[1])), axis=0)
    voxelized_means = voxelized_means[indices_sorted]
    if other_params is None:
        return voxelized_means
    other_params = other_params[indices_sorted]
    return voxelized_means, other_params


VOXELIZE_SCALE_FACTOR = 16


def voxelize(means: 'np.ndarray') ->tuple:
    """
    Voxelization of Gaussians.
    """
    means_min, means_max = means.min(axis=0), means.max(axis=0)
    voxelized_means = (means - means_min) / (means_max - means_min)
    voxelized_means = np.round(voxelized_means * (2 ** VOXELIZE_SCALE_FACTOR - 1))
    return voxelized_means, means_min, means_max


def write_binary_data(dst_file_handle, src_bin_path: 'str') ->None:
    """
    Write binary data to a binary file handle.
    """
    with open(src_bin_path, 'rb') as f:
        data = f.read()
        dst_file_handle.write(np.array([len(data)], dtype=np.uint32).tobytes())
        dst_file_handle.write(data)


def write_ply_geo_ascii(geo_data: 'np.ndarray', ply_path: 'str') ->None:
    """
    Write geometry point cloud to a .ply file in ASCII format.
    """
    assert ply_path.endswith('.ply'), 'Destination path must be a .ply file.'
    assert geo_data.ndim == 2 and geo_data.shape[1] == 3, 'Input data must be a 3D point cloud.'
    geo_data = geo_data.astype(int)
    with open(ply_path, 'w') as f:
        f.writelines(['ply\n', 'format ascii 1.0\n', f'element vertex {geo_data.shape[0]}\n', 'property float x\n', 'property float y\n', 'property float z\n', 'end_header\n'])
        for point in geo_data:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')


def compress_gaussian_params(gaussian_params, bin_path, gpcc_codec_path='tmc3'):
    """
    Compress Gaussian model parameters.
    - Means are compressed by GPCC codec
    - Other parameters except opacity are first quantized, and opacity, indices and codebooks are losslessly compressed by numpy.
    """
    if isinstance(gaussian_params, torch.Tensor):
        gaussian_params = gaussian_params.detach().cpu().numpy()
    means = gaussian_params
    voxelized_means, means_min, means_max = voxelize(means=means)
    voxelized_means = sorted_voxels(voxelized_means=voxelized_means, other_params=None)
    means_enc = None
    with TemporaryDirectory() as temp_dir:
        ply_path = os.path.join(temp_dir, 'voxelized_means.ply')
        write_ply_geo_ascii(geo_data=voxelized_means, ply_path=ply_path)
        means_bin_path = os.path.join(temp_dir, 'compressed.bin')
        gpcc_encode(encoder_path=gpcc_codec_path, ply_path=ply_path, bin_path=means_bin_path)
        with open(bin_path, 'wb') as f:
            head_info = np.array([means_min, means_max], dtype=np.float32)
            f.write(head_info.tobytes())
            write_binary_data(dst_file_handle=f, src_bin_path=means_bin_path)
            file_size = {'means': os.path.getsize(means_bin_path) / 1024 / 1024, 'total': os.path.getsize(bin_path) / 1024 / 1024}
    compress_results = {'num_gaussians': voxelized_means.shape[0], 'file_size': file_size}
    return means_enc, voxelized_means, means_min, means_max, compress_results


chunk_size_cuda = 10000


def decoder(N_len, file_name='tmp.b', device='cuda'):
    assert file_name[-2:] == '.b'
    with open(file_name, 'rb') as fin:
        prob_1 = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy())
        len_cnt_bytes = np.frombuffer(fin.read(4), dtype=np.int32)[0]
        cnt_torch = torch.tensor(np.frombuffer(fin.read(len_cnt_bytes), dtype=np.int32).copy(), device='cuda')
        byte_stream_torch = torch.tensor(np.frombuffer(fin.read(), dtype=np.uint8).copy(), device='cuda')
    p = torch.zeros(size=[N_len], dtype=torch.float32, device='cuda')
    p[...] = prob_1
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    sym_out = arithmetic.arithmetic_decode(output_cdf, byte_stream_torch, cnt_torch, chunk_size_cuda, int(output_cdf.shape[0]), int(output_cdf.shape[1]))
    return sym_out


def decoder_factorized(lower_func, Q, N_len, dim, file_name='tmp.b', device='cuda'):
    assert file_name.endswith('.b')
    with open(file_name, 'rb') as fin:
        min_value = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy(), device='cuda')
        max_value = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy(), device='cuda')
        len_cnt_bytes = np.frombuffer(fin.read(4), dtype=np.int32)[0]
        cnt_torch = torch.tensor(np.frombuffer(fin.read(len_cnt_bytes), dtype=np.int32).copy(), device='cuda')
        byte_stream_torch = torch.tensor(np.frombuffer(fin.read(), dtype=np.uint8).copy(), device='cuda')
    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1)).to(torch.float)
    samples = samples.unsqueeze(0).unsqueeze(0).repeat(dim, 1, 1)
    lower = lower_func((samples - 0.5) * Q, stop_gradient=False)
    upper = lower_func((samples + 0.5) * Q, stop_gradient=False)
    sign = -torch.sign(torch.add(lower, upper))
    sign = sign.detach()
    pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
    cdf = torch.cumsum(pmf, dim=-1)
    lower = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)
    lower = lower.permute(1, 0, 2).contiguous().repeat(N_len, 1, 1)
    lower = lower.view(N_len * dim, -1)
    lower = torch.clamp(lower, min=0.0, max=1.0)
    sym_out = arithmetic.arithmetic_decode(lower, byte_stream_torch, cnt_torch, chunk_size_cuda, int(lower.shape[0]), int(lower.shape[1])).to(device)
    x = sym_out + min_value
    x = x * Q
    x = x.reshape(N_len, dim)
    return x


def decoder_factorized_chunk(lower_func, Q, N_len, dim, file_name='tmp.b', device='cuda', chunk_size=10000000):
    assert file_name.endswith('.b')
    chunks = int(np.ceil(N_len / chunk_size))
    x_c_list = []
    for c in range(chunks):
        x_c = decoder_factorized(lower_func=lower_func, Q=Q, N_len=min(chunk_size, N_len - c * chunk_size), dim=dim, file_name=file_name.replace('.b', f'_{str(c)}.b'), device=device)
        x_c_list.append(x_c)
    x_c_list = torch.cat(x_c_list, dim=0)
    return x_c_list


def decoder_gaussian_mixed(mean_list, scale_list, prob_list, Q, file_name='tmp.b'):
    assert file_name.endswith('.b')
    m0 = mean_list[0]
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=m0.dtype, device=m0.device).repeat(m0.shape[0])
    assert mean_list[0].shape == scale_list[0].shape == prob_list[0].shape == Q.shape
    with open(file_name, 'rb') as fin:
        min_value = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy(), device='cuda')
        max_value = torch.tensor(np.frombuffer(fin.read(4), dtype=np.float32).copy(), device='cuda')
        len_cnt_bytes = np.frombuffer(fin.read(4), dtype=np.int32)[0]
        cnt_torch = torch.tensor(np.frombuffer(fin.read(len_cnt_bytes), dtype=np.int32).copy(), device='cuda')
        byte_stream_torch = torch.tensor(np.frombuffer(fin.read(), dtype=np.uint8).copy(), device='cuda')
    lower_all = int(0)
    for mean, scale, prob in zip(mean_list, scale_list, prob_list):
        lower = arithmetic.calculate_cdf(mean, scale, Q, min_value, max_value) * prob.unsqueeze(-1)
        if isinstance(lower_all, int):
            lower_all = lower
        else:
            lower_all += lower
    lower = torch.clamp(lower_all, min=0.0, max=1.0)
    sym_out = arithmetic.arithmetic_decode(lower, byte_stream_torch, cnt_torch, chunk_size_cuda, int(lower.shape[0]), int(lower.shape[1])).to(mean.device)
    x = sym_out + min_value
    x = x * Q
    return x


def decoder_gaussian_mixed_chunk(mean_list, scale_list, prob_list, Q, file_name='tmp.b', chunk_size=10000000):
    assert file_name.endswith('.b')
    mean_list_view = [mean.view(-1) for mean in mean_list]
    scale_list_view = [scale.view(-1) for scale in scale_list]
    prob_list_view = [prob.view(-1) for prob in prob_list]
    N = mean_list_view[0].shape[0]
    chunks = int(np.ceil(N / chunk_size))
    Is_Q_tensor = isinstance(Q, torch.Tensor)
    if Is_Q_tensor:
        Q_view = Q.view(-1)
    x_c_list = []
    for c in range(chunks):
        x_c = decoder_gaussian_mixed(mean_list=[mean[c * chunk_size:c * chunk_size + chunk_size] for mean in mean_list_view], scale_list=[scale[c * chunk_size:c * chunk_size + chunk_size] for scale in scale_list_view], prob_list=[prob[c * chunk_size:c * chunk_size + chunk_size] for prob in prob_list_view], Q=Q_view[c * chunk_size:c * chunk_size + chunk_size] if Is_Q_tensor else Q, file_name=file_name.replace('.b', f'_{str(c)}.b'))
        x_c_list.append(x_c)
    x_c_list = torch.cat(x_c_list, dim=0).type_as(mean_list[0])
    return x_c_list


def devoxelize(voxelized_means: 'np.ndarray', means_min: 'np.ndarray', means_max: 'np.ndarray') ->np.ndarray:
    voxelized_means = voxelized_means.astype(np.float32)
    means_min = means_min.astype(np.float32)
    means_max = means_max.astype(np.float32)
    means = voxelized_means / (2 ** VOXELIZE_SCALE_FACTOR - 1) * (means_max - means_min) + means_min
    return means


def gpcc_decode(decoder_path: 'str', bin_path: 'str', recon_path: 'str') ->None:
    """
    Decompress geometry point cloud by GPCC codec.
    """
    dec_cmd = f'{decoder_path} --mode=1 --outputBinaryPly=1 --compressedStreamPath={bin_path} --reconstructedDataPath={recon_path} '
    dec_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    exit_code = os.system(dec_cmd)
    assert exit_code == 0, f'GPCC decoder failed with exit code {exit_code}.'


def read_binary_data(dst_bin_path: 'str', src_file_handle) ->None:
    """
    Read binary data from file handle and write it to a binary file.
    """
    length = int(np.frombuffer(src_file_handle.read(4), dtype=np.uint32)[0])
    with open(dst_bin_path, 'wb') as f:
        f.write(src_file_handle.read(length))


def read_ply_geo_bin(ply_path: 'str') ->np.ndarray:
    """
    Read geometry point cloud from a .ply file in binary format.
    """
    assert ply_path.endswith('.ply'), 'Source path must be a .ply file.'
    ply_data = PlyData.read(ply_path).elements[0]
    means = np.stack([ply_data.data[name] for name in ['x', 'y', 'z']], axis=1)
    return means


def decompress_gaussian_params(bin_path, gpcc_codec_path='tmc3'):
    """
    Decompress Gaussian model parameters.
    """
    assert os.path.exists(bin_path), f'Bitstreams {bin_path} not found.'
    with TemporaryDirectory() as temp_dir:
        with open(bin_path, 'rb') as f:
            head_info = np.frombuffer(f.read(24), dtype=np.float32)
            means_min, means_max = head_info[:3], head_info[3:]
            means_bin_path = os.path.join(temp_dir, 'compressed.bin')
            read_binary_data(dst_bin_path=means_bin_path, src_file_handle=f)
        ply_path = os.path.join(temp_dir, 'voxelized_means.ply')
        gpcc_decode(decoder_path=gpcc_codec_path, bin_path=means_bin_path, recon_path=ply_path)
        voxelized_means = read_ply_geo_bin(ply_path=ply_path).astype(np.float32)
        voxelized_means = sorted_voxels(voxelized_means)
        means_dec = devoxelize(voxelized_means=voxelized_means, means_min=means_min, means_max=means_max)
        means_dec = torch.from_numpy(means_dec)
    return means_dec, voxelized_means, means_min, means_max


def encoder(x, file_name='tmp.b'):
    assert file_name[-2:] == '.b'
    x = x.detach().view(-1)
    p = torch.zeros_like(x)
    prob_1 = x.sum() / x.numel()
    p[...] = prob_1
    p_u = 1 - p.unsqueeze(-1)
    p_0 = torch.zeros_like(p_u)
    p_1 = torch.ones_like(p_u)
    output_cdf = torch.cat([p_0, p_u, p_1], dim=-1)
    sym = torch.floor(x)
    byte_stream_torch, cnt_torch = arithmetic.arithmetic_encode(sym, output_cdf, chunk_size_cuda, int(output_cdf.shape[0]), int(output_cdf.shape[1]))
    cnt_bytes = cnt_torch.cpu().numpy().tobytes()
    byte_stream_bytes = byte_stream_torch.cpu().numpy().tobytes()
    len_cnt_bytes = len(cnt_bytes)
    with open(file_name, 'wb') as fout:
        fout.write(prob_1.cpu().numpy().tobytes())
        fout.write(np.array([len_cnt_bytes]).astype(np.int32).tobytes())
        fout.write(cnt_bytes)
        fout.write(byte_stream_bytes)
    bit_len = (len(byte_stream_bytes) + len(cnt_bytes)) * 8 + 32 * 2
    return bit_len


def encoder_factorized(x, lower_func, Q: 'float'=1, file_name='tmp.b'):
    """
    The reason why int(max_value.item()) + 1 or int(max_value.item()) + 1 + 1:
    first 1: range does not include the last value, so +1
    second 1: if directly calculate, we need to use samples - 0.5, in order to include the whole value space,
              the max bound value after -0.5 should be max_value+0.5.

    Here we do not add the second 1, because we use pmf to calculate cdf, instead of directly calculate cdf

    example in here ("`" means sample-0.5 places, "|" means sample places):
                 `  `  `  `                          `  `  `  `                           `  `  `  `  `
    lkl_lower      |  |  |  |       lkl_upper      |  |  |  |         ->    cdf_lower      |  |  |  |

    example in other place ("`" means sample-0.5 places, "|" means sample places):
                  `  `  `  `  `
    cdf_lower      |  |  |  |

    """
    assert file_name.endswith('.b')
    assert len(x.shape) == 2
    x_int_round = torch.round(x / Q)
    max_value = x_int_round.max()
    min_value = x_int_round.min()
    samples = torch.tensor(range(int(min_value.item()), int(max_value.item()) + 1)).to(torch.float)
    samples = samples.unsqueeze(0).unsqueeze(0).repeat(x.shape[-1], 1, 1)
    lower = lower_func((samples - 0.5) * Q, stop_gradient=False)
    upper = lower_func((samples + 0.5) * Q, stop_gradient=False)
    sign = -torch.sign(torch.add(lower, upper))
    sign = sign.detach()
    pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
    cdf = torch.cumsum(pmf, dim=-1)
    lower = torch.cat([torch.zeros_like(cdf[..., 0:1]), cdf], dim=-1)
    lower = lower.permute(1, 0, 2).contiguous().repeat(x.shape[0], 1, 1)
    x_int_round_idx = x_int_round - min_value
    x_int_round_idx = x_int_round_idx.view(-1)
    lower = lower.view(x_int_round_idx.shape[0], -1)
    lower = torch.clamp(lower, min=0.0, max=1.0)
    assert (x_int_round_idx == x_int_round.view(-1) - min_value).all()
    byte_stream_torch, cnt_torch = arithmetic.arithmetic_encode(x_int_round_idx, lower, chunk_size_cuda, int(lower.shape[0]), int(lower.shape[1]))
    cnt_bytes = cnt_torch.cpu().numpy().tobytes()
    byte_stream_bytes = byte_stream_torch.cpu().numpy().tobytes()
    len_cnt_bytes = len(cnt_bytes)
    with open(file_name, 'wb') as fout:
        fout.write(min_value.cpu().numpy().tobytes())
        fout.write(max_value.cpu().numpy().tobytes())
        fout.write(np.array([len_cnt_bytes]).astype(np.int32).tobytes())
        fout.write(cnt_bytes)
        fout.write(byte_stream_bytes)
    bit_len = (len(byte_stream_bytes) + len(cnt_bytes)) * 8 + 32 * 3
    return bit_len


def encoder_factorized_chunk(x, lower_func, Q: 'float'=1, file_name='tmp.b', chunk_size=10000000):
    assert file_name.endswith('.b')
    assert len(x.shape) == 2
    N = x.shape[0]
    chunks = int(np.ceil(N / chunk_size))
    bit_len_list = []
    for c in range(chunks):
        bit_len = encoder_factorized(x=x[c * chunk_size:c * chunk_size + chunk_size], lower_func=lower_func, Q=Q, file_name=file_name.replace('.b', f'_{str(c)}.b'))
        bit_len_list.append(bit_len)
    return sum(bit_len_list)


def encoder_gaussian_mixed(x, mean_list, scale_list, prob_list, Q, file_name='tmp.b'):
    assert file_name.endswith('.b')
    assert len(x.shape) == 1
    if not isinstance(Q, torch.Tensor):
        Q = torch.tensor([Q], dtype=x.dtype, device=x.device).repeat(x.shape[0])
    assert x.shape == mean_list[0].shape == scale_list[0].shape == prob_list[0].shape == Q.shape, f'{x.shape}, {mean_list[0].shape}, {scale_list[0].shape}, {prob_list[0].shape}, {Q.shape}'
    x_int_round = torch.round(x / Q)
    max_value = x_int_round.max()
    min_value = x_int_round.min()
    lower_all = int(0)
    for mean, scale, prob in zip(mean_list, scale_list, prob_list):
        lower = arithmetic.calculate_cdf(mean, scale, Q, min_value, max_value) * prob.unsqueeze(-1)
        if isinstance(lower_all, int):
            lower_all = lower
        else:
            lower_all += lower
    lower = torch.clamp(lower_all, min=0.0, max=1.0)
    del mean
    del scale
    del prob
    x_int_round_idx = x_int_round - min_value
    byte_stream_torch, cnt_torch = arithmetic.arithmetic_encode(x_int_round_idx, lower, chunk_size_cuda, int(lower.shape[0]), int(lower.shape[1]))
    cnt_bytes = cnt_torch.cpu().numpy().tobytes()
    byte_stream_bytes = byte_stream_torch.cpu().numpy().tobytes()
    len_cnt_bytes = len(cnt_bytes)
    with open(file_name, 'wb') as fout:
        fout.write(min_value.cpu().numpy().tobytes())
        fout.write(max_value.cpu().numpy().tobytes())
        fout.write(np.array([len_cnt_bytes]).astype(np.int32).tobytes())
        fout.write(cnt_bytes)
        fout.write(byte_stream_bytes)
    bit_len = (len(byte_stream_bytes) + len(cnt_bytes)) * 8 + 32 * 3
    return bit_len


def encoder_gaussian_mixed_chunk(x, mean_list, scale_list, prob_list, Q, file_name='tmp.b', chunk_size=10000000):
    assert file_name.endswith('.b')
    assert len(x.shape) == 1
    x_view = x.view(-1)
    mean_list_view = [mean.view(-1) for mean in mean_list]
    scale_list_view = [scale.view(-1) for scale in scale_list]
    prob_list_view = [prob.view(-1) for prob in prob_list]
    assert x_view.shape[0] == mean_list_view[0].shape[0] == scale_list_view[0].shape[0] == prob_list_view[0].shape[0]
    N = x_view.shape[0]
    chunks = int(np.ceil(N / chunk_size))
    Is_Q_tensor = isinstance(Q, torch.Tensor)
    if Is_Q_tensor:
        Q_view = Q.view(-1)
    bit_len_list = []
    for c in range(chunks):
        bit_len = encoder_gaussian_mixed(x=x_view[c * chunk_size:c * chunk_size + chunk_size], mean_list=[mean[c * chunk_size:c * chunk_size + chunk_size] for mean in mean_list_view], scale_list=[scale[c * chunk_size:c * chunk_size + chunk_size] for scale in scale_list_view], prob_list=[prob[c * chunk_size:c * chunk_size + chunk_size] for prob in prob_list_view], Q=Q_view[c * chunk_size:c * chunk_size + chunk_size] if Is_Q_tensor else Q, file_name=file_name.replace('.b', f'_{str(c)}.b'))
        bit_len_list.append(bit_len)
    return sum(bit_len_list)


def normalize_xyz(xyz_orig, K=3, means=None, stds=None):
    if means == None:
        xyz_orig = xyz_orig.detach()
        means = torch.mean(xyz_orig, dim=0, keepdim=True)
        stds = torch.std(xyz_orig, dim=0, keepdim=True)
    lower_bound = means - K * stds
    upper_bound = means + K * stds
    norm_xyz = (xyz_orig - lower_bound) / (upper_bound - lower_bound)
    norm_xyz_clamp = torch.clamp(norm_xyz, min=0, max=1)
    mask_xyz = torch.all((norm_xyz == norm_xyz_clamp) + 0.0, dim=1) + 0.0
    return norm_xyz, norm_xyz_clamp, mask_xyz


def sorted_orig_voxels(means, other_params=None):
    means = means.detach().cpu().numpy().astype(np.float32)
    voxelized_means, means_min, means_max = voxelize(means=means)
    voxelized_means, other_params = sorted_voxels(voxelized_means=voxelized_means, other_params=other_params)
    means = devoxelize(voxelized_means=voxelized_means, means_min=means_min, means_max=means_max)
    means = torch.from_numpy(means).cuda()
    return means, other_params


class FCGS(nn.Module):

    def __init__(self, fea_dim=56, hidden=256, lat_dim=256, grid_dim=48, Q=1, Q_fe=0.001, Q_op=0.001, Q_sc=0.01, Q_ro=1e-05, resolutions_list=[300, 400, 500], resolutions_list_3D=[60, 80, 100], num_dim=3, norm_radius=3, binary=0):
        super().__init__()
        self.norm_radius = norm_radius
        self.binary = binary
        self.freq_enc = FreqEncoder(3, 4)
        assert len(resolutions_list) == len(resolutions_list_3D)
        n_levels = len(resolutions_list)
        self.Encoder_mask = nn.Sequential(nn.Linear(fea_dim, hidden), nn.LeakyReLU(inplace=True), nn.Linear(hidden, hidden), nn.LeakyReLU(inplace=True), nn.Linear(hidden, 1), nn.Sigmoid())
        self.Encoder_fea = nn.Sequential(nn.Linear(fea_dim, hidden), nn.LeakyReLU(inplace=True), nn.Linear(hidden, hidden), nn.LeakyReLU(inplace=True), nn.Linear(hidden, hidden), nn.LeakyReLU(inplace=True), nn.Linear(hidden, lat_dim))
        self.Decoder_fea = nn.Sequential(nn.Linear(lat_dim, hidden), nn.LeakyReLU(inplace=True), nn.Linear(hidden, hidden), nn.LeakyReLU(inplace=True), nn.Linear(hidden, hidden), nn.LeakyReLU(inplace=True), nn.Linear(hidden, hidden))
        self.head_f_dc = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(inplace=True), nn.Linear(hidden, 3))
        nn.init.constant_(self.head_f_dc[-1].weight, 0)
        nn.init.constant_(self.head_f_dc[-1].bias, 0)
        self.head_f_rst = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(inplace=True), nn.Linear(hidden, 45))
        nn.init.constant_(self.head_f_rst[-1].weight, 0)
        nn.init.constant_(self.head_f_rst[-1].bias, 0)
        self.latdim_2_griddim_fea = nn.Sequential(nn.Linear(lat_dim, grid_dim))
        self.resolutions_list, self.offsets_list = self.get_offsets(resolutions_list, dim=2)
        self.resolutions_list_3D, self.offsets_list_3D = self.get_offsets(resolutions_list_3D, dim=3)
        self.cafea_indim = 48 * (len(resolutions_list) * 3 + len(resolutions_list_3D)) + self.freq_enc.output_dim
        self.context_analyzer_fea = nn.Sequential(nn.Linear(self.cafea_indim, 256), nn.LeakyReLU(inplace=True), nn.Linear(256, 256), nn.LeakyReLU(inplace=True), nn.Linear(256, 256 * 3))
        self.cafeq_indim = 48 * (len(resolutions_list) * 3 + len(resolutions_list_3D)) + self.freq_enc.output_dim
        self.context_analyzer_feq = nn.Sequential(nn.Linear(self.cafeq_indim, 48 * 12), nn.LeakyReLU(inplace=True), nn.Linear(48 * 12, 48 * 12), nn.LeakyReLU(inplace=True), nn.Linear(48 * 12, 48 * 3))
        self.cageo_indim = 8 * (len(resolutions_list) * 3 + len(resolutions_list_3D)) + self.freq_enc.output_dim
        self.context_analyzer_geo = nn.Sequential(nn.Linear(self.cageo_indim, 8 * 12), nn.LeakyReLU(inplace=True), nn.Linear(8 * 12, 8 * 12), nn.LeakyReLU(inplace=True), nn.Linear(8 * 12, 8 * 3))
        self.feq_channel_ctx = Channel_CTX_feq()
        self.fea_channel_ctx = Channel_CTX_fea()
        self.feq_spatial_ctx = Spatial_CTX(self.resolutions_list_3D, self.offsets_list_3D, self.resolutions_list, self.offsets_list)
        self.Encoder_fea_hyp = nn.Sequential(nn.Linear(lat_dim, lat_dim // 2), nn.LeakyReLU(inplace=True), nn.Linear(lat_dim // 2, lat_dim // 2), nn.LeakyReLU(inplace=True), nn.Linear(lat_dim // 2, lat_dim // 4))
        self.Decoder_fea_hyp = nn.Sequential(nn.Linear(lat_dim // 4, lat_dim // 2), nn.LeakyReLU(inplace=True), nn.Linear(lat_dim // 2, lat_dim), nn.LeakyReLU(inplace=True), nn.Linear(lat_dim, lat_dim * 3))
        self.Encoder_feq_hyp = nn.Sequential(nn.Linear(48, 48 // 2), nn.LeakyReLU(inplace=True), nn.Linear(48 // 2, 48 // 2), nn.LeakyReLU(inplace=True), nn.Linear(48 // 2, 48 // 2))
        self.Decoder_feq_hyp = nn.Sequential(nn.Linear(48 // 2, 48 // 2), nn.LeakyReLU(inplace=True), nn.Linear(48 // 2, 48), nn.LeakyReLU(inplace=True), nn.Linear(48, 48 * 3))
        self.Encoder_geo_hyp = nn.Sequential(nn.Linear(8, 16), nn.LeakyReLU(inplace=True), nn.Linear(16, 16), nn.LeakyReLU(inplace=True), nn.Linear(16, 16))
        self.Decoder_geo_hyp = nn.Sequential(nn.Linear(16, 16), nn.LeakyReLU(inplace=True), nn.Linear(16, 16), nn.LeakyReLU(inplace=True), nn.Linear(16, 8 * 3))
        self.Q = Q
        self.Q_fe = Q_fe
        self.Q_op = Q_op
        self.Q_sc = Q_sc
        self.Q_ro = Q_ro
        self.EF_fea = Entropy_factorized(lat_dim // 4, Q=Q)
        self.EF_feq = Entropy_factorized(48 // 2, Q=Q)
        self.EF_geo = Entropy_factorized(16, Q=Q)
        self.EF_op = Entropy_factorized(1, Q=Q)
        self.EF_sc = Entropy_factorized(3, Q=Q)
        self.EF_ro = Entropy_factorized(4, Q=Q)
        self.EG = Entropy_gaussian(Q=Q)
        self.EG_mix_prob_2 = Entropy_gaussian_mix_prob_2(Q=Q)
        self.EG_mix_prob_3 = Entropy_gaussian_mix_prob_3(Q=Q)
        self.ad_fe = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))
        self.ad_op = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))
        self.ad_sc = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))
        self.ad_ro = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))

    def get_offsets(self, resolutions_list, dim=3):
        offsets_list = [0]
        offsets = 0
        for resolution in resolutions_list:
            offset = resolution ** dim
            offsets_list.append(offsets + offset)
            offsets += offset
        offsets_list = torch.tensor(offsets_list, device='cuda', dtype=torch.int)
        resolutions_list = torch.tensor(resolutions_list, device='cuda', dtype=torch.int)
        return resolutions_list, offsets_list

    def clamp(self, x, Q):
        x_mean = x.mean().detach()
        x_min = x_mean - 15000 * Q
        x_max = x_mean + 15000 * Q
        x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
        return x

    def quantize(self, x, Q, testing):
        if not testing:
            x_q = x + torch.empty_like(x).uniform_(-0.5, 0.5) * Q
        else:
            x_q = STE_multistep.apply(x, Q)
        x_q = self.clamp(x_q, Q)
        return x_q

    def compress_only(self, g_xyz, g_fea, means=None, stds=None, testing=True, root_path='./', chunk_size_list=(), feqonly=False, random_seed=1):
        c_size_fea, c_size_feq, c_size_geo = chunk_size_list
        g_xyz, g_fea = sorted_orig_voxels(g_xyz, g_fea)
        None
        bits_xyz = compress_gaussian_params(gaussian_params=g_xyz, bin_path=os.path.join(root_path, 'xyz_gpcc.bin'))[-1]['file_size']['total'] * 8 * 1024 * 1024
        torch.manual_seed(random_seed)
        shuffled_indices = torch.randperm(g_xyz.size(0))
        g_xyz = g_xyz[shuffled_indices]
        g_fea = g_fea[shuffled_indices]
        fe, op, sc, ro = torch.split(g_fea, split_size_or_sections=[3 + 45, 1, 3, 4], dim=-1)
        norm_xyz, norm_xyz_clamp, mask_xyz = normalize_xyz(g_xyz, K=self.norm_radius, means=means, stds=stds)
        freq_enc_xyz = self.freq_enc(norm_xyz_clamp)
        N_g = g_xyz.shape[0]
        mask_sig = self.Encoder_mask(g_fea)
        mask = ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig
        mask_fea = mask.detach()[:, 0]
        mask_feq = torch.logical_not(mask_fea)
        s0 = 0
        s1 = N_g // 6 * 1
        s2 = N_g // 6 * 2
        s3 = N_g // 6 * 4
        s4 = N_g
        sn = [s0, s1, s2, s3, s4]
        k0 = 0
        k1 = int(mask_fea[s0:s1].sum().item())
        k2 = int(mask_fea[s0:s2].sum().item())
        k3 = int(mask_fea[s0:s3].sum().item())
        k4 = int(mask_fea[s0:s4].sum().item())
        kn = [k0, k1, k2, k3, k4]
        t0 = 0
        t1 = int(mask_feq[s0:s1].sum().item())
        t2 = int(mask_feq[s0:s2].sum().item())
        t3 = int(mask_feq[s0:s3].sum().item())
        t4 = int(mask_feq[s0:s4].sum().item())
        tn = [t0, t1, t2, t3, t4]
        flag_remask = False
        if k1 == k0 or k2 == k1 or k3 == k2 or k4 == k3:
            mask = torch.zeros_like(mask)
            flag_remask = True
        elif t1 == t0 or t2 == t1 or t3 == t2 or t4 == t3:
            mask = torch.ones_like(mask)
            flag_remask = True
        if feqonly:
            mask = torch.zeros_like(mask)
            flag_remask = True
        if flag_remask:
            mask_fea = mask.detach()[:, 0]
            mask_feq = torch.logical_not(mask_fea)
            k0 = 0
            k1 = int(mask_fea[s0:s1].sum().item())
            k2 = int(mask_fea[s0:s2].sum().item())
            k3 = int(mask_fea[s0:s3].sum().item())
            k4 = int(mask_fea[s0:s4].sum().item())
            kn = [k0, k1, k2, k3, k4]
            t0 = 0
            t1 = int(mask_feq[s0:s1].sum().item())
            t2 = int(mask_feq[s0:s2].sum().item())
            t3 = int(mask_feq[s0:s3].sum().item())
            t4 = int(mask_feq[s0:s4].sum().item())
            tn = [t0, t1, t2, t3, t4]
        bits_mask = encoder(x=mask, file_name=os.path.join(root_path, 'mask.b'))
        g_fea_enc = self.Encoder_fea(g_fea[mask_fea])
        g_fea_enc_q = self.quantize(g_fea_enc, self.Q, testing)
        g_fea_out = self.Decoder_fea(g_fea_enc_q)
        fe_dec = torch.cat([self.head_f_dc(g_fea_out), self.head_f_rst(g_fea_out)], dim=-1)
        Q_fe = (self.Q_fe * self.ad_fe[:, 0:1] + self.ad_fe[:, 1:2]) * (1 + torch.tanh(self.ad_fe[:, 2:3]))
        Q_op = (self.Q_op * self.ad_op[:, 0:1] + self.ad_op[:, 1:2]) * (1 + torch.tanh(self.ad_op[:, 2:3]))
        Q_sc = (self.Q_sc * self.ad_sc[:, 0:1] + self.ad_sc[:, 1:2]) * (1 + torch.tanh(self.ad_sc[:, 2:3]))
        Q_ro = (self.Q_ro * self.ad_ro[:, 0:1] + self.ad_ro[:, 1:2]) * (1 + torch.tanh(self.ad_ro[:, 2:3]))
        Q_fe = Q_fe.repeat(mask_feq.sum(), 48)
        Q_op = Q_op.repeat(N_g, 1)
        Q_sc = Q_sc.repeat(N_g, 3)
        Q_ro = Q_ro.repeat(N_g, 4)
        fe_q = self.quantize(fe[mask_feq], Q_fe, testing)
        op_q = self.quantize(op, Q_op, testing)
        sc_q = self.quantize(sc, Q_sc, testing)
        ro_q = self.quantize(ro, Q_ro, testing)
        fe_final = torch.zeros([N_g, 48], dtype=torch.float32, device='cuda')
        fe_final[mask_fea] = fe_dec
        fe_final[mask_feq] = fe_q
        geo_q = torch.cat([op_q, sc_q, ro_q], dim=-1)
        Q_geo = torch.cat([Q_op, Q_sc, Q_ro], dim=-1)
        if mask_fea.sum() > 0:
            None
            g_fea_enc_q_hyp = self.Encoder_fea_hyp(g_fea_enc_q)
            g_fea_enc_q_hyp_q = self.quantize(g_fea_enc_q_hyp, self.Q, testing)
            fea_grid_feature = self.latdim_2_griddim_fea(g_fea_enc_q)
            ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1][mask_fea[s0:s1]], norm_xyz_clamp[s1:s2][mask_fea[s1:s2]], fea_grid_feature[k0:k1])
            ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2][mask_fea[s0:s2]], norm_xyz_clamp[s2:s3][mask_fea[s2:s3]], fea_grid_feature[k0:k2])
            ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3][mask_fea[s0:s3]], norm_xyz_clamp[s3:s4][mask_fea[s3:s4]], fea_grid_feature[k0:k3])
            ctx_s1 = torch.zeros(size=[k1 - k0, ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)
            ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz[mask_fea]], dim=-1)
            mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_fea(ctx_s1234), split_size_or_sections=[256, 256, 256], dim=-1)
            mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_fea_hyp(g_fea_enc_q_hyp_q), split_size_or_sections=[256, 256, 256], dim=-1)
            mean_ch, scale_ch, prob_ch = self.fea_channel_ctx.forward(g_fea_enc_q)
            probs = torch.stack([prob_sp, prob_hp, prob_ch], dim=-1)
            probs = torch.softmax(probs, dim=-1)
            prob_sp, prob_hp, prob_ch = probs[..., 0], probs[..., 1], probs[..., 2]
            bits_fea_hyp = encoder_factorized_chunk(x=g_fea_enc_q_hyp_q, lower_func=self.EF_fea._logits_cumulative, Q=self.Q, file_name=os.path.join(root_path, 'g_fea_enc_q_hyp_q.b'))
            bits_fea_main = 0
            for l_sp in range(4):
                k_st = kn[l_sp]
                k_ed = kn[l_sp + 1]
                for l_ch in range(4):
                    c_st = l_ch * 64
                    c_ed = l_ch * 64 + 64
                    mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[k_st:k_ed, c_st:c_ed], scale_sp[k_st:k_ed, c_st:c_ed], prob_sp[k_st:k_ed, c_st:c_ed]
                    mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[k_st:k_ed, c_st:c_ed], scale_hp[k_st:k_ed, c_st:c_ed], prob_hp[k_st:k_ed, c_st:c_ed]
                    mean_ch_l, scale_ch_l, prob_ch_l = mean_ch[k_st:k_ed, c_st:c_ed], scale_ch[k_st:k_ed, c_st:c_ed], prob_ch[k_st:k_ed, c_st:c_ed]
                    bits_fea_main_tmp = encoder_gaussian_mixed_chunk(x=g_fea_enc_q[k_st:k_ed, c_st:c_ed].contiguous().view(-1), mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)], scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)], prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)], Q=self.Q, file_name=os.path.join(root_path, f'g_fea_enc_q_sp{l_sp}_ch{l_ch}.b'), chunk_size=c_size_fea)
                    bits_fea_main += bits_fea_main_tmp
            bits_fea = bits_fea_main + bits_fea_hyp
        else:
            bits_fea_main = 0
            bits_fea_hyp = 0
            bits_fea = 0
        if mask_feq.sum() > 0:
            None
            fe_q_hyp = self.Encoder_feq_hyp(fe_q)
            fe_q_hyp_q = self.quantize(fe_q_hyp, self.Q, testing)
            ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1], norm_xyz_clamp[s1:s2][mask_feq[s1:s2]], fe_final[s0:s1])
            ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2], norm_xyz_clamp[s2:s3][mask_feq[s2:s3]], fe_final[s0:s2])
            ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3], norm_xyz_clamp[s3:s4][mask_feq[s3:s4]], fe_final[s0:s3])
            ctx_s1 = torch.zeros(size=[mask_feq[s0:s1].sum(), ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)
            ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz[mask_feq]], dim=-1)
            mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_feq(ctx_s1234), split_size_or_sections=[48, 48, 48], dim=-1)
            mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_feq_hyp(fe_q_hyp_q), split_size_or_sections=[48, 48, 48], dim=-1)
            mean_ch, scale_ch, prob_ch = self.feq_channel_ctx.forward(fe_q)
            probs = torch.stack([prob_sp, prob_hp, prob_ch], dim=-1)
            probs = torch.softmax(probs, dim=-1)
            prob_sp, prob_hp, prob_ch = probs[..., 0], probs[..., 1], probs[..., 2]
            bits_feq_hyp = encoder_factorized_chunk(x=fe_q_hyp_q, lower_func=self.EF_feq._logits_cumulative, Q=self.Q, file_name=os.path.join(root_path, 'fe_q_hyp_q.b'))
            bits_feq_main = 0
            for l_sp in range(4):
                t_st = tn[l_sp]
                t_ed = tn[l_sp + 1]
                for l_ch in range(3):
                    mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[t_st:t_ed, l_ch::3], scale_sp[t_st:t_ed, l_ch::3], prob_sp[t_st:t_ed, l_ch::3]
                    mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[t_st:t_ed, l_ch::3], scale_hp[t_st:t_ed, l_ch::3], prob_hp[t_st:t_ed, l_ch::3]
                    mean_ch_l, scale_ch_l, prob_ch_l = mean_ch[t_st:t_ed, l_ch::3], scale_ch[t_st:t_ed, l_ch::3], prob_ch[t_st:t_ed, l_ch::3]
                    bits_feq_main_tmp = encoder_gaussian_mixed_chunk(x=fe_q[t_st:t_ed, l_ch::3].contiguous().view(-1), mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)], scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)], prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)], Q=Q_fe[t_st:t_ed, l_ch::3].contiguous().view(-1), file_name=os.path.join(root_path, f'fe_q_sp{l_sp}_ch{l_ch}.b'), chunk_size=c_size_feq)
                    bits_feq_main += bits_feq_main_tmp
            bits_feq = bits_feq_main + bits_feq_hyp
        else:
            bits_feq_main = 0
            bits_feq_hyp = 0
            bits_feq = 0
        None
        geo_q_hyp = self.Encoder_geo_hyp(geo_q)
        geo_q_hyp_q = self.quantize(geo_q_hyp, self.Q, testing)
        ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1], norm_xyz_clamp[s1:s2], geo_q[s0:s1])
        ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2], norm_xyz_clamp[s2:s3], geo_q[s0:s2])
        ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3], norm_xyz_clamp[s3:s4], geo_q[s0:s3])
        ctx_s1 = torch.zeros(size=[s1 - s0, ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)
        ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz], dim=-1)
        mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_geo(ctx_s1234), split_size_or_sections=[8, 8, 8], dim=-1)
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_geo_hyp(geo_q_hyp_q), split_size_or_sections=[8, 8, 8], dim=-1)
        probs = torch.stack([prob_sp, prob_hp], dim=-1)
        probs = torch.softmax(probs, dim=-1)
        prob_sp, prob_hp = probs[..., 0], probs[..., 1]
        bits_geo_hyp = encoder_factorized_chunk(x=geo_q_hyp_q, lower_func=self.EF_geo._logits_cumulative, Q=self.Q, file_name=os.path.join(root_path, 'geo_q_hyp_q.b'))
        bits_geo_main = 0
        for l_sp in range(4):
            s_st = sn[l_sp]
            s_ed = sn[l_sp + 1]
            mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[s_st:s_ed], scale_sp[s_st:s_ed], prob_sp[s_st:s_ed]
            mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[s_st:s_ed], scale_hp[s_st:s_ed], prob_hp[s_st:s_ed]
            bits_geo_main_tmp = encoder_gaussian_mixed_chunk(x=geo_q[s_st:s_ed].contiguous().view(-1), mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1)], scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1)], prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1)], Q=Q_geo[s_st:s_ed].contiguous().view(-1), file_name=os.path.join(root_path, f'geo_q_sp{l_sp}.b'), chunk_size=c_size_geo)
            bits_geo_main += bits_geo_main_tmp
        bits_geo = bits_geo_main + bits_geo_hyp
        bits = bits_fea + bits_feq + bits_geo + bits_xyz + bits_mask
        g_fea_out = torch.cat([fe_final, geo_q], dim=-1)
        return g_xyz, g_fea_out, mask, bits / b2M, bits_xyz / b2M, bits_mask / b2M, bits_fea / b2M, bits_fea_main / b2M, bits_fea_hyp / b2M, bits_feq / b2M, bits_feq_main / b2M, bits_feq_hyp / b2M, bits_geo / b2M, bits_geo_main / b2M, bits_geo_hyp / b2M

    def compress(self, g_xyz, g_fea, means=None, stds=None, testing=True, root_path='./', chunk_size_list=(), determ_codec=False):
        c_size_fea, c_size_feq, c_size_geo = chunk_size_list
        g_xyz, g_fea = sorted_orig_voxels(g_xyz, g_fea)
        None
        bits_xyz = compress_gaussian_params(gaussian_params=g_xyz, bin_path=os.path.join(root_path, 'xyz_gpcc.bin'))[-1]['file_size']['total'] * 8 * 1024 * 1024
        torch.manual_seed(1)
        shuffled_indices = torch.randperm(g_xyz.size(0))
        g_xyz = g_xyz[shuffled_indices]
        g_fea = g_fea[shuffled_indices]
        fe, op, sc, ro = torch.split(g_fea, split_size_or_sections=[3 + 45, 1, 3, 4], dim=-1)
        norm_xyz, norm_xyz_clamp, mask_xyz = normalize_xyz(g_xyz, K=self.norm_radius, means=means, stds=stds)
        freq_enc_xyz = self.freq_enc(norm_xyz_clamp)
        N_g = g_xyz.shape[0]
        mask_sig = self.Encoder_mask(g_fea)
        mask = ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig
        bits_mask = encoder(x=mask, file_name=os.path.join(root_path, 'mask.b'))
        mask_fea = mask.detach()[:, 0]
        mask_feq = torch.logical_not(mask_fea)
        g_fea_enc = self.Encoder_fea(g_fea[mask_fea])
        g_fea_enc_q = self.quantize(g_fea_enc, self.Q, testing)
        g_fea_out = self.Decoder_fea(g_fea_enc_q)
        fe_dec = torch.cat([self.head_f_dc(g_fea_out), self.head_f_rst(g_fea_out)], dim=-1)
        Q_fe = (self.Q_fe * self.ad_fe[:, 0:1] + self.ad_fe[:, 1:2]) * (1 + torch.tanh(self.ad_fe[:, 2:3]))
        Q_op = (self.Q_op * self.ad_op[:, 0:1] + self.ad_op[:, 1:2]) * (1 + torch.tanh(self.ad_op[:, 2:3]))
        Q_sc = (self.Q_sc * self.ad_sc[:, 0:1] + self.ad_sc[:, 1:2]) * (1 + torch.tanh(self.ad_sc[:, 2:3]))
        Q_ro = (self.Q_ro * self.ad_ro[:, 0:1] + self.ad_ro[:, 1:2]) * (1 + torch.tanh(self.ad_ro[:, 2:3]))
        Q_fe = Q_fe.repeat(mask_feq.sum(), 48)
        Q_op = Q_op.repeat(N_g, 1)
        Q_sc = Q_sc.repeat(N_g, 3)
        Q_ro = Q_ro.repeat(N_g, 4)
        fe_q = self.quantize(fe[mask_feq], Q_fe, testing)
        op_q = self.quantize(op, Q_op, testing)
        sc_q = self.quantize(sc, Q_sc, testing)
        ro_q = self.quantize(ro, Q_ro, testing)
        fe_final = torch.zeros([N_g, 48], dtype=torch.float32, device='cuda')
        fe_final[mask_fea] = fe_dec
        fe_final[mask_feq] = fe_q
        geo_q = torch.cat([op_q, sc_q, ro_q], dim=-1)
        Q_geo = torch.cat([Q_op, Q_sc, Q_ro], dim=-1)
        s0 = 0
        s1 = N_g // 6 * 1
        s2 = N_g // 6 * 2
        s3 = N_g // 6 * 4
        s4 = N_g
        sn = [s0, s1, s2, s3, s4]
        k0 = 0
        k1 = int(mask_fea[s0:s1].sum().item())
        k2 = int(mask_fea[s0:s2].sum().item())
        k3 = int(mask_fea[s0:s3].sum().item())
        k4 = int(mask_fea[s0:s4].sum().item())
        kn = [k0, k1, k2, k3, k4]
        t0 = 0
        t1 = int(mask_feq[s0:s1].sum().item())
        t2 = int(mask_feq[s0:s2].sum().item())
        t3 = int(mask_feq[s0:s3].sum().item())
        t4 = int(mask_feq[s0:s4].sum().item())
        tn = [t0, t1, t2, t3, t4]
        g_fea_enc_q_hyp = self.Encoder_fea_hyp(g_fea_enc_q)
        g_fea_enc_q_hyp_q = self.quantize(g_fea_enc_q_hyp, self.Q, testing)
        fea_grid_feature = self.latdim_2_griddim_fea(g_fea_enc_q)
        ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1][mask_fea[s0:s1]], norm_xyz_clamp[s1:s2][mask_fea[s1:s2]], fea_grid_feature[k0:k1], determ=determ_codec)
        ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2][mask_fea[s0:s2]], norm_xyz_clamp[s2:s3][mask_fea[s2:s3]], fea_grid_feature[k0:k2], determ=determ_codec)
        ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3][mask_fea[s0:s3]], norm_xyz_clamp[s3:s4][mask_fea[s3:s4]], fea_grid_feature[k0:k3], determ=determ_codec)
        ctx_s1 = torch.zeros(size=[k1 - k0, ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)
        ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz[mask_fea]], dim=-1)
        mean_sp_list = []
        scale_sp_list = []
        prob_sp_list = []
        for iii in range(4):
            mean_sp_k, scale_sp_k, prob_sp_k = torch.split(self.context_analyzer_fea(ctx_s1234[kn[iii]:kn[iii + 1]]), split_size_or_sections=[256, 256, 256], dim=-1)
            mean_sp_list.append(mean_sp_k)
            scale_sp_list.append(scale_sp_k)
            prob_sp_list.append(prob_sp_k)
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_fea_hyp(g_fea_enc_q_hyp_q), split_size_or_sections=[256, 256, 256], dim=-1)
        mean_ch, scale_ch, prob_ch = self.fea_channel_ctx.forward(g_fea_enc_q)
        None
        bits_fea_hyp = encoder_factorized_chunk(x=g_fea_enc_q_hyp_q, lower_func=self.EF_fea._logits_cumulative, Q=self.Q, file_name=os.path.join(root_path, 'g_fea_enc_q_hyp_q.b'))
        bits_fea_main = 0
        for l_sp in range(4):
            k_st = kn[l_sp]
            k_ed = kn[l_sp + 1]
            mean_sp = mean_sp_list[l_sp]
            scale_sp = scale_sp_list[l_sp]
            prob_sp = prob_sp_list[l_sp]
            for l_ch in range(4):
                c_st = l_ch * 64
                c_ed = l_ch * 64 + 64
                mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[:, c_st:c_ed], scale_sp[:, c_st:c_ed], prob_sp[:, c_st:c_ed]
                mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[k_st:k_ed, c_st:c_ed], scale_hp[k_st:k_ed, c_st:c_ed], prob_hp[k_st:k_ed, c_st:c_ed]
                mean_ch_l, scale_ch_l, prob_ch_l = mean_ch[k_st:k_ed, c_st:c_ed], scale_ch[k_st:k_ed, c_st:c_ed], prob_ch[k_st:k_ed, c_st:c_ed]
                probs = torch.stack([prob_sp_l, prob_hp_l, prob_ch_l], dim=-1)
                probs = torch.softmax(probs, dim=-1)
                prob_sp_l, prob_hp_l, prob_ch_l = probs[..., 0], probs[..., 1], probs[..., 2]
                bits_fea_main_tmp = encoder_gaussian_mixed_chunk(x=g_fea_enc_q[k_st:k_ed, c_st:c_ed].contiguous().view(-1), mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)], scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)], prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)], Q=self.Q, file_name=os.path.join(root_path, f'g_fea_enc_q_sp{l_sp}_ch{l_ch}.b'), chunk_size=c_size_fea)
                bits_fea_main += bits_fea_main_tmp
        bits_fea = bits_fea_main + bits_fea_hyp
        None
        fe_q_hyp = self.Encoder_feq_hyp(fe_q)
        fe_q_hyp_q = self.quantize(fe_q_hyp, self.Q, testing)
        ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1], norm_xyz_clamp[s1:s2][mask_feq[s1:s2]], fe_final[s0:s1], determ=determ_codec)
        ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2], norm_xyz_clamp[s2:s3][mask_feq[s2:s3]], fe_final[s0:s2], determ=determ_codec)
        ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3], norm_xyz_clamp[s3:s4][mask_feq[s3:s4]], fe_final[s0:s3], determ=determ_codec)
        ctx_s1 = torch.zeros(size=[mask_feq[s0:s1].sum(), ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)
        ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz[mask_feq]], dim=-1)
        mean_sp_list = []
        scale_sp_list = []
        prob_sp_list = []
        for iii in range(4):
            mean_sp_k, scale_sp_k, prob_sp_k = torch.split(self.context_analyzer_feq(ctx_s1234[tn[iii]:tn[iii + 1]]), split_size_or_sections=[48, 48, 48], dim=-1)
            mean_sp_list.append(mean_sp_k)
            scale_sp_list.append(scale_sp_k)
            prob_sp_list.append(prob_sp_k)
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_feq_hyp(fe_q_hyp_q), split_size_or_sections=[48, 48, 48], dim=-1)
        mean_ch, scale_ch, prob_ch = self.feq_channel_ctx.forward(fe_q)
        bits_feq_hyp = encoder_factorized_chunk(x=fe_q_hyp_q, lower_func=self.EF_feq._logits_cumulative, Q=self.Q, file_name=os.path.join(root_path, 'fe_q_hyp_q.b'))
        bits_feq_main = 0
        for l_sp in range(4):
            t_st = tn[l_sp]
            t_ed = tn[l_sp + 1]
            mean_sp = mean_sp_list[l_sp]
            scale_sp = scale_sp_list[l_sp]
            prob_sp = prob_sp_list[l_sp]
            for l_ch in range(3):
                mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[:, l_ch::3], scale_sp[:, l_ch::3], prob_sp[:, l_ch::3]
                mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[t_st:t_ed, l_ch::3], scale_hp[t_st:t_ed, l_ch::3], prob_hp[t_st:t_ed, l_ch::3]
                mean_ch_l, scale_ch_l, prob_ch_l = mean_ch[t_st:t_ed, l_ch::3], scale_ch[t_st:t_ed, l_ch::3], prob_ch[t_st:t_ed, l_ch::3]
                probs = torch.stack([prob_sp_l, prob_hp_l, prob_ch_l], dim=-1)
                probs = torch.softmax(probs, dim=-1)
                prob_sp_l, prob_hp_l, prob_ch_l = probs[..., 0], probs[..., 1], probs[..., 2]
                bits_feq_main_tmp = encoder_gaussian_mixed_chunk(x=fe_q[t_st:t_ed, l_ch::3].contiguous().view(-1), mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)], scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)], prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)], Q=Q_fe[t_st:t_ed, l_ch::3].contiguous().view(-1), file_name=os.path.join(root_path, f'fe_q_sp{l_sp}_ch{l_ch}.b'), chunk_size=c_size_feq)
                bits_feq_main += bits_feq_main_tmp
        bits_feq = bits_feq_main + bits_feq_hyp
        None
        geo_q_hyp = self.Encoder_geo_hyp(geo_q)
        geo_q_hyp_q = self.quantize(geo_q_hyp, self.Q, testing)
        ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1], norm_xyz_clamp[s1:s2], geo_q[s0:s1], determ=determ_codec)
        ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2], norm_xyz_clamp[s2:s3], geo_q[s0:s2], determ=determ_codec)
        ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3], norm_xyz_clamp[s3:s4], geo_q[s0:s3], determ=determ_codec)
        ctx_s1 = torch.zeros(size=[s1 - s0, ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)
        ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz], dim=-1)
        mean_sp_list = []
        scale_sp_list = []
        prob_sp_list = []
        for iii in range(4):
            mean_sp_k, scale_sp_k, prob_sp_k = torch.split(self.context_analyzer_geo(ctx_s1234[sn[iii]:sn[iii + 1]]), split_size_or_sections=[8, 8, 8], dim=-1)
            mean_sp_list.append(mean_sp_k)
            scale_sp_list.append(scale_sp_k)
            prob_sp_list.append(prob_sp_k)
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_geo_hyp(geo_q_hyp_q), split_size_or_sections=[8, 8, 8], dim=-1)
        bits_geo_hyp = encoder_factorized_chunk(x=geo_q_hyp_q, lower_func=self.EF_geo._logits_cumulative, Q=self.Q, file_name=os.path.join(root_path, 'geo_q_hyp_q.b'))
        bits_geo_main = 0
        for l_sp in range(4):
            s_st = sn[l_sp]
            s_ed = sn[l_sp + 1]
            mean_sp = mean_sp_list[l_sp]
            scale_sp = scale_sp_list[l_sp]
            prob_sp = prob_sp_list[l_sp]
            mean_sp_l, scale_sp_l, prob_sp_l = mean_sp, scale_sp, prob_sp
            mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[s_st:s_ed], scale_hp[s_st:s_ed], prob_hp[s_st:s_ed]
            probs = torch.stack([prob_sp_l, prob_hp_l], dim=-1)
            probs = torch.softmax(probs, dim=-1)
            prob_sp_l, prob_hp_l = probs[..., 0], probs[..., 1]
            bits_geo_main_tmp = encoder_gaussian_mixed_chunk(x=geo_q[s_st:s_ed].contiguous().view(-1), mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1)], scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1)], prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1)], Q=Q_geo[s_st:s_ed].contiguous().view(-1), file_name=os.path.join(root_path, f'geo_q_sp{l_sp}.b'), chunk_size=c_size_geo)
            bits_geo_main += bits_geo_main_tmp
        bits_geo = bits_geo_main + bits_geo_hyp
        bits = bits_fea + bits_feq + bits_geo + bits_xyz + bits_mask
        g_fea_out = torch.cat([fe_final, geo_q], dim=-1)
        return g_xyz, g_fea_out, mask, bits / b2M, bits_xyz / b2M, bits_mask / b2M, bits_fea / b2M, bits_fea_main / b2M, bits_fea_hyp / b2M, bits_feq / b2M, bits_feq_main / b2M, bits_feq_hyp / b2M, bits_geo / b2M, bits_geo_main / b2M, bits_geo_hyp / b2M

    def decomprss(self, means=None, stds=None, root_path='./', chunk_size_list=()):
        c_size_fea, c_size_feq, c_size_geo = chunk_size_list
        None
        g_xyz = decompress_gaussian_params(bin_path=os.path.join(root_path, 'xyz_gpcc.bin'))[0]
        torch.manual_seed(1)
        shuffled_indices = torch.randperm(g_xyz.size(0))
        g_xyz = g_xyz[shuffled_indices]
        norm_xyz, norm_xyz_clamp, mask_xyz = normalize_xyz(g_xyz, K=self.norm_radius, means=means, stds=stds)
        freq_enc_xyz = self.freq_enc(norm_xyz_clamp)
        N_g = g_xyz.shape[0]
        mask = decoder(N_len=N_g, file_name=os.path.join(root_path, 'mask.b')).view(N_g, 1)
        mask_fea = mask.detach()[:, 0]
        mask_feq = torch.logical_not(mask_fea)
        Q_fe = (self.Q_fe * self.ad_fe[:, 0:1] + self.ad_fe[:, 1:2]) * (1 + torch.tanh(self.ad_fe[:, 2:3]))
        Q_op = (self.Q_op * self.ad_op[:, 0:1] + self.ad_op[:, 1:2]) * (1 + torch.tanh(self.ad_op[:, 2:3]))
        Q_sc = (self.Q_sc * self.ad_sc[:, 0:1] + self.ad_sc[:, 1:2]) * (1 + torch.tanh(self.ad_sc[:, 2:3]))
        Q_ro = (self.Q_ro * self.ad_ro[:, 0:1] + self.ad_ro[:, 1:2]) * (1 + torch.tanh(self.ad_ro[:, 2:3]))
        Q_fe = Q_fe.repeat(mask_feq.sum(), 48)
        Q_op = Q_op.repeat(N_g, 1)
        Q_sc = Q_sc.repeat(N_g, 3)
        Q_ro = Q_ro.repeat(N_g, 4)
        Q_geo = torch.cat([Q_op, Q_sc, Q_ro], dim=-1)
        s0 = 0
        s1 = N_g // 6 * 1
        s2 = N_g // 6 * 2
        s3 = N_g // 6 * 4
        s4 = N_g
        sn = [s0, s1, s2, s3, s4]
        k0 = 0
        k1 = int(mask_fea[s0:s1].sum().item())
        k2 = int(mask_fea[s0:s2].sum().item())
        k3 = int(mask_fea[s0:s3].sum().item())
        k4 = int(mask_fea[s0:s4].sum().item())
        kn = [k0, k1, k2, k3, k4]
        t0 = 0
        t1 = int(mask_feq[s0:s1].sum().item())
        t2 = int(mask_feq[s0:s2].sum().item())
        t3 = int(mask_feq[s0:s3].sum().item())
        t4 = int(mask_feq[s0:s4].sum().item())
        tn = [t0, t1, t2, t3, t4]
        None
        g_fea_enc_q_hyp_q = decoder_factorized_chunk(lower_func=self.EF_fea._logits_cumulative, Q=self.Q, N_len=mask_fea.sum().item(), dim=64, file_name=os.path.join(root_path, 'g_fea_enc_q_hyp_q.b'))
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_fea_hyp(g_fea_enc_q_hyp_q), split_size_or_sections=[256, 256, 256], dim=-1)
        g_fea_enc_q = torch.zeros(size=[mask_fea.sum(), 256], dtype=torch.float32, device='cuda')
        for l_sp in range(4):
            k_st = kn[l_sp]
            k_ed = kn[l_sp + 1]
            s_st = sn[l_sp]
            s_ed = sn[l_sp + 1]
            if l_sp == 0:
                ctx_sn = torch.zeros(size=[k1 - k0, self.cafea_indim - self.freq_enc.output_dim], dtype=torch.float32, device='cuda')
            else:
                fea_grid_feature_curr = self.latdim_2_griddim_fea(g_fea_enc_q[k0:k_st])
                ctx_sn = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s_st][mask_fea[s0:s_st]], norm_xyz_clamp[s_st:s_ed][mask_fea[s_st:s_ed]], fea_grid_feature_curr, determ=True)
            ctx_sn = torch.cat([ctx_sn, freq_enc_xyz[s_st:s_ed][mask_fea[s_st:s_ed]]], dim=-1)
            mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_fea(ctx_sn), split_size_or_sections=[256, 256, 256], dim=-1)
            for l_ch in range(4):
                c_st = l_ch * 64
                c_ed = l_ch * 64 + 64
                mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[:, c_st:c_ed], scale_sp[:, c_st:c_ed], prob_sp[:, c_st:c_ed]
                mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[k_st:k_ed, c_st:c_ed], scale_hp[k_st:k_ed, c_st:c_ed], prob_hp[k_st:k_ed, c_st:c_ed]
                mean_ch_l, scale_ch_l, prob_ch_l = self.fea_channel_ctx.forward(g_fea_enc_q[k_st:k_ed], to_dec=l_ch)
                probs = torch.stack([prob_sp_l, prob_hp_l, prob_ch_l], dim=-1)
                probs = torch.softmax(probs, dim=-1)
                prob_sp_l, prob_hp_l, prob_ch_l = probs[..., 0], probs[..., 1], probs[..., 2]
                g_fea_enc_q_tmp = decoder_gaussian_mixed_chunk(mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)], scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)], prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)], Q=self.Q, file_name=os.path.join(root_path, f'g_fea_enc_q_sp{l_sp}_ch{l_ch}.b'), chunk_size=c_size_fea).contiguous().view(k_ed - k_st, c_ed - c_st)
                g_fea_enc_q[k_st:k_ed, c_st:c_ed] = g_fea_enc_q_tmp
        g_fea_out = self.Decoder_fea(g_fea_enc_q)
        fe_dec = torch.cat([self.head_f_dc(g_fea_out), self.head_f_rst(g_fea_out)], dim=-1)
        None
        fe_q_hyp_q = decoder_factorized_chunk(lower_func=self.EF_feq._logits_cumulative, Q=self.Q, N_len=mask_feq.sum().item(), dim=24, file_name=os.path.join(root_path, 'fe_q_hyp_q.b'))
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_feq_hyp(fe_q_hyp_q), split_size_or_sections=[48, 48, 48], dim=-1)
        fe_q = torch.zeros(size=[mask_feq.sum(), 48], dtype=torch.float32, device='cuda')
        fe_final = torch.zeros(size=[N_g, 48], dtype=torch.float32, device='cuda')
        fe_final[mask_fea] = fe_dec
        for l_sp in range(4):
            t_st = tn[l_sp]
            t_ed = tn[l_sp + 1]
            s_st = sn[l_sp]
            s_ed = sn[l_sp + 1]
            if l_sp == 0:
                ctx_sn = torch.zeros(size=[t1 - t0, self.cafeq_indim - self.freq_enc.output_dim], dtype=torch.float32, device='cuda')
            else:
                ctx_sn = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s_st], norm_xyz_clamp[s_st:s_ed][mask_feq[s_st:s_ed]], fe_final[s0:s_st], determ=True)
            ctx_sn = torch.cat([ctx_sn, freq_enc_xyz[s_st:s_ed][mask_feq[s_st:s_ed]]], dim=-1)
            mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_feq(ctx_sn), split_size_or_sections=[48, 48, 48], dim=-1)
            for l_ch in range(3):
                mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[:, l_ch::3], scale_sp[:, l_ch::3], prob_sp[:, l_ch::3]
                mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[t_st:t_ed, l_ch::3], scale_hp[t_st:t_ed, l_ch::3], prob_hp[t_st:t_ed, l_ch::3]
                mean_ch_l, scale_ch_l, prob_ch_l = self.feq_channel_ctx.forward(fe_q[t_st:t_ed], to_dec=l_ch)
                probs = torch.stack([prob_sp_l, prob_hp_l, prob_ch_l], dim=-1)
                probs = torch.softmax(probs, dim=-1)
                prob_sp_l, prob_hp_l, prob_ch_l = probs[..., 0], probs[..., 1], probs[..., 2]
                fe_q_tmp = decoder_gaussian_mixed_chunk(mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)], scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)], prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)], Q=Q_fe[t_st:t_ed, l_ch::3].contiguous().view(-1), file_name=os.path.join(root_path, f'fe_q_sp{l_sp}_ch{l_ch}.b'), chunk_size=c_size_feq).contiguous().view(t_ed - t_st, 16)
                fe_q[t_st:t_ed, l_ch::3] = fe_q_tmp
                fe_final[s_st:s_ed][mask_feq[s_st:s_ed], l_ch::3] = fe_q_tmp
        None
        geo_q_hyp_q = decoder_factorized_chunk(lower_func=self.EF_geo._logits_cumulative, Q=self.Q, N_len=N_g, dim=16, file_name=os.path.join(root_path, 'geo_q_hyp_q.b'))
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_geo_hyp(geo_q_hyp_q), split_size_or_sections=[8, 8, 8], dim=-1)
        geo_q = torch.zeros(size=[N_g, 8], dtype=torch.float32, device='cuda')
        for l_sp in range(4):
            s_st = sn[l_sp]
            s_ed = sn[l_sp + 1]
            if l_sp == 0:
                ctx_sn = torch.zeros(size=[s1 - s0, self.cageo_indim - self.freq_enc.output_dim], dtype=torch.float32, device='cuda')
            else:
                ctx_sn = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s_st], norm_xyz_clamp[s_st:s_ed], geo_q[s0:s_st], determ=True)
            ctx_sn = torch.cat([ctx_sn, freq_enc_xyz[s_st:s_ed]], dim=-1)
            mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_geo(ctx_sn), split_size_or_sections=[8, 8, 8], dim=-1)
            mean_sp_l, scale_sp_l, prob_sp_l = mean_sp, scale_sp, prob_sp
            mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[s_st:s_ed], scale_hp[s_st:s_ed], prob_hp[s_st:s_ed]
            probs = torch.stack([prob_sp_l, prob_hp_l], dim=-1)
            probs = torch.softmax(probs, dim=-1)
            prob_sp_l, prob_hp_l = probs[..., 0], probs[..., 1]
            geo_q_tmp = decoder_gaussian_mixed_chunk(mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1)], scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1)], prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1)], Q=Q_geo[s_st:s_ed].contiguous().view(-1), file_name=os.path.join(root_path, f'geo_q_sp{l_sp}.b'), chunk_size=c_size_geo).contiguous().view(s_ed - s_st, 8)
            geo_q[s_st:s_ed] = geo_q_tmp
        g_fea_fused_dec = torch.cat([fe_final, geo_q], dim=-1)
        return g_xyz, g_fea_fused_dec


class Entropy_gaussian_mix_prob_4(nn.Module):

    def __init__(self, Q=1):
        super(Entropy_gaussian_mix_prob_4, self).__init__()
        self.Q = Q

    def forward(self, x, mean1, mean2, mean3, mean4, scale1, scale2, scale3, scale4, probs1, probs2, probs3, probs4, Q=None, x_mean=None, return_lkl=False):
        if Q is None:
            Q = self.Q
        scale1 = torch.clamp(scale1, min=1e-09)
        scale2 = torch.clamp(scale2, min=1e-09)
        scale3 = torch.clamp(scale3, min=1e-09)
        scale4 = torch.clamp(scale4, min=1e-09)
        m1 = torch.distributions.normal.Normal(mean1, scale1)
        m2 = torch.distributions.normal.Normal(mean2, scale2)
        m3 = torch.distributions.normal.Normal(mean3, scale3)
        m4 = torch.distributions.normal.Normal(mean4, scale4)
        likelihood1 = torch.abs(m1.cdf(x + 0.5 * Q) - m1.cdf(x - 0.5 * Q))
        likelihood2 = torch.abs(m2.cdf(x + 0.5 * Q) - m2.cdf(x - 0.5 * Q))
        likelihood3 = torch.abs(m3.cdf(x + 0.5 * Q) - m3.cdf(x - 0.5 * Q))
        likelihood4 = torch.abs(m4.cdf(x + 0.5 * Q) - m4.cdf(x - 0.5 * Q))
        likelihood = Low_bound.apply(probs1 * likelihood1 + probs2 * likelihood2 + probs3 * likelihood3 + probs4 * likelihood4)
        if return_lkl:
            return likelihood
        else:
            likelihood = Low_bound.apply(likelihood)
            bits = -torch.log2(likelihood)
            return bits


class Entropy_bernoulli(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, p):
        p = torch.clamp(p, min=1e-06, max=1 - 1e-06)
        pos_mask = (1 + x) / 2.0
        neg_mask = (1 - x) / 2.0
        pos_prob = p
        neg_prob = 1 - p
        param_bit = -torch.log2(pos_prob) * pos_mask + -torch.log2(neg_prob) * neg_mask
        return param_bit


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


class Camera(nn.Module):

    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device='cuda'):
        super(Camera, self).__init__()
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            None
            None
            self.data_device = torch.device('cuda')
        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        self.scale = scale
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


C0 = 0.28209479177387814


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device='cuda')
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L
    return L


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or lr_init == 0.0 and lr_final == 0.0:
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def mkdir_p(folder_path):
    try:
        makedirs(folder_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device='cuda')
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


class GaussianModel(nn.Module):

    def setup_functions(self):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: 'int'):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.active_sh_degree = self.max_sh_degree

    def capture(self):
        return self.active_sh_degree, self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity, self.max_radii2D, self.xyz_gradient_accum, self.denom, self.optimizer.state_dict(), self.spatial_lr_scale

    def restore(self, model_args, training_args):
        self.active_sh_degree, self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity, self.max_radii2D, xyz_gradient_accum, denom, opt_dict, self.spatial_lr_scale = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1, sc=None, ro=None):
        if sc == None:
            sc = self.get_scaling
            ro = self._rotation
        return self.covariance_activation(sc, scaling_modifier, ro)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: 'BasicPointCloud', spatial_lr_scale: 'float'):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        None
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float()), 1e-07)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device='cuda')
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device='cuda'))
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device='cuda')

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        l = [{'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, 'name': 'xyz'}, {'params': [self._features_dc], 'lr': training_args.feature_lr, 'name': 'f_dc'}, {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, 'name': 'f_rest'}, {'params': [self._opacity], 'lr': training_args.opacity_lr, 'name': 'opacity'}, {'params': [self._scaling], 'lr': training_args.scaling_lr, 'name': 'scaling'}, {'params': [self._rotation], 'lr': training_args.rotation_lr, 'name': 'rotation'}]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale, lr_final=training_args.position_lr_final * self.spatial_lr_scale, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_pkl(self, path):
        torch.save(self.cpu().state_dict(), path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, 'opacity')
        self._opacity = optimizable_tensors['opacity']

    def load_ply(self, path, device='cuda'):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]['x']), np.asarray(plydata.elements[0]['y']), np.asarray(plydata.elements[0]['z'])), axis=1)
        opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['f_dc_0'])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]['f_dc_1'])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['f_dc_2'])
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('f_rest_')]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('scale_')]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('rot')]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def load_pkl(self, path):
        state_dict = torch.load(path)
        N = state_dict['_xyz'].shape[0]
        self._xyz = nn.Parameter(torch.zeros(size=[N, 3]))
        self._features_dc = nn.Parameter(torch.zeros(size=[N, 1, 3]))
        self._features_rest = nn.Parameter(torch.zeros(size=[N, 15, 3]))
        self._opacity = nn.Parameter(torch.zeros(size=[N, 1]))
        self._scaling = nn.Parameter(torch.zeros(size=[N, 3]))
        self._rotation = nn.Parameter(torch.zeros(size=[N, 4]))
        self.load_state_dict(state_dict)
        self.active_sh_degree = self.max_sh_degree
        for n, p in self.named_parameters():
            p.requires_grad = True

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state['exp_avg'] = torch.zeros_like(tensor)
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacity = optimizable_tensors['opacity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group['params']) == 1
            extension_tensor = tensors_dict[group['name']]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=0)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {'xyz': new_xyz, 'f_dc': new_features_dc, 'f_rest': new_features_rest, 'opacity': new_opacities, 'scaling': new_scaling, 'rotation': new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacity = optimizable_tensors['opacity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device='cuda')

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros(n_init_points, device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


class STE_binary(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        p = (input >= 0) * +1.0
        n = (input < 0) * -1.0
        out = p + n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class _grid_encode(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets_list, resolutions_list, calc_grad_inputs=False, max_level=None):
        inputs = inputs.contiguous()
        N, num_dim = inputs.shape
        n_levels = offsets_list.shape[0] - 1
        n_features = embeddings.shape[1]
        max_level = n_levels if max_level is None else min(max_level, n_levels)
        if torch.is_autocast_enabled() and n_features % 2 == 0:
            embeddings = embeddings
        outputs = torch.empty(n_levels, N, n_features, device=inputs.device, dtype=embeddings.dtype)
        if max_level < n_levels:
            outputs.zero_()
        if calc_grad_inputs:
            dy_dx = torch.empty(N, n_levels * num_dim * n_features, device=inputs.device, dtype=embeddings.dtype)
            if max_level < n_levels:
                dy_dx.zero_()
        else:
            dy_dx = None
        _backend.grid_encode_forward(inputs, embeddings, offsets_list, resolutions_list, outputs, N, num_dim, n_features, n_levels, max_level, dy_dx)
        outputs = outputs.permute(1, 0, 2).reshape(N, n_levels * n_features)
        ctx.save_for_backward(inputs, embeddings, offsets_list, resolutions_list, dy_dx)
        ctx.dims = [N, num_dim, n_features, n_levels, max_level]
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        inputs, embeddings, offsets_list, resolutions_list, dy_dx = ctx.saved_tensors
        N, num_dim, n_features, n_levels, max_level = ctx.dims
        grad = grad.view(N, n_levels, n_features).permute(1, 0, 2).contiguous()
        grad_embeddings = torch.zeros_like(embeddings)
        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None
        _backend.grid_encode_backward(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, num_dim, n_features, n_levels, max_level, dy_dx, grad_inputs)
        if dy_dx is not None:
            grad_inputs = grad_inputs
        return grad_inputs, grad_embeddings, None, None, None, None


grid_encode = _grid_encode.apply


class GridEncoder(nn.Module):

    def __init__(self, num_dim=3, n_features=2, resolutions_list=(16, 23, 32, 46, 64, 92, 128, 184, 256, 368, 512, 736), log2_hashmap_size=19, ste_binary=False):
        super().__init__()
        resolutions_list = torch.tensor(resolutions_list)
        n_levels = resolutions_list.numel()
        self.num_dim = num_dim
        self.n_levels = n_levels
        self.n_features = n_features
        self.log2_hashmap_size = log2_hashmap_size
        self.output_dim = n_levels * n_features
        self.ste_binary = ste_binary
        offsets_list = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(n_levels):
            resolution = resolutions_list[i].item()
            params_in_level = min(self.max_params, resolution ** num_dim)
            params_in_level = int(np.ceil(params_in_level / 8) * 8)
            offsets_list.append(offset)
            offset += params_in_level
        offsets_list.append(offset)
        offsets_list = torch.from_numpy(np.array(offsets_list, dtype=np.int32))
        self.register_buffer('offsets_list', offsets_list)
        self.register_buffer('resolutions_list', resolutions_list)
        self.n_params = offsets_list[-1] * n_features
        self.embeddings = nn.Parameter(torch.empty(offset, n_features))
        self.reset_parameters()
        self.n_output_dims = n_levels * n_features

    def reset_parameters(self):
        std = 0.0001
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f'GridEncoder: num_dim={self.num_dim} n_levels={self.n_levels} n_features={self.n_features} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.n_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners} interpolation={self.interpolation}'

    def forward(self, inputs, max_level=None):
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.num_dim)
        if self.ste_binary:
            embeddings = STE_binary.apply(self.embeddings)
        else:
            embeddings = self.embeddings
        outputs = grid_encode(inputs, embeddings, self.offsets_list, self.resolutions_list, inputs.requires_grad, max_level)
        outputs = outputs.view(prefix_shape + [self.output_dim])
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Channel_CTX_feq,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 16, 3])], {}),
     False),
    (Entropy_bernoulli,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Entropy_gaussian,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Entropy_gaussian_mix_prob_2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Entropy_gaussian_mix_prob_3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Entropy_gaussian_mix_prob_4,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_YihangChen_ee_FCGS(_paritybench_base):
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

