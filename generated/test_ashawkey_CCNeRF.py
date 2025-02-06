import sys
_module = sys.modules[__name__]
del sys
compose = _module
gen_config = _module
gen_config_tt = _module
dataLoader = _module
blender = _module
colmap2nerf = _module
llff = _module
nsvf = _module
ray_utils = _module
tankstemple = _module
your_own_data = _module
CCNeRF = _module
models = _module
tensorBase = _module
opt = _module
renderer = _module
shencoder = _module
backend = _module
setup = _module
sphere_harmonics = _module
train = _module
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


from scipy.spatial.transform import Rotation as Rot


import torch


from torch.utils.data import Dataset


from torchvision import transforms as T


import numpy as np


import re


from torch import searchsorted


import torch.nn as nn


import torch.nn.functional as F


from torch.utils.cpp_extension import load


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch.autograd import Function


from torch.autograd.function import once_differentiable


from torch.cuda.amp import custom_bwd


from torch.cuda.amp import custom_fwd


from torch.utils.tensorboard import SummaryWriter


import torchvision.transforms as T


import scipy.signal


class AlphaGridMask(torch.nn.Module):

    def __init__(self, device, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]])

    def sample_alpha(self, xyz_sampled):
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        return alpha_vals


nvcc_flags = ['-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']


class _sh_encoder(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, degree, calc_grad_inputs=False):
        inputs = inputs.contiguous()
        B, input_dim = inputs.shape
        output_dim = degree ** 2
        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)
        if calc_grad_inputs:
            dy_dx = torch.empty(B, input_dim * output_dim, dtype=inputs.dtype, device=inputs.device)
        else:
            dy_dx = torch.empty(1, dtype=inputs.dtype, device=inputs.device)
        _backend.sh_encode_forward(inputs, outputs, B, input_dim, degree, calc_grad_inputs, dy_dx)
        ctx.save_for_backward(inputs, dy_dx)
        ctx.dims = [B, input_dim, degree]
        ctx.calc_grad_inputs = calc_grad_inputs
        return outputs

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        if ctx.calc_grad_inputs:
            grad = grad.contiguous()
            inputs, dy_dx = ctx.saved_tensors
            B, input_dim, degree = ctx.dims
            grad_inputs = torch.zeros_like(inputs)
            _backend.sh_encode_backward(grad, inputs, B, input_dim, degree, dy_dx, grad_inputs)
            return grad_inputs, None, None
        else:
            return None, None, None


sh_encode = _sh_encoder.apply


class SHEncoder(nn.Module):

    def __init__(self, input_dim=3, degree=4):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = degree ** 2
        assert self.input_dim == 3, 'SH encoder only support input dim == 3'
        assert self.degree > 0 and self.degree <= 8, 'SH encoder only supports degree in [1, 8]'

    def __repr__(self):
        return f'SHEncoder: input_dim={self.input_dim} degree={self.degree}'

    def forward(self, inputs, size=1):
        inputs = inputs / size
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, self.input_dim)
        outputs = sh_encode(inputs, self.degree, inputs.requires_grad)
        outputs = outputs.reshape(prefix_shape + [self.output_dim])
        return outputs


def raw2alpha(sigma, dist):
    alpha = 1 - torch.exp(-sigma * dist)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1 - alpha + 1e-10], -1), -1)
    weights = alpha * T[..., :-1]
    return weights


class TensorBase(torch.nn.Module):

    def __init__(self, aabb, gridSize, device, rank_density=[0, 16], rank_mat=[0, 4, 16, 32, 64], rank_vec=[96, 96, 96, 96, 96], degree=3, alphaMask=None, near_far=[2.0, 6.0], density_shift=-10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001, step_ratio=2.0, fea2denseAct='softplus', fea2rgbAct='relu'):
        super(TensorBase, self).__init__()
        self.degree = degree
        self.rank_density = [rank_density]
        self.rank_vec = [rank_vec]
        self.rank_mat = [rank_mat]
        self.K = [len(rank_mat)]
        self.use_vec = [[rank_vec[0] > 0] + [(rank_vec[i] - rank_vec[i - 1] > 0) for i in range(1, self.K[0])]]
        self.use_mat = [[rank_mat[0] > 0] + [(rank_mat[i] - rank_mat[i - 1] > 0) for i in range(1, self.K[0])]]
        self.offset_density_vec = [0]
        self.offset_density_mat = [0]
        self.offset_vec = [0]
        self.offset_mat = [0]
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device = device
        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct
        self.fea2rgbAct = fea2rgbAct
        self.near_far = near_far
        self.step_ratio = step_ratio
        self.update_stepSize(gridSize)
        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]
        self.encoder_dir = SHEncoder(input_dim=3, degree=self.degree)
        self.enc_dir_dim = self.degree ** 2
        self.out_dim = 3 * self.enc_dir_dim
        self.init_svd_volume()

    def update_stepSize(self, gridSize):
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.resolution = gridSize
        self.gridSize = torch.LongTensor(gridSize)
        self.units = self.aabbSize / (self.gridSize - 1)
        self.stepSize = torch.mean(self.units) * self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        None

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass

    def normalize_coord(self, xyz_sampled, oid=0):
        if oid == 0:
            return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1
        else:
            tr = getattr(self, f'T_{oid}')
            y = torch.cat([xyz_sampled, torch.ones_like(xyz_sampled[:, :1])], dim=1)
            y = (y @ tr.T)[:, :3]
            y = (y - getattr(self, f'aabb_{oid}')[0]) * getattr(self, f'invaabbSize_{oid}') - 1
            return y

    def normalize_dir(self, viewdirs, oid=0):
        if oid == 0:
            return viewdirs
        else:
            tr = getattr(self, f'R_{oid}')
            y = viewdirs @ tr.T
            return y

    def get_alphamask(self, oid=0):
        if oid == 0:
            return self.alphaMask
        else:
            return getattr(self, f'alphaMask_{oid}')

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self, K=-1):
        if K <= 0:
            K = self.K[0]
        return {'aabb': self.aabb, 'gridSize': self.gridSize.tolist(), 'rank_density': self.rank_density[0], 'rank_mat': self.rank_mat[0][:K], 'rank_vec': self.rank_vec[0][:K], 'degree': self.degree, 'density_shift': self.density_shift, 'alphaMask_thres': self.alphaMask_thres, 'distance_scale': self.distance_scale, 'rayMarch_weight_thres': self.rayMarch_weight_thres, 'fea2denseAct': self.fea2denseAct, 'fea2rgbAct': self.fea2rgbAct, 'near_far': self.near_far, 'step_ratio': self.step_ratio}

    def save(self, path, K=-1):
        if K <= 0:
            K = self.K[0]
        kwargs = self.get_kwargs(K)
        state_dict = self.state_dict()
        offset_vec = 0
        offset_mat = 0
        for k in range(K):
            if self.use_vec[0][k]:
                offset_vec += 1
            if self.use_mat[0][k]:
                offset_mat += 1
        for key in list(state_dict.keys()):
            if 'sigma' in key:
                continue
            elif 'color_vec' in key:
                k = int(key.split('.')[-1]) // 3
                if k >= offset_vec:
                    del state_dict[key]
            elif 'color_mat' in key:
                k = int(key.split('.')[-1]) // 3
                if k >= offset_mat:
                    del state_dict[key]
            elif 'S_vec' in key:
                k = int(key.split('.')[-1])
                if k >= offset_vec:
                    del state_dict[key]
            elif 'S_mat' in key:
                k = int(key.split('.')[-1])
                if k >= offset_mat:
                    del state_dict[key]
        None
        ckpt = {'kwargs': kwargs, 'state_dict': state_dict}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.shape' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, alpha_volume.float())
            None
        self.load_state_dict(ckpt['state_dict'], strict=False)
        None

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0)
        if is_train:
            interpx += torch.rand_like(interpx) * ((far - near) / N_samples)
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-06), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * rng
        interpx = t_min[..., None] + step
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize
        samples = torch.stack(torch.meshgrid(torch.linspace(0, 1, gridSize[0]), torch.linspace(0, 1, gridSize[1]), torch.linspace(0, 1, gridSize[2])), -1)
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples
        alpha = torch.zeros_like(dense_xyz[..., 0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1, 3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200), return_aabb=True):
        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        alpha = alpha.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]
        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0
        self.alphaMask = AlphaGridMask(self.device, alpha)
        valid_xyz = dense_xyz[alpha > 0.5]
        total = torch.sum(alpha)
        None
        if return_aabb:
            xyz_min = valid_xyz.amin(0)
            xyz_max = valid_xyz.amax(0)
            new_aabb = torch.stack((xyz_min, xyz_max))
            None
            return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240 * 5, bbox_only=False):
        N = torch.tensor(all_rays.shape[:-1]).prod()
        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk]
            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-06), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)
                mask_inbbox = t_max > t_min
            else:
                xyz_sampled, _, _ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                xyz_sampled = self.normalize_coord(xyz_sampled)
                mask_inbbox = (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)
            mask_filtered.append(mask_inbbox.cpu())
        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])
        None
        return all_rays[mask_filtered], all_rgbs[mask_filtered]

    def feature2density(self, density_features):
        if self.fea2denseAct == 'softplus':
            return F.softplus(density_features + self.density_shift)
        elif self.fea2denseAct == 'relu':
            return F.relu(density_features)

    def compute_alpha(self, xyz_locs, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(self.normalize_coord(xyz_locs))
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:, 0], dtype=bool)
        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            xyz_valid = xyz_locs[alpha_mask]
            for oid in range(len(self.K)):
                xyz_sampled = self.normalize_coord(xyz_valid, oid=oid)
                sigma_feature = self.compute_density_features(xyz_sampled, oid=oid)
                validsigma = self.feature2density(sigma_feature)
                sigma[alpha_mask] = sigma[alpha_mask] + validsigma
        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])
        return alpha

    def feature2rgb(self, h):
        if self.fea2rgbAct == 'sigmoid':
            h = torch.sigmoid(h)
        else:
            h = F.relu(h + 0.5).clamp(0, 1)
        return h

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1, K=-1):
        if K <= 0:
            K = self.K[0]
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        sigma = torch.zeros(*xyz_sampled.shape[:-1], device=xyz_sampled.device)
        if is_train:
            rgb = torch.zeros(K, *xyz_sampled.shape[:-1], 3, device=xyz_sampled.device)
        else:
            rgb = torch.zeros(*xyz_sampled.shape[:-1], 3, device=xyz_sampled.device)
        if is_train:
            xyz_model = self.normalize_coord(xyz_sampled[ray_valid], oid=0)
            if self.alphaMask is not None:
                alphas = self.alphaMask.sample_alpha(xyz_model)
                alpha_mask = alphas > 0
                ray_invalid = ~ray_valid
                ray_invalid[ray_valid] |= ~alpha_mask
                ray_valid = ~ray_invalid
                xyz_model = xyz_model[alpha_mask]
            if ray_valid.any():
                sigma_feature = self.compute_density_features(xyz_model)
                validsigma = self.feature2density(sigma_feature)
                sigma[ray_valid] = validsigma
        else:
            ws = []
            last_w = 0
            for oid in range(len(self.K)):
                xyz_model = self.normalize_coord(xyz_sampled[ray_valid], oid=oid)
                if self.alphaMask is not None:
                    alphas = self.get_alphamask(oid).sample_alpha(xyz_model)
                    alpha_mask = alphas > 0
                    ray_invalid_model = ~ray_valid
                    ray_invalid_model[ray_valid] |= ~alpha_mask
                    ray_valid_model = ~ray_invalid_model
                else:
                    ray_valid_model = ray_valid
                    alpha_mask = torch.ones_like(xyz_model[:, 0]).long()
                if ray_valid_model.any():
                    xyz_model = xyz_model[alpha_mask]
                    sigma_feature = self.compute_density_features(xyz_model, oid=oid)
                    validsigma = self.feature2density(sigma_feature)
                    sigma[ray_valid_model] = sigma[ray_valid_model] + validsigma
                w = sigma.detach().clone()
                ws.append(w - last_w)
                last_w = w
            ws = torch.stack(ws, dim=0)
            ws = F.softmax(ws, dim=0)
        weight = raw2alpha(sigma, dists * self.distance_scale)
        app_mask = weight > self.rayMarch_weight_thres
        if app_mask.any():
            xyz_valid = xyz_sampled[app_mask]
            dir_valid = viewdirs[app_mask]
            if is_train:
                xyz_model = self.normalize_coord(xyz_valid, oid=0)
                color_features = self.compute_features(xyz_model, K=K, is_train=is_train, oid=0)
                dir_model = self.normalize_dir(dir_valid, oid=0)
                N = dir_model.shape[0]
                dir_feature = self.encoder_dir(dir_model).view(N, 1, -1)
                color_features = color_features.view(K, N, 3, -1)
                app_features = (color_features * dir_feature.unsqueeze(0)).sum(-1)
                rgb[app_mask.unsqueeze(0).repeat(K, 1, 1)] = self.feature2rgb(app_features).view(-1, 3)
            else:
                app_features = 0
                for oid in range(len(self.K)):
                    xyz_model = self.normalize_coord(xyz_valid, oid=oid)
                    color_feature = self.compute_features(xyz_model, K=K if oid == 0 else -1, is_train=is_train, oid=oid)
                    dir_model = self.normalize_dir(dir_valid, oid=oid)
                    N = dir_model.shape[0]
                    dir_feature = self.encoder_dir(dir_model).view(N, 1, -1)
                    color_feature = color_feature.view(N, 3, -1)
                    app_feature = (color_feature * dir_feature).sum(-1)
                    app_features = app_features + app_feature * ws[oid][app_mask].unsqueeze(-1)
                rgb[app_mask] = self.feature2rgb(app_features)
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)
        if white_bg or is_train and torch.rand((1,)) < 0.5:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)
        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]
        return rgb_map, depth_map

    @torch.no_grad()
    def compose(self, other, T=None, R=None):
        if T is None:
            T = torch.eye(4, dtype=torch.float32)
        elif isinstance(T, np.ndarray):
            T = torch.from_numpy(T.astype(np.float32))
        else:
            T = T.float()
        if R is None:
            R = torch.eye(3, dtype=torch.float32)
        elif isinstance(R, np.ndarray):
            R = torch.from_numpy(R.astype(np.float32))
        else:
            R = R.float()
        T = torch.inverse(T)
        T = T
        R = R.T
        self.offset_density_vec.append(len(self.sigma_vec))
        self.offset_density_mat.append(len(self.sigma_mat))
        self.offset_vec.append(len(self.S_vec))
        self.offset_mat.append(len(self.S_mat))
        self.rank_density.extend(other.rank_density)
        self.rank_mat.extend(other.rank_mat)
        self.rank_vec.extend(other.rank_vec)
        self.rank_vec.extend(other.rank_vec)
        self.rank_mat.extend(other.rank_mat)
        self.use_vec.extend(other.use_vec)
        self.use_mat.extend(other.use_mat)
        self.K.extend(other.K)
        self.sigma_vec.extend(other.sigma_vec)
        self.color_vec.extend(other.color_vec)
        self.S_vec.extend(other.S_vec)
        self.sigma_mat.extend(other.sigma_mat)
        self.color_mat.extend(other.color_mat)
        self.S_mat.extend(other.S_mat)
        oid = len(self.K) - 1
        self.register_buffer(f'T_{oid}', T)
        self.register_buffer(f'R_{oid}', R)
        self.register_buffer(f'aabb_{oid}', other.aabb)
        self.register_buffer(f'invaabbSize_{oid}', other.invaabbSize)
        setattr(self, f'alphaMask_{oid}', AlphaGridMask(self.device, other.alphaMask.alpha_volume))
        return self

    @torch.no_grad()
    def plot_rank(self):
        import matplotlib.pyplot as plt
        if self.use_vec[0]:
            color_vec = [torch.cat([v.data for v in self.color_vec[0::3]], dim=1), torch.cat([v.data for v in self.color_vec[1::3]], dim=1), torch.cat([v.data for v in self.color_vec[2::3]], dim=1)]
            S_vec = torch.cat([s.data for s in self.S_vec], dim=1)
            R = S_vec.shape[1]
            importance = S_vec.detach().clone().abs()
            for i in range(3):
                importance *= color_vec[i].detach().view(R, -1).norm(dim=-1).unsqueeze(0)
            sum = importance.sum(0)
            for k in range(self.K[0]):
                start = k * self.rank_vec[0][0]
                end = (k + 1) * self.rank_vec[0][0]
                inds = torch.argsort(sum[start:end], descending=True)
                importance[:, start:end] = importance[:, start:end][:, inds]
            plt.matshow(importance.cpu().numpy())
            plt.show()
        if self.use_mat[0]:
            color_mat = [torch.cat([v.data for v in self.color_mat[0::3]], dim=1), torch.cat([v.data for v in self.color_mat[1::3]], dim=1), torch.cat([v.data for v in self.color_mat[2::3]], dim=1)]
            S_mat = torch.cat([s.data for s in self.S_mat], dim=1)
            R = S_mat.shape[1]
            importance = S_mat.detach().clone().abs()
            for i in range(3):
                importance *= color_mat[i].detach().view(R, -1).norm(dim=-1).unsqueeze(0)
            sum = importance.sum(0)
            for k in range(self.K[0]):
                start = k * self.rank_vec[0][1]
                end = (k + 1) * self.rank_vec[0][1]
                inds = torch.argsort(sum[start:end], descending=True)
                importance[:, start:end] = importance[:, start:end][:, inds]
            plt.matshow(importance.cpu().numpy())
            plt.show()

    @torch.no_grad()
    def compress(self, target_rank):
        current_rank = self.rank_mat[0]
        rank_vec = self.rank_vec[0]
        assert target_rank[0] <= current_rank[0] and target_rank[1] <= current_rank[1]
        if self.use_vec[0]:
            k, r = target_rank[0] // rank_vec[0], target_rank[0] % rank_vec[0]
            None
            if r == 0:
                self.color_vec = self.color_vec[:3 * k]
                self.S_vec = self.S_vec[:k]
                self.K[0] = k
            else:
                self.color_vec = self.color_vec[:3 * (k + 1)]
                self.S_vec = self.S_vec[:k + 1]
                self.K[0] = k + 1
                importance = self.S_vec[-1].detach().clone().abs().sum(0)
                for i in range(3):
                    importance *= self.color_vec[3 * k + i].detach().view(rank_vec[0], -1).norm(dim=-1)
                vals, inds = torch.topk(importance, r, largest=True)
                self.S_vec[-1] = nn.Parameter(self.S_vec[-1].data[:, inds])
                for i in range(3):
                    self.color_vec[3 * k + i] = nn.Parameter(self.color_vec[3 * k + i].data[:, inds])
        if self.use_mat[0]:
            k, r = target_rank[1] // rank_vec[1], target_rank[1] % rank_vec[1]
            None
            if r == 0:
                self.color_mat = self.color_mat[:3 * k]
                self.S_mat = self.S_mat[:k]
                self.K[0] = k
            else:
                self.color_mat = self.color_mat[:3 * (k + 1)]
                self.S_mat = self.S_mat[:k + 1]
                self.K[0] = k + 1
                importance = self.S_mat[-1].detach().clone().abs().sum(0)
                for i in range(3):
                    importance *= self.color_mat[3 * k + i].detach().view(rank_vec[1], -1).norm(dim=-1)
                vals, inds = torch.topk(importance, r, largest=True)
                self.S_mat[-1] = nn.Parameter(self.S_mat[-1].data[:, inds])
                for i in range(3):
                    self.color_mat[3 * k + i] = nn.Parameter(self.color_mat[3 * k + i].data[:, inds])
        return self


class TVLoss(nn.Module):

    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (TVLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_ashawkey_CCNeRF(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

