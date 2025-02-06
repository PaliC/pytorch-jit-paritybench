import sys
_module = sys.modules[__name__]
del sys
arf_opt = _module
arguments = _module
ref_get_example = _module
ref_loss = _module
ref_pre = _module
ref_render = _module
ref_render_circle = _module
ref_opt = _module
ref_regist = _module
snerf_opt = _module
autotune = _module
calc_metrics = _module
opt = _module
render_imgs = _module
render_imgs_circle = _module
colmap2nsvf = _module
create_split = _module
proc_record3d = _module
run_colmap = _module
unsplit = _module
read_write_model = _module
view_data = _module
to_svox1 = _module
util = _module
co3d_dataset = _module
config_util = _module
dataset = _module
dataset_base = _module
depth_ops = _module
gainmap_stylization = _module
llff_dataset = _module
load_llff = _module
load_llff_nerf = _module
nerf_dataset = _module
nsvf_dataset = _module
run_nerf_helpers = _module
sem_feat_sim = _module
util = _module
setup = _module
__init__ = _module
defs = _module
svox2 = _module
utils = _module
version = _module

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


import torch


import torch.optim


import torch.nn.functional as F


import numpy as np


import math


from warnings import warn


from torch.utils.tensorboard import SummaryWriter


import torchvision


import copy


from math import floor


import random


from typing import List


from typing import Dict


import itertools


import torch.cuda


from typing import NamedTuple


from typing import Optional


from typing import Union


import numpy


from scipy.spatial.transform import Rotation


from collections import deque


import torch.nn as nn


from scipy.interpolate import CubicSpline


from matplotlib import pyplot as plt


import warnings


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


from torch import nn


from torch import autograd


from typing import Tuple


from functools import reduce


from functools import partial


def ModifyMap(Style, Input, gmin, gmax):
    Gain = torch.div(Style, Input + 0.0001)
    Gain = torch.clamp(Gain, min=gmin, max=gmax)
    return Input * Gain


def cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-08)
    b_tmp = b / (b_norm + 1e-08)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()


def argmin_cos_distance_thre(a, b, thre=0.6, center=False):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)
    b_norm = ((b * b).sum(1, keepdims=True) + 1e-08).sqrt()
    b = b / (b_norm + 1e-08)
    z_best = []
    loop_batch_size = int(100000000.0 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i:i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-08).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-08)
        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)
        z_best_min, z_best_batch = torch.min(d_mat, 2)
        z_best_batch[z_best_min > thre] = -1
        z_best.append(z_best_batch)
    z_best = torch.cat(z_best, dim=-1)
    return z_best


def get_nn_feat_relation_thre(tmpl, a):
    n, c, h, w = a.size()
    a_flat = a.view(n, c, -1)
    tmpl_flat = tmpl.view(n, c, -1)
    z_new = []
    z_bests = []
    for i in range(n):
        z_best = argmin_cos_distance_thre(a_flat[i:i + 1], tmpl_flat[i:i + 1])
        z_bests.append(z_best)
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
    return torch.cat(z_bests, 0)


def gram_matrix(feature_maps, center=False):
    """
    feature_maps: b, c, h, w
    gram_matrix: b, c, c
    """
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    if center:
        features = features - features.mean(dim=-1, keepdims=True)
    G = torch.bmm(features, torch.transpose(features, 1, 2))
    return G


def argmin_cos_distance(a, b, center=False):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)
    b_norm = ((b * b).sum(1, keepdims=True) + 1e-08).sqrt()
    b = b / (b_norm + 1e-08)
    z_best = []
    loop_batch_size = int(100000000.0 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i:i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-08).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-08)
        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)
        z_best_batch = torch.argmin(d_mat, 2)
        z_best.append(z_best_batch)
    z_best = torch.cat(z_best, dim=-1)
    return z_best


def nn_feat_replace(a, b):
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()
    assert n == 1 and n2 == 1
    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()
    z_new = []
    for i in range(n):
        z_best = argmin_cos_distance(a_flat[i:i + 1], b_flat[i:i + 1])
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)
        z_new.append(feat)
    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new


def nn_feat_replace_cond(tmpl, a, b):
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()
    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    tmpl_flat = tmpl.view(n, c, -1)
    b_ref = b_flat.clone()
    z_new = []
    z_bests = []
    for i in range(n):
        z_best = argmin_cos_distance(a_flat[i:i + 1], tmpl_flat[i:i + 1])
        z_bests.append(z_best)
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)
        z_new.append(feat)
    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new, torch.cat(z_bests, 0)


class NNFMLoss(torch.nn.Module):

    def __init__(self, device, gainmap=False, is_bn=False, layer=16):
        super().__init__()
        if layer == 19:
            self.vgg = torchvision.models.vgg19(pretrained=True).eval()
            self.block_indexes = [[1, 3], [6, 8], [11, 13, 15, 17], [20, 22, 24, 26], [29, 31, 33, 35]]
        elif not is_bn:
            self.vgg = torchvision.models.vgg16(pretrained=True).eval()
            self.block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]
        else:
            self.vgg = torchvision.models.vgg16_bn(pretrained=True).eval()
            self.block_indexes = [[2, 5], [9, 12], [16, 19, 22], [26, 29, 32], [36, 39, 42]]
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.gainmap = gainmap
        self.gmin = 0.7
        self.gmax = 5.0

    def get_feats(self, x, layers=[]):
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []
        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)
            if ix == final_ix:
                break
        return outputs

    def get_feats_by_blk(self, x, blocks=[]):
        all_layers = []
        for block in blocks:
            all_layers += self.block_indexes[block]
        x = self.normalize(x)
        final_ix = max(all_layers)
        outputs = []
        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in all_layers:
                outputs.append(x)
            if ix == final_ix:
                break
        return outputs

    def preload_golden_template(self, tmpl_imgs=None, tmpl_stys=None, blocks=[2, 4]):
        if not hasattr(self, 'styl_feats_all'):
            blocks.sort()
            all_layers = []
            self.styl_feats_all = []
            self.tmpl_feats_all = []
            self.tmpl_stys = tmpl_stys
            self.tmpl_imgs = tmpl_imgs
            for block in blocks:
                all_layers += self.block_indexes[block]
            with torch.no_grad():
                for tmpl_sty, tmpl_img in zip(tmpl_stys, tmpl_imgs):
                    self.tmpl_feats_all += [self.get_feats(tmpl_img, all_layers)]
                    self.styl_feats_all += [self.get_feats(tmpl_sty.permute(2, 0, 1).unsqueeze(0), all_layers)]
                for layer in range(len(self.styl_feats_all[0])):
                    self.styl_feats_all[0][layer] = torch.concat([self.styl_feats_all[i][layer] for i in range(len(tmpl_stys))], dim=2)
                    self.tmpl_feats_all[0][layer] = torch.concat([self.tmpl_feats_all[i][layer] for i in range(len(tmpl_stys))], dim=2)
                self.styl_feats_all = self.styl_feats_all[0]
                self.tmpl_feats_all = self.tmpl_feats_all[0]
        else:
            blocks.sort()
            all_layers = []
            self.tmpl_feats_all = []
            self.tmpl_imgs = tmpl_imgs
            for block in blocks:
                all_layers += self.block_indexes[block]
            with torch.no_grad():
                for tmpl_img in tmpl_imgs:
                    self.tmpl_feats_all += [self.get_feats(tmpl_img, all_layers)]
                for layer in range(len(self.tmpl_feats_all[0])):
                    self.tmpl_feats_all[0][layer] = torch.concat([self.tmpl_feats_all[i][layer] for i in range(len(tmpl_stys))], dim=2)
                self.tmpl_feats_all = self.tmpl_feats_all[0]

    def forward(self, outputs, styles, blocks=[2], loss_names=['nnfm_loss'], contents=None):
        for x in loss_names:
            assert x in ['nnfm_loss', 'content_loss', 'gram_loss', 'tcm_loss', 'color_patch', 'online_tmp_loss']
        blocks.sort()
        all_layers = []
        for block in blocks:
            all_layers += self.block_indexes[block]
        x_feats_all = self.get_feats(outputs, all_layers)
        with torch.no_grad():
            if hasattr(self, 'styl_feats_all'):
                s_feats_all = self.styl_feats_all
            else:
                s_feats_all = self.get_feats(styles, all_layers)
            if 'content_loss' in loss_names or 'tcm_loss' in loss_names:
                content_feats_all = self.get_feats(contents, all_layers)
        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a
        if 'tcm_loss' in loss_names:
            coarse_style_flat = []
            down_fact = 16
            for tmpl_sty in self.tmpl_stys:
                h_sty, w_sty = tmpl_sty.shape[:2]
                coarse_style_flat.append(F.interpolate(tmpl_sty.unsqueeze(0).permute(0, 3, 1, 2), (h_sty // down_fact, w_sty // down_fact), mode='bilinear', antialias=True, align_corners=True))
            coarse_style_flat = torch.cat(coarse_style_flat, dim=-2)
            coarse_style_flat = coarse_style_flat.view(1, 3, -1)
            coarse_style_flat = torch.cat((coarse_style_flat, torch.FloatTensor([[[0], [0], [0]]])), dim=-1)
            coarse_out_flat = F.interpolate(outputs, (h_sty // down_fact, w_sty // down_fact), mode='bilinear', antialias=True, align_corners=True).view(1, 3, -1)
        loss_dict = dict([(x, 0.0) for x in loss_names])
        if len(blocks) == 1:
            blocks += [-1]
        for block in blocks[:-1]:
            layers = self.block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)
            if 'nnfm_loss' in loss_names:
                target_feats = nn_feat_replace(x_feats, s_feats)
                loss_dict['nnfm_loss'] += cos_loss(x_feats, target_feats)
            if 'gram_loss' in loss_names:
                loss_dict['gram_loss'] += torch.mean((gram_matrix(x_feats) - gram_matrix(s_feats)) ** 2)
            if 'content_loss' in loss_names or 'tcm_loss' in loss_names:
                content_feats = torch.cat([content_feats_all[ix_map[ix]] for ix in layers], 1)
            if 'content_loss' in loss_names:
                if self.gainmap:
                    content_feats = ModifyMap(s_feats, content_feats, self.gmin, self.gmax)
                loss_dict['content_loss'] += torch.mean((content_feats - x_feats) ** 2)
            if 'tcm_loss' in loss_names:
                tmpl_feats = torch.cat([self.tmpl_feats_all[ix_map[ix]] for ix in layers], 1)
                target_feats, relation = nn_feat_replace_cond(tmpl_feats, content_feats, s_feats)
                loss_dict['tcm_loss'] += cos_loss(x_feats, target_feats)
            if 'online_tmp_loss' in loss_names:
                tmpl_feats = torch.cat([self.tmpl_feats_all[ix_map[ix]] for ix in layers], 1)
                target_feats, relation = nn_feat_replace_cond(tmpl_feats, x_feats, s_feats)
                loss_dict['online_tmp_loss'] += cos_loss(x_feats, target_feats) * 0.2
            if 'color_patch' in loss_names:
                layers_last = self.block_indexes[blocks[-1]]
                tmpl_feats_last = torch.cat([self.tmpl_feats_all[ix_map[ix]] for ix in layers_last], 1)
                if 'online_tmp_loss' not in loss_names:
                    content_feats_last = torch.cat([content_feats_all[ix_map[ix]] for ix in layers_last], 1)
                    relation_last = get_nn_feat_relation_thre(tmpl_feats_last, content_feats_last).repeat(1, 3, 1)
                else:
                    content_feats_last = torch.cat([x_feats_all[ix_map[ix]] for ix in layers_last], 1)
                    relation_last = get_nn_feat_relation_thre(tmpl_feats_last, content_feats_last).repeat(1, 3, 1)
                relation_last[relation_last < 0] = coarse_style_flat.shape[-1] - 1
                related_img = torch.gather(coarse_style_flat, 2, relation_last)
                loss_patch = (related_img - coarse_out_flat) ** 2
                loss_patch[relation_last == coarse_style_flat.shape[-1] - 1] = 0
                loss_patch = loss_patch.mean(dim=1)
                coarse_out_flat_mask = coarse_out_flat.mean(dim=1)
                loss_patch[coarse_out_flat_mask > 0.99] = 0
                loss_dict['color_patch'] = torch.mean(loss_patch) * 5
        return loss_dict


class NeRF(nn.Module):

    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] + [(nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)) for i in range(D - 1)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, 'Not implemented if use_viewdirs=False'
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


def viridis_no_norm_cmap(gray: 'np.ndarray'):
    """
    Visualize a single-channel image using matplotlib's viridis color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.turbo(gray.squeeze())[..., :-1]
    return colored.astype(np.float32)


class VGGPatchSim(torch.nn.Module):

    def __init__(self, device, args=None):
        super().__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True).eval()
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.args = args
        self.downrate = args.downrate

    def get_feats(self, x, layers=None):
        if layers is None:
            return []
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []
        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)
            if ix == final_ix:
                break
        return outputs

    def forward(self, img, style, layers=None):
        if layers is None:
            layers = [6, 11, 18, 25]
        h, w = img.shape[-2:]
        img_feats = self.get_feats(img, layers)
        style_feats = self.get_feats(style, layers)
        img_sim_mats = []
        style_sim_mats = []
        for i, (img_feat, style_feat) in enumerate(zip(img_feats, style_feats)):
            img_feat = F.interpolate(img_feat, (h // self.downrate, w // self.downrate), mode='bilinear')
            style_feat = F.interpolate(style_feat, (h // self.downrate, w // self.downrate), mode='bilinear')
            c = img_feat.shape[1]
            img_feat = F.normalize(img_feat[0].reshape(c, -1), dim=0)
            style_feat = F.normalize(style_feat[0].reshape(c, -1), dim=0)
            ic(f'level {i}', img_feat.shape, style_feat.shape)
            img_sim_mat = torch.mm(img_feat.T, img_feat)
            img_sim_mats += [img_sim_mat]
            style_sim_mat = torch.mm(style_feat.T, style_feat)
            style_sim_mats += [style_sim_mat]
        return img_sim_mats, style_sim_mats

    def visualize_patch(self, img, sim_mat, h_in, w_in, path='sim_vis.png'):
        h, w = img.shape[:2]
        feat_h, feat_w = h // self.downrate, w // self.downrate
        idx_h = floor(h_in * h / self.downrate)
        idx_w = floor(w_in * w / self.downrate)
        ic(idx_h, idx_w, sim_mat.shape)
        sim_mat = sim_mat[idx_h * feat_w + idx_w].view(feat_h, feat_w).unsqueeze(0).unsqueeze(0)
        sim_mat = F.interpolate(sim_mat, (h, w), mode='bilinear')[0][0].cpu().detach().numpy()
        sim_mat = viridis_no_norm_cmap(sim_mat)
        imageio.imwrite(path, (sim_mat * 128 + img * 128).astype(np.uint8))
        ic(sim_mat.shape)


BASIS_TYPE_3D_TEXTURE = 4


BASIS_TYPE_MLP = 255


BASIS_TYPE_SH = 1


class _SampleGridAutogradFunction(autograd.Function):

    @staticmethod
    def forward(ctx, data_density: 'torch.Tensor', data_sh: 'torch.Tensor', grid, points: 'torch.Tensor', want_colors: 'bool'):
        assert not points.requires_grad, 'Point gradient not supported'
        out_density, out_sh = _C.sample_grid(grid, points, want_colors)
        ctx.save_for_backward(points)
        ctx.grid = grid
        ctx.want_colors = want_colors
        return out_density, out_sh

    @staticmethod
    def backward(ctx, grad_out_density, grad_out_sh):
        points, = ctx.saved_tensors
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        _C.sample_grid_backward(ctx.grid, points, grad_out_density.contiguous(), grad_out_sh.contiguous(), grad_density_grid, grad_sh_grid, ctx.want_colors)
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None
        return grad_density_grid, grad_sh_grid, None, None, None


class _TotalVariationFunction(autograd.Function):

    @staticmethod
    def forward(ctx, data: 'torch.Tensor', links: 'torch.Tensor', start_dim: 'int', end_dim: 'int', use_logalpha: 'bool', logalpha_delta: 'float', ignore_edge: 'bool', ndc_coeffs: 'Tuple[float, float]'):
        tv = _C.tv(links, data, start_dim, end_dim, use_logalpha, logalpha_delta, ignore_edge, ndc_coeffs[0], ndc_coeffs[1])
        ctx.save_for_backward(links, data)
        ctx.start_dim = start_dim
        ctx.end_dim = end_dim
        ctx.use_logalpha = use_logalpha
        ctx.logalpha_delta = logalpha_delta
        ctx.ignore_edge = ignore_edge
        ctx.ndc_coeffs = ndc_coeffs
        return tv

    @staticmethod
    def backward(ctx, grad_out):
        links, data = ctx.saved_tensors
        grad_grid = torch.zeros_like(data)
        _C.tv_grad(links, data, ctx.start_dim, ctx.end_dim, 1.0, ctx.use_logalpha, ctx.logalpha_delta, ctx.ignore_edge, ctx.ndc_coeffs[0], ctx.ndc_coeffs[1], grad_grid)
        ctx.start_dim = ctx.end_dim = None
        if not ctx.needs_input_grad[0]:
            grad_grid = None
        return grad_grid, None, None, None, None, None, None, None


class _VolumeRenderFunction(autograd.Function):

    @staticmethod
    def forward(ctx, data_density: 'torch.Tensor', data_sh: 'torch.Tensor', data_basis: 'torch.Tensor', data_background: 'torch.Tensor', grid, rays, opt, backend: 'str'):
        cu_fn = _C.__dict__[f'volume_render_{backend}']
        color = cu_fn(grid, rays, opt)
        ctx.save_for_backward(color)
        ctx.grid = grid
        ctx.rays = rays
        ctx.opt = opt
        ctx.backend = backend
        ctx.basis_data = data_basis
        return color

    @staticmethod
    def backward(ctx, grad_out):
        color_cache, = ctx.saved_tensors
        cu_fn = _C.__dict__[f'volume_render_{ctx.backend}_backward']
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        if ctx.grid.basis_type == BASIS_TYPE_MLP:
            grad_basis = torch.zeros_like(ctx.basis_data)
        elif ctx.grid.basis_type == BASIS_TYPE_3D_TEXTURE:
            grad_basis = torch.zeros_like(ctx.grid.basis_data.data)
        if ctx.grid.background_data is not None:
            grad_background = torch.zeros_like(ctx.grid.background_data.data)
        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density_grid
        grad_holder.grad_sh_out = grad_sh_grid
        if ctx.needs_input_grad[2]:
            grad_holder.grad_basis_out = grad_basis
        if ctx.grid.background_data is not None and ctx.needs_input_grad[3]:
            grad_holder.grad_background_out = grad_background
        cu_fn(ctx.grid, ctx.rays, ctx.opt, grad_out.contiguous(), color_cache, grad_holder)
        ctx.grid = ctx.rays = ctx.opt = None
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None
        if not ctx.needs_input_grad[2]:
            grad_basis = None
        if not ctx.needs_input_grad[3]:
            grad_background = None
        ctx.basis_data = None
        return grad_density_grid, grad_sh_grid, grad_basis, grad_background, None, None, None, None


class SparseGrid(nn.Module):
    """
    Main sparse grid data structure.
    initially it will be a dense grid of resolution <reso>.
    Only float32 is supported.

    :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
    :param radius: float or List[float, float, float], the 1/2 side length of the grid, optionally in each direction
    :param center: float or List[float, float, float], the center of the grid
    :param basis_type: int, basis type; may use svox2.BASIS_TYPE_* (1 = SH, 4 = learned 3D texture, 255 = learned MLP)
    :param basis_dim: int, size of basis / number of SH components
                           (must be square number in case of SH)
    :param basis_reso: int, resolution of grid if using BASIS_TYPE_3D_TEXTURE
    :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
    :param mlp_posenc_size: int, if using BASIS_TYPE_MLP, then enables standard axis-aligned positional encoding of
                                 given size on MLP; if 0 then does not use positional encoding
    :param mlp_width: int, if using BASIS_TYPE_MLP, specifies MLP width (hidden dimension)
    :param device: torch.device, device to store the grid
    """

    def __init__(self, reso: 'Union[int, List[int], Tuple[int, int, int]]'=128, radius: 'Union[float, List[float]]'=1.0, center: 'Union[float, List[float]]'=[0.0, 0.0, 0.0], basis_type: 'int'=BASIS_TYPE_SH, basis_dim: 'int'=9, basis_reso: 'int'=16, use_z_order: 'bool'=False, use_sphere_bound: 'bool'=False, mlp_posenc_size: 'int'=0, mlp_width: 'int'=16, background_nlayers: 'int'=0, background_reso: 'int'=256, device: 'Union[torch.device, str]'='cpu'):
        super().__init__()
        self.basis_type = basis_type
        if basis_type == BASIS_TYPE_SH:
            assert utils.isqrt(basis_dim) is not None, 'basis_dim (SH) must be a square number'
        assert basis_dim >= 1 and basis_dim <= utils.MAX_SH_BASIS, f'basis_dim 1-{utils.MAX_SH_BASIS} supported'
        self.basis_dim = basis_dim
        self.mlp_posenc_size = mlp_posenc_size
        self.mlp_width = mlp_width
        self.background_nlayers = background_nlayers
        assert background_nlayers == 0 or background_nlayers > 1, 'Please use at least 2 MSI layers (trilerp limitation)'
        self.background_reso = background_reso
        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert len(reso) == 3, 'reso must be an integer or indexable object of 3 ints'
        if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
            None
            use_z_order = False
        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        if isinstance(radius, torch.Tensor):
            radius = radius
        else:
            radius = torch.tensor(radius, dtype=torch.float32, device='cpu')
        if isinstance(center, torch.Tensor):
            center = center
        else:
            center = torch.tensor(center, dtype=torch.float32, device='cpu')
        self.radius: 'torch.Tensor' = radius
        self.center: 'torch.Tensor' = center
        self._offset = 0.5 * (1.0 - self.center / self.radius)
        self._scaling = 0.5 / self.radius
        n3: 'int' = reduce(lambda x, y: x * y, reso)
        if use_z_order:
            init_links = utils.gen_morton(reso[0], device=device, dtype=torch.int32).flatten()
        else:
            init_links = torch.arange(n3, device=device, dtype=torch.int32)
        if use_sphere_bound:
            X = torch.arange(reso[0], dtype=torch.float32, device=device) - 0.5
            Y = torch.arange(reso[1], dtype=torch.float32, device=device) - 0.5
            Z = torch.arange(reso[2], dtype=torch.float32, device=device) - 0.5
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
            gsz = torch.tensor(reso)
            roffset = 1.0 / gsz - 1.0
            rscaling = 2.0 / gsz
            points = torch.addcmul(roffset, points, rscaling)
            norms = points.norm(dim=-1)
            mask = norms <= 1.0 + 3 ** 0.5 / gsz.max()
            self.capacity: 'int' = mask.sum()
            data_mask = torch.zeros(n3, dtype=torch.int32, device=device)
            idxs = init_links[mask].long()
            data_mask[idxs] = 1
            data_mask = torch.cumsum(data_mask, dim=0) - 1
            init_links[mask] = data_mask[idxs].int()
            init_links[~mask] = -1
        else:
            self.capacity = n3
        self.density_data = nn.Parameter(torch.zeros(self.capacity, 1, dtype=torch.float32, device=device))
        self.sh_data = nn.Parameter(torch.zeros(self.capacity, self.basis_dim * 3, dtype=torch.float32, device=device))
        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            self.basis_data = nn.Parameter(torch.zeros(basis_reso, basis_reso, basis_reso, self.basis_dim, dtype=torch.float32, device=device))
        elif self.basis_type == BASIS_TYPE_MLP:
            D_rgb = mlp_width
            dir_in_dims = 3 + 6 * self.mlp_posenc_size
            self.basis_mlp = nn.Sequential(nn.Linear(dir_in_dims, D_rgb), nn.ReLU(), nn.Linear(D_rgb, D_rgb), nn.ReLU(), nn.Linear(D_rgb, D_rgb), nn.ReLU(), nn.Linear(D_rgb, self.basis_dim))
            self.basis_mlp = self.basis_mlp
            self.basis_mlp.apply(utils.init_weights)
            self.basis_data = nn.Parameter(torch.empty(0, 0, 0, 0, dtype=torch.float32, device=device), requires_grad=False)
        else:
            self.basis_data = nn.Parameter(torch.empty(0, 0, 0, 0, dtype=torch.float32, device=device), requires_grad=False)
        self.background_links: 'Optional[torch.Tensor]'
        self.background_data: 'Optional[torch.Tensor]'
        if self.use_background:
            background_capacity = self.background_reso ** 2 * 2
            background_links = torch.arange(background_capacity, dtype=torch.int32, device=device).reshape(self.background_reso * 2, self.background_reso)
            self.register_buffer('background_links', background_links)
            self.background_data = nn.Parameter(torch.zeros(background_capacity, self.background_nlayers, 4, dtype=torch.float32, device=device))
        else:
            self.background_data = nn.Parameter(torch.empty(0, 0, 0, dtype=torch.float32, device=device), requires_grad=False)
        self.register_buffer('links', init_links.view(reso))
        self.links: 'torch.Tensor'
        self.opt = RenderOptions()
        self.sparse_grad_indexer: 'Optional[torch.Tensor]' = None
        self.sparse_sh_grad_indexer: 'Optional[torch.Tensor]' = None
        self.sparse_background_indexer: 'Optional[torch.Tensor]' = None
        self.density_rms: 'Optional[torch.Tensor]' = None
        self.sh_rms: 'Optional[torch.Tensor]' = None
        self.background_rms: 'Optional[torch.Tensor]' = None
        self.basis_rms: 'Optional[torch.Tensor]' = None
        if self.links.is_cuda and use_sphere_bound:
            self.accelerate()

    @property
    def data_dim(self):
        """
        Get the number of channels in the data, including color + density
        (similar to svox 1)
        """
        return self.sh_data.size(1) + 1

    @property
    def basis_reso(self):
        """
        Return the resolution of the learned spherical function data if using
        3D learned texture, or 0 if only using SH
        """
        return self.basis_data.size(0) if self.BASIS_TYPE_3D_TEXTURE else 0

    @property
    def use_background(self):
        return self.background_nlayers > 0

    @property
    def shape(self):
        return list(self.links.shape) + [self.data_dim]

    def _fetch_links(self, links):
        results_sigma = torch.zeros((links.size(0), 1), device=links.device, dtype=torch.float32)
        results_sh = torch.zeros((links.size(0), self.sh_data.size(1)), device=links.device, dtype=torch.float32)
        mask = links >= 0
        idxs = links[mask].long()
        results_sigma[mask] = self.density_data[idxs]
        results_sh[mask] = self.sh_data[idxs]
        return results_sigma, results_sh

    def sample(self, points: 'torch.Tensor', use_kernel: 'bool'=True, grid_coords: 'bool'=False, want_colors: 'bool'=True):
        """
        Grid sampling with trilinear interpolation.
        Behaves like torch.nn.functional.grid_sample
        with padding mode border and align_corners=False (better for multi-resolution).

        Any voxel with link < 0 (empty) is considered to have 0 values in all channels
        prior to interpolating.

        :param points: torch.Tensor, (N, 3)
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param grid_coords: bool, if true then uses grid coordinates ([-0.5, reso[i]-0.5 ] in each dimension);
                                  more numerically exact for resampling
        :param want_colors: bool, if true (default) returns density and colors,
                            else returns density and a dummy tensor to be ignored
                            (much faster)

        :return: (density, color)
        """
        if use_kernel and self.links.is_cuda and _C is not None:
            return _SampleGridAutogradFunction.apply(self.density_data, self.sh_data, self._to_cpp(grid_coords=grid_coords), points, want_colors)
        else:
            if not grid_coords:
                points = self.world2grid(points)
            points.clamp_min_(0.0)
            for i in range(3):
                points[:, i].clamp_max_(self.links.size(i) - 1)
            l = points
            for i in range(3):
                l[:, i].clamp_max_(self.links.size(i) - 2)
            wb = points - l
            wa = 1.0 - wb
            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]
            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            samples_sigma = c0 * wa[:, :1] + c1 * wb[:, :1]
            if want_colors:
                c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
                c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
                c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
                c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                samples_rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
            else:
                samples_rgb = torch.empty_like(self.sh_data[:0])
            return samples_sigma, samples_rgb

    def forward(self, points: 'torch.Tensor', use_kernel: 'bool'=True):
        return self.sample(points, use_kernel=use_kernel)

    def _volume_render_gradcheck_lerp(self, rays: 'Rays', return_raylen: 'bool'=False):
        """
        trilerp gradcheck version
        """
        origins = self.world2grid(rays.origins)
        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
        viewdirs = dirs
        B = dirs.size(0)
        assert origins.size(0) == B
        gsz = self._grid_size()
        dirs = dirs * (self._scaling * gsz)
        delta_scale = 1.0 / dirs.norm(dim=1)
        dirs *= delta_scale.unsqueeze(-1)
        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            sh_mult = self._eval_learned_bases(viewdirs)
        elif self.basis_type == BASIS_TYPE_MLP:
            sh_mult = torch.sigmoid(self._eval_basis_mlp(viewdirs))
        else:
            sh_mult = utils.eval_sh_bases(self.basis_dim, viewdirs)
        invdirs = 1.0 / dirs
        gsz = self._grid_size()
        gsz_cu = gsz
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz_cu - 0.5 - origins) * invdirs
        t = torch.min(t1, t2)
        t[dirs == 0] = -1000000000.0
        t = torch.max(t, dim=-1).values.clamp_min_(self.opt.near_clip)
        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1000000000.0
        tmax = torch.min(tmax, dim=-1).values
        if return_raylen:
            return tmax - t
        log_light_intensity = torch.zeros(B, device=origins.device)
        out_rgb = torch.zeros((B, 3), device=origins.device)
        good_indices = torch.arange(B, device=origins.device)
        origins_ini = origins
        dirs_ini = dirs
        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]
        del invdirs
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]
        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz[2] - 1)
            l = pos
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0], gsz[0] - 2)
            l[:, 1] = torch.clamp_max(l[:, 1], gsz[1] - 2)
            l[:, 2] = torch.clamp_max(l[:, 2], gsz[2] - 2)
            pos -= l
            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]
            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)
            wa, wb = 1.0 - pos, pos
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            sigma = c0 * wa[:, :1] + c1 * wb[:, :1]
            c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
            c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
            c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
            c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
            log_att = -self.opt.step_size * torch.relu(sigma[..., 0]) * delta_scale[good_indices]
            weight = torch.exp(log_light_intensity[good_indices]) * (1.0 - torch.exp(log_att))
            rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
            rgb = torch.clamp_min(torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5, 0.0)
            rgb = weight[:, None] * rgb[:, :3]
            out_rgb[good_indices] += rgb
            log_light_intensity[good_indices] += log_att
            t += self.opt.step_size
            mask = t <= tmax
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            t = t[mask]
            sh_mult = sh_mult[mask]
            tmax = tmax[mask]
        if self.use_background:
            csi = utils.ConcentricSpheresIntersector(gsz_cu, origins_ini, dirs_ini, delta_scale)
            inner_radius = torch.cross(csi.origins, csi.dirs, dim=-1).norm(dim=-1) + 0.001
            inner_radius = inner_radius.clamp_min(1.0)
            _, t_last = csi.intersect(inner_radius)
            n_steps = int(self.background_nlayers / self.opt.step_size) + 2
            layer_scale = (self.background_nlayers - 1) / (n_steps + 1)

            def fetch_bg_link(lx, ly, lz):
                results = torch.zeros([lx.shape[0], self.background_data.size(-1)], device=lx.device)
                lnk = self.background_links[lx, ly]
                mask = lnk >= 0
                results[mask] = self.background_data[lnk[mask].long(), lz[mask]]
                return results
            for i in range(n_steps):
                r: 'float' = n_steps / (n_steps - i - 0.5)
                normalized_inv_radius = min((i + 1) * layer_scale, self.background_nlayers - 1)
                layerid = min(int(normalized_inv_radius), self.background_nlayers - 2)
                interp_wt = normalized_inv_radius - layerid
                active_mask, t = csi.intersect(r)
                active_mask = active_mask & (r >= inner_radius)
                if active_mask.count_nonzero() == 0:
                    continue
                t_sub = t[active_mask]
                t_mid_sub = (t_sub + t_last[active_mask]) * 0.5
                sphpos = csi.origins[active_mask] + t_mid_sub.unsqueeze(-1) * csi.dirs[active_mask]
                invr_mid = 1.0 / torch.norm(sphpos, dim=-1)
                sphpos *= invr_mid.unsqueeze(-1)
                xy = utils.xyz2equirect(sphpos, self.background_links.size(1))
                z = torch.clamp((1.0 - invr_mid) * self.background_nlayers - 0.5, 0.0, self.background_nlayers - 1)
                points = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
                l = points
                l[..., 0].clamp_max_(self.background_links.size(0) - 1)
                l[..., 1].clamp_max_(self.background_links.size(1) - 1)
                l[..., 2].clamp_max_(self.background_nlayers - 2)
                wb = points - l
                wa = 1.0 - wb
                lx, ly, lz = l.unbind(-1)
                lnx = (lx + 1) % self.background_links.size(0)
                lny = (ly + 1) % self.background_links.size(1)
                lnz = lz + 1
                v000 = fetch_bg_link(lx, ly, lz)
                v001 = fetch_bg_link(lx, ly, lnz)
                v010 = fetch_bg_link(lx, lny, lz)
                v011 = fetch_bg_link(lx, lny, lnz)
                v100 = fetch_bg_link(lnx, ly, lz)
                v101 = fetch_bg_link(lnx, ly, lnz)
                v110 = fetch_bg_link(lnx, lny, lz)
                v111 = fetch_bg_link(lnx, lny, lnz)
                c00 = v000 * wa[:, 2:] + v001 * wb[:, 2:]
                c01 = v010 * wa[:, 2:] + v011 * wb[:, 2:]
                c10 = v100 * wa[:, 2:] + v101 * wb[:, 2:]
                c11 = v110 * wa[:, 2:] + v111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                rgba = c0 * wa[:, :1] + c1 * wb[:, :1]
                log_att = -csi.world_step_scale[active_mask] * torch.relu(rgba[:, -1]) * (t_sub - t_last[active_mask])
                weight = torch.exp(log_light_intensity[active_mask]) * (1.0 - torch.exp(log_att))
                rgb = torch.clamp_min(rgba[:, :3] * utils.SH_C0 + 0.5, 0.0)
                out_rgb[active_mask] += rgb * weight[:, None]
                log_light_intensity[active_mask] += log_att
                t_last[active_mask] = t[active_mask]
        if self.opt.background_brightness:
            out_rgb += torch.exp(log_light_intensity).unsqueeze(-1) * self.opt.background_brightness
        return out_rgb

    def _volume_render_gradcheck_nvol_lerp(self, rays: 'Rays', return_raylen: 'bool'=False):
        """
        trilerp gradcheck version
        """
        origins = self.world2grid(rays.origins)
        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)
        viewdirs = dirs
        B = dirs.size(0)
        assert origins.size(0) == B
        gsz = self._grid_size()
        dirs = dirs * (self._scaling * gsz)
        delta_scale = 1.0 / dirs.norm(dim=1)
        dirs *= delta_scale.unsqueeze(-1)
        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            sh_mult = self._eval_learned_bases(viewdirs)
        elif self.basis_type == BASIS_TYPE_MLP:
            sh_mult = torch.sigmoid(self._eval_basis_mlp(viewdirs))
        else:
            sh_mult = utils.eval_sh_bases(self.basis_dim, viewdirs)
        invdirs = 1.0 / dirs
        gsz = self._grid_size()
        gsz_cu = gsz
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz_cu - 0.5 - origins) * invdirs
        t = torch.min(t1, t2)
        t[dirs == 0] = -1000000000.0
        t = torch.max(t, dim=-1).values.clamp_min_(self.opt.near_clip)
        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1000000000.0
        tmax = torch.min(tmax, dim=-1).values
        if return_raylen:
            return tmax - t
        total_alpha = torch.zeros(B, device=origins.device)
        out_rgb = torch.zeros((B, 3), device=origins.device)
        good_indices = torch.arange(B, device=origins.device)
        origins_ini = origins
        dirs_ini = dirs
        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]
        del invdirs
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]
        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz[2] - 1)
            l = pos
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0], gsz[0] - 2)
            l[:, 1] = torch.clamp_max(l[:, 1], gsz[1] - 2)
            l[:, 2] = torch.clamp_max(l[:, 2], gsz[2] - 2)
            pos -= l
            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]
            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)
            wa, wb = 1.0 - pos, pos
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            sigma = c0 * wa[:, :1] + c1 * wb[:, :1]
            c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
            c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
            c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
            c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
            log_att = -self.opt.step_size * torch.relu(sigma[..., 0]) * delta_scale[good_indices]
            delta_alpha = 1.0 - torch.exp(log_att)
            total_alpha_sub = total_alpha[good_indices]
            new_total_alpha = torch.clamp_max(total_alpha_sub + delta_alpha, 1.0)
            weight = new_total_alpha - total_alpha_sub
            total_alpha[good_indices] = new_total_alpha
            rgb_sh = rgb.reshape(-1, 3, self.basis_dim)
            rgb = torch.clamp_min(torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5, 0.0)
            rgb = weight[:, None] * rgb[:, :3]
            out_rgb[good_indices] += rgb
            t += self.opt.step_size
            mask = t <= tmax
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            t = t[mask]
            sh_mult = sh_mult[mask]
            tmax = tmax[mask]
        if self.opt.background_brightness:
            out_rgb += (1.0 - total_alpha).unsqueeze(-1) * self.opt.background_brightness
        return out_rgb

    def volume_render(self, rays: 'Rays', use_kernel: 'bool'=True, randomize: 'bool'=False, return_raylen: 'bool'=False):
        """
        Standard volume rendering. See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to enable randomness
        :param return_raylen: bool, if true then only returns the length of the
                                    ray-cube intersection and quits
        :return: (N, 3), predicted RGB
        """
        if use_kernel and self.links.is_cuda and _C is not None and not return_raylen:
            basis_data = self._eval_basis_mlp(rays.dirs) if self.basis_type == BASIS_TYPE_MLP else None
            return _VolumeRenderFunction.apply(self.density_data, self.sh_data, basis_data, self.background_data if self.use_background else None, self._to_cpp(replace_basis_data=basis_data), rays._to_cpp(), self.opt._to_cpp(randomize=randomize), self.opt.backend)
        else:
            warn('Using slow volume rendering, should only be used for debugging')
            if self.opt.backend == 'nvol':
                return self._volume_render_gradcheck_nvol_lerp(rays, return_raylen=return_raylen)
            else:
                return self._volume_render_gradcheck_lerp(rays, return_raylen=return_raylen)

    def alloc_grad_indexers(self):
        self.sparse_grad_indexer = torch.zeros((self.density_data.size(0),), dtype=torch.bool, device=self.density_data.device)
        self.sparse_background_indexer = torch.zeros(list(self.background_data.shape[:-1]), dtype=torch.bool, device=self.background_data.device)

    def delete_grad_indexers(self):
        del self.sparse_grad_indexer
        del self.sparse_sh_grad_indexer
        del self.sparse_background_indexer
        self.sparse_grad_indexer = None
        self.sparse_sh_grad_indexer = None
        self.sparse_background_indexer = None

    def delete_grads(self):
        for subitem in ['density_data', 'sh_data', 'basis_data', 'background_data']:
            param = self.__getattr__(subitem)
            if hasattr(param, 'grad'):
                del param.grad
        self.delete_grad_indexers()

    def print_gpu_memory(self):
        mem_str = 'Grid memory footprint: \n'
        total_mem = 0
        for param in dir(self):
            try:
                value = getattr(self, param)
                if torch.is_tensor(value) and value.is_cuda:
                    mem = value.element_size() * value.nelement() / 1000000.0
                    total_mem += mem
                    mem_str += f'{param}: {mem} MB\n'
                    if hasattr(value, 'grad'):
                        value = value.grad
                        if value is not None:
                            mem = value.element_size() * value.nelement() / 1000000.0
                            total_mem += mem
                            mem_str += f'{param}.grad: {mem} MB\n'
            except:
                pass
        mem_str += f'total: {total_mem} MB\n'
        ic(mem_str)

    def volume_render_fused(self, rays: 'Rays', rgb_gt: 'torch.Tensor', randomize: 'bool'=False, beta_loss: 'float'=0.0, sparsity_loss: 'float'=0.0, is_rgb_gt: 'bool'=True, reset_grad_indexers: 'bool'=True):
        """
        Standard volume rendering with fused MSE gradient generation,
            given a ground truth color for each pixel.
        Will update the *.grad tensors for each parameter
        You can then subtract the grad manually or use the optim_*_step methods

        See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param rgb_gt: (N, 3), GT pixel colors, each channel in [0, 1]
        :param randomize: bool, whether to enable randomness
        :param beta_loss: float, weighting for beta loss to add to the gradient.
                                 (fused into the backward pass).
                                 This is average voer the rays in the batch.
                                 Beta loss also from neural volumes:
                                 [Lombardi et al., ToG 2019]
        :return: (N, 3), predicted RGB
        """
        grad_density, grad_sh, grad_basis, grad_bg = self._get_data_grads()
        rgb_out = torch.zeros_like(rgb_gt)
        basis_data: 'Optional[torch.Tensor]' = None
        if self.basis_type == BASIS_TYPE_MLP:
            with torch.enable_grad():
                basis_data = self._eval_basis_mlp(rays.dirs)
            grad_basis = torch.empty_like(basis_data)
        if reset_grad_indexers or self.sparse_grad_indexer is None:
            self.sparse_grad_indexer = torch.zeros((self.density_data.size(0),), dtype=torch.bool, device=self.density_data.device)
        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density
        grad_holder.grad_sh_out = grad_sh
        if self.basis_type != BASIS_TYPE_SH:
            grad_holder.grad_basis_out = grad_basis
        grad_holder.mask_out = self.sparse_grad_indexer
        if self.use_background:
            grad_holder.grad_background_out = grad_bg
            if reset_grad_indexers or self.sparse_background_indexer is None:
                self.sparse_background_indexer = torch.zeros(list(self.background_data.shape[:-1]), dtype=torch.bool, device=self.background_data.device)
            grad_holder.mask_background_out = self.sparse_background_indexer
        cu_fn = _C.__dict__[f'volume_render_{self.opt.backend}_fused']
        cu_fn(self._to_cpp(replace_basis_data=basis_data), rays._to_cpp(), self.opt._to_cpp(randomize=randomize), rgb_gt, beta_loss, sparsity_loss, rgb_out, is_rgb_gt, grad_holder)
        if self.basis_type == BASIS_TYPE_MLP:
            basis_data.backward(grad_basis)
        self.sparse_sh_grad_indexer = self.sparse_grad_indexer.clone()
        return rgb_out

    def volume_render_image(self, camera: 'Camera', use_kernel: 'bool'=True, randomize: 'bool'=False, batch_size: 'int'=5000, return_raylen: 'bool'=False):
        """
        Standard volume rendering (entire image version).
        See grid.opt.* (RenderOptions) for configs.

        :param camera: Camera
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param randomize: bool, whether to enable randomness
        :return: (H, W, 3), predicted RGB image
        """
        imrend_fn_name = f'volume_render_{self.opt.backend}_image'
        if self.basis_type != BASIS_TYPE_MLP and imrend_fn_name in _C.__dict__ and not torch.is_grad_enabled() and not return_raylen:
            cu_fn = _C.__dict__[imrend_fn_name]
            return cu_fn(self._to_cpp(), camera._to_cpp(), self.opt._to_cpp())
        else:
            rays = camera.gen_rays()
            all_rgb_out = []
            for batch_start in range(0, camera.height * camera.width, batch_size):
                rgb_out_part = self.volume_render(rays[batch_start:batch_start + batch_size], use_kernel=use_kernel, randomize=randomize, return_raylen=return_raylen)
                all_rgb_out.append(rgb_out_part)
            all_rgb_out = torch.cat(all_rgb_out, dim=0)
            return all_rgb_out.view(camera.height, camera.width, -1)

    def volume_render_depth(self, rays: 'Rays', sigma_thresh: 'Optional[float]'=None):
        """
        Volumetric depth rendering for rays

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param sigma_thresh: Optional[float]. If None then finds the standard expected termination
                                              (NOTE: this is the absolute length along the ray, not the z-depth as usually expected);
                                              else then finds the first point where sigma strictly exceeds sigma_thresh

        :return: (N,)
        """
        if sigma_thresh is None:
            return _C.volume_render_expected_term(self._to_cpp(), rays._to_cpp(), self.opt._to_cpp())
        else:
            return _C.volume_render_sigma_thresh(self._to_cpp(), rays._to_cpp(), self.opt._to_cpp(), sigma_thresh)

    def volume_render_depth_image(self, camera: 'Camera', sigma_thresh: 'Optional[float]'=None, batch_size: 'int'=5000):
        """
        Volumetric depth rendering for full image

        :param camera: Camera, a single camera
        :param sigma_thresh: Optional[float]. If None then finds the standard expected termination
                                              (NOTE: this is the absolute length along the ray, not the z-depth as usually expected);
                                              else then finds the first point where sigma strictly exceeds sigma_thresh

        :return: depth (H, W)
        """
        rays = camera.gen_rays()
        all_depths = []
        for batch_start in range(0, camera.height * camera.width, batch_size):
            depths = self.volume_render_depth(rays[batch_start:batch_start + batch_size], sigma_thresh)
            all_depths.append(depths)
        all_depth_out = torch.cat(all_depths, dim=0)
        return all_depth_out.view(camera.height, camera.width)

    def volume_render_position_map(self, camera: 'Camera', sigma_thresh: 'Optional[float]'=None, batch_size: 'int'=5000):
        """
        Volumetric position rendering for full image

        :param camera: Camera, a single camera
        :param sigma_thresh: Optional[float]. If None then finds the standard expected termination
                                              (NOTE: this is the absolute length along the ray, not the z-depth as usually expected);
                                              else then finds the first point where sigma strictly exceeds sigma_thresh

        :return: depth (H, W)
        """
        rays = camera.gen_rays()
        all_depths = []
        all_positions = []
        for batch_start in range(0, camera.height * camera.width, batch_size):
            depths = self.volume_render_depth(rays[batch_start:batch_start + batch_size], sigma_thresh)
            all_depths.append(depths)
            all_positions.append(rays[batch_start:batch_start + batch_size].origins + rays[batch_start:batch_start + batch_size].dirs * depths)
        all_depth_out = torch.cat(all_depths, dim=0)
        all_positions = torch.cat(all_positions, dim=0)
        None
        return all_positions.view(camera.height, camera.width, 3)

    def resample(self, reso: 'Union[int, List[int]]', sigma_thresh: 'float'=5.0, weight_thresh: 'float'=0.01, dilate: 'int'=2, cameras: 'Optional[List[Camera]]'=None, use_z_order: 'bool'=False, accelerate: 'bool'=True, weight_render_stop_thresh: 'float'=0.2, max_elements: 'int'=0):
        """
        Resample and sparsify the grid; used to increase the resolution
        :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
        :param sigma_thresh: float, threshold to apply on the sigma (if using sigma thresh i.e. cameras NOT given)
        :param weight_thresh: float, threshold to apply on the weights (if using weight thresh i.e. cameras given)
        :param dilate: int, if true applies dilation of size <dilate> to the 3D mask for nodes to keep in the grid
                             (keep neighbors in all 28 directions, including diagonals, of the desired nodes)
        :param cameras: Optional[List[Camera]], optional list of cameras in OpenCV convention (if given, uses weight thresholding)
        :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
        :param accelerate: bool, if true (default), calls grid.accelerate() after resampling
                           to build distance transform table (only if on CUDA)
        :param weight_render_stop_thresh: float, stopping threshold for grid weight render in [0, 1];
                                                 0.0 = no thresholding, 1.0 = hides everything.
                                                 Useful for force-cutting off
                                                 junk that contributes very little at the end of a ray
        :param max_elements: int, if nonzero, an upper bound on the number of elements in the
                upsampled grid; we will adjust the threshold to match it
        """
        with torch.no_grad():
            device = self.links.device
            if isinstance(reso, int):
                reso = [reso] * 3
            else:
                assert len(reso) == 3, 'reso must be an integer or indexable object of 3 ints'
            if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
                None
                use_z_order = False
            self.capacity: 'int' = reduce(lambda x, y: x * y, reso)
            curr_reso = self.links.shape
            dtype = torch.float32
            reso_facts = [(0.5 * curr_reso[i] / reso[i]) for i in range(3)]
            X = torch.linspace(reso_facts[0] - 0.5, curr_reso[0] - reso_facts[0] - 0.5, reso[0], dtype=dtype)
            Y = torch.linspace(reso_facts[1] - 0.5, curr_reso[1] - reso_facts[1] - 0.5, reso[1], dtype=dtype)
            Z = torch.linspace(reso_facts[2] - 0.5, curr_reso[2] - reso_facts[2] - 0.5, reso[2], dtype=dtype)
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
            if use_z_order:
                morton = utils.gen_morton(reso[0], dtype=torch.long).view(-1)
                points[morton] = points.clone()
            points = points
            use_weight_thresh = cameras is not None
            batch_size = 720720
            all_sample_vals_density = []
            None
            for i in tqdm(range(0, len(points), batch_size)):
                sample_vals_density, _ = self.sample(points[i:i + batch_size], grid_coords=True, want_colors=False)
                sample_vals_density = sample_vals_density
                all_sample_vals_density.append(sample_vals_density)
            self.density_data.grad = None
            self.sh_data.grad = None
            self.sparse_grad_indexer = None
            self.sparse_sh_grad_indexer = None
            self.density_rms = None
            self.sh_rms = None
            sample_vals_density = torch.cat(all_sample_vals_density, dim=0).view(reso)
            del all_sample_vals_density
            if use_weight_thresh:
                gsz = torch.tensor(reso)
                offset = self._offset * gsz - 0.5
                scaling = self._scaling * gsz
                max_wt_grid = torch.zeros(reso, dtype=torch.float32, device=device)
                None
                for i, cam in enumerate(cameras):
                    _C.grid_weight_render(sample_vals_density, cam._to_cpp(), 0.5, weight_render_stop_thresh, False, offset, scaling, max_wt_grid)
                sample_vals_mask = max_wt_grid >= weight_thresh
                if max_elements > 0 and max_elements < max_wt_grid.numel() and max_elements < torch.count_nonzero(sample_vals_mask):
                    weight_thresh_bounded = torch.topk(max_wt_grid.view(-1), k=max_elements, sorted=False).values.min().item()
                    weight_thresh = max(weight_thresh, weight_thresh_bounded)
                    None
                    sample_vals_mask = max_wt_grid >= weight_thresh
                del max_wt_grid
            else:
                sample_vals_mask = sample_vals_density >= sigma_thresh
                if max_elements > 0 and max_elements < sample_vals_density.numel() and max_elements < torch.count_nonzero(sample_vals_mask):
                    sigma_thresh_bounded = torch.topk(sample_vals_density.view(-1), k=max_elements, sorted=False).values.min().item()
                    sigma_thresh = max(sigma_thresh, sigma_thresh_bounded)
                    None
                    sample_vals_mask = sample_vals_density >= sigma_thresh
                if self.opt.last_sample_opaque:
                    sample_vals_mask[:, :, -1] = 1
            if dilate:
                for i in range(int(dilate)):
                    sample_vals_mask = _C.dilate(sample_vals_mask)
            sample_vals_mask = sample_vals_mask.view(-1)
            sample_vals_density = sample_vals_density.view(-1)
            sample_vals_density = sample_vals_density[sample_vals_mask]
            cnz = torch.count_nonzero(sample_vals_mask).item()
            points = points[sample_vals_mask]
            None
            all_sample_vals_sh = []
            for i in tqdm(range(0, len(points), batch_size)):
                _, sample_vals_sh = self.sample(points[i:i + batch_size], grid_coords=True, want_colors=True)
                all_sample_vals_sh.append(sample_vals_sh)
            sample_vals_sh = torch.cat(all_sample_vals_sh, dim=0) if len(all_sample_vals_sh) else torch.empty_like(self.sh_data[:0])
            del self.density_data
            del self.sh_data
            del all_sample_vals_sh
            if use_z_order:
                inv_morton = torch.empty_like(morton)
                inv_morton[morton] = torch.arange(morton.size(0), dtype=morton.dtype)
                inv_idx = inv_morton[sample_vals_mask]
                init_links = torch.full((sample_vals_mask.size(0),), fill_value=-1, dtype=torch.int32)
                init_links[inv_idx] = torch.arange(inv_idx.size(0), dtype=torch.int32)
            else:
                init_links = torch.cumsum(sample_vals_mask, dim=-1).int() - 1
                init_links[~sample_vals_mask] = -1
            self.capacity = cnz
            None
            del sample_vals_mask
            None
            None
            None
            self.density_data = nn.Parameter(sample_vals_density.view(-1, 1))
            self.sh_data = nn.Parameter(sample_vals_sh)
            self.links = init_links.view(reso)
            if accelerate and self.links.is_cuda:
                self.accelerate()

    def sparsify_background(self, sigma_thresh: 'float'=1.0, dilate: 'int'=1):
        device = self.background_links.device
        sigma_mask = torch.zeros(list(self.background_links.shape) + [self.background_nlayers], dtype=torch.bool, device=device).view(-1, self.background_nlayers)
        nonempty_mask = self.background_links.view(-1) >= 0
        data_mask = self.background_data[..., -1] >= sigma_thresh
        sigma_mask[nonempty_mask] = data_mask
        sigma_mask = sigma_mask.view(list(self.background_links.shape) + [self.background_nlayers])
        for _ in range(int(dilate)):
            sigma_mask = _C.dilate(sigma_mask)
        sigma_mask = sigma_mask.any(-1) & nonempty_mask.view(self.background_links.shape)
        self.background_links[~sigma_mask] = -1
        retain_vals = self.background_links[sigma_mask]
        self.background_links[sigma_mask] = torch.arange(retain_vals.size(0), dtype=torch.int32, device=device)
        self.background_data = nn.Parameter(self.background_data.data[retain_vals.long()])

    def resize(self, basis_dim: 'int'):
        """
        Modify the size of the data stored in the voxels. Called expand/shrink in svox 1.

        :param basis_dim: new basis dimension, must be square number
        """
        assert utils.isqrt(basis_dim) is not None, 'basis_dim (SH) must be a square number'
        assert basis_dim >= 1 and basis_dim <= utils.MAX_SH_BASIS, f'basis_dim 1-{utils.MAX_SH_BASIS} supported'
        old_basis_dim = self.basis_dim
        self.basis_dim = basis_dim
        device = self.sh_data.device
        old_data = self.sh_data.data.cpu()
        shrinking = basis_dim < old_basis_dim
        sigma_arr = torch.tensor([0])
        if shrinking:
            shift = old_basis_dim
            arr = torch.arange(basis_dim)
            remap = torch.cat([arr, shift + arr, 2 * shift + arr])
        else:
            shift = basis_dim
            arr = torch.arange(old_basis_dim)
            remap = torch.cat([arr, shift + arr, 2 * shift + arr])
        del self.sh_data
        new_data = torch.zeros((old_data.size(0), 3 * basis_dim + 1), device='cpu')
        if shrinking:
            new_data[:] = old_data[..., remap]
        else:
            new_data[..., remap] = old_data
        new_data = new_data
        self.sh_data = nn.Parameter(new_data)
        self.sh_rms = None

    def accelerate(self):
        """
        Accelerate
        """
        _C.accel_dist_prop(self.links)

    def world2grid(self, points):
        """
        World coordinates to grid coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        offset = self._offset * gsz - 0.5
        scaling = self._scaling * gsz
        return torch.addcmul(offset, points, scaling)

    def grid2world(self, points):
        """
        Grid coordinates to world coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        roffset = self.radius * (1.0 / gsz - 1.0) + self.center
        rscaling = 2.0 * self.radius / gsz
        return torch.addcmul(roffset, points, rscaling)

    def apply_ct(self, apply_ct: 'np.ndarray'):
        sh_data = self.sh_data.data.cpu().numpy()
        cnt = sh_data.shape[0]
        sh_data = sh_data.reshape((cnt, 3, -1))
        sh_data = np.ascontiguousarray(np.transpose(sh_data, (0, 2, 1))).reshape((-1, 3))
        apply_ct = apply_ct.astype(np.float32)
        sh_data = (sh_data @ apply_ct[:3, :3].T + apply_ct[:3, 3][np.newaxis, :]).astype(np.float32)
        sh_data = np.ascontiguousarray(np.transpose(sh_data.reshape((cnt, -1, 3)), (0, 2, 1))).reshape((cnt, -1))
        sh_data = torch.from_numpy(sh_data)
        self.sh_data.data.copy_(sh_data)
        if self.use_background:
            background_data = self.background_data.data.cpu().numpy()
            apply_ct = apply_ct.astype(np.float32)
            shape = background_data.shape
            background_data = np.ascontiguousarray(background_data.reshape(-1, 4))
            background_data[:, :3] = (background_data[:, :3] @ apply_ct[:3, :3].T + apply_ct[:3, 3][np.newaxis, :]).astype(np.float32)
            background_data = background_data.reshape(list(shape))
            background_data = torch.from_numpy(background_data).contiguous()
            self.background_data.data.copy_(background_data)

    def save(self, path: 'str', compress: 'bool'=False, apply_ct: 'np.ndarray'=None):
        """
        Save to a path
        """
        save_fn = np.savez_compressed if compress else np.savez
        sh_data = self.sh_data.data.cpu().numpy()
        if apply_ct is not None:
            cnt = sh_data.shape[0]
            sh_data = sh_data.reshape((cnt, 3, -1))
            sh_data = np.ascontiguousarray(np.transpose(sh_data, (0, 2, 1))).reshape((-1, 3))
            apply_ct = apply_ct.astype(np.float32)
            sh_data = (sh_data @ apply_ct[:3, :3].T + apply_ct[:3, 3][np.newaxis, :]).astype(np.float32)
            sh_data = np.ascontiguousarray(np.transpose(sh_data.reshape((cnt, -1, 3)), (0, 2, 1))).reshape((cnt, -1))
        data = {'radius': self.radius.numpy(), 'center': self.center.numpy(), 'links': self.links.cpu().numpy(), 'density_data': self.density_data.data.cpu().numpy(), 'sh_data': sh_data.astype(np.float16)}
        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            data['basis_data'] = self.basis_data.data.cpu().numpy()
        elif self.basis_type == BASIS_TYPE_MLP:
            utils.net_to_dict(data, 'basis_mlp', self.basis_mlp)
            data['mlp_posenc_size'] = np.int32(self.mlp_posenc_size)
            data['mlp_width'] = np.int32(self.mlp_width)
        if self.use_background:
            data['background_links'] = self.background_links.cpu().numpy()
            background_data = self.background_data.data.cpu().numpy()
            if apply_ct is not None:
                apply_ct = apply_ct.astype(np.float32)
                shape = background_data.shape
                background_data = np.ascontiguousarray(background_data.reshape(-1, 4))
                background_data[:, :3] = (background_data[:, :3] @ apply_ct[:3, :3].T + apply_ct[:3, 3][np.newaxis, :]).astype(np.float32)
                background_data = background_data.reshape(list(shape))
            data['background_data'] = background_data
        data['basis_type'] = self.basis_type
        save_fn(path, **data)

    @classmethod
    def load(cls, path: 'str', device: 'Union[torch.device, str]'='cpu', reset_basis_dim: 'int'=0):
        """
        Load from path
        """
        z = np.load(path)
        if 'data' in z.keys():
            all_data = z.f.data
            sh_data = all_data[..., 1:]
            density_data = all_data[..., :1]
        else:
            sh_data = z.f.sh_data
            density_data = z.f.density_data
        if 'background_data' in z:
            background_data = z['background_data']
            background_links = z['background_links']
        else:
            background_data = None
        basis_dim = sh_data.shape[1] // 3
        if reset_basis_dim > 0:
            sh_data = sh_data.reshape([-1, 3, basis_dim])
            sh_data = np.ascontiguousarray(sh_data[:, :, :reset_basis_dim]).reshape([-1, 3 * reset_basis_dim]).astype(np.float32)
            basis_dim = reset_basis_dim
        links = z.f.links
        radius = z.f.radius.tolist() if 'radius' in z.files else [1.0, 1.0, 1.0]
        center = z.f.center.tolist() if 'center' in z.files else [0.0, 0.0, 0.0]
        grid = cls(1, radius=radius, center=center, basis_dim=basis_dim, use_z_order=False, device='cpu', basis_type=z['basis_type'].item() if 'basis_type' in z else BASIS_TYPE_SH, mlp_posenc_size=z['mlp_posenc_size'].item() if 'mlp_posenc_size' in z else 0, mlp_width=z['mlp_width'].item() if 'mlp_width' in z else 16, background_nlayers=0)
        if sh_data.dtype != np.float32:
            sh_data = sh_data.astype(np.float32)
        if density_data.dtype != np.float32:
            density_data = density_data.astype(np.float32)
        sh_data = torch.from_numpy(sh_data)
        density_data = torch.from_numpy(density_data)
        grid.sh_data = nn.Parameter(sh_data)
        grid.density_data = nn.Parameter(density_data)
        grid.links = torch.from_numpy(links)
        grid.capacity = grid.sh_data.size(0)
        if grid.basis_type == BASIS_TYPE_MLP:
            utils.net_from_dict(z, 'basis_mlp', grid.basis_mlp)
            grid.basis_mlp = grid.basis_mlp
        elif grid.basis_type == BASIS_TYPE_3D_TEXTURE or 'basis_data' in z.keys():
            basis_data = torch.from_numpy(z.f.basis_data)
            grid.basis_type = BASIS_TYPE_3D_TEXTURE
            grid.basis_data = nn.Parameter(basis_data)
        else:
            grid.basis_data = nn.Parameter(grid.basis_data.data)
        if background_data is not None:
            background_data = torch.from_numpy(background_data)
            grid.background_nlayers = background_data.shape[1]
            grid.background_reso = background_links.shape[1]
            grid.background_data = nn.Parameter(background_data)
            grid.background_links = torch.from_numpy(background_links)
        else:
            grid.background_data.data = grid.background_data.data
        if grid.links.is_cuda:
            grid.accelerate()
        return grid

    def to_svox1(self, device: 'Union[torch.device, str, None]'=None):
        """
        Convert the grid to a svox 1 octree. Requires svox (pip install svox)

        :param device: device to put the octree. None = grid data's device
        """
        assert self.is_cubic_pow2, 'Grid must be cubic and power-of-2 to be compatible with svox octree'
        if device is None:
            device = self.sh_data.device
        n_refine = int(np.log2(self.links.size(0))) - 1
        t = svox.N3Tree(data_format=f'SH{self.basis_dim}', init_refine=0, radius=self.radius.tolist(), center=self.center.tolist(), device=device)
        curr_reso = self.links.shape
        dtype = torch.float32
        X = (torch.arange(curr_reso[0], dtype=dtype, device=device) + 0.5) / curr_reso[0]
        Y = (torch.arange(curr_reso[1], dtype=dtype, device=device) + 0.5) / curr_reso[0]
        Z = (torch.arange(curr_reso[2], dtype=dtype, device=device) + 0.5) / curr_reso[0]
        X, Y, Z = torch.meshgrid(X, Y, Z)
        points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
        mask = self.links.view(-1) >= 0
        points = points[mask]
        index = svox.LocalIndex(points)
        None
        for i in tqdm(range(n_refine)):
            t[index].refine()
        t[index, :-1] = self.sh_data.data
        t[index, -1:] = self.density_data.data
        return t

    def tv(self, logalpha: 'bool'=False, logalpha_delta: 'float'=2.0, ndc_coeffs: 'Tuple[float, float]'=(-1.0, -1.0)):
        """
        Compute total variation over sigma,
        similar to Neural Volumes [Lombardi et al., ToG 2019]

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
                 mean over voxels)
        """
        assert not logalpha, 'No longer supported'
        return _TotalVariationFunction.apply(self.density_data, self.links, 0, 1, logalpha, logalpha_delta, False, ndc_coeffs)

    def tv_color(self, start_dim: 'int'=0, end_dim: 'Optional[int]'=None, logalpha: 'bool'=False, logalpha_delta: 'float'=2.0, ndc_coeffs: 'Tuple[float, float]'=(-1.0, -1.0)):
        """
        Compute total variation on color

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
                          Default None = all dimensions until the end.

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
                 mean over voxels)
        """
        assert not logalpha, 'No longer supported'
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim
        return _TotalVariationFunction.apply(self.sh_data, self.links, start_dim, end_dim, logalpha, logalpha_delta, True, ndc_coeffs)

    def tv_basis(self):
        bd = self.basis_data
        return torch.mean(torch.sqrt(1e-05 + (bd[:-1, :-1, 1:] - bd[:-1, :-1, :-1]) ** 2 + (bd[:-1, 1:, :-1] - bd[:-1, :-1, :-1]) ** 2 + (bd[1:, :-1, :-1] - bd[:-1, :-1, :-1]) ** 2).sum(dim=-1))

    def inplace_tv_grad(self, grad: 'torch.Tensor', scaling: 'float'=1.0, sparse_frac: 'float'=0.01, logalpha: 'bool'=False, logalpha_delta: 'float'=2.0, ndc_coeffs: 'Tuple[float, float]'=(-1.0, -1.0), contiguous: 'bool'=True):
        """
        Add gradient of total variation for sigma as in Neural Volumes
        [Lombardi et al., ToG 2019]
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert not logalpha, 'No longer supported'
        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                _C.tv_grad_sparse(self.links, self.density_data, rand_cells, self._get_sparse_grad_indexer(), 0, 1, scaling, logalpha, logalpha_delta, False, self.opt.last_sample_opaque, ndc_coeffs[0], ndc_coeffs[1], grad)
        else:
            _C.tv_grad(self.links, self.density_data, 0, 1, scaling, logalpha, logalpha_delta, False, ndc_coeffs[0], ndc_coeffs[1], grad)
            self.sparse_grad_indexer: 'Optional[torch.Tensor]' = None

    def inplace_tv_color_grad(self, grad: 'torch.Tensor', start_dim: 'int'=0, end_dim: 'Optional[int]'=None, scaling: 'float'=1.0, sparse_frac: 'float'=0.01, logalpha: 'bool'=False, logalpha_delta: 'float'=2.0, ndc_coeffs: 'Tuple[float, float]'=(-1.0, -1.0), contiguous: 'bool'=True):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
                          Default None = all dimensions until the end.
        """
        assert not logalpha, 'No longer supported'
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim
        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                indexer = self._get_sparse_sh_grad_indexer()
                _C.tv_grad_sparse(self.links, self.sh_data, rand_cells, indexer, start_dim, end_dim, scaling, logalpha, logalpha_delta, True, False, ndc_coeffs[0], ndc_coeffs[1], grad)
        else:
            _C.tv_grad(self.links, self.sh_data, start_dim, end_dim, scaling, logalpha, logalpha_delta, True, ndc_coeffs[0], ndc_coeffs[1], grad)
            self.sparse_sh_grad_indexer = None

    def inplace_tv_lumisphere_grad(self, grad: 'torch.Tensor', start_dim: 'int'=0, end_dim: 'Optional[int]'=None, scaling: 'float'=1.0, sparse_frac: 'float'=0.01, logalpha: 'bool'=False, logalpha_delta: 'float'=2.0, ndc_coeffs: 'Tuple[float, float]'=(-1.0, -1.0), dir_factor: 'float'=1.0, dir_perturb_radians: 'float'=0.05):
        assert self.basis_type != BASIS_TYPE_MLP, 'MLP not supported'
        rand_cells = self._get_rand_cells(sparse_frac)
        grad_holder = _C.GridOutputGrads()
        indexer = self._get_sparse_sh_grad_indexer()
        assert indexer is not None
        grad_holder.mask_out = indexer
        grad_holder.grad_sh_out = grad
        batch_size = rand_cells.size(0)
        dirs = torch.randn(3, device=rand_cells.device)
        dirs /= torch.norm(dirs)
        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            sh_mult = self._eval_learned_bases(dirs[None])
        elif self.basis_type == BASIS_TYPE_MLP:
            sh_mult = torch.sigmoid(self._eval_basis_mlp(dirs[None]))
        else:
            sh_mult = utils.eval_sh_bases(self.basis_dim, dirs[None])
        sh_mult = sh_mult[0]
        if dir_factor > 0.0:
            axis = torch.randn((batch_size, 3))
            axis /= torch.norm(axis, dim=-1, keepdim=True)
            axis *= dir_perturb_radians
            R = Rotation.from_rotvec(axis.numpy()).as_matrix()
            R = torch.from_numpy(R).float()
            dirs_perturb = (R * dirs.unsqueeze(-2)).sum(-1)
        else:
            dirs_perturb = dirs
        if self.basis_type == BASIS_TYPE_3D_TEXTURE:
            sh_mult_u = self._eval_learned_bases(dirs_perturb[None])
        elif self.basis_type == BASIS_TYPE_MLP:
            sh_mult_u = torch.sigmoid(self._eval_basis_mlp(dirs_perturb[None]))
        else:
            sh_mult_u = utils.eval_sh_bases(self.basis_dim, dirs_perturb[None])
        sh_mult_u = sh_mult_u[0]
        _C.lumisphere_tv_grad_sparse(self._to_cpp(), rand_cells, sh_mult, sh_mult_u, scaling, ndc_coeffs[0], ndc_coeffs[1], dir_factor, grad_holder)

    def inplace_l2_color_grad(self, grad: 'torch.Tensor', start_dim: 'int'=0, end_dim: 'Optional[int]'=None, scaling: 'float'=1.0):
        """
        Add gradient of L2 regularization for color
        directly into the gradient tensor, multiplied by 'scaling'
        (no CUDA extension used)

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
                          Default None = all dimensions until the end.
        """
        with torch.no_grad():
            if end_dim is None:
                end_dim = self.sh_data.size(1)
            end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
            start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim
            if self.sparse_sh_grad_indexer is None:
                scale = scaling / self.sh_data.size(0)
                grad[:, start_dim:end_dim] += scale * self.sh_data[:, start_dim:end_dim]
            else:
                indexer = self._maybe_convert_sparse_grad_indexer(sh=True)
                nz: 'int' = torch.count_nonzero(indexer).item() if indexer.dtype == torch.bool else indexer.size(0)
                scale = scaling / nz
                grad[indexer, start_dim:end_dim] += scale * self.sh_data[indexer, start_dim:end_dim]

    def inplace_tv_background_grad(self, grad: 'torch.Tensor', scaling: 'float'=1.0, scaling_density: 'Optional[float]'=None, sparse_frac: 'float'=0.01, contiguous: 'bool'=False):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'
        """
        rand_cells_bg = self._get_rand_cells_background(sparse_frac, contiguous)
        indexer = self._get_sparse_background_grad_indexer()
        if scaling_density is None:
            scaling_density = scaling
        _C.msi_tv_grad_sparse(self.background_links, self.background_data, rand_cells_bg, indexer, scaling, scaling_density, grad)

    def inplace_tv_basis_grad(self, grad: 'torch.Tensor', scaling: 'float'=1.0):
        bd = self.basis_data
        tv_val = torch.mean(torch.sqrt(1e-05 + (bd[:-1, :-1, 1:] - bd[:-1, :-1, :-1]) ** 2 + (bd[:-1, 1:, :-1] - bd[:-1, :-1, :-1]) ** 2 + (bd[1:, :-1, :-1] - bd[:-1, :-1, :-1]) ** 2).sum(dim=-1))
        tv_val_scaled = tv_val * scaling
        tv_val_scaled.backward()

    def optim_density_step(self, lr: 'float', beta: 'float'=0.9, epsilon: 'float'=1e-08, optim: 'str'='rmsprop'):
        """
        Execute RMSprop or sgd step on density
        """
        indexer = self._maybe_convert_sparse_grad_indexer()
        if optim == 'rmsprop':
            if self.density_rms is None or self.density_rms.shape != self.density_data.shape:
                del self.density_rms
                self.density_rms = torch.zeros_like(self.density_data.data)
            _C.rmsprop_step(self.density_data.data, self.density_rms, self.density_data.grad, indexer, beta, lr, epsilon, -1000000000.0, lr)
        elif optim == 'sgd':
            _C.sgd_step(self.density_data.data, self.density_data.grad, indexer, lr, lr)
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')

    def optim_sh_step(self, lr: 'float', beta: 'float'=0.9, epsilon: 'float'=1e-08, optim: 'str'='rmsprop'):
        """
        Execute RMSprop/SGD step on SH
        """
        indexer = self._maybe_convert_sparse_grad_indexer(sh=True)
        if optim == 'rmsprop':
            if self.sh_rms is None or self.sh_rms.shape != self.sh_data.shape:
                del self.sh_rms
                self.sh_rms = torch.zeros_like(self.sh_data.data)
            _C.rmsprop_step(self.sh_data.data, self.sh_rms, self.sh_data.grad, indexer, beta, lr, epsilon, -1000000000.0, lr)
        elif optim == 'sgd':
            _C.sgd_step(self.sh_data.data, self.sh_data.grad, indexer, lr, lr)
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')

    def optim_background_step(self, lr_sigma: 'float', lr_color: 'float', beta: 'float'=0.9, epsilon: 'float'=1e-08, optim: 'str'='rmsprop'):
        """
        Execute RMSprop or sgd step on density
        """
        indexer = self._maybe_convert_sparse_grad_indexer(bg=True)
        n_chnl = self.background_data.size(-1)
        if optim == 'rmsprop':
            if self.background_rms is None or self.background_rms.shape != self.background_data.shape:
                del self.background_rms
                self.background_rms = torch.zeros_like(self.background_data.data)
            _C.rmsprop_step(self.background_data.data.view(-1, n_chnl), self.background_rms.view(-1, n_chnl), self.background_data.grad.view(-1, n_chnl), indexer, beta, lr_color, epsilon, -1000000000.0, lr_sigma)
        elif optim == 'sgd':
            _C.sgd_step(self.background_data.data.view(-1, n_chnl), self.background_data.grad.view(-1, n_chnl), indexer, lr_color, lr_sigma)
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')

    def optim_basis_step(self, lr: 'float', beta: 'float'=0.9, epsilon: 'float'=1e-08, optim: 'str'='rmsprop'):
        """
        Execute RMSprop/SGD step on SH
        """
        if optim == 'rmsprop':
            if self.basis_rms is None or self.basis_rms.shape != self.basis_data.shape:
                del self.basis_rms
                self.basis_rms = torch.zeros_like(self.basis_data.data)
            self.basis_rms.mul_(beta).addcmul_(self.basis_data.grad, self.basis_data.grad, value=1.0 - beta)
            denom = self.basis_rms.sqrt().add_(epsilon)
            self.basis_data.data.addcdiv_(self.basis_data.grad, denom, value=-lr)
        elif optim == 'sgd':
            self.basis_data.grad.mul_(lr)
            self.basis_data.data -= self.basis_data.grad
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')
        self.basis_data.grad.zero_()

    @property
    def basis_type_name(self):
        if self.basis_type == BASIS_TYPE_SH:
            return 'SH'
        elif self.basis_type == BASIS_TYPE_3D_TEXTURE:
            return '3D_TEXTURE'
        elif self.basis_type == BASIS_TYPE_MLP:
            return 'MLP'
        return 'UNKNOWN'

    def __repr__(self):
        return f'svox2.SparseGrid(basis_type={self.basis_type_name}, ' + f'basis_dim={self.basis_dim}, ' + f'reso={list(self.links.shape)}, ' + f'capacity:{self.sh_data.size(0)})'

    def is_cubic_pow2(self):
        """
        Check if the current grid is cubic (same in all dims) with power-of-2 size.
        This allows for conversion to svox 1 and Z-order curve (Morton code)
        """
        reso = self.links.shape
        return reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])

    def _to_cpp(self, grid_coords: 'bool'=False, replace_basis_data: 'Optional[torch.Tensor]'=None):
        """
        Generate object to pass to C++
        """
        gspec = _C.SparseGridSpec()
        gspec.density_data = self.density_data
        gspec.sh_data = self.sh_data
        gspec.links = self.links
        if grid_coords:
            gspec._offset = torch.zeros_like(self._offset)
            gspec._scaling = torch.ones_like(self._offset)
        else:
            gsz = self._grid_size()
            gspec._offset = self._offset * gsz - 0.5
            gspec._scaling = self._scaling * gsz
        gspec.basis_dim = self.basis_dim
        gspec.basis_type = self.basis_type
        if replace_basis_data:
            gspec.basis_data = replace_basis_data
        elif self.basis_type == BASIS_TYPE_3D_TEXTURE:
            gspec.basis_data = self.basis_data
        if self.use_background:
            gspec.background_links = self.background_links
            gspec.background_data = self.background_data
        return gspec

    def _grid_size(self):
        return torch.tensor(self.links.shape, device='cpu', dtype=torch.float32)

    def _get_data_grads(self):
        ret = []
        for subitem in ['density_data', 'sh_data', 'basis_data', 'background_data']:
            param = self.__getattr__(subitem)
            if not param.requires_grad:
                ret.append(torch.zeros_like(param.data))
            else:
                if not hasattr(param, 'grad') or param.grad is None or param.grad.shape != param.data.shape:
                    if hasattr(param, 'grad'):
                        del param.grad
                    param.grad = torch.zeros_like(param.data)
                ret.append(param.grad)
        return ret

    def _get_sparse_grad_indexer(self):
        indexer = self.sparse_grad_indexer
        if indexer is None:
            indexer = torch.empty((0,), dtype=torch.bool, device=self.density_data.device)
        return indexer

    def _get_sparse_sh_grad_indexer(self):
        indexer = self.sparse_sh_grad_indexer
        if indexer is None:
            indexer = torch.empty((0,), dtype=torch.bool, device=self.density_data.device)
        return indexer

    def _get_sparse_background_grad_indexer(self):
        indexer = self.sparse_background_indexer
        if indexer is None:
            indexer = torch.empty((0, 0, 0, 0), dtype=torch.bool, device=self.density_data.device)
        return indexer

    def _maybe_convert_sparse_grad_indexer(self, sh=False, bg=False):
        """
        Automatically convert sparse grad indexer from mask to
        indices, if it is efficient
        """
        indexer = self.sparse_sh_grad_indexer if sh else self.sparse_grad_indexer
        if bg:
            indexer = self.sparse_background_indexer
            if indexer is not None:
                indexer = indexer.view(-1)
        if indexer is None:
            return torch.empty((), device=self.density_data.device)
        return indexer

    def _get_rand_cells(self, sparse_frac: 'float', force: 'bool'=False, contiguous: 'bool'=True):
        if sparse_frac < 1.0 or force:
            assert self.sparse_grad_indexer is None or self.sparse_grad_indexer.dtype == torch.bool, 'please call sparse loss after rendering and before gradient updates'
            grid_size = self.links.size(0) * self.links.size(1) * self.links.size(2)
            sparse_num = max(int(sparse_frac * grid_size), 1)
            if contiguous:
                start = np.random.randint(0, grid_size)
                arr = torch.arange(start, start + sparse_num, dtype=torch.int32, device=self.links.device)
                if start > grid_size - sparse_num:
                    arr[grid_size - sparse_num - start:] -= grid_size
                return arr
            else:
                return torch.randint(0, grid_size, (sparse_num,), dtype=torch.int32, device=self.links.device)
        return None

    def _get_rand_cells_background(self, sparse_frac: 'float', contiguous: 'bool'=True):
        assert self.use_background, 'Can only use sparse background loss if using background'
        assert self.sparse_background_indexer is None or self.sparse_background_indexer.dtype == torch.bool, 'please call sparse loss after rendering and before gradient updates'
        grid_size = self.background_links.size(0) * self.background_links.size(1) * self.background_data.size(1)
        sparse_num = max(int(sparse_frac * grid_size), 1)
        if contiguous:
            start = np.random.randint(0, grid_size)
            arr = torch.arange(start, start + sparse_num, dtype=torch.int32, device=self.links.device)
            if start > grid_size - sparse_num:
                arr[grid_size - sparse_num - start:] -= grid_size
            return arr
        else:
            return torch.randint(0, grid_size, (sparse_num,), dtype=torch.int32, device=self.links.device)

    def _eval_learned_bases(self, dirs: 'torch.Tensor'):
        basis_data = self.basis_data.permute([3, 2, 1, 0])[None]
        samples = F.grid_sample(basis_data, dirs[None, None, None], mode='bilinear', padding_mode='zeros', align_corners=True)
        samples = samples[0, :, 0, 0, :].permute([1, 0])
        return samples

    def _eval_basis_mlp(self, dirs: 'torch.Tensor'):
        if self.mlp_posenc_size > 0:
            dirs_enc = utils.posenc(dirs, None, 0, self.mlp_posenc_size, include_identity=True, enable_ipe=False)
        else:
            dirs_enc = dirs
        return self.basis_mlp(dirs_enc)

    def reinit_learned_bases(self, init_type: 'str'='sh', sg_lambda_max: 'float'=1.0, upper_hemi: 'bool'=False):
        """
        Initialize learned basis using either SH orrandom spherical Gaussians
        with concentration parameter sg_lambda (max magnitude) and
        normalization constant sg_sigma

        Spherical Gaussians formula for reference:
        :math:`Output = \\sigma_{i}{exp ^ {\\lambda_i * (\\dot(\\mu_i, \\dirs) - 1)}`

        :param upper_hemi: bool, (SG only) whether to only place Gaussians in z <= 0 (note directions are flipped)
        """
        init_type = init_type.lower()
        n_comps = self.basis_data.size(-1)
        basis_reso = self.basis_data.size(0)
        ax = torch.linspace(-1.0, 1.0, basis_reso, dtype=torch.float32)
        X, Y, Z = torch.meshgrid(ax, ax, ax)
        points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
        points /= points.norm(dim=-1).unsqueeze(-1)
        if init_type == 'sh':
            assert utils.isqrt(n_comps) is not None, 'n_comps (learned basis SH init) must be a square number; maybe try SG init'
            sph_vals = utils.eval_sh_bases(n_comps, points)
        elif init_type == 'sg':
            u1 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u1 /= n_comps
            u1 = u1[torch.randperm(n_comps)]
            u2 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u2 /= n_comps
            sg_dirvecs = utils.spher2cart(u1 * np.pi, u2 * np.pi * 2)
            if upper_hemi:
                sg_dirvecs[..., 2] = -torch.abs(sg_dirvecs[..., 2])
            sg_lambdas = torch.rand_like(sg_dirvecs[:, 0]) * sg_lambda_max
            sg_lambdas[0] = 0.0
            sg_sigmas: 'np.ndarray' = np.sqrt(sg_lambdas / (np.pi * (1.0 - np.exp(-4 * sg_lambdas))))
            sg_sigmas[sg_lambdas == 0.0] = 1.0 / np.sqrt(4 * np.pi)
            sph_vals = utils.eval_sg_at_dirs(sg_lambdas, sg_dirvecs, points) * sg_sigmas
        elif init_type == 'fourier':
            u1 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u1 /= n_comps
            u1 = u1[torch.randperm(n_comps)]
            u2 = torch.arange(0, n_comps) + torch.rand((n_comps,))
            u2 /= n_comps
            fourier_dirvecs = utils.spher2cart(u1 * np.pi, u2 * np.pi * 2)
            fourier_freqs = torch.linspace(0.0, 1.0, n_comps + 1)[:-1]
            fourier_freqs += torch.rand_like(fourier_freqs) * (fourier_freqs[1] - fourier_freqs[0])
            fourier_freqs = torch.exp(fourier_freqs)
            fourier_freqs = fourier_freqs[torch.randperm(n_comps)]
            fourier_scale = 1.0 / torch.sqrt(2 * np.pi - torch.cos(fourier_freqs) * torch.sin(fourier_freqs) / fourier_freqs)
            four_phases = torch.rand_like(fourier_freqs) * np.pi * 2
            dots = (points[:, None] * fourier_dirvecs[None]).sum(-1)
            dots *= fourier_freqs
            sins = torch.sin(dots + four_phases)
            sph_vals = sins * fourier_scale
        else:
            raise NotImplementedError('Unsupported initialization', init_type)
        self.basis_data.data[:] = sph_vals.view(basis_reso, basis_reso, basis_reso, n_comps)

