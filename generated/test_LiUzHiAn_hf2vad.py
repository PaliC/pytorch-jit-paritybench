import sys
_module = sys.modules[__name__]
del sys
datasets = _module
dataset = _module
eval = _module
finetune = _module
losses = _module
loss = _module
ml_memAE_sc_eval = _module
ml_memAE_sc_train = _module
models = _module
basic_modules = _module
mem_cvae = _module
ml_memAE_sc = _module
vunet = _module
pre_process = _module
latest_version_cascade_rcnn_r101_fpn_1x = _module
extract_bboxes = _module
extract_flows = _module
extract_samples = _module
FlowNetC = _module
FlowNetFusion = _module
FlowNetS = _module
FlowNetSD = _module
flownet_networks = _module
channelnorm_package = _module
channelnorm = _module
setup = _module
correlation_package = _module
correlation = _module
setup = _module
flownet2_models = _module
resample2d_package = _module
resample2d = _module
setup = _module
submodules = _module
mmdet_utils = _module
train = _module
utils = _module
cfg_utils = _module
eval_utils = _module
flow_utils = _module
initialization_utils = _module
model_utils = _module
vis_utils = _module

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


import numpy as np


from collections import OrderedDict


import scipy.io as sio


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torchvision.transforms as transforms


import torch.nn as nn


from torch import optim


from torch import nn


from torch.nn.utils import weight_norm


from torch.nn.parameter import Parameter


from torch import norm_except_dim


import torch.nn.functional as F


import math


from torch.nn import ModuleDict


from torch.nn import ModuleList


from torch.nn import Conv2d


from torch.nn import init


from torch.autograd import Function


from torch.autograd import Variable


from torch.nn.modules.module import Module


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import warnings


class Intensity_Loss(nn.Module):

    def __init__(self, l_num):
        super(Intensity_Loss, self).__init__()
        self.l_num = l_num

    def forward(self, gen_frames, gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** self.l_num))


class Gradient_Loss(nn.Module):

    def __init__(self, alpha, channels, device):
        super(Gradient_Loss, self).__init__()
        self.alpha = alpha
        self.device = device
        filter = torch.FloatTensor([[-1.0, 1.0]])
        self.filter_x = filter.view(1, 1, 1, 2).repeat(1, channels, 1, 1)
        self.filter_y = filter.view(1, 1, 2, 1).repeat(1, channels, 1, 1)

    def forward(self, gen_frames, gt_frames):
        gen_frames_x = nn.functional.pad(gen_frames, (1, 0, 0, 0))
        gen_frames_y = nn.functional.pad(gen_frames, (0, 0, 1, 0))
        gt_frames_x = nn.functional.pad(gt_frames, (1, 0, 0, 0))
        gt_frames_y = nn.functional.pad(gt_frames, (0, 0, 1, 0))
        gen_dx = nn.functional.conv2d(gen_frames_x, self.filter_x)
        gen_dy = nn.functional.conv2d(gen_frames_y, self.filter_y)
        gt_dx = nn.functional.conv2d(gt_frames_x, self.filter_x)
        gt_dy = nn.functional.conv2d(gt_frames_y, self.filter_y)
        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)
        return torch.mean(grad_diff_x ** self.alpha + grad_diff_y ** self.alpha)


class Entropy_Loss(nn.Module):

    def __init__(self):
        super(Entropy_Loss, self).__init__()

    def forward(self, x):
        eps = 1e-20
        tmp = torch.sum(-x * torch.log(x + eps), dim=-1)
        return torch.mean(tmp)


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    """
    inconv only changes the number of channels
    """

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1), double_conv(out_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=False, op='none'):
        super(up, self).__init__()
        self.bilinear = bilinear
        self.op = op
        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(in_ch, in_ch // 2, 1))
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        assert op in ['concat', 'none']
        if op == 'concat':
            self.conv = double_conv(in_ch, out_ch)
        else:
            self.conv = double_conv(out_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if self.op == 'concat':
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.bs, self.bs, w // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * self.bs ** 2, h // self.bs, w // self.bs)
        return x


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // self.bs ** 2, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(n, c // self.bs ** 2, h * self.bs, w * self.bs)
        return x


class IDAct(nn.Module):

    def forward(self, input):
        return input


class NormConv2d(nn.Module):
    """
    Convolutional layer with l2 weight normalization and learned scaling parameters
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros([1, out_channels, 1, 1], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.ones([1, out_channels, 1, 1], dtype=torch.float32))
        self.conv = weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), name='weight')

    def forward(self, x):
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out


class Downsample(nn.Module):

    def __init__(self, channels, out_channels=None, conv_layer=NormConv2d):
        super().__init__()
        if out_channels == None:
            self.down = conv_layer(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.down = conv_layer(channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, subpixel=True, conv_layer=NormConv2d):
        super().__init__()
        if subpixel:
            self.up = conv_layer(in_channels, 4 * out_channels, 3, padding=1)
            self.op2 = DepthToSpace(block_size=2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.op2 = IDAct()

    def forward(self, x):
        out = self.up(x)
        out = self.op2(out)
        return out


class VUnetResnetBlock(nn.Module):
    """
    Resnet Block as utilized in the vunet publication
    """

    def __init__(self, out_channels, use_skip=False, kernel_size=3, activate=True, conv_layer=NormConv2d, gated=False, final_act=False, dropout_prob=0.0):
        """

        :param n_channels: The number of output filters
        :param process_skip: the factor between output and input nr of filters
        :param kernel_size:
        :param activate:
        """
        super().__init__()
        self.dout = nn.Dropout(p=dropout_prob)
        self.use_skip = use_skip
        self.gated = gated
        if self.use_skip:
            self.conv2d = conv_layer(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            self.pre = conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.conv2d = conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        if self.gated:
            self.conv2d2 = conv_layer(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            self.dout2 = nn.Dropout(p=dropout_prob)
            self.sigm = nn.Sigmoid()
        if activate:
            self.act_fn = nn.LeakyReLU() if final_act else nn.ELU()
        else:
            self.act_fn = IDAct()

    def forward(self, x, a=None):
        x_prc = x
        if self.use_skip:
            assert a is not None
            a = self.act_fn(a)
            a = self.pre(a)
            x_prc = torch.cat([x_prc, a], dim=1)
        x_prc = self.act_fn(x_prc)
        x_prc = self.dout(x_prc)
        x_prc = self.conv2d(x_prc)
        if self.gated:
            x_prc = self.act_fn(x_prc)
            x_prc = self.dout(x_prc)
            x_prc = self.conv2d2(x_prc)
            a, b = torch.split(x_prc, 2, 1)
            x_prc = a * self.sigm(b)
        return x + x_prc


def hard_shrink_relu(input, lambd=0.0, epsilon=1e-12):
    output = F.relu(input - lambd) * input / (torch.abs(input - lambd) + epsilon)
    return output


class Memory(nn.Module):

    def __init__(self, num_slots, slot_dim, shrink_thres=0.0025):
        super(Memory, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.memMatrix = nn.Parameter(torch.empty(num_slots, slot_dim))
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        att_weight = F.linear(input=x, weight=self.memMatrix)
        att_weight = F.softmax(att_weight, dim=1)
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        out = F.linear(att_weight, self.memMatrix.permute(1, 0))
        return dict(out=out, att_weight=att_weight)


class ML_MemAE_SC(nn.Module):

    def __init__(self, num_in_ch, seq_len, features_root, num_slots, shrink_thres, mem_usage, skip_ops):
        super(ML_MemAE_SC, self).__init__()
        self.num_in_ch = num_in_ch
        self.seq_len = seq_len
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres
        self.mem_usage = mem_usage
        self.num_mem = sum(mem_usage)
        self.skip_ops = skip_ops
        self.in_conv = inconv(num_in_ch * seq_len, features_root)
        self.down_1 = down(features_root, features_root * 2)
        self.down_2 = down(features_root * 2, features_root * 4)
        self.down_3 = down(features_root * 4, features_root * 8)
        self.mem1 = Memory(num_slots=self.num_slots, slot_dim=features_root * 2 * 16 * 16, shrink_thres=self.shrink_thres) if self.mem_usage[1] else None
        self.mem2 = Memory(num_slots=self.num_slots, slot_dim=features_root * 4 * 8 * 8, shrink_thres=self.shrink_thres) if self.mem_usage[2] else None
        self.mem3 = Memory(num_slots=self.num_slots, slot_dim=features_root * 8 * 4 * 4, shrink_thres=self.shrink_thres) if self.mem_usage[3] else None
        self.up_3 = up(features_root * 8, features_root * 4, op=self.skip_ops[-1])
        self.up_2 = up(features_root * 4, features_root * 2, op=self.skip_ops[-2])
        self.up_1 = up(features_root * 2, features_root, op=self.skip_ops[-3])
        self.out_conv = outconv(features_root, num_in_ch * seq_len)

    def forward(self, x):
        """
        :param x: size [bs,C*seq_len,H,W]
        :return:
        """
        x0 = self.in_conv(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        if self.mem_usage[3]:
            bs, C, H, W = x3.shape
            x3 = x3.view(bs, -1)
            mem3_out = self.mem3(x3)
            x3 = mem3_out['out']
            att_weight3 = mem3_out['att_weight']
            x3 = x3.view(bs, C, H, W)
        recon = self.up_3(x3, x2 if self.skip_ops[-1] != 'none' else None)
        if self.mem_usage[2]:
            bs, C, H, W = recon.shape
            recon = recon.view(bs, -1)
            mem2_out = self.mem2(recon)
            recon = mem2_out['out']
            att_weight2 = mem2_out['att_weight']
            recon = recon.view(bs, C, H, W)
        recon = self.up_2(recon, x1 if self.skip_ops[-2] != 'none' else None)
        if self.mem_usage[1]:
            bs, C, H, W = recon.shape
            recon = recon.view(bs, -1)
            mem1_out = self.mem1(recon)
            recon = mem1_out['out']
            att_weight1 = mem1_out['att_weight']
            recon = recon.view(bs, C, H, W)
        recon = self.up_1(recon, x0 if self.skip_ops[-3] != 'none' else None)
        recon = self.out_conv(recon)
        if self.num_mem == 3:
            outs = dict(recon=recon, att_weight3=att_weight3, att_weight2=att_weight2, att_weight1=att_weight1)
        elif self.num_mem == 2:
            outs = dict(recon=recon, att_weight3=att_weight3, att_weight2=att_weight2, att_weight1=torch.zeros_like(att_weight3))
        elif self.num_mem == 1:
            outs = dict(recon=recon, att_weight3=att_weight3, att_weight2=torch.zeros_like(att_weight3), att_weight1=torch.zeros_like(att_weight3))
        return outs


class VUnetBottleneck(nn.Module):

    def __init__(self, n_stages, nf, device, n_rnb=2, n_auto_groups=4, conv_layer=NormConv2d, dropout_prob=0.0):
        super().__init__()
        self.device = device
        self.blocks = ModuleDict()
        self.channel_norm = ModuleDict()
        self.conv1x1 = conv_layer(nf, nf, 1)
        self.up = Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer)
        self.depth_to_space = DepthToSpace(block_size=2)
        self.space_to_depth = SpaceToDepth(block_size=2)
        self.n_stages = n_stages
        self.n_rnb = n_rnb
        self.n_auto_groups = n_auto_groups
        for i_s in range(self.n_stages, self.n_stages - 2, -1):
            self.channel_norm.update({f's{i_s}': conv_layer(2 * nf, nf, 1)})
            for ir in range(self.n_rnb):
                self.blocks.update({f's{i_s}_{ir + 1}': VUnetResnetBlock(nf, use_skip=True, conv_layer=conv_layer, dropout_prob=dropout_prob)})
        self.auto_blocks = ModuleList()
        for i_a in range(4):
            if i_a < 1:
                self.auto_blocks.append(VUnetResnetBlock(nf, conv_layer=conv_layer, dropout_prob=dropout_prob))
                self.param_converter = conv_layer(4 * nf, nf, kernel_size=1)
            else:
                self.auto_blocks.append(VUnetResnetBlock(nf, use_skip=True, conv_layer=conv_layer, dropout_prob=dropout_prob))

    def forward(self, x_e, z_post):
        p_params = {}
        z_prior = {}
        use_z = True
        h = self.conv1x1(x_e[f's{self.n_stages}_2'])
        for i_s in range(self.n_stages, self.n_stages - 2, -1):
            stage = f's{i_s}'
            spatial_size = x_e[stage + '_2'].shape[-1]
            spatial_stage = '%dby%d' % (spatial_size, spatial_size)
            h = self.blocks[stage + '_2'](h, x_e[stage + '_2'])
            if spatial_size == 1:
                p_params[spatial_stage] = h
                prior_samples = self._latent_sample(p_params[spatial_stage])
                z_prior[spatial_stage] = prior_samples
            else:
                if use_z:
                    z_flat = self.space_to_depth(z_post[spatial_stage]) if z_post[spatial_stage].shape[2] > 1 else z_post[spatial_stage]
                    sec_size = z_flat.shape[1] // 4
                    z_groups = torch.split(z_flat, [sec_size, sec_size, sec_size, sec_size], dim=1)
                param_groups = []
                sample_groups = []
                param_features = self.auto_blocks[0](h)
                param_features = self.space_to_depth(param_features)
                param_features = self.param_converter(param_features)
                for i_a in range(len(self.auto_blocks)):
                    param_groups.append(param_features)
                    prior_samples = self._latent_sample(param_groups[-1])
                    sample_groups.append(prior_samples)
                    if i_a + 1 < len(self.auto_blocks):
                        if use_z:
                            feedback = z_groups[i_a]
                        else:
                            feedback = prior_samples
                        param_features = self.auto_blocks[i_a + 1](param_features, feedback)
                p_params_stage = self.__merge_groups(param_groups)
                prior_samples = self.__merge_groups(sample_groups)
                p_params[spatial_stage] = p_params_stage
                z_prior[spatial_stage] = prior_samples
            if use_z:
                z = self.depth_to_space(z_post[spatial_stage]) if z_post[spatial_stage].shape[-1] != h.shape[-1] else z_post[spatial_stage]
            else:
                z = prior_samples
            gz = torch.cat([x_e[stage + '_1'], z], dim=1)
            gz = self.channel_norm[stage](gz)
            h = self.blocks[stage + '_1'](h, gz)
            if i_s == self.n_stages:
                h = self.up(h)
        return h, p_params, z_prior

    def __split_groups(self, x):
        sec_size = x.shape[1] // 4
        return torch.split(self.space_to_depth(x), [sec_size, sec_size, sec_size, sec_size], dim=1)

    def __merge_groups(self, x):
        return self.depth_to_space(torch.cat(x, dim=1))

    def _latent_sample(self, mean):
        normal_sample = torch.randn(mean.size())
        return mean + normal_sample


class VUnetDecoder(nn.Module):

    def __init__(self, n_stages, nf=128, nf_out=3, n_rnb=2, conv_layer=NormConv2d, spatial_size=256, final_act=True, dropout_prob=0.0):
        super().__init__()
        self.final_act = final_act
        self.blocks = ModuleDict()
        self.ups = ModuleDict()
        self.n_stages = n_stages
        self.n_rnb = n_rnb
        for i_s in range(self.n_stages - 2, 0, -1):
            if i_s == 1:
                self.ups.update({f's{i_s + 1}': Upsample(in_channels=nf, out_channels=nf // 2, conv_layer=conv_layer)})
                nf = nf // 2
            else:
                self.ups.update({f's{i_s + 1}': Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer)})
            for ir in range(self.n_rnb, 0, -1):
                stage = f's{i_s}_{ir}'
                self.blocks.update({stage: VUnetResnetBlock(nf, use_skip=True, conv_layer=conv_layer, dropout_prob=dropout_prob)})
        self.final_layer = conv_layer(nf, nf_out, kernel_size=1)
        self.final_act = nn.Sigmoid()

    def forward(self, x, skips):
        out = x
        for i_s in range(self.n_stages - 2, 0, -1):
            out = self.ups[f's{i_s + 1}'](out)
            for ir in range(self.n_rnb, 0, -1):
                stage = f's{i_s}_{ir}'
                out = self.blocks[stage](out, skips[stage])
        out = self.final_layer(out)
        if self.final_act:
            out = self.final_act(out)
        return out


class VUnetEncoder(nn.Module):

    def __init__(self, n_stages, nf_in=3, nf_start=64, nf_max=128, n_rnb=2, conv_layer=NormConv2d, dropout_prob=0.0):
        super().__init__()
        self.in_op = conv_layer(nf_in, nf_start, kernel_size=1)
        nf = nf_start
        self.blocks = ModuleDict()
        self.downs = ModuleDict()
        self.n_rnb = n_rnb
        self.n_stages = n_stages
        for i_s in range(self.n_stages):
            if i_s > 0:
                self.downs.update({f's{i_s + 1}': Downsample(nf, min(2 * nf, nf_max), conv_layer=conv_layer)})
                nf = min(2 * nf, nf_max)
            for ir in range(self.n_rnb):
                stage = f's{i_s + 1}_{ir + 1}'
                self.blocks.update({stage: VUnetResnetBlock(nf, conv_layer=conv_layer, dropout_prob=dropout_prob)})

    def forward(self, x):
        out = {}
        h = self.in_op(x)
        for ir in range(self.n_rnb):
            h = self.blocks[f's1_{ir + 1}'](h)
            out[f's1_{ir + 1}'] = h
        for i_s in range(1, self.n_stages):
            h = self.downs[f's{i_s + 1}'](h)
            for ir in range(self.n_rnb):
                stage = f's{i_s + 1}_{ir + 1}'
                h = self.blocks[stage](h)
                out[stage] = h
        return out


class ZConverter(nn.Module):

    def __init__(self, n_stages, nf, device, conv_layer=NormConv2d, dropout_prob=0.0):
        super().__init__()
        self.n_stages = n_stages
        self.device = device
        self.blocks = ModuleList()
        for i in range(3):
            self.blocks.append(VUnetResnetBlock(nf, use_skip=True, conv_layer=conv_layer, dropout_prob=dropout_prob))
        self.conv1x1 = conv_layer(nf, nf, 1)
        self.up = Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer)
        self.channel_norm = conv_layer(2 * nf, nf, 1)
        self.d2s = DepthToSpace(block_size=2)
        self.s2d = SpaceToDepth(block_size=2)

    def forward(self, x_f):
        params = {}
        zs = {}
        h = self.conv1x1(x_f[f's{self.n_stages}_2'])
        for n, i_s in enumerate(range(self.n_stages, self.n_stages - 2, -1)):
            stage = f's{i_s}'
            spatial_size = x_f[stage + '_2'].shape[-1]
            spatial_stage = '%dby%d' % (spatial_size, spatial_size)
            h = self.blocks[2 * n](h, x_f[stage + '_2'])
            params[spatial_stage] = h
            z = self._latent_sample(params[spatial_stage])
            zs[spatial_stage] = z
            if n == 0:
                gz = torch.cat([x_f[stage + '_1'], z], dim=1)
                gz = self.channel_norm(gz)
                h = self.blocks[2 * n + 1](h, gz)
                h = self.up(h)
        return params, zs

    def _latent_sample(self, mean):
        normal_sample = torch.randn(mean.size())
        return mean + normal_sample


class VUnet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        final_act = retrieve(config, 'model_paras/final_act', default=False)
        nf_max = retrieve(config, 'model_paras/nf_max', default=128)
        nf_start = retrieve(config, 'model_paras/nf_start', default=64)
        spatial_size = retrieve(config, 'model_paras/spatial_size', default=256)
        dropout_prob = retrieve(config, 'model_paras/dropout_prob', default=0.1)
        img_channels = retrieve(config, 'model_paras/img_channels', default=3)
        motion_channels = retrieve(config, 'model_paras/motion_channels', default=2)
        clip_hist = retrieve(config, 'model_paras/clip_hist', default=4)
        clip_pred = retrieve(config, 'model_paras/clip_pred', default=1)
        num_flows = retrieve(config, 'model_paras/num_flows', default=4)
        device = retrieve(config, 'device', default='cuda:0')
        output_channels = img_channels * clip_pred
        n_stages = 1 + int(np.round(np.log2(spatial_size))) - 2
        conv_layer_type = Conv2d if final_act else NormConv2d
        self.f_phi = VUnetEncoder(n_stages=n_stages, nf_in=img_channels * clip_hist + motion_channels * num_flows, nf_start=nf_start, nf_max=nf_max, conv_layer=conv_layer_type, dropout_prob=dropout_prob)
        self.e_theta = VUnetEncoder(n_stages=n_stages, nf_in=motion_channels * num_flows, nf_start=nf_start, nf_max=nf_max, conv_layer=conv_layer_type, dropout_prob=dropout_prob)
        self.zc = ZConverter(n_stages=n_stages, nf=nf_max, device=device, conv_layer=conv_layer_type, dropout_prob=dropout_prob)
        self.bottleneck = VUnetBottleneck(n_stages=n_stages, nf=nf_max, device=device, conv_layer=conv_layer_type, dropout_prob=dropout_prob)
        self.decoder = VUnetDecoder(n_stages=n_stages, nf=nf_max, nf_out=output_channels, conv_layer=conv_layer_type, spatial_size=spatial_size, final_act=final_act, dropout_prob=dropout_prob)
        self.saved_tensors = None

    def forward(self, inputs, mode='train'):
        """
        Two possible usage：

        1. train stage, sampling z from the posterior p(z | x_{1:t},y_{1:t} )
        2. test stage, use the mean of the posterior as sampled z
        """
        x_f_in = torch.cat((inputs['appearance'], inputs['motion']), dim=1)
        x_f = self.f_phi(x_f_in)
        q_means, zs = self.zc(x_f)
        x_e = self.e_theta(inputs['motion'])
        if mode == 'train':
            out_b, p_means, ps = self.bottleneck(x_e, zs)
        else:
            out_b, p_means, ps = self.bottleneck(x_e, q_means)
        out_img = self.decoder(out_b, x_f)
        self.saved_tensors = dict(q_means=q_means, p_means=p_means)
        return out_img


class HFVAD(nn.Module):
    """
    ML-MemAE-SC + CVAE
    """

    def __init__(self, num_hist, num_pred, config, features_root, num_slots, shrink_thres, skip_ops, mem_usage, finetune=False):
        super(HFVAD, self).__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.features_root = features_root
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres
        self.skip_ops = skip_ops
        self.mem_usage = mem_usage
        self.finetune = finetune
        self.x_ch = 3
        self.y_ch = 2
        self.memAE = ML_MemAE_SC(num_in_ch=self.y_ch, seq_len=1, features_root=self.features_root, num_slots=self.num_slots, shrink_thres=self.shrink_thres, mem_usage=self.mem_usage, skip_ops=self.skip_ops)
        self.vunet = VUnet(config)
        self.mse_loss = nn.MSELoss()

    def forward(self, sample_frame, sample_of, mode='train'):
        """
        :param sample_frame: 5 frames in a video clip
        :param sample_of: 4 corresponding flows
        :return:
        """
        att_weight3_cache, att_weight2_cache, att_weight1_cache = [], [], []
        of_recon = torch.zeros_like(sample_of)
        for j in range(self.num_hist):
            memAE_out = self.memAE(sample_of[:, 2 * j:2 * (j + 1), :, :])
            of_recon[:, 2 * j:2 * (j + 1), :, :] = memAE_out['recon']
            att_weight3_cache.append(memAE_out['att_weight3'])
            att_weight2_cache.append(memAE_out['att_weight2'])
            att_weight1_cache.append(memAE_out['att_weight1'])
        att_weight3 = torch.cat(att_weight3_cache, dim=0)
        att_weight2 = torch.cat(att_weight2_cache, dim=0)
        att_weight1 = torch.cat(att_weight1_cache, dim=0)
        if self.finetune:
            loss_recon = self.mse_loss(of_recon, sample_of)
            loss_sparsity = torch.mean(torch.sum(-att_weight3 * torch.log(att_weight3 + 1e-12), dim=1)) + torch.mean(torch.sum(-att_weight2 * torch.log(att_weight2 + 1e-12), dim=1)) + torch.mean(torch.sum(-att_weight1 * torch.log(att_weight1 + 1e-12), dim=1))
        frame_in = sample_frame[:, :-self.x_ch * self.num_pred, :, :]
        frame_target = sample_frame[:, -self.x_ch * self.num_pred:, :, :]
        input_dict = dict(appearance=frame_in, motion=of_recon)
        frame_pred = self.vunet(input_dict, mode=mode)
        out = dict(frame_pred=frame_pred, frame_target=frame_target, of_recon=of_recon, of_target=sample_of)
        out.update(self.vunet.saved_tensors)
        if self.finetune:
            ML_MemAE_SC_dict = dict(loss_recon=loss_recon, loss_sparsity=loss_sparsity)
            out.update(ML_MemAE_SC_dict)
        return out


class CorrelationFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()
            correlation_cuda.forward(input1, input2, rbot1, rbot2, output, ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            grad_input1 = input1.new()
            grad_input2 = input2.new()
            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2, ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)
        return grad_input1, grad_input2, None, None, None, None, None, None


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
        result = CorrelationFunction.apply(input1, input2, self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)
        return result


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True), nn.LeakyReLU(0.1, inplace=True))


def deconv(in_planes, out_planes):
    return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1, inplace=True))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


class tofp16(nn.Module):

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):

    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


class FlowNetC(nn.Module):

    def __init__(self, fp16=False, batchNorm=True, div_flow=20):
        super(FlowNetC, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1)
        if fp16:
            self.corr = nn.Sequential(tofp32(), Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1), tofp16())
        else:
            self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:, :, :]
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)
        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)
        out_corr = self.corr(out_conv3a, out_conv3b)
        out_corr = self.corr_activation(out_corr)
        out_conv_redir = self.conv_redir(out_conv3a)
        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)
        out_conv3_1 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias=True):
    if batchNorm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias), nn.BatchNorm2d(out_planes))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=bias))


class FlowNetFusion(nn.Module):

    def __init__(self, batchNorm=True):
        super(FlowNetFusion, self).__init__()
        self.batchNorm = batchNorm
        self.conv0 = conv(self.batchNorm, 11, 64)
        self.conv1 = conv(self.batchNorm, 64, 64, stride=2)
        self.conv1_1 = conv(self.batchNorm, 64, 128)
        self.conv2 = conv(self.batchNorm, 128, 128, stride=2)
        self.conv2_1 = conv(self.batchNorm, 128, 128)
        self.deconv1 = deconv(128, 32)
        self.deconv0 = deconv(162, 16)
        self.inter_conv1 = i_conv(self.batchNorm, 162, 32)
        self.inter_conv0 = i_conv(self.batchNorm, 82, 16)
        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        flow2 = self.predict_flow2(out_conv2)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1 = self.predict_flow1(out_interconv1)
        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((out_conv0, out_deconv0, flow1_up), 1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0 = self.predict_flow0(out_interconv0)
        return flow0


class FlowNetS(nn.Module):

    def __init__(self, input_channels=12, batchNorm=True):
        super(FlowNetS, self).__init__()
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


class FlowNetSD(nn.Module):

    def __init__(self, batchNorm=True):
        super(FlowNetSD, self).__init__()
        self.batchNorm = batchNorm
        self.conv0 = conv(self.batchNorm, 6, 64)
        self.conv1 = conv(self.batchNorm, 64, 64, stride=2)
        self.conv1_1 = conv(self.batchNorm, 64, 128)
        self.conv2 = conv(self.batchNorm, 128, 128, stride=2)
        self.conv2_1 = conv(self.batchNorm, 128, 128)
        self.conv3 = conv(self.batchNorm, 128, 256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.inter_conv5 = i_conv(self.batchNorm, 1026, 512)
        self.inter_conv4 = i_conv(self.batchNorm, 770, 256)
        self.inter_conv3 = i_conv(self.batchNorm, 386, 128)
        self.inter_conv2 = i_conv(self.batchNorm, 194, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


class ChannelNormFunction(Function):

    @staticmethod
    def forward(ctx, input1, norm_deg=2):
        assert input1.is_contiguous()
        b, _, h, w = input1.size()
        output = input1.new(b, 1, h, w).zero_()
        channelnorm_cuda.forward(input1, output, norm_deg)
        ctx.save_for_backward(input1, output)
        ctx.norm_deg = norm_deg
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, output = ctx.saved_tensors
        grad_input1 = Variable(input1.new(input1.size()).zero_())
        channelnorm_cuda.backward(input1, output, grad_output.data, grad_input1.data, ctx.norm_deg)
        return grad_input1, None


class ChannelNorm(Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        return ChannelNormFunction.apply(input1, self.norm_deg)


class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1, bilinear=True):
        assert input1.is_contiguous()
        assert input2.is_contiguous()
        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        ctx.bilinear = bilinear
        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, d, h, w).zero_()
        resample2d_cuda.forward(input1, input2, output, kernel_size, bilinear)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        assert grad_output.is_contiguous()
        input1, input2 = ctx.saved_tensors
        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())
        resample2d_cuda.backward(input1, input2, grad_output.data, grad_input1.data, grad_input2.data, ctx.kernel_size, ctx.bilinear)
        return grad_input1, grad_input2, None, None


class Resample2d(Module):

    def __init__(self, kernel_size=1, bilinear=True):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.bilinear = bilinear

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size, self.bilinear)


class FlowNet2(nn.Module):

    def __init__(self, rgb_max=255.0, fp16=False, batchNorm=False, div_flow=20.0):
        super(FlowNet2, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = rgb_max
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC.FlowNetC(fp16=fp16, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        if fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()
        self.flownets_1 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if fp16:
            self.resample2 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample2 = Resample2d()
        self.flownets_2 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)
        self.flownets_d = FlowNetSD.FlowNetSD(batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        if fp16:
            self.resample3 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample3 = Resample2d()
        if fp16:
            self.resample4 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample4 = Resample2d()
        self.flownetfusion = FlowNetFusion.FlowNetFusion(batchNorm=self.batchNorm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def init_deconv_bilinear(self, weight):
        f_shape = weight.size()
        heigh, width = f_shape[-2], f_shape[-1]
        f = np.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([heigh, width])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        min_dim = min(f_shape[0], f_shape[1])
        weight.data.fill_(0.0)
        for i in range(min_dim):
            weight.data[i, i, :, :] = torch.from_numpy(bilinear)
        return

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)
        diff_flownets2_flow = self.resample4(x[:, 3:, :, :], flownets2_flow)
        diff_flownets2_img1 = self.channelnorm(x[:, :3, :, :] - diff_flownets2_flow)
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)
        diff_flownetsd_flow = self.resample3(x[:, 3:, :, :], flownetsd_flow)
        diff_flownetsd_img1 = self.channelnorm(x[:, :3, :, :] - diff_flownetsd_flow)
        concat3 = torch.cat((x[:, :3, :, :], flownetsd_flow, flownets2_flow, norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1, diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)
        return flownetfusion_flow


class FlowNet2CS(nn.Module):

    def __init__(self, rgb_max=255.0, fp16=False, batchNorm=False, div_flow=20.0):
        super(FlowNet2CS, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = rgb_max
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC.FlowNetC(fp16=fp16, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        if fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()
        self.flownets_1 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        return flownets1_flow


class FlowNet2CSS(nn.Module):

    def __init__(self, rgb_max=255.0, fp16=False, batchNorm=False, div_flow=20.0):
        super(FlowNet2CSS, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = rgb_max
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC.FlowNetC(fp16=fp16, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        if fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()
        self.flownets_1 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if fp16:
            self.resample2 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample2 = Resample2d()
        self.flownets_2 = FlowNetS.FlowNetS(batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)
        return flownets2_flow


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Downsample,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Entropy_Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FlowNetFusion,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 11, 64, 64])], {}),
     True),
    (FlowNetS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 12, 64, 64])], {}),
     False),
    (FlowNetSD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 6, 64, 64])], {}),
     False),
    (Gradient_Loss,
     lambda: ([], {'alpha': 4, 'channels': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (IDAct,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Intensity_Loss,
     lambda: ([], {'l_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Memory,
     lambda: ([], {'num_slots': 4, 'slot_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NormConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpaceToDepth,
     lambda: ([], {'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Upsample,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VUnetEncoder,
     lambda: ([], {'n_stages': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (VUnetResnetBlock,
     lambda: ([], {'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (double_conv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (down,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (inconv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (outconv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (tofp16,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (tofp32,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_LiUzHiAn_hf2vad(_paritybench_base):
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

