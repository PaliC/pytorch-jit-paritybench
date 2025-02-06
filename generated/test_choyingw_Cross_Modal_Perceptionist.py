import sys
_module = sys.modules[__name__]
del sys
Sim3DR = _module
_init_paths = _module
lighting = _module
setup = _module
mobilenetv2_backbone = _module
cal_size_ARE = _module
cal_size_kpts = _module
config = _module
dataset = _module
demo = _module
demo_mic = _module
distiller_zoo = _module
eval_sup = _module
gan_train_cascade = _module
mfcc = _module
network = _module
parse_dataset = _module
pyaudio_recording = _module
utilf = _module
render = _module
utils = _module
vad = _module

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


from torch import nn


import numpy as np


from torch.utils.data import Dataset


import random


import scipy.io as sio


from scipy.io import wavfile


import torch.nn.functional as F


import torchvision.utils as vutils


import torch.nn as nn


import time


from torch.utils.data import DataLoader


import torch.optim as optim


import logging


from torch.utils.data.dataloader import default_collate


class ConvBNReLU(nn.Sequential):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False), norm_layer(out_planes), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), norm_layer(oup)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None, norm_layer=None, last_CN=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        input_channel = 32
        last_channel = 1280
        if inverted_residual_setting is None:
            inverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty or a 4-element list, got {}'.format(inverted_residual_setting))
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        total = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                total += 1
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        self.features = nn.Sequential(*features)
        self.features_first = self.features[:9]
        self.features_second = self.features[9:]
        if not last_CN:
            self.last_CN = self.last_channel
        else:
            self.last_CN = last_CN
        self.num_ori = 12
        self.num_shape = 40
        self.num_exp = 10
        self.num_texture = 40
        self.num_bin = 121
        self.num_scale = 1
        self.num_trans = 3
        if last_CN is not None:
            self.connector = nn.Sequential(nn.Linear(self.last_CN, self.last_CN // 16), nn.ReLU6(inplace=True), nn.Linear(self.last_CN // 16, self.last_CN), nn.ReLU6(inplace=True), nn.Sigmoid())
            self.adjuster = nn.Sequential(nn.Linear(self.last_CN, self.last_CN), nn.BatchNorm1d(self.last_CN))
        self.classifier_ori = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_CN, self.num_ori))
        self.classifier_shape = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_CN, self.num_shape))
        self.classifier_exp = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_CN, self.num_exp))
        self.classifier_texture = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_CN, self.num_texture))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        inter = self.features_first(x)
        x = self.features_second(inter)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.shape[0], -1)
        pool_x = x.clone()
        x_ori = self.classifier_ori(x)
        x_shape = self.classifier_shape(x)
        x_exp = self.classifier_exp(x)
        x_tex = self.classifier_texture(x)
        x = torch.cat((x_ori, x_shape, x_exp, x_tex), dim=1)
        return x, pool_x, inter

    def forward(self, x):
        return self._forward_impl(x)


class Attention(nn.Module):

    def __init__(self, p=2):
        super(Attention, self).__init__()
        self.p = p

    def forward(self, f_s, f_t):
        if f_s.dim() == 2:
            return (F.normalize(f_s.pow(self.p)) - F.normalize(f_t.pow(self.p))).pow(2).mean()
        else:
            return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))


class Similarity(nn.Module):

    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)
        G_s = torch.mm(f_s, torch.t(f_s))
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        G_t = torch.nn.functional.normalize(G_t)
        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss


class Correlation(nn.Module):

    def __init__(self):
        super(Correlation, self).__init__()

    def forward(self, f_s, f_t):
        delta = torch.abs(f_s - f_t)
        loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
        return loss


class NSTLoss(nn.Module):

    def __init__(self):
        super(NSTLoss, self).__init__()
        pass

    def forward(self, f_s, f_t):
        if f_s.dim() == 4:
            s_H, t_H = f_s.shape[2], f_t.shape[2]
            if s_H > t_H:
                f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
            elif s_H < t_H:
                f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
            else:
                pass
            f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
            f_s = F.normalize(f_s, dim=2)
            f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
            f_t = F.normalize(f_t, dim=2)
        elif f_s.dim() == 2:
            f_s = F.normalize(f_s, dim=1)
            f_t = F.normalize(f_t, dim=1)
        full_loss = True
        if full_loss:
            return self.poly_kernel(f_t, f_t).mean().detach() + self.poly_kernel(f_s, f_s).mean() - 2 * self.poly_kernel(f_s, f_t).mean()
        else:
            return self.poly_kernel(f_s, f_s).mean()

    def poly_kernel(self, a, b):
        a = a.unsqueeze(1)
        b = b.unsqueeze(2)
        res = (a * b).sum(-1).pow(2)
        return res


class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""

    def __init__(self, w_d=25, w_a=50):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        loss_d = F.smooth_l1_loss(d, t_d)
        with torch.no_grad():
            td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        loss_a = F.smooth_l1_loss(s_angle, t_angle)
        loss = self.w_d * loss_d + self.w_a * loss_a
        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
        if not squared:
            res = res.sqrt()
        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res


class PKT(nn.Module):
    """Probabilistic Knowledge Transfer for deep representation learning
    Code from author: https://github.com/passalis/probabilistic_kt"""

    def __init__(self):
        super(PKT, self).__init__()

    def forward(self, f_s, f_t):
        return self.cosine_similarity_loss(f_s, f_t)

    @staticmethod
    def cosine_similarity_loss(output_net, target_net, eps=1e-07):
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0
        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))
        return loss


class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""

    def __init__(self, num_input_channels, num_mid_channel, num_target_channels, init_pred_var=5.0, eps=1e-05):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False, stride=stride)
        self.regressor = nn.Sequential(conv1x1(num_input_channels, num_mid_channel), nn.ReLU(), conv1x1(num_mid_channel, num_mid_channel), nn.ReLU(), conv1x1(num_mid_channel, num_target_channels))
        self.log_scale = torch.nn.Parameter(np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(num_target_channels))
        self.eps = eps

    def forward(self, input, target):
        if input.dim() == 2:
            input = input.unsqueeze(2).unsqueeze(2)
            target = target.unsqueeze(2).unsqueeze(2)
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * ((pred_mean - target) ** 2 / pred_var + torch.log(pred_var))
        loss = torch.mean(neg_log_prob)
        return loss


class VoiceEmbedNet(nn.Module):

    def __init__(self, input_channel, channels, output_channel):
        super(VoiceEmbedNet, self).__init__()
        self.model = nn.Sequential(nn.Conv1d(input_channel, channels[0], 3, 2, 1, bias=False), nn.BatchNorm1d(channels[0], affine=True), nn.ReLU(inplace=True), nn.Conv1d(channels[0], channels[1], 3, 2, 1, bias=False), nn.BatchNorm1d(channels[1], affine=True), nn.ReLU(inplace=True), nn.Conv1d(channels[1], channels[2], 3, 2, 1, bias=False), nn.BatchNorm1d(channels[2], affine=True), nn.ReLU(inplace=True), nn.Conv1d(channels[2], channels[3], 3, 2, 1, bias=False), nn.BatchNorm1d(channels[3], affine=True), nn.ReLU(inplace=True), nn.Conv1d(channels[3], output_channel, 3, 2, 1, bias=True))

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool1d(x, x.size()[2], stride=1)
        x = x.view(x.size()[0], -1, 1, 1)
        return x


class Generator(nn.Module):

    def __init__(self, input_channel, channels, output_channel):
        super(Generator, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(input_channel, channels[0], 4, 1, 0, bias=True), nn.ReLU(inplace=True), nn.ConvTranspose2d(channels[0], channels[1], 4, 2, 1, bias=True), nn.ReLU(inplace=True), nn.ConvTranspose2d(channels[1], channels[2], 4, 2, 1, bias=True), nn.ReLU(inplace=True), nn.ConvTranspose2d(channels[2], channels[3], 4, 2, 1, bias=True), nn.ReLU(inplace=True), nn.ConvTranspose2d(channels[3], channels[4], 4, 2, 1, bias=True), nn.ReLU(inplace=True), nn.ConvTranspose2d(channels[4], output_channel, 1, 1, 0, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x


class FaceEmbedNet(nn.Module):

    def __init__(self, input_channel, channels, output_channel):
        super(FaceEmbedNet, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(input_channel, channels[0], 1, 1, 0, bias=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(channels[0], channels[1], 4, 2, 1, bias=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(channels[1], channels[2], 4, 2, 1, bias=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(channels[2], channels[3], 4, 2, 1, bias=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(channels[3], channels[4], 4, 2, 1, bias=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(channels[4], output_channel, 4, 1, 0, bias=True))

    def forward(self, x):
        x = self.model(x)
        return x


class Classifier(nn.Module):

    def __init__(self, input_channel, channels, output_channel):
        super(Classifier, self).__init__()
        self.model = nn.Linear(input_channel, output_channel, bias=False)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.model(x)
        return x


class SynergyNet(nn.Module):
    """Defintion of 2D-to-3D-part"""

    def __init__(self, pretrained=False, last_CN=None):
        super(SynergyNet, self).__init__()
        self.backbone = getattr(mobilenetv2_backbone, 'mobilenet_v2')(last_CN=last_CN)
        ckpt = torch.load('pretrained_models/2D-to-3D-pretrained.tar')['state_dict']
        model_dict = self.backbone.state_dict()
        for k, v in ckpt.items():
            if 'IGM' in k:
                name_reduced = k.split('.', 3)[-1]
                model_dict[name_reduced] = v
        if pretrained:
            self.backbone.load_state_dict(model_dict)
        self.param_std = ckpt['module.param_std']
        self.param_mean = ckpt['module.param_mean']
        self.w_shp = ckpt['module.w_shp']
        self.w_exp = ckpt['module.w_exp']
        self.u = ckpt['module.u'].unsqueeze(0)

    def forward(self, input, return_onlypose=False, return_interFeature=False):
        _3D_attr, pool_x, inter = self.backbone(input)
        if return_onlypose:
            return _3D_attr[:, :12] * self.param_std[:12] + self.param_mean[:12]
        else:
            _3D_face = self.reconstruct_vertex(_3D_attr, dense=True)
            if return_interFeature:
                return _3D_face, pool_x, inter
            return _3D_face

    def reconstruct_vertex(self, param, whitening=True, dense=False):
        """
        Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
        dense: if True, return dense vertex, else return 68 sparse landmarks.
        Working with batched tensors. Using Fortan-type reshape.
        """
        if whitening:
            if param.shape[1] == 102:
                param_ = param * self.param_std + self.param_mean
            else:
                raise RuntimeError('length of params mismatch')
        p, _, alpha_shp, alpha_exp = self.parse_param_102(param_)
        _, s = self.p_to_Rs(p)
        if dense:
            vertex = s.unsqueeze(1).unsqueeze(1) * (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp).squeeze().contiguous().view(-1, 53215, 3).transpose(1, 2)
        else:
            raise NotImplementedError('Only dense mesh reconstruction supported')
        return vertex

    def parse_param_102(self, param):
        """ Parse param into 3DMM semantics"""
        p_ = param[:, :12].reshape(-1, 3, 4)
        p = p_[:, :, :3]
        offset = p_[:, :, -1].reshape(-1, 3, 1)
        alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
        alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
        return p, offset, alpha_shp, alpha_exp

    def parse_param_102_pose(self, param):
        """ Parse only pose params"""
        p_ = param[:, :12].reshape(-1, 3, 4)
        p = p_[:, :, :3]
        R, s = self.p_to_Rs(p)
        offset = p_[:, :, -1].reshape(-1, 3, 1)
        return R, offset

    def p_to_Rs(self, R):
        """Convert P to R and s as in 3DDFA-V2"""
        s = (R[:, 0, :3].norm(dim=1) + R[:, 1, :3].norm(dim=1)) / 2.0
        return F.normalize(R, p=2, dim=2), s


class Generator1D_directMLP(nn.Module):

    def __init__(self):
        super(Generator1D_directMLP, self).__init__()
        self.num_scale = 1
        self.num_shape = 40
        self.num_exp = 10
        self.last_channel = 64
        self.classifier_scale = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, self.num_scale))
        self.classifier_shape = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, self.num_shape))
        self.classifier_exp = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.last_channel, self.num_exp))
        ckpt = torch.load('pretrained_models/2D-to-3D-pretrained.tar')['state_dict']
        None
        self.param_std = ckpt['module.param_std']
        self.param_mean = ckpt['module.param_mean']
        self.w_shp = ckpt['module.w_shp']
        self.w_exp = ckpt['module.w_exp']
        self.u = ckpt['module.u'].unsqueeze(0)

    def forward_test(self, x):
        """return mesh
        """
        x = x.reshape(x.shape[0], -1)
        x_scale = self.classifier_scale(x)
        x_shape = self.classifier_shape(x)
        x_exp = self.classifier_exp(x)
        _3D_attr = torch.cat((x_scale, x_shape, x_exp), dim=1)
        _3D_face = self.reconstruct_vertex_51_onlyDeform(_3D_attr, dense=True)
        return _3D_face

    def forward_test_param(self, x):
        """return 3dmm parameters
        """
        x = x.reshape(x.shape[0], -1)
        x_scale = self.classifier_scale(x)
        x_shape = self.classifier_shape(x)
        x_exp = self.classifier_exp(x)
        _3D_attr = torch.cat((x_scale, x_shape, x_exp), dim=1)
        return _3D_attr

    def reconstruct_vertex_51_onlyDeform(self, param, whitening=True, dense=False):
        """51 = 1 (scale) + 40 (shape) + 10 (expr)
        """
        if whitening:
            if param.shape[1] == 51:
                s = param[:, 0] * 1.538597731841497e-05 + 0.0005920184194110334
                param_ = param[:, 1:] * self.param_std[12:62] + self.param_mean[12:62]
            else:
                raise RuntimeError('length of params mismatch')
        alpha_shp, alpha_exp = self.parse_param_50(param_)
        if dense:
            vertex = s.unsqueeze(1).unsqueeze(1) * (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp).squeeze().contiguous().view(-1, 53215, 3).transpose(1, 2)
        return vertex

    def parse_param_50(self, param):
        """Work for only tensor"""
        alpha_shp = param[:, :40].reshape(-1, 40, 1)
        alpha_exp = param[:, 40:50].reshape(-1, 10, 1)
        return alpha_shp, alpha_exp


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Classifier,
     lambda: ([], {'input_channel': 4, 'channels': 4, 'output_channel': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (ConvBNReLU,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Correlation,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FaceEmbedNet,
     lambda: ([], {'input_channel': 4, 'channels': [4, 4, 4, 4, 4], 'output_channel': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Generator,
     lambda: ([], {'input_channel': 4, 'channels': [4, 4, 4, 4, 4], 'output_channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (NSTLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (PKT,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (RKDLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Similarity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (VIDLoss,
     lambda: ([], {'num_input_channels': 4, 'num_mid_channel': 4, 'num_target_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (VoiceEmbedNet,
     lambda: ([], {'input_channel': 4, 'channels': [4, 4, 4, 4], 'output_channel': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_choyingw_Cross_Modal_Perceptionist(_paritybench_base):
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

