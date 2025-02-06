import sys
_module = sys.modules[__name__]
del sys
default_img = _module
default_vid = _module
data = _module
dataloader = _module
dataset_loader = _module
ccvid = _module
deepchange = _module
last = _module
ltcc = _module
prcc = _module
vcclothes = _module
img_transforms = _module
samplers = _module
spatial_transforms = _module
temporal_transforms = _module
losses = _module
arcface_loss = _module
circle_loss = _module
clothes_based_adversarial_loss = _module
contrastive_loss = _module
cosface_loss = _module
cross_entropy_loss_with_label_smooth = _module
gather = _module
triplet_loss = _module
main = _module
models = _module
classifier = _module
img_resnet = _module
c3d_blocks = _module
inflate = _module
nonlocal_blocks = _module
pooling = _module
vid_resnet = _module
test = _module
eval_metrics = _module
utils = _module
train = _module

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


from torch.utils.data import DataLoader


import torch


import queue


from torch import distributed as dist


import functools


from torch.utils.data import Dataset


import copy


import math


import random


import numpy as np


from collections import defaultdict


from torch.utils.data.sampler import Sampler


import numbers


import collections


import torchvision.transforms as T


from torch import nn


import torch.nn.functional as F


import torch.distributed as dist


import time


import logging


import torch.nn as nn


import torch.optim as optim


from torch.optim import lr_scheduler


from torch.nn import init


from torch.nn import functional as F


from torch.nn import Parameter


import torchvision


class ArcFaceLoss(nn.Module):
    """ ArcFace loss.

    Reference:
        Deng et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In CVPR, 2019.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """

    def __init__(self, scale=16, margin=0.1):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        index = inputs.data * 0.0
        index.scatter_(1, targets.data.view(-1, 1), 1)
        index = index.bool()
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        cos_t = inputs[index]
        sin_t = torch.sqrt(1.0 - cos_t * cos_t)
        cos_t_add_m = cos_t * cos_m - sin_t * sin_m
        cond_v = cos_t - math.cos(math.pi - self.m)
        cond = F.relu(cond_v)
        keep = cos_t - math.sin(math.pi - self.m) * self.m
        cos_t_add_m = torch.where(cond.bool(), cos_t_add_m, keep)
        output = inputs * 1.0
        output[index] = cos_t_add_m
        output = self.s * output
        return F.cross_entropy(output, targets)


class CircleLoss(nn.Module):
    """ Circle Loss based on the predictions of classifier.

    Reference:
        Sun et al. Circle Loss: A Unified Perspective of Pair Similarity Optimization. In CVPR, 2020.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """

    def __init__(self, scale=96, margin=0.3, **kwargs):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        mask = torch.zeros_like(inputs)
        mask.scatter_(1, targets.view(-1, 1), 1.0)
        pos_scale = self.s * F.relu(1 + self.m - inputs.detach())
        neg_scale = self.s * F.relu(inputs.detach() + self.m)
        scale_matrix = pos_scale * mask + neg_scale * (1 - mask)
        scores = (inputs - (1 - self.m) * mask - self.m * (1 - mask)) * scale_matrix
        loss = F.cross_entropy(scores, targets)
        return loss


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class PairwiseCircleLoss(nn.Module):
    """ Circle Loss among sample pairs.

    Reference:
        Sun et al. Circle Loss: A Unified Perspective of Pair Similarity Optimization. In CVPR, 2020.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """

    def __init__(self, scale=48, margin=0.35, **kwargs):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        inputs = F.normalize(inputs, p=2, dim=1)
        gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        m, n = targets.size(0), gallery_targets.size(0)
        similarities = torch.matmul(inputs, gallery_inputs.t())
        targets, gallery_targets = targets.view(-1, 1), gallery_targets.view(-1, 1)
        mask = torch.eq(targets, gallery_targets.T).float()
        mask_self = torch.zeros_like(mask)
        rank = dist.get_rank()
        mask_self[:, rank * m:(rank + 1) * m] += torch.eye(m).float()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask
        pos_scale = self.s * F.relu(1 + self.m - similarities.detach())
        neg_scale = self.s * F.relu(similarities.detach() + self.m)
        scale_matrix = pos_scale * mask_pos + neg_scale * mask_neg
        scores = (similarities - self.m) * mask_neg + (1 - self.m - similarities) * mask_pos
        scores = scores * scale_matrix
        neg_scores_LSE = torch.logsumexp(scores * mask_neg - 99999999 * (1 - mask_neg), dim=1)
        pos_scores_LSE = torch.logsumexp(scores * mask_pos - 99999999 * (1 - mask_pos), dim=1)
        loss = F.softplus(neg_scores_LSE + pos_scores_LSE).mean()
        return loss


class ClothesBasedAdversarialLoss(nn.Module):
    """ Clothes-based Adversarial Loss.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
    """

    def __init__(self, scale=16, epsilon=0.1):
        super().__init__()
        self.scale = scale
        self.epsilon = epsilon

    def forward(self, inputs, targets, positive_mask):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). The clothes classes with 
                the same identity as the anchor sample are defined as positive clothes classes and their mask 
                values are 1. The clothes classes with different identities from the anchor sample are defined 
                as negative clothes classes and their mask values in positive_mask are 0.
        """
        inputs = self.scale * inputs
        negtive_mask = 1 - positive_mask
        identity_mask = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        exp_logits = torch.exp(inputs)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = inputs - log_sum_exp_pos_and_all_neg
        mask = (1 - self.epsilon) * identity_mask + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (-mask * log_prob).sum(1).mean()
        return loss


class ClothesBasedAdversarialLossWithMemoryBank(nn.Module):
    """ Clothes-based Adversarial Loss between mini batch and the samples in memory.

    Reference:
        Gu et al. Clothes-Changing Person Re-identification with RGB Modality Only. In CVPR, 2022.

    Args:
        num_clothes (int): the number of clothes classes.
        feat_dim (int): the dimensions of feature.
        momentum (float): momentum to update memory.
        scale (float): scaling factor.
        epsilon (float): a trade-off hyper-parameter.
    """

    def __init__(self, num_clothes, feat_dim, momentum=0.0, scale=16, epsilon=0.1):
        super().__init__()
        self.num_clothes = num_clothes
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.epsilon = epsilon
        self.scale = scale
        self.register_buffer('feature_memory', torch.zeros((num_clothes, feat_dim)))
        self.register_buffer('label_memory', torch.zeros(num_clothes, dtype=torch.int64) - 1)
        self.has_been_filled = False

    def forward(self, inputs, targets, positive_mask):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
            positive_mask: positive mask matrix with shape (batch_size, num_classes). 
        """
        gathered_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gathered_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        self._update_memory(gathered_inputs.detach(), gathered_targets)
        inputs_norm = F.normalize(inputs, p=2, dim=1)
        memory_norm = F.normalize(self.feature_memory.detach(), p=2, dim=1)
        similarities = torch.matmul(inputs_norm, memory_norm.t()) * self.scale
        negtive_mask = 1 - positive_mask
        mask_identity = torch.zeros(positive_mask.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if not self.has_been_filled:
            invalid_index = self.label_memory == -1
            positive_mask[:, invalid_index] = 0
            negtive_mask[:, invalid_index] = 0
            if sum(invalid_index.type(torch.int)) == 0:
                self.has_been_filled = True
                None
        exp_logits = torch.exp(similarities)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * negtive_mask).sum(1, keepdim=True) + exp_logits)
        log_prob = similarities - log_sum_exp_pos_and_all_neg
        mask = (1 - self.epsilon) * mask_identity + self.epsilon / positive_mask.sum(1, keepdim=True) * positive_mask
        loss = (-mask * log_prob).sum(1).mean()
        return loss

    def _update_memory(self, features, labels):
        label_to_feat = {}
        for x, y in zip(features, labels):
            if y not in label_to_feat:
                label_to_feat[y] = [x.unsqueeze(0)]
            else:
                label_to_feat[y].append(x.unsqueeze(0))
        if not self.has_been_filled:
            for y in label_to_feat:
                feat = torch.mean(torch.cat(label_to_feat[y], dim=0), dim=0)
                self.feature_memory[y] = feat
                self.label_memory[y] = y
        else:
            for y in label_to_feat:
                feat = torch.mean(torch.cat(label_to_feat[y], dim=0), dim=0)
                self.feature_memory[y] = self.momentum * self.feature_memory[y] + (1.0 - self.momentum) * feat


class ContrastiveLoss(nn.Module):
    """ Supervised Contrastive Learning Loss among sample pairs.

    Args:
        scale (float): scaling factor.
    """

    def __init__(self, scale=16, **kwargs):
        super().__init__()
        self.s = scale

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        inputs = F.normalize(inputs, p=2, dim=1)
        gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        m, n = targets.size(0), gallery_targets.size(0)
        similarities = torch.matmul(inputs, gallery_inputs.t()) * self.s
        targets, gallery_targets = targets.view(-1, 1), gallery_targets.view(-1, 1)
        mask = torch.eq(targets, gallery_targets.T).float()
        mask_self = torch.zeros_like(mask)
        rank = dist.get_rank()
        mask_self[:, rank * m:(rank + 1) * m] += torch.eye(m).float()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask
        exp_logits = torch.exp(similarities) * (1 - mask_self)
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * mask_neg).sum(1, keepdim=True) + exp_logits)
        log_prob = similarities - log_sum_exp_pos_and_all_neg
        loss = (mask_pos * log_prob).sum(1) / mask_pos.sum(1)
        loss = -loss.mean()
        return loss


class CosFaceLoss(nn.Module):
    """ CosFace Loss based on the predictions of classifier.

    Reference:
        Wang et al. CosFace: Large Margin Cosine Loss for Deep Face Recognition. In CVPR, 2018.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """

    def __init__(self, scale=16, margin=0.1, **kwargs):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        one_hot = torch.zeros_like(inputs)
        one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        output = self.s * (inputs - one_hot * self.m)
        return F.cross_entropy(output, targets)


class PairwiseCosFaceLoss(nn.Module):
    """ CosFace Loss among sample pairs.

    Reference:
        Sun et al. Circle Loss: A Unified Perspective of Pair Similarity Optimization. In CVPR, 2020.

    Args:
        scale (float): scaling factor.
        margin (float): pre-defined margin.
    """

    def __init__(self, scale=16, margin=0):
        super().__init__()
        self.s = scale
        self.m = margin

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        inputs = F.normalize(inputs, p=2, dim=1)
        gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        m, n = targets.size(0), gallery_targets.size(0)
        similarities = torch.matmul(inputs, gallery_inputs.t())
        targets, gallery_targets = targets.view(-1, 1), gallery_targets.view(-1, 1)
        mask = torch.eq(targets, gallery_targets.T).float()
        mask_self = torch.zeros_like(mask)
        rank = dist.get_rank()
        mask_self[:, rank * m:(rank + 1) * m] += torch.eye(m).float()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask
        scores = (similarities + self.m) * mask_neg - similarities * mask_pos
        scores = scores * self.s
        neg_scores_LSE = torch.logsumexp(scores * mask_neg - 99999999 * (1 - mask_neg), dim=1)
        pos_scores_LSE = torch.logsumexp(scores * mask_pos - 99999999 * (1 - mask_pos), dim=1)
        loss = F.softplus(neg_scores_LSE + pos_scores_LSE).mean()
        return loss


class CrossEntropyWithLabelSmooth(nn.Module):
    """ Cross entropy loss with label smoothing regularization.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. In CVPR, 2016.
    Equation: 
        y = (1 - epsilon) * y + epsilon / K.

    Args:
        epsilon (float): a hyper-parameter in the above equation.
    """

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        _, num_classes = inputs.size()
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """ Triplet loss with hard example mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Args:
        margin (float): pre-defined margin.

    Note that we use cosine similarity, rather than Euclidean distance in the original paper.
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.m = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: sample features (before classifier) with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        inputs = F.normalize(inputs, p=2, dim=1)
        gallery_inputs = torch.cat(GatherLayer.apply(inputs), dim=0)
        gallery_targets = torch.cat(GatherLayer.apply(targets), dim=0)
        dist = 1 - torch.matmul(inputs, gallery_inputs.t())
        targets, gallery_targets = targets.view(-1, 1), gallery_targets.view(-1, 1)
        mask_pos = torch.eq(targets, gallery_targets.T).float()
        mask_neg = 1 - mask_pos
        dist_ap, _ = torch.max(dist - mask_neg * 99999999.0, dim=1)
        dist_an, _ = torch.min(dist + mask_pos * 99999999.0, dim=1)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class Classifier(nn.Module):

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        y = self.classifier(x)
        return y


class NormalizedClassifier(nn.Module):

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_classes, feature_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-05).mul_(100000.0)

    def forward(self, x):
        w = self.weight
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(w, p=2, dim=1)
        return F.linear(x, w)


class ResNet50(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride = 1, 1
            resnet50.layer4[0].downsample[0].stride = 1, 1
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))
        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def forward(self, x):
        x = self.base(x)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        f = self.bn(x)
        return f


class APM(nn.Module):

    def __init__(self, in_channels, out_channels, time_dim=3, temperature=4, contrastive_att=True):
        super(APM, self).__init__()
        self.time_dim = time_dim
        self.temperature = temperature
        self.contrastive_att = contrastive_att
        padding = 0, 0, 0, 0, (time_dim - 1) // 2, (time_dim - 1) // 2
        self.padding = nn.ConstantPad3d(padding, value=0)
        self.semantic_mapping = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        if self.contrastive_att:
            self.x_mapping = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            self.n_mapping = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            self.contrastive_att_net = nn.Sequential(nn.Conv3d(out_channels, 1, kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, t, h, w = x.size()
        N = self.time_dim
        neighbor_time_index = torch.cat([(torch.arange(0, t) + i).unsqueeze(0) for i in range(N) if i != N // 2], dim=0).t().flatten().long()
        semantic = self.semantic_mapping(x)
        x_norm = F.normalize(semantic, p=2, dim=1)
        x_norm_padding = self.padding(x_norm)
        x_norm_expand = x_norm.unsqueeze(3).expand(-1, -1, -1, N - 1, -1, -1).permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, h * w, c // 16)
        neighbor_norm = x_norm_padding[:, :, neighbor_time_index, :, :].permute(0, 2, 1, 3, 4).contiguous().view(-1, c // 16, h * w)
        similarity = torch.matmul(x_norm_expand, neighbor_norm) * self.temperature
        similarity = F.softmax(similarity, dim=-1)
        x_padding = self.padding(x)
        neighbor = x_padding[:, :, neighbor_time_index, :, :].permute(0, 2, 3, 4, 1).contiguous().view(-1, h * w, c)
        neighbor_new = torch.matmul(similarity, neighbor).view(b, t * (N - 1), h, w, c).permute(0, 4, 1, 2, 3)
        if self.contrastive_att:
            x_att = self.x_mapping(x.unsqueeze(3).expand(-1, -1, -1, N - 1, -1, -1).contiguous().view(b, c, (N - 1) * t, h, w).detach())
            n_att = self.n_mapping(neighbor_new.detach())
            contrastive_att = self.contrastive_att_net(x_att * n_att)
            neighbor_new = neighbor_new * contrastive_att
        x_offset = torch.zeros([b, c, N * t, h, w], dtype=x.data.dtype, device=x.device.type)
        x_index = torch.tensor([i for i in range(t * N) if i % N == N // 2])
        neighbor_index = torch.tensor([i for i in range(t * N) if i % N != N // 2])
        x_offset[:, :, x_index, :, :] += x
        x_offset[:, :, neighbor_index, :, :] += neighbor_new
        return x_offset


class C2D(nn.Module):

    def __init__(self, conv2d, **kwargs):
        super(C2D, self).__init__()
        kernel_dim = 1, conv2d.kernel_size[0], conv2d.kernel_size[1]
        stride = 1, conv2d.stride[0], conv2d.stride[0]
        padding = 0, conv2d.padding[0], conv2d.padding[1]
        self.conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=conv2d.bias)
        weight_2d = conv2d.weight.data
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2)
        weight_3d[:, :, 0, :, :] = weight_2d
        self.conv3d.weight = nn.Parameter(weight_3d)
        self.conv3d.bias = conv2d.bias

    def forward(self, x):
        out = self.conv3d(x)
        return out


class I3D(nn.Module):

    def __init__(self, conv2d, time_dim=3, time_stride=1, **kwargs):
        super(I3D, self).__init__()
        kernel_dim = time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1]
        stride = time_stride, conv2d.stride[0], conv2d.stride[0]
        padding = time_dim // 2, conv2d.padding[0], conv2d.padding[1]
        self.conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=conv2d.bias)
        weight_2d = conv2d.weight.data
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
        self.conv3d.weight = nn.Parameter(weight_3d)
        self.conv3d.bias = conv2d.bias

    def forward(self, x):
        out = self.conv3d(x)
        return out


class API3D(nn.Module):

    def __init__(self, conv2d, time_dim=3, time_stride=1, temperature=4, contrastive_att=True):
        super(API3D, self).__init__()
        self.APM = APM(conv2d.in_channels, conv2d.in_channels // 16, time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        kernel_dim = time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1]
        stride = time_stride * time_dim, conv2d.stride[0], conv2d.stride[0]
        padding = 0, conv2d.padding[0], conv2d.padding[1]
        self.conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=conv2d.bias)
        weight_2d = conv2d.weight.data
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
        self.conv3d.weight = nn.Parameter(weight_3d)
        self.conv3d.bias = conv2d.bias

    def forward(self, x):
        x_offset = self.APM(x)
        out = self.conv3d(x_offset)
        return out


class P3DA(nn.Module):

    def __init__(self, conv2d, time_dim=3, time_stride=1, **kwargs):
        super(P3DA, self).__init__()
        kernel_dim = 1, conv2d.kernel_size[0], conv2d.kernel_size[1]
        stride = 1, conv2d.stride[0], conv2d.stride[0]
        padding = 0, conv2d.padding[0], conv2d.padding[1]
        self.spatial_conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=conv2d.bias)
        weight_2d = conv2d.weight.data
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2)
        weight_3d[:, :, 0, :, :] = weight_2d
        self.spatial_conv3d.weight = nn.Parameter(weight_3d)
        self.spatial_conv3d.bias = conv2d.bias
        kernel_dim = time_dim, 1, 1
        stride = time_stride, 1, 1
        padding = time_dim // 2, 0, 0
        self.temporal_conv3d = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=False)
        weight_2d = torch.eye(conv2d.out_channels).unsqueeze(2).unsqueeze(2)
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
        self.temporal_conv3d.weight = nn.Parameter(weight_3d)

    def forward(self, x):
        x = self.spatial_conv3d(x)
        out = self.temporal_conv3d(x)
        return out


class P3DB(nn.Module):

    def __init__(self, conv2d, time_dim=3, time_stride=1, **kwargs):
        super(P3DB, self).__init__()
        kernel_dim = 1, conv2d.kernel_size[0], conv2d.kernel_size[1]
        stride = 1, conv2d.stride[0], conv2d.stride[0]
        padding = 0, conv2d.padding[0], conv2d.padding[1]
        self.spatial_conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=conv2d.bias)
        weight_2d = conv2d.weight.data
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2)
        weight_3d[:, :, 0, :, :] = weight_2d
        self.spatial_conv3d.weight = nn.Parameter(weight_3d)
        self.spatial_conv3d.bias = conv2d.bias
        kernel_dim = time_dim, 1, 1
        stride = time_stride, conv2d.stride[0], conv2d.stride[0]
        padding = time_dim // 2, 0, 0
        self.temporal_conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=False)
        nn.init.constant_(self.temporal_conv3d.weight, 0)

    def forward(self, x):
        out1 = self.spatial_conv3d(x)
        out2 = self.temporal_conv3d(x)
        out = out1 + out2
        return out


class P3DC(nn.Module):

    def __init__(self, conv2d, time_dim=3, time_stride=1, **kwargs):
        super(P3DC, self).__init__()
        kernel_dim = 1, conv2d.kernel_size[0], conv2d.kernel_size[1]
        stride = 1, conv2d.stride[0], conv2d.stride[0]
        padding = 0, conv2d.padding[0], conv2d.padding[1]
        self.spatial_conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=conv2d.bias)
        weight_2d = conv2d.weight.data
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2)
        weight_3d[:, :, 0, :, :] = weight_2d
        self.spatial_conv3d.weight = nn.Parameter(weight_3d)
        self.spatial_conv3d.bias = conv2d.bias
        kernel_dim = time_dim, 1, 1
        stride = time_stride, 1, 1
        padding = time_dim // 2, 0, 0
        self.temporal_conv3d = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=False)
        nn.init.constant_(self.temporal_conv3d.weight, 0)

    def forward(self, x):
        out = self.spatial_conv3d(x)
        residual = self.temporal_conv3d(out)
        out = out + residual
        return out


class APP3DA(nn.Module):

    def __init__(self, conv2d, time_dim=3, time_stride=1, temperature=4, contrastive_att=True):
        super(APP3DA, self).__init__()
        self.APM = APM(conv2d.out_channels, conv2d.out_channels // 16, time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        kernel_dim = 1, conv2d.kernel_size[0], conv2d.kernel_size[1]
        stride = 1, conv2d.stride[0], conv2d.stride[0]
        padding = 0, conv2d.padding[0], conv2d.padding[1]
        self.spatial_conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=conv2d.bias)
        weight_2d = conv2d.weight.data
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2)
        weight_3d[:, :, 0, :, :] = weight_2d
        self.spatial_conv3d.weight = nn.Parameter(weight_3d)
        self.spatial_conv3d.bias = conv2d.bias
        kernel_dim = time_dim, 1, 1
        stride = time_stride * time_dim, 1, 1
        padding = 0, 0, 0
        self.temporal_conv3d = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=False)
        weight_2d = torch.eye(conv2d.out_channels).unsqueeze(2).unsqueeze(2)
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
        self.temporal_conv3d.weight = nn.Parameter(weight_3d)

    def forward(self, x):
        x = self.spatial_conv3d(x)
        out = self.temporal_conv3d(self.APM(x))
        return out


class APP3DB(nn.Module):

    def __init__(self, conv2d, time_dim=3, time_stride=1, temperature=4, contrastive_att=True):
        super(APP3DB, self).__init__()
        self.APM = APM(conv2d.in_channels, conv2d.in_channels // 16, time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        kernel_dim = 1, conv2d.kernel_size[0], conv2d.kernel_size[1]
        stride = 1, conv2d.stride[0], conv2d.stride[0]
        padding = 0, conv2d.padding[0], conv2d.padding[1]
        self.spatial_conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=conv2d.bias)
        weight_2d = conv2d.weight.data
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2)
        weight_3d[:, :, 0, :, :] = weight_2d
        self.spatial_conv3d.weight = nn.Parameter(weight_3d)
        self.spatial_conv3d.bias = conv2d.bias
        kernel_dim = time_dim, 1, 1
        stride = time_stride * time_dim, conv2d.stride[0], conv2d.stride[0]
        padding = 0, 0, 0
        self.temporal_conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=False)
        nn.init.constant_(self.temporal_conv3d.weight, 0)

    def forward(self, x):
        out1 = self.spatial_conv3d(x)
        out2 = self.temporal_conv3d(self.APM(x))
        out = out1 + out2
        return out


class APP3DC(nn.Module):

    def __init__(self, conv2d, time_dim=3, time_stride=1, temperature=4, contrastive_att=True):
        super(APP3DC, self).__init__()
        self.APM = APM(conv2d.out_channels, conv2d.out_channels // 16, time_dim=time_dim, temperature=temperature, contrastive_att=contrastive_att)
        kernel_dim = 1, conv2d.kernel_size[0], conv2d.kernel_size[1]
        stride = 1, conv2d.stride[0], conv2d.stride[0]
        padding = 0, conv2d.padding[0], conv2d.padding[1]
        self.spatial_conv3d = nn.Conv3d(conv2d.in_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=conv2d.bias)
        weight_2d = conv2d.weight.data
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2)
        weight_3d[:, :, 0, :, :] = weight_2d
        self.spatial_conv3d.weight = nn.Parameter(weight_3d)
        self.spatial_conv3d.bias = conv2d.bias
        kernel_dim = time_dim, 1, 1
        stride = time_stride * time_dim, 1, 1
        padding = 0, 0, 0
        self.temporal_conv3d = nn.Conv3d(conv2d.out_channels, conv2d.out_channels, kernel_size=kernel_dim, padding=padding, stride=stride, bias=False)
        nn.init.constant_(self.temporal_conv3d.weight, 0)

    def forward(self, x):
        out = self.spatial_conv3d(x)
        residual = self.temporal_conv3d(self.APM(out))
        out = out + residual
        return out


class MaxPool2dFor3dInput(nn.Module):
    """
    Since nn.MaxPool3d is nondeterministic operation, using fixed random seeds can't get consistent results.
    So we attempt to use max_pool2d to implement MaxPool3d with kernelsize (1, kernel_size, kernel_size).
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * t, c, h, w)
        x = self.maxpool(x)
        _, _, h, w = x.size()
        x = x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return x


class NonLocalBlockND(nn.Module):

    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d
        self.g = conv_nd(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.theta = conv_nd(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.phi = conv_nd(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        if sub_sample:
            if dimension == 3:
                self.g = nn.Sequential(self.g, max_pool((1, 2, 2)))
                self.phi = nn.Sequential(self.phi, max_pool((1, 2, 2)))
            else:
                self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))
        if bn_layer:
            self.W = nn.Sequential(conv_nd(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=True), bn(self.in_channels))
        else:
            self.W = conv_nd(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, conv_nd):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if bn_layer:
            nn.init.constant_(self.W[1].weight.data, 0.0)
            nn.init.constant_(self.W[1].bias.data, 0.0)
        else:
            nn.init.constant_(self.W.weight.data, 0.0)
            nn.init.constant_(self.W.bias.data, 0.0)

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f = F.softmax(f, dim=-1)
        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        y = self.W(y)
        z = y + x
        return z


class NonLocalBlock1D(NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock1D, self).__init__(in_channels, inter_channels=inter_channels, dimension=1, sub_sample=sub_sample, bn_layer=bn_layer)


class NonLocalBlock2D(NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock2D, self).__init__(in_channels, inter_channels=inter_channels, dimension=2, sub_sample=sub_sample, bn_layer=bn_layer)


class NonLocalBlock3D(NonLocalBlockND):

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock3D, self).__init__(in_channels, inter_channels=inter_channels, dimension=3, sub_sample=sub_sample, bn_layer=bn_layer)


class GeMPooling(nn.Module):

    def __init__(self, p=3, eps=1e-06):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), x.size()[2:]).pow(1.0 / self.p)


class MaxAvgPooling(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_f = self.maxpooling(x)
        avg_f = self.avgpooling(x)
        return torch.cat((max_f, avg_f), 1)


class Bottleneck3D(nn.Module):

    def __init__(self, bottleneck2d, block, inflate_time=False, temperature=4, contrastive_att=True):
        super().__init__()
        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        if inflate_time == True:
            self.conv2 = block(bottleneck2d.conv2, temperature=temperature, contrastive_att=contrastive_att)
        else:
            self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)
        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(inflate.inflate_conv(downsample2d[0], time_dim=1, time_stride=time_stride), inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

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


class ResNet503D(nn.Module):

    def __init__(self, config, block, c3d_idx, nl_idx, **kwargs):
        super().__init__()
        self.block = block
        self.temperature = config.MODEL.AP3D.TEMPERATURE
        self.contrastive_att = config.MODEL.AP3D.CONTRACTIVE_ATT
        resnet2d = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet2d.layer4[0].conv2.stride = 1, 1
            resnet2d.layer4[0].downsample[0].stride = 1, 1
        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)
        self.layer1 = self._inflate_reslayer(resnet2d.layer1, c3d_idx=c3d_idx[0], nonlocal_idx=nl_idx[0], nonlocal_channels=256)
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, c3d_idx=c3d_idx[1], nonlocal_idx=nl_idx[1], nonlocal_channels=512)
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, c3d_idx=c3d_idx[2], nonlocal_idx=nl_idx[2], nonlocal_channels=1024)
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, c3d_idx=c3d_idx[3], nonlocal_idx=nl_idx[3], nonlocal_channels=2048)
        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def _inflate_reslayer(self, reslayer2d, c3d_idx, nonlocal_idx=[], nonlocal_channels=0):
        reslayers3d = []
        for i, layer2d in enumerate(reslayer2d):
            if i not in c3d_idx:
                layer3d = Bottleneck3D(layer2d, c3d_blocks.C2D, inflate_time=False)
            else:
                layer3d = Bottleneck3D(layer2d, self.block, inflate_time=True, temperature=self.temperature, contrastive_att=self.contrastive_att)
            reslayers3d.append(layer3d)
            if i in nonlocal_idx:
                non_local_block = nonlocal_blocks.NonLocalBlock3D(nonlocal_channels, sub_sample=True)
                reslayers3d.append(non_local_block)
        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b * t, c, h, w)
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        x = x.mean(1)
        f = self.bn(x)
        return f


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Classifier,
     lambda: ([], {'feature_dim': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ClothesBasedAdversarialLoss,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=torch.int64), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossEntropyWithLabelSmooth,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (GeMPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxAvgPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPool2dFor3dInput,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     True),
    (NonLocalBlock2D,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NormalizedClassifier,
     lambda: ([], {'feature_dim': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_guxinqian_Simple_CCReID(_paritybench_base):
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

