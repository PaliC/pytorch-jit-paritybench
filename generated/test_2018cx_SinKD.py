import sys
_module = sys.modules[__name__]
del sys
distillation = _module
distillation_t = _module
lossd = _module
distillation = _module
distillation_s = _module
distillation_t = _module
lossd = _module
glue_train = _module
loss = _module
main_glue = _module
main_glue_distill = _module
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


import logging


import random


from itertools import chain


from typing import Optional


from typing import Union


import math


import torch


from torch.utils.data import DataLoader


import torch.nn.functional as F


from torch.cuda.amp import autocast


from torch.cuda.amp import GradScaler


import warnings


import torch.nn as nn


import numpy as np


import torch.optim as optim


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data.distributed import DistributedSampler


import time


import torch.distributed as dist


from sklearn.metrics import matthews_corrcoef


from torch.utils.data import Dataset


class KL(nn.Module):

    def __init__(self):
        super(KL, self).__init__()
        self.T = 1

    def forward(self, y_s, y_t, mode='classification'):
        y_s = y_s.view(-1, 32128)
        y_t = y_t.view(-1, 32128)
        y_s = torch.log_softmax(y_s, dim=-1)
        y_t = torch.log_softmax(y_t, dim=-1)
        if mode == 'regression':
            loss = F.mse_loss((y_s / self.T).view(-1), (y_t / self.T).view(-1))
        else:
            p_s = F.log_softmax(y_s / self.T, dim=-1)
            p_t = F.softmax(y_t / self.T, dim=-1)
            loss = -torch.sum(p_t * p_s, dim=-1).mean()
        return loss


class Sinkhorn(nn.Module):

    def __init__(self, T):
        super(Sinkhorn, self).__init__()
        self.T = 2

    def sinkhorn_normalized(self, x, n_iters=10):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=20):
        Wxy = torch.cdist(x, y, p=1)
        K = torch.exp(-Wxy / epsilon)
        P = self.sinkhorn_normalized(K, n_iters)
        return torch.sum(P * Wxy)

    def forward(self, y_s, y_t, mode='classification'):
        softmax = nn.Softmax(dim=1)
        p_s = softmax(y_s / self.T)
        p_t = softmax(y_t / self.T)
        emd_loss = 0.001 * self.sinkhorn_loss(x=p_s, y=p_t)
        return emd_loss


class RKL(nn.Module):

    def __init__(self):
        super(RKL, self).__init__()
        self.T = 2

    def forward(self, y_s, y_t, mode='classification'):
        temperature = self.T
        if mode == 'regression':
            loss = F.mse_loss((y_s / temperature).view(-1), (y_t / temperature).view(-1))
        else:
            p_s = F.softmax(y_s / temperature, dim=-1)
            p_s1 = F.log_softmax(y_s / temperature, dim=-1)
            p_t = F.log_softmax(y_t / temperature, dim=-1)
            loss = torch.sum(p_s1 * p_s, dim=-1).mean() - torch.sum(p_t * p_s, dim=-1).mean()
        return 0.1 * loss


class JSKL(nn.Module):

    def __init__(self):
        super(JSKL, self).__init__()
        self.T = 2

    def js_divergence(self, p, q):
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(p, m, reduction='batchmean') + F.kl_div(q, m, reduction='batchmean'))

    def forward(self, y_s, y_t, mode='classification'):
        temperature = self.T
        if mode == 'regression':
            loss = F.mse_loss((y_s / temperature).view(-1), (y_t / temperature).view(-1))
        else:
            p_s = F.softmax(y_s / temperature, dim=-1)
            p_t = F.softmax(y_t / temperature, dim=-1)
            loss = (0.5 * self.js_divergence(p_s, p_t)).mean()
        return 0.1 * loss


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear = nn.Linear(2048, 4096)

    def forward(self, x):
        x = x
        x = x.view(-1, 2048)
        x = self.linear(x)
        return x


class saliency_mse(nn.Module):

    def __init__(self, top_k, norm, loss_func):
        super(saliency_mse, self).__init__()
        self.top_k = top_k
        self.norm = norm
        self.loss_func = loss_func

    def forward(self, s_loss, t_loss, s_hidden, t_hidden):
        t_grad = torch.autograd.grad(t_loss, [t_hidden[0]], create_graph=False)
        t_input_grad = t_grad[0]
        t_saliency = t_input_grad * t_hidden[0].detach()
        t_topk_saliency = torch.topk(torch.abs(t_saliency), self.top_k, dim=-1)[0]
        t_saliency = torch.norm(t_topk_saliency, dim=-1)
        t_saliency = F.normalize(t_saliency, p=self.norm, dim=1)
        s_grad = torch.autograd.grad(s_loss, [s_hidden[0]], create_graph=True, retain_graph=True)
        s_input_grad = s_grad[0]
        s_saliency = s_input_grad * s_hidden[0]
        s_saliency = torch.norm(s_saliency, dim=-1)
        s_saliency = F.normalize(s_saliency, p=self.norm, dim=1)
        if self.loss_func == 'L1':
            return F.l1_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()
        elif self.loss_func == 'L2':
            return F.mse_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()
        elif self.loss_func == 'smoothL1':
            return F.smooth_l1_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()


class DistillKL(nn.Module):

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, mode='classification'):
        if mode == 'regression':
            loss = F.mse_loss((y_s / self.T).view(-1), (y_t / self.T).view(-1))
        else:
            p_s = F.log_softmax(y_s / self.T, dim=-1)
            p_t = F.softmax(y_t / self.T, dim=-1)
            loss = -torch.sum(p_t * p_s, dim=-1).mean()
        return loss


class DistillKL_anneal(nn.Module):

    def __init__(self, T):
        super(DistillKL_anneal, self).__init__()
        self.T = T

    def forward(self, cur_epoch, total_epoch, y_s, y_t, mode='classification'):
        temperature = (1 - self.T) / (total_epoch - 1) * cur_epoch + self.T
        if mode == 'regression':
            loss = F.mse_loss((y_s / temperature).view(-1), (y_t / temperature).view(-1))
        else:
            p_s = F.log_softmax(y_s / temperature, dim=-1)
            p_t = F.softmax(y_t / temperature, dim=-1)
            loss = -torch.sum(p_t * p_s, dim=-1).mean()
        return loss


class DistillJS(nn.Module):

    def __init__(self, T):
        super(saliency_mse, self).__init__()
        self.T = T

    def js_divergence(self, p, q):
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(p, m, reduction='batchmean') + F.kl_div(q, m, reduction='batchmean'))

    def forward(self, y_s, y_t, mode='classification'):
        temperature = self.T
        if mode == 'regression':
            loss = F.mse_loss((y_s / temperature).view(-1), (y_t / temperature).view(-1))
        else:
            p_s = F.softmax(y_s / temperature, dim=-1)
            p_t = F.softmax(y_t / temperature, dim=-1)
            loss = (0.5 * self.js_divergence(p_s, p_t)).mean()
        return 0.1 * loss


class DistillTVD(nn.Module):

    def __init__(self, T):
        super(saliency_mse, self).__init__()
        self.T = T

    def tvd_loss(self, p, q):
        return 0.5 * (p - q).abs().mean()

    def forward(self, y_s, y_t, mode='classification'):
        temperature = self.T
        if mode == 'regression':
            loss = F.mse_loss((y_s / temperature).view(-1), (y_t / temperature).view(-1))
        else:
            p_s = F.softmax(y_s / temperature, dim=-1)
            p_t = F.softmax(y_t / temperature, dim=-1)
        loss = self.tvd_loss(p_s, p_t)
        return 0.1 * loss


class Sinkhorn_sample(nn.Module):

    def __init__(self, T):
        super(Sinkhorn_sample, self).__init__()
        self.T = 2

    def sinkhorn_normalized(self, x, n_iters=10):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=20):
        Wxy = torch.cdist(x, y, p=1)
        K = torch.exp(-Wxy / epsilon)
        P = self.sinkhorn_normalized(K, n_iters)
        return torch.sum(P * Wxy)

    def forward(self, y_s, y_t, mode='classification'):
        batch_size = y_s.size(0)
        p_s = F.softmax(y_s, dim=-1)
        p_t = F.softmax(y_t, dim=-1)
        emd_loss = 0.0
        for i in range(batch_size):
            emd_loss += self.sinkhorn_loss(x=p_s[i:i + 1], y=p_t[i:i + 1])
        return 0.001 * emd_loss


class saliency_mse_for_multiclass(nn.Module):

    def __init__(self, top_k, norm, loss_func):
        super(saliency_mse_for_multiclass, self).__init__()
        self.top_k = top_k
        self.norm = norm
        self.loss_func = loss_func

    def forward(self, s_logits, t_logits, s_hidden, t_hidden):
        loss = 0
        num_labels = s_logits.size(1)
        t_probs = F.softmax(t_logits, dim=-1)
        s_probs = F.softmax(s_logits, dim=-1)
        for i in range(num_labels):
            t_loss = -torch.log(t_probs[:, i]).mean()
            t_grad = torch.autograd.grad(t_loss, [t_hidden[0]], create_graph=False, retain_graph=True)
            t_input_grad = t_grad[0]
            t_saliency = t_input_grad * t_hidden[0].detach()
            t_topk_saliency = torch.topk(torch.abs(t_saliency), self.top_k, dim=-1)[0]
            t_saliency = torch.norm(t_topk_saliency, dim=-1)
            t_saliency = F.normalize(t_saliency, p=self.norm, dim=1)
            s_loss = -torch.log(s_probs[:, i]).mean()
            s_grad = torch.autograd.grad(s_loss, [s_hidden[0]], create_graph=True, retain_graph=True)
            s_input_grad = s_grad[0]
            s_saliency = s_input_grad * s_hidden[0]
            s_saliency = torch.norm(s_saliency, dim=-1)
            s_saliency = F.normalize(s_saliency, p=self.norm, dim=1)
            if self.loss_func == 'L1':
                loss += F.l1_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()
            elif self.loss_func == 'L2':
                loss += F.mse_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()
            elif self.loss_func == 'smoothL1':
                loss += F.smooth_l1_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()
        return loss


class integrated_gradient_mse(nn.Module):

    def __init__(self, total_step, top_k, norm, loss_func):
        super(integrated_gradient_mse, self).__init__()
        self.total_step = total_step
        self.top_k = top_k
        self.norm = norm
        self.loss_func = loss_func

    def forward(self, inputs, student, teacher):
        t_grad_tmp, s_grad_tmp = 0, 0
        for step in range(1, 1 + self.total_step):
            alpha = step / self.total_step
            if isinstance(teacher, nn.DataParallel) or isinstance(teacher, nn.parallel.distributed.DistributedDataParallel):
                t_alpha_hook = teacher.module.bert.embeddings.register_forward_hook(lambda module, input_data, output_data: output_data * alpha)
                s_alpha_hook = student.module.bert.embeddings.register_forward_hook(lambda module, input_data, output_data: output_data * alpha)
            else:
                t_alpha_hook = teacher.bert.embeddings.register_forward_hook(lambda module, input_data, output_data: output_data * alpha)
                s_alpha_hook = student.bert.embeddings.register_forward_hook(lambda module, input_data, output_data: output_data * alpha)
            teacher.eval()
            t_loss, teacher_logits, t_hidden, _ = teacher(**inputs)
            t_grad = torch.autograd.grad(t_loss, [t_hidden[0]], create_graph=False)
            t_input_grad = t_grad[0]
            t_grad_tmp = t_grad_tmp + t_input_grad
            t_alpha_hook.remove()
            s_loss, student_logits, s_hidden, _ = student(**inputs)
            s_grad = torch.autograd.grad(s_loss, [s_hidden[0]], create_graph=True)
            s_input_grad = s_grad[0]
            s_grad_tmp = s_grad_tmp + s_input_grad
            s_alpha_hook.remove()
        t_mean_grad = t_grad_tmp / self.total_step
        t_saliency = t_mean_grad * t_hidden[0].detach()
        t_topk_saliency = torch.topk(torch.abs(t_saliency), self.top_k, dim=-1)[0]
        t_saliency = torch.norm(t_topk_saliency, dim=-1)
        t_saliency = F.normalize(t_saliency, p=self.norm, dim=1)
        s_mean_grad = s_grad_tmp / self.total_step
        s_saliency = s_mean_grad * s_hidden[0]
        s_saliency = torch.norm(s_saliency, dim=-1)
        s_saliency = F.normalize(s_saliency, p=self.norm, dim=1)
        if self.loss_func == 'L1':
            return F.l1_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()
        elif self.loss_func == 'L2':
            return F.mse_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()
        elif self.loss_func == 'smoothL1':
            return F.smooth_l1_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()


class integrated_gradient_mse_for_multiclass(nn.Module):

    def __init__(self, total_step, top_k, norm, loss_func):
        super(integrated_gradient_mse_for_multiclass, self).__init__()
        self.total_step = total_step
        self.top_k = top_k
        self.norm = norm
        self.loss_func = loss_func

    def forward(self, inputs, student, teacher, num_labels):
        loss = 0
        t_grad_tmp, s_grad_tmp = [(0) for _ in range(num_labels)], [(0) for _ in range(num_labels)]
        for step in range(1, 1 + self.total_step):
            alpha = step / self.total_step
            if isinstance(teacher, nn.DataParallel) or isinstance(teacher, nn.parallel.distributed.DistributedDataParallel):
                t_alpha_hook = teacher.module.bert.embeddings.register_forward_hook(lambda module, input_data, output_data: output_data * alpha)
                s_alpha_hook = student.module.bert.embeddings.register_forward_hook(lambda module, input_data, output_data: output_data * alpha)
            else:
                t_alpha_hook = teacher.bert.embeddings.register_forward_hook(lambda module, input_data, output_data: output_data * alpha)
                s_alpha_hook = student.bert.embeddings.register_forward_hook(lambda module, input_data, output_data: output_data * alpha)
            teacher.eval()
            _, teacher_logits, t_hidden, _ = teacher(**inputs)
            _, student_logits, s_hidden, _ = student(**inputs)
            t_probs = F.softmax(teacher_logits, dim=-1)
            s_probs = F.softmax(student_logits, dim=-1)
            for i in range(num_labels):
                t_loss = -torch.log(t_probs[:, i]).mean()
                t_grad = torch.autograd.grad(t_loss, [t_hidden[0]], create_graph=False, retain_graph=True)
                t_input_grad = t_grad[0]
                t_grad_tmp[i] = t_grad_tmp[i] + t_input_grad
                s_loss = -torch.log(s_probs[:, i]).mean()
                s_grad = torch.autograd.grad(s_loss, [s_hidden[0]], create_graph=True, retain_graph=True)
                s_input_grad = s_grad[0]
                s_grad_tmp[i] = s_grad_tmp[i] + s_input_grad
            t_alpha_hook.remove()
            s_alpha_hook.remove()
        for i in range(num_labels):
            t_mean_grad = t_grad_tmp[i] / self.total_step
            t_saliency = t_mean_grad * t_hidden[0].detach()
            t_topk_saliency = torch.topk(torch.abs(t_saliency), self.top_k, dim=-1)[0]
            t_saliency = torch.norm(t_topk_saliency, dim=-1)
            t_saliency = F.normalize(t_saliency, p=self.norm, dim=1)
            s_mean_grad = s_grad_tmp[i] / self.total_step
            s_saliency = s_mean_grad * s_hidden[0]
            s_saliency = torch.norm(s_saliency, dim=-1)
            s_saliency = F.normalize(s_saliency, p=self.norm, dim=1)
            if self.loss_func == 'L1':
                loss += F.l1_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()
            elif self.loss_func == 'L2':
                loss += F.mse_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()
            elif self.loss_func == 'smoothL1':
                loss += F.smooth_l1_loss(t_saliency, s_saliency, reduction='sum') / (t_saliency != 0).sum()
        return loss


class SampleLoss(nn.Module):

    def __init__(self, n_relation_heads):
        super(SampleLoss, self).__init__()
        self.n_relation_heads = n_relation_heads

    def mean_pooling(self, hidden, attention_mask):
        hidden = hidden * attention_mask.unsqueeze(-1)
        hidden = torch.sum(hidden, dim=1)
        hidden = hidden / torch.sum(attention_mask, dim=1).unsqueeze(-1)
        return hidden

    def cal_angle_multihead(self, hidden):
        batch, dim = hidden.shape
        hidden = hidden.view(batch, self.n_relation_heads, -1).permute(1, 0, 2)
        norm_ = F.normalize(hidden.unsqueeze(1) - hidden.unsqueeze(2), p=2, dim=-1)
        angle = torch.einsum('hijd, hidk->hijk', norm_, norm_.transpose(-2, -1))
        return angle

    def forward(self, s_rep, t_rep, attention_mask):
        s_rep = self.mean_pooling(s_rep, attention_mask)
        t_rep = self.mean_pooling(t_rep, attention_mask)
        with torch.no_grad():
            t_angle = self.cal_angle_multihead(t_rep)
        s_angle = self.cal_angle_multihead(s_rep)
        loss = F.smooth_l1_loss(s_angle.view(-1), t_angle.view(-1), reduction='elementwise_mean')
        return loss


def expand_gather(input, dim, index):
    size = list(input.size())
    size[dim] = -1
    return input.gather(dim, index.expand(*size))


class TokenPhraseLoss(nn.Module):

    def __init__(self, n_relation_heads, k1, k2):
        super(TokenPhraseLoss, self).__init__()
        self.n_relation_heads = n_relation_heads
        self.k1 = k1
        self.k2 = k2

    def forward(self, s_rep, t_rep, attention_mask):
        attention_mask_extended = torch.einsum('bl, bp->blp', attention_mask, attention_mask)
        attention_mask_extended = attention_mask_extended.unsqueeze(1).repeat(1, self.n_relation_heads, 1, 1).float()
        s_pair, s_global_topk, s_local_topk = self.cal_pairinteraction_multihead(s_rep, attention_mask_extended, self.n_relation_heads, k1=min(self.k1, s_rep.shape[1]), k2=min(self.k2, s_rep.shape[1]))
        with torch.no_grad():
            t_pair, t_global_topk, t_local_topk = self.cal_pairinteraction_multihead(t_rep, attention_mask_extended, self.n_relation_heads, k1=min(self.k1, t_rep.shape[1]), k2=min(self.k2, t_rep.shape[1]))
        loss_pair = F.mse_loss(s_pair.view(-1), t_pair.view(-1), reduction='sum') / torch.sum(attention_mask_extended)
        s_angle, s_mask = self.calculate_tripletangleseq_multihead(s_rep, attention_mask, 1, t_global_topk, t_local_topk)
        with torch.no_grad():
            t_angle, t_mask = self.calculate_tripletangleseq_multihead(t_rep, attention_mask, 1, t_global_topk, t_local_topk)
        loss_triplet = F.smooth_l1_loss(s_angle.view(-1), t_angle.view(-1), reduction='sum') / torch.sum(s_mask)
        return loss_pair + loss_triplet

    def cal_pairinteraction_multihead(self, hidden, attention_mask_extended, n_relation_heads, k1, k2):
        batch, seq_len, dim = hidden.shape
        hidden = hidden.view(batch, seq_len, n_relation_heads, -1).permute(0, 2, 1, 3)
        scores = torch.matmul(hidden, hidden.transpose(-1, -2))
        scores = scores / math.sqrt(dim // n_relation_heads)
        scores = scores * attention_mask_extended
        scores_out = scores
        attention_mask_extended_add = (1.0 - attention_mask_extended) * -10000.0
        scores = scores + attention_mask_extended_add
        scores = F.softmax(scores, dim=-1)
        scores = scores * attention_mask_extended
        global_score = scores.sum(2).sum(1)
        global_topk = global_score.topk(k1, dim=1)[1]
        local_score = scores.sum(1)
        mask = torch.ones_like(local_score)
        mask[:, range(mask.shape[-2]), range(mask.shape[-1])] = 0.0
        local_score = local_score * mask
        local_topk = local_score.topk(k2, dim=2)[1]
        index_ = global_topk.unsqueeze(-1)
        local_topk = expand_gather(local_topk, 1, index_)
        return scores_out, global_topk, local_topk

    def calculate_tripletangleseq_multihead(self, hidden, attention_mask, n_relation_heads, global_topk, local_topk):
        """
            hidden: batch, len, dim
            attention_mask: batch, len
            global_topk: batch, k1
            local_topk: batch, k1, k2
        """
        batch, seq_len, dim = hidden.shape
        hidden = hidden.view(batch, seq_len, n_relation_heads, -1).permute(0, 2, 1, 3)
        index_ = global_topk.unsqueeze(1).unsqueeze(-1)
        index_ = index_.repeat(1, n_relation_heads, 1, 1)
        hidden1 = expand_gather(hidden, 2, index_)
        sd = hidden1.unsqueeze(3) - hidden.unsqueeze(2)
        index_ = local_topk.unsqueeze(1).repeat(1, n_relation_heads, 1, 1).unsqueeze(-1)
        sd = expand_gather(sd, 3, index_)
        norm_sd = F.normalize(sd, p=2, dim=-1)
        angle = torch.einsum('bhijd, bhidk->bhijk', norm_sd, norm_sd.transpose(-2, -1))
        attention_mask1 = attention_mask.gather(-1, global_topk)
        attention_mask_extended = attention_mask1.unsqueeze(2) + attention_mask.unsqueeze(1)
        attention_mask_extended = attention_mask_extended.unsqueeze(1).repeat(1, n_relation_heads, 1, 1)
        attention_mask_extended = attention_mask_extended.unsqueeze(-1)
        index_ = local_topk.unsqueeze(1).repeat(1, n_relation_heads, 1, 1).unsqueeze(-1)
        attention_mask_extended = expand_gather(attention_mask_extended, 3, index_)
        attention_mask_extended = (torch.einsum('bhijd, bhidk->bhijk', attention_mask_extended.float(), attention_mask_extended.transpose(-2, -1).float()) == 4).float()
        mask = angle.ne(0).float()
        mask[:, :, :, range(mask.shape[-2]), range(mask.shape[-1])] = 0.0
        attention_mask_extended = attention_mask_extended * mask
        angle = angle * attention_mask_extended
        return angle, attention_mask_extended


class MGSKDLoss(nn.Module):

    def __init__(self, n_relation_heads=64, k1=20, k2=20, M=2, weights=(4.0, 1.0)):
        super(MGSKDLoss, self).__init__()
        self.n_relation_heads = n_relation_heads
        self.k1 = k1
        self.k2 = k2
        self.M = M
        self.w1, self.w2 = weights
        self.sample_loss = SampleLoss(n_relation_heads)
        self.tokenphrase_loss = TokenPhraseLoss(n_relation_heads, k1, k2)

    def forward(self, s_reps, t_reps, attention_mask):
        token_loss = 0.0
        sample_loss = 0.0
        for layer_id in range(s_reps.size(1)):
            if layer_id < self.M:
                token_loss += self.tokenphrase_loss(s_reps[:, layer_id], t_reps[:, layer_id], attention_mask)
            else:
                sample_loss += self.sample_loss(s_reps[:, layer_id], t_reps[:, layer_id], attention_mask)
        loss = self.w1 * sample_loss + self.w2 * token_loss
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DistillKL,
     lambda: ([], {'T': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DistillKL_anneal,
     lambda: ([], {'T': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (JSKL,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (KL,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 32128]), torch.rand([4, 32128])], {}),
     False),
    (MLP,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2048])], {}),
     True),
    (RKL,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Sinkhorn,
     lambda: ([], {'T': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Sinkhorn_sample,
     lambda: ([], {'T': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TokenPhraseLoss,
     lambda: ([], {'n_relation_heads': 4, 'k1': 4, 'k2': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_2018cx_SinKD(_paritybench_base):
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

