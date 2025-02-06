import sys
_module = sys.modules[__name__]
del sys
dataset = _module
loss = _module
main = _module
main_test = _module
metrics = _module
model = _module
sampler = _module
test = _module
train = _module
util = _module
arcface = _module
dataset = _module
main = _module
main2 = _module
model = _module
sampler = _module
test = _module
train = _module
triplet = _module
util = _module

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


from torch.utils.data import Dataset


from torchvision import transforms


import pandas as pd


import torch


from torch.utils.data import DataLoader


import torch.nn as nn


from torch.nn import functional as F


from torch.nn.parameter import Parameter


from torch.utils.data.sampler import Sampler


import itertools


import numpy as np


import matplotlib.pyplot as plt


from sklearn.metrics import classification_report


from sklearn.metrics import confusion_matrix


from torch import optim


import math


def gem(x, p=3, eps=1e-06):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-06, p_trainable=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class InceptionResnet(nn.Module):

    def __init__(self, device, pool=None, dropout=0.3, pretrain=True):
        super(InceptionResnet, self).__init__()
        if pretrain:
            self.net = InceptionResnetV1(pretrained='vggface2', dropout_prob=dropout, device=device)
        else:
            self.net = InceptionResnetV1(dropout_prob=dropout, device=device)
        self.out_features = self.net.last_linear.in_features
        if pool == 'gem':
            self.net.avgpool_1a = GeM(p_trainable=True)

    def forward(self, x):
        return self.net(x)


class EfficientNetEncoderHead(nn.Module):

    def __init__(self, depth, pretrain=True):
        super(EfficientNetEncoderHead, self).__init__()
        self.depth = depth
        if pretrain:
            self.net = EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
        else:
            self.net = EfficientNet.from_name(f'efficientnet-b{self.depth}')
        self.out_features = self.net._fc.in_features

    def forward(self, x):
        return self.net.extract_features(x)


class SEResNeXt101(nn.Module):

    def __init__(self, pretrained=True):
        super(SEResNeXt101, self).__init__()
        self.net = timm.create_model('gluon_seresnext101_32x4d', pretrained=pretrained)
        self.out_features = self.net.fc.in_features

    def forward(self, x):
        return self.net.forward_features(x)


class FaceNet(nn.Module):

    def __init__(self, model_name=None, pool=None, dropout=0.0, embedding_size=512, device='cuda', pretrain=True):
        super(FaceNet, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet':
            self.model = SEResNeXt101(pretrain)
        elif model_name == 'effnet':
            self.model = EfficientNetEncoderHead(depth=3, pretrain=pretrain)
        else:
            self.model = InceptionResnet(device, pool=pool, dropout=dropout, pretrain=pretrain)
        if pool == 'gem':
            self.global_pool = GeM(p_trainable=True)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(nn.Linear(self.model.out_features, embedding_size, bias=True), nn.BatchNorm1d(embedding_size, eps=0.001))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.model_name == None:
            return self.model(x)
        x = self.model(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        x = x[:, :, 0, 0]
        embeddings = self.neck(x)
        return embeddings


class ArcMarginProduct(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class FaceNet2(nn.Module):

    def __init__(self, num_classes, model_name=None, pool=None, dropout=0.0, embedding_size=512, device='cuda', pretrain=True):
        super(FaceNet2, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet':
            self.model = SEResNeXt101(pretrain)
        elif model_name == 'effnet':
            self.model = EfficientNetEncoderHead(depth=3, pretrain=pretrain)
        else:
            self.model = InceptionResnet(device, pool=pool, dropout=dropout, pretrain=pretrain)
        if pool == 'gem':
            self.global_pool = GeM(p_trainable=True)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(nn.Linear(self.model.out_features, embedding_size, bias=True), nn.BatchNorm1d(embedding_size, eps=0.001))
        self.dropout = nn.Dropout(p=dropout)
        self.head = ArcMarginProduct(embedding_size, num_classes)

    def forward(self, x):
        if self.model_name == None:
            embeddings = self.model(x)
            logits = self.head(embeddings)
            return {'logits': logits, 'embeddings': embeddings}
        x = self.model(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        x = x[:, :, 0, 0]
        embeddings = self.neck(x)
        logits = self.head(embeddings)
        return {'logits': logits, 'embeddings': embeddings}


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-07):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class ArcFaceLoss(nn.modules.Module):

    def __init__(self, s=45.0, m=0.1, crit='bce', weight=None, reduction='mean', class_weights_norm='batch'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm
        if crit == 'focal':
            self.crit = FocalLoss(gamma=2)
        elif crit == 'bce':
            self.crit = nn.CrossEntropyLoss(reduction='none')
        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.0], requires_grad=True, device='cuda'))
        else:
            self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = labels2 * phi + (1.0 - labels2) * cosine
        s = self.s
        output = output * s
        loss = self.crit(output, labels)
        if self.weight is not None:
            w = self.weight[labels]
            loss = loss * w
            if self.class_weights_norm == 'batch':
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == 'global':
                loss = loss.mean()
            else:
                loss = loss.mean()
            return loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ArcMarginProduct,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GeM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_SamYuen101234_Masked_Face_Recognition(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

