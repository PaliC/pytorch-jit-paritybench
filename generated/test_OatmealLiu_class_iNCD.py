import sys
_module = sys.modules[__name__]
del sys
data = _module
cifarloader = _module
concat = _module
imagenetloader = _module
make_tinyimagenet = _module
omniglot = _module
omniglotloader = _module
rotationloader = _module
svhnloader = _module
tinyimagenetloader = _module
utils = _module
incd_2step_cifar100 = _module
incd_2step_tinyimagenet = _module
incd_ablation_expt = _module
models = _module
resnet = _module
supervised_learning_wo_ssl = _module
logging = _module
painter = _module
ramps = _module
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


import numpy as np


import random


import torch


import torch.utils.data as data


import torchvision.transforms as transforms


import warnings


from torch import randperm


import torch.backends.cudnn as cudnn


import torchvision


from torch.utils.data.dataloader import default_collate


from torch.utils.data.dataloader import DataLoader


from torchvision import transforms


import torchvision.datasets as datasets


import itertools


from torch.utils.data.sampler import Sampler


import torch.nn as nn


import torch.nn.functional as F


from torch.optim import SGD


from torch.optim import lr_scheduler


from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


from sklearn.metrics import adjusted_rand_score as ari_score


from sklearn.cluster import KMeans


import copy


from collections.abc import Iterable


import torch.optim as optim


from torch.autograd import Variable


from matplotlib import pyplot as plt


import matplotlib.lines as mline


import matplotlib.cm as cm


from sklearn.manifold import TSNE


from sklearn.metrics import confusion_matrix


from torch.nn.parameter import Parameter


import matplotlib


from scipy.optimize import linear_sum_assignment as linear_assignment


import scipy.io


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_labeled_classes=5, num_unlabeled_classes=5):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.head1 = nn.Linear(512 * block.expansion, num_labeled_classes)
        self.head2 = nn.Linear(512 * block.expansion, num_unlabeled_classes)
        self.l2_classifier = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_feat(self, feat):
        out = feat
        if self.l2_classifier:
            out1 = self.head1(F.normalize(out, dim=-1))
        else:
            out1 = self.head1(out)
        return out1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if out.size(2) > 4:
            out = F.avg_pool2d(out, out.size(2))
        else:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.relu(out)
        if self.l2_classifier:
            out1 = self.head1(F.normalize(out, dim=-1))
            out1 /= 0.1
        else:
            out1 = self.head1(out)
        out2 = self.head2(out)
        return out1, out2, out


class ResNetTri(nn.Module):

    def __init__(self, block, num_blocks, num_labeled_classes=80, num_unlabeled_classes1=10, num_unlabeled_classes2=10):
        super(ResNetTri, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.head1 = nn.Linear(512 * block.expansion, num_labeled_classes)
        self.head2 = nn.Linear(512 * block.expansion, num_unlabeled_classes1)
        self.head3 = nn.Linear(512 * block.expansion, num_unlabeled_classes2)
        self.l2_classifier = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_feat(self, feat):
        out = feat
        if self.l2_classifier:
            out1 = self.head1(F.normalize(out, dim=-1))
        else:
            out1 = self.head1(out)
        return out1

    def forward(self, x, output='None'):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if out.size(2) > 4:
            out = F.avg_pool2d(out, out.size(2))
        else:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.relu(out)
        if self.l2_classifier:
            out1 = self.head1(F.normalize(out, dim=-1))
            out1 /= 0.1
        else:
            out1 = self.head1(out)
        out2 = self.head2(out)
        out3 = self.head3(out)
        if output == 'test':
            return out1, out2, out3, out
        else:
            return out1, out3, out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.is_padding = 0
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.AvgPool2d(2)
            if in_planes != self.expansion * planes:
                self.is_padding = 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.is_padding:
            shortcut = self.shortcut(x)
            out += torch.cat([shortcut, torch.zeros(shortcut.shape).type(torch.FloatTensor)], 1)
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class BCE(nn.Module):
    eps = 1e-07

    def forward(self, prob1, prob2, simi):
        assert len(prob1) == len(prob2) == len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)), str(len(prob2)), str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_OatmealLiu_class_iNCD(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

