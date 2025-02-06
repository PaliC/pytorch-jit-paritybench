import sys
_module = sys.modules[__name__]
del sys
chesscog = _module
__version__ = _module
core = _module
coordinates = _module
dataset = _module
dataset = _module
datasets = _module
transforms = _module
evaluation = _module
exceptions = _module
io = _module
download = _module
models = _module
registry = _module
statistics = _module
training = _module
create_configs = _module
optimizer = _module
train = _module
train_classifier = _module
corner_detection = _module
detect_corners = _module
evaluate = _module
find_best_configs = _module
visualize = _module
data_synthesis = _module
create_fens = _module
download_dataset = _module
download_pgn = _module
split_dataset = _module
occupancy_classifier = _module
create_dataset = _module
download_model = _module
models = _module
piece_classifier = _module
models = _module
recognition = _module
recognition = _module
report = _module
analyze_misclassified_positions = _module
prepare_classifier_results = _module
prepare_confusion_matrix = _module
prepare_error_distribution = _module
prepare_recognition_results = _module
transfer_learning = _module
download_models = _module
train = _module
conf = _module
synthesize_data = _module
test_download_models = _module
test_transforms = _module
test_coordinates = _module
test_io = _module
test_registry = _module
test_statistics = _module
test_detect_corners = _module
test_version = _module
bump_version = _module

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


import torch


import typing


import functools


from collections.abc import Iterable


import torchvision


import logging


from enum import Enum


from torchvision import transforms as T


from abc import ABC


from torch import nn


from torch.utils.tensorboard import SummaryWriter


import copy


from torchvision import models


import torch.nn.functional as F


NUM_CLASSES = len({'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'}) * 2


class CNN100_3Conv_3Pool_3FC(nn.Module):
    """CNN (100, 3, 3, 3) model.
    """
    input_size = 100, 200
    pretrained = False

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 10 * 22, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 10 * 22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN100_3Conv_3Pool_2FC(nn.Module):
    """CNN (100, 3, 3, 2) model.
    """
    input_size = 100, 200
    pretrained = False

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 10 * 22, 1000)
        self.fc2 = nn.Linear(1000, NUM_CLASSES)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 10 * 22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN50_2Conv_2Pool_3FC(nn.Module):
    """CNN (50, 2, 2, 3) model.
    """
    input_size = 50, 50
    pretrained = False

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 11 * 11, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN50_2Conv_2Pool_2FC(nn.Module):
    """CNN (50, 2, 2, 2) model.
    """
    input_size = 50, 50
    pretrained = False

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 11 * 11, 1000)
        self.fc2 = nn.Linear(1000, NUM_CLASSES)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN50_3Conv_1Pool_2FC(nn.Module):
    """CNN (50, 3, 1, 2) model.
    """
    input_size = 50, 50
    pretrained = False

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 17 * 17, 1000)
        self.fc2 = nn.Linear(1000, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 17 * 17)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN50_3Conv_1Pool_3FC(nn.Module):
    """CNN (50, 3, 1, 3) model.
    """
    input_size = 50, 50
    pretrained = False

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 17 * 17, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 17 * 17)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet(nn.Module):
    """AlexNet model.
    """
    input_size = 100, 200
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        n = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(n, NUM_CLASSES)
        self.params = {'head': list(self.model.classifier[6].parameters())}

    def forward(self, x):
        return self.model(x)


class ResNet(nn.Module):
    """ResNet model.
    """
    input_size = 100, 200
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, NUM_CLASSES)
        self.params = {'head': list(self.model.fc.parameters())}

    def forward(self, x):
        return self.model(x)


class VGG(nn.Module):
    """VGG model.
    """
    input_size = 100, 200
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.vgg11_bn(pretrained=True)
        n = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(n, NUM_CLASSES)
        self.params = {'head': list(self.model.classifier[6].parameters())}

    def forward(self, x):
        return self.model(x)


class InceptionV3(nn.Module):
    """InceptionV3 model.
    """
    input_size = 299, 299
    pretrained = True

    def __init__(self):
        super().__init__()
        self.model = models.inception_v3(pretrained=True)
        n = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(n, NUM_CLASSES)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n, NUM_CLASSES)
        self.params = {'head': list(self.model.AuxLogits.fc.parameters()) + list(self.model.fc.parameters())}

    def forward(self, x):
        return self.model(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlexNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (InceptionV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 128, 128])], {}),
     True),
    (ResNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (VGG,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_georg_wolflein_chesscog(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

