import sys
_module = sys.modules[__name__]
del sys
hub = _module
compressor = _module
hubconf = _module
lossyless = _module
architectures = _module
callbacks = _module
classical_compressors = _module
distortions = _module
distributions = _module
helpers = _module
learnable_compressors = _module
predictors = _module
rates = _module
main = _module
Z_linear_eval = _module
utils = _module
aggregate = _module
data = _module
augmentations = _module
label_augment = _module
base = _module
distributions = _module
helpers = _module
images = _module
helpers = _module
load_pretrained = _module
postplotting = _module
decorators = _module
helpers = _module
postplotter = _module
pretty_renamer = _module
save_hub = _module
visualizations = _module
images = _module

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


import numpy as np


import torch


import torch.nn as nn


from torch.utils.data import DataLoader


import logging


import math


from functools import partial


from typing import Iterable


import torchvision


from torchvision import transforms as transform_lib


from typing import Hashable


import matplotlib


import matplotlib.pyplot as plt


import pandas as pd


from matplotlib.lines import Line2D


from torch.nn import functional as F


from torchvision.transforms import ToPILImage


from torchvision.transforms import ToTensor


import torch.nn.functional as F


from torch.distributions import Categorical


from torch.distributions import Independent


from torch.distributions import MixtureSameFamily


from torch.distributions import Normal


from torch.nn.modules.conv import Conv2d


import itertools


import random


import warnings


from collections.abc import MutableSet


from functools import reduce


from numbers import Number


from torch import nn


from torch.distributions import Distribution


from torch.distributions import constraints


from torch.distributions.utils import broadcast_all


from torch.nn.utils.rnn import PackedSequence


from collections.abc import Sequence


import copy


from time import sleep


from sklearn.metrics import accuracy_score


from sklearn.metrics import balanced_accuracy_score


from sklearn.model_selection import PredefinedSplit


from sklearn.model_selection import RandomizedSearchCV


from sklearn.svm import LinearSVC


from torch.optim import Adam


from torch.optim import lr_scheduler


from torchvision.datasets import CIFAR10


from torchvision.datasets import STL10


import numbers


import numpy.random as random


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import RandomRotation


import abc


import torch.distributions as dist


from torch.utils.data import Dataset


from torchvision.datasets.folder import default_loader


from torchvision.transforms import functional as F_trnsf


from torch.utils.data import random_split


from torchvision.datasets import CIFAR100


from torchvision.datasets import MNIST


from torchvision.datasets import CocoCaptions


from torchvision.datasets import ImageFolder


from torchvision.datasets import ImageNet


from torchvision.transforms import CenterCrop


from torchvision.transforms import ColorJitter


from torchvision.transforms import Compose


from torchvision.transforms import Lambda


from torchvision.transforms import RandomAffine


from torchvision.transforms import RandomApply


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomErasing


from torchvision.transforms import RandomGrayscale


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import RandomVerticalFlip


from torchvision.transforms import Resize


import collections


from torch.utils.data import Subset


from copy import deepcopy


import inspect


from torchvision.utils import make_grid


def atleast_ndim(x, ndim):
    """Reshapes a tensor so that it has at least n dimensions."""
    if x is None:
        return None
    return x.view(list(x.shape) + [1] * (ndim - x.ndim))


def read_bytes(fd, n, fmt='>{:d}s'):
    sz = struct.calcsize('s')
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def read_uints(fd, n, fmt='>{:d}I'):
    sz = struct.calcsize('I')
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt='>{:d}s'):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def write_uints(fd, values, fmt='>{:d}I'):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def batch_flatten(x):
    """Batch wise flattenting of an array."""
    shape = x.shape
    return x.reshape(-1, shape[-1]), shape


def batch_unflatten(x, shape):
    """Revert `batch_flatten`."""
    return x.reshape(*shape[:-1], -1)


def get_Activation(activation, inverse=False):
    """Return an uninistantiated activation that takes the number of channels as inputs.

    Parameters
    ----------
    activation : {"gdn"}U{any torch.nn activation}
        Activation to use.

    inverse : bool, optional
        Whether using the activation in a transposed model.
    """
    if activation == 'gdn':
        return partial(GDN, inverse=inverse)
    return getattr(torch.nn, activation)


def get_Normalization(norm_layer, dim=2):
    """Return the correct normalization layer.

    Parameters
    ----------
    norm_layer : callable or {"batchnorm", "identity"}U{any torch.nn layer}
        Layer to return.

    dim : int, optional
        Number of dimension of the input (e.g. 2 for images).
    """
    Batchnorms = [None, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
    if 'batch' in norm_layer:
        Norm = Batchnorms[dim]
    elif norm_layer == 'identity':
        Norm = nn.Identity
    elif isinstance(norm_layer, str):
        Norm = getattr(torch.nn, norm_layer)
    else:
        Norm = norm_layer
    return Norm


def weights_init(module, nonlinearity='relu'):
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.
    """
    for m in module.children():
        if isinstance(m, torch.nn.modules.conv._ConvNd):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            try:
                nn.init.zeros_(m.bias)
            except AttributeError:
                pass
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
            try:
                nn.init.zeros_(m.bias)
            except AttributeError:
                pass
        elif isinstance(m, nn.BatchNorm2d):
            try:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            except AttributeError:
                pass
        elif hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        else:
            weights_init(m, nonlinearity=nonlinearity)


class MLP(nn.Module):
    """Multi Layer Perceptron.

    Parameters
    ----------
    in_dim : int

    out_dim : int

    hid_dim : int, optional
        Number of hidden neurones.

    n_hid_layers : int, optional
        Number of hidden layers.

    norm_layer : nn.Module or {"identity","batch"}, optional
        Normalizing layer to use.

    activation : {"gdn"}U{any torch.nn activation}, optional
        Activation to use.

    dropout_p : float, optional
        Dropout rate.
    """

    def __init__(self, in_dim, out_dim, n_hid_layers=1, hid_dim=128, norm_layer='identity', activation='ReLU', dropout_p=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid_layers = n_hid_layers
        self.hid_dim = hid_dim
        Activation = get_Activation(activation)
        Dropout = nn.Dropout if dropout_p > 0 else nn.Identity
        Norm = get_Normalization(norm_layer, dim=1)
        is_bias = Norm == nn.Identity
        layers = [nn.Linear(in_dim, hid_dim, bias=is_bias), Norm(hid_dim), Activation(), Dropout(p=dropout_p)]
        for _ in range(1, n_hid_layers):
            layers += [nn.Linear(hid_dim, hid_dim, bias=is_bias), Norm(hid_dim), Activation(), Dropout(p=dropout_p)]
        layers += [nn.Linear(hid_dim, out_dim)]
        self.module = nn.Sequential(*layers)
        self.reset_parameters()

    def forward(self, X):
        X, shape = batch_flatten(X)
        X = self.module(X)
        X = batch_unflatten(X, shape)
        return X

    def reset_parameters(self):
        weights_init(self)


def prod(iterable):
    """Take product of iterable like."""
    return reduce(operator.mul, iterable, 1)


class FlattenMLP(MLP):
    """
    MLP that can take a multi dimensional array as input and output (i.e. can be used with same
    input and output shape as CNN but permutation invariant.). E.g. for predicting an image use
    `out_shape=(32,32,3)` and this will predict 32*32*3 and then reshape.

    Parameters
    ----------
    in_shape : tuple or int

    out_shape : tuple or int

    kwargs :
        Additional arguments to `MLP`.
    """

    def __init__(self, in_shape, out_shape, **kwargs):
        self.in_shape = [in_shape] if isinstance(in_shape, int) else in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        in_dim = prod(self.in_shape)
        out_dim = prod(self.out_shape)
        super().__init__(in_dim=in_dim, out_dim=out_dim, **kwargs)

    def forward(self, X):
        X = X.flatten(start_dim=X.ndim - len(self.in_shape))
        X = super().forward(X)
        X = X.unflatten(dim=-1, sizes=self.out_shape)
        return X


class FlattenLinear(torch.nn.Linear):
    """
    Linear that can take a multi dimensional array as input and output . E.g. for predicting an image use
    `out_shape=(32,32,3)` and this will predict 32*32*3 and then reshape.

    Parameters
    ----------
    in_shape : tuple or int

    out_shape : tuple or int

    kwargs :
        Additional arguments to `torch.nn.Linear`.
    """

    def __init__(self, in_shape, out_shape, **kwargs):
        self.in_shape = [in_shape] if isinstance(in_shape, int) else in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        in_dim = prod(self.in_shape)
        out_dim = prod(self.out_shape)
        super().__init__(in_features=in_dim, out_features=out_dim, **kwargs)

    def forward(self, X):
        X = X.flatten(start_dim=X.ndim - len(self.in_shape))
        X = super().forward(X)
        X = X.unflatten(dim=-1, sizes=self.out_shape)
        return X


class Resnet(nn.Module):
    """Base class for renets.

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). This is used to see whether to change the underlying
        resnet or not. If first dim < 100, then will decrease the kernel size  and stride of the
        first conv, and remove the max pooling layer as done (for cifar10) in
        https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469.

    out_shape : int or tuple
        Size of the output.

    base : {'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2'}, optional
        Base resnet to use, any model `torchvision.models.resnet` should work (the larger models were
        not tested).

    is_pretrained : bool, optional
        Whether to load a model pretrained on imagenet. Might not work well with `is_small=True`.

    norm_layer : nn.Module or {"identity","batch"}, optional
        Normalizing layer to use.
    """

    def __init__(self, in_shape, out_shape, base='resnet18', is_pretrained=False, norm_layer='batchnorm'):
        super().__init__()
        kwargs = {}
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.out_dim = prod(self.out_shape)
        self.is_pretrained = is_pretrained
        if not self.is_pretrained:
            kwargs['num_classes'] = self.out_dim
        self.resnet = torchvision.models.__dict__[base](pretrained=self.is_pretrained, norm_layer=get_Normalization(norm_layer, 2), **kwargs)
        if self.is_pretrained:
            assert self.out_dim == self.resnet.fc.in_features
            self.resnet.fc = torch.nn.Identity()
        if self.in_shape[1] < 100:
            self.resnet.conv1 = nn.Conv2d(in_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet.maxpool = nn.Identity()
        self.reset_parameters()

    def forward(self, X):
        Y_pred = self.resnet(X)
        Y_pred = Y_pred.unflatten(dim=-1, sizes=self.out_shape)
        return Y_pred

    def reset_parameters(self):
        if self.in_shape[1] < 100:
            weights_init(self.resnet.conv1)


class PretrainedSSL(nn.Module):
    """Pretrained self supervised models.

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). Needs to be 3,224,224.

    out_shape : int or tuple
        Size of the output. Flattened needs to be 512 for clip_vit, 1024 for clip_rn50, and
        2048 for swav and simclr.

    model : {"swav", "simclr", "clip_vit", "clip_rn50"}
        Which SSL model to use.
    """

    def __init__(self, in_shape, out_shape, model):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = [out_shape] if isinstance(out_shape, int) else out_shape
        self.model = model
        self.out_dim = prod(self.out_shape)
        self.load_weights_()
        if self.model == 'clip_vit':
            assert self.out_dim == 512
        elif self.model == 'clip_rn50':
            assert self.out_dim == 1024
        elif self.model in ['swav', 'simclr']:
            assert self.out_dim == 2048
        else:
            raise ValueError(f'Unkown model={self.model}.')
        assert self.in_shape[0] == 3
        assert self.in_shape[1] == self.in_shape[2] == 224
        self.reset_parameters()

    def forward(self, X):
        z = self.encoder(X)
        z = z.unflatten(dim=-1, sizes=self.out_shape)
        return z

    def load_weights_(self):
        if self.model == 'simclr':
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
            self.encoder = SimCLR.load_from_checkpoint(weight_path, strict=False)
        elif self.model == 'swav':
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
            self.encoder = SwAV.load_from_checkpoint(weight_path, strict=False)
        elif 'clip' in self.model:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            arch = 'ViT-B/32' if 'vit' in self.model else 'RN50'
            model, _ = clip.load(arch, device, jit=False)
            self.encoder = model.visual
        else:
            raise ValueError(f'Unkown model={self.model}.')
        self.encoder.float()

    def reset_parameters(self):
        self.load_weights_()


def closest_pow(n, base=2):
    """Return the closest (in log space) power of 2 from a number."""
    return base ** round(math.log(n, base))


def is_pow2(n):
    """Check if a number is a power of 2."""
    return n != 0 and n & n - 1 == 0


logger = logging.getLogger(__name__)


class CNN(nn.Module):
    """CNN in shape of pyramid, which doubles hidden after each layer but decreases image size by 2.

    Notes
    -----
    - if some of the sides of the inputs are not power of 2 they will be resized to the closest power
    of 2 for prediction.
    - If `in_shape` and `out_dim` are reversed (i.e. `in_shape` is int) then will transpose the CNN.

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). If integer and `out_dim` is a tuple of int, then will
        transpose ("reverse") the CNN.
    out_dim : int
        Number of output channels. If tuple of int  and `in_shape` is an int, then will transpose
        ("reverse") the CNN.
    hid_dim : int, optional
        Base number of temporary channels (will be multiplied by 2 after each layer).
    norm_layer : callable or {"batchnorm", "identity"}
        Layer to return.
    activation : {"gdn"}U{any torch.nn activation}, optional
        Activation to use.
    n_layers : int, optional
        Number of layers. If `None` uses the required number of layers so that the smallest side
        is 2 after encoding (i.e. one less than the maximum).
    kwargs :
        Additional arguments to `ConvBlock`.
    """

    def __init__(self, in_shape, out_dim, hid_dim=32, norm_layer='batchnorm', activation='ReLU', n_layers=None, **kwargs):
        super().__init__()
        in_shape, out_dim, resizer = self.validate_sizes(out_dim, in_shape)
        self.in_shape = in_shape
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.norm_layer = norm_layer
        self.activation = activation
        self.n_layers = n_layers
        if self.n_layers is None:
            min_side = min(self.in_shape[1], self.in_shape[2])
            self.n_layers = int(math.log2(min_side) - 1)
        Norm = get_Normalization(self.norm_layer, 2)
        is_bias = Norm == nn.Identity
        channels = [self.in_shape[0]]
        channels += [(self.hid_dim * 2 ** i) for i in range(0, self.n_layers)]
        end_h = self.in_shape[1] // 2 ** self.n_layers
        end_w = self.in_shape[2] // 2 ** self.n_layers
        if self.is_transpose:
            channels.reverse()
        layers = []
        in_chan = channels[0]
        for i, out_chan in enumerate(channels[1:]):
            is_last = i == len(channels[1:]) - 1
            layers += self.make_block(in_chan, out_chan, Norm, is_bias, is_last, **kwargs)
            in_chan = out_chan
        if self.is_transpose:
            pre_layers = [nn.Linear(self.out_dim, channels[0] * end_w * end_h, bias=is_bias), nn.Unflatten(dim=-1, unflattened_size=(channels[0], end_h, end_w))]
            post_layers = [resizer]
        else:
            pre_layers = [resizer]
            post_layers = [nn.Flatten(start_dim=1), nn.Linear(channels[-1] * end_w * end_h, self.out_dim)]
        self.model = nn.Sequential(*(pre_layers + layers + post_layers))
        self.reset_parameters()

    def validate_sizes(self, out_dim, in_shape):
        if isinstance(out_dim, int) and not isinstance(in_shape, int):
            self.is_transpose = False
        else:
            in_shape, out_dim = out_dim, in_shape
            self.is_transpose = True
        resizer = torch.nn.Identity()
        is_input_pow2 = is_pow2(in_shape[1]) and is_pow2(in_shape[2])
        if not is_input_pow2:
            in_shape_pow2 = list(in_shape)
            in_shape_pow2[1] = closest_pow(in_shape[1], base=2)
            in_shape_pow2[2] = closest_pow(in_shape[2], base=2)
            if self.is_transpose:
                resizer = transform_lib.Resize((in_shape[1], in_shape[2]))
            else:
                resizer = transform_lib.Resize((in_shape_pow2[1], in_shape_pow2[2]))
            logger.warning(f'The input shape={in_shape} is not powers of 2 so we will rescale it and work with shape {in_shape_pow2}.')
            in_shape = in_shape_pow2
        return in_shape, out_dim, resizer

    def make_block(self, in_chan, out_chan, Norm, is_bias, is_last, **kwargs):
        if self.is_transpose:
            Activation = get_Activation(self.activation, inverse=True)
            return [Norm(in_chan), Activation(in_chan), nn.ConvTranspose2d(in_chan, out_chan, stride=2, padding=1, kernel_size=3, output_padding=1, bias=is_bias or is_last, **kwargs)]
        else:
            Activation = get_Activation(self.activation, inverse=True)
            return [nn.Conv2d(in_chan, out_chan, stride=2, padding=1, kernel_size=3, bias=is_bias, **kwargs), Norm(out_chan), Activation(out_chan)]

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):
        return self.model(X)


class BALLE(nn.Module):
    """CNN from Balle's factorized prior. The key difference with the other encoders, is that it
    keeps some spatial structure in Z. I.e. representation can be seen as a flattened latent image.

    Notes
    -----
    - replicates https://github.com/InterDigitalInc/CompressAI/blob/a73c3378e37a52a910afaf9477d985f86a06634d/compressai/models/priors.py#L104

    Parameters
    ----------
    in_shape : tuple of int
        Size of the inputs (channels first). If integer and `out_dim` is a tuple of int, then will
        transpose ("reverse") the CNN.

    out_dim : int
        Number of output channels. If tuple of int  and `in_shape` is an int, then will transpose
        ("reverse") the CNN.

    hid_dim : int, optional
        Number of channels for every layer.

    n_layers : int, optional
        Number of layers, after every layer divides image by 2 on each side.

    norm_layer : callable or {"batchnorm", "identity"}
        Normalization layer.

    activation : {"gdn"}U{any torch.nn activation}, optional
        Activation to use. Typically that would be GDN for lossy image compression, but did not
        work for Galaxy (maybe because all black pixels).
    """
    validate_sizes = CNN.validate_sizes

    def __init__(self, in_shape, out_dim, hid_dim=256, n_layers=4, norm_layer='batchnorm', activation='ReLU'):
        super().__init__()
        in_shape, out_dim, resizer = self.validate_sizes(out_dim, in_shape)
        self.in_shape = in_shape
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.activation = activation
        self.norm_layer = norm_layer
        end_h = self.in_shape[1] // 2 ** self.n_layers
        end_w = self.in_shape[2] // 2 ** self.n_layers
        self.channel_out_dim = self.out_dim // (end_w * end_h)
        layers = [self.make_block(self.hid_dim, self.hid_dim) for _ in range(self.n_layers - 2)]
        if self.is_transpose:
            pre_layers = [nn.Unflatten(dim=-1, unflattened_size=(self.channel_out_dim, end_h, end_w)), self.make_block(self.channel_out_dim, self.hid_dim)]
            post_layers = [self.make_block(self.hid_dim, self.in_shape[0], is_last=True), resizer]
        else:
            pre_layers = [resizer, self.make_block(self.in_shape[0], self.hid_dim)]
            post_layers = [self.make_block(self.hid_dim, self.channel_out_dim, is_last=True), nn.Flatten(start_dim=1)]
        self.model = nn.Sequential(*(pre_layers + layers + post_layers))
        self.reset_parameters()

    def make_block(self, in_chan, out_chan, is_last=False, kernel_size=5, stride=2):
        if is_last:
            Norm = nn.Identity
        else:
            Norm = get_Normalization(self.norm_layer, 2)
        is_bias = Norm == nn.Identity
        if self.is_transpose:
            conv = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2, bias=is_bias)
        else:
            conv = nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=is_bias)
        if not is_last:
            Activation = get_Activation(self.activation, inverse=self.is_transpose)
            conv = nn.Sequential(conv, Norm(out_chan), Activation(out_chan))
        return conv

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):
        return self.model(X)


BASE_LOG = 2


MEANS = dict(imagenet=[0.485, 0.456, 0.406], cifar10=[0.4914009, 0.48215896, 0.4465308], galaxy=[0.03294565, 0.04387402, 0.04995899], clip=[0.48145466, 0.4578275, 0.40821073], stl10=[0.43, 0.42, 0.39], stl10_unlabeled=[0.43, 0.42, 0.39])


STDS = dict(imagenet=[0.229, 0.224, 0.225], cifar10=[0.24703279, 0.24348423, 0.26158753], galaxy=[0.07004886, 0.07964786, 0.09574898], clip=[0.26862954, 0.26130258, 0.27577711], stl10=[0.27, 0.26, 0.27], stl10_unlabeled=[0.27, 0.26, 0.27])


class UnNormalizer(torch.nn.Module):

    def __init__(self, dataset, is_raise=True):
        super().__init__()
        self.dataset = dataset.lower()
        try:
            mean, std = MEANS[self.dataset], STDS[self.dataset]
            self.unnormalizer = transform_lib.Normalize([(-m / s) for m, s in zip(mean, std)], std=[(1 / s) for s in std])
        except KeyError:
            if is_raise:
                raise KeyError(f"dataset={self.dataset} wasn't found in MEANS={MEANS.keys()} orSTDS={STDS.keys()}. Please add mean and std.")
            else:
                self.normalizer = None

    def forward(self, x):
        if self.unnormalizer is None:
            return x
        return self.unnormalizer(x)


def get_Architecture(mode, **kwargs):
    """Return the (uninstantiated) correct architecture.

    Parameters
    ----------
    mode : {"mlp","linear","resnet","identity", "balle", "clip", "clip_rn50", "simclr", "swav"}

    kwargs :
        Additional arguments to the Module.

    Return
    ------
    Architecture : uninstantiated nn.Module
        Architecture that can be instantiated by `Architecture(in_shape, out_shape)`
    """
    if mode == 'mlp':
        return partial(FlattenMLP, **kwargs)
    elif mode == 'linear':
        return partial(FlattenLinear, **kwargs)
    elif mode == 'identity':
        return torch.nn.Identity
    elif mode == 'resnet':
        return partial(Resnet, **kwargs)
    elif mode == 'cnn':
        return partial(CNN, **kwargs)
    elif mode == 'balle':
        return partial(BALLE, **kwargs)
    elif mode == 'clip':
        return partial(PretrainedSSL, model='clip_vit', **kwargs)
    elif mode == 'clip_rn50':
        return partial(PretrainedSSL, model='clip_rn50', **kwargs)
    elif mode == 'simclr':
        return partial(PretrainedSSL, model='simclr', **kwargs)
    elif mode == 'swav':
        return partial(PretrainedSSL, model='swav', **kwargs)
    else:
        raise ValueError(f'Unkown mode={mode}.')


def is_colored_img(x):
    """Check if an image or batch of image is colored."""
    if x.shape[-3] not in [1, 3]:
        raise ValueError(f"x doesn't seem to be a (batch of) image as shape={x.shape}.")
    return x.shape[-3] == 3


def prediction_loss(Y_hat, y, is_classification=True, agg_over_tasks='mean'):
    """Compute the prediction loss for a task.

    Parameters
    ----------
    Y_hat : Tensor
        Predictions.

    y : Tensor
        Targets. Should be shape (batch_size, Y_dim, n_tasks), or (batch_size, Y_dim) if single task.

    is_classification : bool, optional
        Whether we are in a classification task, in which case we use log loss insteasd of (r)mse.

    agg_over_tasks : {"mean","sum","max","std",or Non}
        How to aggregate over tasks.
    """
    if is_classification:
        loss = F.cross_entropy(Y_hat, y.long(), reduction='none')
    else:
        loss = F.mse_loss(Y_hat, y, reduction='none')
    loss = atleast_ndim(loss, 3)
    batch_size, Y_dim, *_ = loss.shape
    loss = loss.view(batch_size, Y_dim, -1).mean(keepdim=False, dim=1)
    if agg_over_tasks == 'mean':
        loss = loss.mean(keepdim=True, dim=1)
    elif agg_over_tasks == 'max':
        loss = loss.max(keepdim=True, dim=1)[0]
    elif agg_over_tasks == 'sum':
        loss = loss.sum(keepdim=True, dim=1)
    elif agg_over_tasks == 'std':
        loss = loss.std(keepdim=True, dim=1)
    elif agg_over_tasks == 'min':
        loss = loss.min(keepdim=True, dim=1)[0]
    elif agg_over_tasks == 'median':
        loss = loss.median(keepdim=True, dim=1)[0]
    elif agg_over_tasks is None:
        loss = loss
    else:
        raise ValueError(f'Unkown agg_over_tasks={agg_over_tasks}.')
    return loss


class DirectDistortion(nn.Module):
    """Computes the loss using an direct variational bound (i.e. trying to predict an other variable).

    Parameters
    ----------
    z_dim : int
        Dimensionality of the representation.

    y_shape : tuple of int or int
        Shape of Y.

    arch : str, optional
        Architecture of the decoder. See `get_Architecture`.

    arch_kwargs : dict, optional
        Additional parameters to `get_Architecture`.

    dataset : str, optional
        Name of the dataset, used to undo normalization.

    is_normalized : bool, optional
        Whether the data is normalized. This is important to know whether needs to be unormalized
        when comparing in case you are reconstructing the input. Currently only works for colored
        images.

    data_mode : {"image","distribution"}, optional
        Mode of the data input.

    kwargs :
        Additional arguments to `prediction_loss`.
    """

    def __init__(self, z_dim, y_shape, arch=None, arch_kwargs=dict(), dataset=None, is_normalized=True, data_mode='image', name=None, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.is_img_out = data_mode == 'image'
        if arch is None:
            arch = 'cnn' if self.is_img_out else 'mlp'
        Decoder = get_Architecture(arch, **arch_kwargs)
        self.q_YlZ = Decoder(z_dim, y_shape)
        self.is_normalized = is_normalized
        self.kwargs = kwargs
        if self.is_normalized:
            if self.is_img_out:
                self.unnormalizer = UnNormalizer(self.dataset)
            else:
                raise NotImplementedError("Can curently only deal with normalized data if it's an image.")
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, z_hat, aux_target, _, __):
        """Compute the distortion.

        Parameters
        ----------
        z_hat : Tensor shape=[batch_size, z_dim]
            Reconstructed representations.

        aux_target : Tensor shape=[batch_size, *aux_shape]
            Targets to predict.

        Returns
        -------
        distortions : torch.Tensor shape=[batch_shape]
            Estimates distortion.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        Y_hat = self.q_YlZ(z_hat)
        if self.is_img_out:
            if is_colored_img(aux_target):
                if self.is_normalized:
                    aux_target = self.unnormalizer(aux_target)
                Y_hat = torch.sigmoid(Y_hat)
                neg_log_q_ylz = F.mse_loss(Y_hat, aux_target, reduction='none')
            else:
                neg_log_q_ylz = F.binary_cross_entropy_with_logits(Y_hat, aux_target, reduction='none')
                Y_hat = torch.sigmoid(Y_hat)
        else:
            neg_log_q_ylz = prediction_loss(Y_hat, aux_target, **self.kwargs)
        neg_log_q_ylz = einops.reduce(neg_log_q_ylz, 'b ... -> b', reduction='sum')
        logs = dict(H_q_TlZ=neg_log_q_ylz.mean() / math.log(BASE_LOG))
        other = dict()
        other['Y_hat'] = Y_hat[0].detach().cpu()
        other['Y'] = aux_target[0].detach().cpu()
        return neg_log_q_ylz, logs, other


class GatherFromGpus(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor)
        return tuple(gathered_tensor)

    @staticmethod
    def backward(ctx, *grads):
        tensor, = ctx.saved_tensors
        grad_out = torch.zeros_like(tensor)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out


gather_from_gpus = GatherFromGpus.apply


class ContrastiveDistortion(nn.Module):
    """Computes the loss using contrastive variational bound (i.e. with positive and negative examples).

    Notes
    -----
    - For the case of distribution, simply does NCE after sampling. This is not ideal and it would
    probably be better to derive a distributional InfoNCE.
    - parts of code taken from https://github.com/lucidrains/contrastive-learner

    Parameters
    ----------
    temperature : float, optional
        Temperature scaling in InfoNCE. Recommended less than 1.

    is_train_temperature : bool, optional
        Whether to treat the temperature as a parameter. Uses the same sceme as CLIP.
        If true then `temperature` becomes the lower bound on temperature.

    effective_batch_size : float, optional
        Effective batch size to use for estimating InfoNCE. Larger means that more variance but less bias,
        but if too large can become not a lower bound anymore. In [1] this is (m+1)/(2*alpha), where
        +1 and / 2 comes from the fact that talking about batch size rather than sample size.
        If `None` will use the standard unweighted `effective_batch_size`. Another good possibility
        is `effective_batch_size=len_dataset` which ensures that least bias while still lower bound.

    is_cosine : bool, optional
        Whether to use cosine similarity instead of dot products fot the logits of deterministic functions.
        This seems necessary for training, probably because if not norm of Z matters++ and then
        large loss in entropy bottleneck. Recommended True.

    is_already_featurized : bool, optional
        Whether the posivite examples are already featurized => no need to use p_ZlX again.
        In this case `p_ZlX` will be replaced by a placeholder distribution. Useful
        for clip, where the positive examples are text sentences that are already featuized.

    is_project : bool, optional
        Whether to use a porjection head. True seems to work better.

    project_kwargs : dict, optional
        Additional arguments to `Projector` in case `is_project`. Noe that is `out_shape` is <= 1
        it will be a percentage of z_dim.

    References
    ----------
    [1] Song, Jiaming, and Stefano Ermon. "Multi-label contrastive predictive coding." Advances in
    Neural Information Processing Systems 33 (2020).
    """

    def __init__(self, temperature=0.01, is_train_temperature=True, is_cosine=True, effective_batch_size=None, is_already_featurized=False, is_project=True, project_kwargs={'mode': 'mlp', 'out_shape': 128, 'in_shape': 128}):
        super().__init__()
        self.temperature = temperature
        self.is_train_temperature = is_train_temperature
        self.is_cosine = is_cosine
        self.effective_batch_size = effective_batch_size
        self.is_already_featurized = is_already_featurized
        self.is_project = is_project
        if self.is_project:
            if project_kwargs['out_shape'] <= 1:
                project_kwargs['out_shape'] = max(10, int(project_kwargs['in_shape'] * project_kwargs['out_shape']))
            Projector = get_Architecture(**project_kwargs)
            self.projector = Projector()
        else:
            self.projector = torch.nn.Identity()
        if self.is_train_temperature:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)
        if self.is_train_temperature:
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, z_hat, x_pos, _, parent):
        """Compute the distortion.

        Parameters
        ----------
        z_hat : Tensor shape=[batch_size, z_dim]
            Reconstructed representations.

        x_pos : Tensor shape=[batch_size, *x_shape]
            Other positive inputs., i.e., input on the same orbit.

        parent : LearnableCompressor, optional
            Parent module. This is useful for some distortion if they need access to other parts of the
            model.

        Returns
        -------
        distortions : torch.Tensor shape=[batch_shape]
            Estimates distortion.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        batch_size, z_dim = z_hat.shape
        logits = self.compute_logits(z_hat, x_pos, parent)
        hat_H_mlz, logs = self.compute_loss(logits)
        hat_H_mlz = (hat_H_mlz[:batch_size] + hat_H_mlz[batch_size:]) / 2
        other = dict()
        return hat_H_mlz, logs, other

    def compute_logits(self, z_hat, x_pos, parent):
        if self.is_already_featurized:
            z_pos_hat = x_pos
        else:
            z_pos_hat = parent(x_pos, is_features=True, is_dist=True)
        z = self.projector(z_hat)
        z_pos = self.projector(z_pos_hat)
        zs = torch.cat([z, z_pos], dim=0)
        if self.is_cosine:
            zs = F.normalize(zs, dim=1, p=2)
        logits = zs @ zs.T
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            list_logits = gather_from_gpus(logits)
            curr_gpu = torch.distributed.get_rank()
            curr_logits = list_logits[curr_gpu]
            other_logits = torch.cat(list_logits[:curr_gpu] + list_logits[curr_gpu + 1:], dim=-1)
            logits = torch.cat([curr_logits, other_logits], dim=1)
        return logits

    def compute_loss(self, logits):
        n_classes = logits.size(1)
        new_batch_size = logits.size(0)
        batch_size = new_batch_size // 2
        device = logits.device
        mask = ~torch.eye(new_batch_size, device=device).bool()
        n_to_add = n_classes - new_batch_size
        ones = torch.ones(new_batch_size, n_to_add, device=device).bool()
        mask = torch.cat([mask, ones], dim=1)
        n_classes -= 1
        logits = logits[mask].view(new_batch_size, n_classes)
        arange = torch.arange(batch_size, device=device)
        pos_idx = torch.cat([arange + batch_size - 1, arange], dim=0)
        if self.effective_batch_size is not None:
            effective_n_classes = 2 * self.effective_batch_size - 1
            to_mult = (effective_n_classes - 1) / (n_classes - 1)
            to_add = -math.log(to_mult)
            to_add = to_add * torch.ones_like(logits[:, 0:1])
            logits.scatter_add_(1, pos_idx.unsqueeze(1), to_add)
        else:
            effective_n_classes = n_classes
        if self.is_train_temperature:
            temperature = 1 / torch.clamp(self.logit_scale.exp(), max=1 / self.temperature)
        else:
            temperature = self.temperature
        logits = logits / temperature
        hat_H_m = math.log(effective_n_classes)
        hat_H_mlz = F.cross_entropy(logits, pos_idx, reduction='none')
        logs = dict(I_q_zm=(hat_H_m - hat_H_mlz.mean()) / math.log(BASE_LOG), hat_H_m=hat_H_m / math.log(BASE_LOG), n_negatives=n_classes)
        return hat_H_mlz, logs


class LossyZDistortion(nn.Module):
    """Computes the distortion by simply trying to reconstruct the given Z => Lossy compression of the
    representation without looking at X. Uses Mikowsky distance between z and z_hat.

    Parameters
    ----------
    p_norm : float, optional
        Which Lp norm to use for computing the distance.
    """

    def __init__(self, p_norm=1):
        super().__init__()
        self.distance = nn.PairwiseDistance(p=p_norm)

    def forward(self, z_hat, _, p_Zlx, __):
        """Compute the distortion.

        Parameters
        ----------
        z_hat : Tensor shape=[batch_size, z_dim]
            Reconstructed representations.

        p_Zlx : torch.Distribution batch_shape=[batch_size] event_shape=[z_dim]
            Encoded distribution of Z. Will take the mean to get the target z.

        Returns
        -------
        distortions : torch.Tensor shape=[batch_shape]
            Estimates distortion.

        logs : dict
            Additional values to log.

        other : dict
            Additional values to return.
        """
        dist = self.distance(z_hat, p_Zlx.base_dist.mean)
        logs = dict()
        other = dict()
        return dist, logs, other


class Delta(Distribution):
    """
    Degenerate discrete distribution (a single point).

    Parameters
    ----------
    v: torch.Tensor
        The single support element.

    log_density: torch.Tensor, optional
        An optional density for this Delta. This is useful to keep the class of :class:`Delta`
        distributions closed under differentiable transformation.

    event_dim: int, optional
        Optional event dimension.
    """
    has_rsample = True
    arg_constraints = {'loc': constraints.real, 'log_density': constraints.real}
    support = constraints.real

    def __init__(self, loc, log_density=0.0, validate_args=None):
        self.loc, self.log_density = broadcast_all(loc, log_density)
        if isinstance(loc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return torch.zeros_like(self.loc)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Delta, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.log_density = self.log_density.expand(batch_shape)
        super().__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = list(sample_shape) + list(self.loc.shape)
        return self.loc.expand(shape)

    def log_prob(self, x):
        log_prob = (x == self.loc).type(x.dtype).log()
        return log_prob + self.log_density


class Distributions:
    """Base class for distributions that can be instantiated with joint suff stat."""
    n_param = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_suff_param(cls, concat_suff_params, **kwargs):
        """Initialize the distribution using the concatenation of sufficient parameters (output of NN)."""
        suff_params = einops.rearrange(concat_suff_params, 'b (z p) -> b z p', p=cls.n_param).unbind(-1)
        suff_params = cls.preprocess_suff_params(*suff_params)
        return cls(*suff_params, **kwargs)

    @classmethod
    def preprocess_suff_params(cls, *suff_params):
        """Preprocesses parameters outputed from network (usually to satisfy some constraints)."""
        return suff_params

    def detach(self, is_grad_flow=False):
        """
        Detaches all the parameters. With optional `is_grad_flow` that would ensure pytorch does
        not complain about no grad (by setting grad to 0.
        """
        raise NotImplementedError()


class Deterministic(Distributions, Independent):
    """Delta function distribution (i.e. no stochasticity)."""
    n_param = 1

    def __init__(self, param):
        super().__init__(Delta(param), 1)

    def detach(self, is_grad_flow=False):
        loc = self.base_dist.loc.detach()
        if is_grad_flow:
            loc = loc + 0 * self.base_dist.loc
        return Deterministic(loc)


class DiagGaussian(Distributions, Independent):
    """Gaussian with diagonal covariance."""
    n_param = 2
    min_std = 1e-05

    def __init__(self, diag_loc, diag_scale):
        super().__init__(Normal(diag_loc, diag_scale), 1)
        self.min_std

    @classmethod
    def preprocess_suff_params(cls, diag_loc, diag_log_var):
        diag_scale = F.softplus(diag_log_var) + cls.min_std
        return diag_loc, diag_scale

    def detach(self, is_grad_flow=False):
        loc = self.base_dist.loc.detach()
        scale = self.base_dist.scale.detach()
        if is_grad_flow:
            loc = loc + 0 * self.base_dist.loc
            scale = scale + 0 * self.base_dist.scale
        return DiagGaussian(loc, scale)


class CondDist(nn.Module):
    """Return the (uninstantiated) correct CondDist.

    Parameters
    ----------
    in_shape : tuple of int

    out_dim : int

    Architecture : nn.Module
        Module to be instantiated by `Architecture(in_shape, out_dim)`.

    family : {"gaussian","uniform"}
        Family of the distribution (after conditioning), this can be easily extandable to any
        distribution in `torch.distribution`.

    kwargs :
        Additional arguments to the `Family`.
    """

    def __init__(self, in_shape, out_dim, Architecture, family, **kwargs):
        super().__init__()
        if family == 'diaggaussian':
            self.Family = DiagGaussian
        elif family == 'deterministic':
            self.Family = Deterministic
        else:
            raise ValueError(f'Unkown family={family}.')
        self.in_shape = in_shape
        self.out_dim = out_dim
        self.kwargs = kwargs
        self.mapper = Architecture(in_shape, out_dim * self.Family.n_param)
        self.reset_parameters()

    def forward(self, x):
        """Compute the distribution conditioned on `X`.

        Parameters
        ----------
        Xx: torch.Tensor, shape: [batch_size, *in_shape]
            Input on which to condition the output distribution.

        Return
        ------
        p(.|x) : torch.Distribution, batch shape: [batch_size] event shape: [out_dim]
        """
        suff_param = self.mapper(x)
        p__lx = self.Family.from_suff_param(suff_param, **self.kwargs)
        return p__lx

    def reset_parameters(self):
        weights_init(self)


class MarginalUnitGaussian(nn.Module):
    """Mean 0 covariance 1 Gaussian."""

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.register_buffer('loc', torch.as_tensor([0.0] * self.out_dim))
        self.register_buffer('scale', torch.as_tensor([1.0] * self.out_dim))

    def forward(self):
        return Independent(Normal(self.loc, self.scale), 1)


class Normalizer(torch.nn.Module):

    def __init__(self, dataset, is_raise=True):
        super().__init__()
        self.dataset = dataset.lower()
        try:
            self.normalizer = transform_lib.Normalize(mean=MEANS[self.dataset], std=STDS[self.dataset])
        except KeyError:
            if is_raise:
                raise KeyError(f"dataset={self.dataset} wasn't found in MEANS={MEANS.keys()} orSTDS={STDS.keys()}. Please add mean and std.")
            else:
                self.normalizer = None

    def forward(self, x):
        if self.normalizer is None:
            return x
        return self.normalizer(x)


class Timer:
    """Timer context manager"""

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start


class OnlineEvaluator(torch.nn.Module):
    """
    Attaches MLP/linear predictor for evaluating the quality of a representation as usual in self-supervised.

    Notes
    -----
    -  generalizes `pl_bolts.callbacks.ssl_online.SSLOnlineEvaluator` for multilabel clf and regression
    and does not use a callback as pytorch lightning doesn't work well with trainable callbacks.

    Parameters
    ----------
    in_dim : int
        Input dimension.

    y_shape : tuple of in
        Shape of the output

    Architecture : nn.Module
        Module to be instantiated by `Architecture(in_shape, out_dim)`.

    is_classification : bool, optional
        Whether or not the task is a classification one.

    kwargs:
        Additional kwargs to `prediction_loss`.
    """

    def __init__(self, in_dim, out_dim, Architecture, is_classification=True, **kwargs):
        super().__init__()
        self.model = Architecture(in_dim, out_dim)
        self.is_classification = is_classification
        self.kwargs = kwargs

    def aux_parameters(self):
        """Return iterator over parameters."""
        for m in self.children():
            for p in m.parameters():
                yield p

    def forward(self, batch, encoder):
        x, y = batch
        if isinstance(y, (tuple, list)):
            y = y[0]
        with torch.no_grad():
            z = encoder(x, is_features=True)
        z = z.detach()
        with Timer() as inference_timer:
            Y_hat = self.model(z)
        loss = prediction_loss(Y_hat, y, self.is_classification, **self.kwargs)
        loss = loss.mean()
        logs = dict(online_loss=loss, inference_time=inference_timer.duration)
        if self.is_classification:
            logs['online_acc'] = accuracy(Y_hat.argmax(dim=-1), y)
            logs['online_err'] = 1 - logs['online_acc']
        return loss, logs


def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def mean(array):
    """Take mean of array like."""
    return sum(array) / len(array)


def to_numpy(X):
    """Convert tensors,list,tuples,dataframes to numpy arrays."""
    if isinstance(X, np.ndarray):
        return X
    if hasattr(X, 'iloc'):
        return X.values
    if isinstance(X, (tuple, list)):
        return np.array(X)
    if not isinstance(X, (torch.Tensor, PackedSequence)):
        raise TypeError(f'Cannot convert {type(X)} to a numpy array.')
    if X.is_cuda:
        X = X.cpu()
    if X.requires_grad:
        X = X.detach()
    return X.numpy()


def kl_divergence(p, q, z_samples=None, is_lower_var=False):
    """Computes KL[p||q], analytically if possible but with MC."""
    try:
        kl_pq = torch.distributions.kl_divergence(p, q)
    except NotImplementedError:
        log_q = q.log_prob(z_samples)
        log_p = p.log_prob(z_samples)
        if is_lower_var:
            log_r = log_q - log_p
            kl_pq = log_r.exp() - 1 - log_r
        else:
            kl_pq = log_p - log_q
    return kl_pq


class OrderedSet(MutableSet):
    """A set that preserves insertion order by internally using a dict."""

    def __init__(self, iterable):
        self._d = dict.fromkeys(iterable)

    def add(self, x):
        self._d[x] = None

    def discard(self, x):
        self._d.pop(x)

    def __contains__(self, x):
        return self._d.__contains__(x)

    def __len__(self):
        return self._d.__len__()

    def __iter__(self):
        return self._d.__iter__()


def _torch_random_choice(x, pdf):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    assert len(x.shape) == 1, 'Random choice array must be 1d, your got shape {}'.format(x.shape)
    assert np.all(np.array(pdf) >= 0), 'PDF need not to be normalized but counts must be positive.'
    x_size = torch.tensor(x.size())
    pdf = pdf / np.sum(pdf)
    cdf = torch.tensor(np.cumsum(pdf))
    idx = x_size - torch.sum(torch.rand((1,)) < cdf)
    return int(x[idx].numpy()[0])


class EquivariantTransformation(torch.nn.Module):
    """Abstraction layer for augmentations on the label and the input. If sampled augmentation in invariant range:
        augments image and keeps the orignal target, but if augmentation in the equivariant range, change label with
        probability p.

        Corresponding augmentation parameter range:

        | left equivariant range |    invariant range           |     right equivariant range               |

        If the image is torch Tensor, it is expected to have [..., H, W] shape, where ... means an arbitrary number of
        leading dimensions. This class refers to torch.transforms.RandomRotation, args and kwargs apply.

        Args:
            p (float): Probability of changing the label if rotation is in the equivariant range.
            num_classes (integer): Size of the sample space for the label.
        """

    def __init__(self, p=1.0, num_classes=10):
        super().__init__()
        self.p = torch.tensor([p])
        self.num_classes = num_classes

    @property
    def pdf(self):
        """ Probability of augmentation being in the [left equivariant range, invariant range, right equivariant range].
            For uniform sampling should  be proportional to size of the respective range.
        """
        return self._pdf

    @pdf.setter
    def pdf(self, value):
        self._pdf = value

    def forward(self, data):
        """
                Args:
                    data (PIL Image or Tensor, int or Tensor): (image, label) to be augmented.

                Returns:
                    PIL Image or Tensor, int or Tensor: Rotated image and possibly shuffled target.
        """
        img, label = data
        aug_idx = _torch_random_choice([0, 1, 2], pdf=self.pdf)
        if aug_idx == 1:
            img = self.invariant_aug(img)
        elif aug_idx == 0:
            img = self.equivariant_aug_left(img)
            if torch.rand((1,)) < self.p:
                label = torch.randint(high=self.num_classes, size=(1,)).numpy()[0]
        else:
            img = self.equivariant_aug_right(img)
            if torch.rand((1,)) < self.p:
                label = torch.randint(high=self.num_classes, size=(1,)).numpy()[0]
        return img, label


class EquivariantRandomResizedCrop(EquivariantTransformation):
    """EquivariantTransformation based on randomly resized crops.

        If the image is torch Tensor, it is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

        A crop of random size (default: of 0.08 to 1.0) of the original size and a random
        aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
        is finally resized to given size.
        This is popularly used to train the Inception networks.

        Args:
            size (int or sequence): expected output size of each edge. If size is an
                int instead of sequence like (h, w), a square output size ``(size, size)`` is
                made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
            invariant_scale (tuple of float): scale range of the cropped image before resizing, relatively to the
                origin image. This should contain the interval invariant_scale.
            equivariant_scale (tuple of float): scale range of the cropped image before resizing, relatively to the
                origin image.
        """

    def __init__(self, size, invariant_scale, equivariant_scale, p=1.0, num_classes=10, *args, **kwargs):
        super().__init__(p=p, num_classes=num_classes)
        if not isinstance(invariant_scale, Sequence):
            raise TypeError('Scale should be a sequence')
        if not isinstance(equivariant_scale, Sequence):
            raise TypeError('Scale should be a sequence')
        self.invariant_aug = RandomResizedCrop(*args, size=size, scale=invariant_scale, **kwargs)
        assert equivariant_scale[0] <= invariant_scale[0], 'Problem with data augmentations: Range of equivariant scale should entail invariant scale.'
        assert invariant_scale[1] <= equivariant_scale[1], 'Problem with data augmentations: Range of equivariant scale should entail invariant scale.'
        self.equivariant_aug_left = RandomResizedCrop(*args, size=size, scale=(equivariant_scale[0], invariant_scale[0]), **kwargs)
        self.equivariant_aug_right = RandomResizedCrop(*args, size=size, scale=(invariant_scale[1], equivariant_scale[1]), **kwargs)
        self.pdf = [invariant_scale[0] - equivariant_scale[0], invariant_scale[1] - invariant_scale[0], equivariant_scale[1] - invariant_scale[1]]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CNN,
     lambda: ([], {'in_shape': [4, 4, 4], 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MarginalUnitGaussian,
     lambda: ([], {'out_dim': 4}),
     lambda: ([], {}),
     False),
]

class Test_YannDubs_lossyless(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

