import sys
_module = sys.modules[__name__]
del sys
celldetection = _module
__meta__ = _module
callbacks = _module
dropout = _module
keepalive = _module
data = _module
cpn = _module
datasets = _module
bbbc038 = _module
bbbc039 = _module
bbbc041 = _module
generic = _module
synth = _module
instance_eval = _module
misc = _module
segmentation = _module
toydata = _module
transforms = _module
models = _module
commons = _module
convnext = _module
convnextv2 = _module
cpn = _module
densenet = _module
features = _module
filters = _module
fpn = _module
hosted = _module
inference = _module
lightning_base = _module
lightning_cpn = _module
loss = _module
mamba = _module
manet = _module
mobilenetv3 = _module
normalization = _module
ppm = _module
resnet = _module
smp = _module
timmodels = _module
unet = _module
mpi = _module
ops = _module
boxes = _module
commons = _module
cpn = _module
draw = _module
features = _module
loss = _module
normalization = _module
optim = _module
lr_scheduler = _module
util = _module
logging = _module
schedule = _module
shm_cache = _module
timer = _module
util = _module
visualization = _module
cmaps = _module
images = _module
celldetection_scripts = _module
cpn_inference = _module
conf = _module
hubconf = _module
setup = _module
test_cpn_inference = _module

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


from torch.nn.modules.dropout import _DropoutNd


from typing import Any


import warnings


import numpy as np


import torch


from collections import OrderedDict


from scipy.ndimage import distance_transform_edt


from itertools import product


from itertools import chain


from typing import Union


from warnings import warn


import pandas as pd


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from torch import tanh


from torch import sigmoid


from torchvision import transforms as trans


from torch.nn.common_types import _size_2_t


from typing import Type


from functools import partial


from torchvision.models.convnext import CNBlockConfig


from torchvision.models import convnext as cnx


from typing import List


from typing import Optional


from typing import Callable


from typing import Sequence


from torch import nn


from torchvision.ops import misc


from torchvision.ops import Permute


from torchvision.ops.stochastic_depth import StochasticDepth


from torch.hub import load_state_dict_from_url


from typing import Dict


from torchvision.models.densenet import _DenseLayer


from torchvision.models.densenet import _DenseBlock


from torchvision.models.densenet import _Transition


from torchvision.models import densenet


from torch.nn import functional as F


from torchvision.models.detection import backbone_utils


from torchvision.models._utils import IntermediateLayerGetter


from torchvision.ops import feature_pyramid_network


from torchvision.ops.feature_pyramid_network import ExtraFPNBlock as _ExtraFPNBlock


from typing import Tuple


import copy


import matplotlib.pyplot as plt


from torch.distributed import is_available


from torch.distributed import all_gather_object


from torch.distributed import get_world_size


from torch.distributed import is_initialized


from collections import ChainMap


from torchvision.ops.boxes import remove_small_boxes


from torchvision.ops.boxes import nms


from torch.nn.modules.loss import _Loss


from torchvision.ops.focal_loss import sigmoid_focal_loss


from torch.cuda.amp import autocast


from torch.nn.functional import softmax


from torchvision.models.mobilenetv3 import InvertedResidualConfig


from torchvision.models.mobilenetv3 import InvertedResidual


from torchvision.models.mobilenetv3 import _mobilenet_v3_conf


from torchvision.models.segmentation.deeplabv3 import ASPP


from torchvision.models import resnet as tvr


import re


from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


from torchvision.ops.feature_pyramid_network import ExtraFPNBlock


import torchvision.ops.boxes as bx


from itertools import combinations_with_replacement


from torch.optim.lr_scheduler import MultiplicativeLR


from torch.optim.lr_scheduler import SequentialLR as _SequentialLR


from torch.optim.lr_scheduler import ReduceLROnPlateau as _ReduceLROnPlateau


from torch.optim import Optimizer


from torch.nn import Module


from torch import optim


import inspect


from time import time


from typing import Dict as TDict


from typing import Iterator


from typing import Iterable


from inspect import currentframe


from inspect import signature


from functools import wraps


from matplotlib import pyplot as plt


from matplotlib import patches


from matplotlib.image import AxesImage


import matplotlib.animation as animation


import matplotlib.patheffects as path_effects


from matplotlib.axes import SubplotBase


from torch import squeeze


from torch import as_tensor


from torchvision.utils import make_grid


from torch.utils.data import DataLoader


from torch.distributed import get_rank


from torch.distributed import gather_object


from torch import device as _device


def get_nd_conv(dim: 'int'):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'Conv%dd' % dim)


def replace_ndim(s: 'Union[str, type, Callable]', dim: 'int', allowed_dims=(1, 2, 3)):
    """Replace ndim.

    Replaces dimension statement of ``string``or ``type``.

    Notes:
        - Dimensions are expected to be at the end of the type name.
        - If there is no dimension statement, nothing is changed.

    Examples:
        >>> replace_ndim('BatchNorm2d', 3)
        'BatchNorm3d'
        >>> replace_ndim(nn.BatchNorm2d, 3)
        torch.nn.modules.batchnorm.BatchNorm3d
        >>> replace_ndim(nn.GroupNorm, 3)
        torch.nn.modules.normalization.GroupNorm
        >>> replace_ndim(F.conv2d, 3)
        <function torch._VariableFunctionsClass.conv3d>

    Args:
        s: String or type.
        dim: Desired dimension.
        allowed_dims: Allowed dimensions to look for.

    Returns:
        Input with replaced dimension.
    """
    if isinstance(s, str) and dim in allowed_dims:
        return re.sub(f'[1-3]d$', f'{int(dim)}d', s)
    elif isinstance(s, type) or callable(s):
        return getattr(sys.modules[s.__module__], replace_ndim(s.__name__, dim))
    return s


def lookup_nn(item: 'str', *a, src=None, call=True, inplace=True, nd=None, **kw):
    """

    Examples:
        >>> lookup_nn('batchnorm2d', 32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn(torch.nn.BatchNorm2d, 32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn('batchnorm2d', num_features=32)
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        >>> lookup_nn('tanh')
            Tanh()
        >>> lookup_nn('tanh', call=False)
            torch.nn.modules.activation.Tanh
        >>> lookup_nn('relu')
            ReLU(inplace=True)
        >>> lookup_nn('relu', inplace=False)
            ReLU()
        >>> # Dict notation to contain all keyword arguments for calling in `item`. Always called once.
        ... lookup_nn(dict(relu=dict(inplace=True)), call=False)
            ReLU(inplace=True)
        >>> lookup_nn({'NormProxy': {'norm': 'GroupNorm', 'num_groups': 32}}, call=False)
            NormProxy(GroupNorm, kwargs={'num_groups': 32})
        >>> lookup_nn({'NormProxy': {'norm': 'GroupNorm', 'num_groups': 32}}, 32, call=True)
            GroupNorm(32, 32, eps=1e-05, affine=True)

    Args:
        item: Lookup item. None is equivalent to `identity`.
        *a: Arguments passed to item if called.
        src: Lookup source.
        call: Whether to call item.
        inplace: Default setting for items that take an `inplace` argument when called.
            As default is True, `lookup_nn('relu')` returns a ReLu instance with `inplace=True`.
        nd: If set, replace dimension statement (e.g. '2d' in nn.Conv2d) with ``nd``.
        **kw: Keyword arguments passed to item when it is called.

    Returns:
        Looked up item.
    """
    if src is None:
        src = nn, models
    if isinstance(item, tuple):
        if len(item) == 1:
            item, = item
        elif len(item) == 2:
            item, _kw = item
            kw.update(_kw)
        else:
            raise ValueError('Allowed formats for item: (item,) or (item, kwargs).')
    if item is None:
        v = nn.Identity
    elif isinstance(item, str):
        l_item = item.lower()
        if nd is not None:
            l_item = replace_ndim(l_item, nd)
        if not isinstance(src, (list, tuple)):
            src = src,
        v = None
        for src_ in src:
            try:
                v = next(getattr(src_, i) for i in dir(src_) if i.lower() == l_item)
            except StopIteration:
                continue
            break
        if v is None:
            raise ValueError(f'Could not find `{item}` in {src}.')
    elif isinstance(item, nn.Module):
        return item
    elif isinstance(item, dict):
        assert len(item) == 1
        key, = item
        val = item[key]
        assert isinstance(val, dict)
        cls = lookup_nn(key, src=src, call=False, inplace=inplace, nd=nd)
        if issubclass(cls, nn.modules.loss._WeightedLoss):
            if 'weight' in val and not isinstance(val['weight'], Tensor):
                val['weight'] = torch.as_tensor(val['weight'])
        v = cls(**val)
    elif isinstance(item, type) and nd is not None:
        v = replace_ndim(item, nd)
    else:
        v = item
    if call:
        kwargs = {'inplace': inplace} if 'inplace' in inspect.getfullargspec(v).args else {}
        kwargs.update(kw)
        v = v(*a, **kwargs)
    return v


class ConvNorm(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, norm_layer=nn.BatchNorm2d, nd=2, **kwargs):
        """ConvNorm.

        Just a convolution and a normalization layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            padding: Padding.
            stride: Stride.
            norm_layer: Normalization layer (e.g. ``nn.BatchNorm2d``).
            **kwargs: Additional keyword arguments.
        """
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm_layer, nd=nd, call=False)
        super().__init__(Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs), Norm(out_channels))


class ConvNormRelu(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, norm_layer=nn.BatchNorm2d, activation='relu', nd=2, **kwargs):
        """ConvNormReLU.

        Just a convolution, normalization layer and an activation.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            padding: Padding.
            stride: Stride.
            norm_layer: Normalization layer (e.g. ``nn.BatchNorm2d``).
            activation: Activation function. (e.g. ``nn.ReLU``, ``'relu'``)
            **kwargs: Additional keyword arguments.
        """
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm_layer, nd=nd, call=False)
        super().__init__(Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs), Norm(out_channels), lookup_nn(activation))


class TwoConvNormRelu(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, mid_channels=None, norm_layer=nn.BatchNorm2d, activation='relu', nd=2, **kwargs):
        """TwoConvNormReLU.

        A sequence of conv, norm, activation, conv, norm, activation.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size.
            padding: Padding.
            stride: Stride.
            mid_channels: Mid-channels. Default: Same as ``out_channels``.
            norm_layer: Normalization layer (e.g. ``nn.BatchNorm2d``).
            activation: Activation function. (e.g. ``nn.ReLU``, ``'relu'``)
            **kwargs: Additional keyword arguments.
        """
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm_layer, nd=nd, call=False)
        if mid_channels is None:
            mid_channels = out_channels
        super().__init__(Conv(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs), Norm(mid_channels), lookup_nn(activation), Conv(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, **kwargs), Norm(out_channels), lookup_nn(activation))


class TwoConvNormLeaky(TwoConvNormRelu):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, mid_channels=None, norm_layer=nn.BatchNorm2d, nd=2, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, mid_channels=mid_channels, norm_layer=norm_layer, activation='leakyrelu', nd=nd, **kwargs)


class ScaledX(nn.Module):

    def __init__(self, fn, factor, shift=0.0):
        super().__init__()
        self.factor = factor
        self.shift = shift
        self.fn = fn

    def forward(self, inputs: 'Tensor') ->Tensor:
        return self.fn(inputs) * self.factor + self.shift

    def extra_repr(self) ->str:
        return 'factor={}, shift={}'.format(self.factor, self.shift)


class ScaledTanh(ScaledX):

    def __init__(self, factor, shift=0.0):
        """Scaled Tanh.

        Computes the scaled and shifted hyperbolic tangent:

        .. math:: tanh(x) * factor + shift

        Args:
            factor: Scaling factor.
            shift: Shifting constant.
        """
        super().__init__(tanh, factor, shift)


class ScaledSigmoid(ScaledX):

    def __init__(self, factor, shift=0.0):
        """Scaled Sigmoid.

        Computes the scaled and shifted sigmoid:

        .. math:: sigmoid(x) * factor + shift

        Args:
            factor: Scaling factor.
            shift: Shifting constant.
        """
        super().__init__(sigmoid, factor, shift)


class _ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, block, activation='ReLU', stride=1, downsample=None, norm_layer='BatchNorm2d', nd=2) ->None:
        """ResBlock.

        Typical ResBlock with variable kernel size and an included mapping of the identity to correct dimensions.

        References:
            https://arxiv.org/abs/1512.03385

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            padding: Padding.
            norm_layer: Norm layer.
            activation: Activation.
            stride: Stride.
            downsample: Downsample module that maps identity to correct dimensions. Default is an optionally strided
                1x1 Conv2d with BatchNorm2d, as per He et al. (2015) (`3.3. Network Architectures`, `Residual Network`,
                "option (B)").
            nd: Number of spatial dimensions.
        """
        super().__init__()
        downsample = downsample or partial(ConvNorm, nd=nd, norm_layer=norm_layer)
        if in_channels != out_channels or stride != 1:
            self.downsample = downsample(in_channels, out_channels, 1, stride=stride, bias=False, padding=0)
        else:
            self.downsample = nn.Identity()
        self.block = block
        self.activation = lookup_nn(activation)

    def forward(self, x: 'Tensor') ->Tensor:
        identity = self.downsample(x)
        out = self.block(x)
        out += identity
        return self.activation(out)


class ResBlock(_ResBlock):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, norm_layer='BatchNorm2d', activation='ReLU', stride=1, downsample=None, nd=2, **kwargs) ->None:
        """ResBlock.

        Typical ResBlock with variable kernel size and an included mapping of the identity to correct dimensions.

        References:
            - https://doi.org/10.1109/CVPR.2016.90

        Notes:
            - Similar to ``torchvision.models.resnet.BasicBlock``, with different interface and defaults.
            - Consistent with standard signature ``in_channels, out_channels, kernel_size, ...``.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            padding: Padding.
            norm_layer: Norm layer.
            activation: Activation.
            stride: Stride.
            downsample: Downsample module that maps identity to correct dimensions. Default is an optionally strided
                1x1 Conv2d with BatchNorm2d, as per He et al. (2015) (`3.3. Network Architectures`, `Residual Network`,
                "option (B)").
            **kwargs: Keyword arguments for Conv2d layers.
        """
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm_layer, nd=nd, call=False)
        super().__init__(in_channels, out_channels, block=nn.Sequential(Conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=stride, **kwargs), Norm(out_channels), lookup_nn(activation), Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False, **kwargs), Norm(out_channels)), activation=activation, stride=stride, downsample=downsample, nd=nd, norm_layer=norm_layer)


class BottleneckBlock(_ResBlock):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, mid_channels=None, compression=4, base_channels=64, norm_layer='BatchNorm2d', activation='ReLU', stride=1, downsample=None, nd=2, **kwargs) ->None:
        """Bottleneck Block.

        Typical Bottleneck Block with variable kernel size and an included mapping of the identity to correct
        dimensions.

        References:
            - https://doi.org/10.1109/CVPR.2016.90
            - https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch

        Notes:
            - Similar to ``torchvision.models.resnet.Bottleneck``, with different interface and defaults.
            - Consistent with standard signature ``in_channels, out_channels, kernel_size, ...``.
            - Stride handled in bottleneck.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Kernel size.
            padding: Padding.
            mid_channels:
            compression: Compression rate of the bottleneck. The default 4 compresses 256 channels to 64=256/4.
            base_channels: Minimum number of ``mid_channels``.
            norm_layer: Norm layer.
            activation: Activation.
            stride: Stride.
            downsample: Downsample module that maps identity to correct dimensions. Default is an optionally strided
                1x1 Conv2d with BatchNorm2d, as per He et al. (2015) (`3.3. Network Architectures`, `Residual Network`,
                "option (B)").
            **kwargs: Keyword arguments for Conv2d layers.
        """
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm_layer, nd=nd, call=False)
        mid_channels = mid_channels or np.max([base_channels, out_channels // compression, in_channels // compression])
        super().__init__(in_channels, out_channels, block=nn.Sequential(Conv(in_channels, mid_channels, kernel_size=1, padding=0, bias=False, **kwargs), Norm(mid_channels), lookup_nn(activation), Conv(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False, stride=stride, **kwargs), Norm(mid_channels), lookup_nn(activation), Conv(mid_channels, out_channels, kernel_size=1, padding=0, bias=False, **kwargs), Norm(out_channels)), activation=activation, stride=stride, downsample=downsample)


def tensor_to(inputs: 'Union[list, tuple, dict, Tensor]', *args, **kwargs):
    """Tensor to device/dtype/other.

    Recursively calls ``tensor.to(*args, **kwargs)`` for all ``Tensors`` in ``inputs``.

    Notes:
        - Works recursively.
        - Non-Tensor items are not altered.

    Args:
        inputs: Tensor, list, tuple or dict. Non-Tensor objects are ignored. Tensors are substituted by result of
            ``tensor.to(*args, **kwargs)`` call.
        *args: Arguments. See docstring of ``torch.Tensor.to``.
        **kwargs: Keyword arguments. See docstring of ``torch.Tensor.to``.

    Returns:
        Inputs with Tensors replaced by ``tensor.to(*args, **kwargs)``.
    """
    if isinstance(inputs, Tensor):
        inputs = inputs
    elif isinstance(inputs, (dict, OrderedDict)):
        inputs = {k: tensor_to(b, *args, **kwargs) for k, b in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        inputs = type(inputs)([tensor_to(b, *args, **kwargs) for b in inputs])
    return inputs


class NoAmp(nn.Module):

    def __init__(self, module: 'Type[nn.Module]'):
        """No AMP.

        Wrap a ``Module`` object and disable ``torch.cuda.amp.autocast`` during forward pass if it is enabled.

        Examples:
            >>> import celldetection as cd
            ... model = cd.models.CpnU22(1)
            ... # Wrap all ReadOut modules in model with NoAmp, thus disabling autocast for those modules
            ... cd.wrap_module_(model, cd.models.ReadOut, cd.models.NoAmp)

        Args:
            module: Module.
        """
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        if torch.is_autocast_enabled():
            with torch.amp.autocast(enabled=False):
                result = self.module(*tensor_to(args, torch.float32), **tensor_to(kwargs, torch.float32))
        else:
            result = self.module(*args, **kwargs)
        return result


class ReadOut(nn.Module):

    def __init__(self, channels_in, channels_out, kernel_size=3, padding=1, activation='relu', norm='batchnorm2d', final_activation=None, dropout=0.1, channels_mid=None, stride=1, nd=2, attention=None):
        super().__init__()
        Conv = get_nd_conv(nd)
        Norm = lookup_nn(norm, nd=nd, call=False)
        Dropout = lookup_nn(nn.Dropout2d, nd=nd, call=False)
        self.channels_out = channels_out
        if channels_mid is None:
            channels_mid = channels_in
        self.attention = None
        if attention is not None:
            if isinstance(attention, dict):
                attention_kwargs, = list(attention.values())
                attention, = list(attention.keys())
            else:
                attention_kwargs = {}
            self.attention = lookup_nn(attention, nd=nd, call=False)(channels_in, **attention_kwargs)
        self.block = nn.Sequential(Conv(channels_in, channels_mid, kernel_size, padding=padding, stride=stride), Norm(channels_mid), lookup_nn(activation), Dropout(p=dropout) if dropout else nn.Identity(), Conv(channels_mid, channels_out, 1))
        if final_activation is ...:
            self.activation = lookup_nn(activation)
        else:
            self.activation = lookup_nn(final_activation)

    def forward(self, x):
        if self.attention is not None:
            x = self.attention(x)
        out = self.block(x)
        return self.activation(out)


def split_spatially(x, size):
    """Split spatially.

    Splits spatial dimensions of Tensor ``x`` into patches of given ``size`` and adds the patches
    to the batch dimension.

    Args:
        x: Input Tensor[n, c, h, w, ...].
        size: Patch size of the splits.

    Returns:
        Tensor[n * h//height * w//width, c, height, width].
    """
    n, c = x.shape[:2]
    spatial = x.shape[2:]
    nd = len(spatial)
    assert len(spatial) == len(size)
    v = n, c
    for cur, new in zip(spatial, size):
        v += cur // new, new
    perm = (0,) + tuple(range(2, nd * 2 + 1, 2)) + tuple(range(1, nd * 3, 2))
    return x.view(v).permute(perm).reshape((-1, c) + tuple(size))


class SpatialSplit(nn.Module):

    def __init__(self, height, width=None):
        """Spatial split.

        Splits spatial dimensions of input Tensor into patches of size ``(height, width)`` and adds the patches
        to the batch dimension.

        Args:
            height: Patch height.
            width: Patch width.
        """
        super().__init__()
        self.height = height
        self.width = width or height

    def forward(self, x):
        return split_spatially(x, self.height, self.width)


def minibatch_std_layer(x, channels=1, group_channels=None, epsilon=1e-08):
    """Minibatch standard deviation layer.

    The minibatch standard deviation layer first splits the batch dimension into slices of size ``group_channels``.
    The channel dimension is split into ``channels`` slices. For the groups the standard deviation is calculated and
    averaged over spatial dimensions and channel slice depth. The result is broadcasted to the spatial dimensions,
    repeated for the batch dimension and then concatenated to the channel dimension of ``x``.

    References:
        - https://arxiv.org/pdf/1710.10196.pdf

    Args:
        x: Input Tensor[n, c, h, w].
        channels: Number of averaged standard deviation channels.
        group_channels: Number of channels per group. Default: batch size.
        epsilon: Epsilon.

    Returns:
        Tensor[n, c + channels, h, w].
    """
    n, c, h, w = x.shape
    gc = min(group_channels or n, n)
    cc, g = c // channels, n // gc
    y = x.view(gc, g, channels, cc, h, w)
    y = y.var(0, False).add(epsilon).sqrt().mean([2, 3, 4], True).squeeze(-1).repeat(gc, 1, h, w)
    return torch.cat([x, y], 1)


class MinibatchStdLayer(torch.nn.Module):

    def __init__(self, channels=1, group_channels=None, epsilon=1e-08):
        """Minibatch standard deviation layer.

        The minibatch standard deviation layer first splits the batch dimension into slices of size ``group_channels``.
        The channel dimension is split into ``channels`` slices. For the groups the standard deviation is calculated and
        averaged over spatial dimensions and channel slice depth. The result is broadcasted to the spatial dimensions,
        repeated for the batch dimension and then concatenated to the channel dimension of ``x``.

        References:
            - https://arxiv.org/pdf/1710.10196.pdf

        Args:
            channels: Number of averaged standard deviation channels.
            group_channels: Number of channels per group. Default: batch size.
            epsilon: Epsilon.
        """
        super().__init__()
        self.channels = channels
        self.group_channels = group_channels
        self.epsilon = epsilon

    def forward(self, x):
        return minibatch_std_layer(x, self.channels, self.group_channels, epsilon=self.epsilon)

    def extra_repr(self) ->str:
        return f'channels={self.channels}, group_channels={self.group_channels}'


class _AdditiveNoise(nn.Module):

    def __init__(self, in_channels, noise_channels=1, mean=0.0, std=1.0, weighted=False, nd=2):
        super().__init__()
        self.noise_channels = noise_channels
        self.in_channels = in_channels
        self.reps = (1, self.in_channels // self.noise_channels) + (1,) * nd
        self.weighted = weighted
        self.weight = nn.Parameter(torch.zeros((1, in_channels) + (1,) * nd)) if weighted else 1.0
        self.constant = False
        self.mean = mean
        self.std = std
        self._constant = None

    def sample_noise(self, shape, device, dtype):
        return torch.randn(shape, device=device, dtype=dtype) * self.std + self.mean

    def forward(self, x):
        shape = x.shape
        constant = getattr(self, 'constant', False)
        _constant = getattr(self, '_constant', None)
        if constant and _constant is None or not constant:
            noise = self.sample_noise((shape[0], self.noise_channels) + shape[2:], x.device, x.dtype)
            if constant and _constant is None:
                self._constant = noise
        else:
            noise = _constant
        return x + noise.repeat(self.reps) * self.weight

    def extra_repr(self):
        s = f'in_channels={self.in_channels}, noise_channels={self.noise_channels}, mean={self.mean}, std={self.std}, weighted={self.weighted}'
        if getattr(self, 'constant', False):
            s += ', constant=True'
        return s


class AdditiveNoise2d(_AdditiveNoise):

    def __init__(self, in_channels, noise_channels=1, weighted=True, **kwargs):
        super().__init__(in_channels=in_channels, noise_channels=noise_channels, weighted=weighted, nd=2, **kwargs)


class AdditiveNoise3d(_AdditiveNoise):

    def __init__(self, in_channels, noise_channels=1, weighted=True, **kwargs):
        super().__init__(in_channels=in_channels, noise_channels=noise_channels, weighted=weighted, nd=3, **kwargs)


def ensure_num_tuple(v, num=2, msg=''):
    if isinstance(v, (int, float)):
        v = (v,) * num
    elif isinstance(v, (list, tuple)):
        pass
    else:
        raise ValueError(msg)
    return v


class _Stride(nn.Module):

    def __init__(self, stride, start=0, nd=2):
        super().__init__()
        self.stride = ensure_num_tuple(stride, nd)
        self.start = start

    def forward(self, x):
        return x[(...,) + tuple(slice(self.start, None, s) for s in self.stride)]


class Stride1d(_Stride):

    def __init__(self, stride, start=0):
        super().__init__(stride, start, 1)


class Stride2d(_Stride):

    def __init__(self, stride, start=0):
        super().__init__(stride, start, 2)


class Stride3d(_Stride):

    def __init__(self, stride, start=0):
        super().__init__(stride, start, 3)


class _Fuse(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, activation='relu', norm_layer='batchnorm2d', nd=2, dim=1, **kwargs):
        super().__init__()
        modules = [get_nd_conv(nd)(in_channels, out_channels, kernel_size, padding=padding, **kwargs)]
        if norm_layer is not None:
            modules.append(lookup_nn(norm_layer, out_channels, nd=nd))
        if activation is not None:
            modules.append(lookup_nn(activation, inplace=False))
        self.block = nn.Sequential(*modules)
        self.nd = nd
        self.dim = dim

    def forward(self, x: 'tuple'):
        x = tuple(x)
        target_size = x[0].shape[-self.nd:]
        x = torch.cat([(F.interpolate(x_, target_size) if x_.shape[-self.nd:] != target_size else x_) for x_ in x], dim=self.dim)
        return self.block(x)


class Fuse1d(_Fuse):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, activation='relu', norm_layer='batchnorm1d', **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, activation=activation, norm_layer=norm_layer, nd=1, **kwargs)


class Fuse2d(_Fuse):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, activation='relu', norm_layer='batchnorm2d', **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, activation=activation, norm_layer=norm_layer, nd=2, **kwargs)


class Fuse3d(_Fuse):

    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, activation='relu', norm_layer='batchnorm3d', **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, activation=activation, norm_layer=norm_layer, nd=3, **kwargs)


class Normalize(nn.Module):

    def __init__(self, mean=0.0, std=1.0, assert_range=(0.0, 1.0)):
        super().__init__()
        self.assert_range = assert_range
        self.transform = trans.Compose([trans.Normalize(mean=mean, std=std)])

    def forward(self, inputs: 'Tensor'):
        if self.assert_range is not None:
            assert torch.all(inputs >= self.assert_range[0]) and torch.all(inputs <= self.assert_range[1]), f'Inputs should be in interval {self.assert_range}'
        if self.transform is not None:
            inputs = self.transform(inputs)
        return inputs

    def extra_repr(self) ->str:
        s = ''
        if self.assert_range is not None:
            s += f'(assert_range): {self.assert_range}\n'
        s += f'(norm): {repr(self.transform)}'
        return s


class SelfAttention(nn.Module):

    def __init__(self, in_channels, out_channels=None, mid_channels=None, kernel_size=1, padding=0, beta=True, nd=2):
        """Self-Attention.

        References:
            - https://arxiv.org/pdf/1805.08318.pdf

        Args:
            in_channels:
            out_channels: Equal to `in_channels` by default.
            mid_channels: Set to `in_channels // 8` by default.
            kernel_size:
            padding:
            beta:
            nd:
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels // 8
        if out_channels is None:
            out_channels = in_channels
        Conv = lookup_nn('Conv2d', nd=nd, call=False)
        self.beta = nn.Parameter(torch.zeros(1)) if beta else 1.0
        if in_channels != out_channels:
            self.in_conv = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.in_conv = None
        self.proj_b, self.proj_a = [Conv(out_channels, mid_channels, kernel_size=1) for _ in range(2)]
        self.proj = Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.out_conv = Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, inputs):
        x = inputs if self.in_conv is None else self.in_conv(inputs)
        a = self.proj_a(x).flatten(2)
        b = self.proj_b(x).flatten(2)
        p = torch.matmul(a.permute(0, 2, 1), b)
        p = F.softmax(p, dim=1)
        c = self.proj(x).flatten(2)
        out = torch.matmul(p, c.permute(0, 2, 1)).view(*c.shape[:2], *inputs.shape[2:])
        out = self.out_conv(self.beta * out + x)
        return out


def channels_first_permute(nd):
    return (0, nd + 1) + tuple(range(1, nd + 1))


def channels_last_permute(nd):
    return (0,) + tuple(range(2, nd + 2)) + (1,)


class LayerNormNd(nn.LayerNorm):

    def __init__(self, normalized_shape, eps: 'float'=1e-05, elementwise_affine: 'bool'=True, nd=2, device=None, dtype=None) ->None:
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)
        self._perm0 = channels_last_permute(nd)
        self._perm1 = channels_first_permute(nd)

    def forward(self, x: 'Tensor') ->Tensor:
        x = x.permute(*self._perm0)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(*self._perm1)
        return x


class LayerNorm1d(LayerNormNd):

    def __init__(self, normalized_shape, eps: 'float'=1e-05, elementwise_affine: 'bool'=True, device=None, dtype=None) ->None:
        """Layer Norm.

        By default, ``LayerNorm1d(channels)`` operates on feature vectors, i.e. the channel dimension.

        Args:
            normalized_shape: Input shape from an expected input of size
            eps: A value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine: A boolean value that when set to ``True``, this module
                has learnable per-element affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: ``True``.
            device: Device.
            dtype: Data type.
        """
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype, nd=1)


class LayerNorm2d(LayerNormNd):

    def __init__(self, normalized_shape, eps: 'float'=1e-05, elementwise_affine: 'bool'=True, device=None, dtype=None) ->None:
        """Layer Norm.

        By default, ``LayerNorm2d(channels)`` operates on feature vectors, i.e. the channel dimension.

        Args:
            normalized_shape: Input shape from an expected input of size
            eps: A value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine: A boolean value that when set to ``True``, this module
                has learnable per-element affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: ``True``.
            device: Device.
            dtype: Data type.
        """
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype, nd=2)


class LayerNorm3d(LayerNormNd):

    def __init__(self, normalized_shape, eps: 'float'=1e-05, elementwise_affine: 'bool'=True, device=None, dtype=None) ->None:
        """Layer Norm.

        By default, ``LayerNorm3d(channels)`` operates on feature vectors, i.e. the channel dimension.

        Args:
            normalized_shape: Input shape from an expected input of size
            eps: A value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine: A boolean value that when set to ``True``, this module
                has learnable per-element affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: ``True``.
            device: Device.
            dtype: Data type.
        """
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype, nd=3)


class CNBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, layer_scale: 'float'=1e-06, stochastic_depth_prob: 'float'=0, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation='gelu', stride: 'int'=1, identity_norm_layer=None, nd: 'int'=2, conv_kwargs=None) ->None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-06)
        if conv_kwargs is None:
            conv_kwargs = {}
        Conv = lookup_nn('Conv2d', nd=nd, call=False)
        out_channels = in_channels if out_channels is None else out_channels
        self.identity = None
        if in_channels != out_channels or stride != 1:
            if identity_norm_layer is None:
                identity_norm_layer = [LayerNorm1d, LayerNorm2d, LayerNorm3d][nd - 1]
            self.identity = nn.Sequential(Conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), identity_norm_layer(out_channels))
        self.block = nn.Sequential(Conv(in_channels, out_channels, kernel_size=conv_kwargs.pop('kernel_size', 7), padding=conv_kwargs.pop('padding', 3), groups=conv_kwargs.pop('groups', out_channels), bias=conv_kwargs.pop('bias', True), **conv_kwargs), Permute(list(channels_last_permute(nd))), norm_layer(out_channels), nn.Linear(in_features=out_channels, out_features=4 * out_channels, bias=True), lookup_nn(activation), nn.Linear(in_features=4 * out_channels, out_features=out_channels, bias=True), Permute(list(channels_first_permute(nd))))
        if layer_scale is None:
            self.layer_scale = 1
        else:
            self.layer_scale = nn.Parameter(torch.ones(out_channels, *((1,) * nd)) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')

    def forward(self, inputs: 'Tensor') ->Tensor:
        identity = inputs if self.identity is None else self.identity(inputs)
        result = self.layer_scale * self.block(inputs)
        result = self.stochastic_depth(result)
        result += identity
        return result


class ConvNormActivation(misc.ConvNormActivation):

    def __init__(self, in_channels: 'int', out_channels: 'int', kernel_size: 'int'=3, stride: 'int'=1, padding: 'Optional[int]'=None, groups: 'int'=1, norm_layer: 'Optional[Callable[..., torch.nn.Module]]'=torch.nn.BatchNorm2d, activation_layer: 'Optional[Callable[..., torch.nn.Module]]'=torch.nn.ReLU, dilation: 'int'=1, inplace: 'Optional[bool]'=True, bias: 'Optional[bool]'=None, nd=2) ->None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, groups, norm_layer, activation_layer, dilation, inplace, bias, lookup_nn('Conv2d', nd=nd, call=False))


class Ppm(nn.Module):

    def __init__(self, in_channels, out_channels, scales: 'Union[list, tuple]'=(1, 2, 3, 6), kernel_size=1, norm='BatchNorm2d', activation='relu', concatenate=True, nd=2, **kwargs):
        """Pyramid Pooling Module.

        References:
            - https://ieeexplore.ieee.org/document/8100143

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels per pyramid scale.
            scales: Pyramid scales. Default: (1, 2, 3, 6).
            kernel_size: Kernel size.
            norm: Normalization.
            activation: Activation.
            concatenate: Whether to concatenate module inputs to pyramid pooling output before returning results.
            **kwargs: Keyword arguments for ``nn.Conv2d``.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        self.concatenate = concatenate
        self.out_channels = out_channels * len(scales) + in_channels * int(concatenate)
        Conv = get_nd_conv(nd)
        AdaptiveAvgPool = lookup_nn(nn.AdaptiveAvgPool2d, call=False, nd=nd)
        norm = lookup_nn(norm, call=False, nd=nd)
        activation = lookup_nn(activation, call=False, nd=nd)
        for scale in scales:
            self.blocks.append(nn.Sequential(AdaptiveAvgPool(output_size=scale), Conv(in_channels, out_channels, kernel_size, **kwargs), norm(out_channels), activation()))

    def forward(self, x):
        prefix = [x] if self.concatenate else []
        return torch.cat(prefix + [F.interpolate(m(x), x.shape[2:], mode='bilinear', align_corners=False) for m in self.blocks], 1)


def append_pyramid_pooling_(module: 'nn.Sequential', out_channels, scales=(1, 2, 3, 6), method='ppm', in_channels=None, **kwargs):
    if in_channels is None:
        in_channels = module.out_channels[-1]
    method = method.lower()
    if method == 'ppm':
        assert out_channels % len(scales) == 0
        p = Ppm(in_channels, out_channels, scales=scales, **kwargs)
        out_channels = p.out_channels
    elif method == 'aspp':
        scales = sorted(tuple(set(scales) - {1}))
        nd = kwargs.pop('nd', 2)
        assert nd == 2, NotImplementedError('Only nd=2 supported.')
        p = ASPP(in_channels, scales, out_channels, **kwargs)
    else:
        raise ValueError
    module.append(p)
    if hasattr(module, 'out_channels'):
        module.out_channels += out_channels,
    if hasattr(module, 'out_strides'):
        module.out_strides += module.out_strides[-1:]


def get_nn(item: "Union[str, 'nn.Module', Type['nn.Module']]", src=None, nd=None, call_if_type=False):
    ret = lookup_nn(item, src=src, nd=nd, call=False)
    if call_if_type and type(ret) is type:
        ret = ret()
    return ret


def _apply_mapping_rules(key, rules: 'dict'):
    for prefix, repl in rules.items():
        if key.startswith(prefix):
            key = key.replace(prefix, repl, 1)
    return key


def map_state_dict(in_channels, state_dict, fused_initial):
    """Map state dict.

    Map state dict from torchvision format to celldetection format.

    Args:
        in_channels: Number of input channels.
        state_dict: State dict.
        fused_initial:

    Returns:
        State dict in celldetection format.
    """
    mapping = {}
    for k, v in state_dict.items():
        if 'fc' in k:
            continue
        if k.startswith('conv1.') and v.data.shape[1] != in_channels:
            v.data = F.interpolate(v.data[None], (in_channels,) + v.data.shape[-2:]).squeeze(0)
        if fused_initial:
            rules = {'conv1.': '0.0.', 'bn1.': '0.1.', 'layer1.': '0.4.', 'layer2.': '1.', 'layer3.': '2.', 'layer4.': '3.', 'layer5.': '4.'}
        else:
            rules = {'conv1.': '0.0.', 'bn1.': '0.1.', 'layer1.': '1.1.', 'layer2.': '2.', 'layer3.': '3.', 'layer4.': '4.', 'layer5.': '5.'}
        mapping[_apply_mapping_rules(k, rules)] = v
    return mapping


default_model_urls = {'ResNet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth', 'ResNet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth', 'ResNet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth', 'ResNet101': 'https://download.pytorch.org/models/resnet101-cd907fc2.pth', 'ResNet152': 'https://download.pytorch.org/models/resnet152-f82ba261.pth', 'ResNeXt50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth', 'ResNeXt101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth', 'WideResNet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth', 'WideResNet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth'}


class GRN(nn.Module):

    def __init__(self, channels, nd=2, channels_axis=-1, epsilon=1e-06):
        """Global Response Normalization.

        References:
            - https://arxiv.org/abs/2301.00808

        Note:
            - Expects channels last format

        Args:
            channels: Number of channels.
            nd: Number of spatial dimensions.
            channels_axis: Channels axis. Expects channels-last format by default.
        """
        super().__init__()
        self.channels_axis = channels_axis
        dims = [1] * (nd + 2)
        dims[self.channels_axis] = channels
        self.spatial_dims = tuple(range(1, nd + 1))
        self.nd = nd
        self.gamma = nn.Parameter(torch.zeros(*dims))
        self.beta = nn.Parameter(torch.zeros(*dims))
        self.epsilon = epsilon

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=self.spatial_dims, keepdim=True)
        nx = gx / (gx.mean(dim=self.channels_axis, keepdim=True) + self.epsilon)
        return self.gamma * (x * nx) + self.beta + x


class CNBlockV2(CNBlock):

    def __init__(self, in_channels, out_channels=None, layer_scale: 'float'=None, stochastic_depth_prob: 'float'=0, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, activation='gelu', stride: 'int'=1, identity_norm_layer=None, nd: 'int'=2, conv_kwargs=None) ->None:
        """ConvNeXt Block V2.

        References:
            - https://arxiv.org/abs/2301.00808

        Args:
            in_channels:
            out_channels:
            layer_scale:
            stochastic_depth_prob:
            norm_layer:
            activation:
            stride:
            identity_norm_layer:
            nd:
            conv_kwargs:
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels, layer_scale=layer_scale, stochastic_depth_prob=stochastic_depth_prob, norm_layer=norm_layer, activation=activation, stride=stride, identity_norm_layer=identity_norm_layer, nd=nd, conv_kwargs=conv_kwargs)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-06)
        if conv_kwargs is None:
            conv_kwargs = {}
        Conv = lookup_nn('Conv2d', nd=nd, call=False)
        out_channels = in_channels if out_channels is None else out_channels
        self.block = nn.Sequential(Conv(in_channels, out_channels, kernel_size=conv_kwargs.pop('kernel_size', 7), padding=conv_kwargs.pop('padding', 3), groups=conv_kwargs.pop('groups', out_channels), bias=conv_kwargs.pop('bias', True), **conv_kwargs), Permute(list(channels_last_permute(nd))), norm_layer(out_channels), nn.Linear(in_features=out_channels, out_features=4 * out_channels, bias=True), lookup_nn(activation), GRN(4 * out_channels, nd=nd, channels_axis=-1), nn.Linear(in_features=4 * out_channels, out_features=out_channels, bias=True), Permute(list(channels_first_permute(nd))))


def _init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def _equal_size(x, reference, mode='bilinear', align_corners=False):
    if reference.shape[2:] != x.shape[2:]:
        x = F.interpolate(x, reference.shape[2:], mode=mode, align_corners=align_corners)
    return x


def _resolve_channels(encoder_channels, backbone_channels, keys: 'Union[list, tuple, str]', encoder_prefix: 'str'):
    channels = 0
    reference = None
    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    for k in keys:
        if k.startswith(encoder_prefix):
            channels += encoder_channels[int(k[len(encoder_prefix):])]
        else:
            channels += backbone_channels[int(k)]
        if reference is None:
            reference = channels
    return channels, reference, len(keys)


def _resolve_features(features, keys):
    if isinstance(keys, (tuple, list)):
        return [features[k] for k in keys]
    return features[keys]


class CPNCore(nn.Module):

    def __init__(self, backbone: 'nn.Module', backbone_channels, order, score_channels: 'int', refinement: 'bool'=True, refinement_margin: 'float'=3.0, uncertainty_head=False, contour_features='1', location_features='1', uncertainty_features='1', score_features='1', refinement_features='0', contour_head_channels=None, contour_head_stride=1, refinement_head_channels=None, refinement_head_stride=1, refinement_interpolation='bilinear', refinement_buckets=1, refinement_full_res=True, encoder_channels=None, **kwargs):
        super().__init__()
        self.order = order
        self.backbone = backbone
        self.refinement_interpolation = refinement_interpolation
        assert refinement_buckets >= 1
        self.refinement_buckets = refinement_buckets
        if encoder_channels is None:
            encoder_channels = backbone_channels
        channels = encoder_channels, backbone_channels
        kw = {'encoder_prefix': kwargs.get('encoder_prefix', 'encoder.')}
        self.contour_features = contour_features
        self.location_features = location_features
        self.score_features = score_features
        self.refinement_features = refinement_features
        self.uncertainty_features = uncertainty_features
        self.refinement_full_res = refinement_full_res
        fourier_channels, fourier_channels_, num_fourier_inputs = _resolve_channels(*channels, contour_features, **kw)
        loc_channels, loc_channels_, num_loc_inputs = _resolve_channels(*channels, location_features, **kw)
        sco_channels, sco_channels_, num_score_inputs = _resolve_channels(*channels, score_features, **kw)
        ref_channels, ref_channels_, num_ref_inputs = _resolve_channels(*channels, refinement_features, **kw)
        unc_channels, unc_channels_, num_unc_inputs = _resolve_channels(*channels, uncertainty_features, **kw)
        fuse_kw = kwargs.get('fuse_kwargs', {})
        self.score_fuse = Fuse2d(sco_channels, sco_channels_, **fuse_kw) if num_score_inputs > 1 else None
        self.score_head = ReadOut(sco_channels_, score_channels, kernel_size=kwargs.get('kernel_size_score', 7), padding=kwargs.get('kernel_size_score', 7) // 2, channels_mid=contour_head_channels, stride=contour_head_stride, activation=kwargs.pop('head_activation_score', kwargs.get('head_activation', 'relu')))
        self.location_fuse = Fuse2d(loc_channels, loc_channels_, **fuse_kw) if num_loc_inputs > 1 else None
        self.location_head = ReadOut(loc_channels_, 2, kernel_size=kwargs.get('kernel_size_location', 7), padding=kwargs.get('kernel_size_location', 7) // 2, channels_mid=contour_head_channels, stride=contour_head_stride, activation=kwargs.pop('head_activation_location', kwargs.get('head_activation', 'relu')))
        self.fourier_fuse = Fuse2d(fourier_channels, fourier_channels_, **fuse_kw) if num_fourier_inputs > 1 else None
        self.fourier_head = ReadOut(fourier_channels_, order * 4, kernel_size=kwargs.get('kernel_size_fourier', 7), padding=kwargs.get('kernel_size_fourier', 7) // 2, channels_mid=contour_head_channels, stride=contour_head_stride, activation=kwargs.pop('head_activation_fourier', kwargs.get('head_activation', 'relu')))
        if uncertainty_head:
            self.uncertainty_fuse = Fuse2d(unc_channels, unc_channels_, **fuse_kw) if num_unc_inputs > 1 else None
            self.uncertainty_head = ReadOut(unc_channels_, 4, kernel_size=kwargs.get('kernel_size_uncertainty', 7), padding=kwargs.get('kernel_size_uncertainty', 7) // 2, channels_mid=contour_head_channels, stride=contour_head_stride, final_activation='sigmoid', activation=kwargs.pop('head_activation_uncertainty', kwargs.get('head_activation', 'relu')))
        else:
            self.uncertainty_fuse = self.uncertainty_head = None
        if refinement:
            self.refinement_fuse = Fuse2d(ref_channels, ref_channels_, **fuse_kw) if num_ref_inputs > 1 else None
            self.refinement_head = ReadOut(ref_channels_, 2 * refinement_buckets, kernel_size=kwargs.get('kernel_size_refinement', 7), padding=kwargs.get('kernel_size_refinement', 7) // 2, final_activation=ScaledTanh(refinement_margin), channels_mid=refinement_head_channels, stride=refinement_head_stride, activation=kwargs.pop('head_activation_refinement', kwargs.get('head_activation', 'relu')))
        else:
            self.refinement_fuse = self.refinement_head = None

    def forward(self, inputs):
        features = self.backbone(inputs)
        if isinstance(features, torch.Tensor):
            score_features = fourier_features = location_features = unc_features = ref_features = features
        else:
            score_features = _resolve_features(features, self.score_features)
            fourier_features = _resolve_features(features, self.contour_features)
            location_features = _resolve_features(features, self.location_features)
            unc_features = _resolve_features(features, self.uncertainty_features)
            ref_features = _resolve_features(features, self.refinement_features)
        if self.score_fuse is not None:
            score_features = self.score_fuse(score_features)
        scores = self.score_head(score_features)
        if self.location_fuse is not None:
            location_features = self.location_fuse(location_features)
        locations = self.location_head(location_features)
        if self.fourier_fuse is not None:
            fourier_features = self.fourier_fuse(fourier_features)
        fourier = self.fourier_head(fourier_features)
        if self.uncertainty_head is not None:
            if self.uncertainty_fuse is not None:
                unc_features = self.uncertainty_fuse(unc_features)
            uncertainty = self.uncertainty_head(unc_features)
        else:
            uncertainty = None
        if self.refinement_head is not None:
            if self.refinement_fuse is not None:
                ref_features = self.refinement_fuse(ref_features)
            if self.refinement_full_res:
                ref_features = _equal_size(ref_features, inputs, mode=self.refinement_interpolation)
            refinement = _equal_size(self.refinement_head(ref_features), inputs, mode=self.refinement_interpolation)
        else:
            refinement = None
        return scores, locations, refinement, fourier, uncertainty


def _pairwise_box_inter_union(boxes1: 'Tensor', boxes2: 'Tensor') ->Tuple[Tensor, Tensor]:
    area1 = bx.box_area(boxes1)
    area2 = bx.box_area(boxes2)
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = bx._upcast(rb - lt).clamp(min=0)
    intersection = torch.prod(wh, dim=1)
    union = area1 + area2 - intersection
    return intersection, union


def pairwise_box_iou(boxes1: 'Tensor', boxes2: 'Tensor') ->Tensor:
    inter, union = _pairwise_box_inter_union(boxes1, boxes2)
    return torch.abs(inter / union)


def reduce_loss(x: 'Tensor', reduction: 'str', **kwargs):
    """Reduce loss.

    Reduces Tensor according to ``reduction``.

    Args:
        x: Input.
        reduction: Reduction method. Must be a symbol of ``torch``.
        **kwargs: Additional keyword arguments.

    Returns:
        Reduced Tensor.
    """
    if reduction == 'none':
        return x
    fn = getattr(torch, reduction, None)
    if fn is None:
        raise ValueError(f'Unknown reduction: {reduction}')
    return fn(x, **kwargs)


def box_npll_loss(uncertainty, boxes, boxes_targets, factor=10.0, sigmoid=False, epsilon=1e-08, reduction='mean', min_size=None):
    """NPLL.

    References:
        https://arxiv.org/abs/2006.15607

    Args:
        uncertainty: Tensor[n, 4].
        boxes: Tensor[n, 4].
        boxes_targets: Tensor[n, 4].
        sigmoid: Whether to apply the ``sigmoid`` function to ``uncertainty``.
        factor: Uncertainty factor.
        epsilon: Epsilon.
        reduction: Loss reduction.
        min_size: Minimum box size. May be used to remove degenerate boxes.

    Returns:
        Loss.
    """
    if min_size is not None:
        keep = bx.remove_small_boxes(boxes, min_size)
        boxes, boxes_targets, uncertainty = (c[keep] for c in (boxes, boxes_targets, uncertainty))
    delta_sq = torch.square((torch.sigmoid(uncertainty) if sigmoid else uncertainty) * factor)
    a = torch.square(boxes - boxes_targets) / (2 * delta_sq + epsilon)
    b = 0.5 * torch.log(delta_sq + epsilon)
    iou = pairwise_box_iou(boxes, boxes_targets)
    loss = iou * ((a + b).sum(dim=1) + 2 * np.log(2 * np.pi))
    loss = reduce_loss(loss, reduction=reduction)
    return loss


class BoxNpllLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, factor=10.0, sigmoid=False, min_size=None, epsilon=1e-08, size_average=None, reduce=None, reduction: 'str'='mean') ->None:
        super().__init__(size_average, reduce, reduction)
        self.factor = factor
        self.sigmoid = sigmoid
        self.min_size = min_size
        self.epsilon = epsilon

    def forward(self, uncertainty: 'Tensor', input: 'Tensor', target: 'Tensor') ->Tensor:
        return box_npll_loss(uncertainty, input, target, factor=self.factor, sigmoid=self.sigmoid, epsilon=self.epsilon, reduction=self.reduction, min_size=self.min_size)


def pairwise_generalized_box_iou(boxes1: 'Tensor', boxes2: 'Tensor') ->Tensor:
    inter, union = _pairwise_box_inter_union(boxes1, boxes2)
    iou = inter / union
    lti = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    rbi = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    whi = bx._upcast(rbi - lti).clamp(min=0)
    areai = torch.prod(whi, dim=1)
    return iou - (areai - union) / areai


def iou_loss(boxes, boxes_targets, reduction='mean', generalized=True, method='linear', min_size=None):
    if min_size is not None:
        keep = bx.remove_small_boxes(boxes, min_size)
        boxes, boxes_targets = (c[keep] for c in (boxes, boxes_targets))
    if generalized:
        iou = pairwise_generalized_box_iou(boxes, boxes_targets)
    else:
        iou = pairwise_box_iou(boxes, boxes_targets)
    if method == 'log':
        if generalized:
            iou = iou * 0.5 + 0.5
        loss = -torch.log(iou + 1e-08)
    elif method == 'linear':
        loss = 1 - iou
    else:
        raise ValueError
    loss = reduce_loss(loss, reduction=reduction)
    return loss


class IoULoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, generalized=True, method='linear', min_size=None, size_average=None, reduce=None, reduction: 'str'='mean') ->None:
        super().__init__(size_average, reduce, reduction)
        self.generalized = generalized
        self.method = method
        self.min_size = min_size

    def forward(self, input: 'Tensor', target: 'Tensor') ->Tensor:
        return iou_loss(input, target, self.reduction, generalized=self.generalized, method=self.method, min_size=self.min_size)

    def extra_repr(self) ->str:
        return f"generalized={self.generalized}, method='{self.method}'"


def equal_size(x, reference, mode='bilinear', align_corners=False):
    if reference.shape[2:] != x.shape[2:]:
        x = F.interpolate(x, reference.shape[2:], mode=mode, align_corners=align_corners)
    return x


def _apply_score_bounds(scores, scores_lower_bound, scores_upper_bound):
    if scores_upper_bound is not None:
        assert scores_upper_bound.ndim >= 4, f'Please make sure scores_upper_bound comes in NCHW format: {scores_upper_bound.shape}'
        assert scores_upper_bound.dtype.is_floating_point, f'Please make sure to pass scores_upper_bound as float instead of {scores_upper_bound.dtype}'
        scores = torch.minimum(scores, equal_size(scores_upper_bound, scores))
    if scores_lower_bound is not None:
        assert scores_lower_bound.ndim >= 4, f'Please make sure scores_upper_bound comes in NCHW format: {scores_lower_bound.shape}'
        assert scores_lower_bound.dtype.is_floating_point, f'Please make sure to pass scores_upper_bound as float instead of {scores_lower_bound.dtype}'
        scores = torch.maximum(scores, equal_size(scores_lower_bound, scores))
    return scores


def add_to_loss_dict(d: 'dict', key: 'str', loss: 'torch.Tensor', weight=None):
    if loss is None:
        return
    dk = d.get(key, None)
    torch.nan_to_num_(loss, 0.0, 0.0, 0.0)
    if weight is not None:
        loss = loss * weight
    d[key] = loss if dk is None else dk + loss


NMS_BATCH_SIZE = 50000


def batched_box_nmsi(boxes: 'List[Tensor]', scores: 'List[Tensor]', iou_threshold: 'float', batch_size: 'int'=None) ->List[Tensor]:
    """
    Apply Non-Maximum Suppression (NMS) in batches to avoid OOM errors for very large numbers of boxes.

    Args:
        boxes (List[Tensor]): List of tensors where each tensor contains bounding box coordinates of
            shape [num_boxes, 4].
        scores (List[Tensor]): List of tensors where each tensor contains scores for each box of shape [num_boxes].
        iou_threshold (float): The IoU threshold for suppression.
        batch_size (int): Maximum number of boxes to process in each batch.

    Returns:
        List[Tensor]: A list of tensors where each tensor contains the indices of the boxes that are kept after NMS.
    """
    assert len(scores) == len(boxes), 'The number of score tensors must match the number of box tensors.'
    batch_size = NMS_BATCH_SIZE if batch_size is None else batch_size
    keeps = []
    for con, sco in zip(boxes, scores):
        num_boxes = con.size(0)
        if num_boxes <= batch_size:
            indices = torch.ops.torchvision.nms(con, sco, iou_threshold)
        else:
            indices = torch.zeros(0, dtype=torch.long, device=con.device)
            for start_idx in range(0, num_boxes, batch_size):
                end_idx = min(start_idx + batch_size, num_boxes)
                batch_indices = torch.ops.torchvision.nms(con[start_idx:end_idx], sco[start_idx:end_idx], iou_threshold)
                indices = torch.cat((indices, batch_indices + start_idx))
            if indices.numel() > 0:
                final_boxes = con[indices]
                final_scores = sco[indices]
                keep_final_indices = torch.ops.torchvision.nms(final_boxes, final_scores, iou_threshold)
                indices = indices[keep_final_indices]
        keeps.append(indices)
    return keeps


def downsample_labels(inputs, size: 'List[int]'):
    """

    Down-sample via max-pooling and interpolation

    Notes:
        - Downsampling can lead to loss of labeled instances, both during max pooling and interpolation.
        - Typical timing: 0.08106 ms for 256x256

    Args:
        inputs: Label Tensor to resize. Shape (n, c, h, w)
        size: Tuple containing target height and width.

    Returns:

    """
    sizeh, sizew = size
    if inputs.shape[-2:] == (sizeh, sizew):
        return inputs
    if inputs.dtype != torch.float32:
        inputs = inputs.float()
    h, w = inputs.shape[-2:]
    th, tw = size
    k = h // th, w // tw
    r = F.max_pool2d(inputs, k, k)
    if r.shape[-2:] != (sizeh, sizew):
        r = F.interpolate(r, size, mode='nearest')
    return r


def dummy_loss(*a, sub=1):
    return 0.0 * sum([i[:sub].mean() for i in a if isinstance(i, Tensor)])


def fouriers2contours(fourier, locations, samples=64, sampling=None, cache: 'Dict[str, Tensor]'=None, cache_size: 'int'=16):
    """

    Args:
        fourier: Tensor[..., order, 4]
        locations: Tensor[..., 2]
        samples: Number of samples. Only used for default sampling, ignored otherwise.
        sampling: Sampling t. Default is linspace 0..1. Device should match `fourier` and `locations`.
        cache: Cache for initial zero tensors. When fourier shapes are consistent this can increase execution times.
        cache_size: Cache size.

    Returns:
        Contours.
    """
    if isinstance(fourier, (tuple, list)):
        if sampling is None:
            sampling = [sampling] * len(fourier)
        return [fouriers2contours(f, l, samples=samples, sampling=s) for f, l, s in zip(fourier, locations, sampling)]
    order = fourier.shape[-2]
    d = fourier.device
    sampling_ = sampling
    if sampling is None:
        sampling = sampling_ = torch.linspace(0, 1.0, samples, device=d)
    samples = sampling.shape[-1]
    sampling = sampling[..., None, :]
    c = float(np.pi) * 2 * torch.arange(1, order + 1, device=d)[..., None] * sampling
    c_cos = torch.cos(c)
    c_sin = torch.sin(c)
    con = None
    con_shape = fourier.shape[:-2] + (samples, 2)
    con_key = str(tuple(con_shape) + (d,))
    if cache is not None:
        con = cache.get(con_key, None)
    if con is None:
        con = torch.zeros(fourier.shape[:-2] + (samples, 2), device=d)
        if cache is not None:
            if len(cache) >= cache_size:
                del cache[next(iter(cache.keys()))]
            cache[con_key] = con
    con = con + locations[..., None, :]
    con += (fourier[..., None, (1, 3)] * c_sin[(...,) + (None,) * 1]).sum(-3)
    con += (fourier[..., None, (0, 2)] * c_cos[(...,) + (None,) * 1]).sum(-3)
    return con, sampling_


def refinement_bucket_weight(index, base_index):
    dist = torch.abs(index + 0.5 - base_index)
    sel = dist > 1
    dist = 1.0 - dist
    dist[sel] = 0
    dist.detach_()
    return dist


def resolve_refinement_buckets(samplings, num_buckets):
    base_index = samplings * num_buckets
    base_index_int = base_index.long()
    a, b, c = base_index_int - 1, base_index_int, base_index_int + 1
    return (a % num_buckets, refinement_bucket_weight(a, base_index)), (b % num_buckets, refinement_bucket_weight(b, base_index)), (c % num_buckets, refinement_bucket_weight(c, base_index))


def local_refinement(det_indices, refinement, num_loops, num_buckets, original_size, sampling, b):
    all_det_indices = []
    for _ in torch.arange(0, num_loops):
        det_indices = torch.round(det_indices.detach())
        det_indices[..., 0].clamp_(0, original_size[1] - 1)
        det_indices[..., 1].clamp_(0, original_size[0] - 1)
        indices = det_indices.detach().long()
        if num_buckets == 1:
            responses = refinement[b[:, None], :, indices[:, :, 1], indices[:, :, 0]]
        else:
            buckets = resolve_refinement_buckets(sampling, num_buckets)
            responses = None
            for bucket_indices, bucket_weights in buckets:
                bckt_idx = torch.stack((bucket_indices * 2, bucket_indices * 2 + 1), -1)
                cur_ref = refinement[b[:, None, None], bckt_idx, indices[:, :, 1, None], indices[:, :, 0, None]]
                cur_ref = cur_ref * bucket_weights[..., None]
                if responses is None:
                    responses = cur_ref
                else:
                    responses = responses + cur_ref
        det_indices = det_indices + responses
        all_det_indices.append(det_indices)
    return det_indices, all_det_indices


def order_weighting(order, max_w=5, min_w=1, spread=None) ->torch.Tensor:
    x = torch.arange(order).float()
    if spread is None:
        spread = order - 1
    y = min_w + (max_w - min_w) * (1 - (x / spread).clamp(0.0, 1.0)) ** 2
    return y[:, None]


def reduce_loss_dict(losses: 'dict', divisor, ignore_prefix='_'):
    return sum(i for k, i in losses.items() if i is not None and not k.startswith(ignore_prefix)) / divisor


def rel_location2abs_location(locations, cache: 'Dict[str, Tensor]'=None, cache_size: 'int'=16):
    """

    Args:
        locations: Tensor[..., 2, h, w]. In xy format.
        cache: can be None.
        cache_size:

    Returns:

    """
    d = locations.device
    h, w = locations.shape[-2:]
    offset = None
    if cache is not None:
        key = str((h, w, d))
        if key in cache.keys():
            offset = cache[key]
    if offset is None:
        offset = torch.stack((torch.arange(w, device=d)[None] + torch.zeros(h, device=d)[:, None], torch.zeros(w, device=d)[None] + torch.arange(h, device=d)[:, None]), 0)
        if cache is not None:
            cache[str((h, w, d))] = offset
    if cache is not None and len(cache) > cache_size:
        del cache[list(cache.keys())[0]]
    r = locations + offset
    return r


def resolve_batch_index(inputs: 'dict', n, b) ->dict:
    outputs = OrderedDict({k: (None if v is None else []) for k, v in inputs.items()})
    for batch_index in range(n):
        sel = b == batch_index
        for k, v in inputs.items():
            o = outputs[k]
            if o is not None:
                o.append(v[sel])
    return outputs


def resolve_keep_indices(inputs: 'dict', keep: 'list') ->dict:
    outputs = OrderedDict({k: (None if v is None else []) for k, v in inputs.items()})
    for j, indices in enumerate(keep):
        for k, v in inputs.items():
            o = outputs[k]
            if o is not None:
                o.append(v[j][indices])
    return outputs


def get_scale(actual_size, original_size, flip=True, dtype=torch.float):
    scale = torch.as_tensor(original_size, dtype=dtype) / torch.as_tensor(actual_size, dtype=dtype)
    if flip:
        scale = scale.flip(-1)
    return scale


def scale_contours(actual_size, original_size, contours):
    """

    Args:
        actual_size: Image size. E.g. (256, 256)
        original_size: Original image size. E.g. (512, 512)
        contours: Contours that are to be scaled to from `actual_size` to `original_size`.
            E.g. array of shape (1, num_points, 2) for a single contour or tuple/list of (num_points, 2) arrays.
            Last dimension is interpreted as (x, y).

    Returns:
        Rescaled contours.
    """
    assert len(actual_size) == len(original_size)
    scale = get_scale(actual_size, original_size, flip=True)
    if isinstance(contours, Tensor):
        contours = contours * scale
    else:
        assert isinstance(contours, (tuple, list))
        scale = scale
        for i in range(len(contours)):
            contours[i] = contours[i] * scale
    return contours


def _scale_fourier(fourier, location, scale):
    fourier[..., [0, 1]] = fourier[..., [0, 1]] * scale[0]
    fourier[..., [2, 3]] = fourier[..., [2, 3]] * scale[1]
    location = location * scale
    return fourier, location


def scale_fourier(actual_size, original_size, fourier, location):
    """

    Args:
        actual_size: Image size. E.g. (256, 256)
        original_size: Original image size. E.g. (512, 512)
        fourier: Fourier descriptor. E.g. array of shape (..., order, 4).
        location: Location. E.g. array of shape (..., 2). Last dimension is interpreted as (x, y).

    Returns:
        Rescaled fourier, rescaled location
    """
    assert len(actual_size) == len(original_size)
    scale = get_scale(actual_size, original_size, flip=True)
    if isinstance(fourier, Tensor):
        return _scale_fourier(fourier, location, scale)
    else:
        assert isinstance(fourier, (list, tuple))
        scale = scale
        rfo, rlo = [], []
        for fo, lo in zip(fourier, location):
            a, b = _scale_fourier(fo, lo, scale)
            rfo.append(a)
            rlo.append(b)
        return rfo, rlo


def get_nd_linear(dim: 'int'):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return ['', 'bi', 'tri'][dim - 1] + 'linear'


def update_dict_(dst, src, override=False, keys: 'Union[List[str], Tuple[str]]'=None):
    for k, v in src.items():
        if keys is not None and k not in keys:
            continue
        if override or k not in dst:
            dst[k] = v


class GeneralizedUNet(FeaturePyramidNetwork):

    def __init__(self, in_channels_list, out_channels: 'int', block_cls: 'nn.Module', block_kwargs: 'dict'=None, final_activation=None, interpolate='nearest', final_interpolate=None, initialize=True, keep_features=True, bridge_strides=True, bridge_block_cls: "'nn.Module'"=None, bridge_block_kwargs: 'dict'=None, secondary_block: "'nn.Module'"=None, in_strides_list: 'Union[List[int], Tuple[int]]'=None, out_channels_list: 'Union[List[int], Tuple[int]]'=None, nd=2, **kwargs):
        super().__init__([], 0, extra_blocks=kwargs.get('extra_blocks'))
        block_kwargs = {} if block_kwargs is None else block_kwargs
        Conv = get_nd_conv(nd)
        if out_channels_list is None:
            out_channels_list = in_channels_list
        if in_strides_list is None or bridge_strides is False:
            in_strides_list = [(2 ** i) for i in range(len(in_channels_list))]
        self.bridges = np.log2(in_strides_list[0])
        assert self.bridges % 1 == 0
        self.bridges = int(self.bridges)
        if bridge_block_cls is None:
            bridge_block_cls = partial(TwoConvNormRelu, bias=False)
        else:
            bridge_block_cls = get_nn(bridge_block_cls, nd=nd)
        bridge_block_kwargs = {} if bridge_block_kwargs is None else bridge_block_kwargs
        update_dict_(bridge_block_kwargs, block_kwargs, ('activation', 'norm_layer'))
        if self.bridges:
            num = len(in_channels_list)
            for _ in range(self.bridges):
                in_channels_list = (0,) + tuple(in_channels_list)
                if len(out_channels_list) < num + self.bridges - 1:
                    out_channels_list = (out_channels_list[0],) + tuple(out_channels_list)
        self.cat_order = kwargs.get('cat_order', 0)
        assert self.cat_order in (0, 1)
        self.block_channel_reduction = kwargs.get('block_channel_reduction', False)
        self.block_interpolate = kwargs.get('block_interpolate', False)
        self.block_cat = kwargs.get('block_cat', False)
        self.bridge_block_interpolate = kwargs.get('bridge_block_interpolate', False)
        self.apply_cat = {}
        self.has_lat = {}
        len_in_channels_list = len(in_channels_list)
        for i in range(len_in_channels_list):
            if i > 0:
                inner_ouc = out_channels_list[i - 1]
                inner_inc = out_channels_list[i] if i < len_in_channels_list - 1 else in_channels_list[i]
                if not self.block_channel_reduction and inner_inc > 0 and inner_ouc < inner_inc:
                    inner = Conv(inner_inc, inner_ouc, 1)
                else:
                    inner = nn.Identity()
                self.inner_blocks.append(inner)
            if i < len_in_channels_list - 1:
                lat = in_channels_list[i]
                if self.block_channel_reduction:
                    inc = out_channels_list[i + 1] if i < len_in_channels_list - 2 else in_channels_list[i + 1]
                else:
                    inc = min(out_channels_list[i:i + 2])
                ouc = out_channels_list[i]
                self.apply_cat[i] = False
                self.has_lat[i] = has_lat = lat > 0
                cls, kw = block_cls, block_kwargs
                if not has_lat:
                    self.has_lat[i] = False
                    cls, kw = bridge_block_cls, bridge_block_kwargs
                    inp = inc,
                elif self.block_cat:
                    inp = inc, lat
                else:
                    self.apply_cat[i] = True
                    inp = inc + lat,
                layer_block = cls(*inp, ouc, nd=nd, **kw)
                if secondary_block is not None:
                    layer_block = nn.Sequential(layer_block, secondary_block(ouc, nd=nd))
                self.layer_blocks.append(layer_block)
        self.depth = len(self.layer_blocks)
        self.interpolate = interpolate
        self.keep_features = keep_features
        self.features_prefix = 'encoder'
        self.out_layer = Conv(out_channels_list[0], out_channels, 1) if out_channels > 0 else None
        self.nd = nd
        self.final_interpolate = final_interpolate
        if self.final_interpolate is None:
            self.final_interpolate = get_nd_linear(nd)
        self.final_activation = None if final_activation is None else lookup_nn(final_activation)
        self.out_channels_list = out_channels_list
        self.out_channels = out_channels if out_channels else out_channels_list
        if initialize:
            for m in self.modules():
                if isinstance(m, Conv):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: 'Dict[str, Tensor]', size: 'List[int]') ->Union[Dict[str, Tensor], Tensor]:
        """

        Args:
            x: Input dictionary. E.g. {
                    0: Tensor[1, 64, 128, 128]
                    1: Tensor[1, 128, 64, 64]
                    2: Tensor[1, 256, 32, 32]
                    3: Tensor[1, 512, 16, 16]
                }
            size: Desired final output size. If set to None output remains as it is.

        Returns:
            Output dictionary. For each key in `x` a corresponding output is returned; the final output
            has the key `'out'`.
            E.g. {
                out: Tensor[1, 2, 128, 128]
                0: Tensor[1, 64, 128, 128]
                1: Tensor[1, 128, 64, 64]
                2: Tensor[1, 256, 32, 32]
                3: Tensor[1, 512, 16, 16]
            }
        """
        features = x
        names = list(x.keys())
        x = list(x.values())
        last_inner = x[-1]
        results = [last_inner]
        kw = {} if self.interpolate == 'nearest' else {'align_corners': False}
        for i in range(self.depth - 1, -1, -1):
            lateral = lateral_size = None
            if self.has_lat[i]:
                lateral = x[i - self.bridges]
                lateral_size = lateral.shape[2:]
            inner_top_down = last_inner
            if self.interpolate and (not self.block_interpolate and lateral is not None or not self.bridge_block_interpolate and lateral is None):
                inner_top_down = F.interpolate(inner_top_down, **dict(scale_factor=2) if lateral_size is None else dict(size=lateral_size), mode=self.interpolate, **kw)
            inner_top_down = self.get_result_from_inner_blocks(inner_top_down, i)
            if self.apply_cat[i]:
                if self.cat_order == 0:
                    cat = lateral, inner_top_down
                else:
                    cat = inner_top_down, lateral
                layer_block_inputs = torch.cat(cat, 1)
            elif lateral is None:
                layer_block_inputs = inner_top_down
            else:
                layer_block_inputs = inner_top_down, lateral
            last_inner = self.get_result_from_layer_blocks(layer_block_inputs, i)
            results.insert(0, last_inner)
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)
        if size is None:
            final = results[0]
        else:
            final = F.interpolate(last_inner, size=size, mode=self.final_interpolate, align_corners=False)
        if self.out_layer is not None:
            final = self.out_layer(final)
        if self.final_activation is not None:
            final = self.final_activation(final)
        if self.out_layer is not None:
            return final
        results.insert(0, final)
        names.insert(0, 'out')
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        if self.keep_features:
            out.update(OrderedDict([('.'.join([self.features_prefix, k]), v) for k, v in features.items()]))
        return out


class BackboneAsUNet(nn.Module):

    def __init__(self, backbone, return_layers, in_channels_list, out_channels, block, block_kwargs: 'dict'=None, final_activation=None, interpolate='nearest', ilg=None, nd=2, in_strides_list=None, **kwargs):
        super(BackboneAsUNet, self).__init__()
        if ilg is None:
            ilg = isinstance(backbone, nn.Sequential)
        if block is None:
            block = TwoConvNormRelu
        else:
            block = get_nn(block, nd=nd)
        self.nd = nd
        pretrained_cfg = backbone.__dict__.get('pretrained_cfg', {})
        if kwargs.pop('normalize', True):
            self.normalize = Normalize(mean=kwargs.get('inputs_mean', pretrained_cfg.get('mean', 0.0)), std=kwargs.get('inputs_std', pretrained_cfg.get('std', 1.0)), assert_range=kwargs.get('assert_range', (0.0, 1.0)))
        else:
            self.normalize = None
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers) if ilg else backbone
        self.intermediate_blocks = kwargs.get('intermediate_blocks')
        if self.intermediate_blocks is not None:
            in_channels_list = in_channels_list + type(in_channels_list)(self.intermediate_blocks.out_channels)
            if in_strides_list is not None:
                in_strides_list = in_strides_list + type(in_strides_list)([(i * in_strides_list[-1]) for i in self.intermediate_blocks.out_strides])
        self.unet = GeneralizedUNet(in_channels_list=in_channels_list, out_channels=out_channels, block_cls=block, block_kwargs=block_kwargs, final_activation=final_activation, interpolate=interpolate, in_strides_list=in_strides_list, nd=nd, **kwargs)
        self.out_channels = list(self.unet.out_channels_list)
        self.nd = nd

    def forward(self, inputs):
        x = inputs
        if self.normalize is not None:
            x = self.normalize(x)
        x = self.body(x)
        if self.intermediate_blocks is not None:
            x = self.intermediate_blocks(x)
        x = self.unet(x, size=inputs.shape[-self.nd:])
        return x


def get_nd_max_pool(dim: 'int'):
    assert isinstance(dim, int) and dim in (1, 2, 3)
    return getattr(nn, 'MaxPool%dd' % dim)


class UNetEncoder(nn.Sequential):

    def __init__(self, in_channels, depth=5, base_channels=64, factor=2, pool=True, block_cls: 'Type[nn.Module]'=None, nd=2):
        """U-Net Encoder.

        Args:
            in_channels: Input channels.
            depth: Model depth.
            base_channels: Base channels.
            factor: Growth factor of base_channels.
            pool: Whether to use max pooling or stride 2 for downsampling.
            block_cls: Block class. Callable as `block_cls(in_channels, out_channels, stride=stride)`.
        """
        if block_cls is None:
            block_cls = partial(TwoConvNormRelu, nd=nd)
        else:
            block_cls = get_nn(block_cls, nd=nd)
        MaxPool = get_nd_max_pool(nd)
        layers = []
        self.out_channels = []
        self.out_strides = list(range(1, depth + 1))
        for i in range(depth):
            in_c = base_channels * int(factor ** (i - 1)) * int(i > 0) + int(i <= 0) * in_channels
            out_c = base_channels * factor ** i
            self.out_channels.append(out_c)
            block = block_cls(in_c, out_c, stride=int((not pool and i > 0) + 1))
            if i > 0 and pool:
                block = nn.Sequential(MaxPool(2, stride=2), block)
            layers.append(block)
        super().__init__(*layers)


def _ni_pretrained(pretrained):
    if pretrained:
        raise NotImplementedError('The `pretrained` option is not yet available for this model.')


def _default_unet_kwargs(backbone_kwargs, pretrained=False):
    _ni_pretrained(pretrained)
    kw = dict()
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


def _make_cpn_doc(title, text, backbone):
    return f"""{title}
    
    {text}
    
    References:
        https://www.sciencedirect.com/science/article/pii/S136184152200024X
    
    Args:
        in_channels: Number of input channels.
        order: Contour order. The higher, the more complex contours can be proposed.
            ``order=1`` restricts the CPN to propose ellipses, ``order=3`` allows for non-convex rough outlines,
            ``order=8`` allows even finer detail.
        nms_thresh: IoU threshold for non-maximum suppression (NMS). NMS considers all objects with
            ``iou > nms_thresh`` to be identical.
        score_thresh: Score threshold. For binary classification problems (object vs. background) an object must
            have ``score > score_thresh`` to be proposed as a result.
        samples: Number of samples. This sets the number of coordinates with which a contour is defined.
            This setting can be changed on the fly, e.g. small for training and large for inference.
            Small settings reduces computational costs, while larger settings capture more detail.
        classes: Number of classes. Default: 2 (object vs. background).
        refinement: Whether to use local refinement or not. Local refinement generally improves pixel precision of
            the proposed contours.
        refinement_iterations: Number of refinement iterations.
        refinement_margin: Maximum refinement margin (step size) per iteration.
        refinement_buckets: Number of refinement buckets. Bucketed refinement is especially recommended for data
            with overlapping objects. ``refinement_buckets=1`` practically disables bucketing,
            ``refinement_buckets=6`` uses 6 different buckets, each influencing different fractions of a contour.
        backbone_kwargs: Additional backbone keyword arguments. See docstring of ``{backbone}``.
        **kwargs: Additional CPN keyword arguments. See docstring of ``cd.models.CPN``.
    
    """


def conv1x1(in_planes: 'int', out_planes: 'int', stride: 'int'=1, nd=2) ->nn.Conv2d:
    """1x1 convolution"""
    return get_nd_conv(nd)(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes: 'int', out_planes: 'int', stride: 'int'=1, groups: 'int'=1, dilation: 'int'=1, kernel_size=3, nd=2) ->nn.Conv2d:
    """3x3 convolution with padding"""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * nd
    if isinstance(dilation, int):
        dilation = (dilation,) * nd
    padding = tuple((ks - 1) * dil // 2 for ks, dil in zip(kernel_size, dilation))
    return get_nd_conv(nd)(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion: 'int' = tvr.Bottleneck.expansion
    forward = tvr.Bottleneck.forward

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer='batchnorm2d', kernel_size=3, nd=2) ->None:
        super().__init__()
        norm_layer = lookup_nn(norm_layer, call=False, nd=nd)
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width, nd=nd)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, kernel_size=kernel_size, nd=nd)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, nd=nd)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


def resolve_pretrained(pretrained, state_dict_mapper=None, **kwargs):
    if isinstance(pretrained, str):
        if isfile(pretrained):
            state_dict = torch.load(pretrained)
        else:
            state_dict = load_state_dict_from_url(pretrained)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if '.pytorch.org' in pretrained:
            if state_dict_mapper is not None:
                state_dict = state_dict_mapper(state_dict=state_dict, **kwargs)
    else:
        raise ValueError('There is no default set of weights for this model. Please specify a URL or filename using the `pretrained` argument.')
    return state_dict


def _make_layer(self, block: 'Type[Union[BasicBlock, Bottleneck]]', planes: 'int', blocks: 'int', stride: 'int'=1, dilate: 'bool'=False, kernel_size: 'int'=3, nd=2, secondary_block=None, downsample_method=None) ->nn.Sequential:
    """

    References:
        - [1] https://arxiv.org/abs/1812.01187.pdf

    Args:
        self:
        block:
        planes:
        blocks:
        stride:
        dilate:
        kernel_size:
        nd:
        secondary_block:
        downsample_method: Downsample method. None: 1x1Conv with stride, Norm (standard ResNet),
            'avg': AvgPool, 1x1Conv, Norm (ResNet-D in [1])

    Returns:

    """
    if secondary_block is not None:
        secondary_block = get_nn(secondary_block, nd=nd)
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
        self.dilation *= stride
        stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
        if downsample_method is None or stride <= 1:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride, nd=nd), norm_layer(planes * block.expansion))
        elif downsample_method == 'avg':
            downsample = nn.Sequential(get_nn(nn.AvgPool2d, nd=nd)(2, stride=stride), conv1x1(self.inplanes, planes * block.expansion, nd=nd), norm_layer(planes * block.expansion))
        else:
            raise ValueError(f'Unknown downsample_method: {downsample_method}')
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, kernel_size=kernel_size, nd=nd))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer, kernel_size=kernel_size, nd=nd))
    if secondary_block is not None:
        layers.append(secondary_block(self.inplanes, nd=nd))
    return nn.Sequential(*layers)


def make_res_layer(block, inplanes, planes, blocks, norm_layer=nn.BatchNorm2d, base_width=64, groups=1, stride=1, dilation=1, dilate=False, nd=2, secondary_block=None, downsample_method=None, kernel_size=3, **kwargs) ->nn.Module:
    """

    Args:
        block: Module class. For example `BasicBlock` or `Bottleneck`.
        inplanes: Number of in planes
        planes: Number of planes
        blocks: Number of blocks
        norm_layer: Norm Module class
        base_width: Base width. Acts as a factor of the bottleneck size of the Bottleneck block and is used with groups.
        groups:
        stride:
        dilation:
        dilate:
        nd:
        secondary_block:
        downsample_method:
        kernel_size:
        kwargs:

    Returns:

    """
    norm_layer = lookup_nn(norm_layer, nd=nd, call=False)
    d = Dict(inplanes=inplanes, _norm_layer=norm_layer, base_width=base_width, groups=groups, dilation=dilation)
    return _make_layer(self=d, block=block, planes=planes, blocks=blocks, stride=stride, dilate=dilate, nd=nd, secondary_block=secondary_block, downsample_method=downsample_method, kernel_size=kernel_size)


class BasicBlock(nn.Module):
    expansion: 'int' = tvr.BasicBlock.expansion
    forward = tvr.BasicBlock.forward

    def __init__(self, inplanes: 'int', planes: 'int', stride: 'int'=1, downsample: 'Optional[nn.Module]'=None, groups: 'int'=1, base_width: 'int'=64, dilation: 'int'=1, norm_layer='batchnorm2d', kernel_size=3, nd=2) ->None:
        super().__init__()
        norm_layer = lookup_nn(norm_layer, call=False, nd=nd)
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride, nd=nd, kernel_size=kernel_size)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, nd=nd)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride


def _default_res_kwargs(backbone_kwargs, pretrained=False):
    kw = dict(fused_initial=False, pretrained=pretrained)
    kw.update({} if backbone_kwargs is None else backbone_kwargs)
    return kw


class LastLevelMaxPool(ExtraFPNBlock):

    def __init__(self, nd=2):
        """
        This is an adapted class from torchvision to support n-dimensional data.

        References:
            https://github.com/pytorch/vision/blob/d2d448c71b4cb054d160000a0f63eecad7867bdb/torchvision/ops/feature_pyramid_network.py#L207

        Notes:
            This class just applies stride 2 to spatial dimensions, and uses pytorch's max_pool function to do it.
        """
        super().__init__()
        self._fn = lookup_nn('max_pool2d', nd=nd, call=False, src=F)

    def adapt_out_channel_list(self, channel_list):
        return channel_list + channel_list[-1:]

    def forward(self, x: 'List[Tensor]', y: 'List[Tensor]', names: 'List[str]') ->Tuple[List[Tensor], List[str]]:
        names.append('pool')
        x.append(self._fn(x[-1], 1, 2, 0))
        return x, names


class BackboneWithFPN(backbone_utils.BackboneWithFPN):

    def __init__(self, backbone: 'nn.Module', return_layers: 'Dict[str, str]', in_channels_list: 'List[int]', out_channels: 'int', out_channel_list: 'List[int]', extra_blocks: "Optional['ExtraFPNBlock']"=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, ilg=None, nd: 'int'=2, **kwargs) ->None:
        super(backbone_utils.BackboneWithFPN, self).__init__()
        if ilg is None:
            ilg = isinstance(backbone, nn.Sequential)
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool(nd=nd)
            if hasattr(extra_blocks, 'adapt_out_channel_list'):
                out_channel_list = extra_blocks.adapt_out_channel_list(out_channel_list)
        pretrained_cfg = backbone.__dict__.get('pretrained_cfg', {})
        if not len(pretrained_cfg) and hasattr(backbone, 'hparams') and isinstance(backbone.hparams, dict):
            pretrained_cfg = backbone.hparams.get('pretrained_cfg', pretrained_cfg)
        if kwargs.pop('normalize', True):
            self.normalize = Normalize(mean=kwargs.get('inputs_mean', pretrained_cfg.get('mean', 0.0)), std=kwargs.get('inputs_std', pretrained_cfg.get('std', 1.0)), assert_range=kwargs.get('assert_range', (0.0, 1.0)))
        else:
            self.normalize = None
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers) if ilg else backbone
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer, nd=nd)
        self.out_channels = out_channel_list

    def forward(self, x: 'Tensor') ->Dict[str, Tensor]:
        if self.normalize is not None:
            x = self.normalize(x)
        x = self.body(x)
        x = self.fpn(x)
        return x


def resolve_model(model_name, model_parameters, verbose=True, **kwargs):
    if isinstance(model_name, nn.Module):
        model = model_name
    elif callable(model_name):
        model = model_name(map_location='cpu')
    elif model_name.endswith('.ckpt'):
        if len(kwargs):
            warn(f'Cannot use kwargs when loading Lightning Checkpoints. Ignoring the following: {kwargs}')
        model = cd.models.LitCpn.load_from_checkpoint(model_name, map_location='cpu')
    else:
        model = cd.load_model(model_name, map_location='cpu', **kwargs)
    if not isinstance(model, cd.models.LitCpn):
        if verbose:
            None
        model = cd.models.LitCpn(model, **kwargs)
    model.model.max_imsize = None
    model.eval()
    model.requires_grad_(False)
    if model_parameters is not None:
        for k, v in model_parameters.items():
            if hasattr(model.model, k):
                setattr(model.model, k, type(getattr(model.model, k))(v))
            else:
                raise ValueError(f'Could not find attribute {k} in model {model_name}! Please check your configuration.')
    return model


def model2dict(model: "'nn.Module'"):
    kwargs = dict(model.hparams)
    updated_kwargs = dict()
    for k, v in kwargs.items():
        if k in model.__dict__:
            cv = model.__dict__[k]
            r = v != cv
            if hasattr(r, 'any'):
                r = r.any()
            if r:
                updated_kwargs[k] = cv
    return dict(model=model.__class__.__name__, kwargs=kwargs, updated_kwargs=updated_kwargs)


def update_model_hparams_(obj, resolve=True, **kwargs):
    assert hasattr(obj, '_set_hparams')
    assert hasattr(obj, '_hparams_initial')
    assert hasattr(obj, '_hparams')
    changes = {}
    for key, value in kwargs.items():
        if isinstance(value, nn.Module):
            if resolve:
                value = model2dict(value)
        changes[key] = value
    if len(changes):
        obj._set_hparams(changes)
        obj._hparams_initial = copy.deepcopy(obj._hparams)


def init_modules_(mod: 'nn.Module'):
    for m in mod.modules():
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


class MobileNetV3Base(nn.Sequential):
    """Adaptation of torchvision.models.mobilenetv3.MobileNetV3"""

    def __init__(self, in_channels, inverted_residual_setting: 'List[InvertedResidualConfig]', block: 'Optional[Callable[..., nn.Module]]'=None, norm_layer: 'Optional[Callable[..., nn.Module]]'=None, **kwargs: Any) ->None:
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError('The inverted_residual_setting should not be empty')
        elif not (isinstance(inverted_residual_setting, Sequence) and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError('The inverted_residual_setting should be List[InvertedResidualConfig]')
        if block is None:
            block = InvertedResidual
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        layers: 'List[nn.Sequential]' = [nn.Sequential()]
        cbna_kw = {}
        if norm_layer is not None:
            cbna_kw['norm_layer'] = norm_layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.out_channels = [firstconv_output_channels]
        layers[-1].add_module(str(len(layers[-1])), ConvBNActivation(in_channels, firstconv_output_channels, kernel_size=3, stride=2, activation_layer=nn.Hardswish, **cbna_kw))
        for cnf in inverted_residual_setting:
            if cnf.stride > 1:
                layers.append(nn.Sequential())
                self.out_channels.append(cnf.out_channels)
            else:
                self.out_channels[-1] = cnf.out_channels
            layers[-1].add_module(str(len(layers[-1])), block(cnf, norm_layer))
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.out_channels[-1] = lastconv_output_channels
        assert len(self.out_channels) == len(layers)
        layers[-1].add_module(str(len(layers[-1])), ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1, activation_layer=nn.Hardswish, **cbna_kw))
        super().__init__(*layers)
        init_modules_(self)


class MobileNetV3Small(MobileNetV3Base):

    def __init__(self, in_channels, width_mult: 'float'=1.0, reduced_tail: 'bool'=False, dilated: 'bool'=False):
        super().__init__(in_channels=in_channels, inverted_residual_setting=_mobilenet_v3_conf('mobilenet_v3_small', width_mult=width_mult, reduced_tail=reduced_tail, dilated=dilated)[0])


def _ni_3d(nd):
    if nd != 2:
        raise NotImplementedError('The `nd` option is not yet available for this model.')


class MobileNetV3Large(MobileNetV3Base):

    def __init__(self, in_channels, width_mult: 'float'=1.0, reduced_tail: 'bool'=False, dilated: 'bool'=False):
        super().__init__(in_channels=in_channels, inverted_residual_setting=_mobilenet_v3_conf('mobilenet_v3_large', width_mult=width_mult, reduced_tail=reduced_tail, dilated=dilated)[0])


class MultiscaleFusionAttention(nn.Module):

    def __init__(self, in_channels, in_channels2, out_channels, norm_layer='BatchNorm2d', activation='relu', compression=16, interpolation=None, nd=2):
        super().__init__()
        kw = dict(activation=activation, norm_layer=norm_layer, nd=nd, bias=False)
        self.in_block = nn.Sequential(ConvNormRelu(in_channels, in_channels, **kw), ConvNormRelu(in_channels, in_channels2, kernel_size=1, padding=0, **kw))
        self.se_high = SqueezeExcitation(in_channels2, compression=compression, activation=activation, nd=nd)
        self.se_low = SqueezeExcitation(in_channels2, compression=compression, activation=activation, nd=nd)
        self.out_block = nn.Sequential(ConvNormRelu(in_channels2 + in_channels2, out_channels, **kw), ConvNormRelu(out_channels, out_channels, **kw))
        if interpolation is True:
            interpolation = 'nearest'
        elif interpolation is False:
            interpolation = None
        self.interpolation = interpolation

    def forward(self, x, x2=None):
        if isinstance(x, (tuple, list)):
            assert x2 is None
            x, x2 = x
        x = self.in_block(x)
        if self.interpolation is not None:
            x = F.interpolate(x, x2.shape[2:], mode=self.interpolation)
        if x2 is not None:
            a = self.se_high(x)
            b = self.se_low(x2)
            x = x * (a + b)
            x = torch.cat((x, x2), 1)
        return self.out_block(x)


class IntermediateUNetBlock(nn.Module):

    def __init__(self, out_channels: 'Tuple[int]', out_strides: 'Tuple[int]'):
        super().__init__()
        self.out_channels = out_channels
        self.out_strides = out_strides

    def forward(self, x: 'Dict[str, Tensor]') ->Dict[str, Tensor]:
        pass


class PositionWiseAttention(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=64, kernel_size=3, padding=1, beta=False, nd=2):
        super().__init__()
        Conv = lookup_nn('Conv2d', nd=nd, call=False)
        self.beta = nn.Parameter(torch.zeros(1)) if beta else 1.0
        if in_channels != out_channels:
            self.in_conv = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.in_conv = None
        self.proj_b, self.proj_a = [Conv(out_channels, mid_channels, kernel_size=1) for _ in range(2)]
        self.proj = Conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.out_conv = Conv(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        x = inputs if self.in_conv is None else self.in_conv(inputs)
        a = self.proj_a(x).flatten(2)
        b = self.proj_b(x).flatten(2)
        p = torch.matmul(a.permute(0, 2, 1), b)
        p = softmax(p.flatten(1), dim=1).view(p.shape)
        c = self.proj(x).flatten(2)
        out = torch.matmul(p, c.permute(0, 2, 1)).view(*c.shape[:2], *inputs.shape[2:])
        out = self.out_conv(self.beta * out + x)
        return out


class PAB(IntermediateUNetBlock):

    def __init__(self, in_channels, out_channels, mid_channels=64, kernel_size=3, padding=1, nd=2, replace=False):
        kwargs = dict(out_channels=(out_channels,), out_strides=(1,))
        if replace:
            kwargs = dict(out_channels=(), out_strides=())
        super().__init__(**kwargs)
        self.module = PositionWiseAttention(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels, kernel_size=kernel_size, padding=padding, nd=nd)
        self.replace = replace

    def forward(self, x: 'Dict[str, Tensor]') ->Dict[str, Tensor]:
        in_key = list(x.keys())[-1]
        out_key = in_key if self.replace else str(len(x))
        x[out_key] = self.module(x[in_key])
        return x


HOSTED_MODELS = dict(ginoro='ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c')


HOST_TEMPLATE = 'https://celldetection.org/torch/models/{name}'

