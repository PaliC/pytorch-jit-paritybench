import sys
_module = sys.modules[__name__]
del sys
image_augmentation = _module
models = _module
padding_same_conv = _module
train = _module
training_data = _module
umeyama = _module
util = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.utils.data


from torch import nn


from torch import optim


from torch.nn import functional as F


import math


from torch.nn.parameter import Parameter


from torch.nn.functional import pad


from torch.nn.modules import Module


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


import numpy as np


import torch.backends.cudnn as cudnn


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1,
    dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] +
        effective_filter_size_rows - input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) *
        dilation[0] + 1 - input_rows)
    rows_odd = padding_rows % 2 != 0
    padding_cols = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) *
        dilation[0] + 1 - input_rows)
    cols_odd = padding_rows % 2 != 0
    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.conv2d(input, weight, bias, stride, padding=(padding_rows // 2,
        padding_cols // 2), dilation=dilation, groups=groups)


class _ConvLayer(nn.Sequential):

    def __init__(self, input_features, output_features):
        super(_ConvLayer, self).__init__()
        self.add_module('conv2', Conv2d(input_features, output_features,
            kernel_size=5, stride=2))
        self.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))


class _UpScale(nn.Sequential):

    def __init__(self, input_features, output_features):
        super(_UpScale, self).__init__()
        self.add_module('conv2_', Conv2d(input_features, output_features * 
            4, kernel_size=3))
        self.add_module('leakyrelu', nn.LeakyReLU(0.1, inplace=True))
        self.add_module('pixelshuffler', _PixelShuffler())


class Flatten(nn.Module):

    def forward(self, input):
        output = input.view(input.size(0), -1)
        return output


class Reshape(nn.Module):

    def forward(self, input):
        output = input.view(-1, 1024, 4, 4)
        return output


class _PixelShuffler(nn.Module):

    def forward(self, input):
        batch_size, c, h, w = input.size()
        rh, rw = 2, 2
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)
        out = input.view(batch_size, rh, rw, oc, h, w)
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous()
        out = out.view(batch_size, oc, oh, ow)
        return out


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(_ConvLayer(3, 128), _ConvLayer(128, 
            256), _ConvLayer(256, 512), _ConvLayer(512, 1024), Flatten(),
            nn.Linear(1024 * 4 * 4, 1024), nn.Linear(1024, 1024 * 4 * 4),
            Reshape(), _UpScale(1024, 512))
        self.decoder_A = nn.Sequential(_UpScale(512, 256), _UpScale(256, 
            128), _UpScale(128, 64), Conv2d(64, 3, kernel_size=5, padding=1
            ), nn.Sigmoid())
        self.decoder_B = nn.Sequential(_UpScale(512, 256), _UpScale(256, 
            128), _UpScale(128, 64), Conv2d(64, 3, kernel_size=5, padding=1
            ), nn.Sigmoid())

    def forward(self, x, select='A'):
        if select == 'A':
            out = self.encoder(x)
            out = self.decoder_A(out)
        else:
            out = self.encoder(x)
            out = self.decoder_B(out)
        return out


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels //
                groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(out_channels, in_channels //
                groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = (
            '{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
            )
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Oldpan_Faceswap_Deepfake_Pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Reshape(*[], **{}), [torch.rand([4, 1024, 4, 4])], {})

    def test_002(self):
        self._check(_PixelShuffler(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

