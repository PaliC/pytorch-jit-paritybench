import sys
_module = sys.modules[__name__]
del sys
setup_dataset_kitti = _module
setup_dataset_nyu_v2 = _module
setup_dataset_void = _module
data_utils = _module
datasets = _module
eval_utils = _module
global_constants = _module
kbnet = _module
kbnet_model = _module
log_utils = _module
losses = _module
net_utils = _module
networks = _module
posenet_model = _module
run_kbnet = _module
train_kbnet = _module
transforms = _module

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


import torch.utils.data


import time


import torch


from torch.utils.tensorboard import SummaryWriter


import torchvision


from matplotlib import pyplot as plt


class Conv2d(torch.nn.Module):
    """
    2D convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False):
        super(Conv2d, self).__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))
        self.activation_func = activation_func
        assert not (use_batch_norm and use_instance_norm), 'Unable to apply both batch and instance normalization'
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm
        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        """
        Forward input x through a convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        conv = self.conv(x)
        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)
        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class DepthwiseSeparableConv2d(torch.nn.Module):
    """
    Depthwise separable convolution class
    Performs
    1. separate k x k convolution per channel (depth-wise)
    2. 1 x 1 convolution across all channels (point-wise)

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel (k x k)
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        padding = kernel_size // 2
        self.conv_depthwise = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=in_channels)
        self.conv_pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv_depthwise.weight)
            torch.nn.init.kaiming_normal_(self.conv_pointwise.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv_depthwise.weight)
            torch.nn.init.xavier_normal_(self.conv_pointwise.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv_depthwise.weight)
            torch.nn.init.xavier_uniform_(self.conv_pointwise.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))
        self.conv = torch.nn.Sequential(self.conv_depthwise, self.conv_pointwise)
        assert not (use_batch_norm and use_instance_norm), 'Unable to apply both batch and instance normalization'
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm
        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)
        self.activation_func = activation_func

    def forward(self, x):
        """
        Forward input x through a depthwise convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        conv = self.conv(x)
        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)
        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class AtrousConv2d(torch.nn.Module):
    """
    2D atrous convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        dilation : int
            dilation of convolution (skips rate - 1 pixels)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False):
        super(AtrousConv2d, self).__init__()
        padding = dilation
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=dilation, padding=padding, bias=False)
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))
        self.activation_func = activation_func
        assert not (use_batch_norm and use_instance_norm), 'Unable to apply both batch and instance normalization'
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm
        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        """
        Forward input x through an atrous convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        conv = self.conv(x)
        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)
        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class TransposeConv2d(torch.nn.Module):
    """
    Transpose convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False):
        super(TransposeConv2d, self).__init__()
        padding = kernel_size // 2
        self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False)
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.deconv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))
        self.activation_func = activation_func
        assert not (use_batch_norm and use_instance_norm), 'Unable to apply both batch and instance normalization'
        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm
        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        """
        Forward input x through a transposed convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        """
        deconv = self.deconv(x)
        if self.use_batch_norm:
            deconv = self.batch_norm(deconv)
        elif self.use_instance_norm:
            deconv = self.instance_norm(deconv)
        if self.activation_func is not None:
            return self.activation_func(deconv)
        else:
            return deconv


class UpConv2d(torch.nn.Module):
    """
    Up-convolution (upsample + convolution) block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        shape : list[int]
            two element tuple of ints (height, width)
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False):
        super(UpConv2d, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)

    def forward(self, x, shape):
        """
        Forward input x through an up convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        """
        upsample = torch.nn.functional.interpolate(x, size=shape, mode='nearest')
        conv = self.conv(upsample)
        return conv


class FullyConnected(torch.nn.Module):
    """
    Fully connected layer

    Arg(s):
        in_channels : int
            number of input neurons
        out_channels : int
            number of output neurons
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        dropout_rate : float
            probability to use dropout
    """

    def __init__(self, in_features, out_features, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), dropout_rate=0.0):
        super(FullyConnected, self).__init__()
        self.fully_connected = torch.nn.Linear(in_features, out_features)
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.fully_connected.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.fully_connected.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.fully_connected.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))
        self.activation_func = activation_func
        if dropout_rate > 0.0 and dropout_rate <= 1.0:
            self.dropout = torch.nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        """
        Forward input x through a fully connected block

        Arg(s):
            x : torch.Tensor[float32]
                N x C input tensor
        Returns:
            torch.Tensor[float32] : N x K output tensor
        """
        fully_connected = self.fully_connected(x)
        if self.activation_func is not None:
            fully_connected = self.activation_func(fully_connected)
        if self.dropout is not None:
            return self.dropout(fully_connected)
        else:
            return fully_connected


class ResNetBlock(torch.nn.Module):
    """
    Basic ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    """

    def __init__(self, in_channels, out_channels, stride=1, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False, use_depthwise_separable=False):
        super(ResNetBlock, self).__init__()
        self.activation_func = activation_func
        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.conv2 = conv2d(out_channels, out_channels, kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.projection = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, weight_initializer=weight_initializer, activation_func=None, use_batch_norm=False, use_instance_norm=False)

    def forward(self, x):
        """
        Forward input x through a basic ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x
        return self.activation_func(conv2 + X)


class ResNetBottleneckBlock(torch.nn.Module):
    """
    ResNet bottleneck block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    """

    def __init__(self, in_channels, out_channels, stride=1, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False, use_depthwise_separable=False):
        super(ResNetBottleneckBlock, self).__init__()
        self.activation_func = activation_func
        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.conv2 = conv2d(out_channels, out_channels, kernel_size=3, stride=stride, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.conv3 = conv2d(out_channels, 4 * out_channels, kernel_size=1, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.projection = Conv2d(in_channels, 4 * out_channels, kernel_size=1, stride=stride, weight_initializer=weight_initializer, activation_func=None, use_batch_norm=False, use_instance_norm=False)

    def forward(self, x):
        """
        Forward input x through a ResNet bottleneck block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x
        return self.activation_func(conv3 + X)


class AtrousResNetBlock(torch.nn.Module):
    """
    Basic atrous ResNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dilation : int
            dilation of convolution (skips rate - 1 pixels)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    """

    def __init__(self, in_channels, out_channels, dilation=2, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False, use_depthwise_separable=False):
        super(AtrousResNetBlock, self).__init__()
        self.activation_func = activation_func
        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d
        self.conv1 = AtrousConv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.conv2 = conv2d(out_channels, out_channels, kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.projection = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, weight_initializer=weight_initializer, activation_func=None, use_batch_norm=False, use_instance_norm=False)

    def forward(self, x):
        """
        Forward input x through an atrous ResNet block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x
        return self.activation_func(conv2 + X)


class VGGNetBlock(torch.nn.Module):
    """
    VGGNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_convolution : int
            number of convolution layers
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    """

    def __init__(self, in_channels, out_channels, n_convolution=1, stride=1, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False, use_depthwise_separable=False):
        super(VGGNetBlock, self).__init__()
        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d
        layers = []
        for n in range(n_convolution - 1):
            conv = conv2d(in_channels, out_channels, kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
            layers.append(conv)
            in_channels = out_channels
        conv = conv2d(in_channels, out_channels, kernel_size=3, stride=stride, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        layers.append(conv)
        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward input x through a VGG block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        return self.conv_block(x)


class AtrousVGGNetBlock(torch.nn.Module):
    """
    Atrous VGGNet block class
    (last block performs atrous convolution instead of convolution with stride)

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_convolution : int
            number of convolution layers
        dilation : int
            dilation of atrous convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    """

    def __init__(self, in_channels, out_channels, n_convolution=1, dilation=2, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False, use_depthwise_separable=False):
        super(AtrousVGGNetBlock, self).__init__()
        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d
        layers = []
        for n in range(n_convolution - 1):
            conv = conv2d(in_channels, out_channels, kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
            layers.append(conv)
            in_channels = out_channels
        conv = AtrousConv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        layers.append(conv)
        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward input x through an atrous VGG block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        return self.conv_block(x)


class AtrousSpatialPyramidPooling(torch.nn.Module):
    """
    Atrous Spatial Pyramid Pooling class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        dilations : list[int]
            dilations for different atrous convolution of each branch
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """

    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18], weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False):
        super(AtrousSpatialPyramidPooling, self).__init__()
        output_channels = out_channels // (len(dilations) + 1)
        self.conv1 = Conv2d(in_channels, output_channels, kernel_size=1, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.atrous_convs = torch.nn.ModuleList()
        for dilation in dilations:
            atrous_conv = AtrousConv2d(in_channels, output_channels, kernel_size=3, dilation=dilation, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
            self.atrous_convs.append(atrous_conv)
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.global_pool_conv = Conv2d(in_channels, output_channels, kernel_size=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.conv_fuse = Conv2d((len(dilations) + 2) * output_channels, out_channels, kernel_size=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=False, use_instance_norm=False)

    def forward(self, x):
        """
        Forward input x through a ASPP block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        branches = []
        branches.append(self.conv1(x))
        for atrous_conv in self.atrous_convs:
            branches.append(atrous_conv(x))
        global_pool = self.global_pool(x)
        global_pool = self.global_pool_conv(global_pool)
        global_pool = torch.nn.functional.interpolate(global_pool, size=x.shape[2:], mode='bilinear', align_corners=True)
        branches.append(global_pool)
        return self.conv_fuse(torch.cat(branches, dim=1))


class SpatialPyramidPooling(torch.nn.Module):
    """
    Spatial Pyramid Pooling class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_sizes : list[int]
            pooling kernel size of each branch
        pool_func : str
            max, average
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """

    def __init__(self, in_channels, out_channels, kernel_sizes, pool_func='max', weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False):
        super(SpatialPyramidPooling, self).__init__()
        output_channels = out_channels // len(kernel_sizes)
        self.kernel_sizes = kernel_sizes
        if pool_func == 'max':
            self.pool_func = torch.nn.functional.max_pool2d
        elif pool_func == 'average':
            self.pool_func = torch.nn.functional.avg_pool2d
        else:
            raise ValueError('Unsupported pooling function: {}'.format(pool_func))
        self.convs = torch.nn.ModuleList()
        for n in kernel_sizes:
            conv = Conv2d(in_channels, output_channels, kernel_size=1, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
            self.convs.append(conv)
        self.conv_fuse = torch.nn.Sequential(Conv2d(2 * len(kernel_sizes) * output_channels, out_channels, kernel_size=3, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm), Conv2d(out_channels, out_channels, kernel_size=1, weight_initializer=weight_initializer, activation_func=None, use_batch_norm=False, use_instance_norm=False))

    def forward(self, x):
        """
        Forward input x through SPP block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        branches = [x]
        for kernel_size, conv in zip(self.kernel_sizes, self.convs):
            pool = self.pool_func(x, kernel_size=(kernel_size, kernel_size), stride=(kernel_size, kernel_size))
            pool = torch.nn.functional.interpolate(pool, size=x.shape[2:], mode='bilinear', align_corners=True)
            branches.append(conv(pool))
        return self.conv_fuse(torch.cat(branches, dim=1))


class CalibratedBackprojectionBlock(torch.nn.Module):
    """
    Calibrated backprojection (KB) layer class

    Arg(s):
        in_channels_image : int
            number of input channels for image (RGB) branch
        in_channels_depth : int
            number of input channels for depth branch
        in_channels_fused : int
            number of input channels for RGB 3D fusion branch
        n_filter_image : int
            number of filters for image (RGB) branch
        n_filter_depth : int
            number of filters for depth branch
        n_filter_fused : int
            number of filters for RGB 3D fusion branch
        n_convolution_image : int
            number of convolution layers in image branch
        n_convolution_depth : int
            number of convolution layers in depth branch
        n_convolution_fused : int
            number of convolution layers in RGB 3D fusion branch
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    """

    def __init__(self, in_channels_image, in_channels_depth, in_channels_fused, n_filter_image=48, n_filter_depth=16, n_filter_fused=48, n_convolution_image=1, n_convolution_depth=1, n_convolution_fused=1, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)):
        super(CalibratedBackprojectionBlock, self).__init__()
        self.conv_image = VGGNetBlock(in_channels=in_channels_image, out_channels=n_filter_image, n_convolution=n_convolution_image, stride=2, weight_initializer=weight_initializer, activation_func=activation_func)
        self.conv_depth = VGGNetBlock(in_channels=in_channels_depth + 3, out_channels=n_filter_depth, n_convolution=n_convolution_depth, stride=2, weight_initializer=weight_initializer, activation_func=activation_func)
        self.proj_depth = Conv2d(in_channels=in_channels_depth, out_channels=1, kernel_size=1, stride=1, weight_initializer=weight_initializer, activation_func=activation_func)
        self.conv_fused = Conv2d(in_channels=in_channels_fused + 3, out_channels=n_filter_fused, kernel_size=1, stride=2, weight_initializer=weight_initializer, activation_func=activation_func)

    def forward(self, image, depth, coordinates, fused=None):
        layers_fused = []
        conv_image = self.conv_image(image)
        conv_depth = self.conv_depth(torch.cat([depth, coordinates], dim=1))
        layers_fused.append(image)
        z = self.proj_depth(depth)
        xyz = coordinates * z
        layers_fused.append(xyz)
        if fused is not None:
            layers_fused.append(fused)
        layers_fused = torch.cat(layers_fused, dim=1)
        conv_fused = self.conv_fused(layers_fused)
        return conv_image, conv_depth, conv_fused


class DecoderBlock(torch.nn.Module):
    """
    Decoder block with skip connection

    Arg(s):
        in_channels : int
            number of input channels
        skip_channels : int
            number of skip connection channels
        out_channels : int
            number of output channels
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        deconv_type : str
            deconvolution types: transpose, up
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    """

    def __init__(self, in_channels, skip_channels, out_channels, weight_initializer='kaiming_uniform', activation_func=torch.nn.LeakyReLU(negative_slope=0.1, inplace=True), use_batch_norm=False, use_instance_norm=False, deconv_type='up', use_depthwise_separable=False):
        super(DecoderBlock, self).__init__()
        self.skip_channels = skip_channels
        self.deconv_type = deconv_type
        if deconv_type == 'transpose':
            self.deconv = TransposeConv2d(in_channels, out_channels, kernel_size=3, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        elif deconv_type == 'up':
            self.deconv = UpConv2d(in_channels, out_channels, kernel_size=3, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        concat_channels = skip_channels + out_channels
        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d
        self.conv = conv2d(concat_channels, out_channels, kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)

    def forward(self, x, skip=None, shape=None):
        """
        Forward input x through a decoder block and fuse with skip connection

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            skip : torch.Tensor[float32]
                N x F x H x W skip connection
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        """
        if self.deconv_type == 'transpose':
            deconv = self.deconv(x)
        elif self.deconv_type == 'up':
            if skip is not None:
                shape = skip.shape[2:4]
            elif shape is not None:
                pass
            else:
                n_height, n_width = x.shape[2:4]
                shape = int(2 * n_height), int(2 * n_width)
            deconv = self.deconv(x, shape=shape)
        if self.skip_channels > 0:
            concat = torch.cat([deconv, skip], dim=1)
        else:
            concat = deconv
        return self.conv(concat)


class KBNetEncoder(torch.nn.Module):
    """
    Calibrated backprojection network (KBNet) encoder with skip connections

    Arg(s):
        in_channels_image : int
            number of input channels for image (RGB) branch
        in_channels_depth : int
            number of input channels for depth branch
        n_filters_image : int
            number of filters for image (RGB) branch for each KB layer
         n_filters_depth : int
            number of filters for depth branch  for each KB layer
        n_filters_fused : int
            number of filters for RGB 3D fusion branch  for each KB layer
        n_convolution_image : list[int]
            number of convolution layers in image branch  for each KB layer
        n_convolution_depth : list[int]
            number of convolution layers in depth branch  for each KB layer
        n_convolution_fused : list[int]
            number of convolution layers in RGB 3D fusion branch  for each KB layer
        resolutions_backprojection : list[int]
            resolutions at which to use calibrated backprojection layers
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    """

    def __init__(self, input_channels_image=3, input_channels_depth=1, n_filters_image=[48, 96, 192, 384, 384], n_filters_depth=[16, 32, 64, 128, 128], n_filters_fused=[48, 96, 192, 384, 384], n_convolutions_image=[1, 1, 1, 1, 1], n_convolutions_depth=[1, 1, 1, 1, 1], n_convolutions_fused=[1, 1, 1, 1, 1], resolutions_backprojection=[0, 1, 2], weight_initializer='kaiming_uniform', activation_func='leaky_relu'):
        super(KBNetEncoder, self).__init__()
        self.resolutions_backprojection = resolutions_backprojection
        network_depth = 5
        assert len(n_convolutions_image) == network_depth
        assert len(n_convolutions_depth) == network_depth
        assert len(n_convolutions_fused) == network_depth
        assert len(n_filters_image) == network_depth
        assert len(n_filters_depth) == network_depth
        assert len(n_filters_fused) == network_depth
        activation_func = net_utils.activation_func(activation_func)
        n = 0
        if n in resolutions_backprojection:
            self.conv0_image = net_utils.Conv2d(in_channels=input_channels_image, out_channels=n_filters_image[n], kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=activation_func)
            self.conv0_depth = net_utils.Conv2d(in_channels=input_channels_depth, out_channels=n_filters_depth[n], kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=activation_func)
            in_channels_image = n_filters_image[n]
            in_channels_depth = n_filters_depth[n]
            in_channels_fused = n_filters_image[n]
            self.calibrated_backprojection1 = net_utils.CalibratedBackprojectionBlock(in_channels_image=in_channels_image, in_channels_depth=in_channels_depth, in_channels_fused=in_channels_fused, n_filter_image=n_filters_image[n], n_filter_depth=n_filters_depth[n], n_filter_fused=n_filters_fused[n], n_convolution_image=n_convolutions_image[n], n_convolution_depth=n_convolutions_depth[n], n_convolution_fused=n_convolutions_fused[n], weight_initializer=weight_initializer, activation_func=activation_func)
        else:
            self.conv1_image = net_utils.VGGNetBlock(in_channels=input_channels_image, out_channels=n_filters_image[n], n_convolution=n_convolutions_image[n], stride=2, weight_initializer=weight_initializer, activation_func=activation_func)
            self.conv1_depth = net_utils.VGGNetBlock(in_channels=input_channels_depth, out_channels=n_filters_depth[n], n_convolution=n_convolutions_depth[n], stride=2, weight_initializer=weight_initializer, activation_func=activation_func)
        n = 1
        in_channels_image = n_filters_image[n - 1]
        in_channels_depth = n_filters_depth[n - 1]
        if n in resolutions_backprojection:
            if n - 1 in resolutions_backprojection:
                in_channels_fused = n_filters_image[n - 1] + n_filters_fused[n - 1]
            else:
                in_channels_fused = n_filters_image[n - 1]
            self.calibrated_backprojection2 = net_utils.CalibratedBackprojectionBlock(in_channels_image=in_channels_image, in_channels_depth=in_channels_depth, in_channels_fused=in_channels_fused, n_filter_image=n_filters_image[n], n_filter_depth=n_filters_depth[n], n_filter_fused=n_filters_fused[n], n_convolution_image=n_convolutions_image[n], n_convolution_depth=n_convolutions_depth[n], n_convolution_fused=n_convolutions_fused[n], weight_initializer=weight_initializer, activation_func=activation_func)
        else:
            self.conv2_image = net_utils.VGGNetBlock(in_channels=in_channels_image, out_channels=n_filters_image[n], n_convolution=n_convolutions_image[n], stride=2, weight_initializer=weight_initializer, activation_func=activation_func)
            self.conv2_depth = net_utils.VGGNetBlock(in_channels=in_channels_depth, out_channels=n_filters_depth[n], n_convolution=n_convolutions_depth[n], stride=2, weight_initializer=weight_initializer, activation_func=activation_func)
        n = 2
        in_channels_image = n_filters_image[n - 1]
        in_channels_depth = n_filters_depth[n - 1]
        if n in resolutions_backprojection:
            if n - 1 in resolutions_backprojection:
                in_channels_fused = n_filters_image[n - 1] + n_filters_fused[n - 1]
            else:
                in_channels_fused = n_filters_image[n - 1]
            self.calibrated_backprojection3 = net_utils.CalibratedBackprojectionBlock(in_channels_image=in_channels_image, in_channels_depth=in_channels_depth, in_channels_fused=in_channels_fused, n_filter_image=n_filters_image[n], n_filter_depth=n_filters_depth[n], n_filter_fused=n_filters_fused[n], n_convolution_image=n_convolutions_image[n], n_convolution_depth=n_convolutions_depth[n], n_convolution_fused=n_convolutions_fused[n], weight_initializer=weight_initializer, activation_func=activation_func)
        else:
            self.conv3_image = net_utils.VGGNetBlock(in_channels=in_channels_image, out_channels=n_filters_image[n], n_convolution=n_convolutions_image[n], stride=2, weight_initializer=weight_initializer, activation_func=activation_func)
            self.conv3_depth = net_utils.VGGNetBlock(in_channels=in_channels_depth, out_channels=n_filters_depth[n], n_convolution=n_convolutions_depth[n], stride=2, weight_initializer=weight_initializer, activation_func=activation_func)
        n = 3
        in_channels_image = n_filters_image[n - 1]
        in_channels_depth = n_filters_depth[n - 1]
        if n in resolutions_backprojection:
            if n - 1 in resolutions_backprojection:
                in_channels_fused = n_filters_image[n - 1] + n_filters_fused[n - 1]
            else:
                in_channels_fused = n_filters_image[n - 1]
            self.calibrated_backprojection4 = net_utils.CalibratedBackprojectionBlock(in_channels_image=in_channels_image, in_channels_depth=in_channels_depth, in_channels_fused=in_channels_fused, n_filter_image=n_filters_image[n], n_filter_depth=n_filters_depth[n], n_filter_fused=n_filters_fused[n], n_convolution_image=n_convolutions_image[n], n_convolution_depth=n_convolutions_depth[n], n_convolution_fused=n_convolutions_fused[n], weight_initializer=weight_initializer, activation_func=activation_func)
        else:
            self.conv4_image = net_utils.VGGNetBlock(in_channels=in_channels_image, out_channels=n_filters_image[n], n_convolution=n_convolutions_image[n], stride=2, weight_initializer=weight_initializer, activation_func=activation_func)
            self.conv4_depth = net_utils.VGGNetBlock(in_channels=in_channels_depth, out_channels=n_filters_depth[n], n_convolution=n_convolutions_depth[n], stride=2, weight_initializer=weight_initializer, activation_func=activation_func)
        n = 4
        in_channels_image = n_filters_image[n - 1]
        in_channels_depth = n_filters_depth[n - 1]
        if n in resolutions_backprojection:
            if n - 1 in resolutions_backprojection:
                in_channels_fused = n_filters_image[n - 1] + n_filters_fused[n - 1]
            else:
                in_channels_fused = n_filters_image[n - 1]
            self.calibrated_backprojection5 = net_utils.CalibratedBackprojectionBlock(in_channels_image=in_channels_image, in_channels_depth=in_channels_depth, in_channels_fused=in_channels_fused, n_filter_image=n_filters_image[n], n_filter_depth=n_filters_depth[n], n_filter_fused=n_filters_fused[n], n_convolution_image=n_convolutions_image[n], n_convolution_depth=n_convolutions_depth[n], n_convolution_fused=n_convolutions_fused[n], weight_initializer=weight_initializer, activation_func=activation_func)
        else:
            self.conv5_image = net_utils.VGGNetBlock(in_channels=in_channels_image, out_channels=n_filters_image[n], n_convolution=n_convolutions_image[n], stride=2, weight_initializer=weight_initializer, activation_func=activation_func)
            self.conv5_depth = net_utils.VGGNetBlock(in_channels=in_channels_depth, out_channels=n_filters_depth[n], n_convolution=n_convolutions_depth[n], stride=2, weight_initializer=weight_initializer, activation_func=activation_func)

    def forward(self, image, depth, intrinsics):
        """
        Forward image, depth and calibration through encoder

        Arg(s):
            image : torch.Tensor[float32]
                N x C x H x W image
            depth : torch.Tensor[float32]
                N x 1 x H x W depth map
            intrinsics : torch.Tensor[float32]
                N x C x 3 x 3 calibration
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
            list[torch.Tensor[float32]] : list of skip connections
        """

        def camera_coordinates(batch, height, width, k):
            xy_h = net_utils.meshgrid(n_batch=batch, n_height=height, n_width=width, device=k.device, homogeneous=True)
            xy_h = xy_h.view(batch, 3, -1)
            coordinates = torch.matmul(torch.inverse(k), xy_h)
            coordinates = coordinates.view(n_batch, 3, height, width)
            return coordinates

        def scale_intrinsics(batch, height0, width0, height1, width1, k):
            device = k.device
            width0 = torch.tensor(width0, dtype=torch.float32, device=device)
            height0 = torch.tensor(height0, dtype=torch.float32, device=device)
            width1 = torch.tensor(width1, dtype=torch.float32, device=device)
            height1 = torch.tensor(height1, dtype=torch.float32, device=device)
            scale_x = n_width1 / n_width0
            scale_y = n_height1 / n_height0
            scale = torch.tensor([[scale_x, 1.0, scale_x], [1.0, scale_y, scale_y], [1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
            scale = scale.view(1, 3, 3).repeat(n_batch, 1, 1)
            return k * scale
        layers = []
        if 0 in self.resolutions_backprojection:
            n_batch, _, n_height0, n_width0 = image.shape
            coordinates0 = camera_coordinates(n_batch, n_height0, n_width0, intrinsics)
            conv0_image = self.conv0_image(image)
            conv0_depth = self.conv0_depth(depth)
            conv1_image, conv1_depth, conv1_fused = self.calibrated_backprojection1(image=conv0_image, depth=conv0_depth, coordinates=coordinates0, fused=None)
            skips1 = [conv1_fused, conv1_depth]
        else:
            conv1_image = self.conv1_image(image)
            conv1_depth = self.conv1_depth(depth)
            conv1_fused = None
            skips1 = [conv1_image, conv1_depth]
        layers.append(torch.cat(skips1, dim=1))
        _, _, n_height1, n_width1 = conv1_image.shape
        if 1 in self.resolutions_backprojection:
            intrinsics1 = scale_intrinsics(batch=n_batch, height0=n_height0, width0=n_width0, height1=n_height1, width1=n_width1, k=intrinsics)
            coordinates1 = camera_coordinates(n_batch, n_height1, n_width1, intrinsics1)
            conv2_image, conv2_depth, conv2_fused = self.calibrated_backprojection2(image=conv1_image, depth=conv1_depth, coordinates=coordinates1, fused=conv1_fused)
            skips2 = [conv2_fused, conv2_depth]
        else:
            if conv1_fused is not None:
                conv2_image = self.conv2_image(conv1_fused)
            else:
                conv2_image = self.conv2_image(conv1_image)
            conv2_depth = self.conv2_depth(conv1_depth)
            conv2_fused = None
            skips2 = [conv2_image, conv2_depth]
        layers.append(torch.cat(skips2, dim=1))
        _, _, n_height2, n_width2 = conv2_image.shape
        if 2 in self.resolutions_backprojection:
            intrinsics2 = scale_intrinsics(batch=n_batch, height0=n_height0, width0=n_width0, height1=n_height2, width1=n_width2, k=intrinsics)
            coordinates2 = camera_coordinates(n_batch, n_height2, n_width2, intrinsics2)
            conv3_image, conv3_depth, conv3_fused = self.calibrated_backprojection3(image=conv2_image, depth=conv2_depth, coordinates=coordinates2, fused=conv2_fused)
            skips3 = [conv3_fused, conv3_depth]
        else:
            if conv2_fused is not None:
                conv3_image = self.conv3_image(conv2_fused)
            else:
                conv3_image = self.conv3_image(conv2_image)
            conv3_depth = self.conv3_depth(conv2_depth)
            conv3_fused = None
            skips3 = [conv3_image, conv3_depth]
        layers.append(torch.cat(skips3, dim=1))
        _, _, n_height3, n_width3 = conv3_image.shape
        if 3 in self.resolutions_backprojection:
            intrinsics3 = scale_intrinsics(batch=n_batch, height0=n_height0, width0=n_width0, height1=n_height3, width1=n_width3, k=intrinsics)
            coordinates3 = camera_coordinates(n_batch, n_height3, n_width3, intrinsics3)
            conv4_image, conv4_depth, conv4_fused = self.calibrated_backprojection4(image=conv3_image, depth=conv3_depth, coordinates=coordinates3, fused=conv3_fused)
            skips4 = [conv4_fused, conv4_depth]
        else:
            if conv3_fused is not None:
                conv4_image = self.conv4_image(conv3_fused)
            else:
                conv4_image = self.conv4_image(conv3_image)
            conv4_depth = self.conv4_depth(conv3_depth)
            conv4_fused = None
            skips4 = [conv4_image, conv4_depth]
        layers.append(torch.cat(skips4, dim=1))
        _, _, n_height4, n_width4 = conv4_image.shape
        if 4 in self.resolutions_backprojection:
            intrinsics4 = scale_intrinsics(batch=n_batch, height0=n_height0, width0=n_width0, height1=n_height4, width1=n_width4, k=intrinsics)
            coordinates4 = camera_coordinates(n_batch, n_height4, n_width4, intrinsics4)
            conv5_image, conv5_depth, conv5_fused = self.calibrated_backprojection4(image=conv4_image, depth=conv4_depth, coordinates=coordinates4, fused=conv4_fused)
            skips5 = [conv5_fused, conv5_depth]
        else:
            if conv4_fused is not None:
                conv5_image = self.conv5_image(conv4_fused)
            else:
                conv5_image = self.conv5_image(conv4_image)
            conv5_depth = self.conv5_depth(conv4_depth)
            conv5_fused = None
            skips5 = [conv5_image, conv5_depth]
        layers.append(torch.cat(skips5, dim=1))
        return layers[-1], layers[0:-1]


class PoseEncoder(torch.nn.Module):
    """
    Pose network encoder

    Arg(s):
        input_channels : int
            number of channels in input data
        n_filters : list[int]
            number of filters to use for each convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """

    def __init__(self, input_channels=6, n_filters=[16, 32, 64, 128, 256, 256, 256], weight_initializer='kaiming_uniform', activation_func='leaky_relu', use_batch_norm=False, use_instance_norm=False):
        super(PoseEncoder, self).__init__()
        activation_func = net_utils.activation_func(activation_func)
        self.conv1 = net_utils.Conv2d(input_channels, n_filters[0], kernel_size=7, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.conv2 = net_utils.Conv2d(n_filters[0], n_filters[1], kernel_size=5, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.conv3 = net_utils.Conv2d(n_filters[1], n_filters[2], kernel_size=3, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.conv4 = net_utils.Conv2d(n_filters[2], n_filters[3], kernel_size=3, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.conv5 = net_utils.Conv2d(n_filters[3], n_filters[4], kernel_size=3, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.conv6 = net_utils.Conv2d(n_filters[4], n_filters[5], kernel_size=3, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.conv7 = net_utils.Conv2d(n_filters[5], n_filters[6], kernel_size=3, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)

    def forward(self, x):
        """
        Forward input x through encoder

        Arg(s):
            x : torch.Tensor[float32]
                input image N x C x H x W
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
            None
        """
        layers = [x]
        layers.append(self.conv1(layers[-1]))
        layers.append(self.conv2(layers[-1]))
        layers.append(self.conv3(layers[-1]))
        layers.append(self.conv4(layers[-1]))
        layers.append(self.conv5(layers[-1]))
        layers.append(self.conv6(layers[-1]))
        layers.append(self.conv7(layers[-1]))
        return layers[-1], None


class ResNetEncoder(torch.nn.Module):
    """
    ResNet encoder with skip connections

    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34, 50
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    """

    def __init__(self, n_layer, input_channels=3, n_filters=[32, 64, 128, 256, 256], weight_initializer='kaiming_uniform', activation_func='leaky_relu', use_batch_norm=False, use_instance_norm=False, use_depthwise_separable=False):
        super(ResNetEncoder, self).__init__()
        use_bottleneck = False
        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 50:
            n_blocks = [3, 4, 6, 3]
            use_bottleneck = True
            resnet_block = net_utils.ResNetBottleneckBlock
        else:
            raise ValueError('Only supports 18, 34, 50 layer architecture')
        for n in range(len(n_filters) - len(n_blocks) - 1):
            n_blocks = n_blocks + [n_blocks[-1]]
        assert len(n_filters) == len(n_blocks) + 1
        block_idx = 0
        filter_idx = 0
        activation_func = net_utils.activation_func(activation_func)
        in_channels, out_channels = [input_channels, n_filters[filter_idx]]
        self.conv1 = net_utils.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        filter_idx = filter_idx + 1
        blocks2 = []
        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=False)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=False)
            blocks2.append(block)
        self.blocks2 = torch.nn.Sequential(*blocks2)
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        blocks3 = []
        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(in_channels, out_channels, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=False)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=False)
            blocks3.append(block)
        self.blocks3 = torch.nn.Sequential(*blocks3)
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        blocks4 = []
        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(in_channels, out_channels, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
            blocks4.append(block)
        self.blocks4 = torch.nn.Sequential(*blocks4)
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        blocks5 = []
        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):
            if n == 0:
                in_channels = 4 * in_channels if use_bottleneck else in_channels
                block = resnet_block(in_channels, out_channels, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
            else:
                in_channels = 4 * out_channels if use_bottleneck else out_channels
                block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
            blocks5.append(block)
        self.blocks5 = torch.nn.Sequential(*blocks5)
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        if filter_idx < len(n_filters):
            blocks6 = []
            in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
            for n in range(n_blocks[block_idx]):
                if n == 0:
                    in_channels = 4 * in_channels if use_bottleneck else in_channels
                    block = resnet_block(in_channels, out_channels, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
                else:
                    in_channels = 4 * out_channels if use_bottleneck else out_channels
                    block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
                blocks6.append(block)
            self.blocks6 = torch.nn.Sequential(*blocks6)
        else:
            self.blocks6 = None
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        if filter_idx < len(n_filters):
            blocks7 = []
            in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
            for n in range(n_blocks[block_idx]):
                if n == 0:
                    in_channels = 4 * in_channels if use_bottleneck else in_channels
                    block = resnet_block(in_channels, out_channels, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
                else:
                    in_channels = 4 * out_channels if use_bottleneck else out_channels
                    block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
                blocks7.append(block)
            self.blocks7 = torch.nn.Sequential(*blocks7)
        else:
            self.blocks7 = None

    def forward(self, x):
        """
        Forward input x through a ResNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
            list[torch.Tensor[float32]] : list of skip connections
        """
        layers = [x]
        layers.append(self.conv1(layers[-1]))
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))
        layers.append(self.blocks3(layers[-1]))
        layers.append(self.blocks4(layers[-1]))
        layers.append(self.blocks5(layers[-1]))
        if self.blocks6 is not None:
            layers.append(self.blocks6(layers[-1]))
        if self.blocks7 is not None:
            layers.append(self.blocks7(layers[-1]))
        return layers[-1], layers[1:-1]


class AtrousResNetEncoder(torch.nn.Module):
    """
    ResNet encoder with skip connections

    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        atrous_spatial_pyramid_pool_dilations : list[int]
            list of dilation rates for atrous spatial pyramid pool (ASPP)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """

    def __init__(self, n_layer, input_channels=3, n_filters=[32, 64, 128, 256, 256], atrous_spatial_pyramid_pool_dilations=None, weight_initializer='kaiming_uniform', activation_func='leaky_relu', use_batch_norm=False, use_instance_norm=False):
        super(AtrousResNetEncoder, self).__init__()
        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
            atrous_resnet_block = net_utils.AtrousResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
            atrous_resnet_block = net_utils.AtrousResNetBlock
        else:
            raise ValueError('Only supports 18, 34 layer architecture')
        assert len(n_filters) == len(n_blocks) + 1
        activation_func = net_utils.activation_func(activation_func)
        dilation = 2
        in_channels, out_channels = [input_channels, n_filters[0]]
        self.conv1 = net_utils.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_channels, out_channels = [n_filters[0], n_filters[1]]
        blocks2 = []
        for n in range(n_blocks[0]):
            if n == 0:
                block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
                blocks2.append(block)
            else:
                in_channels = out_channels
                block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
                blocks2.append(block)
        self.blocks2 = torch.nn.Sequential(*blocks2)
        blocks3 = []
        in_channels, out_channels = [n_filters[1], n_filters[2]]
        for n in range(n_blocks[1]):
            if n == 0:
                block = resnet_block(in_channels, out_channels, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
                blocks3.append(block)
            else:
                in_channels = out_channels
                block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
                blocks3.append(block)
        self.blocks3 = torch.nn.Sequential(*blocks3)
        blocks4 = []
        in_channels, out_channels = [n_filters[2], n_filters[3]]
        for n in range(n_blocks[2]):
            if n == 0:
                block = atrous_resnet_block(in_channels, out_channels, dilation=dilation, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
                dilation = dilation * 2
                blocks4.append(block)
            else:
                in_channels = out_channels
                block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
                blocks4.append(block)
        self.blocks4 = torch.nn.Sequential(*blocks4)
        blocks5 = []
        in_channels, out_channels = [n_filters[3], n_filters[4]]
        for n in range(n_blocks[3]):
            if n == 0:
                block = atrous_resnet_block(in_channels, out_channels, dilation=dilation, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
                dilation = dilation * 2
                blocks5.append(block)
            else:
                in_channels = out_channels
                block = resnet_block(in_channels, out_channels, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
                blocks5.append(block)
        self.blocks5 = torch.nn.Sequential(*blocks5)
        if atrous_spatial_pyramid_pool_dilations is not None:
            self.atrous_spatial_pyramid_pool = net_utils.AtrousSpatialPyramidPooling(in_channels, out_channels, dilations=atrous_spatial_pyramid_pool_dilations, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        else:
            self.atrous_spatial_pyramid_pool = torch.nn.Identity()

    def forward(self, x):
        """
        Forward input x through an atrous ResNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
            list[torch.Tensor[float32]] : list of skip connections
        """
        layers = [x]
        layers.append(self.conv1(layers[-1]))
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))
        layers.append(self.blocks3(layers[-1]))
        layers.append(self.blocks4(layers[-1]))
        block5 = self.blocks5(layers[-1])
        layers.append(self.atrous_spatial_pyramid_pool(block5))
        return layers[-1], layers[1:-1]


class VGGNetEncoder(torch.nn.Module):
    """
    VGGNet encoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input data
        n_layer : int
            architecture type based on layers: 8, 11, 13
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    """

    def __init__(self, n_layer, input_channels=3, n_filters=[32, 64, 128, 256, 256], weight_initializer='kaiming_uniform', activation_func='leaky_relu', use_batch_norm=False, use_instance_norm=False, use_depthwise_separable=False):
        super(VGGNetEncoder, self).__init__()
        if n_layer == 8:
            n_convolutions = [1, 1, 1, 1, 1]
        elif n_layer == 11:
            n_convolutions = [1, 1, 2, 2, 2]
        elif n_layer == 13:
            n_convolutions = [2, 2, 2, 2, 2]
        else:
            raise ValueError('Only supports 8, 11, 13 layer architecture')
        for n in range(len(n_filters) - len(n_convolutions) - 1):
            n_convolutions = n_convolutions + [n_convolutions[-1]]
        block_idx = 0
        filter_idx = 0
        assert len(n_filters) == len(n_convolutions)
        activation_func = net_utils.activation_func(activation_func)
        stride = 1 if n_convolutions[block_idx] - 1 > 0 else 2
        in_channels, out_channels = [input_channels, n_filters[filter_idx]]
        conv1 = net_utils.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        if n_convolutions[block_idx] - 1 > 0:
            self.conv1 = torch.nn.Sequential(conv1, net_utils.VGGNetBlock(out_channels, out_channels, n_convolution=n_convolutions[filter_idx] - 1, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=False))
        else:
            self.conv1 = conv1
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
        self.conv2 = net_utils.VGGNetBlock(in_channels, out_channels, n_convolution=n_convolutions[block_idx], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=False)
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
        self.conv3 = net_utils.VGGNetBlock(in_channels, out_channels, n_convolution=n_convolutions[block_idx], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=False)
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
        self.conv4 = net_utils.VGGNetBlock(in_channels, out_channels, n_convolution=n_convolutions[block_idx], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
        self.conv5 = net_utils.VGGNetBlock(in_channels, out_channels, n_convolution=n_convolutions[block_idx], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        if filter_idx < len(n_filters):
            in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
            self.conv6 = net_utils.VGGNetBlock(in_channels, out_channels, n_convolution=n_convolutions[block_idx], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
        else:
            self.conv6 = None
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1
        if filter_idx < len(n_filters):
            in_channels, out_channels = [n_filters[filter_idx - 1], n_filters[filter_idx]]
            self.conv7 = net_utils.VGGNetBlock(in_channels, out_channels, n_convolution=n_convolutions[block_idx], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_depthwise_separable=use_depthwise_separable)
        else:
            self.conv7 = None

    def forward(self, x):
        """
        Forward input x through a VGGNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        layers = [x]
        layers.append(self.conv1(layers[-1]))
        layers.append(self.conv2(layers[-1]))
        layers.append(self.conv3(layers[-1]))
        layers.append(self.conv4(layers[-1]))
        layers.append(self.conv5(layers[-1]))
        if self.conv6 is not None:
            layers.append(self.conv6(layers[-1]))
        if self.conv7 is not None:
            layers.append(self.conv7(layers[-1]))
        return layers[-1], layers[1:-1]


class AtrousVGGNetEncoder(torch.nn.Module):
    """
    Atrous VGGNet encoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input data
        n_layer : int
            architecture type based on layers: 8, 11, 13
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """

    def __init__(self, n_layer, input_channels=3, n_filters=[32, 64, 128, 256, 256], weight_initializer='kaiming_uniform', activation_func='leaky_relu', use_batch_norm=False, use_instance_norm=False):
        super(AtrousVGGNetEncoder, self).__init__()
        if n_layer == 8:
            n_convolutions = [1, 1, 1, 1, 1]
        elif n_layer == 11:
            n_convolutions = [1, 1, 2, 2, 2]
        elif n_layer == 13:
            n_convolutions = [2, 2, 2, 2, 2]
        else:
            raise ValueError('Only supports 8, 11, 13 layer architecture')
        assert len(n_filters) == len(n_convolutions)
        activation_func = net_utils.activation_func(activation_func)
        dilation = 2
        stride = 1 if n_convolutions[0] - 1 > 0 else 2
        in_channels, out_channels = [input_channels, n_filters[0]]
        conv1 = net_utils.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        if n_convolutions[0] - 1 > 0:
            self.conv1 = torch.nn.Sequential(conv1, net_utils.VGGNetBlock(out_channels, out_channels, n_convolution=n_convolutions[0] - 1, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))
        else:
            self.conv1 = conv1
        in_channels, out_channels = [n_filters[0], n_filters[1]]
        self.conv2 = net_utils.VGGNetBlock(in_channels, out_channels, n_convolution=n_convolutions[1], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        in_channels, out_channels = [n_filters[1], n_filters[2]]
        self.conv3 = net_utils.VGGNetBlock(in_channels, out_channels, n_convolution=n_convolutions[2], stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        in_channels, out_channels = [n_filters[2], n_filters[3]]
        self.conv4 = net_utils.AtrousVGGNetBlock(in_channels, out_channels, n_convolution=n_convolutions[3], dilation=dilation, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
        in_channels, out_channels = [n_filters[3], n_filters[4]]
        self.conv5 = net_utils.AtrousVGGNetBlock(in_channels, out_channels, n_convolution=n_convolutions[4], dilation=dilation, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)

    def forward(self, x):
        """
        Forward input x through an atrous VGGNet encoder

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        """
        layers = [x]
        layers.append(self.conv1(layers[-1]))
        layers.append(self.conv2(layers[-1]))
        layers.append(self.conv3(layers[-1]))
        layers.append(self.conv4(layers[-1]))
        layers.append(self.conv5(layers[-1]))
        return layers[-1], layers[1:-1]


class MultiScaleDecoder(torch.nn.Module):
    """
    Multi-scale decoder with skip connections

    Arg(s):
        input_channels : int
            number of channels in input latent vector
        output_channels : int
            number of channels or classes in output
        n_resolution : int
            number of output resolutions (scales) for multi-scale prediction
        n_filters : list[int]
            number of filters to use at each decoder block
        n_skips : list[int]
            number of filters from skip connections
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        output_func : func
            activation function for output
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        deconv_type : str
            deconvolution types available: transpose, up
    """

    def __init__(self, input_channels=256, output_channels=1, n_resolution=4, n_filters=[256, 128, 64, 32, 16], n_skips=[256, 128, 64, 32, 0], weight_initializer='kaiming_uniform', activation_func='leaky_relu', output_func='linear', use_batch_norm=False, use_instance_norm=False, deconv_type='transpose'):
        super(MultiScaleDecoder, self).__init__()
        network_depth = len(n_filters)
        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert n_resolution > 0 and n_resolution < network_depth
        self.n_resolution = n_resolution
        self.output_func = output_func
        activation_func = net_utils.activation_func(activation_func)
        output_func = net_utils.activation_func(output_func)
        if 'upsample' in self.output_func and self.n_resolution < 2:
            self.n_resolution = 2
        filter_idx = 0
        in_channels, skip_channels, out_channels = [input_channels, n_skips[filter_idx], n_filters[filter_idx]]
        if network_depth > 6:
            self.deconv6 = net_utils.DecoderBlock(in_channels, skip_channels, out_channels, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, deconv_type=deconv_type)
            filter_idx = filter_idx + 1
            in_channels, skip_channels, out_channels = [n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]]
        else:
            self.deconv6 = None
        if network_depth > 5:
            self.deconv5 = net_utils.DecoderBlock(in_channels, skip_channels, out_channels, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, deconv_type=deconv_type)
            filter_idx = filter_idx + 1
            in_channels, skip_channels, out_channels = [n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]]
        else:
            self.deconv5 = None
        if network_depth > 4:
            self.deconv4 = net_utils.DecoderBlock(in_channels, skip_channels, out_channels, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, deconv_type=deconv_type)
            filter_idx = filter_idx + 1
            in_channels, skip_channels, out_channels = [n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]]
        else:
            self.deconv4 = None
        if network_depth > 3:
            self.deconv3 = net_utils.DecoderBlock(in_channels, skip_channels, out_channels, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, deconv_type=deconv_type)
            if self.n_resolution > 3:
                self.output3 = net_utils.Conv2d(out_channels, output_channels, kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=None, use_batch_norm=False, use_instance_norm=False)
            else:
                self.output3 = None
            filter_idx = filter_idx + 1
            in_channels, skip_channels, out_channels = [n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]]
            if self.n_resolution > 3:
                skip_channels = skip_channels + output_channels
        else:
            self.deconv3 = None
        if network_depth > 2:
            self.deconv2 = net_utils.DecoderBlock(in_channels, skip_channels, out_channels, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, deconv_type=deconv_type)
            if self.n_resolution > 2:
                self.output2 = net_utils.Conv2d(out_channels, output_channels, kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=output_func, use_batch_norm=False, use_instance_norm=False)
            else:
                self.output2 = None
            filter_idx = filter_idx + 1
            in_channels, skip_channels, out_channels = [n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]]
            if self.n_resolution > 2:
                skip_channels = skip_channels + output_channels
        else:
            self.deconv2 = None
        self.deconv1 = net_utils.DecoderBlock(in_channels, skip_channels, out_channels, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, deconv_type=deconv_type)
        if self.n_resolution > 1:
            self.output1 = net_utils.Conv2d(out_channels, output_channels, kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=output_func, use_batch_norm=False, use_instance_norm=False)
        else:
            self.output1 = None
        filter_idx = filter_idx + 1
        in_channels, skip_channels, out_channels = [n_filters[filter_idx - 1], n_skips[filter_idx], n_filters[filter_idx]]
        if self.n_resolution > 1:
            skip_channels = skip_channels + output_channels
        self.deconv0 = net_utils.DecoderBlock(in_channels, skip_channels, out_channels, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, deconv_type=deconv_type)
        self.output0 = net_utils.Conv2d(out_channels, output_channels, kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=output_func, use_batch_norm=False, use_instance_norm=False)

    def forward(self, x, skips, shape=None):
        """
        Forward latent vector x through decoder network

        Arg(s):
            x : torch.Tensor[float32]
                latent vector
            skips : list[torch.Tensor[float32]]
                list of skip connection tensors (earlier are larger resolution)
            shape : tuple[int]
                (height, width) tuple denoting output size
        Returns:
            list[torch.Tensor[float32]] : list of outputs at multiple scales
        """
        layers = [x]
        outputs = []
        n = len(skips) - 1
        if self.deconv6 is not None:
            layers.append(self.deconv6(layers[-1], skips[n]))
            n = n - 1
        if self.deconv5 is not None:
            layers.append(self.deconv5(layers[-1], skips[n]))
            n = n - 1
        if self.deconv4 is not None:
            layers.append(self.deconv4(layers[-1], skips[n]))
            n = n - 1
        if self.deconv3 is not None:
            layers.append(self.deconv3(layers[-1], skips[n]))
            if self.n_resolution > 3:
                output3 = self.output3(layers[-1])
                outputs.append(output3)
                if n > 0:
                    upsample_output3 = torch.nn.functional.interpolate(input=outputs[-1], size=skips[n - 1].shape[-2:], mode='bilinear', align_corners=True)
                else:
                    upsample_output3 = torch.nn.functional.interpolate(input=outputs[-1], scale_factor=2, mode='bilinear', align_corners=True)
            n = n - 1
        if self.deconv2 is not None:
            if skips[n] is not None:
                skip = torch.cat([skips[n], upsample_output3], dim=1) if self.n_resolution > 3 else skips[n]
            else:
                skip = skips[n]
            layers.append(self.deconv2(layers[-1], skip))
            if self.n_resolution > 2:
                output2 = self.output2(layers[-1])
                outputs.append(output2)
                if n > 0:
                    upsample_output2 = torch.nn.functional.interpolate(input=outputs[-1], size=skips[n - 1].shape[-2:], mode='bilinear', align_corners=True)
                else:
                    upsample_output2 = torch.nn.functional.interpolate(input=outputs[-1], scale_factor=2, mode='bilinear', align_corners=True)
            n = n - 1
        if skips[n] is not None:
            skip = torch.cat([skips[n], upsample_output2], dim=1) if self.n_resolution > 2 else skips[n]
        else:
            skip = skips[n]
        layers.append(self.deconv1(layers[-1], skip))
        if self.n_resolution > 1:
            output1 = self.output1(layers[-1])
            outputs.append(output1)
            if n > 0:
                upsample_output1 = torch.nn.functional.interpolate(input=outputs[-1], size=skips[n - 1].shape[-2:], mode='bilinear', align_corners=True)
            else:
                upsample_output1 = torch.nn.functional.interpolate(input=outputs[-1], scale_factor=2, mode='bilinear', align_corners=True)
        n = n - 1
        if 'upsample' in self.output_func:
            output0 = upsample_output1
        else:
            if self.n_resolution > 1:
                if skips[n] is not None and n == 0:
                    skip = torch.cat([skips[n], upsample_output1], dim=1) if n == 0 else upsample_output1
                else:
                    skip = upsample_output1
                layers.append(self.deconv0(layers[-1], skip))
            elif skips[n] is not None and n == 0:
                layers.append(self.deconv0(layers[-1], skips[n]))
            else:
                layers.append(self.deconv0(layers[-1], shape=shape[-2:]))
            output0 = self.output0(layers[-1])
        outputs.append(output0)
        return outputs


class PoseDecoder(torch.nn.Module):
    """
    Pose Decoder 6 DOF

    Arg(s):
        rotation_parameterization : str
            axis
        input_channels : int
            number of channels in input latent vector
        n_filters : int list
            number of filters to use at each decoder block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    """

    def __init__(self, rotation_parameterization, input_channels=256, n_filters=[], weight_initializer='kaiming_uniform', activation_func='leaky_relu', use_batch_norm=False, use_instance_norm=False):
        super(PoseDecoder, self).__init__()
        self.rotation_parameterization = rotation_parameterization
        activation_func = net_utils.activation_func(activation_func)
        if len(n_filters) > 0:
            layers = []
            in_channels = input_channels
            for out_channels in n_filters:
                conv = net_utils.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm)
                layers.append(conv)
                in_channels = out_channels
            conv = net_utils.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=1, stride=1, weight_initializer=weight_initializer, activation_func=None, use_batch_norm=False, use_instance_norm=False)
            layers.append(conv)
            self.conv = torch.nn.Sequential(*layers)
        else:
            self.conv = net_utils.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=1, stride=1, weight_initializer=weight_initializer, activation_func=None, use_batch_norm=False, use_instance_norm=False)

    def forward(self, x):
        conv_output = self.conv(x)
        pose_mean = torch.mean(conv_output, [2, 3])
        dof = 0.01 * pose_mean
        posemat = net_utils.pose_matrix(dof, rotation_parameterization=self.rotation_parameterization)
        return posemat


class SparseToDensePool(torch.nn.Module):
    """
    Converts sparse inputs to dense outputs using max and min pooling
    with different kernel sizes and combines them with 1 x 1 convolutions

    Arg(s):
        input_channels : int
            number of channels to be fed to max and/or average pool(s)
        min_pool_sizes : list[int]
            list of min pool sizes s (kernel size is s x s)
        max_pool_sizes : list[int]
            list of max pool sizes s (kernel size is s x s)
        n_filter : int
            number of filters for 1 x 1 convolutions
        n_convolution : int
            number of 1 x 1 convolutions to use for balancing detail and density
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    """

    def __init__(self, input_channels, min_pool_sizes=[3, 5, 7, 9], max_pool_sizes=[3, 5, 7, 9], n_filter=8, n_convolution=3, weight_initializer='kaiming_uniform', activation_func='leaky_relu'):
        super(SparseToDensePool, self).__init__()
        activation_func = net_utils.activation_func(activation_func)
        self.min_pool_sizes = [s for s in min_pool_sizes if s > 1]
        self.max_pool_sizes = [s for s in max_pool_sizes if s > 1]
        self.min_pools = []
        for s in self.min_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.min_pools.append(pool)
        self.max_pools = []
        for s in self.max_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.max_pools.append(pool)
        self.len_pool_sizes = len(self.min_pool_sizes) + len(self.max_pool_sizes)
        in_channels = len(self.min_pool_sizes) + len(self.max_pool_sizes)
        pool_convs = []
        for n in range(n_convolution):
            conv = net_utils.Conv2d(in_channels, n_filter, kernel_size=1, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=False, use_instance_norm=False)
            pool_convs.append(conv)
            in_channels = n_filter
        self.pool_convs = torch.nn.Sequential(*pool_convs)
        in_channels = n_filter + input_channels
        self.conv = net_utils.Conv2d(in_channels, n_filter, kernel_size=3, stride=1, weight_initializer=weight_initializer, activation_func=activation_func, use_batch_norm=False, use_instance_norm=False)

    def forward(self, x):
        z = torch.unsqueeze(x[:, 0, ...], dim=1)
        pool_pyramid = []
        for pool, s in zip(self.min_pools, self.min_pool_sizes):
            z_pool = -pool(torch.where(z == 0, -999 * torch.ones_like(z), -z))
            z_pool = torch.where(z_pool == 999, torch.zeros_like(z), z_pool)
            pool_pyramid.append(z_pool)
        for pool, s in zip(self.max_pools, self.max_pool_sizes):
            z_pool = pool(z)
            pool_pyramid.append(z_pool)
        pool_pyramid = torch.cat(pool_pyramid, dim=1)
        pool_convs = self.pool_convs(pool_pyramid)
        pool_convs = torch.cat([pool_convs, x], dim=1)
        return self.conv(pool_convs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AtrousConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AtrousResNetBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AtrousSpatialPyramidPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (AtrousVGGNetBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DepthwiseSeparableConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FullyConnected,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNetBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SpatialPyramidPooling,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TransposeConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VGGNetBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_alexklwong_calibrated_backprojection_network(_paritybench_base):
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

