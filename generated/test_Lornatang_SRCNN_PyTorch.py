import sys
_module = sys.modules[__name__]
del sys
config = _module
dataset = _module
image_quality_assessment = _module
imgproc = _module
inference = _module
model = _module
prepare_dataset = _module
run = _module
split_train_valid_dataset = _module
test = _module
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


import random


import numpy as np


import torch


from torch.backends import cudnn


import queue


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import warnings


from torch import nn


from torch.nn import functional as F


import math


from typing import Any


from torchvision.transforms import functional as F


import time


from enum import Enum


from torch import optim


from torch.cuda import amp


from torch.utils.tensorboard import SummaryWriter


def _check_tensor_shape(raw_tensor: 'torch.Tensor', dst_tensor: 'torch.Tensor'):
    """Check if the dimensions of the two tensors are the same

    Args:
        raw_tensor (np.ndarray or torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (np.ndarray or torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]

    """
    assert raw_tensor.shape == dst_tensor.shape, f'Supplied images have different sizes {str(raw_tensor.shape)} and {str(dst_tensor.shape)}'


def _psnr_torch(raw_tensor: 'torch.Tensor', dst_tensor: 'torch.Tensor', crop_border: 'int', only_test_y_channel: 'bool') ->float:
    """PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function

    Args:
        raw_tensor (torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image

    Returns:
        psnr_metrics (torch.Tensor): PSNR metrics

    """
    _check_tensor_shape(raw_tensor, dst_tensor)
    if crop_border > 0:
        raw_tensor = raw_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
        dst_tensor = dst_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
    if only_test_y_channel:
        raw_tensor = imgproc.rgb2ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = imgproc.rgb2ycbcr_torch(dst_tensor, only_use_y_channel=True)
    raw_tensor = raw_tensor
    dst_tensor = dst_tensor
    mse_value = torch.mean((raw_tensor * 255.0 - dst_tensor * 255.0) ** 2 + 1e-08, dim=[1, 2, 3])
    psnr_metrics = 10 * torch.log10_(255.0 ** 2 / mse_value)
    return psnr_metrics


class PSNR(nn.Module):
    """PyTorch implements PSNR (Peak Signal-to-Noise Ratio, peak signal-to-noise ratio) function

    Attributes:
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image

    Returns:
        psnr_metrics (torch.Tensor): PSNR metrics

    """

    def __init__(self, crop_border: 'int', only_test_y_channel: 'bool') ->None:
        super().__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel

    def forward(self, raw_tensor: 'torch.Tensor', dst_tensor: 'torch.Tensor') ->torch.Tensor:
        psnr_metrics = _psnr_torch(raw_tensor, dst_tensor, self.crop_border, self.only_test_y_channel)
        return psnr_metrics


def _ssim_torch(raw_tensor: 'torch.Tensor', dst_tensor: 'torch.Tensor', window_size: 'int', gaussian_kernel_window: 'np.ndarray') ->float:
    """PyTorch implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        raw_tensor (torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 255]
        dst_tensor (torch.Tensor): reference image tensorflow, RGB format, data range [0, 255]
        window_size (int): Gaussian filter size
        gaussian_kernel_window (np.ndarray): Gaussian filter

    Returns:
        ssim_metrics (torch.Tensor): SSIM metrics

    """
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    gaussian_kernel_window = torch.from_numpy(gaussian_kernel_window).view(1, 1, window_size, window_size)
    gaussian_kernel_window = gaussian_kernel_window.expand(raw_tensor.size(1), 1, window_size, window_size)
    gaussian_kernel_window = gaussian_kernel_window
    raw_mean = F.conv2d(raw_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0), groups=raw_tensor.shape[1])
    dst_mean = F.conv2d(dst_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0), groups=dst_tensor.shape[1])
    raw_mean_square = raw_mean ** 2
    dst_mean_square = dst_mean ** 2
    raw_dst_mean = raw_mean * dst_mean
    raw_variance = F.conv2d(raw_tensor * raw_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0), groups=raw_tensor.shape[1]) - raw_mean_square
    dst_variance = F.conv2d(dst_tensor * dst_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0), groups=raw_tensor.shape[1]) - dst_mean_square
    raw_dst_covariance = F.conv2d(raw_tensor * dst_tensor, gaussian_kernel_window, stride=1, padding=(0, 0), groups=raw_tensor.shape[1]) - raw_dst_mean
    ssim_molecular = (2 * raw_dst_mean + c1) * (2 * raw_dst_covariance + c2)
    ssim_denominator = (raw_mean_square + dst_mean_square + c1) * (raw_variance + dst_variance + c2)
    ssim_metrics = ssim_molecular / ssim_denominator
    ssim_metrics = torch.mean(ssim_metrics, [1, 2, 3])
    return ssim_metrics


def _ssim_single_torch(raw_tensor: 'torch.Tensor', dst_tensor: 'torch.Tensor', crop_border: 'int', only_test_y_channel: 'bool', window_size: 'int', gaussian_kernel_window: 'torch.Tensor') ->torch.Tensor:
    """PyTorch implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        raw_tensor (torch.Tensor): image tensor flow to be compared, RGB format, data range [0, 1]
        dst_tensor (torch.Tensor): reference image tensorflow, RGB format, data range [0, 1]
        crop_border (int): crop border a few pixels
        only_test_y_channel (bool): Whether to test only the Y channel of the image
        window_size (int): Gaussian filter size
        gaussian_kernel_window (torch.Tensor): Gaussian filter

    Returns:
        ssim_metrics (torch.Tensor): SSIM metrics

    """
    _check_tensor_shape(raw_tensor, dst_tensor)
    if crop_border > 0:
        raw_tensor = raw_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
        dst_tensor = dst_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
    if only_test_y_channel:
        raw_tensor = imgproc.rgb2ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = imgproc.rgb2ycbcr_torch(dst_tensor, only_use_y_channel=True)
    raw_tensor = raw_tensor
    dst_tensor = dst_tensor
    ssim_metrics = _ssim_torch(raw_tensor * 255.0, dst_tensor * 255.0, window_size, gaussian_kernel_window)
    return ssim_metrics


class SSIM(nn.Module):
    """PyTorch implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        crop_border (int): crop border a few pixels
        only_only_test_y_channel (bool): Whether to test only the Y channel of the image
        window_size (int): Gaussian filter size
        gaussian_sigma (float): sigma parameter in Gaussian filter

    Returns:
        ssim_metrics (torch.Tensor): SSIM metrics

    """

    def __init__(self, crop_border: 'int', only_only_test_y_channel: 'bool', window_size: 'int'=11, gaussian_sigma: 'float'=1.5) ->None:
        super().__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_only_test_y_channel
        self.window_size = window_size
        gaussian_kernel = cv2.getGaussianKernel(window_size, gaussian_sigma)
        self.gaussian_kernel_window = np.outer(gaussian_kernel, gaussian_kernel.transpose())

    def forward(self, raw_tensor: 'torch.Tensor', dst_tensor: 'torch.Tensor') ->torch.Tensor:
        ssim_metrics = _ssim_single_torch(raw_tensor, dst_tensor, self.crop_border, self.only_test_y_channel, self.window_size, self.gaussian_kernel_window)
        return ssim_metrics


class SRCNN(nn.Module):

    def __init__(self) ->None:
        super(SRCNN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)), nn.ReLU(True))
        self.map = nn.Sequential(nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)), nn.ReLU(True))
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))
        self._initialize_weights()

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: 'torch.Tensor') ->torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        return out

    def _initialize_weights(self) ->None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)
        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SRCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
]

class Test_Lornatang_SRCNN_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

