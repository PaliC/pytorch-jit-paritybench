import sys
_module = sys.modules[__name__]
del sys
MutualInformation = _module
histogram = _module

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


import torch.nn as nn


import matplotlib.pyplot as plt


from torchvision import transforms


from sklearn.metrics import normalized_mutual_info_score


class MutualInformation(nn.Module):

    def __init__(self, sigma=0.1, num_bins=256, normalize=True):
        super(MutualInformation, self).__init__()
        self.sigma = sigma
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10
        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins).float(), requires_grad=False)

    def marginalPdf(self, values):
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        return pdf, kernel_values

    def jointPdf(self, kernel_values1, kernel_values2):
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
        normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization
        return pdf

    def getMutualInformation(self, input1, input2):
        """
			input1: B, C, H, W
			input2: B, C, H, W

			return: scalar
		"""
        input1 = input1 * 255
        input2 = input2 * 255
        B, C, H, W = input1.shape
        assert input1.shape == input2.shape
        x1 = input1.view(B, H * W, C)
        x2 = input2.view(B, H * W, C)
        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)
        H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.epsilon), dim=(1, 2))
        mutual_information = H_x1 + H_x2 - H_x1x2
        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x1 + H_x2)
        return mutual_information

    def forward(self, input1, input2):
        """
			input1: B, C, H, W
			input2: B, C, H, W

			return: scalar
		"""
        return self.getMutualInformation(input1, input2)

