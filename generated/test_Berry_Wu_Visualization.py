import sys
_module = sys.modules[__name__]
del sys
main = _module
data = _module
gen_heatmap = _module
get_parameter = _module
Grad_CAM = _module
guided_backpro = _module
model_test = _module
img_patch = _module
separate_num = _module
visual = _module

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


import torch


import numpy as np


from matplotlib import pyplot as plt


from torch import nn


from torch.autograd import Function


import torchvision


from torchvision import transforms


import math


import matplotlib


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input_img):
        output = torch.clamp(input_img, min=0.0)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = grad_output * positive_mask_1 * positive_mask_2
        return grad_input


class GuidedBackpropReLUasModule(torch.nn.Module):

    def __init__(self):
        super(GuidedBackpropReLUasModule, self).__init__()

    def forward(self, input_img):
        return GuidedBackpropReLU.apply(input_img)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GuidedBackpropReLUasModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Berry_Wu_Visualization(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

