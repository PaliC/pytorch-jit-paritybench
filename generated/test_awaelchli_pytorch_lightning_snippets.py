import sys
_module = sys.modules[__name__]
del sys
checkpoint = _module
code_snapshot = _module
peek = _module
monitor = _module
data_monitor_base = _module
module_data_monitor = _module
training_data_monitor = _module
setup = _module
tests = _module
templates = _module
base = _module
test_batch_gradient = _module
test_batch_norm = _module
test_data_monitor = _module
verification = _module
base = _module
batch_gradient = _module
batch_norm = _module

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


from collections.abc import Mapping


from collections.abc import Sequence


import torch


from typing import Any


from typing import Sequence


from typing import Dict


import numpy as np


from torch import Tensor


from typing import List


from typing import Optional


from typing import Union


import torch.nn as nn


from torch.utils.hooks import RemovableHandle


import torch.nn.functional as F


import torchvision.transforms as transforms


from torch import optim


from torch.utils.data import DataLoader


from torchvision.datasets import MNIST


from torch import nn as nn


from torch.optim import Adam


from torch.utils.data import TensorDataset


from abc import abstractmethod


from copy import deepcopy


from typing import Callable


from typing import Tuple


class TemplateModel(nn.Module):

    def __init__(self, mix_data=False):
        super().__init__()
        self.mix_data = mix_data
        self.linear = nn.Linear(10, 5)
        self.input_array = torch.rand(10, 5, 2)

    def forward(self, *args, **kwargs):
        return self.forward__standard(*args, **kwargs)

    def forward__standard(self, x):
        if self.mix_data:
            x = x.view(10, -1).permute(1, 0).view(-1, 10)
        else:
            x = x.view(-1, 10)
        return self.linear(x)


class MultipleInputModel(TemplateModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_array = torch.rand(10, 5, 2), torch.rand(10, 5, 2)

    def forward(self, x, y, some_kwarg=True):
        out = super().forward(x) + super().forward(y)
        return out


class MultipleOutputModel(TemplateModel):

    def forward(self, x):
        out = super().forward(x)
        return None, out, out, False


class DictInputDictOutputModel(TemplateModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_array = {'w': 42, 'x': {'a': torch.rand(3, 5, 2)}, 'y': torch.rand(3, 1, 5, 2), 'z': torch.tensor(2)}

    def forward(self, y, x, z, w):
        out1 = super().forward(x['a'])
        out2 = super().forward(y)
        out3 = out1 + out2
        out = {(1): out1, (2): out2, (3): [out1, out3]}
        return out


class ConvBiasBatchNormModel(nn.Module):

    def __init__(self, use_bias=True):
        super().__init__()
        self.conv = nn.Conv2d(3, 5, kernel_size=1, bias=use_bias)
        self.bn = nn.BatchNorm2d(5)
        self.input_array = torch.rand(2, 3, 10, 10)

    def forward(self, x):
        return self.bn(self.conv(x))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvBiasBatchNormModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (MultipleInputModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 10]), torch.rand([4, 10])], {}),
     False),
    (MultipleOutputModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 10])], {}),
     False),
]

class Test_awaelchli_pytorch_lightning_snippets(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

