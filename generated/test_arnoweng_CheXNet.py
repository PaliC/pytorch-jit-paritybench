import sys
_module = sys.modules[__name__]
del sys
model = _module
read_data = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.backends.cudnn as cudnn


import torchvision


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(num_ftrs,
            out_size), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_arnoweng_CheXNet(_paritybench_base):
    pass
    def test_000(self):
        self._check(DenseNet121(*[], **{'out_size': 4}), [torch.rand([4, 3, 64, 64])], {})

