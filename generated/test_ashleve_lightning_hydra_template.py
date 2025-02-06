import sys
_module = sys.modules[__name__]
del sys
configs = _module
setup = _module
src = _module
data = _module
components = _module
mnist_datamodule = _module
eval = _module
models = _module
simple_dense_net = _module
mnist_module = _module
train = _module
utils = _module
instantiators = _module
logging_utils = _module
pylogger = _module
rich_utils = _module
tests = _module
conftest = _module
helpers = _module
package_available = _module
run_if = _module
run_sh_command = _module
test_configs = _module
test_datamodules = _module
test_eval = _module
test_sweeps = _module
test_train = _module

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


from typing import Any


from typing import Dict


from typing import Optional


from typing import Tuple


import torch


from torch.utils.data import ConcatDataset


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import random_split


from torchvision.datasets import MNIST


from torchvision.transforms import transforms


from torch import nn


from typing import List


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(self, input_size: 'int'=784, lin1_size: 'int'=256, lin2_size: 'int'=256, lin3_size: 'int'=256, output_size: 'int'=10) ->None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_size, lin1_size), nn.BatchNorm1d(lin1_size), nn.ReLU(), nn.Linear(lin1_size, lin2_size), nn.BatchNorm1d(lin2_size), nn.ReLU(), nn.Linear(lin2_size, lin3_size), nn.BatchNorm1d(lin3_size), nn.ReLU(), nn.Linear(lin3_size, output_size))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        return self.model(x)

