import sys
_module = sys.modules[__name__]
del sys
main = _module
schedulefree = _module
adamw_schedulefree = _module
adamw_schedulefree_closure = _module
adamw_schedulefree_paper = _module
adamw_schedulefree_reference = _module
submission = _module
submission = _module
submission = _module
radam_schedulefree = _module
radam_schedulefree_closure = _module
sgd_schedulefree = _module
sgd_schedulefree_closure = _module
sgd_schedulefree_reference = _module
test_schedulefree = _module
wrap_schedulefree = _module
wrap_schedulefree_reference = _module
setup = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torchvision import datasets


from torchvision import transforms


from typing import Tuple


from typing import Union


from typing import Optional


from typing import Iterable


from typing import Dict


from typing import Callable


from typing import Any


import torch.optim


import math


from typing import Iterator


from typing import List


import torch.distributed.nn as dist_nn


from math import sqrt


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

