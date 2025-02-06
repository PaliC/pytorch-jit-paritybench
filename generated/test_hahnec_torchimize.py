import sys
_module = sys.modules[__name__]
del sys
conf = _module
draw_svg_logo = _module
embedded_svg_base64_font = _module
setup = _module
tests = _module
emg = _module
unit_test_all = _module
unit_test_analytical_jacobian = _module
unit_test_parallel = _module
unit_test_raw_fit = _module
unit_test_skewed_gaussian = _module
torchimize = _module
functions = _module
fun_dims = _module
jacobian = _module
parallel = _module
gda_fun_parallel = _module
gna_fun_parallel = _module
lma_fun_parallel = _module
newton_parallel = _module
single = _module
gda_fun_single = _module
gna_fun_single = _module
lma_fun_single = _module
optimizer = _module
gna_opt = _module

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


from torch import nn


import torch.nn.functional as F


import torch.utils.data as Data


from sklearn.model_selection import train_test_split


from typing import Callable


from typing import Union


from typing import Tuple


from typing import List


import warnings


from torch import Tensor


import torch.nn as nn


from torch.optim.optimizer import Optimizer


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(1, 20)
        self.predict = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

