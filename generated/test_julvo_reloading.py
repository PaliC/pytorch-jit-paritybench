import sys
_module = sys.modules[__name__]
del sys
demo = _module
train = _module
train = _module
reloading = _module
test_reloading = _module
setup = _module

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


from torch import nn


from torch.optim import Adam


import torch.nn.functional as F


from torchvision.models import resnet18


from torchvision.datasets import FashionMNIST


from torchvision.transforms import ToTensor


from torch.utils.data import DataLoader


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_julvo_reloading(_paritybench_base):
    pass
