import sys
_module = sys.modules[__name__]
del sys
example_aug = _module
example_norm = _module
compare = _module
setup = _module
tests = _module
test_color_conv = _module
test_tf = _module
test_torch = _module
torchstain = _module
base = _module
augmentors = _module
he_augmentor = _module
macenko = _module
normalizers = _module
he_normalizer = _module
macenko = _module
multitarget = _module
reinhard = _module
numpy = _module
utils = _module
lab2rgb = _module
rgb2lab = _module
split = _module
stats = _module
tf = _module
cov = _module
percentile = _module
solveLS = _module
macenko = _module
normalizers = _module
macenko = _module
multitarget = _module
reinhard = _module
utils = _module
cov = _module
lab2rgb = _module
percentile = _module
rgb2lab = _module
split = _module
stats = _module

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


import matplotlib.pyplot as plt


import torch


from torchvision import transforms


import time


import numpy as np


import torchvision


from typing import Union

