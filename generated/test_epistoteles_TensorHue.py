import sys
_module = sys.modules[__name__]
del sys
setup = _module
tensorhue = _module
_print_opts = _module
colors = _module
converters = _module
eastereggs = _module
viz = _module
test__print_opts = _module
test_colors = _module
test_converter_jax = _module
test_converter_pillow = _module
test_converter_tensorflow = _module
test_converter_torch = _module
test_eastereggs = _module
test_viz = _module

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


import inspect


import warnings


import numpy as np


import tensorflow as tf


import torch

