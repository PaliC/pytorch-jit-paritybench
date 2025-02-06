import sys
_module = sys.modules[__name__]
del sys
clean = _module
setup = _module
liberate = _module
csprng = _module
chacha20_naive = _module
csprng = _module
discrete_gaussian_sampler = _module
setup = _module
fhe = _module
cache = _module
ckks_engine = _module
context = _module
ckks_context = _module
generate_primes = _module
prim_test = _module
security_parameters = _module
data_struct = _module
encdec = _module
encdec = _module
presets = _module
errors = _module
params = _module
types = _module
test_generate_engine = _module
version = _module
ntt = _module
ntt_context = _module
rns_partition = _module
setup = _module
utils = _module
helpers = _module

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


from torch.utils.cpp_extension import BuildExtension


from torch.utils.cpp_extension import CUDAExtension


import math


import torch


import numpy as np


import warnings


import time

