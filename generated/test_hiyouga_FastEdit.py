import sys
_module = sys.modules[__name__]
del sys
fastedit = _module
editor = _module
rome = _module
compute_u = _module
compute_v = _module
repr_tools = _module
rome_hparams = _module
rome_main = _module
utils = _module
context = _module
generate = _module
hparams = _module
mtloader = _module
nethook = _module
prints = _module
template = _module
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


from typing import Dict


from typing import List


from typing import Optional


import numpy as np


from typing import Tuple


from typing import Literal


import time


from copy import deepcopy


from typing import Union


import copy


import inspect


from collections import OrderedDict


import re

