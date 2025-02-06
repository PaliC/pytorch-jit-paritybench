import sys
_module = sys.modules[__name__]
del sys
conf = _module
setup = _module
aac_datasets = _module
check = _module
datasets = _module
audiocaps = _module
base = _module
clotho = _module
functional = _module
common = _module
macs = _module
wavcaps = _module
macs = _module
wavcaps = _module
download = _module
info = _module
utils = _module
audioset_mapping = _module
cmdline = _module
collate = _module
collections = _module
download = _module
globals = _module
type_checks = _module
test_datasets_base = _module
test_utils_collections = _module

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


import logging


from typing import Any


from typing import Callable


from typing import ClassVar


from typing import Dict


from typing import List


from typing import Optional


from typing import Union


import torch


import torchaudio


from torch import Tensor


from typing import Generic


from typing import Iterable


from typing import Tuple


from typing import TypeVar


from typing import overload


from torch.utils.data.dataset import Dataset


import copy


from torch.hub import download_url_to_file


from torch.nn import functional as F

