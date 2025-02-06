import sys
_module = sys.modules[__name__]
del sys
arguments = _module
base_dataset = _module
coreference_metrics = _module
datasets = _module
evaluate = _module
input_example = _module
input_formats = _module
output_formats = _module
extract_examples = _module
prepare_multi_woz = _module
run = _module
utils = _module

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


import random


from typing import Dict


from typing import Generator


from typing import Tuple


from typing import List


from abc import ABC


from abc import abstractmethod


import torch


from torch.utils.data import DataLoader


from torch.utils.data.dataset import Dataset


import copy


from itertools import islice


from collections import Counter


from collections import defaultdict


import numpy as np


from typing import Set


from typing import Optional


from typing import Any


from typing import Union


import itertools

