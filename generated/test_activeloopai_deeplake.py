import sys
_module = sys.modules[__name__]
del sys
setup_actions = _module
test_activeloop_code_analysis = _module
test_activeloop_deeplake = _module
test_activeloop_deeplake_selfquery = _module
test_activeloop_semanitic_search = _module
test_activeloop_twitter = _module
test_code_analysis_deeplake = _module
test_semanitic_search = _module
test_twitter = _module
deeplake = _module
_tensorflow = _module
_torch = _module
core = _module
formats = _module
ingestion = _module
coco = _module
exceptions = _module
from_coco = _module
ingest_coco = _module
integrations = _module
constants = _module
mm = _module
get_indexes = _module
ipc = _module
mm_common = _module
mm_runners = _module
upcast_array = _module
warnings = _module
worker_init_fn = _module
mmdet = _module
mmdet_ = _module
mmdet_dataset_ = _module
mmdet_utils_ = _module
test_ = _module
mmseg = _module
compose_transform_ = _module
mmseg_ = _module
mmseg_dataset_ = _module
test_ = _module
schemas = _module
storage = _module
tql = _module
types = _module

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


import math


from typing import Optional


import torch


import warnings


import logging


from torch.utils.data import DataLoader


import time


from typing import List


from typing import Tuple


import numpy as np


import random


from collections import OrderedDict


from typing import Callable


from typing import Dict


from typing import Sequence


from functools import partial


import types


from torch.utils.data import Dataset


import torch.distributed as dist


from typing import Union


from torch.utils.data import IterableDataset

