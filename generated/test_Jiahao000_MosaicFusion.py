import sys
_module = sys.modules[__name__]
del sys
merge_ann = _module
mosaicfusion = _module
bilateral_solver = _module
canvas = _module
utils = _module
vis = _module
sam_iou_metric = _module
sam_mask_refiner = _module
seg2ann = _module
text2seg = _module
vis_gen = _module

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


import re


from copy import deepcopy


from enum import Enum


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import numpy as np


import torch


from numpy import pi


from numpy import exp


from numpy import sqrt


from torchvision.transforms.functional import resize


import abc


import random


from typing import Callable


from typing import Dict


import torch.nn.functional as F


import math


from torchvision.transforms import ToTensor


from torchvision.utils import make_grid

