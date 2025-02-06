import sys
_module = sys.modules[__name__]
del sys
signerf = _module
camera_arc_dataset = _module
signerf_dataloader = _module
signerf_datamanager = _module
signerf_dataparser = _module
signerf_patch_pixel_sampler = _module
datasetgenerator = _module
datasetgenerator = _module
diffuser = _module
interface = _module
interface = _module
viewer = _module
viewer_elements_extended = _module
renderer = _module
renderer = _module
signerf = _module
signerf_config = _module
signerf_nerfacto_config = _module
signerf_pipeline = _module
signerf_trainer = _module
utils = _module
image_base64_converter = _module
image_tensor_converter = _module
intersection = _module
load_previous_experiment_cameras = _module
poses_generation = _module

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


from typing import Type


from typing import List


from typing import Dict


from torch.utils.data import Dataset


import random


from abc import abstractmethod


from typing import Optional


from typing import Tuple


from typing import Union


from typing import Sequence


import torch


from torch.utils.data.dataloader import DataLoader


from typing import Literal


from torch.nn import Parameter


import numpy as np


from torch import Tensor


from typing import Any


from typing import Callable


import time


import math


import torch.nn.functional as F


from collections import defaultdict


from typing import DefaultDict


from typing import TYPE_CHECKING


from typing import Mapping


from torch.cuda.amp.grad_scaler import GradScaler

