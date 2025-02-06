import sys
_module = sys.modules[__name__]
del sys
hack_attention = _module
mrsa = _module
freecustom_controlnet = _module
freecustom_stable_diffusion = _module
pipeline_blip_diffusion_freecustom = _module
pipeline_controlnet_freecustom = _module
pipeline_stable_diffusion_freecustom = _module
run_clip_image_score_multi = _module
run_clip_image_score_single = _module
run_clipscore_mulit = _module
run_clipscore_single = _module
run_dinov2_image_score_multi = _module
run_dinov2_image_score_single = _module
run_iqa_single = _module
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


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from torchvision.transforms import ToTensor


from torchvision.utils import save_image


from typing import List


from typing import Optional


from typing import Union


import inspect


from typing import Any


from typing import Callable


from typing import Dict


from typing import Tuple


from torchvision.io import read_image


from matplotlib import cm

