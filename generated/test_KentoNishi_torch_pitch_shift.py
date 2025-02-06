import sys
_module = sys.modules[__name__]
del sys
example = _module
setup = _module
torch_pitch_shift = _module
main = _module

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


import numpy as np


import torch


from scipy.io import wavfile


from collections import Counter


from functools import reduce


from itertools import chain


from itertools import count


from itertools import islice


from itertools import repeat


from math import log2


from typing import Callable


from typing import List


from typing import Optional


from typing import Union


import torchaudio


import torchaudio.transforms as T


from torch.nn.functional import pad

