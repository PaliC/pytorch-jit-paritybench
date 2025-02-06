import sys
_module = sys.modules[__name__]
del sys
autodistill = _module
classification = _module
classification_base_model = _module
classification_target_model = _module
cli = _module
core = _module
base_model = _module
composed_detection_model = _module
embedding_model = _module
embedding_ontology = _module
ontology = _module
target_model = _module
detection = _module
caption_ontology = _module
detection_base_model = _module
detection_ontology = _module
detection_target_model = _module
helpers = _module
registry = _module
text_classification = _module
text_classification_base_model = _module
text_classification_ontology = _module
text_classification_target_model = _module
utils = _module
setup = _module
test_hello = _module
test_load_image = _module

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

