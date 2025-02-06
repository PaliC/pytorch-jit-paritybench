import sys
_module = sys.modules[__name__]
del sys
model = _module
bot = _module
embeddings_inf2 = _module
test_embeddings_inf2 = _module
test_embeddings_inf2_client = _module
sdxl_inf2 = _module
test_sdxl_inf2 = _module
test_sdxl_inf2_client = _module
test_openai_client = _module
model = _module
test_model = _module
model = _module
chat = _module
test_grpc_chat = _module
test_http_chat = _module
summarize_audio = _module
test_m2m_auth = _module
nos = _module
cli = _module
hub = _module
predict = _module
profile = _module
serve = _module
system = _module
utils = _module
client = _module
grpc = _module
common = _module
cloudpickle = _module
exceptions = _module
git = _module
helpers = _module
io = _module
base = _module
opencv = _module
metaclass = _module
profiler = _module
runtime = _module
shm = _module
spec = _module
system = _module
tasks = _module
types = _module
constants = _module
executors = _module
ray = _module
config = _module
hf = _module
logging = _module
managers = _module
model = _module
pool = _module
models = _module
_noop = _module
blip = _module
clip = _module
dreambooth = _module
hub = _module
faster_rcnn = _module
llm = _module
monodepth = _module
mmdetection = _module
owlvit = _module
sam = _module
stable_diffusion = _module
super_resolution = _module
ldm = _module
swin2sr = _module
tts = _module
whisper = _module
yolox = _module
device = _module
protoc = _module
server = _module
_docker = _module
_runtime = _module
_service = _module
_exceptions = _module
_security = _module
_utils = _module
telemetry = _module
benchmark = _module
conftest = _module
utils = _module
version = _module
test_cli_hub = _module
test_cli_predict = _module
test_cli_profile = _module
test_cli_serve = _module
test_cli_system = _module
test_grpc_client = _module
test_grpc_client_integration = _module
test_http_client = _module
test_opencv = _module
test_common = _module
test_common_git = _module
test_common_metaclass = _module
test_common_model_resources = _module
test_common_spec = _module
test_common_types = _module
test_helpers = _module
test_system = _module
test_ray = _module
test_hub = _module
test_hub_hf = _module
test_hub_inference = _module
test_client_integration = _module
locustfile = _module
test_model_manager = _module
test_dreambooth = _module
test_dreambooth_config = _module
test_mmdetection = _module
test_openmmlab_config = _module
test_blip = _module
test_clip = _module
test_controlnet = _module
test_llm = _module
test_monodepth = _module
test_noop = _module
test_object_detection = _module
test_owlvit = _module
test_sam = _module
test_stable_diffusion = _module
test_superresolution = _module
test_tts = _module
test_whisper = _module
test_neuron_device = _module
test_docker_runtime = _module
test_inference_service = _module
test_inference_service_runtime = _module
test_server_utils = _module
test_constants = _module
test_exceptions = _module
test_imports = _module
test_logging = _module
test_protoc = _module
test_version = _module
test_data = _module
test_utils = _module

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


from typing import List


from typing import Union


import torch


from typing import Iterable


from typing import Any


from typing import Dict


import numpy as np


import time


from typing import Iterator


from typing import Tuple


from typing import Callable


from typing import Optional


import pandas as pd


from torch.profiler import ProfilerActivity


from torch.profiler import profile


from torch.profiler import record_function as _record_function


import inspect


import math


import re


from functools import cached_property


from typing import Literal


from functools import lru_cache


import typing


from typing import Generic


from typing import TypeVar


from collections import OrderedDict


from enum import Enum


import torchvision.transforms.functional as F


from torchvision import ops


from torch.utils.benchmark import Timer


from typing import get_args

