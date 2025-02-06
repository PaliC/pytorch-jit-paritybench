import sys
_module = sys.modules[__name__]
del sys
contrib = _module
conftest = _module
sentence_transformers = _module
test_sentence_transformers = _module
spacy = _module
test_spacy = _module
timm = _module
test_timm = _module
utils = _module
setup = _module
huggingface_hub = _module
_commit_api = _module
_commit_scheduler = _module
_inference_endpoints = _module
_local_folder = _module
_login = _module
_snapshot_download = _module
_space_api = _module
_tensorboard_logger = _module
_upload_large_folder = _module
_webhooks_payload = _module
_webhooks_server = _module
commands = _module
_cli_utils = _module
delete_cache = _module
download = _module
env = _module
huggingface_cli = _module
lfs = _module
repo_files = _module
scan_cache = _module
tag = _module
upload = _module
upload_large_folder = _module
user = _module
version = _module
community = _module
constants = _module
errors = _module
fastai_utils = _module
file_download = _module
hf_api = _module
hf_file_system = _module
hub_mixin = _module
inference = _module
_client = _module
_common = _module
_generated = _module
_async_client = _module
types = _module
audio_classification = _module
audio_to_audio = _module
automatic_speech_recognition = _module
base = _module
chat_completion = _module
depth_estimation = _module
document_question_answering = _module
feature_extraction = _module
fill_mask = _module
image_classification = _module
image_segmentation = _module
image_to_image = _module
image_to_text = _module
object_detection = _module
question_answering = _module
sentence_similarity = _module
summarization = _module
table_question_answering = _module
text2text_generation = _module
text_classification = _module
text_generation = _module
text_to_audio = _module
text_to_image = _module
text_to_speech = _module
text_to_video = _module
token_classification = _module
translation = _module
video_classification = _module
visual_question_answering = _module
zero_shot_classification = _module
zero_shot_image_classification = _module
zero_shot_object_detection = _module
_providers = _module
fal_ai = _module
hf_inference = _module
replicate = _module
sambanova = _module
together = _module
inference_api = _module
keras_mixin = _module
repocard = _module
repocard_data = _module
repository = _module
serialization = _module
_base = _module
_dduf = _module
_tensorflow = _module
_torch = _module
_auth = _module
_cache_assets = _module
_cache_manager = _module
_chunk_utils = _module
_datetime = _module
_deprecation = _module
_experimental = _module
_fixes = _module
_git_credential = _module
_headers = _module
_hf_folder = _module
_http = _module
_lfs = _module
_pagination = _module
_paths = _module
_runtime = _module
_safetensors = _module
_subprocess = _module
_telemetry = _module
_typing = _module
_validators = _module
endpoint_helpers = _module
insecure_hashlib = _module
logging = _module
sha = _module
tqdm = _module
tests = _module
test_auth = _module
test_auth_cli = _module
test_cache_layout = _module
test_cache_no_symlinks = _module
test_cli = _module
test_command_delete_cache = _module
test_commit_api = _module
test_commit_scheduler = _module
test_dduf = _module
test_endpoint_helpers = _module
test_fastai_integration = _module
test_file_download = _module
test_hf_api = _module
test_hf_file_system = _module
test_hub_mixin = _module
test_hub_mixin_pytorch = _module
test_inference_api = _module
test_inference_async_client = _module
test_inference_client = _module
test_inference_endpoints = _module
test_inference_providers = _module
test_inference_text_generation = _module
test_inference_types = _module
test_init_lazy_loading = _module
test_keras_integration = _module
test_lfs = _module
test_local_folder = _module
test_login_utils = _module
test_offline_utils = _module
test_repocard = _module
test_repocard_data = _module
test_repository = _module
test_serialization = _module
test_snapshot_download = _module
test_testing_configuration = _module
test_tf_import = _module
test_utils_assets = _module
test_utils_cache = _module
test_utils_chunks = _module
test_utils_cli = _module
test_utils_datetime = _module
test_utils_deprecation = _module
test_utils_errors = _module
test_utils_experimental = _module
test_utils_fixes = _module
test_utils_git_credentials = _module
test_utils_headers = _module
test_utils_hf_folder = _module
test_utils_http = _module
test_utils_pagination = _module
test_utils_paths = _module
test_utils_runtime = _module
test_utils_sha = _module
test_utils_telemetry = _module
test_utils_tqdm = _module
test_utils_typing = _module
test_utils_validators = _module
test_webhooks_server = _module
test_windows = _module
testing_constants = _module
testing_utils = _module
_legacy_check_future_compatible_signatures = _module
check_all_variable = _module
check_contrib_list = _module
check_inference_input_params = _module
check_static_imports = _module
check_task_parameters = _module
generate_async_inference_client = _module
generate_inference_types = _module
helpers = _module
push_repocard_examples = _module

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


from typing import TYPE_CHECKING


from typing import List


from typing import Optional


from typing import Union


import inspect


from typing import Any


from typing import Callable


from typing import ClassVar


from typing import Dict


from typing import Protocol


from typing import Tuple


from typing import Type


from typing import TypeVar


import logging


import re


import time


import warnings


from typing import Iterable


from typing import Literal


from typing import overload


from typing import AsyncIterable


from typing import Set


from typing import Iterator


from typing import TypedDict


from typing import Generator


from collections import defaultdict


from collections import namedtuple


from functools import lru_cache


from typing import NamedTuple

