import sys
_module = sys.modules[__name__]
del sys
coremltools = _module
_deps = _module
converters = _module
_converters_entry = _module
_profile_utils = _module
libsvm = _module
_libsvm_converter = _module
_libsvm_util = _module
mil = _module
_deployment_compatibility = _module
backend = _module
backend_helper = _module
helper = _module
load = _module
passes = _module
adjust_io_to_supported_types = _module
fuse_activation_silu = _module
fuse_pow2_sqrt = _module
insert_image_preprocessing_op = _module
sanitize_name_strings = _module
test_passes = _module
test_helper = _module
test_load = _module
nn = _module
mil_to_nn_mapping_registry = _module
op_mapping = _module
alert_return_type_cast = _module
commingle_loop_vars = _module
conv1d_decomposition = _module
handle_return_inputs_as_outputs = _module
handle_return_unused_inputs = _module
handle_unused_inputs = _module
mlmodel_passes = _module
test_mlmodel_passes = _module
conftest = _module
converter = _module
debugging_utils = _module
experimental = _module
generic_conv_batchnorm_fusion = _module
generic_conv_bias_fusion = _module
generic_conv_scale_fusion = _module
generic_layernorm_instancenorm_pattern_fusion = _module
generic_linear_bias_fusion = _module
generic_pass_infrastructure = _module
frontend = _module
_utils = _module
milproto = _module
test_load = _module
tensorflow = _module
basic_graph_ops = _module
convert_utils = _module
dialect_ops = _module
dot_visitor = _module
naming_utils = _module
ops = _module
parse = _module
parsed_tf_node = _module
ssa_passes = _module
backfill_make_list_elem_type = _module
expand_tf_lstm = _module
tf_lstm_to_core_lstm = _module
test = _module
test_composite_ops = _module
test_custom_ops = _module
test_graphs = _module
test_ops = _module
test_parse = _module
test_parsed_tf_node = _module
test_tf_conversion_api = _module
testing_utils = _module
tf_graph_pass = _module
cond_to_where = _module
constant_propagation = _module
delete_asserts = _module
delete_constant = _module
delete_disconnected_nodes = _module
functionalize_loops = _module
fuse_dilation_conv = _module
insert_get_tuple = _module
quantization_pass = _module
tensor_array_transform = _module
variable_node_transform = _module
visitors = _module
tf_op_registry = _module
tfssa = _module
tensorflow2 = _module
remove_vacuous_cond = _module
test_v2_passes = _module
test_tf2_conversion_api = _module
test_v2_load = _module
test_v2_ops = _module
test_v2_ops_tf_keras = _module
rewrite_control_flow_functions = _module
converter = _module
dialect_ops = _module
exir_utils = _module
internal_graph = _module
load = _module
ops = _module
quantization_ops = _module
torch_tensor_assign_to_core = _module
torch_upsample_to_core_upsample = _module
test_custom_ops = _module
test_examples = _module
test_internal_graph = _module
test_passes = _module
test_torch_conversion_api = _module
test_torch_export_conversion_api = _module
test_torch_export_quantization = _module
test_torch_ops = _module
test_torch_quantization_ops = _module
test_torch_stateful_model = _module
testing_utils = _module
torch_op_registry = _module
torchir_passes = _module
torchscript_utils = _module
utils = _module
input_types = _module
block = _module
builder = _module
input_type = _module
operation = _module
defs = _module
_op_reqs = _module
complex_dialect_ops = _module
coreml_dialect = _module
iOS15 = _module
activation = _module
classify = _module
control_flow = _module
conv = _module
elementwise_binary = _module
elementwise_unary = _module
image_resizing = _module
linear = _module
normalization = _module
pool = _module
random = _module
recurrent = _module
reduction = _module
scatter_gather = _module
tensor_operation = _module
tensor_transformation = _module
iOS16 = _module
constexpr_ops = _module
tensor_transformation = _module
iOS17 = _module
iOS18 = _module
compression = _module
states = _module
transformers = _module
registry = _module
tests = _module
test_coreml_dialect = _module
iOS14 = _module
test_activation = _module
test_control_flow = _module
test_conv = _module
test_elementwise_binary = _module
test_elementwise_unary = _module
test_image_resizing = _module
test_linear = _module
test_normalization = _module
test_pool = _module
test_random = _module
test_recurrent = _module
test_reduction = _module
test_scatter_gather = _module
test_tensor_operation = _module
test_tensor_transformation = _module
test_constexpr_ops = _module
test_tensor_transformation = _module
test_quantization = _module
test_compression = _module
test_recurrent = _module
test_states = _module
test_transformers = _module
test_utils = _module
passes = _module
cleanup = _module
const_deduplication = _module
const_elimination = _module
dead_code_elimination = _module
dedup_op_and_var_names = _module
expand_dynamic_linear = _module
fuse_reduce_mean = _module
loop_invariant_elimination = _module
noop_elimination = _module
remove_redundant_ops = _module
remove_symbolic_reshape = _module
topological_reorder = _module
lower_complex_dialect_ops = _module
optimize_activation = _module
optimize_activation_quantization = _module
optimize_conv = _module
optimize_elementwise_binary = _module
optimize_linear = _module
optimize_normalization = _module
optimize_quantization = _module
optimize_repeat_ops = _module
optimize_state = _module
optimize_tensor_operation = _module
preprocess = _module
quantization = _module
randomize = _module
symbol_transform = _module
transformer = _module
graph_pass = _module
pass_pipeline = _module
pass_registry = _module
test_cleanup_passes = _module
test_lower_complex_dialect_ops = _module
test_optimize_linear_passes = _module
test_pass_pipeline = _module
test_passes = _module
test_quantization_passes = _module
test_reduce_transposes_pass = _module
test_state_passes = _module
test_symbol_transform = _module
program = _module
scope = _module
test_block = _module
test_debug = _module
test_programs = _module
test_types = _module
types = _module
annotate = _module
get_type_info = _module
global_methods = _module
symbolic = _module
type_bool = _module
type_complex = _module
type_dict = _module
type_double = _module
type_globals_pseudo_type = _module
type_int = _module
type_list = _module
type_mapping = _module
type_spec = _module
type_state = _module
type_str = _module
type_tensor = _module
type_tuple = _module
type_unknown = _module
type_void = _module
var = _module
test_inputs_outputs_shape = _module
testing_reqs = _module
_LinearSVC = _module
_LinearSVR = _module
_NuSVC = _module
_NuSVR = _module
_SVC = _module
_SVR = _module
sklearn = _module
_converter = _module
_converter_internal = _module
_decision_tree_classifier = _module
_decision_tree_regressor = _module
_dict_vectorizer = _module
_gradient_boosting_classifier = _module
_gradient_boosting_regressor = _module
_imputer = _module
_k_neighbors_classifier = _module
_linear_regression = _module
_logistic_regression = _module
_normalizer = _module
_one_hot_encoder = _module
_random_forest_classifier = _module
_random_forest_regressor = _module
_ridge_regression = _module
_sklearn_util = _module
_standard_scaler = _module
_svm_common = _module
_tree_ensemble = _module
xgboost = _module
_tree = _module
models = _module
_compiled_model = _module
_deprecation = _module
_feature_management = _module
_interface_management = _module
array_feature_extractor = _module
compute_device = _module
compute_plan = _module
datatypes = _module
feature_vectorizer = _module
ml_program = _module
compression_utils = _module
model = _module
nearest_neighbors = _module
neural_network = _module
flexible_shape_utils = _module
optimization_utils = _module
printer = _module
quantization_utils = _module
spec_inspection_utils = _module
update_optimizer_utils = _module
pipeline = _module
tree_ensemble = _module
optimize = _module
coreml = _module
_config = _module
_post_training_quantization = _module
_quantization_passes = _module
_model_debugger = _module
test_post_training_quantization = _module
_logging = _module
_typing = _module
dist_utils = _module
fsdp_utils = _module
graph_utils = _module
joint_compression_utils = _module
k_means = _module
math_utils = _module
metadata_utils = _module
python_utils = _module
report_utils = _module
state_dict_utils = _module
torch_utils = _module
transforms = _module
validation_utils = _module
version_utils = _module
base_model_optimizer = _module
layerwise_compression = _module
_quant = _module
algorithms = _module
input_cacher = _module
layerwise_compressor = _module
optimization_config = _module
palettization = _module
_custom_conversion = _module
_efficient_kmeans = _module
_fake_palettizer_tensor_hook = _module
_partitioner = _module
_supported_modules = _module
_utils = _module
fake_palettize = _module
palettization_config = _module
palettizer = _module
post_training_palettization = _module
sensitive_k_means = _module
pruning = _module
_base_pruner = _module
_base_pruning_method = _module
_utils = _module
magnitude_pruner = _module
pruning_scheduler = _module
quantization = _module
_annotation_config = _module
_backend_config = _module
_backend_config_utils = _module
_configure = _module
_coreml_quantizer = _module
_coreml_quantizer_utils = _module
_qconfig_mapping = _module
_utils = _module
modules = _module
conv_transpose = _module
conv_transpose_fused = _module
fused_modules = _module
observers = _module
qat_modules = _module
quantized_modules = _module
post_training_quantization = _module
quantization_config = _module
quantizer = _module
ArrayFeatureExtractor_pb2 = _module
AudioFeaturePrint_pb2 = _module
BayesianProbitRegressor_pb2 = _module
CategoricalMapping_pb2 = _module
ClassConfidenceThresholding_pb2 = _module
CustomModel_pb2 = _module
DataStructures_pb2 = _module
DictVectorizer_pb2 = _module
FeatureTypes_pb2 = _module
FeatureVectorizer_pb2 = _module
GLMClassifier_pb2 = _module
GLMRegressor_pb2 = _module
Gazetteer_pb2 = _module
Identity_pb2 = _module
Imputer_pb2 = _module
ItemSimilarityRecommender_pb2 = _module
LinkedModel_pb2 = _module
MIL_pb2 = _module
Model_pb2 = _module
NamedParameters_pb2 = _module
NearestNeighbors_pb2 = _module
NeuralNetwork_pb2 = _module
NonMaximumSuppression_pb2 = _module
Normalizer_pb2 = _module
OneHotEncoder_pb2 = _module
Parameters_pb2 = _module
SVM_pb2 = _module
Scaler_pb2 = _module
SoundAnalysisPreprocessing_pb2 = _module
TextClassifier_pb2 = _module
TreeEnsemble_pb2 = _module
VisionFeaturePrint_pb2 = _module
WordEmbedding_pb2 = _module
WordTagger_pb2 = _module
proto = _module
api = _module
test_api_examples = _module
test_api_visibilities = _module
blob = _module
test_weights = _module
test_compression = _module
test_utils = _module
modelpackage = _module
test_mlmodel = _module
test_modelpackage = _module
test_compiled_model = _module
test_custom_neural_nets = _module
test_model = _module
test_neural_networks = _module
test_nn_builder = _module
test_numpy_nn_layers = _module
test_simple_nn_inference = _module
test_tf_numeric = _module
test_optimize_api = _module
test_passes = _module
test_post_training_quantization = _module
conftest = _module
conversion = _module
conversion_utils = _module
joint = _module
test_joint_compression_conversion = _module
test_palettization_conversion = _module
test_pruning_conversion = _module
test_quantization_conversion = _module
test_api_ordering = _module
test_algorithms = _module
test_quant = _module
mnist = _module
multi_input_net = _module
palettization_utils = _module
test_palettization_api = _module
test_palettization_utils = _module
test_palettizer = _module
test_post_training_palettization = _module
test_sensitive_k_means = _module
pruning_utils = _module
test_base_pruner = _module
test_magnitude_pruner = _module
test_pruning_scheduler = _module
test_configure = _module
test_coreml_quantizer = _module
test_post_training_quantization = _module
test_quantizer = _module
test_utils = _module
smoke_test = _module
test_api_surface = _module
test_base_optimizer = _module
test_fsdp_utils = _module
test_k_means = _module
test_metadata_utils = _module
test_report_utils = _module
test_torch_utils = _module
test_validation_utils = _module
utils = _module
test_model_updatable = _module
test_pipeline = _module
sklearn_tests = _module
test_NuSVC = _module
test_NuSVR = _module
test_SVC = _module
test_SVR = _module
test_categorical_imputer = _module
test_composite_pipelines = _module
test_dict_vectorizer = _module
test_feature_names = _module
test_glm_classifier = _module
test_imputer = _module
test_io_types = _module
test_k_neighbors_classifier = _module
test_linear_regression = _module
test_nearest_neighbors_builder = _module
test_normalizer = _module
test_one_hot_encoder = _module
test_random_forest_classifier = _module
test_random_forest_classifier_numeric = _module
test_random_forest_regression = _module
test_random_forest_regression_numeric = _module
test_ridge_regression = _module
test_standard_scalar = _module
xgboost_tests = _module
test_boosted_trees_classifier = _module
test_boosted_trees_classifier_numeric = _module
test_boosted_trees_regression = _module
test_boosted_trees_regression_numeric = _module
test_decision_tree_classifier = _module
test_decision_tree_classifier_numeric = _module
test_decision_tree_regression = _module
test_decision_tree_regression_numeric = _module
version = _module
kmeans1d = _module
core = _module
setup = _module
test_kmeans1d = _module
benchmarks = _module
python = _module
py_benchmark = _module
util = _module
big_query_utils = _module
result_parser = _module
result_uploader = _module
conformance_python = _module
update_failure_list = _module
add_person = _module
list_people = _module
generate_changelog = _module
make_test_output = _module
pddm = _module
pddm_tests = _module
conf = _module
generate_docs = _module
google = _module
protobuf = _module
compiler = _module
descriptor = _module
descriptor_database = _module
descriptor_pool = _module
internal = _module
_parameterized = _module
api_implementation = _module
containers = _module
decoder = _module
descriptor_database_test = _module
descriptor_pool_test = _module
descriptor_test = _module
encoder = _module
enum_type_wrapper = _module
extension_dict = _module
generator_test = _module
import_test_package = _module
json_format_test = _module
keywords_test = _module
message_factory_test = _module
message_listener = _module
message_test = _module
proto_builder_test = _module
python_message = _module
reflection_test = _module
service_reflection_test = _module
symbol_database_test = _module
test_util = _module
testing_refleaks = _module
text_encoding_test = _module
text_format_test = _module
type_checkers = _module
unknown_fields_test = _module
well_known_types = _module
well_known_types_test = _module
wire_format = _module
wire_format_test = _module
json_format = _module
message = _module
message_factory = _module
proto_builder = _module
pyext = _module
cpp_message = _module
reflection = _module
service = _module
service_reflection = _module
symbol_database = _module
text_encoding = _module
text_format = _module
mox = _module
protobuf_distutils = _module
generate_py_protobufs = _module
stubout = _module
update_compatibility_version = _module
update_version = _module
benchmark = _module
noxfile = _module
pybind11 = _module
_version = _module
commands = _module
setup_helpers = _module
env = _module
test_files = _module
test_setuphelper = _module
test_async = _module
test_buffers = _module
test_builtin_casters = _module
test_call_policies = _module
test_callbacks = _module
test_chrono = _module
test_class = _module
test_const_name = _module
test_constants_and_functions = _module
test_copy_move = _module
test_custom_type_casters = _module
test_custom_type_setup = _module
test_docstring_options = _module
test_eigen_matrix = _module
test_eigen_tensor = _module
test_interpreter = _module
test_trampoline = _module
test_enum = _module
test_eval = _module
test_eval_call = _module
test_exceptions = _module
test_factory_constructors = _module
test_gil_scoped = _module
test_iostream = _module
test_kwargs_and_defaults = _module
test_local_bindings = _module
test_methods_and_attributes = _module
test_modules = _module
test_multiple_inheritance = _module
test_numpy_array = _module
test_numpy_dtypes = _module
test_numpy_vectorize = _module
test_opaque_types = _module
test_operator_overloading = _module
test_pickling = _module
test_python_multiple_inheritance = _module
test_pytypes = _module
test_sequences_and_iterators = _module
test_smart_ptr = _module
test_stl = _module
test_stl_binders = _module
test_tagbased_polymorphic = _module
test_thread = _module
test_type_caster_pyobject_ptr = _module
test_union = _module
test_unnamed_namespace_a = _module
test_unnamed_namespace_b = _module
test_vector_unique_ptr_member = _module
test_virtual_functions = _module
codespell_ignore_lines_from_errors = _module
libsize = _module
make_changelog = _module
conf = _module
dkm_palettization = _module
magnitude_pruning = _module
linear_quantization = _module
readme_session = _module
build_tag = _module

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


import re as _re


import collections


from typing import List


from typing import Optional


from typing import Text


from typing import Union


import warnings as _warnings


from typing import Tuple


import itertools


import numpy as np


import math


from collections import OrderedDict


from enum import Enum


from typing import Dict


import torch as torch


from torch.jit._script import RecursiveScriptModule


import torch


from typing import Any


import torch as _torch


import math as _math


import numbers


import re


from collections.abc import Iterable


import numpy as _np


import torch.nn as nn


import torch.nn.functional as F


from torch.export import export_for_training


from torch.ao.quantization.quantize_pt2e import convert_pt2e


from torch.ao.quantization.quantize_pt2e import prepare_pt2e


from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e


from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer


from torch.ao.quantization.quantizer.xnnpack_quantizer import get_symmetric_quantization_config


import numpy.testing


import torchvision


from typing import Callable


from collections import defaultdict


import scipy


import functools


import time


from typing import Set


import copy


from typing import ClassVar


from copy import deepcopy as _deepcopy


from typing import Dict as _Dict


from typing import List as _List


from typing import Optional as _Optional


import numpy as _numpy


from abc import ABC


from abc import abstractmethod


from typing import IO


import logging


from typing import Any as _Any


from typing import Callable as _Callable


import torch.distributed as _dist


from abc import ABC as _ABC


from abc import abstractmethod as _abstractmethod


from functools import partial as _partial


from typing import Iterable as _Iterable


from typing import Type as _Type


from torch.distributed.fsdp.wrap import ModuleWrapPolicy as _TorchModuleWrapPolicy


from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy as _size_based_auto_wrap_policy


from typing import Tuple as _Tuple


import logging as _logging


import queue as _queue


from typing import Union as _Union


import torch.multiprocessing as _mp


from collections import OrderedDict as _OrderedDict


from typing import IO as _IO


from typing import Type


from typing import Mapping


from typing import NamedTuple


import torch.nn as _nn


from enum import Enum as _Enum


import copy as _copy


from collections import UserDict as _UserDict


import time as _time


from typing import NewType as _NewType


import torch.nn.functional as _F


import torch.nn.qat as _nnqat


from torch.ao.quantization.observer import ObserverBase as _ObserverBase


from torch.quantization import FakeQuantize as _FakeQuantize


from torch.ao.quantization import FakeQuantize as _FakeQuantize


from torch.distributed.fsdp import FullStateDictConfig as _FullStateDictConfig


from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP


from torch.distributed.fsdp import ShardingStrategy as _ShardingStrategy


from torch.distributed.fsdp import StateDictType as _StateDictType


import types as _types


from typing import NamedTuple as _NamedTuple


from typing import cast as _cast


import torch.nn.utils.prune as _prune


import torch.utils.hooks as _hooks


import torch.ao.quantization as _aoquant


from torch.ao.quantization.quantizer.quantizer import QuantizationSpec as _TorchQuantizationSpec


from typing import Set as _Set


import torch.ao.nn.qat as _nnq


import torch.ao.nn.quantized.reference as _nnr


import torch.nn.intrinsic as _nni


import torch.nn.intrinsic.qat as _nniq


from torch.ao.quantization.backend_config import BackendConfig as _BackendConfig


from torch.ao.quantization.backend_config import BackendPatternConfig as _BackendPatternConfig


from torch.ao.quantization.backend_config import DTypeWithConstraints as _DTypeWithConstraints


from torch.ao.quantization.backend_config import DTypeConfig as _DTypeConfig


from torch.ao.quantization.backend_config import ObservationType as _ObservationType


from collections import defaultdict as _defaultdict


import torch.fx as _fx


import torch.nn.intrinsic.qat as _nniqat


from torch.ao.quantization.fx.custom_config import PrepareCustomConfig as _PrepareCustomConfig


from torch.quantization.quantize_fx import prepare_qat_fx as _prepare_qat_fx


from torch.ao.quantization.quantizer.quantizer import Quantizer as _TorchQuantizer


from torch.ao.quantization.quantizer.xnnpack_quantizer import _get_module_name_filter


from torch.fx import Node as _Node


import itertools as _itertools


from torch.ao.quantization.quantizer.quantizer import FixedQParamsQuantizationSpec as _FixedQParamsQuantizationSpec


from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation as _QuantizationAnnotation


from torch.ao.quantization.quantizer.quantizer import QuantizationSpecBase as _TorchQuantizationSpecBase


from torch.ao.quantization.quantizer.quantizer import SharedQuantizationSpec as _SharedQuantizationSpec


from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import _is_annotated


from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import _mark_nodes_as_annotated


from torch.fx.passes.utils.matcher_with_name_node_map_utils import SubgraphMatcherWithNameNodeMap as _SubgraphMatcherWithNameNodeMap


from torch.fx.passes.utils.source_matcher_utils import get_source_partitions as _get_source_partitions


from torch.ao.nn.quantized.reference.modules.utils import _quantize_and_dequantize_weight_decomposed


from typing import TypeVar as _TypeVar


from torch import Tensor as _Tensor


from torch.ao.nn.intrinsic import _FusedModule


from torch.nn.common_types import _size_1_t


from torch.nn.common_types import _size_2_t


from torch.nn.common_types import _size_3_t


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _triple


import torch.ao.nn.intrinsic as nni


from torch import Tensor


from torch.nn.utils import fuse_conv_bn_weights


import torch.ao.nn.intrinsic as _nni


import torch.ao.nn.qat as _nnqat


import torch.ao.nn.quantized.reference as _reference


from enum import unique as _unique


from torch.ao.quantization.fx.custom_config import ConvertCustomConfig as _ConvertCustomConfig


from torch.ao.quantization.quantize_fx import convert_to_reference_fx as _convert_to_reference_fx


from collections import Counter


import random


from torchvision import datasets


from torchvision import transforms


import torch.functional as F


import torch.ao.nn.quantized.reference


import torch.ao.quantization


import torch.nn.intrinsic


import torch.nn.intrinsic.qat


import torch.nn.qat


import torch.nn.quantized


from torch.fx import Node


import torch.nn.quantized.modules.utils


import torch.utils.data

