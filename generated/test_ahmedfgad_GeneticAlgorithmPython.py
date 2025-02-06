import sys
_module = sys.modules[__name__]
del sys
Example_GeneticAlgorithm = _module
ga = _module
conf = _module
XOR_classification = _module
cancer_dataset = _module
cancer_dataset_generator = _module
image_classification_CNN = _module
image_classification_Dense = _module
regression_example = _module
XOR_classification = _module
image_classification_CNN = _module
image_classification_Dense = _module
regression_example = _module
example_clustering_2 = _module
example_clustering_3 = _module
example_image_classification = _module
example = _module
example_custom_operators = _module
example_dynamic_population_size = _module
example_fitness_wrapper = _module
example_logger = _module
example_multi_objective = _module
example_parallel_processing = _module
example_XOR_classification = _module
example_classification = _module
example_regression = _module
example_regression_fish = _module
extract_features = _module
pygad_lifecycle = _module
pygad = _module
cnn = _module
gacnn = _module
gann = _module
helper = _module
unique = _module
kerasga = _module
nn = _module
torchga = _module
torchga = _module
utils = _module
crossover = _module
mutation = _module
nsga2 = _module
parent_selection = _module
visualize = _module
plot = _module
setup = _module
test_adaptive_mutation = _module
test_allow_duplicate_genes = _module
test_crossover_mutation = _module
test_gene_space = _module
test_gene_space_allow_duplicate_genes = _module
test_lifecycle_callbacks_calls = _module
test_number_fitness_function_calls = _module
test_save_solutions = _module
test_stop_criteria = _module

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


import numpy


import copy

