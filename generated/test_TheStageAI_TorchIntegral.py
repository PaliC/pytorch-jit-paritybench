import sys
_module = sys.modules[__name__]
del sys
conf = _module
imagenet = _module
mnist = _module
nin_cifar = _module
edsr = _module
setup = _module
tsp_test = _module
torch_integral = _module
graph = _module
group = _module
operations = _module
trace = _module
grid = _module
integral_group = _module
model = _module
parametrizations = _module
base_parametrization = _module
interpolation_weights = _module
permutation = _module
quadrature = _module
tsp_solver = _module
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


import re


import torch


from torchvision import models


import torchvision.transforms as transforms


from torchvision import datasets


import torch.nn as nn


import torchvision


import time


import random


from scipy.special import roots_legendre


import copy


from typing import Any


from typing import Mapping


from torch.nn.utils import parametrize


from functools import reduce


import torch.nn.functional as F


from typing import Tuple


from typing import Dict


from typing import List


import torch.fx as fx


from collections import OrderedDict


class MnistNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3, padding=1, bias=True, padding_mode='replicate')
        self.conv_2 = nn.Conv2d(16, 32, 5, padding=2, bias=True, padding_mode='replicate')
        self.conv_3 = nn.Conv2d(32, 64, 5, padding=2, bias=True, padding_mode='replicate')
        self.f_1 = nn.ReLU()
        self.f_2 = nn.ReLU()
        self.f_3 = nn.ReLU()
        self.pool = nn.AvgPool2d(2, 2)
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.f_1(self.conv_1(x))
        x = self.pool(x)
        x = self.f_2(self.conv_2(x))
        x = self.pool(x)
        x = self.f_3(self.conv_3(x))
        x = self.pool(x)
        x = self.linear(x[:, :, 0, 0])
        return x


class RelatedGroup(torch.nn.Module):
    """
    Class for grouping tensors and parameters.
    Group is a collection of paris of tensor and it's dimension.
    Two parameter tensors are considered to be in the same group
    if they should have the same integration grid.
    Group can contain subgroups. This means that parent group's grid is a con
    catenation of subgroups grids.

    Parameters
    ----------
    size: int.
        Each tensor in the group should have the same size along certain dimension.
    """

    def __init__(self, size):
        super(RelatedGroup, self).__init__()
        self.size = size
        self.subgroups = None
        self.parents = []
        self.params = []
        self.tensors = []
        self.operations = []

    def forward(self):
        pass

    def copy_attributes(self, group):
        self.size = group.size
        self.subgroups = group.subgroups
        self.parents = group.parents
        self.params = group.params
        self.tensors = group.tensors
        self.operations = group.operations
        for parent in self.parents:
            if group in parent.subgroups:
                i = parent.subgroups.index(group)
                parent.subgroups[i] = self
        if self.subgroups is not None:
            for sub in self.subgroups:
                if group in sub.parents:
                    i = sub.parents.index(group)
                    sub.parents[i] = self
        for param in self.params:
            param['value'].related_groups[param['dim']] = self
        for tensor in self.tensors:
            tensor['value'].related_groups[tensor['dim']] = self

    def append_param(self, name, value, dim, operation=None):
        """
        Adds parameter tensor to the group.

        Parameters
        ----------
        name: str.
        value: torch.Tensor.
        dim: int.
        operation: str.
        """
        self.params.append({'value': value, 'name': name, 'dim': dim, 'operation': operation})

    def append_tensor(self, value, dim, operation=None):
        """
        Adds tensor to the group.

        Parameters
        ----------
        value: torch.Tensor.
        dim: int.
        operation: str.
        """
        self.tensors.append({'value': value, 'dim': dim, 'operation': operation})

    def clear_params(self):
        self.params = []

    def clear_tensors(self):
        self.tensors = []

    def set_subgroups(self, groups):
        self.subgroups = groups
        for subgroup in self.subgroups:
            if subgroup is not None:
                subgroup.parents.append(self)

    def build_operations_set(self):
        """Builds set of operations in the group."""
        self.operations = set([t['operation'] for t in self.tensors])

    def count_parameters(self):
        ans = 0
        for p in self.params:
            ans += p['value'].numel()
        return ans

    def __str__(self):
        result = ''
        for p in self.params:
            result += p['name'] + ': ' + str(p['dim']) + '\n'
        return result

    @staticmethod
    def append_to_groups(tensor, operation=None):
        attr_name = 'related_groups'
        if hasattr(tensor, attr_name):
            for i, g in enumerate(getattr(tensor, attr_name)):
                if g is not None:
                    g.append_tensor(tensor, i, operation)


class IGrid(torch.nn.Module):
    """Base Grid class."""

    def __init__(self):
        super(IGrid, self).__init__()
        self.curr_grid = None
        self.eval_size = None

    def forward(self):
        """
        Performs forward pass. Generates new grid if
        last generated grid is not saved, else returns saved one.

        Returns
        -------
        torch.Tensor.
            Generated grid points.
        """
        if self.curr_grid is None:
            out = self.generate_grid()
        else:
            out = self.curr_grid
        return out

    def ndim(self):
        """Returns dimensionality of grid object."""
        return 1

    def size(self):
        return self.eval_size

    def generate_grid(self):
        """Samples new grid points."""
        raise NotImplementedError('Implement this method in derived class.')


class ConstantGrid1D(IGrid):
    """
    Class implements IGrid interface for fixed grid.

    Parameters
    ----------
    init_value: torch.Tensor.
    """

    def __init__(self, init_value):
        super(ConstantGrid1D, self).__init__()
        self.curr_grid = init_value

    def generate_grid(self):
        return self.curr_grid


class TrainableGrid1D(IGrid):
    """Grid with TrainablePartition.

    Parameters
    ----------
    size: int.
    init_value: torch.Tensor.
    """

    def __init__(self, size, init_value=None):
        super(TrainableGrid1D, self).__init__()
        self.eval_size = size
        self.curr_grid = torch.nn.Parameter(torch.linspace(-1, 1, size))
        if init_value is not None:
            assert size == init_value.shape[0]
            self.curr_grid.data = init_value

    def generate_grid(self):
        return self.curr_grid


class L1Grid1D(IGrid):

    def __init__(self, group, size):
        super().__init__()
        indices = self.get_indices(group, size).cpu()
        self.curr_grid = torch.linspace(-1, 1, group.size).index_select(0, indices)
        self.curr_grid = self.curr_grid.sort().values

    def generate_grid(self):
        return self.generate_grid

    def get_indices(self, group, size):
        device = group.params[0]['value'].device
        channels_importance = torch.zeros(group.size, device=device)
        for param in group.params:
            if 'bias' not in param['name']:
                tensor = param['value']
                tensor = param['function'](tensor)
                dim = param['dim']
                tensor = tensor.transpose(0, dim).reshape(group.size, -1)
                mean = tensor.abs().mean(dim=1)
                channels_importance += mean
        return torch.argsort(channels_importance)[:size]


class MultiTrainableGrid1D(IGrid):

    def __init__(self, full_grid, index, num_grids):
        super(MultiTrainableGrid1D, self).__init__()
        self.curr_grid = None
        self.num_grids = num_grids
        self.full_grid = full_grid
        self.index = index
        self.generate_grid()

    def generate_grid(self):
        grid_len = 1.0 / self.num_grids
        start = self.index * grid_len
        end = start + grid_len
        grid = self.full_grid[(self.full_grid >= start) & (self.full_grid < end)]
        self.curr_grid = 2 * (grid - start) / grid_len - 1.0
        return self.curr_grid


class TrainableDeltasGrid1D(IGrid):
    """Grid with TrainablePartition parametrized with deltas.

    Parameters
    ----------
    size: int.
    """

    def __init__(self, size):
        super(TrainableDeltasGrid1D, self).__init__()
        self.eval_size = size
        self.deltas = torch.nn.Parameter(torch.zeros(size - 1))
        self.curr_grid = None

    def generate_grid(self):
        self.curr_grid = torch.cumsum(self.deltas.abs(), dim=0)
        self.curr_grid = torch.cat([torch.zeros(1), self.curr_grid])
        self.curr_grid = self.curr_grid * 2 - 1
        return self.curr_grid


class RandomLinspace(IGrid):
    """
    Grid which generates random sized tensor each time,
    when generate_grid method is called.
    Size of tensor is sampled from ``size_distribution``.

    Parameters
    ----------
    size_distribution: Distribution.
    noise_std: float.
    """

    def __init__(self, size_distribution, noise_std=0):
        super(RandomLinspace, self).__init__()
        self.distribution = size_distribution
        self.eval_size = size_distribution.max_val
        self.noise_std = noise_std
        self.generate_grid()

    def generate_grid(self):
        if self.training:
            size = self.distribution.sample()
        else:
            size = self.eval_size
        self.curr_grid = torch.linspace(-1, 1, size)
        if self.noise_std > 0:
            noise = torch.normal(torch.zeros(size), self.noise_std * torch.ones(size))
            self.curr_grid = self.curr_grid + noise
        return self.curr_grid

    def resize(self, new_size):
        """Set new value for evaluation size."""
        self.eval_size = new_size
        self.generate_grid()


class RandomLegendreGrid(RandomLinspace):

    def __init__(self, size_distribution):
        super(RandomLinspace, self).__init__()
        self.distribution = size_distribution
        self.eval_size = size_distribution.max_val
        self.generate_grid()

    def generate_grid(self):
        if self.training:
            size = self.distribution.sample()
        else:
            size = self.eval_size
        self.curr_grid, _ = roots_legendre(size)
        self.curr_grid = torch.tensor(self.curr_grid, dtype=torch.float32)
        return self.curr_grid


class CompositeGrid1D(IGrid):
    """Grid which consist of concatenated IGrid objects."""

    def __init__(self, grids):
        super(CompositeGrid1D, self).__init__()
        self.grids = torch.nn.ModuleList(grids)
        size = self.size()
        self.proportions = [((grid.size() - 1) / (size - 1)) for grid in grids]
        self.generate_grid()

    def reset_grid(self, index, new_grid):
        self.grids[index] = new_grid
        self.generate_grid()

    def generate_grid(self):
        g_list = []
        start = 0.0
        h = 1 / (self.size() - 1)
        device = None
        for i, grid in enumerate(self.grids):
            g = grid.generate_grid()
            device = g.device if device is None else device
            g = (g + 1.0) / 2.0
            g = start + g * self.proportions[i]
            g_list.append(g)
            start += self.proportions[i] + h
        self.curr_grid = 2.0 * torch.cat(g_list) - 1.0
        return self.curr_grid

    def size(self):
        return sum([g.size() for g in self.grids])


class GridND(IGrid):
    """N-dimensional grid, each dimension of which is an object of type IGrid."""

    def __init__(self, grid_objects):
        super(GridND, self).__init__()
        self.grid_objects = torch.nn.ModuleList(grid_objects)
        self.generate_grid()

    def ndim(self):
        """Returns dimensionality of grid object."""
        return sum([grid.ndim() for grid in self.grid_objects])

    def reset_grid(self, dim, new_grid):
        """Replaces grid at given index."""
        self.grid_objects[dim] = new_grid
        self.generate_grid()

    def generate_grid(self):
        self.curr_grid = [grid.generate_grid() for grid in self.grid_objects]
        return self.curr_grid

    def forward(self):
        self.curr_grid = [grid() for grid in self.grid_objects]
        return self.curr_grid

    def __iter__(self):
        return iter(self.grid_objects)


class Distribution:
    """
    Base class for grid size distribution.

    Attributes
    ----------
    min_val: int.
        Minimal possible random value.
    max_val: int.
        Maximal possible random value.
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def sample(self):
        """Samples random integer number from distribution."""
        raise NotImplementedError('Implement this method in derived class.')


class UniformDistribution(Distribution):

    def __init__(self, min_val, max_val):
        super().__init__(min_val, max_val)

    def sample(self):
        return random.randint(self.min_val, self.max_val)


class IntegralGroup(RelatedGroup):
    """ """

    def __init__(self, size):
        super(RelatedGroup, self).__init__()
        self.size = size
        self.subgroups = None
        self.parents = []
        self.grid = None
        self.params = []
        self.tensors = []
        self.operations = []

    def forward(self):
        self.grid.generate_grid()

    def grid_size(self):
        """Returns size of the grid."""
        return self.grid.size()

    def clear(self, new_grid=None):
        """Resets grid and removes cached values."""
        for param_dict in self.params:
            function = param_dict['function']
            dim = list(function.grid).index(self.grid)
            grid = new_grid if new_grid is not None else self.grid
            function.grid.reset_grid(dim, grid)
            function.clear()

    def initialize_grids(self):
        """Sets default RandomLinspace grid."""
        if self.grid is None:
            if self.subgroups is not None:
                for subgroup in self.subgroups:
                    if subgroup.grid is None:
                        subgroup.initialize_grids()
                self.grid = CompositeGrid1D([sub.grid for sub in self.subgroups])
            else:
                distrib = UniformDistribution(self.size, self.size)
                self.grid = RandomLinspace(distrib)

    def reset_grid(self, new_grid):
        """
        Set new integration grid for the group.

        Parameters
        ----------
        new_grid: IntegralGrid.
        """
        self.clear(new_grid)
        for parent in self.parents:
            parent.reset_child_grid(self, new_grid)
        self.grid = new_grid

    def reset_child_grid(self, child, new_grid):
        """Sets new integration grid for given child of the group."""
        i = self.subgroups.index(child)
        self.grid.reset_grid(i, new_grid)
        self.clear()

    def resize(self, new_size):
        """If grid supports resizing, resizes it."""
        if hasattr(self.grid, 'resize'):
            self.grid.resize(new_size)
        self.clear()
        for parent in self.parents:
            parent.clear()

    def reset_distribution(self, distribution):
        """Sets new distribution for the group."""
        if hasattr(self.grid, 'distribution'):
            self.grid.distribution = distribution


def reapply_parametrizations(model, parametrized_modules, unsafe=True):
    """Function to reapply parameterizations to a model."""
    for name, params in parametrized_modules.items():
        module = dict(model.named_modules())[name]
        for p_name, parametrizations, orig_parameter in params:
            for parametrization in parametrizations:
                parametrize.register_parametrization(module, p_name, parametrization, unsafe=unsafe)
            getattr(module.parametrizations, p_name).original.data = orig_parameter


def remove_parametrizations(model):
    """Function to remove parameterizations from a model."""
    parametrized_modules = {}
    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations'):
            parametrized_modules[name] = []
            for p_name in list(module.parametrizations.keys()):
                orig_parameter = getattr(module.parametrizations, p_name)
                orig_parameter = orig_parameter.original.data.detach().clone()
                parametrized_modules[name].append((p_name, module.parametrizations[p_name], orig_parameter))
                parametrize.remove_parametrizations(module, p_name, True)
    return parametrized_modules


class ParametrizedModel(nn.Module):

    def __init__(self, model):
        super(ParametrizedModel, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        """ """
        self.forward_groups()
        return self.model(*args, **kwargs)

    def forward_groups(self):
        pass

    def get_unparametrized_model(self):
        """Samples weights, removes parameterizations and returns discrete model."""
        self.forward_groups()
        parametrized_modules = remove_parametrizations(self.model)
        unparametrized_model = copy.deepcopy(self.model)
        reapply_parametrizations(self.model, parametrized_modules, True)
        return unparametrized_model

    def __getattr__(self, item):
        if item in dir(self):
            out = super().__getattr__(item)
        else:
            out = getattr(self.model, item)
        return out

    def __getstate__(self):
        """
        Return the state of the module, removing the non-picklable parametrizations.
        """
        parametrized_modules = remove_parametrizations(self.model)
        state = self.state_dict()
        state['parametrized_modules'] = parametrized_modules
        return state

    def __setstate__(self, state):
        """Initialize the module from its state."""
        parametrized_modules = state.pop('parametrized_modules')
        super().__setstate__(state)
        reapply_parametrizations(self.model, parametrized_modules, True)


class PrunableModel(ParametrizedModel):

    def __init__(self, model, groups):
        super(PrunableModel, self).__init__(model)
        groups.sort(key=lambda g: g.count_parameters())
        self.groups = nn.ModuleList(groups)

    def forward_groups(self):
        for group in self.groups:
            group()


def get_attr_by_name(module, name):
    """ """
    for s in name.split('.'):
        module = getattr(module, s)
    return module


def get_parent_name(qualname: 'str') ->Tuple[str, str]:
    """
    Splits a ``qualname`` into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = qualname.rsplit('.', 1)
    return parent[0] if parent else '', name


def get_parent_module(module, attr_path):
    """
    Returns parent module of module.attr_path.

    Parameters
    ----------
    module: torch.nn.Module.
    attr_path: str.
    """
    parent_name, _ = get_parent_name(attr_path)
    if parent_name != '':
        parent = get_attr_by_name(module, parent_name)
    else:
        parent = module
    return parent


def reset_batchnorm(model):
    """
    Set new BatchNorm2d in place of fused batch norm layers.

    Parameters
    ----------
    model: torch.nn.Module.
    """
    fx_model = torch.fx.symbolic_trace(model)
    modules = dict(model.named_modules())
    for node in fx_model.graph.nodes:
        if node.op != 'call_module':
            continue
        if type(modules[node.target]) is nn.Identity:
            conv = modules[node.args[0].target]
            size = conv.weight.shape[0]
            bn = nn.BatchNorm2d(size)
            _, attr_name = get_parent_name(node.target)
            parent = get_parent_module(model, node.target)
            setattr(parent, attr_name, bn)


class IntegralModel(PrunableModel):
    """
    Contains original model with parametrized layers and RelatedGroups list.

    Parameters
    ----------
    model: torch.nn.Module.
        Model with parametrized layers.
    groups: List[RelatedGroup].
        List related groups.
    """

    def __init__(self, model, groups):
        super(IntegralModel, self).__init__(model, groups)
        self.original_size = None
        self.original_size = self.calculate_compression()

    def clear(self):
        """Clears cached tensors in all integral groups."""
        for group in self.groups:
            group.clear()

    def load_state_dict(self, state_dict: 'Mapping[str, Any]', strict: 'bool'=True):
        out = super().load_state_dict(state_dict, strict)
        self.clear()
        return out

    def calculate_compression(self):
        """
        Returns 1 - ratio of the size of the current
        model to the original size of the model.
        """
        self.forward_groups()
        for group in self.groups:
            group.clear()
        parametrized = remove_parametrizations(self.model)
        out = sum(p.numel() for p in self.model.parameters())
        reapply_parametrizations(self.model, parametrized, True)
        if self.original_size is not None:
            out = 1.0 - out / self.original_size
        return out

    def resize(self, sizes):
        """
        Resizes grids in each group.

        Parameters
        ----------
        sizes: List[int].
            List of new sizes.
        """
        for group, size in zip(self.groups, sizes):
            group.resize(size)

    def reset_grids(self, grids):
        for group, grid in zip(self.groups, grids):
            group.reset_grid(grid)

    def reset_distributions(self, distributions):
        """
        Sets new distributions in each RelatedGroup.grid.

        Parameters
        ----------
        distributions: List[torch_integral.grid.Distribution].
            List of new distributions.
        """
        for group, dist in zip(self.groups, distributions):
            group.reset_distribution(dist)

    def grids(self):
        """Returns list of grids of each integral group."""
        return [group.grid for group in self.groups]

    def grid_tuning(self, train_bn=False, train_bias=False, use_all_grids=False):
        """Turns on grid tuning mode for fast post-training pruning.
        Sets requires_grad = False for all parameters except TrainableGrid's parameters,
        biases and BatchNorm parameters (if corresponding flag is True).

        Parameters
        ----------
        train_bn: bool.
            Set True to train BatchNorm parameters.
        train_bias: bool.
            Set True to train biases.
        use_all_grids: bool.
            Set True to use all grids in each group.
        """
        if use_all_grids:
            for group in self.groups:
                if group.subgroups is None:
                    group.reset_grid(TrainableGrid1D(group.grid_size()))
        for name, param in self.named_parameters():
            parent = get_parent_module(self, name)
            if isinstance(parent, TrainableGrid1D):
                param.requires_grad = True
            else:
                param.requires_grad = False
        if train_bn:
            reset_batchnorm(self)
        if train_bias:
            for group in self.groups:
                for p in group.params:
                    if 'bias' in p['name']:
                        parent = get_parent_module(self.model, p['name'])
                        if parametrize.is_parametrized(parent, 'bias'):
                            parametrize.remove_parametrizations(parent, 'bias', True)
                        getattr(parent, 'bias').requires_grad = True


class IWeights(torch.nn.Module):
    """
    Class for weights parametrization. Can be registereg as parametrization
    with torch.nn.utils.parametrize.register_parametrization

    Parameters
    ----------
    weight_function: torch.nn.Module.
    grid: torch_integral.grid.IGrid.
    quadrature: torch_integral.quadrature.BaseIntegrationQuadrature.
    """

    def __init__(self, grid, quadrature):
        super().__init__()
        self.quadrature = quadrature
        self.grid = grid
        self.last_value = None
        self.train_volume = None

    def reset_quadrature(self, quadrature):
        """Replaces quadrature object."""
        weight = self.sample_weights(None)
        self.quadrature = quadrature
        self.right_inverse(weight)

    def clear(self):
        self.last_value = None

    def right_inverse(self, x):
        """Initialization method which is used when setattr of parametrized tensor called."""
        train_volume = 1.0
        if self.quadrature is not None:
            ones = torch.ones_like(x, device=x.device)
            q_coeffs = self.quadrature.multiply_coefficients(ones, self.grid())
            x = x / q_coeffs
            for dim in self.quadrature.integration_dims:
                train_volume *= x.shape[dim] - 1
            train_volume *= 0.5
            if self.train_volume is None:
                self.train_volume = train_volume
            x = x / self.train_volume
        x = self.init_values(x)
        return x

    def init_values(self, weight):
        """ """
        return weight

    def forward(self, w):
        """
        Performs forward pass. Samples new weights on grid
        if training or last sampled tensor is not cached.

        Parameters
        ----------
        w: torch.Tensor.
        """
        weight = self.sample_weights(w)
        return weight

    def sample_weights(self, w):
        """
        Evaluate pparametrization function on grid.

        Parameters
        ----------
        w: torch.Tensor.

        Returns
        -------
        torch.Tensor.
            Sampled weight function on grid.
        """
        x = self.grid()
        weight = self.evaluate_function(x, w)
        if self.quadrature is not None:
            weight = self.quadrature(weight, x) * self.train_volume
        return weight

    def evaluate_function(self, grid, weight):
        """ """
        raise NotImplementedError('Implement this method in derived class.')


class GridSampleWeightsBase(IWeights):
    """
    Base class for parametrization based on torch.nn.functional.grid_sample.

    Parameters
    ----------
    cont_size: List[int].
        Shape of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    interpolate_mode: str.
        Same modes as in torch.nn.functional.grid_sample.
    padding_mode: str.
    align_corners: bool.
    """

    def __init__(self, grid, quadrature, cont_size, discrete_shape=None, interpolate_mode='bicubic', padding_mode='border', align_corners=True):
        super(GridSampleWeightsBase, self).__init__(grid, quadrature)
        self.iterpolate_mode = interpolate_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.cont_size = cont_size
        self.discrete_shape = discrete_shape
        if discrete_shape is not None:
            self.planes_num = int(reduce(lambda a, b: a * b, discrete_shape))
        else:
            self.planes_num = 1

    def _postprocess_output(self, out):
        """ """
        raise NotImplementedError('Implement this method in derived class.')

    def evaluate_function(self, grid, weight):
        """
        Performs forward pass

        Parameters
        ----------
        grid: List[torch.Tensor].
            List of discretization grids along each dimension.

        Returns
        -------
        torch.Tensor.
            Sampled ``self.values`` on grid.

        """
        for i in range(len(grid)):
            grid[i] = grid[i]
        if len(grid) == 1:
            grid.append(torch.tensor(0.0, device=weight.device))
        grid = torch.stack(torch.meshgrid(grid[::-1], indexing='ij'), dim=-1)
        grid = grid.unsqueeze(0)
        out = F.grid_sample(weight, grid, mode=self.iterpolate_mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
        return self._postprocess_output(out)


class GridSampleWeights1D(GridSampleWeightsBase):
    """
    Class implementing InterpolationWeightsBase for parametrization
    of tensor with one continuous dimension.

    Parameters
    ----------
    cont_size: int.
        Size of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    cont_dim: int.
        Index of continuous dimension.
    interpolate_mode: str.
        See torch.nn.functional.grid_sample.
    padding_mode: str.
        See torch.nn.functional.grid_sample.
    align_corners: bool.
        See torch.nn.functional.grid_sample.
    """

    def __init__(self, grid, quadrature, cont_size, discrete_shape=None, cont_dim=0, interpolate_mode='bicubic', padding_mode='border', align_corners=True):
        super(GridSampleWeights1D, self).__init__(grid, quadrature, (cont_size, 1), discrete_shape, interpolate_mode, padding_mode, align_corners)
        self.cont_dim = cont_dim

    def init_values(self, x):
        """ """
        weight = x
        if x.ndim == 1:
            x = x[None, None, :, None]
        else:
            permutation = [i for i in range(x.ndim) if i != self.cont_dim]
            x = x.permute(*permutation, self.cont_dim)
            x = x.reshape(1, -1, x.shape[-1], 1)
        if x.shape[-2:] == self.cont_size:
            weight = x.contiguous()
        else:
            weight = F.interpolate(x, self.cont_size, mode=self.iterpolate_mode).contiguous()
        return weight

    def _postprocess_output(self, out):
        """ """
        discrete_shape = self.discrete_shape
        if discrete_shape is None:
            discrete_shape = []
        shape = out.shape[-1:]
        out = out.view(*discrete_shape, *shape)
        permutation = list(range(out.ndim))
        permutation[self.cont_dim] = out.ndim - 1
        j = 0
        for i in range(len(permutation)):
            if i != self.cont_dim:
                permutation[i] = j
                j += 1
        out = out.permute(*permutation).contiguous()
        return out


class GridSampleWeights2D(GridSampleWeightsBase):
    """
    Class implementing InterpolationWeightsBase for parametrization
    of tensor with two continuous dimensions.

    Parameters
    ----------
    cont_size: List[int].
        Shape of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    interpolate_mode: str.
        See torch.nn.functional.grid_sample.
    padding_mode: str.
        See torch.nn.functional.grid_sample.
    align_corners: bool.
        See torch.nn.functional.grid_sample.
    """

    def init_values(self, x):
        """ """
        weight = x
        if x.ndim == 2:
            x = x[None, None, :, :]
        else:
            permutation = list(range(2, x.ndim))
            shape = x.shape[:2]
            x = x.permute(*permutation, 0, 1)
            x = x.reshape(1, -1, *shape)
        if x.shape[-2:] == self.cont_size:
            weight = x.contiguous()
        else:
            weight = F.interpolate(x, self.cont_size, mode=self.iterpolate_mode).contiguous()
        return weight

    def _postprocess_output(self, out):
        discrete_shape = self.discrete_shape
        if discrete_shape is None:
            discrete_shape = []
        shape = out.shape[-2:]
        out = out.view(*discrete_shape, *shape)
        dims = range(out.ndim - 2)
        out = out.permute(out.ndim - 1, out.ndim - 2, *dims)
        return out.contiguous()


class InterpolationWeightsBase(IWeights):
    """
    Base class for parametrization based on torch.nn.functional.grid_sample.

    Parameters
    ----------
    cont_size: List[int].
        Shape of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    interpolate_mode: str.
        Same modes as in torch.nn.functional.grid_sample.
    padding_mode: str.
    align_corners: bool.
    """

    def __init__(self, grid, quadrature, cont_size, discrete_shape=None, interpolate_mode='bicubic', padding_mode='border', align_corners=True):
        super(InterpolationWeightsBase, self).__init__(grid, quadrature)
        self.iterpolate_mode = interpolate_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.cont_size = cont_size
        self.discrete_shape = discrete_shape
        if discrete_shape is not None:
            self.planes_num = int(reduce(lambda a, b: a * b, discrete_shape))
        else:
            self.planes_num = 1

    def _postprocess_output(self, out):
        """ """
        raise NotImplementedError('Implement this method in derived class.')

    def evaluate_function(self, grid, weight):
        """
        Performs forward pass

        Parameters
        ----------
        grid: List[torch.Tensor].
            List of discretization grids along each dimension.

        Returns
        -------
        torch.Tensor.
            Sampled ``self.values`` on grid.

        """
        shape = [g.shape[0] for g in grid]
        if len(shape) == 1:
            shape.append(1)
        out = F.interpolate(weight, size=shape, mode=self.iterpolate_mode)
        return self._postprocess_output(out)


class InterpolationWeights1D(InterpolationWeightsBase):
    """
    Class implementing InterpolationWeightsBase for parametrization
    of tensor with one continuous dimension.

    Parameters
    ----------
    cont_size: int.
        Size of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    cont_dim: int.
        Index of continuous dimension.
    interpolate_mode: str.
        See torch.nn.functional.grid_sample.
    padding_mode: str.
        See torch.nn.functional.grid_sample.
    align_corners: bool.
        See torch.nn.functional.grid_sample.
    """

    def __init__(self, grid, quadrature, cont_size, discrete_shape=None, cont_dim=0, interpolate_mode='bicubic', padding_mode='border', align_corners=True):
        super(InterpolationWeights1D, self).__init__(grid, quadrature, (cont_size, 1), discrete_shape, interpolate_mode, padding_mode, align_corners)
        self.cont_dim = cont_dim

    def init_values(self, x):
        """ """
        weight = x
        if x.ndim == 1:
            x = x[None, None, :, None]
        else:
            permutation = [i for i in range(x.ndim) if i != self.cont_dim]
            x = x.permute(*permutation, self.cont_dim)
            x = x.reshape(1, -1, x.shape[-1], 1)
        if x.shape[-2:] == self.cont_size:
            weight = x.contiguous()
        else:
            weight = F.interpolate(x, self.cont_size, mode=self.iterpolate_mode).contiguous()
        return weight

    def _postprocess_output(self, out):
        """ """
        discrete_shape = self.discrete_shape
        if discrete_shape is None:
            discrete_shape = []
        out = out.view(*discrete_shape, out.shape[-2])
        permutation = list(range(out.ndim))
        permutation[self.cont_dim] = out.ndim - 1
        j = 0
        for i in range(len(permutation)):
            if i != self.cont_dim:
                permutation[i] = j
                j += 1
        out = out.permute(*permutation).contiguous()
        return out


class InterpolationWeights2D(InterpolationWeightsBase):
    """
    Class implementing InterpolationWeightsBase for parametrization
    of tensor with two continuous dimensions.

    Parameters
    ----------
    cont_size: List[int].
        Shape of trainable parameter along continuous dimensions.
    discrete_shape: List[int].
        Sizes of parametrized tensor along discrete dimension.
    interpolate_mode: str.
        See torch.nn.functional.grid_sample.
    padding_mode: str.
        See torch.nn.functional.grid_sample.
    align_corners: bool.
        See torch.nn.functional.grid_sample.
    """

    def init_values(self, x):
        """ """
        weight = x
        if x.ndim == 2:
            x = x[None, None, :, :]
        else:
            permutation = list(range(2, x.ndim))
            shape = x.shape[:2]
            x = x.permute(*permutation, 0, 1)
            x = x.reshape(1, -1, *shape)
        if x.shape[-2:] == self.cont_size:
            weight = x.contiguous()
        else:
            weight = F.interpolate(x, self.cont_size, mode=self.iterpolate_mode).contiguous()
        return weight

    def _postprocess_output(self, out):
        discrete_shape = self.discrete_shape
        if discrete_shape is None:
            discrete_shape = []
        shape = out.shape[-2:]
        out = out.view(*discrete_shape, *shape)
        dims = range(out.ndim - 2)
        out = out.permute(out.ndim - 2, out.ndim - 1, *dims)
        return out.contiguous()


class BaseIntegrationQuadrature(torch.nn.Module):
    """
    Base quadrature class.

    Parameters
    ----------
    integration_dims: List[int].
        Numbers of dimensions along which we multiply by the quadrature weights
    grid_indices: List[int].
        Indices of corresponding grids.

    Attributes
    ----------
    integration_dims: List[int].
    grid_indices: List[int].
    """

    def __init__(self, integration_dims, grid_indices=None):
        super().__init__()
        self.integration_dims = integration_dims
        if grid_indices is None:
            self.grid_indices = integration_dims
        else:
            self.grid_indices = grid_indices
            assert len(grid_indices) == len(integration_dims)

    def multiply_coefficients(self, discretization, grid):
        """
        Multiply discretization tensor by quadrature weights along integration_dims.

        Parameters
        ----------
        discretization: torch.Tensor.
            Tensor to be multiplied by quadrature weights.
        grid: List[torch.Tensor].
            List of tensors with sampling points.

        Returns
        -------
        torch.Tensor.
            ``discretization`` multiplied by quadrature weights.
        """
        raise NotImplementedError('Implement this method in derived class.')

    def forward(self, function, grid):
        """
        Performs forward pass of the Module.

        Parameters
        ----------
        function: callable or torch.Tensor.
            Function to be integrated.
        grid: List[torch.Tensor].
            List of tensors with sampling points.

        Returns
        -------
        torch.Tensor.
            ``function`` discretized and multiplied by quadrature weights.
        """
        if callable(function):
            out = function(grid)
        else:
            out = function
        out = self.multiply_coefficients(out, grid)
        return out


class TrapezoidalQuadrature(BaseIntegrationQuadrature):
    """Class for integration with trapezoidal rule."""

    def multiply_coefficients(self, discretization, grid):
        """ """
        for i in range(len(self.integration_dims)):
            grid_i = self.grid_indices[i]
            dim = self.integration_dims[i]
            x = grid[grid_i]
            h = torch.zeros_like(x)
            h[1:-1] = x[2:] - x[:-2]
            h[0] = x[1] - x[0]
            h[-1] = x[-1] - x[-2]
            size = [1] * discretization.ndim
            size[dim] = h.size(0)
            h = h.view(size)
            discretization = discretization * (h * 0.5)
        return discretization


class RiemannQuadrature(BaseIntegrationQuadrature):
    """Rectangular integration rule."""

    def multiply_coefficients(self, discretization, grid):
        """ """
        for i in range(len(self.integration_dims)):
            grid_i = self.grid_indices[i]
            dim = self.integration_dims[i]
            x = grid[grid_i]
            h = x[1:] - x[:-1]
            first = (0.5 * h[0]).unsqueeze(0)
            last = (0.5 * h[-1]).unsqueeze(0)
            h = torch.cat([first, 0.5 * (h[:-1] + h[1:]), last])
            size = [1] * discretization.ndim
            size[dim] = h.size(0)
            h = h.view(size)
            discretization = discretization * h
        return discretization


class SimpsonQuadrature(BaseIntegrationQuadrature):
    """
    Integratioin of the function in propositioin
    that function is quadratic between sampling points.
    """

    def multiply_coefficients(self, discretization, grid):
        """ """
        for i in range(len(self.integration_dims)):
            grid_i = self.grid_indices[i]
            dim = self.integration_dims[i]
            x = grid[grid_i]
            step = x[1] - x[0]
            h = torch.ones_like(x)
            h[1::2] *= 4.0
            h[2:-1:2] *= 2.0
            h *= step / 3.0
            size = [1] * discretization.ndim
            size[dim] = h.size(0)
            h = h.view(size)
            discretization = discretization * h
        return discretization


class LegendreQuadrature(BaseIntegrationQuadrature):
    """ """

    def multiply_coefficients(self, discretization, grid):
        """ """
        for i in range(len(self.integration_dims)):
            grid_i = self.grid_indices[i]
            dim = self.integration_dims[i]
            x = grid[grid_i]
            _, weights = roots_legendre(x.shape[0])
            h = torch.tensor(weights, dtype=torch.float32, device=discretization.device)
            size = [1] * discretization.ndim
            size[dim] = h.size(0)
            h = h.view(size)
            discretization = discretization * h
        return discretization


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConstantGrid1D,
     lambda: ([], {'init_value': 4}),
     lambda: ([], {}),
     True),
    (MnistNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (ParametrizedModel,
     lambda: ([], {'model': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (RelatedGroup,
     lambda: ([], {'size': 4}),
     lambda: ([], {}),
     True),
    (TrainableDeltasGrid1D,
     lambda: ([], {'size': 4}),
     lambda: ([], {}),
     True),
    (TrainableGrid1D,
     lambda: ([], {'size': 4}),
     lambda: ([], {}),
     True),
]

class Test_TheStageAI_TorchIntegral(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

