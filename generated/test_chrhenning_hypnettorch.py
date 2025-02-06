import sys
_module = sys.modules[__name__]
del sys
conf = _module
hypnettorch = _module
data = _module
celeba_data = _module
cifar100_data = _module
cifar10_data = _module
cub_200_2011_data = _module
dataset = _module
fashion_mnist = _module
ilsvrc2012_data = _module
large_img_dataset = _module
mnist_data = _module
sequential_dataset = _module
special = _module
donuts = _module
gaussian_mixture_data = _module
gmm_data = _module
permuted_mnist = _module
regression1d_bimodal_data = _module
regression1d_data = _module
split_cifar = _module
split_mnist = _module
svhn_data = _module
timeseries = _module
audioset_data = _module
cognitive_tasks = _module
cognitive_data = _module
parameters = _module
stimulus = _module
copy_data = _module
mud_data = _module
permuted_copy = _module
preprocess_audioset = _module
preprocess_mud = _module
preprocess_smnist = _module
rnd_rec_teacher = _module
seq_smnist = _module
smnist_data = _module
split_audioset = _module
split_smnist = _module
udacity_ch2 = _module
run = _module
hnets = _module
chunked_deconv_hnet = _module
chunked_mlp_hnet = _module
deconv_hnet = _module
hnet_container = _module
hnet_helpers = _module
hnet_interface = _module
hnet_perturbation_wrapper = _module
mlp_hnet = _module
structured_hmlp_examples = _module
structured_mlp_hnet = _module
hpsearch = _module
gather_random_seeds = _module
hpsearch_config_template = _module
hpsearch_postprocessing = _module
mnets = _module
bi_rnn = _module
bio_conv_net = _module
chunk_squeezer = _module
classifier_interface = _module
lenet = _module
mlp = _module
mnet_interface = _module
resnet = _module
resnet_imgnet = _module
simple_rnn = _module
wide_resnet = _module
zenkenet = _module
utils = _module
batchnorm_layer = _module
cli_args = _module
context_mod_layer = _module
ewc_regularizer = _module
gan_helpers = _module
hmc = _module
hnet_regularizer = _module
init_utils = _module
local_conv2d_layer = _module
logger_config = _module
misc = _module
optim_step = _module
self_attention_layer = _module
si_regularizer = _module
sim_utils = _module
torch_ckpts = _module
torch_utils = _module
setup = _module
tests = _module

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


import time


import matplotlib.pyplot as plt


from warnings import warn


from abc import ABC


from abc import abstractmethod


from sklearn.preprocessing import OneHotEncoder


import matplotlib.gridspec as gridspec


import numpy.matlib as npm


from torchvision.datasets import FashionMNIST


import torchvision


import warnings


import torch


import torchvision.transforms as transforms


import torchvision.datasets as datasets


from scipy.io import loadmat


from copy import deepcopy


import copy


from torchvision.datasets import SVHN


from torch import from_numpy


from scipy.stats import ortho_group


import matplotlib.lines as lines


import pandas as pd


from sklearn.model_selection import train_test_split


from time import time


import torch.nn.functional as F


import torch.nn as nn


from collections import defaultdict


import math


from torch.nn import functional as F


import logging


from queue import Queue


from torch.distributions import Normal


from torch.distributions import MultivariateNormal


import inspect


import matplotlib


from torch import nn


from torch import optim


import random


import types


class BatchNormLayer(nn.Module):
    """Hypernetwork-compatible batch-normalization layer.

    Note, batch normalization performs the following operation

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\
            \\gamma + \\beta

    This class allows to deviate from this standard implementation in order to
    provide the flexibility required when using hypernetworks. Therefore, we
    slightly change the notation to

    .. math::

        y = \\frac{x - m_{\\text{stats}}^{(t)}}{\\sqrt{v_{\\text{stats}}^{(t)} + \\
                  \\epsilon}} * \\gamma^{(t)} + \\beta^{(t)}

    We use this notation to highlight that the running statistics
    :math:`m_{\\text{stats}}^{(t)}` and :math:`v_{\\text{stats}}^{(t)}` are not
    necessarily estimates resulting from mean and variance computation but might
    be learned parameters (e.g., the outputs of a hypernetwork).

    We additionally use the superscript :math:`(t)` to denote that the gain
    :math:`\\gamma`, offset :math:`\\beta` and statistics may be dynamically
    selected based on some external context information.

    This class provides the possibility to checkpoint statistics
    :math:`m_{\\text{stats}}^{(t)}` and :math:`v_{\\text{stats}}^{(t)}`, but
    **not** gains and offsets.

    .. note::
        If context-dependent gains :math:`\\gamma^{(t)}` and offsets
        :math:`\\beta^{(t)}` are required, then they have to be maintained
        externally, e.g., via a task-conditioned hypernetwork (see
        `this paper`_ for an example) and passed to the :meth:`forward` method.

        .. _this paper: https://arxiv.org/abs/1906.00695
    """

    def __init__(self, num_features, momentum=0.1, affine=True, track_running_stats=True, frozen_stats=False, learnable_stats=False):
        """
        Args:
            num_features: See argument ``num_features``, for instance, of class
                :class:`torch.nn.BatchNorm1d`.
            momentum: See argument ``momentum`` of class
                :class:`torch.nn.BatchNorm1d`.
            affine: See argument ``affine`` of class
                :class:`torch.nn.BatchNorm1d`. If set to :code:`False`, the
                input activity will simply be "whitened" according to the
                applied layer statistics (except if gain :math:`\\gamma` and
                offset :math:`\\beta` are passed to the :meth:`forward` method).

                Note, if ``learnable_stats`` is :code:`False`, then setting
                ``affine`` to :code:`False` results in no learnable weights for
                this layer (running stats might still be updated, but not via
                gradient descent).

                Note, even if this option is ``False``, one may still pass a
                gain :math:`\\gamma` and offset :math:`\\beta` to the
                :meth:`forward` method.
            track_running_stats: See argument ``track_running_stats`` of class
                :class:`torch.nn.BatchNorm1d`.
            frozen_stats: If ``True``, the layer statistics are frozen at their
                initial values of :math:`\\gamma = 1` and :math:`\\beta = 0`,
                i.e., layer activity will not be whitened.

                Note, this option requires ``track_running_stats`` to be set to
                ``False``.
            learnable_stats: If ``True``, the layer statistics are initialized
                as learnable parameters (:code:`requires_grad=True`).

                Note, these extra parameters will be maintained internally and
                not added to the :attr:`weights`. Statistics can always be
                maintained externally and passed to the :meth:`forward` method.

                Note, this option requires ``track_running_stats`` to be set to
                ``False``.
        """
        super(BatchNormLayer, self).__init__()
        if learnable_stats:
            raise NotImplementedError('Option "learnable_stats" has not been ' + 'implemented yet!')
        if momentum is None:
            raise NotImplementedError('This reimplementation of PyTorch its ' + 'batchnorm layer does not support ' + 'setting "momentum" to None.')
        if learnable_stats and track_running_stats:
            raise ValueError('Option "track_running_stats" must be set to ' + 'False when enabling "learnable_stats".')
        if frozen_stats and track_running_stats:
            raise ValueError('Option "track_running_stats" must be set to ' + 'False when enabling "frozen_stats".')
        self._num_features = num_features
        self._momentum = momentum
        self._affine = affine
        self._track_running_stats = track_running_stats
        self._frozen_stats = frozen_stats
        self._learnable_stats = learnable_stats
        self.register_buffer('_num_stats', torch.tensor(0, dtype=torch.long))
        self._weights = nn.ParameterList()
        self._param_shapes = [[num_features], [num_features]]
        if affine:
            self.register_parameter('scale', nn.Parameter(torch.Tensor(num_features), requires_grad=True))
            self.register_parameter('bias', nn.Parameter(torch.Tensor(num_features), requires_grad=True))
            self._weights.append(self.scale)
            self._weights.append(self.bias)
            nn.init.ones_(self.scale)
            nn.init.zeros_(self.bias)
        elif not learnable_stats:
            self._weights = None
        if learnable_stats:
            raise NotImplementedError()
        elif track_running_stats or frozen_stats:
            self.checkpoint_stats()
        else:
            mname, vname = self._stats_names(0)
            self.register_buffer(mname, None)
            self.register_buffer(vname, None)

    @property
    def weights(self):
        """A list of all internal weights of this layer. If all weights are
        assumed to be generated externally, then this attribute will be
        ``None``.

        :type: list or None
        """
        return self._weights

    @property
    def param_shapes(self):
        """A list of list of integers. Each list represents the shape of a
        parameter tensor.

        Note, this attribute is independent of the attribute :attr:`weights`,
        it always comprises the shapes of all weight tensors as if the network
        would be stand-alone (i.e., no weights being passed to the
        :meth:`forward` method).
        Note, unless ``learnable_stats`` is enabled, the layer statistics are
        not considered here.

        :type: list
        """
        return self._param_shapes

    @property
    def hyper_shapes(self):
        """A list of list of integers. Each list represents the shape of a
        weight tensor that can be passed to the :meth:`forward` method. If all
        weights are maintained internally, then this attribute will be ``None``.

        Specifically, this attribute is controlled by the argument ``affine``.
        If ``affine`` is ``True``, this attribute will be ``None``. Otherwise
        this attribute contains the shape of :math:`\\gamma` and :math:`\\beta`.

        :type: list or None
        """
        raise NotImplementedError('Not implemented yet!')
        return self._hyper_shapes

    @property
    def num_stats(self):
        """The number :math:`T` of internally managed statistics
        :math:`\\{(m_{\\text{stats}}^{(1)}, v_{\\text{stats}}^{(1)}), \\dots, \\
        (m_{\\text{stats}}^{(T)}, v_{\\text{stats}}^{(T)}) \\}`. This number is
        incremented everytime the method :meth:`checkpoint_stats` is called.

        :type: int
        """
        return self._num_stats

    def forward(self, inputs, running_mean=None, running_var=None, weight=None, bias=None, stats_id=None):
        """Apply batch normalization to given layer activations.

        Based on the state if this module (attribute :attr:`training`), the
        configuration of this layer and the parameters currently passed, the
        behavior of this function will be different.

        The core of this method still relies on the function
        :func:`torch.nn.functional.batch_norm`. In the following we list the
        different behaviors of this method based on the context.

        **In training mode:**

        We first consider the case that this module is in training mode, i.e.,
        :meth:`torch.nn.Module.train` has been called.

        Usually, during training, the running statistics are not used when
        computing the output, instead the statistics computed on the current
        batch are used (denoted by *use batch stats* in the table below).
        However, the batch statistics are typically updated during training
        (denoted by *update running stats* in the table below).

        The above described scenario would correspond to passing batch
        statistics to the function :func:`torch.nn.functional.batch_norm` and
        setting the parameter ``training`` to ``True``.

        +----------------------+---------------------+-------------------------+
        | **training mode**    | **use batch stats** | **update running stats**|
        +----------------------+---------------------+-------------------------+
        | given stats          | Yes                 | Yes                     |
        +----------------------+---------------------+-------------------------+
        | track running stats  | Yes                 | Yes                     |
        +----------------------+---------------------+-------------------------+
        | frozen stats         | No                  | No                      |
        +----------------------+---------------------+-------------------------+
        | learnable stats      | Yes                 | Yes [1]_                |
        +----------------------+---------------------+-------------------------+
        |no track running stats| Yes                 | No                      |
        +----------------------+---------------------+-------------------------+

        The meaning of each row in this table is as follows:

            - **given stats**: External stats are provided via the parameters
              ``running_mean`` and ``running_var``.
            - **track running stats**: If ``track_running_stats`` was set to
              ``True`` in the constructor and no stats were given.
            - **frozen stats**: If ``frozen_stats`` was set to ``True`` in the
              constructor and no stats were given.
            - **learnable stats**: If ``learnable_stats`` was set to ``True`` in
              the constructor and no stats were given.
            - **no track running stats**: If none of the above options apply,
              then the statistics will always be computed from the current batch
              (also in eval mode).

        .. note::
            If provided, running stats specified via ``running_mean`` and
            ``running_var`` always have priority.

        .. [1] We use a custom implementation to update the running statistics,
           that is compatible with backpropagation.

        **In evaluation mode:**

        We now consider the case that this module is in evaluation mode, i.e.,
        :meth:`torch.nn.Module.eval` has been called.

        Here is the same table as above just for the evaluation mode.

        +----------------------+---------------------+-------------------------+
        | **evaluation mode**  | **use batch stats** | **update running stats**|
        +----------------------+---------------------+-------------------------+
        | track running stats  | No                  | No                      |
        +----------------------+---------------------+-------------------------+
        | frozen stats         | No                  | No                      |
        +----------------------+---------------------+-------------------------+
        | learnable stats      | No                  | No                      |
        +----------------------+---------------------+-------------------------+
        | given stats          | No                  | No                      |
        +----------------------+---------------------+-------------------------+
        |no track running stats| Yes                 | No                      |
        +----------------------+---------------------+-------------------------+

        Args:
            inputs: The inputs to the batchnorm layer.
            running_mean (optional): Running mean stats
                :math:`m_{\\text{stats}}`. This option has priority, i.e., any
                internally maintained statistics are ignored if given.

                .. note::
                    If specified, then ``running_var`` also has to be specified.
            running_var (optional): Similar to option ``running_mean``, but for
                the running variance stats :math:`v_{\\text{stats}}`

                .. note::
                    If specified, then ``running_mean`` also has to be
                    specified.
            weight (optional): The gain factors :math:`\\gamma`. If given, any
                internal gains are ignored. If option ``affine`` was set to
                ``False`` in the constructor and this option remains ``None``,
                then no gains are multiplied to the "whitened" inputs.
            bias (optional): The behavior of this option is similar to option
                ``weight``, except that this option represents the offsets
                :math:`\\beta`.
            stats_id: This argument is optional except if multiple running
                stats checkpoints exist (i.e., attribute :attr:`num_stats` is
                greater than 1) and no running stats have been provided to this
                method.

                .. note::
                    This argument is ignored if running stats have been passed.

        Returns:
            The layer activation ``inputs`` after batch-norm has been applied.
        """
        assert running_mean is None and running_var is None or running_mean is not None and running_var is not None
        if not self._affine:
            if weight is None or bias is None:
                raise ValueError('Layer was generated in non-affine mode. ' + 'Therefore, arguments "weight" and "bias" ' + 'may not be None.')
        if weight is None and self._affine:
            weight = self.scale
        if bias is None and self._affine:
            bias = self.bias
        stats_given = running_mean is not None
        if running_mean is None or running_var is None:
            if stats_id is None and self.num_stats > 1:
                raise ValueError('Parameter "stats_id" is not defined but ' + 'multiple running stats are available.')
            elif self._track_running_stats:
                if stats_id is None:
                    stats_id = 0
                assert stats_id < self.num_stats
                rm, rv = self.get_stats(stats_id)
                if running_mean is None:
                    running_mean = rm
                if running_var is None:
                    running_var = rv
        elif stats_id is not None:
            warn('Parameter "stats_id" is ignored since running stats have ' + 'been provided.')
        momentum = self._momentum
        if stats_given or self._track_running_stats:
            return F.batch_norm(inputs, running_mean, running_var, weight=weight, bias=bias, training=self.training, momentum=momentum)
        if self._learnable_stats:
            raise NotImplementedError()
        if self._frozen_stats:
            return F.batch_norm(inputs, running_mean, running_var, weight=weight, bias=bias, training=False)
        else:
            assert not self._track_running_stats
            return F.batch_norm(inputs, None, None, weight=weight, bias=bias, training=True, momentum=momentum)

    def checkpoint_stats(self, device=None):
        """Buffers for a new set of running stats will be registered.

        Calling this function will also increment the attribute
        :attr:`num_stats`.

        Args:
            device (optional): If not provided, the newly created statistics
                will either be moved to the device of the most recent statistics
                or to CPU if no prior statistics exist.
        """
        assert self._track_running_stats or self._frozen_stats and self._num_stats == 0
        if device is None:
            if self.num_stats > 0:
                mname_old, _ = self._stats_names(self._num_stats - 1)
                device = getattr(self, mname_old).device
        if self._learnable_stats:
            raise NotImplementedError()
        mname, vname = self._stats_names(self._num_stats)
        self._num_stats += 1
        self.register_buffer(mname, torch.zeros(self._num_features, device=device))
        self.register_buffer(vname, torch.ones(self._num_features, device=device))

    def get_stats(self, stats_id=None):
        """Get a set of running statistics (means and variances).

        Args:
            stats_id (optional): ID of stats. If not provided, the most recent
                stats are returned.

        Returns:
            (tuple): Tuple containing:

            - **running_mean**
            - **running_var**
        """
        if stats_id is None:
            stats_id = self.num_stats - 1
        assert stats_id < self.num_stats
        mname, vname = self._stats_names(stats_id)
        running_mean = getattr(self, mname)
        running_var = getattr(self, vname)
        return running_mean, running_var

    def _stats_names(self, stats_id):
        """Get the buffer names for mean and variance statistics depending on
        the ``stats_id``, i.e., the ID of the stats checkpoint.

        Args:
            stats_id: ID of stats.

        Returns:
            (tuple): Tuple containing:

            - **mean_name**
            - **var_name**
        """
        mean_name = 'mean_%d' % stats_id
        var_name = 'var_%d' % stats_id
        return mean_name, var_name


class ContextModLayer(nn.Module):
    """Implementation of a layer that can apply context-dependent modulation on
    the level of neuronal computation.

    The layer consists of two parameter vectors: gains :math:`\\mathbf{g}`
    and shifts :math:`\\mathbf{s}`, whereas gains represent a multiplicative
    modulation of input activations and shifts an additive modulation,
    respectively.

    Note, the weight vectors :math:`\\mathbf{g}` and :math:`\\mathbf{s}` might
    also be passed to the :meth:`forward` method, where one may pass a separate
    set of parameters for each sample in the input batch.

    Example:
        Assume that a :class:`ContextModLayer` is applied between a linear
        (fully-connected) layer
        :math:`\\mathbf{y} \\equiv W \\mathbf{x} + \\mathbf{b}` with input
        :math:`\\mathbf{x}` and a nonlinear activation function
        :math:`z \\equiv \\sigma(y)`.

        The layer-computation in such a case will become

        .. math::

            \\sigma \\big( (W \\mathbf{x} + \\mathbf{b}) \\odot \\mathbf{g} + \\
            \\mathbf{s} \\big)

    Args:
        num_features (int or tuple): Number of units in the layer (size of
            parameter vectors :math:`\\mathbf{g}` and :math:`\\mathbf{s}`).

            In case a ``tuple`` of integers is provided, the gain
            :math:`\\mathbf{g}` and shift :math:`\\mathbf{s}` parameters will
            become multidimensional tensors with the shape being prescribed
            by ``num_features``. Please note the `broadcasting rules`_ as
            :math:`\\mathbf{g}` and :math:`\\mathbf{s}` are simply multiplied
            or added to the input.

            Example:
                Consider the output of a convolutional layer with output shape
                ``[B,C,W,H]``. In case there should be a scalar gain and shift
                per feature map, ``num_features`` could be ``[C,1,1]`` or
                ``[1,C,1,1]`` (one might also pass a shape ``[B,C,1,1]`` to the
                :meth:`forward` method to apply separate shifts and gains per
                sample in the batch).

                Alternatively, one might want to provide shift and gain per
                output unit, i.e., ``num_features`` should be ``[C,W,H]``. Note,
                that due to weight sharing, all output activities within a
                feature map are computed using the same weights, which is why it
                is common practice to share shifts and gains within a feature
                map (e.g., in Spatial Batch-Normalization).
        no_weights (bool): If ``True``, the layer will have no trainable weights
            (:math:`\\mathbf{g}` and :math:`\\mathbf{s}`). Hence, weights are
            expected to be passed to the :meth:`forward` method.
        no_gains (bool): If ``True``, no gain parameters :math:`\\mathbf{g}` will
            be modulating the input activity.

            .. note::
                Arguments ``no_gains`` and ``no_shifts`` might not be activated
                simultaneously!
        no_shifts (bool): If ``True``, no shift parameters :math:`\\mathbf{s}`
            will be modulating the input activity.
        apply_gain_offset (bool, optional): If activated, this option will apply
            a constant offset of 1 to all gains, i.e., the computation becomes

            .. math::

                \\sigma \\big( (W \\mathbf{x} + \\mathbf{b}) \\odot \\
                (1 + \\mathbf{g}) + \\mathbf{s} \\big)

            When could that be useful? In case the gains and shifts are
            generated by the same hypernetwork, a meaningful initialization
            might be difficult to achieve (e.g., such that gains are close to 1
            and shifts are close to 0 at the beginning). Therefore, one might
            initialize the hypernetwork such that all outputs are close to zero
            at the beginning and the constant shift ensures that meaningful
            gains are applied.
        apply_gain_softplus (bool, optional): If activated, this option will
            enforce poitive gain modulation by sending the gain weights
            :math:`\\mathbf{g}` through a softplus function (scaled by :math:`s`,
            see ``softplus_scale``).

            .. math::

                \\mathbf{g} = \\frac{1}{s} \\log(1+\\exp(\\mathbf{g} \\cdot s))
        softplus_scale (float): If option ``apply_gain_softplus`` is ``True``,
            then this will determine the sclae of the softplus function.

    .. _broadcasting rules:
        https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-\\
        semantics
    """

    def __init__(self, num_features, no_weights=False, no_gains=False, no_shifts=False, apply_gain_offset=False, apply_gain_softplus=False, softplus_scale=1.0):
        super(ContextModLayer, self).__init__()
        assert isinstance(num_features, (int, list, tuple))
        if not isinstance(num_features, int):
            for nf in num_features:
                assert isinstance(nf, int)
        else:
            num_features = [num_features]
        assert not no_gains or not no_shifts
        self._num_features = num_features
        self._no_weights = no_weights
        self._no_gains = no_gains
        self._no_shifts = no_shifts
        self._apply_gain_offset = apply_gain_offset
        self._apply_gain_softplus = apply_gain_softplus
        self._sps = softplus_scale
        if apply_gain_offset and apply_gain_softplus:
            raise ValueError('Options "apply_gain_offset" and ' + '"apply_gain_softplus" are not compatible.')
        self._weights = None
        self._param_shapes = [num_features] * (1 if no_gains or no_shifts else 2)
        self._param_shapes_meta = ([] if no_gains else ['gain']) + ([] if no_shifts else ['shift'])
        self.register_buffer('_num_ckpts', torch.tensor(0, dtype=torch.long))
        if not no_weights:
            self._weights = nn.ParameterList()
            if not no_gains:
                self.register_parameter('gain', nn.Parameter(torch.Tensor(*num_features), requires_grad=True))
                self._weights.append(self.gain)
                if apply_gain_offset:
                    nn.init.zeros_(self.gain)
                else:
                    nn.init.ones_(self.gain)
            else:
                self.register_parameter('gain', None)
            if not no_shifts:
                self.register_parameter('shift', nn.Parameter(torch.Tensor(*num_features), requires_grad=True))
                self._weights.append(self.shift)
                nn.init.zeros_(self.shift)
            else:
                self.register_parameter('shift', None)

    @property
    def weights(self):
        """A list of all internal weights of this layer.

        If all weights are assumed to be generated externally, then this
        attribute will be ``None``.

        :type: torch.nn.ParameterList or None
        """
        return self._weights

    @property
    def param_shapes(self):
        """A list of list of integers. Each list represents the shape of a
        parameter tensor. Note, this attribute is independent of the attribute
        :attr:`weights`, it always comprises the shapes of all weight tensors as
        if the network would be stand- alone (i.e., no weights being passed to
        the :meth:`forward` method).

        .. note::
            The weights passed to the :meth:`forward` method might deviate
            from these shapes, as we allow passing a distinct set of
            parameters per sample in the input batch.

        :type: list
        """
        return self._param_shapes

    @property
    def param_shapes_meta(self):
        """List of strings. Each entry represents the meaning of the
        corresponding entry in :attr:`param_shapes`. The following keywords are
        possible:

        - ``'gain'``: The corresponding shape in :attr:`param_shapes`
          denotes the gain :math:`\\mathbf{g}` parameter.
        - ``'shift'``: The corresponding shape in :attr:`param_shapes`
          denotes the shift :math:`\\mathbf{s}` parameter.

        :type: list
        """
        return self._param_shapes_meta

    @property
    def num_ckpts(self):
        """The number of existing weight checkpoints (i.e., how often the method
        :meth:`checkpoint_weights` was called).

        :type: int
        """
        return self._num_ckpts

    @property
    def gain_offset_applied(self):
        """Whether constructor argument ``apply_gain_offset`` was activated.

        Thus, whether an offset for the gain :math:`\\mathbf{g}` is applied.

        :type: bool
        """
        return self._apply_gain_offset

    @property
    def gain_softplus_applied(self):
        """Whether constructor argument ``apply_gain_softplus`` was activated.

        Thus, whether a softplus function for the gain :math:`\\mathbf{g}` is
        applied.

        :type: bool
        """
        return self._apply_gain_softplus

    @property
    def has_gains(self):
        """Is ``True`` if ``no_gains`` was not set in the constructor.

        Thus, whether gains :math:`\\mathbf{g}` are part of the computation of
        this layer.

        :type: bool
        """
        return not self._no_gains

    @property
    def has_shifts(self):
        """Is ``True`` if ``no_shifts`` was not set in the constructor.

        Thus, whether shifts :math:`\\mathbf{s}` are part of the computation of
        this layer.

        :type: bool
        """
        return not self._no_shifts

    def forward(self, x, weights=None, ckpt_id=None, bs_dim=0):
        """Apply context-dependent gain modulation.

        Computes :math:`\\mathbf{x} \\odot \\mathbf{g} + \\mathbf{s}`, where
        :math:`\\mathbf{x}` denotes the input activity ``x``.

        Args:
            x: The input activity.
            weights: Weights that should be used instead of the internally
                maintained once (determined by attribute :attr:`weights`). Note,
                if ``no_weights`` was ``True`` in the constructor, then this
                parameter is mandatory.

                Usually, the shape of the passed weights should follow the
                attribute :attr:`param_shapes`, which is a tuple of shapes
                ``[[num_features], [num_features]]`` (at least for linear
                layers, see docstring of argument ``num_features`` in the
                constructor for more details). However, one may also
                specify a seperate set of context-mod parameters per input
                sample. Assume ``x`` has shape ``[num_samples, num_features]``.
                Then ``weights`` may have the shape
                ``[[num_samples, num_features], [num_samples, num_features]]``.
            ckpt_id (int): This argument can be set in case a checkpointed set
                of weights should be used to compute the forward pass (see
                method :meth:`checkpoint_weights`).

                .. note::
                    This argument is ignored if ``weights`` is not ``None``.
            bs_dim (int): Batch size dimension in input tensor ``x``.

        Returns:
            The modulated input activity.
        """
        if self._no_weights and weights is None:
            raise ValueError('Layer was generated without weights. ' + 'Hence, "weights" option may not be None.')
        if weights is not None and ckpt_id is not None:
            warn('Context-mod layer received weights as well as the request ' + 'to load checkpointed weights. The request to load ' + 'checkpointed weights will be ignored.')
        batch_size = x.shape[bs_dim]
        if weights is None:
            gain, shift = self.get_weights(ckpt_id=ckpt_id)
            if self._no_gains:
                weights = [shift]
            elif self._no_shifts:
                weights = [gain]
            else:
                weights = [gain, shift]
        else:
            assert len(weights) in [1, 2]
            nfl = len(self._num_features)
            nb = len(x.shape)
            for p in weights:
                assert len(p.shape) in [nfl, nb]
                if len(p.shape) == nfl:
                    assert np.all(np.equal(p.shape, self._num_features))
                else:
                    assert p.shape[0] == batch_size and np.all(np.equal(p.shape[1:], self._num_features))
        gain = None
        shift = None
        if self._no_gains:
            assert len(weights) == 1
            shift = weights[0]
        elif self._no_shifts:
            assert len(weights) == 1
            gain = weights[0]
        else:
            assert len(weights) == 2
            gain = weights[0]
            shift = weights[1]
        if gain is not None:
            x = x.mul(self.preprocess_gain(gain))
        if shift is not None:
            x = x.add(shift)
        return x

    def preprocess_gain(self, gain):
        """Obtains gains :math:`\\mathbf{g}` used for mudulation.
        
        Depending on the user configuration, gains might be preprocessed before
        applied for context-modulation (e.g., see attributes
        :attr:`gain_offset_applied` or :attr:`gain_softplus_applied`). This
        method transforms raw gains such that they can be applied to the network
        activation.

        Note:
            This method is called by the :meth:`forward` to transform given
            gains.

        Args:
            gain (torch.Tensor): A gain tensor.

        Returns:
            (torch.Tensor): The transformed gains.
        """
        if self._apply_gain_softplus:
            gain = 1.0 / self._sps * F.softplus(gain * self._sps)
        elif self._apply_gain_offset:
            gain = gain + 1.0
        return gain

    def checkpoint_weights(self, device=None, no_reinit=False):
        """Checkpoint and reinit the current weights.

        Buffers for a new checkpoint will be registered and the current weights
        will be copied into them. Additionally, the current weights will be
        reinitialized (gains to 1 and shifts to 0).

        Calling this function will also increment the attribute
        :attr:`num_ckpts`.

        Note:
            This method uses the method :meth:`torch.nn.Module.register_buffer`
            rather than the method :meth:`torch.nn.Module.register_parameter` to
            create checkpoints. The reason is, that we don't want the
            checkpoints to appear as trainable weights (when calling
            :meth:`torch.nn.Module.parameters`). However, that means that
            training on checkpointed weights cannot be continued unless they are
            copied back into an actual :class:`torch.nn.Parameter` object.

        Args:
            device (optional): If not provided, the newly created checkpoint
                will be moved to the device of the current weights.
            no_reinit (bool): If ``True``, the actual :attr:`weights` will not
                be reinitialized.
        """
        assert not self._no_weights
        if device is None:
            if self.gain is not None:
                device = self.gain.device
            else:
                device = self.shift.device
        gname, sname = self._weight_names(self._num_ckpts)
        self._num_ckpts += 1
        if not self._no_gains:
            self.register_buffer(gname, torch.empty_like(self.gain, device=device))
            getattr(self, gname).data = self.gain.detach().clone()
            if not no_reinit:
                if self._apply_gain_offset:
                    nn.init.zeros_(self.gain)
                else:
                    nn.init.ones_(self.gain)
        else:
            self.register_buffer(gname, None)
        if not self._no_shifts:
            self.register_buffer(sname, torch.empty_like(self.shift, device=device))
            getattr(self, sname).data = self.shift.detach().clone()
            if not no_reinit:
                nn.init.zeros_(self.shift)
        else:
            self.register_buffer(gname, None)

    def get_weights(self, ckpt_id=None):
        """Get the current (or a set of checkpointed) weights of this context-
        mod layer.

        Args:
            ckpt_id (optional): ID of checkpoint. If not provided, the current
                set of weights is returned.
                If :code:`ckpt_id == self.num_ckpts`, then this method also
                returns the current weights, as the checkpoint has not been
                created yet.

        Returns:
            (tuple): Tuple containing:

            - **gain**: Is ``None`` if layer has no gains.
            - **shift**: Is ``None`` if layer has no shifts.
        """
        if ckpt_id is None or ckpt_id == self.num_ckpts:
            return self.gain, self.shift
        assert ckpt_id >= 0 and ckpt_id < self.num_ckpts
        gname, sname = self._weight_names(ckpt_id)
        gain = getattr(self, gname)
        shift = getattr(self, sname)
        return gain, shift

    def _weight_names(self, ckpt_id):
        """Get the buffer names for checkpointed gain and shift weights
        depending on the ``ckpt_id``, i.e., the ID of the checkpoint.

        Args:
            ckpt_id: ID of weight checkpoint.

        Returns:
            (tuple): Tuple containing:

            - **gain_name**
            - **shift_name**
        """
        gain_name = 'gain_ckpt_%d' % ckpt_id
        shift_name = 'shift_ckpt_%d' % ckpt_id
        return gain_name, shift_name

    def normal_init(self, std=1.0):
        """Reinitialize internal weights using a normal distribution.

        Args:
            std (float): Standard deviation of init.
        """
        if self._no_weights:
            raise ValueError('Method is not applicable to layers without ' + 'internally maintained weights.')
        if not self._no_gains:
            if self._apply_gain_offset:
                nn.init.normal_(self.gain, std=std)
            else:
                nn.init.normal_(self.gain, mean=1.0, std=std)
        if not self._no_shifts:
            nn.init.normal_(self.shift, std=std)

    def uniform_init(self, width=1.0):
        """Reinitialize internal weights using a uniform distribution.

        Args:
            width (float): The range of the uniform init will be determined
                as ``[mean-width, mean+width]``, where ``mean`` is 0 for shifts
                and 1 for gains.
        """
        if self._no_weights:
            raise ValueError('Method is not applicable to layers without ' + 'internally maintained weights.')
        if not self._no_gains:
            if self._apply_gain_offset:
                nn.init.uniform_(self.gain, a=-width, b=width)
            else:
                nn.init.uniform_(self.gain, a=1.0 - width, b=1.0 + width)
        if not self._no_shifts:
            nn.init.uniform_(self.shift, a=-width, b=width)

    def sparse_init(self, sparsity=0.8):
        """Reinitialize internal weights sparsely.

        Gains will be initialized such that ``sparisity * 100`` percent of them
        will be 0, the remaining ones will be 1. Shifts are initialized to 0.

        Args:
            sparsity (float): A number between 0 and 1 determining the
                spasity level of gains.
        """
        if self._no_weights:
            raise ValueError('Method is not applicable to layers without ' + 'internally maintained weights.')
        assert 0 <= sparsity <= 1
        if not self._no_gains:
            num_zeros = int(self.gain.numel() * sparsity)
            inds = np.zeros(self.gain.numel(), dtype=bool)
            inds = inds.reshape(-1)
            inds[:num_zeros] = True
            np.random.shuffle(inds)
            inds = inds.reshape(*self.gain.shape)
            inds = torch.from_numpy(inds)
            if self._apply_gain_offset:
                nn.init.zeros_(self.gain)
                self.gain.data[inds] = -1.0
            else:
                nn.init.ones_(self.gain)
                self.gain.data[inds] = 0.0
        if not self._no_shifts:
            nn.init.zeros_(self.shift)

