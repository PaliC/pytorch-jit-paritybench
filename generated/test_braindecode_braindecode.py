import sys
_module = sys.modules[__name__]
del sys
braindecode = _module
augmentation = _module
base = _module
functional = _module
transforms = _module
classifier = _module
datasets = _module
base = _module
bbci = _module
bcicomp = _module
mne = _module
moabb = _module
nmt = _module
sleep_physio_challe_18 = _module
sleep_physionet = _module
tuh = _module
xy = _module
datautil = _module
serialization = _module
util = _module
eegneuralnet = _module
models = _module
atcnet = _module
attentionbasenet = _module
base = _module
biot = _module
contrawr = _module
ctnet = _module
deep4 = _module
deepsleepnet = _module
eegconformer = _module
eeginception_erp = _module
eeginception_mi = _module
eegitnet = _module
eegminer = _module
eegnet = _module
eegnex = _module
eegresnet = _module
eegsimpleconv = _module
eegtcnet = _module
functions = _module
hybrid = _module
labram = _module
modules = _module
modules_attention = _module
msvtnet = _module
sccnet = _module
shallow_fbcsp = _module
sinc_shallow = _module
sleep_stager_blanco_2020 = _module
sleep_stager_chambon_2018 = _module
sleep_stager_eldele_2021 = _module
sparcnet = _module
syncnet = _module
tcn = _module
tidnet = _module
tsinception = _module
usleep = _module
util = _module
preprocessing = _module
mne_preprocess = _module
preprocess = _module
windowers = _module
regressor = _module
samplers = _module
base = _module
ssl = _module
training = _module
callbacks = _module
losses = _module
scoring = _module
util = _module
version = _module
visualization = _module
confusion_matrices = _module
gradients = _module
conf = _module
gh_substitutions = _module
plot_bcic_iv_4_ecog_cropped = _module
plot_data_augmentation = _module
plot_data_augmentation_search = _module
plot_relative_positioning = _module
plot_bcic_iv_4_ecog_trial = _module
plot_sleep_staging_chambon2018 = _module
plot_sleep_staging_eldele2021 = _module
plot_sleep_staging_usleep = _module
plot_tuh_eeg_corpus = _module
benchmark_lazy_eager_loading = _module
plot_benchmark_preprocessing = _module
plot_custom_dataset_example = _module
plot_load_save_datasets = _module
plot_mne_dataset_example = _module
plot_moabb_dataset_example = _module
plot_split_dataset = _module
plot_tuh_discrete_multitarget = _module
plot_basic_training_epochs = _module
plot_bcic_iv_2a_moabb_cropped = _module
plot_bcic_iv_2a_moabb_trial = _module
plot_how_train_test_and_tune = _module
plot_regression = _module
plot_train_in_pure_pytorch_and_pytorch_lightning = _module
test = _module
acceptance_tests = _module
test_cropped_decoding = _module
test_eeg_classifier = _module
test_model_architectures = _module
test_trialwise_decoding = _module
test_variable_length_trials_decoding = _module
dataset = _module
unit_tests = _module
conftest = _module
test_base = _module
test_functional = _module
test_transforms = _module
test_bbci = _module
test_bcicomp = _module
test_dataset = _module
test_mne = _module
test_moabb = _module
test_nmt = _module
test_sleep_physionet = _module
test_tuh = _module
test_xy = _module
test_init = _module
test_serialization = _module
test_util = _module
test_base = _module
test_correctness = _module
test_functional = _module
test_integration = _module
test_models = _module
test_modules = _module
test_util = _module
test_mne_preprocessor = _module
test_preprocess = _module
test_windowers = _module
test_samplers = _module
test_eegneuralnet = _module
test_util = _module
test_losses = _module
test_scoring = _module
test_gradients = _module

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


from typing import Tuple


from typing import Any


from typing import Optional


from typing import Union


from typing import Callable


from numbers import Real


from sklearn.utils import check_random_state


import torch


from torch import Tensor


from torch import nn


from torch.utils.data import DataLoader


from torch.utils.data._utils.collate import default_collate


import numpy as np


from scipy.interpolate import Rbf


from torch.fft import fft


from torch.fft import ifft


from torch.nn.functional import pad


from torch.nn.functional import one_hot


import warnings


from torch.nn import CrossEntropyLoss


from collections.abc import Callable


from typing import Iterable


from typing import no_type_check


import pandas as pd


from torch.utils.data import Dataset


from torch.utils.data import ConcatDataset


import abc


import logging


import inspect


from sklearn.metrics import get_scorer


from typing import Dict


from collections import OrderedDict


import math


from warnings import warn


import torch.nn as nn


from torch.nn import init


import torch.nn.functional as F


from numpy import prod


from functools import partial


from torch.fft import fftfreq


from torchaudio.transforms import Resample


from torch.nn import ConstantPad2d


from torch.nn.init import trunc_normal_


from torch import from_numpy


from torchaudio.functional import fftconvolve


from torchaudio.functional import filtfilt


from typing import Type


import copy


from copy import deepcopy


from math import floor


from math import log2


from numpy import ceil


from numpy import arange


from torch.nn.utils import weight_norm


from math import ceil


from scipy.special import log_softmax


from sklearn.utils import deprecated


from torch.utils.data.sampler import Sampler


import random


import matplotlib


import sklearn


import matplotlib.pyplot as plt


from matplotlib.lines import Line2D


from numpy import multiply


from numpy import array


from numpy import linspace


from sklearn.model_selection import KFold


from sklearn.model_selection import GridSearchCV


from sklearn.preprocessing import scale as standard_scale


from sklearn.model_selection import train_test_split


from sklearn.metrics import confusion_matrix


from sklearn.metrics import classification_report


from sklearn.metrics import balanced_accuracy_score


from sklearn.linear_model import LogisticRegression


from sklearn.preprocessing import StandardScaler


from sklearn.pipeline import make_pipeline


from sklearn.decomposition import PCA


from matplotlib import colormaps


from numbers import Integral


from sklearn.utils import compute_class_weight


from sklearn.preprocessing import robust_scale


from itertools import product


import time


from torch import optim


from matplotlib.patches import Patch


from torch.utils.data import Subset


from sklearn.model_selection import cross_val_score


from torch.nn import Module


from torch.optim.lr_scheduler import LRScheduler


from torch.nn.functional import nll_loss


from scipy.fft import fft


from scipy.fft import fftfreq


from scipy.fft import fftshift


from scipy.signal import find_peaks


from scipy.signal import welch


from scipy.signal import hilbert


from scipy.signal import fftconvolve as fftconvolve_scipy


from scipy.signal import freqz


from scipy.signal import lfilter as lfilter_scipy


from sklearn.preprocessing import OneHotEncoder


from scipy.special import softmax


from sklearn.base import clone


from numpy.testing import assert_array_equal


from numpy.testing import assert_allclose


import sklearn.datasets


from sklearn.metrics import accuracy_score


from sklearn.metrics import f1_score


Output = Union[torch.Tensor, Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]]


class Transform(torch.nn.Module):
    """Basic transform class used for implementing data augmentation
    operations.

    Parameters
    ----------
    operation : callable
        A function taking arrays X, y (inputs and targets resp.) and
        other required arguments, and returning the transformed X and y.
    probability : float, optional
        Float between 0 and 1 defining the uniform probability of applying the
        operation. Set to 1.0 by default (e.g always apply the operation).
    random_state: int, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """
    operation: 'Operation'

    def __init__(self, probability=1.0, random_state=None):
        super().__init__()
        if self.forward.__func__ is Transform.forward:
            assert callable(self.operation), 'operation should be a ``callable``. '
        assert isinstance(probability, Real), f'probability should be a ``real``. Got {type(probability)}.'
        assert probability <= 1.0 and probability >= 0.0, 'probability should be between 0 and 1.'
        self._probability = probability
        self.rng = check_random_state(random_state)

    def get_augmentation_params(self, *batch):
        return dict()

    def forward(self, X: 'Tensor', y: 'Optional[Tensor]'=None) ->Output:
        """General forward pass for an augmentation transform.

        Parameters
        ----------
        X : torch.Tensor
            EEG input example or batch.
        y : torch.Tensor | None
            EEG labels for the example or batch. Defaults to None.

        Returns
        -------
        torch.Tensor
            Transformed inputs.
        torch.Tensor, optional
            Transformed labels. Only returned when y is not None.
        """
        X = torch.as_tensor(X).float()
        out_X = X.clone()
        if len(out_X.shape) < 3:
            out_X = out_X[None, ...]
        if y is not None:
            y = torch.as_tensor(y)
            out_y = y.clone()
            if len(out_y.shape) == 0:
                out_y = out_y.reshape(1)
        else:
            out_y = torch.zeros(out_X.shape[0], device=out_X.device)
        mask = self._get_mask(out_X.shape[0], out_X.device)
        num_valid = mask.sum().long()
        if num_valid > 0:
            out_X[mask, ...], tr_y = self.operation(out_X[mask, ...], out_y[mask], **self.get_augmentation_params(out_X[mask, ...], out_y[mask]))
            if isinstance(tr_y, tuple):
                out_y = tuple(tmp_y[mask] for tmp_y in tr_y)
            else:
                out_y[mask] = tr_y
        out_X = out_X.reshape_as(X)
        if y is not None:
            return out_X, out_y
        else:
            return out_X

    def _get_mask(self, batch_size, device) ->torch.Tensor:
        """Samples whether to apply operation or not over the whole batch"""
        return torch.as_tensor(self.probability > self.rng.uniform(size=batch_size))

    @property
    def probability(self):
        return self._probability


def identity(x):
    return x


class IdentityTransform(Transform):
    """Identity transform.

    Transform that does not change the input.
    """
    operation = staticmethod(identity)


class Compose(Transform):
    """Transform composition.

    Callable class allowing to cast a sequence of Transform objects into a
    single one.

    Parameters
    ----------
    transforms: list
        Sequence of Transforms to be composed.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        super().__init__()

    def forward(self, X, y):
        for transform in self.transforms:
            X, y = transform(X, y)
        return X, y


def time_reverse(X, y):
    """Flip the time axis of each input.

    Parameters
    ----------
    X : torch.Tensor
        EEG input example or batch.
    y : torch.Tensor
        EEG labels for the example or batch.

    Returns
    -------
    torch.Tensor
        Transformed inputs.
    torch.Tensor
        Transformed labels.
    """
    return torch.flip(X, [-1]), y


class TimeReverse(Transform):
    """Flip the time axis of each input with a given probability.

    Parameters
    ----------
    probability : float
        Float setting the probability of applying the operation.
    random_state: int | numpy.random.Generator, optional
        Seed to be used to instantiate numpy random number generator instance.
        Used to decide whether or not to transform given the probability
        argument. Defaults to None.
    """
    operation = staticmethod(time_reverse)

    def __init__(self, probability, random_state=None):
        super().__init__(probability=probability, random_state=random_state)

