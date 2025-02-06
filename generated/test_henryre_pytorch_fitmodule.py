import sys
_module = sys.modules[__name__]
del sys
pytorch_fitmodule = _module
fit_module = _module
utils = _module
run_example = _module

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


from collections import OrderedDict


from functools import partial


from torch.autograd import Variable


from torch.nn import CrossEntropyLoss


from torch.nn import Module


from torch.optim import SGD


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import TensorDataset


import torch.nn as nn


import torch.nn.functional as F


from sklearn.datasets import make_multilabel_classification


DEFAULT_LOSS = CrossEntropyLoss()


DEFAULT_OPTIMIZER = partial(SGD, lr=0.001, momentum=0.9)


class ProgressBar(object):
    """Cheers @ajratner"""

    def __init__(self, n, length=40):
        self.n = max(1, n)
        self.nf = float(n)
        self.length = length
        self.ticks = set([round(i / 100.0 * n) for i in range(101)])
        self.ticks.add(n - 1)
        self.bar(0)

    def bar(self, i, message=''):
        """Assumes i ranges through [0, n-1]"""
        if i in self.ticks:
            b = int(np.ceil((i + 1) / self.nf * self.length))
            sys.stdout.write('\r[{0}{1}] {2}%\t{3}'.format('=' * b, ' ' * (self.length - b), int(100 * ((i + 1) / self.nf)), message))
            sys.stdout.flush()

    def close(self, message=''):
        self.bar(self.n - 1)
        sys.stdout.write('{0}\n\n'.format(message))
        sys.stdout.flush()


def add_metrics_to_log(log, metrics, y_true, y_pred, prefix=''):
    for metric in metrics:
        q = metric(y_true, y_pred)
        log[prefix + metric.__name__] = q
    return log


def get_loader(X, y=None, batch_size=1, shuffle=False):
    """Convert X and y Tensors to a DataLoader
        
        If y is None, use a dummy Tensor
    """
    if y is None:
        y = torch.Tensor(X.size()[0])
    return DataLoader(TensorDataset(X, y), batch_size, shuffle)


def log_to_message(log, precision=4):
    fmt = '{0}: {1:.' + str(precision) + 'f}'
    return '    '.join(fmt.format(k, v) for k, v in log.items())


class FitModule(Module):

    def fit(self, X, y, batch_size=32, epochs=1, verbose=1, validation_split=0.0, validation_data=None, shuffle=True, initial_epoch=0, seed=None, loss=DEFAULT_LOSS, optimizer=DEFAULT_OPTIMIZER, metrics=None):
        """Trains the model similar to Keras' .fit(...) method

        # Arguments
            X: training data Tensor.
            y: target data Tensor.
            batch_size: integer. Number of samples per gradient update.
            epochs: integer, the number of times to iterate
                over the training data arrays.
            verbose: 0, 1. Verbosity mode.
                0 = silent, 1 = verbose.
            validation_split: float between 0 and 1:
                fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
            validation_data: (x_val, y_val) tuple on which to evaluate
                the loss and any model metrics
                at the end of each epoch. The model will not
                be trained on this data.
            shuffle: boolean, whether to shuffle the training data
                before each epoch.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
            seed: random seed.
            optimizer: training optimizer
            loss: training loss
            metrics: list of functions with signatures `metric(y_true, y_pred)`
                where y_true and y_pred are both Tensors

        # Returns
            list of OrderedDicts with training metrics
        """
        if seed and seed >= 0:
            torch.manual_seed(seed)
        if validation_data:
            X_val, y_val = validation_data
        elif validation_split and 0.0 < validation_split < 1.0:
            split = int(X.size()[0] * (1.0 - validation_split))
            X, X_val = X[:split], X[split:]
            y, y_val = y[:split], y[split:]
        else:
            X_val, y_val = None, None
        train_data = get_loader(X, y, batch_size, shuffle)
        opt = optimizer(self.parameters())
        logs = []
        self.train()
        for t in range(initial_epoch, epochs):
            if verbose:
                None
            if verbose:
                pb = ProgressBar(len(train_data))
            log = OrderedDict()
            epoch_loss = 0.0
            for batch_i, batch_data in enumerate(train_data):
                X_batch = Variable(batch_data[0])
                y_batch = Variable(batch_data[1])
                opt.zero_grad()
                y_batch_pred = self(X_batch)
                batch_loss = loss(y_batch_pred, y_batch)
                batch_loss.backward()
                opt.step()
                epoch_loss += batch_loss.data[0]
                log['loss'] = float(epoch_loss) / (batch_i + 1)
                if verbose:
                    pb.bar(batch_i, log_to_message(log))
            if metrics:
                y_train_pred = self.predict(X, batch_size)
                add_metrics_to_log(log, metrics, y, y_train_pred)
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val, batch_size)
                val_loss = loss(Variable(y_val_pred), Variable(y_val))
                log['val_loss'] = val_loss.data[0]
                if metrics:
                    add_metrics_to_log(log, metrics, y_val, y_val_pred, 'val_')
            logs.append(log)
            if verbose:
                pb.close(log_to_message(log))
        return logs

    def predict(self, X, batch_size=32):
        """Generates output predictions for the input samples.

        Computation is done in batches.

        # Arguments
            X: input data Tensor.
            batch_size: integer.

        # Returns
            prediction Tensor.
        """
        data = get_loader(X, batch_size=batch_size)
        self.eval()
        r, n = 0, X.size()[0]
        for batch_data in data:
            X_batch = Variable(batch_data[0])
            y_batch_pred = self(X_batch).data
            if r == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:])
            y_pred[r:min(n, r + batch_size)] = y_batch_pred
            r += batch_size
        return y_pred


class MLP(FitModule):

    def __init__(self, n_feats, n_classes, hidden_size=50):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_feats, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        return F.log_softmax(self.fc2(F.relu(self.fc1(x))))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MLP,
     lambda: ([], {'n_feats': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_henryre_pytorch_fitmodule(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

