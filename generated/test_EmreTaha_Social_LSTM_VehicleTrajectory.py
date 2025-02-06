import sys
_module = sys.modules[__name__]
del sys
social_lstm = _module
criterion = _module
grid = _module
helper = _module
model = _module
sample = _module
st_graph = _module
train = _module
utils_vehicle = _module
visualize = _module

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


import numpy as np


from torch.autograd import Variable


import torch.nn as nn


import time


import matplotlib.pyplot as plt


class SocialLSTM(nn.Module):
    """
    Class representing the Social LSTM model
    """

    def __init__(self, args, infer=False):
        """
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        """
        super(SocialLSTM, self).__init__()
        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda
        if infer:
            self.seq_length = 1
        else:
            self.seq_length = args.seq_length
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.cell = nn.LSTMCell(2 * self.embedding_size, self.rnn_size)
        if self.use_cuda:
            self.cell = self.cell
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        if self.use_cuda:
            self.input_embedding_layer = self.input_embedding_layer
        self.tensor_embedding_layer = nn.Linear(self.grid_size * self.grid_size * self.rnn_size, self.embedding_size)
        if self.use_cuda:
            self.tensor_embedding_layer = self.tensor_embedding_layer
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)
        if self.use_cuda:
            self.output_layer = self.output_layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
        if self.use_cuda:
            self.relu = self.relu
            self.dropout = self.dropout

    def getSocialTensor(self, grid, hidden_states):
        """
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        """
        numNodes = grid.size()[0]
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size * self.grid_size, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor
        for node in range(numNodes):
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)
        social_tensor = social_tensor.view(numNodes, self.grid_size * self.grid_size * self.rnn_size)
        return social_tensor

    def forward(self, nodes, grids, nodesPresent, hidden_states, cell_states):
        """
        Forward pass for the model
        params:
        nodes: Input positions
        grids: Grid masks
        nodesPresent: Peds present in each frame
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        """
        numNodes = nodes.size()[1]
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:
            outputs = outputs
        for framenum in range(self.seq_length):
            nodeIDs = nodesPresent[framenum]
            if len(nodeIDs) == 0:
                continue
            list_of_nodes = Variable(torch.LongTensor(nodeIDs))
            if self.use_cuda:
                list_of_nodes = list_of_nodes
            if self.use_cuda:
                hidden_states = hidden_states
            if self.use_cuda:
                cell_states = cell_states
            if self.use_cuda:
                nodes = nodes
            nodes_current = torch.index_select(nodes[framenum], 0, list_of_nodes)
            grid_current = grids[framenum]
            if self.use_cuda:
                grid_current = grid_current
            hidden_states_current = torch.index_select(hidden_states, 0, list_of_nodes)
            cell_states_current = torch.index_select(cell_states, 0, list_of_nodes)
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)
            h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
            outputs[framenum * numNodes + list_of_nodes.data] = self.output_layer(h_nodes)
            hidden_states[list_of_nodes.data] = h_nodes
            cell_states[list_of_nodes.data] = c_nodes
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum * numNodes + node, :]
        return outputs_return, hidden_states, cell_states

