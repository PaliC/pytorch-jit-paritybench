import sys
_module = sys.modules[__name__]
del sys
getting_started = _module
autograd_example = _module
custom_linear_layer = _module
datautils = _module
fizbuz = _module
numpy_like_fizbuz = _module
datautils = _module
torchtext_example = _module
bottleneck_support = _module
otherenv = _module
profile_support = _module
ignite_with_checkpointing = _module
dataset = _module
segmentation = _module
segmentationModel = _module
simpleCNN = _module
simpleCNNModel = _module
model = _module
train = _module
model = _module
train = _module
model = _module
train = _module
pixelcnn = _module
wavenet = _module
wavenet_data = _module
mode = _module
util = _module
model = _module
reinforcement_learning = _module
app = _module
controller = _module
model = _module
fizbuz_service = _module
fizbuz = _module
model = _module
run = _module
run_redis = _module
addition = _module
frompython = _module
multinomial = _module
locustfile = _module
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


import torch


import numpy as np


import time


from torch import nn


import torch.optim as optim


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from scipy.signal import convolve2d


from scipy.signal import correlate2d


from torch.nn.modules.module import Module


from torch.nn.parameter import Parameter


from torch.autograd import Function


from numpy.fft import rfft2


from numpy.fft import irfft2


import torch.nn.functional as F


import logging


from torch.optim import SGD


from torchvision.transforms import Compose


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from torchvision.datasets import MNIST


from torch.utils import data


from scipy import misc


from torchvision import transforms


from torch import optim


import torchvision


import torchvision.transforms as transforms


import torch.nn as nn


from collections import namedtuple


import itertools


from torch import backends


from torchvision import datasets


from torchvision import utils


import torch.utils.data as data


import random


import math


from itertools import count


import torchvision.transforms as T


class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(*size, -1)


class Linear(Bottle, nn.Linear):
    pass


class FizBuzNet(nn.Module):
    """
    2 layer network for predicting fiz or buz
    param: input_size -> int
    param: output_size -> int
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(FizBuzNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, batch):
        hidden = self.hidden(batch)
        activated = torch.sigmoid(hidden)
        out = self.out(activated)
        return out


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


class DummyNN(torch.nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.embed = torch.nn.Embedding(len(TEXT.vocab), emb_dim)
        self.embed.weight.data.copy_(TEXT.vocab.vectors)


class ScipyConv2dFunction(Function):

    @staticmethod
    def forward(ctx, input, filter):
        input, filter = input.detach(), filter.detach()
        result = correlate2d(input.numpy(), filter.detach().numpy(), mode='valid')
        ctx.save_for_backward(input, filter)
        return input.new(result)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach()
        input, filter = ctx.saved_tensors
        grad_input = convolve2d(grad_output.numpy(), filter.t().numpy(), mode='full')
        grad_filter = convolve2d(input.numpy(), grad_output.numpy(), mode='valid')
        return grad_output.new_tensor(grad_input), grad_output.new_tensor(grad_filter)


class ScipyConv2d(Module):

    def __init__(self, kh, kw):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(kh, kw))

    def forward(self, input):
        return ScipyConv2dFunction.apply(input, self.filter)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class ConvBlock(nn.Module):
    """ LinkNet uses initial block with conv -> batchnorm -> relu """

    def __init__(self, inp, out, kernal, stride, pad, bias, act):
        super().__init__()
        if act:
            self.conv_block = nn.Sequential(nn.Conv2d(inp, out, kernal, stride, pad, bias=bias), nn.BatchNorm2d(num_features=out), nn.ReLU())
        else:
            self.conv_block = nn.Sequential(nn.Conv2d(inp, out, kernal, stride, pad, bias=bias), nn.BatchNorm2d(num_features=out))

    def forward(self, x):
        return self.conv_block(x)


class DeconvBlock(nn.Module):
    """ LinkNet uses Deconv block with transposeconv -> batchnorm -> relu """

    def __init__(self, inp, out, kernal, stride, pad):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(inp, out, kernal, stride, pad)
        self.batchnorm = nn.BatchNorm2d(out)
        self.relu = nn.ReLU()

    def forward(self, x, output_size):
        convt_out = self.conv_transpose(x, output_size=output_size)
        batchnormout = self.batchnorm(convt_out)
        return self.relu(batchnormout)


class DecoderBlock(nn.Module):
    """ Residucal Block in linknet that does Encoding """

    def __init__(self, inp, out):
        super().__init__()
        self.conv1 = ConvBlock(inp=inp, out=inp // 4, kernal=1, stride=1, pad=0, bias=True, act=True)
        self.deconv = DeconvBlock(inp=inp // 4, out=inp // 4, kernal=3, stride=2, pad=1)
        self.conv2 = ConvBlock(inp=inp // 4, out=out, kernal=1, stride=1, pad=0, bias=True, act=True)

    def forward(self, x, output_size):
        conv1 = self.conv1(x)
        deconv = self.deconv(conv1, output_size=output_size)
        conv2 = self.conv2(deconv)
        return conv2


class EncoderBlock(nn.Module):
    """ Residucal Block in linknet that does Encoding - layers in ResNet18 """

    def __init__(self, inp, out):
        """
        Resnet18 has first layer without downsampling.
        The parameter ``downsampling`` decides that
        # TODO - mention about how n - f/s + 1 is handling output size in
        # in downsample
        """
        super().__init__()
        self.block1 = nn.Sequential(ConvBlock(inp=inp, out=out, kernal=3, stride=2, pad=1, bias=True, act=True), ConvBlock(inp=out, out=out, kernal=3, stride=1, pad=1, bias=True, act=True))
        self.block2 = nn.Sequential(ConvBlock(inp=out, out=out, kernal=3, stride=1, pad=1, bias=True, act=True), ConvBlock(inp=out, out=out, kernal=3, stride=1, pad=1, bias=True, act=True))
        self.residue = ConvBlock(inp=inp, out=out, kernal=3, stride=2, pad=1, bias=True, act=True)

    def forward(self, x):
        out1 = self.block1(x)
        residue = self.residue(x)
        out2 = self.block2(out1 + residue)
        return out2 + out1


class SegmentationModel(nn.Module):
    """
    LinkNet for Semantic segmentation. Inspired heavily by
    https://github.com/meetshah1995/pytorch-semseg
    # TODO -> pad = kernal // 2
    # TODO -> change the var names
    # find size > a = lambda n, f, p, s: (((n + (2 * p)) - f) / s) + 1
    # Cannot have resnet18 architecture because it doesn't do downsampling on first layer
    """

    def __init__(self):
        super().__init__()
        self.init_conv = ConvBlock(inp=3, out=64, kernal=7, stride=2, pad=3, bias=True, act=True)
        self.init_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder1 = EncoderBlock(inp=64, out=64)
        self.encoder2 = EncoderBlock(inp=64, out=128)
        self.encoder3 = EncoderBlock(inp=128, out=256)
        self.encoder4 = EncoderBlock(inp=256, out=512)
        self.decoder4 = DecoderBlock(inp=512, out=256)
        self.decoder3 = DecoderBlock(inp=256, out=128)
        self.decoder2 = DecoderBlock(inp=128, out=64)
        self.decoder1 = DecoderBlock(inp=64, out=64)
        self.final_deconv1 = DeconvBlock(inp=64, out=32, kernal=3, stride=2, pad=1)
        self.final_conv = ConvBlock(inp=32, out=32, kernal=3, stride=1, pad=1, bias=True, act=True)
        self.final_deconv2 = DeconvBlock(inp=32, out=2, kernal=2, stride=2, pad=0)

    def forward(self, x):
        init_conv = self.init_conv(x)
        init_maxpool = self.init_maxpool(init_conv)
        e1 = self.encoder1(init_maxpool)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        d4 = self.decoder4(e4, e3.size()) + e3
        d3 = self.decoder3(d4, e2.size()) + e2
        d2 = self.decoder2(d3, e1.size()) + e1
        d1 = self.decoder1(d2, init_maxpool.size())
        final_deconv1 = self.final_deconv1(d1, init_conv.size())
        final_conv = self.final_conv(final_deconv1)
        final_deconv2 = self.final_deconv2(final_conv, x.size())
        return final_deconv2


class Conv(nn.Module):
    """
    Custom conv layer
    Assumes the image is squre
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = Parameter(torch.zeros(out_channels))

    def forward(self, x):
        if len(x.size()) != 4:
            raise Exception('Batch should be 4 dimensional')
        batch_size = x.size(0)
        height = x.size(2)
        width = x.size(3)
        new_depth = self.weight.size(0)
        new_height = int((height - self.kernel_size) / self.stride + 1)
        new_width = int((width - self.kernel_size) / self.stride + 1)
        if height != width:
            raise Exception('Only processing square Image')
        out = torch.zeros(batch_size, new_depth, new_height, new_width)
        padded_input = F.pad(x, (self.padding,) * 4)
        for nf, f in enumerate(self.weight):
            for h in range(new_height):
                for w in range(new_width):
                    val = padded_input[:, :, h:h + self.kernel_size, w:w + self.kernel_size]
                    out[:, nf, h, w] = val.contiguous().view(batch_size, -1) @ f.view(-1)
                    out[:, nf, h, w] += self.bias[nf]
        return out


class MaxPool(nn.Module):
    """
    Custom max pool layer
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        if len(x.size()) != 4:
            raise Exception('Batch should be 4 dimensional')
        batch_size = x.size(0)
        depth = x.size(1)
        height = x.size(2)
        width = x.size(3)
        new_height = int((height - self.kernel_size) / self.kernel_size + 1)
        new_width = int((width - self.kernel_size) / self.kernel_size + 1)
        if height != width:
            raise Exception('Only processing square Image')
        if height % self.kernel_size != 0:
            raise Exception('Kernal cannot be moved completely, change Kernal size')
        out = torch.zeros(batch_size, depth, new_height, new_width)
        for h in range(new_height):
            for w in range(new_width):
                for d in range(depth):
                    val = x[:, d, h:h + self.kernel_size, w:w + self.kernel_size]
                    out[:, d, h, w] = val.max(2)[0].max(1)[0]
        return out


class SimpleCNNModel(nn.Module):
    """ A basic CNN model implemented with the the basic building blocks """

    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 6, 5)
        self.pool = MaxPool(2)
        self.conv2 = Conv(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RNNCell(nn.Module):

    def __init__(self, embed_dim, hidden_size, vocab_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(embed_dim + hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        combined = torch.cat((inputs, hidden), 2)
        hidden = torch.relu(self.input2hidden(combined))
        return hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class Encoder(nn.Module):

    def __init__(self, embed_dim, vocab_dim, hidden_size):
        super(Encoder, self).__init__()
        self.rnn = RNNCell(embed_dim, hidden_size, vocab_dim)

    def forward(self, inputs):
        ht = self.rnn.init_hidden(inputs.size(1))
        for word in inputs.split(1, dim=0):
            ht = self.rnn(word, ht)
        return ht


class Merger(nn.Module):

    def __init__(self, size, dropout=0.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        prem = data[0]
        hypo = data[1]
        diff = prem - hypo
        prod = prem * hypo
        cated_data = torch.cat([prem, hypo, diff, prod], 2)
        cated_data = cated_data.squeeze()
        return self.dropout(self.bn(cated_data))


class RNNClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_dim, config.embed_dim)
        self.encoder = Encoder(config.embed_dim, config.vocab_dim, config.hidden_size)
        self.classifier = nn.Sequential(Merger(4 * config.hidden_size, config.dropout), nn.Linear(4 * config.hidden_size, config.fc1_dim), nn.ReLU(), nn.BatchNorm1d(config.fc1_dim), nn.Dropout(p=config.dropout), nn.Linear(config.fc1_dim, config.fc2_dim))

    def forward(self, batch):
        prem_embed = self.embed(batch.premise)
        hypo_embed = self.embed(batch.hypothesis)
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)
        scores = self.classifier((premise, hypothesis))
        return scores


class BatchNorm(Bottle, nn.BatchNorm1d):
    pass


def bundle(lstm_iter):
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    return torch.cat(lstm_iter, 0).chunk(2, 1)


def tree_lstm(c1, c2, lstm_in):
    a, i, f1, f2, o = lstm_in.chunk(5, 1)
    c = a.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
    h = o.sigmoid() * c.tanh()
    return h, c


def unbundle(state):
    if state is None:
        return itertools.repeat(None)
    return torch.split(torch.cat(state, 1), 1, 0)


class Reduce(nn.Module):

    def __init__(self, size, tracker_size=None):
        super().__init__()
        self.left = nn.Linear(size, 5 * size)
        self.right = nn.Linear(size, 5 * size, bias=False)
        if tracker_size is not None:
            self.track = nn.Linear(tracker_size, 5 * size, bias=False)

    def forward(self, left_in, right_in, tracking=None):
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)
        lstm_in = self.left(left[0])
        lstm_in += self.right(right[0])
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking[0])
        out = unbundle(tree_lstm(left[1], right[1], lstm_in))
        return out


class Tracker(nn.Module):

    def __init__(self, size, tracker_size, predict):
        super().__init__()
        self.rnn = nn.LSTMCell(3 * size, tracker_size)
        if predict:
            self.transition = nn.Linear(tracker_size, 4)
        self.state_size = tracker_size

    def reset_state(self):
        self.state = None

    def forward(self, bufs, stacks):
        buf = bundle(buf[-1] for buf in bufs)[0]
        stack1 = bundle(stack[-1] for stack in stacks)[0]
        stack2 = bundle(stack[-2] for stack in stacks)[0]
        x = torch.cat((buf, stack1, stack2), 1)
        if self.state is None:
            self.state = 2 * [x.data.new(x.size(0), self.state_size).zero_()]
        self.state = self.rnn(x, self.state)
        if hasattr(self, 'transition'):
            return unbundle(self.state), self.transition(self.state[0])
        return unbundle(self.state), None


class SPINN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.d_hidden == config.d_proj / 2
        self.reduce = Reduce(config.d_hidden, config.d_tracker)
        if config.d_tracker is not None:
            self.tracker = Tracker(config.d_hidden, config.d_tracker, predict=config.predict)

    def forward(self, buffers, transitions):
        buffers = [list(torch.split(b.squeeze(1), 1, 0)) for b in torch.split(buffers, 1, 1)]
        stacks = [[buf[0], buf[0]] for buf in buffers]
        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        else:
            assert transitions is not None
        if transitions is not None:
            num_transitions = transitions.size(0)
        else:
            num_transitions = len(buffers[0]) * 2 - 3
        for i in range(num_transitions):
            if transitions is not None:
                trans = transitions[i]
            if hasattr(self, 'tracker'):
                tracker_states, trans_hyp = self.tracker(buffers, stacks)
                if trans_hyp is not None:
                    trans = trans_hyp.max(1)[1]
            else:
                tracker_states = itertools.repeat(None)
            lefts, rights, trackings = [], [], []
            batch = zip(trans.data, buffers, stacks, tracker_states)
            for transition, buf, stack, tracking in batch:
                if transition == 3:
                    stack.append(buf.pop())
                elif transition == 2:
                    rights.append(stack.pop())
                    lefts.append(stack.pop())
                    trackings.append(tracking)
            if rights:
                reduced = iter(self.reduce(lefts, rights, trackings))
                for transition, stack in zip(trans.data, stacks):
                    if transition == 2:
                        stack.append(next(reduced))
        return bundle([stack.pop() for stack in stacks])[0]


class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.embed_bn = BatchNorm(config.d_proj)
        self.embed_dropout = nn.Dropout(p=config.embed_dropout)
        self.encoder = SPINN(config)
        feat_in_size = config.d_hidden * (2 if self.config.birnn and not self.config.spinn else 1)
        self.merger = Merger(feat_in_size, config.mlp_dropout)
        self.mlp_dropout = nn.Dropout(p=config.mlp_dropout)
        self.relu = nn.ReLU()
        mlp_in_size = 4 * feat_in_size
        mlp = [nn.Linear(mlp_in_size, config.d_mlp), self.relu, nn.BatchNorm1d(config.d_mlp), self.mlp_dropout]
        for i in range(config.n_mlp_layers - 1):
            mlp.extend([nn.Linear(config.d_mlp, config.d_mlp), self.relu, nn.BatchNorm1d(config.d_mlp), self.mlp_dropout])
        mlp.append(nn.Linear(config.d_mlp, config.d_out))
        self.out = nn.Sequential(*mlp)

    def forward(self, batch):
        prem_embed = self.projection(self.embed(batch.premise))
        hypo_embed = self.projection(self.embed(batch.hypothesis))
        prem_embed = self.embed_dropout(self.embed_bn(prem_embed))
        hypo_embed = self.embed_dropout(self.embed_bn(hypo_embed))
        if hasattr(batch, 'premise_transitions'):
            prem_trans = batch.premise_transitions
            hypo_trans = batch.hypothesis_transitions
        else:
            prem_trans = hypo_trans = None
        premise = self.encoder(prem_embed, prem_trans)
        hypothesis = self.encoder(hypo_embed, hypo_trans)
        scores = self.out(self.merger(premise, hypothesis))
        return scores


class MaskedConv2d(nn.Conv2d):

    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class FinalConv(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels, 1)
        self.conv2 = torch.nn.Conv1d(channels, channels, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.softmax(x)


class ResidualBlock(nn.Module):

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3), nn.InstanceNorm2d(in_features), nn.ReLU(inplace=True), nn.ReflectionPad2d(1), nn.Conv2d(in_features, in_features, 3), nn.InstanceNorm2d(in_features)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResidualStack(torch.nn.Module):

    def __init__(self, layer_size, stack_size, res_channels, skip_channels):
        super().__init__()
        self.res_blocks = torch.nn.ModuleList()
        for s in range(stack_size):
            for l in range(layer_size):
                dilation = 2 ** l
                block = ResidualBlock(res_channels, skip_channels, dilation)
                self.res_blocks.append(block)

    def forward(self, x, skip_size):
        skip_connections = []
        for res_block in self.res_blocks:
            x, skip = res_block(x, skip_size)
            skip_connections.append(skip)
        return torch.stack(skip_connections)


class WaveNet(torch.nn.Module):

    def __init__(self, layer_size, stack_size, in_channels, res_channels):
        super().__init__()
        self.rf_size = sum([(2 ** i) for i in range(layer_size)] * stack_size)
        self.causalconv = torch.nn.Conv1d(in_channels, res_channels, kernel_size=2, padding=1, bias=False)
        self.res_stack = ResidualStack(layer_size, stack_size, res_channels, in_channels)
        self.final_conv = FinalConv(in_channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        sample_size = x.size(2)
        out_size = sample_size - self.rf_size
        if out_size < 1:
            None
        else:
            x = self.causalconv(x)[:, :, :-1]
            skip_connections = self.res_stack(x, out_size)
            x = torch.sum(skip_connections, dim=0)
            x = self.final_conv(x)
            return x.transpose(1, 2).contiguous()


class Generator(nn.Module):

    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(256, 512, 4, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self):
        super().__init__()
        n_features = 784
        n_out = 1
        self.hidden0 = nn.Sequential(nn.Linear(n_features, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.hidden1 = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.hidden2 = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.out = nn.Sequential(torch.nn.Linear(256, n_out), torch.nn.Sigmoid())

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self):
        super().__init__()
        n_features = 100
        n_out = 784
        self.hidden0 = nn.Sequential(nn.Linear(n_features, 256), nn.LeakyReLU(0.2))
        self.hidden1 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU(0.2))
        self.hidden2 = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU(0.2))
        self.out = nn.Sequential(nn.Linear(1024, n_out), nn.Tanh())

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Conv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'inp': 4, 'out': 4, 'kernal': 4, 'stride': 1, 'pad': 4, 'bias': 4, 'act': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {'embed_dim': 4, 'vocab_dim': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (EncoderBlock,
     lambda: ([], {'inp': 4, 'out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FinalConv,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (FizBuzNet,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (MaskedConv2d,
     lambda: ([], {'mask_type': 'A', 'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MaxPool,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Merger,
     lambda: ([], {'size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RNNCell,
     lambda: ([], {'embed_dim': 4, 'hidden_size': 4, 'vocab_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (ResidualBlock,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScipyConv2d,
     lambda: ([], {'kh': 4, 'kw': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (SegmentationModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_hhsecond_HandsOnDeepLearningWithPytorch(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

