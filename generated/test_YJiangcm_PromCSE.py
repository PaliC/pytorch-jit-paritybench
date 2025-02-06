import sys
_module = sys.modules[__name__]
del sys
bow = _module
gensen = _module
googleuse = _module
infersent = _module
models = _module
skipthought = _module
senteval = _module
binary = _module
engine = _module
mrpc = _module
probing = _module
rank = _module
sick = _module
snli = _module
sst = _module
sts = _module
classifier = _module
ranking = _module
relatedness = _module
validation = _module
trec = _module
utils = _module
setup = _module
evaluation = _module
promcse = _module
models = _module
tool = _module
trainers = _module
setup = _module
train = _module

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


import logging


import numpy as np


import time


import torch.nn as nn


import copy


from torch import nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.optim as optim


from scipy.stats import pearsonr


from scipy.stats import spearmanr


import re


import inspect


from torch import optim


import matplotlib.pyplot as plt


import torch.distributed as dist


from numpy import ndarray


from typing import List


from typing import Tuple


from typing import Union


from torch import Tensor


from sklearn.metrics.pairwise import cosine_similarity


import collections


import math


import warnings


from typing import TYPE_CHECKING


from typing import Any


from typing import Callable


from typing import Dict


from typing import Optional


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataset import Dataset


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


import random


class InferSent(nn.Module):

    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = 1 if 'version' not in config else config['version']
        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1, bidirectional=True, dropout=self.dpout_model)
        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self):
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)
        idx_sort = torch.from_numpy(idx_sort) if self.is_cuda() else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, idx_sort)
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]
        idx_unsort = torch.from_numpy(idx_unsort) if self.is_cuda() else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, idx_unsort)
        if self.pool_type == 'mean':
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == 'max':
            if not self.max_pad:
                sent_output[sent_output == 0] = -1000000000.0
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2
        return emb

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        word_dict = {}
        sentences = [(s.split() if not tokenize else self.tokenize(s)) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        None
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')
                if k > K and all([(w in word_vec) for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        None

    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        None

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        None

    def get_batch(self, batch):
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]
        return torch.FloatTensor(embed)

    def tokenize(self, s):
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [([self.bos] + s.split() + [self.eos] if not tokenize else [self.bos] + self.tokenize(s) + [self.eos]) for s in sentences]
        n_w = np.sum([len(x) for x in sentences])
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors.                                Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f
        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            None
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]
        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(sentences, bsize, tokenize, verbose)
        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = self.get_batch(sentences[stidx:stidx + bsize])
            if self.is_cuda():
                batch = batch
            with torch.no_grad():
                batch = self.forward((batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]
        if verbose:
            None
        return embeddings

    def visualize(self, sent, tokenize=True):
        sent = sent.split() if not tokenize else self.tokenize(sent)
        sent = [[self.bos] + [word for word in sent if word in self.word_vec] + [self.eos]]
        if ' '.join(sent[0]) == '%s %s' % (self.bos, self.eos):
            import warnings
            warnings.warn('No words in "%s" have w2v vectors. Replacing                            by "%s %s"..' % (sent, self.bos, self.eos))
        batch = self.get_batch(sent)
        if self.is_cuda():
            batch = batch
        output = self.enc_lstm(batch)[0]
        output, idxs = torch.max(output, 0)
        idxs = idxs.data.cpu().numpy()
        argmaxs = [np.sum(idxs == k) for k in range(len(sent[0]))]
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [(100.0 * n / np.sum(argmaxs)) for n in argmaxs]
        plt.xticks(x, sent[0], rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()
        return output, idxs


class COCOProjNet(nn.Module):

    def __init__(self, config):
        super(COCOProjNet, self).__init__()
        self.imgdim = config['imgdim']
        self.sentdim = config['sentdim']
        self.projdim = config['projdim']
        self.imgproj = nn.Sequential(nn.Linear(self.imgdim, self.projdim))
        self.sentproj = nn.Sequential(nn.Linear(self.sentdim, self.projdim))

    def forward(self, img, sent, imgc, sentc):
        img = img.unsqueeze(1).expand_as(imgc).contiguous()
        img = img.view(-1, self.imgdim)
        imgc = imgc.view(-1, self.imgdim)
        sent = sent.unsqueeze(1).expand_as(sentc).contiguous()
        sent = sent.view(-1, self.sentdim)
        sentc = sentc.view(-1, self.sentdim)
        imgproj = self.imgproj(img)
        imgproj = imgproj / torch.sqrt(torch.pow(imgproj, 2).sum(1, keepdim=True)).expand_as(imgproj)
        imgcproj = self.imgproj(imgc)
        imgcproj = imgcproj / torch.sqrt(torch.pow(imgcproj, 2).sum(1, keepdim=True)).expand_as(imgcproj)
        sentproj = self.sentproj(sent)
        sentproj = sentproj / torch.sqrt(torch.pow(sentproj, 2).sum(1, keepdim=True)).expand_as(sentproj)
        sentcproj = self.sentproj(sentc)
        sentcproj = sentcproj / torch.sqrt(torch.pow(sentcproj, 2).sum(1, keepdim=True)).expand_as(sentcproj)
        anchor1 = torch.sum(imgproj * sentproj, 1)
        anchor2 = torch.sum(sentproj * imgproj, 1)
        img_sentc = torch.sum(imgproj * sentcproj, 1)
        sent_imgc = torch.sum(sentproj * imgcproj, 1)
        return anchor1, anchor2, img_sentc, sent_imgc

    def proj_sentence(self, sent):
        output = self.sentproj(sent)
        output = output / torch.sqrt(torch.pow(output, 2).sum(1, keepdim=True)).expand_as(output)
        return output

    def proj_image(self, img):
        output = self.imgproj(img)
        output = output / torch.sqrt(torch.pow(output, 2).sum(1, keepdim=True)).expand_as(output)
        return output


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss
    """

    def __init__(self, margin):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor1, anchor2, img_sentc, sent_imgc):
        cost_sent = torch.clamp(self.margin - anchor1 + img_sentc, min=0.0).sum()
        cost_img = torch.clamp(self.margin - anchor2 + sent_imgc, min=0.0).sum()
        loss = cost_sent + cost_img
        return loss


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config, model_args):
        super().__init__()
        self.prefix_projection = model_args.prefix_projection
        if self.prefix_projection:
            self.embedding = torch.nn.Embedding(model_args.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(torch.nn.Linear(config.hidden_size, model_args.prefix_hidden_size), torch.nn.Tanh(), torch.nn.Linear(model_args.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size))
        else:
            self.embedding = torch.nn.Embedding(model_args.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: 'torch.Tensor'):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 'unrecognized pooling type %s' % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states
        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == 'avg':
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == 'avg_first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (COCOProjNet,
     lambda: ([], {'config': _mock_config(imgdim=4, sentdim=4, projdim=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPLayer,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PairwiseRankingLoss,
     lambda: ([], {'margin': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Similarity,
     lambda: ([], {'temp': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_YJiangcm_PromCSE(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

