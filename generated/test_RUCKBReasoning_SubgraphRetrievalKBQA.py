import sys
_module = sys.modules[__name__]
del sys
bow = _module
gensen = _module
googleuse = _module
infersent = _module
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
tools = _module
classifier = _module
ranking = _module
relatedness = _module
validation = _module
trec = _module
utils = _module
setup = _module
config = _module
data_loader = _module
graftnet = _module
main = _module
build_relation_set_for_end2end = _module
retrieve_subgraph_for_end2end = _module
run_retrieve_subgraph = _module
script = _module
util = _module
BaseAgent = _module
NSMAgent = _module
TeacherAgent = _module
TeacherAgent2 = _module
base_instruction = _module
seq_instruction = _module
base_reasoning = _module
gnn_backward_reasoning = _module
gnn_reasoning = _module
layer_nsm = _module
layers_att = _module
evaluate_nsm = _module
init = _module
trainer_hybrid = _module
trainer_nsm = _module
trainer_parallel = _module
trainer_student = _module
config = _module
build_search_tree_for_cwq_dataset = _module
build_search_tree_for_dataset = _module
build_train_example = _module
cal_path_for_cwq = _module
main_nsm = _module
retrieve_subgraph_for_end2end = _module
build_ent_type_ary = _module
convert_all_triple_to_int = _module
map_test_set = _module
graftnet = _module
main = _module
step0_preprocess_webqsp = _module
step1_process_entity_links = _module
step2_relation_embeddings = _module
step3_question_embeddings = _module
step4_extract_subgraphs = _module
util = _module
knowledge_graph = _module
knowledge_graph_base = _module
knowledge_graph_cache = _module
knowledge_graph_freebase = _module
BaseAgent = _module
NSMAgent = _module
TeacherAgent = _module
TeacherAgent2 = _module
base_instruction = _module
seq_instruction = _module
base_reasoning = _module
gnn_backward_reasoning = _module
gnn_reasoning = _module
layer_nsm = _module
layers_att = _module
evaluate_nsm = _module
trainer_hybrid = _module
trainer_nsm = _module
trainer_parallel = _module
trainer_student = _module
config = _module
main_nsm = _module
build_training_dataset = _module
filter_legal_path = _module
negative_sampling = _module
score_path = _module
search_to_get_path = _module
build_relation_set = _module
eval = _module
generate_relation_cache = _module
retrieve_subgraph = _module
convert_to_plm = _module
generate_tokenized_data = _module
model = _module
convert_to_plm = _module
model = _module
tokenize_csv_file = _module
tokenize_jsonl_file = _module
train = _module
trainer = _module
trainer = _module
run_convert_retriever_output_to_graftnet = _module
run_convert_retriever_output_to_graftnet_end2end = _module
run_preprocess = _module
run_train_nsm = _module
infersent = _module
classifier = _module
ranking = _module
relatedness = _module
utils = _module
graftnet = _module
load_data_from_nsm = _module
main = _module
util = _module
BaseAgent = _module
NSMAgent = _module
TeacherAgent = _module
TeacherAgent2 = _module
backward_model = _module
base_model = _module
forward_model = _module
hybrid_model = _module
nsm_model = _module
base_instruction = _module
seq_instruction = _module
base_reasoning = _module
gnn_backward_reasoning = _module
gnn_reasoning = _module
layer_nsm = _module
layers_att = _module
basic_dataset = _module
dataset_single = _module
dataset_super = _module
load_data = _module
load_data_super = _module
read_tree = _module
evaluate_nsm = _module
trainer_hybrid = _module
trainer_nsm = _module
trainer_parallel = _module
trainer_student = _module
config = _module
main_nsm = _module
build_vocab_from_dep = _module
get_2hop_subgraph = _module
get_seed_set = _module
load_emb_glove = _module
manual_filter_rel = _module
map_kb_id = _module
preprocess_step0 = _module
preprocess_step1 = _module
update_vocab_with_rel = _module
deal_cvt = _module
ppr_util = _module
prepare_data = _module
simplify_dataset = _module
con_parse = _module
dep_parse = _module
load_dataset = _module
negative_sampling_for_unsup = _module
retrieve_subgraph = _module
retrieve_subgraph_for_finetune = _module
retrieve_subgraph_for_graftnet = _module
retrieve_subgraph_for_test = _module
evaluation = _module
huggingface_to_simcse = _module
simcse_to_huggingface = _module
models = _module
trainers = _module
train = _module
BaseAgent = _module
NSMAgent = _module
TeacherAgent = _module
TeacherAgent2 = _module
backward_model = _module
base_model = _module
forward_model = _module
hybrid_model = _module
nsm_model = _module
base_instruction = _module
seq_instruction = _module
base_reasoning = _module
gnn_backward_reasoning = _module
gnn_reasoning = _module
layer_nsm = _module
layers_att = _module
basic_dataset = _module
dataset_single = _module
dataset_super = _module
evaluate_nsm = _module
trainer_hybrid = _module
trainer_nsm = _module
trainer_parallel = _module
trainer_student = _module
config = _module
build_train_set_from_search_tree = _module
cal_path_score_from_reader = _module
main_nsm = _module
run_retriever_finetune = _module
run_train_graftnet = _module
run_train_retriever = _module

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


import copy


from torch import nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.optim as optim


from scipy.stats import pearsonr


import re


import inspect


from torch import optim


import torch.nn as nn


from typing import Tuple


from typing import List


from typing import Any


from typing import Dict


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import time


import torch.sparse as sparse


import math


import random


from torch.optim.lr_scheduler import ExponentialLR


from numpy import positive


from torch import negative


from torch import threshold


from typing import Optional


from typing import Union


from collections import Counter


from typing import Set


from numpy import tri


import torch.distributed as dist


import collections


import warnings


from typing import TYPE_CHECKING


from typing import Callable


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataset import Dataset


from torch.utils.data.distributed import DistributedSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data.sampler import SequentialSampler


import pandas as pd


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


VERY_NEG_NUMBER = -100000000000


VERY_SMALL_NUMBER = 1e-10


class LeftMMFixed(torch.autograd.Function):
    """
    Implementation of matrix multiplication of a Sparse Variable with a Dense Variable, returning a Dense one.
    This is added because there's no autograd for sparse yet. No gradient computed on the sparse weights.
    """

    def __init__(self):
        super(LeftMMFixed, self).__init__()
        self.sparse_weights = None

    @staticmethod
    def forward(self, sparse_weights, x):
        if self.sparse_weights is None:
            self.sparse_weights = sparse_weights
        return torch.mm(self.sparse_weights, x)

    @staticmethod
    def backward(self, grad_output):
        sparse_weights = self.sparse_weights
        return None, torch.mm(sparse_weights.t(), grad_output)


def use_cuda(var):
    if torch.cuda.is_available():
        return var
    else:
        return var


def sparse_bmm(X, Y):
    """Batch multiply X and Y where X is sparse, Y is dense.
    Args:
        X: Sparse tensor of size BxMxN. Consists of two tensors,
            I:3xZ indices, and V:1xZ values.
        Y: Dense tensor of size BxNxK.
    Returns:
        batched-matmul(X, Y): BxMxK
    """
    I = X._indices()
    V = X._values()
    B, M, N = X.size()
    _, _, K = Y.size()
    Z = I.size()[1]
    lookup = Y[I[0, :], I[2, :], :]
    X_I = torch.stack((I[0, :] * M + I[1, :], use_cuda(torch.arange(Z).type(torch.LongTensor))), 0)
    S = use_cuda(Variable(torch.sparse.FloatTensor(X_I, V, torch.Size([B * M, Z])), requires_grad=False))
    prod_op = LeftMMFixed()
    prod = torch.mm(S, lookup)
    return prod.view(B, M, K)


class GraftNet(nn.Module):

    def __init__(self, pretrained_word_embedding_file, pretrained_entity_emb_file, pretrained_entity_kge_file, pretrained_relation_emb_file, pretrained_relation_kge_file, num_layer, num_relation, num_entity, num_word, entity_dim, word_dim, kge_dim, pagerank_lambda, fact_scale, lstm_dropout, linear_dropout, use_kb, use_doc):
        """
        num_relation: number of relation including self-connection
        """
        super(GraftNet, self).__init__()
        self.num_layer = num_layer
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        self.entity_dim = entity_dim
        self.word_dim = word_dim
        self.pagerank_lambda = pagerank_lambda
        self.fact_scale = fact_scale
        self.has_entity_kge = False
        self.has_relation_kge = False
        self.use_kb = use_kb
        self.use_doc = use_doc
        self.entity_embedding = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=word_dim, padding_idx=num_entity)
        if pretrained_entity_emb_file is not None:
            self.entity_embedding.weight = nn.Parameter(torch.from_numpy(np.pad(np.load(pretrained_entity_emb_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
            self.entity_embedding.weight.requires_grad = False
        else:
            self.entity_embedding.weight = nn.Parameter(torch.from_numpy(np.pad(np.zeros((num_entity, word_dim)), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
            self.entity_embedding.weight.requires_grad = False
        if pretrained_entity_kge_file is not None:
            self.has_entity_kge = True
            self.entity_kge = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=kge_dim, padding_idx=num_entity)
            self.entity_kge.weight = nn.Parameter(torch.from_numpy(np.pad(np.load(pretrained_entity_kge_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
            self.entity_kge.weight.requires_grad = False
        if self.has_entity_kge:
            self.entity_linear = nn.Linear(in_features=word_dim + kge_dim, out_features=entity_dim)
        else:
            self.entity_linear = nn.Linear(in_features=word_dim, out_features=entity_dim)
        self.relation_embedding = nn.Embedding(num_embeddings=num_relation + 1, embedding_dim=2 * word_dim, padding_idx=num_relation)
        if pretrained_relation_emb_file is not None:
            self.relation_embedding.weight = nn.Parameter(torch.from_numpy(np.pad(np.load(pretrained_relation_emb_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
        if pretrained_relation_kge_file is not None:
            self.has_relation_kge = True
            self.relation_kge = nn.Embedding(num_embeddings=num_relation + 1, embedding_dim=kge_dim, padding_idx=num_relation)
            self.relation_kge.weight = nn.Parameter(torch.from_numpy(np.pad(np.load(pretrained_relation_kge_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
        if self.has_relation_kge:
            self.relation_linear = nn.Linear(in_features=2 * word_dim + kge_dim, out_features=entity_dim)
        else:
            self.relation_linear = nn.Linear(in_features=2 * word_dim, out_features=entity_dim)
        self.k = 3
        for i in range(self.num_layer):
            self.add_module('q2e_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('d2e_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2q_linear' + str(i), nn.Linear(in_features=self.k * entity_dim, out_features=entity_dim))
            self.add_module('e2d_linear' + str(i), nn.Linear(in_features=self.k * entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=self.k * entity_dim, out_features=entity_dim))
            if self.use_kb:
                self.add_module('kb_head_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
                self.add_module('kb_tail_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
                self.add_module('kb_self_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
        self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=word_dim, padding_idx=num_word)
        if pretrained_word_embedding_file is not None:
            self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.pad(np.load(pretrained_word_embedding_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False
        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, batch_first=True, bidirectional=False)
        self.query_rel_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, batch_first=True, bidirectional=False)
        self.bi_text_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, batch_first=True, bidirectional=True)
        self.doc_info_carrier = nn.LSTM(input_size=entity_dim, hidden_size=entity_dim, batch_first=True, bidirectional=True)
        self.lstm_drop = nn.Dropout(p=lstm_dropout)
        self.linear_drop = nn.Dropout(p=linear_dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax_d1 = nn.Softmax(dim=1)
        self.kld_loss = nn.KLDivLoss()
        self.bce_loss = nn.BCELoss()
        self.bce_loss_logits = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        """
        :local_entity: global_id of each entity                     (batch_size, max_local_entity)
        :q2e_adj_mat: adjacency matrices (dense)                    (batch_size, max_local_entity, 1)
        :kb_adj_mat: adjacency matrices (sparse)                    (batch_size, max_fact, max_local_entity), (batch_size, max_local_entity, max_fact)
        :kb_fact_rel:                                               (batch_size, max_fact)
        :query_text: a list of words in the query                   (batch_size, max_query_word)
        :document_text:                                             (batch_size, max_relevant_doc, max_document_word)
        :entity_pos: sparse entity_pos_mat                          (batch_size, max_local_entity, max_relevant_doc * max_document_word) 
        :answer_dist: an distribution over local_entity             (batch_size, max_local_entity)
        """
        local_entity, q2e_adj_mat, kb_adj_mat, kb_fact_rel, query_text, document_text, entity_pos, answer_dist = batch
        batch_size, max_local_entity = local_entity.shape
        _, max_relevant_doc, max_document_word = document_text.shape
        _, max_fact = kb_fact_rel.shape
        local_entity = use_cuda(Variable(torch.from_numpy(local_entity).type('torch.LongTensor'), requires_grad=False))
        local_entity_mask = use_cuda((local_entity != self.num_entity).type('torch.FloatTensor'))
        kb_fact_rel = use_cuda(Variable(torch.from_numpy(kb_fact_rel).type('torch.LongTensor'), requires_grad=False))
        query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor'), requires_grad=False))
        query_mask = use_cuda((query_text != self.num_word).type('torch.FloatTensor'))
        answer_dist = use_cuda(Variable(torch.from_numpy(answer_dist).type('torch.FloatTensor'), requires_grad=False))
        pagerank_f = use_cuda(Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor'), requires_grad=True)).squeeze(dim=2)
        q2e_adj_mat = use_cuda(Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor'), requires_grad=False))
        assert pagerank_f.requires_grad == True
        query_word_emb = self.word_embedding(query_text)
        query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb), self.init_hidden(1, batch_size, self.entity_dim))
        query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1)
        if self.use_kb:
            (e2f_batch, e2f_f, e2f_e, e2f_val), (f2e_batch, f2e_e, f2e_f, f2e_val) = kb_adj_mat
            entity2fact_index = torch.LongTensor([e2f_batch, e2f_f, e2f_e])
            entity2fact_val = torch.FloatTensor(e2f_val)
            entity2fact_mat = use_cuda(torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size([batch_size, max_fact, max_local_entity])))
            fact2entity_index = torch.LongTensor([f2e_batch, f2e_e, f2e_f])
            fact2entity_val = torch.FloatTensor(f2e_val)
            fact2entity_mat = use_cuda(torch.sparse.FloatTensor(fact2entity_index, fact2entity_val, torch.Size([batch_size, max_local_entity, max_fact])))
            local_fact_emb = self.relation_embedding(kb_fact_rel)
            if self.has_relation_kge:
                local_fact_emb = torch.cat((local_fact_emb, self.relation_kge(kb_fact_rel)), dim=2)
            local_fact_emb = self.relation_linear(local_fact_emb)
            div = float(np.sqrt(self.entity_dim))
            fact2query_sim = torch.bmm(query_hidden_emb, local_fact_emb.transpose(1, 2)) / div
            fact2query_sim = self.softmax_d1(fact2query_sim + (1 - query_mask.unsqueeze(dim=2)) * VERY_NEG_NUMBER)
            fact2query_att = torch.sum(fact2query_sim.unsqueeze(dim=3) * query_hidden_emb.unsqueeze(dim=2), dim=1)
            W = torch.sum(fact2query_att * local_fact_emb, dim=2) / div
            W_max = torch.max(W, dim=1, keepdim=True)[0]
            W_tilde = torch.exp(W - W_max)
            e2f_softmax = sparse_bmm(entity2fact_mat.transpose(1, 2), W_tilde.unsqueeze(dim=2)).squeeze(dim=2)
            e2f_softmax = torch.clamp(e2f_softmax, min=VERY_SMALL_NUMBER)
            e2f_out_dim = use_cuda(Variable(torch.sum(entity2fact_mat.to_dense(), dim=1), requires_grad=False))
        local_entity_emb = self.entity_embedding(local_entity)
        if self.has_entity_kge:
            local_entity_emb = torch.cat((local_entity_emb, self.entity_kge(local_entity)), dim=2)
        if self.word_dim != self.entity_dim:
            local_entity_emb = self.entity_linear(local_entity_emb)
        for i in range(self.num_layer):
            q2e_linear = getattr(self, 'q2e_linear' + str(i))
            e2q_linear = getattr(self, 'e2q_linear' + str(i))
            e2e_linear = getattr(self, 'e2e_linear' + str(i))
            kb_self_linear = getattr(self, 'kb_self_linear' + str(i))
            kb_head_linear = getattr(self, 'kb_head_linear' + str(i))
            kb_tail_linear = getattr(self, 'kb_tail_linear' + str(i))
            next_local_entity_emb = local_entity_emb
            q2e_emb = q2e_linear(self.linear_drop(query_node_emb)).expand(batch_size, max_local_entity, self.entity_dim)
            next_local_entity_emb = torch.cat((next_local_entity_emb, q2e_emb), dim=2)
            e2f_emb = self.relu(kb_self_linear(local_fact_emb) + sparse_bmm(entity2fact_mat, kb_head_linear(self.linear_drop(local_entity_emb))))
            e2f_softmax_normalized = W_tilde.unsqueeze(dim=2) * sparse_bmm(entity2fact_mat, (pagerank_f / e2f_softmax).unsqueeze(dim=2))
            e2f_emb = e2f_emb * e2f_softmax_normalized
            f2e_emb = self.relu(kb_self_linear(local_entity_emb) + sparse_bmm(fact2entity_mat, kb_tail_linear(self.linear_drop(e2f_emb))))
            pagerank_f = self.pagerank_lambda * sparse_bmm(fact2entity_mat, e2f_softmax_normalized).squeeze(dim=2) + (1 - self.pagerank_lambda) * pagerank_f
            next_local_entity_emb = torch.cat((next_local_entity_emb, self.fact_scale * f2e_emb), dim=2)
            query_node_emb = torch.bmm(pagerank_f.unsqueeze(dim=1), e2q_linear(self.linear_drop(next_local_entity_emb)))
            local_entity_emb = self.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))
        score = self.score_func(self.linear_drop(local_entity_emb)).squeeze(dim=2)
        loss = self.bce_loss_logits(score, answer_dist)
        score = score + (1 - local_entity_mask) * VERY_NEG_NUMBER
        pred_dist = self.sigmoid(score) * local_entity_mask
        pred = torch.max(score, dim=1)[1]
        return loss, pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))), use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size)))


class BaseAgent(nn.Module):

    def __init__(self, args, logger, num_entity, num_relation, num_word):
        super(BaseAgent, self).__init__()
        self.parse_args(args, num_entity, num_relation, num_word)

    def parse_args(self, args, num_entity, num_relation, num_word):
        self.args = args
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        self.use_inverse_relation = args['use_inverse_relation']
        self.use_self_loop = args['use_self_loop']
        None
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.learning_rate = self.args['lr']
        self.q_type = args['q_type']
        self.num_step = args['num_step']
        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)
        self.reset_time = 0

    @staticmethod
    def get_node_emb(query_hidden_emb, action):
        """

        :param query_hidden_emb: (batch_size, max_hyper, emb)
        :param action: (batch_size)
        :return: (batch_size, emb)
        """
        batch_size, max_hyper, _ = query_hidden_emb.size()
        row_idx = torch.arange(0, batch_size).type(torch.LongTensor)
        q_rep = query_hidden_emb[row_idx, action, :]
        return q_rep

    def deal_input_seq(self, batch):
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor')
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor')
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor')
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor')
        current_dist = Variable(seed_dist, requires_grad=True)
        query_text = torch.from_numpy(query_text).type('torch.LongTensor')
        query_mask = (query_text != self.num_word).float()
        return current_dist, query_text, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id

    def forward(self, *args):
        pass

    @staticmethod
    def mask_max(values, mask, keepdim=True):
        return torch.max(values + (1 - mask) * VERY_NEG_NUMBER, dim=-1, keepdim=keepdim)[0]

    @staticmethod
    def mask_argmax(values, mask):
        return torch.argmax(values + (1 - mask) * VERY_NEG_NUMBER, dim=-1, keepdim=True)


class TypeLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, linear_drop, device):
        super(TypeLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_drop = linear_drop
        self.kb_self_linear = nn.Linear(in_features, out_features)
        self.device = device

    def forward(self, local_entity, edge_list, rel_features):
        """
        input_vector: (batch_size, max_local_entity)
        curr_dist: (batch_size, max_local_entity)
        instruction: (batch_size, hidden_size)
        """
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = edge_list
        num_fact = len(fact_ids)
        batch_size, max_local_entity = local_entity.size()
        hidden_size = self.in_features
        fact2head = torch.LongTensor([batch_heads, fact_ids])
        fact2tail = torch.LongTensor([batch_tails, fact_ids])
        batch_rels = torch.LongTensor(batch_rels)
        batch_ids = torch.LongTensor(batch_ids)
        val_one = torch.ones_like(batch_ids).float()
        fact_rel = torch.ones(batch_rels.size(0), hidden_size, dtype=torch.float32, device=self.device)
        fact_val = self.kb_self_linear(fact_rel)
        fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        fact2head_mat = self._build_sparse_tensor(fact2head, val_one, (batch_size * max_local_entity, num_fact))
        f2e_emb = F.relu(torch.sparse.mm(fact2tail_mat, fact_val) + torch.sparse.mm(fact2head_mat, fact_val))
        assert not torch.isnan(f2e_emb).any()
        f2e_emb = f2e_emb.view(batch_size, max_local_entity, hidden_size)
        return f2e_emb

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size)


batch_size = 100


def f1_and_hits_new(answers, candidate2prob, eps=0.5):
    retrieved = []
    correct = 0
    cand_list = sorted(candidate2prob, key=lambda x: x[1], reverse=True)
    if len(cand_list) == 0:
        best_ans = -1
    else:
        best_ans = cand_list[0][0]
    tp_prob = 0.0
    for c, prob in cand_list:
        retrieved.append((c, prob))
        tp_prob += prob
        if c in answers:
            correct += 1
        if tp_prob > eps:
            break
    if len(answers) == 0:
        if len(retrieved) == 0:
            return 1.0, 1.0, 1.0, 1.0, 0, retrieved
        else:
            return 0.0, 1.0, 0.0, 1.0, 1, retrieved
    else:
        hits = float(best_ans in answers)
        if len(retrieved) == 0:
            return 1.0, 0.0, 0.0, hits, 2, retrieved
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return p, r, f1, hits, 3, retrieved


class BaseModel(torch.nn.Module):

    def __init__(self, args, num_entity, num_relation, num_word):
        super(BaseModel, self).__init__()
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        self._parse_args(args)
        self.embedding_def()
        self.share_module_def()
        self.model_name = args['model_name'].lower()
        None

    def _parse_args(self, args):
        self.use_inverse_relation = args['use_inverse_relation']
        self.use_self_loop = args['use_self_loop']
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.has_entity_kge = False
        self.has_relation_kge = False
        self.q_type = args['q_type']
        self.num_layer = args['num_layer']
        self.num_step = args['num_step']
        self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']
        self.encode_type = args['encode_type']
        self.reason_kb = args['reason_kb']
        self.eps = args['eps']
        self.loss_type = args['loss_type']
        self.label_f1 = args['label_f1']
        self.entropy_weight = args['entropy_weight']
        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)
        self.reset_time = 0

    def share_module_def(self):
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        if self.has_entity_kge:
            self.entity_linear = nn.Linear(in_features=kg_dim + kge_dim, out_features=entity_dim)
        else:
            self.entity_linear = nn.Linear(in_features=kg_dim, out_features=entity_dim)
        if self.has_relation_kge:
            self.relation_linear = nn.Linear(in_features=2 * kg_dim + kge_dim, out_features=entity_dim)
        else:
            self.relation_linear = nn.Linear(in_features=2 * kg_dim, out_features=entity_dim)
        self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim, linear_drop=self.linear_drop, device=self.device)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity, edge_list=kb_adj_mat, rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)
            if self.has_entity_kge:
                local_entity_emb = torch.cat((local_entity_emb, self.entity_kge(local_entity)), dim=2)
            if self.word_dim != self.entity_dim:
                local_entity_emb = self.entity_linear(local_entity_emb)
        return local_entity_emb

    def embedding_def(self):
        word_dim = self.word_dim
        kge_dim = self.kge_dim
        kg_dim = self.kg_dim
        num_entity = self.num_entity
        num_relation = self.num_relation
        num_word = self.num_word
        if not self.encode_type:
            self.entity_embedding = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=kg_dim, padding_idx=num_entity)
            if self.entity_emb_file is not None:
                self.entity_embedding.weight = nn.Parameter(torch.from_numpy(np.pad(np.load(self.entity_emb_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
                self.entity_embedding.weight.requires_grad = False
            if self.entity_kge_file is not None:
                self.has_entity_kge = True
                self.entity_kge = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=kge_dim, padding_idx=num_entity)
                self.entity_kge.weight = nn.Parameter(torch.from_numpy(np.pad(np.load(self.entity_kge_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
                self.entity_kge.weight.requires_grad = False
            else:
                self.entity_kge = None
        self.relation_embedding = nn.Embedding(num_embeddings=num_relation, embedding_dim=2 * kg_dim)
        if self.relation_emb_file is not None:
            np_tensor = self.load_relation_file(self.relation_emb_file)
            self.relation_embedding.weight = nn.Parameter(torch.from_numpy(np_tensor).type('torch.FloatTensor'))
        if self.relation_kge_file is not None:
            self.has_relation_kge = True
            self.relation_kge = nn.Embedding(num_embeddings=num_relation, embedding_dim=kge_dim)
            np_tensor = self.load_relation_file(self.relation_kge_file)
            self.relation_kge.weight = nn.Parameter(torch.from_numpy(np_tensor).type('torch.FloatTensor'))
        else:
            self.relation_kge = None
        self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=word_dim, padding_idx=num_word)
        if self.word_emb_file is not None:
            self.word_embedding.weight = nn.Parameter(torch.from_numpy(np.pad(np.load(self.word_emb_file), ((0, 1), (0, 0)), 'constant')).type('torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False

    def load_relation_file(self, filename):
        half_tensor = np.load(filename)
        num_pad = 0
        if self.use_self_loop:
            num_pad = 1
        if self.use_inverse_relation:
            load_tensor = np.concatenate([half_tensor, half_tensor])
        else:
            load_tensor = half_tensor
        return np.pad(load_tensor, ((0, num_pad), (0, 0)), 'constant')

    def get_rel_feature(self):
        rel_features = self.relation_embedding.weight
        if self.has_relation_kge:
            rel_features = torch.cat((rel_features, self.relation_kge.weight), dim=-1)
        rel_features = self.relation_linear(rel_features)
        return rel_features

    def get_rel_feature_from_text_encoder(self, lstm_instruction, relation_text):
        rel_features, _ = lstm_instruction.encode_question(relation_text)
        rel_features = torch.mean(rel_features, dim=1)
        return rel_features

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return self.instruction.init_hidden(num_layer, batch_size, hidden_size)

    def encode_question(self, q_input):
        return self.instruction.encode_question(q_input)

    def get_instruction(self, query_hidden_emb, query_mask, states):
        return self.instruction.get_instruction(query_hidden_emb, query_mask, states)

    def get_loss_bce(self, pred_dist_score, answer_dist):
        answer_dist = (answer_dist > 0) * 0.95
        loss = self.bce_loss_logits(pred_dist_score, answer_dist)
        return loss

    def get_loss_kl(self, pred_dist, answer_dist):
        answer_len = torch.sum(answer_dist, dim=1, keepdim=True)
        answer_len[answer_len == 0] = 1.0
        answer_prob = answer_dist.div(answer_len)
        log_prob = torch.log(pred_dist + 1e-08)
        loss = self.kld_loss(log_prob, answer_prob)
        return loss

    def get_loss_new(self, pred_dist, answer_dist, reduction='mean'):
        if self.loss_type == 'bce':
            tp_loss = self.get_loss_bce(pred_dist, answer_dist)
            if reduction == 'none':
                return tp_loss
            else:
                return torch.mean(tp_loss)
        else:
            tp_loss = self.get_loss_kl(pred_dist, answer_dist)
            if reduction == 'none':
                return tp_loss
            else:
                return torch.sum(tp_loss) / pred_dist.size(0)

    def calc_loss_label(self, label_dist, label_valid):
        loss_label = None
        for i in range(self.num_step):
            cur_dist = self.dist_history[i + 1]
            cur_label_dist = label_dist[i]
            cur_label_dist = torch.from_numpy(cur_label_dist).type('torch.FloatTensor')
            tp_loss = self.get_loss_new(pred_dist=cur_dist, answer_dist=cur_label_dist, reduction='none')
            tp_loss = tp_loss * label_valid
            if self.loss_type == 'bce':
                cur_loss = torch.mean(tp_loss)
            elif self.loss_type == 'kl':
                cur_loss = torch.sum(tp_loss) / cur_dist.size(0)
            else:
                raise NotImplementedError
            if loss_label is None:
                loss_label = cur_loss
            else:
                loss_label += cur_loss
        return loss_label

    def calc_f1(self, curr_dist, dist_ans, eps=0.01, metric='f1'):
        dist_now = (curr_dist > eps).float()
        dist_ans = (dist_ans > eps).float()
        correct_num = torch.sum(dist_now * dist_ans, dim=-1)
        answer_num = torch.sum(dist_ans, dim=-1)
        answer_num[answer_num == 0] = 1.0
        pred_num = torch.sum(dist_ans, dim=-1)
        pred_num[pred_num == 0] = 1.0
        recall = correct_num.div(answer_num)
        if metric == 'recall':
            return recall
        precision = correct_num.div(pred_num)
        if metric == 'precision':
            return precision
        mask = (correct_num == 0).float()
        precision_ = precision + mask * VERY_SMALL_NUMBER
        recall_ = recall + mask * VERY_SMALL_NUMBER
        f1 = 2.0 / (1.0 / precision_ + 1.0 / recall_)
        f1_0 = torch.zeros_like(f1)
        f1 = torch.where(correct_num > 0, f1, f1_0)
        return precision, recall, f1

    def calc_f1_new(self, curr_dist, dist_ans, h1_vec):
        batch_size = curr_dist.size(0)
        max_local_entity = curr_dist.size(1)
        seed_dist = self.dist_history[0]
        local_entity = self.local_entity
        ignore_prob = (1 - self.eps) / max_local_entity
        pad_ent_id = self.num_entity
        f1_list = []
        for batch_id in range(batch_size):
            if h1_vec[batch_id].item() == 0.0:
                f1_list.append(0.0)
                continue
            candidates = local_entity[batch_id, :].tolist()
            probs = curr_dist[batch_id, :].tolist()
            answer_prob = dist_ans[batch_id, :].tolist()
            seed_entities = seed_dist[batch_id, :].tolist()
            answer_list = []
            candidate2prob = []
            for c, p, p_a, s in zip(candidates, probs, answer_prob, seed_entities):
                if s > 0:
                    continue
                if c == pad_ent_id:
                    continue
                if p_a > 0:
                    answer_list.append(c)
                if p < ignore_prob:
                    continue
                candidate2prob.append((c, p))
            precision, recall, f1, hits = f1_and_hits_new(answer_list, candidate2prob, self.eps)
            f1_list.append(f1)
        f1_vec = torch.FloatTensor(f1_list)
        return f1_vec

    def calc_h1(self, curr_dist, dist_ans, eps=0.01):
        greedy_option = curr_dist.argmax(dim=-1, keepdim=True)
        dist_top1 = torch.zeros_like(curr_dist).scatter_(1, greedy_option, 1.0)
        dist_ans = (dist_ans > eps).float()
        h1 = torch.sum(dist_top1 * dist_ans, dim=-1)
        return (h1 > 0).float()

    def get_eval_metric(self, pred_dist, answer_dist):
        with torch.no_grad():
            h1 = self.calc_h1(curr_dist=pred_dist, dist_ans=answer_dist, eps=VERY_SMALL_NUMBER)
            f1 = self.calc_f1_new(pred_dist, answer_dist, h1)
        return h1, f1

    def get_label_valid(self, pred_dist, answer_dist, label_f1=0.8):
        with torch.no_grad():
            h1 = self.calc_h1(curr_dist=pred_dist, dist_ans=answer_dist, eps=VERY_SMALL_NUMBER)
            f1 = self.calc_f1_new(pred_dist, answer_dist, h1)
        f1_valid = (f1 > label_f1).float()
        return (h1 * f1_valid).unsqueeze(1)

    def get_attn_align_loss(self, attn_list):
        align_loss = None
        for i in range(self.num_step):
            other_step = self.num_step - 1 - i
            cur_dist = self.attn_list[i]
            other_dist = attn_list[other_step].detach()
            if align_loss is None:
                align_loss = self.mse_loss(cur_dist, other_dist)
            else:
                align_loss += self.mse_loss(cur_dist, other_dist)
        return align_loss

    def get_dist_align_loss(self, dist_history):
        align_loss = None
        for i in range(self.num_step - 1):
            forward_pos = i + 1
            backward_pos = self.num_step - 1 - i
            cur_dist = self.dist_history[forward_pos]
            back_dist = dist_history[backward_pos].detach()
            if align_loss is None:
                align_loss = self.mse_loss(cur_dist, back_dist)
            else:
                align_loss += self.mse_loss(cur_dist, back_dist)
        return align_loss

    def get_cotraining_loss(self, target_dist, answer_dist):
        pred_dist = self.dist_history[-1]
        cur_label_dist = target_dist[-1].detach()
        avg_dist = (pred_dist + cur_label_dist) / 2
        loss_merge = self.get_loss_new(pred_dist=avg_dist, answer_dist=answer_dist)
        loss_constraint = self.mse_loss(pred_dist, cur_label_dist)
        return loss_merge, loss_constraint

    def get_constraint_loss(self, target_dist, answer_dist, consider_last=True):
        loss_constraint = None
        label_valid = self.get_label_valid(pred_dist=target_dist[-1], answer_dist=answer_dist, label_f1=self.label_f1)
        if consider_last:
            total_step = self.num_step
        else:
            total_step = self.num_step - 1
        for i in range(total_step):
            pred_dist = self.dist_history[i + 1]
            cur_label_dist = target_dist[i + 1].detach()
            tp_loss = self.get_loss_new(pred_dist=pred_dist, answer_dist=cur_label_dist, reduction='none')
            tp_loss = tp_loss * label_valid
            if self.loss_type == 'bce':
                cur_loss = torch.mean(tp_loss)
            elif self.loss_type == 'kl':
                cur_loss = torch.sum(tp_loss) / answer_dist.size(0)
            else:
                raise NotImplementedError
            if loss_constraint is None:
                loss_constraint = cur_loss
            else:
                loss_constraint += cur_loss
        return loss_constraint

    def calc_loss_basic(self, answer_dist):
        extras = []
        pred_dist = self.dist_history[-1]
        loss = self.get_loss_new(pred_dist, answer_dist)
        extras.append(loss.item())
        if self.entropy_weight > 0:
            ent_loss = None
            for action_prob in self.action_probs:
                dist = torch.distributions.Categorical(probs=action_prob)
                entropy = dist.entropy()
                if ent_loss is None:
                    ent_loss = torch.mean(entropy)
                else:
                    ent_loss += torch.mean(entropy)
            loss = loss + ent_loss * self.entropy_weight
            extras.append(ent_loss.item())
        else:
            extras.append(0.0)
        return loss, extras

    def calc_loss(self, answer_dist, use_label=False, label_dist=None, label_valid=None):
        extras = []
        pred_dist = self.dist_history[-1]
        if use_label and self.num_step > 1:
            label_valid = torch.from_numpy(label_valid).type('torch.FloatTensor')
            main_loss = self.get_loss_new(pred_dist, answer_dist, reduction='none')
            main_loss = main_loss * (1 - label_valid)
            if self.loss_type == 'bce':
                main_loss = torch.mean(main_loss)
            elif self.loss_type == 'kl':
                main_loss = torch.sum(main_loss) / batch_size
            else:
                raise NotImplementedError
            loss_label = self.calc_loss_label(label_dist, label_valid)
            loss = main_loss + loss_label * self.lambda_label
            extras.append(main_loss.item())
            extras.append(loss_label.item())
        else:
            loss = self.get_loss_new(pred_dist, answer_dist)
            extras.append(loss.item())
            extras.append(0.0)
        if self.entropy_weight > 0:
            ent_loss = None
            for action_prob in self.action_probs:
                dist = torch.distributions.Categorical(probs=action_prob)
                entropy = dist.entropy()
                if ent_loss is None:
                    ent_loss = torch.mean(entropy)
                else:
                    ent_loss += torch.mean(entropy)
            loss = loss + ent_loss * self.entropy_weight
            extras.append(ent_loss.item())
        else:
            extras.append(0.0)
        return loss, extras


class BaseReasoning(torch.nn.Module):

    def __init__(self, args, num_entity, num_relation):
        super(BaseReasoning, self).__init__()
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.use_inverse_relation = args['use_inverse_relation']
        self.use_self_loop = args['use_self_loop']
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.num_step = args['num_step']
        self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']
        self.reason_kb = args['reason_kb']
        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)
        self.reset_time = 0

    def share_module_def(self):
        self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

    def build_matrix(self):
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = self.edge_list
        num_fact = len(fact_ids)
        num_relation = self.num_relation
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        self.num_fact = num_fact
        fact2head = torch.LongTensor([batch_heads, fact_ids])
        fact2tail = torch.LongTensor([batch_tails, fact_ids])
        head2fact = torch.LongTensor([fact_ids, batch_heads])
        tail2fact = torch.LongTensor([fact_ids, batch_tails])
        rel2fact = torch.LongTensor([batch_rels + batch_ids * num_relation, fact_ids])
        self.batch_rels = torch.LongTensor(batch_rels)
        self.batch_ids = torch.LongTensor(batch_ids)
        self.batch_heads = torch.LongTensor(batch_heads)
        self.batch_tails = torch.LongTensor(batch_tails)
        val_one = torch.ones_like(self.batch_ids).float()
        self.fact2head_mat = self._build_sparse_tensor(fact2head, val_one, (batch_size * max_local_entity, num_fact))
        self.head2fact_mat = self._build_sparse_tensor(head2fact, val_one, (num_fact, batch_size * max_local_entity))
        self.fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        self.tail2fact_mat = self._build_sparse_tensor(tail2fact, val_one, (num_fact, batch_size * max_local_entity))
        self.rel2fact_mat = self._build_sparse_tensor(rel2fact, val_one, (batch_size * num_relation, num_fact))

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size)


class GNNBackwardReasoning(BaseReasoning):

    def __init__(self, args, num_entity, num_relation):
        super(GNNBackwardReasoning, self).__init__(args, num_entity, num_relation)
        self.share_module_def()
        self.private_module_def()

    def private_module_def(self):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_step):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2 * entity_dim, out_features=entity_dim))

    def reason_layer(self, curr_dist, instruction, rel_linear):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        rel_features = self.rel_features
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.tail2fact_mat, curr_dist.view(-1, 1))
        possible_head = torch.sparse.mm(self.fact2head_mat, fact_prior)
        possible_head = (possible_head > VERY_SMALL_NUMBER).float().view(batch_size, max_local_entity)
        fact_val = fact_val * fact_prior
        f2e_emb = torch.sparse.mm(self.fact2head_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()
        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        return neighbor_rep, possible_head

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, query_entities, query_node_emb):
        batch_size, max_local_entity = local_entity.size()
        self.query_entities = query_entities
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.local_entity_mask = (self.local_entity_mask + self.query_entities > 0).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.build_matrix()

    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        score_func = self.score_func
        relational_ins = relational_ins.squeeze(1)
        neighbor_rep, possible_head = self.reason_layer(current_dist, relational_ins, rel_linear)
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_rep), dim=2)
        self.local_entity_emb = F.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))
        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        if self.reason_kb:
            answer_mask = possible_head * self.local_entity_mask
            score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        else:
            answer_mask = self.local_entity_mask
            score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        if return_score:
            return score_tp, current_dist
        return current_dist

    def forward_all(self, curr_dist, instruction_list):
        dist_history = [curr_dist]
        score_list = []
        for i in range(self.num_step):
            cur_step = self.num_step - 1 - i
            score_tp, curr_dist = self.forward(curr_dist, instruction_list[cur_step], step=cur_step, return_score=True)
            dist_history.insert(0, curr_dist)
            score_list.insert(0, score_tp)
        return dist_history, score_list


class BaseInstruction(torch.nn.Module):

    def __init__(self, args):
        super(BaseInstruction, self).__init__()
        self._parse_args(args)
        self.share_module_def()

    def _parse_args(self, args):
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.q_type = args['q_type']
        self.num_step = args['num_step']
        self.lstm_dropout = args['lstm_dropout']
        self.linear_dropout = args['linear_dropout']
        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)
        self.reset_time = 0

    def share_module_def(self):
        self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return torch.zeros(num_layer, batch_size, hidden_size), torch.zeros(num_layer, batch_size, hidden_size)

    def encode_question(self, *args):
        pass

    def get_instruction(self, *args):
        pass

    @staticmethod
    def get_node_emb(query_hidden_emb, action):
        """

        :param query_hidden_emb: (batch_size, max_hyper, emb)
        :param action: (batch_size)
        :return: (batch_size, 1, emb)
        """
        batch_size, max_hyper, _ = query_hidden_emb.size()
        row_idx = torch.arange(0, batch_size).type(torch.LongTensor)
        q_rep = query_hidden_emb[row_idx, action, :]
        return q_rep.unsqueeze(1)


class LSTMInstruction(BaseInstruction):

    def __init__(self, args, word_embedding, num_word):
        super(LSTMInstruction, self).__init__(args)
        self.word_embedding = word_embedding
        self.num_word = num_word
        self.encoder_def()
        entity_dim = self.entity_dim
        self.cq_linear = nn.Linear(in_features=2 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_step):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))

    def encoder_def(self):
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim, batch_first=True, bidirectional=False)

    def encode_question(self, query_text):
        batch_size = query_text.size(0)
        query_word_emb = self.word_embedding(query_text)
        query_hidden_emb, (h_n, c_n) = self.node_encoder(self.lstm_drop(query_word_emb), self.init_hidden(1, batch_size, self.entity_dim))
        self.instruction_hidden = h_n
        self.instruction_mem = c_n
        self.query_node_emb = h_n.squeeze(dim=0).unsqueeze(dim=1)
        self.query_hidden_emb = query_hidden_emb
        self.query_mask = (query_text != self.num_word).float()
        return query_hidden_emb, self.query_node_emb

    def init_reason(self, query_text):
        batch_size = query_text.size(0)
        self.encode_question(query_text)
        self.relational_ins = torch.zeros(batch_size, self.entity_dim)
        self.instructions = []
        self.attn_list = []

    def get_instruction(self, relational_ins, step=0, query_node_emb=None):
        query_hidden_emb = self.query_hidden_emb
        query_mask = self.query_mask
        if query_node_emb is None:
            query_node_emb = self.query_node_emb
        relational_ins = relational_ins.unsqueeze(1)
        question_linear = getattr(self, 'question_linear' + str(step))
        q_i = question_linear(self.linear_drop(query_node_emb))
        cq = self.cq_linear(self.linear_drop(torch.cat((relational_ins, q_i), dim=-1)))
        ca = self.ca_linear(self.linear_drop(cq * query_hidden_emb))
        attn_weight = F.softmax(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1)
        relational_ins = torch.sum(attn_weight * query_hidden_emb, dim=1)
        return relational_ins, attn_weight

    def forward(self, query_text):
        self.init_reason(query_text)
        for i in range(self.num_step):
            relational_ins, attn_weight = self.get_instruction(self.relational_ins, step=i)
            self.instructions.append(relational_ins)
            self.attn_list.append(attn_weight)
            self.relational_ins = relational_ins
        return self.instructions, self.attn_list


class BackwardReasonModel(BaseModel):

    def __init__(self, args, num_entity, num_relation, num_word, forward_model=None):
        """
        num_relation: number of relation including self-connection
        """
        super(BackwardReasonModel, self).__init__(args, num_entity, num_relation, num_word)
        share_embedding = args['share_embedding']
        share_instruction = args['share_instruction']
        if share_embedding:
            self.share_embedding(forward_model)
        else:
            self.embedding_def()
            self.share_module_def()
        if share_instruction:
            self.instruction = forward_model.instruction
        else:
            self.instruction_def(args)
        self.reasoning_def(args, num_entity, num_relation)
        self.loss_type = args['loss_type']
        self

    def instruction_def(self, args):
        self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)

    def reasoning_def(self, args, num_entity, num_relation):
        self.reasoning = GNNBackwardReasoning(args, num_entity, num_relation)

    def share_embedding(self, model):
        self.relation_embedding = model.relation_embedding
        self.word_embedding = model.word_embedding
        self.type_layer = model.type_layer
        self.entity_linear = model.entity_linear
        self.relation_linear = model.relation_linear
        self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        self.query_node_emb = self.instruction.query_node_emb
        self.rel_features = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, self.rel_features)
        self.curr_dist = curr_dist
        self.dist_history = [curr_dist]
        self.action_probs = []
        self.reasoning.init_reason(local_entity=local_entity, kb_adj_mat=kb_adj_mat, local_entity_emb=self.local_entity_emb, rel_features=self.rel_features, query_entities=query_entities, query_node_emb=self.query_node_emb)

    def get_loss_constraint(self, forewad_dist, backward_dist):
        log_prob = torch.log(forewad_dist + 1e-08)
        loss = torch.mean(-(log_prob * backward_dist.detach()))
        return loss

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def label_data(self, batch):
        middle_dist = []
        self.model(batch, training=False)
        self.back_model(batch, training=False)
        forward_history = self.model.dist_history
        backward_history = self.back_model.dist_history
        for i in range(self.num_step - 1):
            middle_dist.append(forward_history[i + 1] + backward_history[i + 1] / 2)
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
        pred_dist = self.model.dist_history[-1]
        label_valid = self.model.get_label_valid(pred_dist, answer_dist, label_f1=self.label_f1)
        return middle_dist, label_valid

    def forward(self, batch, training=False):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity, kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        answer_len = torch.sum(answer_dist, dim=1, keepdim=True)
        answer_len[answer_len == 0] = 1.0
        answer_prob = answer_dist.div(answer_len)
        dist_history, score_list = self.reasoning.forward_all(curr_dist=answer_prob, instruction_list=self.instruction_list)
        self.dist_history = dist_history
        self.score_list = score_list
        pred_dist = dist_history[0]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=query_entities, label_valid=case_valid)
        extras = [loss.item(), 0.0, 0.0]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, query_entities)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, extras, pred_dist, tp_list


class GNNReasoning(BaseReasoning):

    def __init__(self, args, num_entity, num_relation):
        super(GNNReasoning, self).__init__(args, num_entity, num_relation)
        self.share_module_def()
        self.private_module_def()

    def private_module_def(self):
        entity_dim = self.entity_dim
        self.softmax_d1 = nn.Softmax(dim=1)
        self.sigmoid_d1 = nn.Sigmoid()
        self.score_func = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_step):
            self.add_module('rel_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2 * entity_dim, out_features=entity_dim))

    def reason_layer(self, curr_dist, instruction, rel_linear):
        batch_size = self.batch_size
        max_local_entity = self.max_local_entity
        rel_features = self.rel_features
        fact_rel = torch.index_select(rel_features, dim=0, index=self.batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=self.batch_ids)
        fact_val = F.relu(rel_linear(fact_rel) * fact_query)
        fact_prior = torch.sparse.mm(self.head2fact_mat, curr_dist.view(-1, 1))
        possible_tail = torch.sparse.mm(self.fact2tail_mat, fact_prior)
        possible_tail = (possible_tail > VERY_SMALL_NUMBER).float().view(batch_size, max_local_entity)
        fact_val = fact_val * fact_prior
        f2e_emb = torch.sparse.mm(self.fact2tail_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()
        neighbor_rep = f2e_emb.view(batch_size, max_local_entity, self.entity_dim)
        return neighbor_rep, possible_tail

    def init_reason(self, local_entity, kb_adj_mat, local_entity_emb, rel_features, query_node_emb=None):
        batch_size, max_local_entity = local_entity.size()
        self.local_entity_mask = (local_entity != self.num_entity).float()
        self.batch_size = batch_size
        self.max_local_entity = max_local_entity
        self.edge_list = kb_adj_mat
        self.rel_features = rel_features
        self.local_entity_emb = local_entity_emb
        self.num_relation = self.rel_features.size(0)
        self.possible_cand = []
        self.build_matrix()

    def forward(self, current_dist, relational_ins, step=0, return_score=False):
        rel_linear = getattr(self, 'rel_linear' + str(step))
        e2e_linear = getattr(self, 'e2e_linear' + str(step))
        score_func = self.score_func
        relational_ins = relational_ins.squeeze(1)
        neighbor_rep, possible_tail = self.reason_layer(current_dist, relational_ins, rel_linear)
        next_local_entity_emb = torch.cat((self.local_entity_emb, neighbor_rep), dim=2)
        self.local_entity_emb = F.relu(e2e_linear(self.linear_drop(next_local_entity_emb)))
        score_tp = score_func(self.linear_drop(self.local_entity_emb)).squeeze(dim=2)
        if self.reason_kb:
            answer_mask = self.local_entity_mask * possible_tail
        else:
            answer_mask = self.local_entity_mask
        self.possible_cand.append(answer_mask)
        score_tp = score_tp + (1 - answer_mask) * VERY_NEG_NUMBER
        current_dist = self.softmax_d1(score_tp)
        if return_score:
            return score_tp, current_dist
        return current_dist

    def forward_all(self, curr_dist, instruction_list):
        dist_history = [curr_dist]
        score_list = []
        for i in range(self.num_step):
            score_tp, curr_dist = self.forward(curr_dist, instruction_list[i], step=i, return_score=True)
            score_list.append(score_tp)
            dist_history.append(curr_dist)
        return dist_history, score_list


class GNNModel(BaseModel):

    def __init__(self, args, num_entity, num_relation, num_word):
        """
        num_relation: number of relation including self-connection
        """
        super(GNNModel, self).__init__(args, num_entity, num_relation, num_word)
        self.embedding_def()
        self.share_module_def()
        self.private_module_def(args, num_entity, num_relation)
        self.loss_type = args['loss_type']
        self.model_name = args['model_name'].lower()
        self.lambda_label = args['lambda_label']
        self.filter_label = args['filter_label']
        self

    def insert_relation_tokens(self, relation_tokens):
        self.relation_tokens = relation_tokens

    def private_module_def(self, args, num_entity, num_relation):
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        self.relation_tokens = None
        self.reasoning = GNNReasoning(args, num_entity, num_relation)
        self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input):
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        self.rel_features = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, self.rel_features)
        self.curr_dist = curr_dist
        self.dist_history = [curr_dist]
        self.action_probs = []
        self.reasoning.init_reason(local_entity=local_entity, kb_adj_mat=kb_adj_mat, local_entity_emb=self.local_entity_emb, rel_features=self.rel_features)

    def one_step(self, num_step):
        relational_ins = self.instruction_list[num_step]
        self.curr_dist = self.reasoning(self.curr_dist, relational_ins, step=num_step)
        self.dist_history.append(self.curr_dist)

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def train_batch(self, batch, middle_dist, label_valid=None):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity, kb_adj_mat=kb_adj_mat, q_input=q_input)
        for i in range(self.num_step):
            self.one_step(num_step=i)
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        main_loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        distill_loss = None
        for i in range(self.num_step - 1):
            curr_dist = self.dist_history[i + 1]
            teacher_dist = middle_dist[i].squeeze(1).detach()
            if self.filter_label:
                assert not label_valid is None
                tp_label_loss = self.calc_loss_label(curr_dist=curr_dist, teacher_dist=teacher_dist, label_valid=label_valid)
            else:
                tp_label_loss = self.calc_loss_label(curr_dist=curr_dist, teacher_dist=teacher_dist, label_valid=case_valid)
            if distill_loss is None:
                distill_loss = tp_label_loss
            else:
                distill_loss += tp_label_loss
        extras = [main_loss.item(), distill_loss.item()]
        loss = main_loss + distill_loss * self.lambda_label
        h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
        tp_list = [h1.tolist(), f1.tolist()]
        return loss, extras, pred_dist, tp_list

    def forward(self, batch, training=False):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity, kb_adj_mat=kb_adj_mat, q_input=q_input)
        for i in range(self.num_step):
            self.one_step(num_step=i)
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list


class NsmAgent(BaseAgent):

    def __init__(self, args, logger, num_entity, num_relation, num_word):
        super(NsmAgent, self).__init__(args, logger, num_entity, num_relation, num_word)
        self.q_type = 'seq'
        model_name = args['model_name'].lower()
        self.label_f1 = args['label_f1']
        self.model_name = model_name
        if model_name.startswith('gnn'):
            self.model = GNNModel(args, num_entity, num_relation, num_word)
        elif model_name.startswith('back'):
            self.model = BackwardReasonModel(args, num_entity, num_relation, num_word)
        else:
            raise NotImplementedError

    def insert_relation_tokens_to_model(self, relation_tokens):
        self.model.insert_relation_tokens(relation_tokens)

    def forward(self, batch, training=False):
        batch = self.deal_input(batch)
        return self.model(batch, training=training)

    def label_data(self, batch):
        batch = self.deal_input(batch)
        middle_dist = []
        self.model(batch, training=False)
        forward_history = self.model.dist_history
        for i in range(self.num_step - 1):
            middle_dist.append(forward_history[i + 1])
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
        if self.model_name == 'back':
            pred_dist = self.model.dist_history[0]
            label_valid = self.model.get_label_valid(pred_dist, query_entities, label_f1=self.label_f1)
        else:
            pred_dist = self.model.dist_history[-1]
            label_valid = self.model.get_label_valid(pred_dist, answer_dist, label_f1=self.label_f1)
        return middle_dist, label_valid

    def train_batch(self, batch, middle_dist, label_valid=None):
        batch = self.deal_input(batch)
        return self.model.train_batch(batch, middle_dist, label_valid)

    def deal_input(self, batch):
        return self.deal_input_seq(batch)


class HybridModel(BaseModel):

    def __init__(self, args, num_entity, num_relation, num_word):
        """
        num_relation: number of relation including self-connection
        """
        super(HybridModel, self).__init__(args, num_entity, num_relation, num_word)
        self.embedding_def()
        self.share_module_def()
        self.private_module_def(args, num_entity, num_relation)
        self.loss_type = 'kl'
        self.model_name = args['model_name'].lower()
        self.lambda_back = args['lambda_back']
        self.lambda_constrain = args['lambda_constrain']
        self.constrain_type = args['constrain_type']
        self.constraint_loss = torch.nn.MSELoss(reduction='none')
        self.kld_loss_1 = nn.KLDivLoss(reduction='none')
        self.num_step = args['num_step']
        self

    def reasoning_def(self, args, num_entity, num_relation):
        self.reasoning = GNNReasoning(args, num_entity, num_relation)
        self.back_reasoning = GNNBackwardReasoning(args, num_entity, num_relation)

    def instruction_def(self, args):
        self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)

    def private_module_def(self, args, num_entity, num_relation):
        self.instruction_def(args)
        self.reasoning_def(args, num_entity, num_relation)
        self.constraint_loss = torch.nn.MSELoss()

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input):
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        self.query_node_emb = self.instruction.query_node_emb
        self.rel_features = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, self.rel_features)
        self.curr_dist = curr_dist
        self.dist_history = [curr_dist]
        self.action_probs = []
        self.reasoning.init_reason(local_entity=local_entity, kb_adj_mat=kb_adj_mat, local_entity_emb=self.local_entity_emb, rel_features=self.rel_features, query_node_emb=self.query_node_emb)

    def one_step(self, num_step):
        relational_ins = self.instruction_list[num_step]
        self.curr_dist = self.reasoning(self.curr_dist, relational_ins, step=num_step)
        self.dist_history.append(self.curr_dist)

    def get_js_div(self, dist_1, dist_2):
        mean_dist = (dist_1 + dist_2) / 2
        log_mean_dist = torch.log(mean_dist + 1e-08)
        loss = 0.5 * (self.kld_loss_1(log_mean_dist, dist_1) + self.kld_loss_1(log_mean_dist, dist_2))
        return loss

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def calc_loss_backward(self, case_valid):
        back_loss = None
        constrain_loss = None
        for i in range(self.num_step):
            forward_dist = self.dist_history[i]
            backward_dist = self.backward_history[i]
            if i == 0:
                back_loss = self.calc_loss_label(curr_dist=backward_dist, teacher_dist=forward_dist, label_valid=case_valid)
            else:
                tp_loss = self.get_js_div(forward_dist, backward_dist)
                tp_loss = torch.sum(tp_loss * case_valid) / forward_dist.size(0)
                if constrain_loss is None:
                    constrain_loss = tp_loss
                else:
                    constrain_loss += tp_loss
        return back_loss, constrain_loss

    def label_data(self, batch):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity, kb_adj_mat=kb_adj_mat, q_input=q_input)
        self.dist_history, score_list = self.reasoning.forward_all(current_dist, self.instruction_list)
        final_emb = self.reasoning.local_entity_emb
        self.back_reasoning.init_reason(local_entity=local_entity, kb_adj_mat=kb_adj_mat, local_entity_emb=final_emb, rel_features=self.rel_features, query_entities=query_entities, query_node_emb=self.query_node_emb)
        answer_len = torch.sum(answer_dist, dim=1, keepdim=True)
        answer_len[answer_len == 0] = 1.0
        answer_prob = answer_dist.div(answer_len)
        self.backward_history, back_score_list = self.back_reasoning.forward_all(answer_prob, self.instruction_list)
        middle_dist = []
        for i in range(self.num_step - 1):
            mix_dist = (self.dist_history[i + 1] + self.backward_history[i + 1]) / 2
            middle_dist.append(mix_dist)
        return middle_dist

    def forward(self, batch, training=False):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity, kb_adj_mat=kb_adj_mat, q_input=q_input)
        self.dist_history, score_list = self.reasoning.forward_all(current_dist, self.instruction_list)
        extras = []
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        extras.append(loss.item())
        if training:
            final_emb = self.reasoning.local_entity_emb
            self.back_reasoning.init_reason(local_entity=local_entity, kb_adj_mat=kb_adj_mat, local_entity_emb=final_emb, rel_features=self.rel_features, query_entities=query_entities, query_node_emb=self.query_node_emb)
            answer_len = torch.sum(answer_dist, dim=1, keepdim=True)
            answer_len[answer_len == 0] = 1.0
            answer_prob = answer_dist.div(answer_len)
            self.backward_history, back_score_list = self.back_reasoning.forward_all(answer_prob, self.instruction_list)
            back_loss, constrain_loss = self.calc_loss_backward(case_valid)
            extras.append(back_loss.item())
            extras.append(constrain_loss.item())
            loss = loss + self.lambda_back * back_loss + self.lambda_constrain * constrain_loss
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, extras, pred_dist, tp_list


class TeacherAgent_hybrid(BaseAgent):

    def __init__(self, args, logger, num_entity, num_relation, num_word):
        super(TeacherAgent_hybrid, self).__init__(args, logger, num_entity, num_relation, num_word)
        self.q_type = args['q_type']
        self.label_f1 = args['label_f1']
        self.model = HybridModel(args, num_entity, num_relation, num_word)

    def forward(self, batch, training=False):
        batch = self.deal_input(batch)
        return self.model(batch, training=training)

    def label_data(self, batch):
        batch = self.deal_input(batch)
        middle_dist = self.model.label_data(batch)
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
        pred_dist = self.model.dist_history[-1]
        label_valid = self.model.get_label_valid(pred_dist, answer_dist, label_f1=self.label_f1)
        return middle_dist, label_valid

    def deal_input(self, batch):
        return self.deal_input_seq(batch)


class ForwardReasonModel(BaseModel):

    def __init__(self, args, num_entity, num_relation, num_word):
        """
        num_relation: number of relation including self-connection
        """
        super(ForwardReasonModel, self).__init__(args, num_entity, num_relation, num_word)
        self.embedding_def()
        self.share_module_def()
        self.instruction_def(args)
        self.reasoning_def(args, num_entity, num_relation)
        self.loss_type = args['loss_type']
        self.model_name = args['model_name'].lower()
        self.lambda_label = args['lambda_label']
        self.filter_label = args['filter_label']
        self

    def instruction_def(self, args):
        self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)

    def reasoning_def(self, args, num_entity, num_relation):
        self.reasoning = GNNReasoning(args, num_entity, num_relation)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input):
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        query_node_emb = self.instruction.query_node_emb
        self.rel_features = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, self.rel_features)
        self.curr_dist = curr_dist
        self.dist_history = [curr_dist]
        self.action_probs = []
        self.reasoning.init_reason(local_entity=local_entity, kb_adj_mat=kb_adj_mat, local_entity_emb=self.local_entity_emb, rel_features=self.rel_features, query_node_emb=query_node_emb)

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def train_batch(self, batch, middle_dist, label_valid=None):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity, kb_adj_mat=kb_adj_mat, q_input=q_input)
        for i in range(self.num_step):
            self.one_step(num_step=i)
        pred_dist = self.dist_history[-1]
        main_loss = self.get_loss_new(pred_dist, answer_dist)
        distill_loss = None
        for i in range(self.num_step - 1):
            curr_dist = self.dist_history[i + 1]
            teacher_dist = middle_dist[i].squeeze(1).detach()
            if self.filter_label:
                assert not label_valid is None
                tp_label_loss = self.calc_loss_label(curr_dist=curr_dist, teacher_dist=teacher_dist, label_valid=label_valid)
            else:
                tp_label_loss = self.get_loss_new(curr_dist, teacher_dist)
            if distill_loss is None:
                distill_loss = tp_label_loss
            else:
                distill_loss += tp_label_loss
        extras = [main_loss.item(), distill_loss.item()]
        loss = main_loss + distill_loss * self.lambda_label
        h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
        tp_list = [h1.tolist(), f1.tolist()]
        return loss, extras, pred_dist, tp_list

    def one_step(self, num_step):
        relational_ins = self.instruction_list[num_step]
        self.curr_dist = self.reasoning(self.curr_dist, relational_ins, step=num_step)
        self.dist_history.append(self.curr_dist)

    def forward(self, batch, training=False):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity, kb_adj_mat=kb_adj_mat, q_input=q_input)
        for i in range(self.num_step):
            self.one_step(num_step=i)
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list


class TeacherAgent_parallel(BaseAgent):

    def __init__(self, args, logger, num_entity, num_relation, num_word):
        super(TeacherAgent_parallel, self).__init__(args, logger, num_entity, num_relation, num_word)
        self.q_type = args['q_type']
        self.label_f1 = args['label_f1']
        self.model = ForwardReasonModel(args, num_entity, num_relation, num_word)
        self.back_model = BackwardReasonModel(args, num_entity, num_relation, num_word, self.model)
        self.lambda_back = args['lambda_back']
        self.lambda_constrain = args['lambda_constrain']
        self.constrain_type = args['constrain_type']
        self.constraint_loss = torch.nn.MSELoss(reduction='none')
        self.kld_loss_1 = nn.KLDivLoss(reduction='none')
        self.num_step = args['num_step']

    def get_js_div(self, dist_1, dist_2):
        mean_dist = (dist_1 + dist_2) / 2
        log_mean_dist = torch.log(mean_dist + 1e-08)
        loss = 0.5 * (self.kld_loss_1(log_mean_dist, dist_1) + self.kld_loss_1(log_mean_dist, dist_2))
        return loss

    def get_kl_div(self, dist_1, dist_2):
        log_dist_1 = torch.log(dist_1 + 1e-08)
        log_dist_2 = torch.log(dist_2 + 1e-08)
        loss = 0.5 * (self.kld_loss_1(log_dist_1, dist_2) + self.kld_loss_1(log_dist_2, dist_1))
        return loss

    def get_constraint_loss(self, forward_dist, backward_dist, case_valid):
        loss_constraint = None
        for i in range(self.num_step - 1):
            cur_forward_dist = forward_dist[i + 1]
            cur_backward_dist = backward_dist[i + 1]
            tp_loss = self.get_js_div(cur_forward_dist, cur_backward_dist)
            tp_loss = torch.sum(tp_loss * case_valid) / cur_forward_dist.size(0)
            if loss_constraint is None:
                loss_constraint = tp_loss
            else:
                loss_constraint += tp_loss
        return loss_constraint

    def forward(self, batch, training=False):
        batch = self.deal_input(batch)
        loss, pred, pred_dist, tp_list = self.model(batch, training=training)
        extras = [loss.item()]
        if training:
            current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
            answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
            case_valid = (answer_number > 0).float()
            back_loss, _, _, _ = self.back_model(batch, training=training)
            forward_history = self.model.dist_history
            backward_history = self.back_model.dist_history
            constrain_loss = self.get_constraint_loss(forward_history, backward_history, case_valid)
            loss = loss + self.lambda_back * back_loss + self.lambda_constrain * constrain_loss
            extras.append(back_loss.item())
            extras.append(constrain_loss.item())
        return loss, extras, pred_dist, tp_list

    def label_data(self, batch):
        batch = self.deal_input(batch)
        middle_dist = []
        self.model(batch, training=False)
        self.back_model(batch, training=False)
        forward_history = self.model.dist_history
        backward_history = self.back_model.dist_history
        for i in range(self.num_step - 1):
            middle_dist.append((forward_history[i + 1] + backward_history[i + 1]) / 2)
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, local_entity, query_entities, true_batch_id = batch
        pred_dist = self.model.dist_history[-1]
        label_valid = self.model.get_label_valid(pred_dist, answer_dist, label_f1=self.label_f1)
        return middle_dist, label_valid

    def deal_input(self, batch):
        return self.deal_input_seq(batch)


class STLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, linear_drop, device):
        super(STLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear_drop = linear_drop
        self.kb_self_linear = nn.Linear(in_features, out_features)
        self.device = device

    def forward(self, input_vector, edge_list, curr_dist, instruction, rel_features):
        """
        input_vector: (batch_size, max_local_entity, hidden_size)
        curr_dist: (batch_size, max_local_entity)
        instruction: (batch_size, hidden_size)
        """
        batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list = edge_list
        num_fact = len(fact_ids)
        batch_size, max_local_entity, hidden_size = input_vector.size()
        fact2tail = torch.LongTensor([batch_tails, fact_ids])
        head2fact = torch.LongTensor([fact_ids, batch_heads])
        batch_rels = torch.LongTensor(batch_rels)
        batch_ids = torch.LongTensor(batch_ids)
        val_one = torch.ones_like(batch_ids).float()
        fact_rel = torch.index_select(rel_features, dim=0, index=batch_rels)
        fact_query = torch.index_select(instruction, dim=0, index=batch_ids)
        fact_val = F.relu(self.kb_self_linear(fact_rel) * fact_query)
        head2fact_mat = self._build_sparse_tensor(head2fact, val_one, (num_fact, batch_size * max_local_entity))
        fact2tail_mat = self._build_sparse_tensor(fact2tail, val_one, (batch_size * max_local_entity, num_fact))
        fact_prior = torch.sparse.mm(head2fact_mat, curr_dist.view(-1, 1))
        fact_val = fact_val * fact_prior
        f2e_emb = torch.sparse.mm(fact2tail_mat, fact_val)
        assert not torch.isnan(f2e_emb).any()
        f2e_emb = f2e_emb.view(batch_size, max_local_entity, hidden_size)
        return f2e_emb

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size)


class Attn(nn.Module):

    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs, query_mask):
        """
        :param hidden: (B, 1, d)
        :param encoder_outputs: (B, ML_Q, d)
        :param query_mask: (B, ML_Q, 1)
        :return: attn_weight (B, ML_Q, 1)
        """
        batch_size = hidden.size(0)
        max_len = encoder_outputs.size(1)
        H = hidden.expand(batch_size, max_len, self.hidden_size)
        att_energies = torch.tanh(self.score(H, encoder_outputs)) + (1 - query_mask) * VERY_NEG_NUMBER
        return F.softmax(att_energies, dim=1)

    def score(self, hidden, encoder_outputs):
        batch_size, max_len, hidden_size = encoder_outputs.size()
        energy = self.attn(torch.cat([hidden, encoder_outputs], -1))
        energy = energy.view(-1, self.hidden_size)
        v = self.v.unsqueeze(1)
        energy = energy.mm(v)
        att_energies = energy.view(batch_size, max_len, 1)
        return att_energies


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
    (Attn,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
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

class Test_RUCKBReasoning_SubgraphRetrievalKBQA(_paritybench_base):
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

