import sys
_module = sys.modules[__name__]
del sys
configs = _module
configs_base = _module
configs_data = _module
configs_inference = _module
protenix = _module
config = _module
extend_types = _module
data = _module
ccd = _module
constants = _module
data_pipeline = _module
dataloader = _module
dataset = _module
featurizer = _module
filter = _module
infer_data_pipeline = _module
json_maker = _module
json_parser = _module
json_to_feature = _module
msa_featurizer = _module
msa_utils = _module
parser = _module
substructure_perms = _module
tokenizer = _module
utils = _module
metrics = _module
clash = _module
lddt_metrics = _module
rmsd = _module
model = _module
generator = _module
layer_norm = _module
layer_norm = _module
torch_ext_compile = _module
loss = _module
modules = _module
confidence = _module
diffusion = _module
embedders = _module
frames = _module
head = _module
pairformer = _module
primitives = _module
transformer = _module
protenix = _module
sample_confidence = _module
utils = _module
openfold_local = _module
data_transforms = _module
errors = _module
mmcif_parsing = _module
msa_identifiers = _module
msa_pairing = _module
parsers = _module
templates = _module
tools = _module
jackhmmer = _module
dropout = _module
outer_product_mean = _module
primitives = _module
triangular_attention = _module
triangular_multiplicative_update = _module
np = _module
residue_constants = _module
all_atom_multimer = _module
checkpointing = _module
chunk_utils = _module
feats = _module
geometry = _module
quat_rigid = _module
rigid_matrix_vector = _module
rotation_matrix = _module
test_utils = _module
vector = _module
kernel = _module
attention_core = _module
precision_utils = _module
rigid_utils = _module
tensor_utils = _module
cropping = _module
distributed = _module
file_io = _module
logger = _module
lr_scheduler = _module
metrics = _module
permutation = _module
atom_permutation = _module
chain_permutation = _module
heuristic = _module
pocket_based_permutation = _module
utils = _module
permutation = _module
utils = _module
scatter_utils = _module
seed = _module
torch_utils = _module
training = _module
web_service = _module
colab_request_parser = _module
colab_request_utils = _module
dependency_url = _module
prediction_visualization = _module
viewer = _module
runner = _module
batch_inference = _module
dumper = _module
ema = _module
inference = _module
msa_search = _module
train = _module
scripts = _module
colabfold_msa = _module
gen_ccd_cache = _module
prepare_training_data = _module
setup = _module

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


import logging


from collections import defaultdict


from typing import Any


from typing import Optional


from typing import Union


import numpy as np


import pandas as pd


import torch


import math


from typing import Iterator


from typing import Sequence


import torch.distributed as dist


from torch.utils.data import DataLoader


from torch.utils.data import DistributedSampler


from torch.utils.data import Sampler


import random


from copy import deepcopy


from typing import Callable


from torch.utils.data import Dataset


import copy


from sklearn.neighbors import KDTree


import time


import warnings


from typing import Mapping


from abc import ABC


from abc import abstractmethod


import functools


import re


import torch.nn as nn


import numbers


from torch.nn.parameter import Parameter


from torch.utils.cpp_extension import load


import torch.nn.functional as F


from torch.nn import Linear


from functools import partial


from scipy.spatial.transform import Rotation


from functools import partialmethod


from typing import List


from typing import Tuple


from scipy.stats import truncnorm


from typing import Dict


from typing import Text


import torch.utils.checkpoint


from functools import reduce


from functools import lru_cache


from scipy.spatial.distance import cdist


from torch.optim.lr_scheduler import LRScheduler


from torch import nn


import inspect


from torch.nn.parallel import DistributedDataParallel as DDP


ID2TYPE = {(0): 'UNK', (1): 'lig', (2): 'prot', (3): 'dna', (4): 'rna'}


rdkit_vdws = [1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54, 2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88, 2.8, 2.4, 2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1, 2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3, 2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25, 2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.45, 2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3, 2.28, 2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05, 2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4, 2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]


RDKIT_VDWS = torch.tensor(rdkit_vdws)


def get_vdw_radii(elements_one_hot):
    """get vdw radius for each atom according to their elements"""
    element_order = elements_one_hot.argmax(dim=1)
    return RDKIT_VDWS[element_order]


class Clash(nn.Module):

    def __init__(self, af3_clash_threshold=1.1, vdw_clash_threshold=0.75, compute_af3_clash=True, compute_vdw_clash=True):
        super().__init__()
        self.af3_clash_threshold = af3_clash_threshold
        self.vdw_clash_threshold = vdw_clash_threshold
        self.compute_af3_clash = compute_af3_clash
        self.compute_vdw_clash = compute_vdw_clash

    def forward(self, pred_coordinate, asym_id, atom_to_token_idx, is_ligand, is_protein, is_dna, is_rna, mol_id: 'Optional[torch.Tensor]'=None, elements_one_hot: 'Optional[torch.Tensor]'=None):
        chain_info = self.get_chain_info(asym_id=asym_id, atom_to_token_idx=atom_to_token_idx, is_ligand=is_ligand, is_protein=is_protein, is_dna=is_dna, is_rna=is_rna, mol_id=mol_id, elements_one_hot=elements_one_hot)
        return self._check_clash_per_chain_pairs(pred_coordinate=pred_coordinate, **chain_info)

    def get_chain_info(self, asym_id, atom_to_token_idx, is_ligand, is_protein, is_dna, is_rna, mol_id: 'Optional[torch.Tensor]'=None, elements_one_hot: 'Optional[torch.Tensor]'=None):
        asym_id = asym_id.long()
        asym_id_to_asym_mask = {aid.item(): (asym_id == aid) for aid in torch.unique(asym_id)}
        N_chains = len(asym_id_to_asym_mask)
        assert N_chains == asym_id.max() + 1
        chain_types = []
        mol_id_to_asym_ids, asym_id_to_mol_id = {}, {}
        atom_type = (1 * is_ligand + 2 * is_protein + 3 * is_dna + 4 * is_rna).long()
        if self.compute_vdw_clash:
            assert mol_id is not None
            assert elements_one_hot is not None
        for aid in range(N_chains):
            atom_chain_mask = asym_id_to_asym_mask[aid][atom_to_token_idx]
            atom_type_i = atom_type[atom_chain_mask]
            assert len(atom_type_i.unique()) == 1
            if atom_type_i[0].item() == 0:
                logging.warning('Unknown asym_id type: not in ligand / protein / dna / rna')
            chain_types.append(ID2TYPE[atom_type_i[0].item()])
            if self.compute_vdw_clash:
                mol_id_i = mol_id[atom_chain_mask].unique().item()
                mol_id_to_asym_ids.setdefault(mol_id_i, []).append(aid)
                asym_id_to_mol_id[aid] = mol_id_i
        chain_info = {'N_chains': N_chains, 'atom_to_token_idx': atom_to_token_idx, 'asym_id_to_asym_mask': asym_id_to_asym_mask, 'atom_type': atom_type, 'mol_id': mol_id, 'elements_one_hot': elements_one_hot, 'chain_types': chain_types}
        if self.compute_vdw_clash:
            chain_info.update({'asym_id_to_mol_id': asym_id_to_mol_id})
        return chain_info

    def get_chain_pair_violations(self, pred_coordinate, violation_type, chain_1_mask, chain_2_mask, elements_one_hot: 'Optional[torch.Tensor]'=None):
        chain_1_coords = pred_coordinate[chain_1_mask, :]
        chain_2_coords = pred_coordinate[chain_2_mask, :]
        pred_dist = torch.cdist(chain_1_coords, chain_2_coords)
        if violation_type == 'af3':
            clash_per_atom_pair = pred_dist < self.af3_clash_threshold
            clashed_col, clashed_row = torch.where(clash_per_atom_pair)
            clash_atom_pairs = torch.stack((clashed_col, clashed_row), dim=-1)
        else:
            assert elements_one_hot is not None
            vdw_radii_i, vdw_radii_j = get_vdw_radii(elements_one_hot[chain_1_mask, :]), get_vdw_radii(elements_one_hot[chain_2_mask, :])
            vdw_sum_pair = vdw_radii_i[:, None] + vdw_radii_j[None, :]
            relative_vdw_distance = pred_dist / vdw_sum_pair
            clash_per_atom_pair = relative_vdw_distance < self.vdw_clash_threshold
            clashed_col, clashed_row = torch.where(clash_per_atom_pair)
            clash_rel_dist = relative_vdw_distance[clashed_col, clashed_row]
            clashed_global_col = torch.where(chain_1_mask)[0][clashed_col]
            clashed_global_row = torch.where(chain_2_mask)[0][clashed_row]
            clash_atom_pairs = torch.stack((clashed_global_col, clashed_global_row, clash_rel_dist), dim=-1)
        return clash_atom_pairs

    def _check_clash_per_chain_pairs(self, pred_coordinate, atom_to_token_idx, N_chains, atom_type, chain_types, elements_one_hot, asym_id_to_asym_mask, mol_id: 'Optional[torch.Tensor]'=None, asym_id_to_mol_id: 'Optional[torch.Tensor]'=None):
        device = pred_coordinate.device
        N_sample = pred_coordinate.shape[0]
        if self.compute_af3_clash:
            has_af3_clash_flag = torch.zeros(N_sample, N_chains, N_chains, device=device, dtype=torch.bool)
            af3_clash_details = torch.zeros(N_sample, N_chains, N_chains, 2, device=device, dtype=torch.bool)
        if self.compute_vdw_clash:
            has_vdw_clash_flag = torch.zeros(N_sample, N_chains, N_chains, device=device, dtype=torch.bool)
            vdw_clash_details = {}
        skipped_pairs = []
        for sample_id in range(N_sample):
            for i in range(N_chains):
                if chain_types[i] == 'UNK':
                    continue
                atom_chain_mask_i = asym_id_to_asym_mask[i][atom_to_token_idx]
                N_chain_i = torch.sum(atom_chain_mask_i).item()
                for j in range(i + 1, N_chains):
                    if chain_types[j] == 'UNK':
                        continue
                    chain_pair_type = set([chain_types[i], chain_types[j]])
                    skip_bonded_ligand = False
                    if self.compute_vdw_clash and 'lig' in chain_pair_type and len(chain_pair_type) > 1 and asym_id_to_mol_id[i] == asym_id_to_mol_id[j]:
                        common_mol_id = asym_id_to_mol_id[i]
                        logging.warning(f'mol_id {common_mol_id} may contain bonded ligand to polymers')
                        skip_bonded_ligand = True
                        skipped_pairs.append((i, j))
                    atom_chain_mask_j = asym_id_to_asym_mask[j][atom_to_token_idx]
                    N_chain_j = torch.sum(atom_chain_mask_j).item()
                    if self.compute_vdw_clash and not skip_bonded_ligand:
                        vdw_clash_pairs = self.get_chain_pair_violations(pred_coordinate=pred_coordinate[sample_id, :, :], violation_type='vdw', chain_1_mask=atom_chain_mask_i, chain_2_mask=atom_chain_mask_j, elements_one_hot=elements_one_hot)
                        if vdw_clash_pairs.shape[0] > 0:
                            vdw_clash_details[sample_id, i, j] = vdw_clash_pairs
                            has_vdw_clash_flag[sample_id, i, j] = True
                            has_vdw_clash_flag[sample_id, j, i] = True
                    if chain_types[i] == 'lig' or chain_types[j] == 'lig':
                        continue
                    if self.compute_af3_clash:
                        af3_clash_pairs = self.get_chain_pair_violations(pred_coordinate=pred_coordinate[sample_id, :, :], violation_type='af3', chain_1_mask=atom_chain_mask_i, chain_2_mask=atom_chain_mask_j)
                        total_clash = af3_clash_pairs.shape[0]
                        relative_clash = total_clash / min(N_chain_i, N_chain_j)
                        af3_clash_details[sample_id, i, j, 0] = total_clash
                        af3_clash_details[sample_id, i, j, 1] = relative_clash
                        has_af3_clash_flag[sample_id, i, j] = total_clash > 100 or relative_clash > 0.5
                        af3_clash_details[sample_id, j, i, :] = af3_clash_details[sample_id, i, j, :]
                        has_af3_clash_flag[sample_id, j, i] = has_af3_clash_flag[sample_id, i, j]
        return {'summary': {'af3_clash': has_af3_clash_flag if self.compute_af3_clash else None, 'vdw_clash': has_vdw_clash_flag if self.compute_vdw_clash else None, 'chain_types': chain_types, 'skipped_pairs': skipped_pairs}, 'details': {'af3_clash': af3_clash_details if self.compute_af3_clash else None, 'vdw_clash': vdw_clash_details if self.compute_vdw_clash else None}}


class LDDT(nn.Module):
    """LDDT base metrics"""

    def __init__(self, eps: 'float'=1e-10):
        super(LDDT, self).__init__()
        self.eps = eps

    def _chunk_base_forward(self, pred_distance, true_distance) ->torch.Tensor:
        distance_error_l1 = torch.abs(pred_distance - true_distance)
        thresholds = [0.5, 1, 2, 4]
        sparse_pair_lddt = torch.stack([(distance_error_l1 < t) for t in thresholds], dim=-1).mean(dim=-1)
        del distance_error_l1
        if sparse_pair_lddt.numel() == 0:
            sparse_pair_lddt = torch.zeros_like(sparse_pair_lddt)
        lddt = torch.mean(sparse_pair_lddt, dim=-1)
        return lddt

    def _chunk_forward(self, pred_distance, true_distance, chunk_size: 'Optional[int]'=None) ->torch.Tensor:
        if chunk_size is None:
            return self._chunk_base_forward(pred_distance, true_distance)
        else:
            lddt = []
            N_sample = pred_distance.shape[-2]
            no_chunks = N_sample // chunk_size + (N_sample % chunk_size != 0)
            for i in range(no_chunks):
                lddt_i = self._chunk_base_forward(pred_distance[..., i * chunk_size:(i + 1) * chunk_size, :], true_distance)
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)
            return lddt

    def _calc_sparse_dist(self, pred_coordinate, true_coordinate, l_index, m_index):
        pred_coords_l = pred_coordinate.index_select(-2, l_index)
        pred_coords_m = pred_coordinate.index_select(-2, m_index)
        true_coords_l = true_coordinate.index_select(-2, l_index)
        true_coords_m = true_coordinate.index_select(-2, m_index)
        pred_distance_sparse_lm = torch.norm(pred_coords_l - pred_coords_m, p=2, dim=-1)
        true_distance_sparse_lm = torch.norm(true_coords_l - true_coords_m, p=2, dim=-1)
        return pred_distance_sparse_lm, true_distance_sparse_lm

    def forward(self, pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', lddt_mask: 'torch.Tensor', chunk_size: 'Optional[int]'=None) ->dict[str, torch.Tensor]:
        """LDDT: evaluated on complex, chains and interfaces
        sparse implementation, which largely reduce cuda memory when atom num reaches 10^4 +

        Args:
            pred_coordinate (torch.Tensor): the pred coordinates
                [N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [N_atom, 3]
            lddt_mask (torch.Tensor):
                sparse version of [N_atom, N_atom] atompair mask based on bespoke radius of true distance
                [N_nonzero_mask, 2]

        Returns:
            Dict[str, torch.Tensor]:
                "best": [N_eval]
                "worst": [N_eval]
        """
        lddt_indices = torch.nonzero(lddt_mask, as_tuple=True)
        l_index = lddt_indices[0]
        m_index = lddt_indices[1]
        pred_distance_sparse_lm, true_distance_sparse_lm = self._calc_sparse_dist(pred_coordinate, true_coordinate, l_index, m_index)
        group_lddt = self._chunk_forward(pred_distance_sparse_lm, true_distance_sparse_lm, chunk_size=chunk_size)
        return group_lddt

    @staticmethod
    def compute_lddt_mask(true_coordinate: 'torch.Tensor', true_coordinate_mask: 'torch.Tensor', is_nucleotide: 'torch.Tensor'=None, is_nucleotide_threshold: 'float'=30.0, threshold: 'float'=15.0):
        distance_mask = true_coordinate_mask[..., None] * true_coordinate_mask[..., None, :]
        distance = torch.cdist(true_coordinate, true_coordinate) * distance_mask
        c_lm = distance < threshold
        if is_nucleotide is not None:
            is_nucleotide_mask = is_nucleotide.bool()[..., None]
            c_lm = (distance < is_nucleotide_threshold) * is_nucleotide_mask + c_lm * ~is_nucleotide_mask
        c_lm = c_lm * (1 - torch.eye(n=c_lm.size(-1), device=c_lm.device, dtype=distance.dtype))
        c_lm = c_lm * distance_mask
        return c_lm


def add_diff_metrics(scores, ranker_keys):
    diff_metrics = {'diff/best_worst': scores['best'] - scores['worst'], 'diff/best_random': scores['best'] - scores['random'], 'diff/best_median': scores['best'] - scores['median']}
    for key in ranker_keys:
        diff_metrics.update({f'diff/best_{key}': scores['best'] - scores[f'{key}.rank1'], f'diff/{key}_median': scores[f'{key}.rank1'] - scores['median']})
    scores.update(diff_metrics)
    return scores


def get_complex_level_rankers(scores, keys):
    assert all([(k in ['plddt', 'gpde', 'ranking_score']) for k in keys])
    rankers = {}
    for key in keys:
        if key == 'gpde':
            descending = False
        else:
            descending = True
        ranking = scores[key].argsort(dim=0, descending=descending)
        rankers[f'{key}.rank1'] = lambda x, rank1_idx=ranking[0].item(): x[..., rank1_idx]
    return rankers


class LDDTMetrics(nn.Module):
    """LDDT: evaluated on chains and interfaces"""

    def __init__(self, configs):
        super(LDDTMetrics, self).__init__()
        self.eps = configs.metrics.lddt.eps
        self.configs = configs
        self.chunk_size = self.configs.infer_setting.lddt_metrics_chunk_size
        self.lddt_base = LDDT(eps=self.eps)
        self.complex_ranker_keys = configs.metrics.get('complex_ranker_keys', ['plddt', 'gpde', 'ranking_score'])

    def compute_lddt(self, pred_dict: 'dict', label_dict: 'dict'):
        """compute complex-level and chain/interface-level lddt

        Args:
            pred_dict (Dict): a dictionary containing
                coordinate: [N_sample, N_atom, 3]
            label_dict (Dict): a dictionary containing
                coordinate: [N_sample, N_atom, 3]
                lddt_mask: [N_atom, N_atom]
        """
        out = {}
        lddt = self.lddt_base.forward(pred_coordinate=pred_dict['coordinate'], true_coordinate=label_dict['coordinate'], lddt_mask=label_dict['lddt_mask'], chunk_size=self.chunk_size)
        out['complex'] = lddt
        return out

    def aggregate(self, vals, dim: 'int'=-1, aggregators: 'dict'={}):
        N_sample = vals.size(dim)
        median_index = N_sample // 2
        basic_sample_aggregators = {'best': lambda x: x.max(dim=dim)[0], 'worst': lambda x: x.min(dim=dim)[0], 'random': lambda x: x.select(dim=dim, index=0), 'mean': lambda x: x.mean(dim=dim), 'median': lambda x: x.sort(dim=dim, descending=True)[0].select(dim=dim, index=median_index)}
        sample_aggregators = {**basic_sample_aggregators, **aggregators}
        return {agg_name: agg_func(vals) for agg_name, agg_func in sample_aggregators.items()}

    def aggregate_lddt(self, lddt_dict, per_sample_summary_confidence):
        confidence_scores = sample_confidence.merge_per_sample_confidence_scores(per_sample_summary_confidence)
        complex_level_ranker = get_complex_level_rankers(confidence_scores, self.complex_ranker_keys)
        complex_lddt = self.aggregate(lddt_dict['complex'], aggregators=complex_level_ranker)
        complex_lddt = add_diff_metrics(complex_lddt, self.complex_ranker_keys)
        complex_lddt = {f'lddt/complex/{name}': value for name, value in complex_lddt.items()}
        return complex_lddt, {}


class FusedLayerNormAffineFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, normalized_shape, eps):
        d = input.dtype
        if d is torch.bfloat16:
            with torch.amp.autocast(enabled=False):
                ctx.normalized_shape = normalized_shape
                ctx.eps = eps
                input_ = input.contiguous()
                weight_ = weight.contiguous()
                bias_ = bias.contiguous()
                output, mean, invvar = fastfold_layer_norm_cuda.forward_affine(input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
                ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        else:
            ctx.normalized_shape = normalized_shape
            ctx.eps = eps
            input_ = input.contiguous()
            weight_ = weight.contiguous()
            bias_ = bias.contiguous()
            output, mean, invvar = fastfold_layer_norm_cuda.forward_affine(input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
            ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        d = grad_output.dtype
        if d is torch.bfloat16:
            with torch.amp.autocast(enabled=False):
                input_, weight_, bias_, mean, invvar = ctx.saved_tensors
                grad_input = grad_weight = grad_bias = None
                grad_input, grad_weight, grad_bias = fastfold_layer_norm_cuda.backward_affine(grad_output.contiguous(), mean, invvar, input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
        else:
            input_, weight_, bias_, mean, invvar = ctx.saved_tensors
            grad_input = grad_weight = grad_bias = None
            grad_input, grad_weight, grad_bias = fastfold_layer_norm_cuda.backward_affine(grad_output.contiguous(), mean, invvar, input_, ctx.normalized_shape, weight_, bias_, ctx.eps)
        return grad_input, grad_weight, grad_bias, None, None


class FusedLayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-05):
        super(FusedLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = normalized_shape,
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.ones(*normalized_shape))
        self.bias = Parameter(torch.ones(*normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return self.kernel_forward(input)

    def kernel_forward(self, input):
        return FusedLayerNormAffineFunction.apply(input, self.weight, self.bias, self.normalized_shape, self.eps)


def get_checkpoint_fn():
    deepspeed_is_configured = deepspeed_is_installed and deepspeed.checkpointing.is_configured()
    if deepspeed_is_configured:
        checkpoint = deepspeed.checkpointing.checkpoint
    else:
        checkpoint = torch.utils.checkpoint.checkpoint
    return checkpoint


def loss_reduction(loss: 'torch.Tensor', method: 'str'='mean') ->torch.Tensor:
    """reduction wrapper

    Args:
        loss (torch.Tensor): loss
            [...]
        method (str, optional): reduction method. Defaults to "mean".

    Returns:
        torch.Tensor: reduced loss
            [] or [...]
    """
    if method is None:
        return loss
    assert method in ['mean', 'sum', 'add', 'max', 'min']
    if method == 'add':
        method = 'sum'
    return getattr(torch, method)(loss)


class SmoothLDDTLoss(nn.Module):
    """
    Implements Algorithm 27 [SmoothLDDTLoss] in AF3
    """

    def __init__(self, eps: 'float'=1e-10, reduction: 'str'='mean') ->None:
        """SmoothLDDTLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-10.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(SmoothLDDTLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, c_lm=None):
        dist_diff = torch.abs(pred_distance - true_distance)
        dist_diff_epsilon = 0
        for threshold in [0.5, 1, 2, 4]:
            dist_diff_epsilon += 0.25 * torch.sigmoid(threshold - dist_diff)
        if c_lm is not None:
            lddt = torch.sum(c_lm * dist_diff_epsilon, dim=(-1, -2)) / (torch.sum(c_lm, dim=(-1, -2)) + self.eps)
        else:
            lddt = torch.mean(dist_diff_epsilon, dim=-1)
        return lddt

    def forward(self, pred_distance: 'torch.Tensor', true_distance: 'torch.Tensor', distance_mask: 'torch.Tensor', lddt_mask: 'torch.Tensor', diffusion_chunk_size: 'Optional[int]'=None) ->torch.Tensor:
        """SmoothLDDTLoss

        Args:
            pred_distance (torch.Tensor): the diffusion denoised atom-atom distance
                [..., N_sample, N_atom, N_atom]
            true_distance (torch.Tensor): the ground truth coordinates
                [..., N_atom, N_atom]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()
        if diffusion_chunk_size is None:
            lddt = self._chunk_forward(pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm)
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_distance.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
            for i in range(no_chunks):
                lddt_i = checkpoint_fn(self._chunk_forward, pred_distance[..., i * diffusion_chunk_size:(i + 1) * diffusion_chunk_size, :, :], true_distance, c_lm)
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)
        lddt = lddt.mean(dim=-1)
        return 1 - loss_reduction(lddt, method=self.reduction)

    def sparse_forward(self, pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', lddt_mask: 'torch.Tensor', diffusion_chunk_size: 'Optional[int]'=None) ->torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        lddt_indices = torch.nonzero(lddt_mask, as_tuple=True)
        true_coords_l = true_coordinate.index_select(-2, lddt_indices[0])
        true_coords_m = true_coordinate.index_select(-2, lddt_indices[1])
        true_distance_sparse_lm = torch.norm(true_coords_l - true_coords_m, p=2, dim=-1)
        if diffusion_chunk_size is None:
            pred_coords_l = pred_coordinate.index_select(-2, lddt_indices[0])
            pred_coords_m = pred_coordinate.index_select(-2, lddt_indices[1])
            pred_distance_sparse_lm = torch.norm(pred_coords_l - pred_coords_m, p=2, dim=-1)
            lddt = self._chunk_forward(pred_distance_sparse_lm, true_distance_sparse_lm, c_lm=None)
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
            for i in range(no_chunks):
                pred_coords_i_l = pred_coordinate[i * diffusion_chunk_size:(i + 1) * diffusion_chunk_size, :, :].index_select(-2, lddt_indices[0])
                pred_coords_i_m = pred_coordinate[i * diffusion_chunk_size:(i + 1) * diffusion_chunk_size, :, :].index_select(-2, lddt_indices[1])
                pred_distance_sparse_i_lm = torch.norm(pred_coords_i_l - pred_coords_i_m, p=2, dim=-1)
                lddt_i = checkpoint_fn(self._chunk_forward, pred_distance_sparse_i_lm, true_distance_sparse_lm)
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)
        lddt = lddt.mean(dim=-1)
        return 1 - loss_reduction(lddt, method=self.reduction)

    def dense_forward(self, pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', lddt_mask: 'torch.Tensor', diffusion_chunk_size: 'Optional[int]'=None) ->torch.Tensor:
        """SmoothLDDTLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            lddt_mask (torch.Tensor, optional): whether true distance is within radius (30A for nuc and 15A for others)
                [N_atom, N_atom]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the smooth lddt loss
                [...] if reduction is None else []
        """
        c_lm = lddt_mask.bool().unsqueeze(dim=-3).detach()
        true_distance = torch.cdist(true_coordinate, true_coordinate)
        if diffusion_chunk_size is None:
            pred_distance = torch.cdist(pred_coordinate, pred_coordinate)
            lddt = self._chunk_forward(pred_distance=pred_distance, true_distance=true_distance, c_lm=c_lm)
        else:
            checkpoint_fn = get_checkpoint_fn()
            lddt = []
            N_sample = pred_coordinate.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
            for i in range(no_chunks):
                pred_distance_i = torch.cdist(pred_coordinate[i * diffusion_chunk_size:(i + 1) * diffusion_chunk_size, :, :], pred_coordinate[i * diffusion_chunk_size:(i + 1) * diffusion_chunk_size, :, :])
                lddt_i = checkpoint_fn(self._chunk_forward, pred_distance_i, true_distance, c_lm)
                lddt.append(lddt_i)
            lddt = torch.cat(lddt, dim=-1)
        lddt = lddt.mean(dim=-1)
        return 1 - loss_reduction(lddt, method=self.reduction)


class BondLoss(nn.Module):
    """
    Implements Formula 5 [BondLoss] in AF3
    """

    def __init__(self, eps: 'float'=1e-06, reduction: 'str'='mean') ->None:
        """BondLoss

        Args:
            eps (float, optional): avoid nan. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(BondLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def _chunk_forward(self, pred_distance, true_distance, bond_mask):
        dist_squared_err = (pred_distance - true_distance.unsqueeze(dim=-3)) ** 2
        bond_loss = torch.sum(dist_squared_err * bond_mask, dim=(-1, -2)) / torch.sum(bond_mask + self.eps, dim=(-1, -2))
        return bond_loss

    def forward(self, pred_distance: 'torch.Tensor', true_distance: 'torch.Tensor', distance_mask: 'torch.Tensor', bond_mask: 'torch.Tensor', per_sample_scale: 'torch.Tensor'=None, diffusion_chunk_size: 'Optional[int]'=None) ->torch.Tensor:
        """BondLoss

        Args:
            pred_distance (torch.Tensor): the diffusion denoised atom-atom distance
                [..., N_sample, N_atom, N_atom]
            true_distance (torch.Tensor): the ground truth coordinates
                [..., N_atom, N_atom]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom] or [..., N_atom, N_atom]
            bond_mask (torch.Tensor): bonds considered in this loss
                [N_atom, N_atom] or [..., N_atom, N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]
            diffusion_chunk_size (Optional[int]): Chunk size over the N_sample dimension. Defaults to None.

        Returns:
            torch.Tensor: the bond loss
                [...] if reduction is None else []
        """
        bond_mask = (bond_mask * distance_mask).unsqueeze(dim=-3)
        if diffusion_chunk_size is None:
            bond_loss = self._chunk_forward(pred_distance=pred_distance, true_distance=true_distance, bond_mask=bond_mask)
        else:
            checkpoint_fn = get_checkpoint_fn()
            bond_loss = []
            N_sample = pred_distance.shape[-3]
            no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
            for i in range(no_chunks):
                bond_loss_i = checkpoint_fn(self._chunk_forward, pred_distance[..., i * diffusion_chunk_size:(i + 1) * diffusion_chunk_size, :, :], true_distance, bond_mask)
                bond_loss.append(bond_loss_i)
            bond_loss = torch.cat(bond_loss, dim=-1)
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale
        bond_loss = bond_loss.mean(dim=-1)
        return loss_reduction(bond_loss, method=self.reduction)

    def sparse_forward(self, pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', distance_mask: 'torch.Tensor', bond_mask: 'torch.Tensor', per_sample_scale: 'torch.Tensor'=None) ->torch.Tensor:
        """BondLoss sparse implementation

        Args:
            pred_coordinate (torch.Tensor): the diffusion denoised atom coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth atom coordinates
                [..., N_atom, 3]
            distance_mask (torch.Tensor): whether true coordinates exist.
                [N_atom, N_atom] or [..., N_atom, N_atom]
            bond_mask (torch.Tensor): bonds considered in this loss
                [N_atom, N_atom] or [..., N_atom, N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]
        Returns:
            torch.Tensor: the bond loss
                [...] if reduction is None else []
        """
        bond_mask = bond_mask * distance_mask
        bond_indices = torch.nonzero(bond_mask, as_tuple=True)
        pred_coords_i = pred_coordinate.index_select(-2, bond_indices[0])
        pred_coords_j = pred_coordinate.index_select(-2, bond_indices[1])
        true_coords_i = true_coordinate.index_select(-2, bond_indices[0])
        true_coords_j = true_coordinate.index_select(-2, bond_indices[1])
        pred_distance_sparse = torch.norm(pred_coords_i - pred_coords_j, p=2, dim=-1)
        true_distance_sparse = torch.norm(true_coords_i - true_coords_j, p=2, dim=-1)
        dist_squared_err_sparse = (pred_distance_sparse - true_distance_sparse) ** 2
        if dist_squared_err_sparse.numel() == 0:
            return torch.tensor(0.0, device=dist_squared_err_sparse.device, requires_grad=True)
        bond_loss = torch.mean(dist_squared_err_sparse, dim=-1)
        if per_sample_scale is not None:
            bond_loss = bond_loss * per_sample_scale
        bond_loss = bond_loss.mean(dim=-1)
        return bond_loss


def softmax_cross_entropy(logits: 'torch.Tensor', labels: 'torch.Tensor') ->torch.Tensor:
    """Softmax cross entropy

    Args:
        logits (torch.Tensor): classification logits
            [..., num_class]
        labels (torch.Tensor): classification labels (value = probability)
            [..., num_class]

    Returns:
        torch.Tensor: softmax cross entropy
            [...]
    """
    loss = -1 * torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
    return loss


class DistogramLoss(nn.Module):
    """
    Implements DistogramLoss in AF3
    """

    def __init__(self, min_bin: 'float'=2.3125, max_bin: 'float'=21.6875, no_bins: 'int'=64, eps: 'float'=1e-06, reduction: 'str'='mean') ->None:
        """Distogram loss
        This head and loss are identical to AlphaFold 2, where the pairwise token distances use the representative atom for each token:
            Cβ for protein residues (Cα for glycine),
            C4 for purines and C2 for pyrimidines.
            All ligands already have a single atom per token.

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 2.3125.
            max_bin (float, optional): max boundary of bins. Defaults to 21.6875.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduce (bool, optional): reduce dim. Defaults to True.
        """
        super(DistogramLoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(self, true_coordinate: 'torch.Tensor', coordinate_mask: 'torch.Tensor', rep_atom_mask: 'torch.Tensor') ->tuple[torch.Tensor, torch.Tensor]:
        """calculate the label as bins

        Args:
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor): representative atom mask
                [N_atom]

        Returns:
            true_bins (torch.Tensor): distance error assigned into bins (one-hot).
                [..., N_token, N_token, no_bins]
            pair_coordinate_mask (torch.Tensor): whether the coordinates of representative atom pairs exist.
                [N_token, N_token] or [..., N_token, N_token]
        """
        boundaries = torch.linspace(start=self.min_bin, end=self.max_bin, steps=self.no_bins - 1, device=true_coordinate.device)
        rep_atom_mask = rep_atom_mask.bool()
        true_coordinate = true_coordinate[..., rep_atom_mask, :]
        gt_dist = cdist(true_coordinate, true_coordinate)
        true_bins = torch.sum(gt_dist.unsqueeze(dim=-1) > boundaries, dim=-1)
        token_mask = coordinate_mask[..., rep_atom_mask]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        return F.one_hot(true_bins, self.no_bins), pair_mask

    def forward(self, logits: 'torch.Tensor', true_coordinate: 'torch.Tensor', coordinate_mask: 'torch.Tensor', rep_atom_mask: 'torch.Tensor') ->torch.Tensor:
        """Distogram loss

        Args:
            logits (torch.Tensor): logits.
                [..., N_token, N_token, no_bins]
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor): representative atom mask.
                [N_atom]

        Returns:
            torch.Tensor: the return loss.
                [...] if self.reduction is not None else []
        """
        with torch.no_grad():
            true_bins, pair_mask = self.calculate_label(true_coordinate=true_coordinate, coordinate_mask=coordinate_mask, rep_atom_mask=rep_atom_mask)
        errors = softmax_cross_entropy(logits=logits, labels=true_bins)
        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))
        loss = torch.sum(errors * pair_mask, dim=(-1, -2))
        loss = loss / denom
        return loss_reduction(loss, method=self.reduction)


class PDELoss(nn.Module):
    """
    Implements Predicted distance loss in AF3
    """

    def __init__(self, min_bin: 'float'=0, max_bin: 'float'=32, no_bins: 'int'=64, eps: 'float'=1e-06, reduction: 'str'='mean') ->None:
        """PDELoss
        This loss are between representative token atoms i and j in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 32.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(PDELoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(self, pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', coordinate_mask: 'torch.Tensor', rep_atom_mask: 'torch.Tensor') ->tuple[torch.Tensor, torch.Tensor]:
        """calculate the label as bins

        Args:
            pred_coordinate (torch.Tensor): predicted coordinates.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor):
                [N_atom]

        Returns:
            true_bins (torch.Tensor): distance error assigned into bins (one-hot).
                [..., N_sample, N_token, N_token, no_bins]
            pair_coordinate_mask (torch.Tensor): whether the coordinates of representative atom pairs exist.
                [N_token, N_token] or [..., N_token, N_token]
        """
        boundaries = torch.linspace(start=self.min_bin, end=self.max_bin, steps=self.no_bins + 1, device=pred_coordinate.device)
        rep_atom_mask = rep_atom_mask.bool()
        true_coordinate = true_coordinate[..., rep_atom_mask, :]
        gt_dist = cdist(true_coordinate, true_coordinate)
        pred_coordinate = pred_coordinate[..., rep_atom_mask, :]
        pred_dist = cdist(pred_coordinate, pred_coordinate)
        dist_error = torch.abs(pred_dist - gt_dist.unsqueeze(dim=-3))
        true_bins = torch.sum(dist_error.unsqueeze(dim=-1) > boundaries, dim=-1)
        true_bins = torch.clamp(true_bins, min=1, max=self.no_bins)
        token_mask = coordinate_mask[..., rep_atom_mask]
        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        return F.one_hot(true_bins - 1, self.no_bins).detach(), pair_mask.detach()

    def forward(self, logits: 'torch.Tensor', pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', coordinate_mask: 'torch.Tensor', rep_atom_mask: 'torch.Tensor') ->torch.Tensor:
        """PDELoss

        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_token, N_token, no_bins]
            pred_coordinate: (torch.Tensor): predict coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            rep_atom_mask (torch.Tensor): representative atom mask for this loss
                [N_atom]

        Returns:
            torch.Tensor: the return loss
                [...] if reduction is None else []
        """
        with torch.no_grad():
            true_bins, pair_mask = self.calculate_label(pred_coordinate=pred_coordinate, true_coordinate=true_coordinate, coordinate_mask=coordinate_mask, rep_atom_mask=rep_atom_mask)
        errors = softmax_cross_entropy(logits=logits, labels=true_bins)
        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))
        loss = errors * pair_mask.unsqueeze(dim=-3)
        loss = torch.sum(loss, dim=(-1, -2))
        loss = loss / denom.unsqueeze(dim=-1)
        loss = loss.mean(dim=-1)
        return loss_reduction(loss, method=self.reduction)


def expressCoordinatesInFrame(coordinate: 'torch.Tensor', frames: 'torch.Tensor', eps: 'float'=1e-08) ->torch.Tensor:
    """Algorithm 29 Express coordinate in frame

    Args:
        coordinate (torch.Tensor): the input coordinate
            [..., N_atom, 3]
        frames (torch.Tensor): the input frames
            [..., N_frame, 3, 3]
        eps (float): Small epsilon value

    Returns:
        torch.Tensor: the transformed coordinate projected onto frame basis
            [..., N_frame, N_atom, 3]
    """
    a, b, c = torch.unbind(frames, dim=-2)
    w1 = F.normalize(a - b, dim=-1, eps=eps)
    w2 = F.normalize(c - b, dim=-1, eps=eps)
    e1 = F.normalize(w1 + w2, dim=-1, eps=eps)
    e2 = F.normalize(w2 - w1, dim=-1, eps=eps)
    e3 = torch.cross(e1, e2, dim=-1)
    d = coordinate[..., None, :, :] - b[..., None, :]
    x_transformed = torch.cat([torch.sum(d * e1[..., None, :], dim=-1, keepdim=True), torch.sum(d * e2[..., None, :], dim=-1, keepdim=True), torch.sum(d * e3[..., None, :], dim=-1, keepdim=True)], dim=-1)
    return x_transformed


def compute_alignment_error_squared(pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', pred_frames: 'torch.Tensor', true_frames: 'torch.Tensor') ->torch.Tensor:
    """Implements Algorithm 30 Compute alignment error, but do not take the square root

    Args:
        pred_coordinate (torch.Tensor): the predict coords [frame center]
            [..., N_sample, N_token, 3]
        true_coordinate (torch.Tensor): the ground truth coords [frame center]
            [..., N_token, 3]
        pred_frames (torch.Tensor): the predict frame
            [..., N_sample, N_frame, 3, 3]
        true_frames (torch.Tensor): the ground truth frame
            [..., N_frame, 3, 3]

    Returns:
        torch.Tensor: the computed alignment error
            [..., N_sample, N_frame, N_token]
    """
    x_transformed_pred = expressCoordinatesInFrame(coordinate=pred_coordinate, frames=pred_frames)
    x_transformed_true = expressCoordinatesInFrame(coordinate=true_coordinate, frames=true_frames)
    squared_pae = torch.sum((x_transformed_pred - x_transformed_true.unsqueeze(dim=-4)) ** 2, dim=-1)
    return squared_pae


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)
    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def gather_frame_atom_by_indices(coordinate: 'torch.Tensor', frame_atom_index: 'torch.Tensor', dim: 'int'=-2) ->torch.Tensor:
    """construct frames from coordinate

    Args:
        coordinate (torch.Tensor):  the input coordinate
            [..., N_atom, 3]
        frame_atom_index (torch.Tensor): indices of three atoms in each frame
            [..., N_frame, 3] or [N_frame, 3]
        dim (torch.Tensor): along which dimension to select the frame atoms
    Returns:
        torch.Tensor: the constructed frames
            [..., N_frame, 3[three atom], 3[three coordinate]]
    """
    if len(frame_atom_index.shape) == 2:
        x1 = torch.index_select(coordinate, dim=dim, index=frame_atom_index[:, 0])
        x2 = torch.index_select(coordinate, dim=dim, index=frame_atom_index[:, 1])
        x3 = torch.index_select(coordinate, dim=dim, index=frame_atom_index[:, 2])
        return torch.stack([x1, x2, x3], dim=dim)
    else:
        assert frame_atom_index.shape[:dim] == coordinate.shape[:dim], 'batch size dims should match'
    x1 = batched_gather(data=coordinate, inds=frame_atom_index[..., 0], dim=dim, no_batch_dims=len(coordinate.shape[:dim]))
    x2 = batched_gather(data=coordinate, inds=frame_atom_index[..., 1], dim=dim, no_batch_dims=len(coordinate.shape[:dim]))
    x3 = batched_gather(data=coordinate, inds=frame_atom_index[..., 2], dim=dim, no_batch_dims=len(coordinate.shape[:dim]))
    return torch.stack([x1, x2, x3], dim=dim)


class PAELoss(nn.Module):
    """
    Implements Predicted Aligned distance loss in AF3
    """

    def __init__(self, min_bin: 'float'=0, max_bin: 'float'=32, no_bins: 'int'=64, eps: 'float'=1e-06, reduction: 'str'='mean') ->None:
        """PAELoss
        This loss are between representative token atoms i and j in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 32.
            no_bins (int, optional): number of bins. Defaults to 64.
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduce (bool, optional): reduce dim. Defaults to True.
        """
        super(PAELoss, self).__init__()
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction

    def calculate_label(self, pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', coordinate_mask: 'torch.Tensor', rep_atom_mask: 'torch.Tensor', frame_atom_index: 'torch.Tensor', has_frame: 'torch.Tensor') ->tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """calculate true PAE (squared) and true bins

        Args:
            pred_coordinate: (torch.Tensor): predict coordinates.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom]
            rep_atom_mask (torch.Tensor): masks of the representative atom for each token.
                [N_atom]
            frame_atom_index (torch.Tensor): indices of frame atoms (three atoms per token(=per frame)).
                [N_token, 3[three atom]]
            has_frame (torch.Tensor): indicates whether token_i has a valid frame.
                [N_token]
        Returns:
            squared_pae (torch.Tensor): pairwise alignment error squared
                [..., N_sample, N_frame, N_token] where N_token = rep_atom_mask.sum()
            true_bins (torch.Tensor): the true bins
                [..., N_sample, N_frame, N_token, no_bins]
            frame_token_pair_mask (torch.Tensor): whether frame_i token_j both have true coordinates.
                [N_frame, N_token]
        """
        coordinate_mask = coordinate_mask.bool()
        rep_atom_mask = rep_atom_mask.bool()
        has_frame = has_frame.bool()
        assert len(frame_atom_index.shape) == 2
        frame_atom_index = frame_atom_index[has_frame, :]
        pred_frames = gather_frame_atom_by_indices(coordinate=pred_coordinate, frame_atom_index=frame_atom_index, dim=-2)
        true_frames = gather_frame_atom_by_indices(coordinate=true_coordinate, frame_atom_index=frame_atom_index, dim=-2)
        true_frame_coord_mask = gather_frame_atom_by_indices(coordinate=coordinate_mask, frame_atom_index=frame_atom_index, dim=-1)
        true_frame_coord_mask = true_frame_coord_mask.sum(dim=-1) >= 3
        token_mask = coordinate_mask[rep_atom_mask]
        frame_token_pair_mask = true_frame_coord_mask[..., None] * token_mask[..., None, :]
        squared_pae = compute_alignment_error_squared(pred_coordinate=pred_coordinate[..., rep_atom_mask, :], true_coordinate=true_coordinate[..., rep_atom_mask, :], pred_frames=pred_frames, true_frames=true_frames) * frame_token_pair_mask
        boundaries = torch.linspace(start=self.min_bin, end=self.max_bin, steps=self.no_bins + 1, device=pred_coordinate.device)
        boundaries = boundaries ** 2
        true_bins = torch.sum(squared_pae.unsqueeze(dim=-1) > boundaries, dim=-1)
        true_bins = torch.where(frame_token_pair_mask, true_bins, torch.ones_like(true_bins) * self.no_bins)
        true_bins = torch.clamp(true_bins, min=1, max=self.no_bins)
        return squared_pae.detach(), F.one_hot(true_bins - 1, self.no_bins).detach(), frame_token_pair_mask.detach()

    def forward(self, logits: 'torch.Tensor', pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', coordinate_mask: 'torch.Tensor', frame_atom_index: 'torch.Tensor', rep_atom_mask: 'torch.Tensor', has_frame: 'torch.Tensor') ->torch.Tensor:
        """PAELoss

        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_token, N_token, no_bins]
            pred_coordinate: (torch.Tensor): predict coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom]
            rep_atom_mask (torch.Tensor): masks of the representative atom for each token.
                [N_atom]
            frame_atom_index (torch.Tensor): indices of frame atoms (three atoms per token(=per frame)).
                [N_token, 3[three atom]]
            has_frame (torch.Tensor): indicates whether token_i has a valid frame.
                [N_token]
        Returns:
            torch.Tensor: the return loss
                [] if reduce
                [..., n] else
        """
        has_frame = has_frame.bool()
        rep_atom_mask = rep_atom_mask.bool()
        assert len(has_frame.shape) == 1
        assert len(frame_atom_index.shape) == 2
        with torch.no_grad():
            _, true_bins, pair_mask = self.calculate_label(pred_coordinate=pred_coordinate, true_coordinate=true_coordinate, frame_atom_index=frame_atom_index, rep_atom_mask=rep_atom_mask, coordinate_mask=coordinate_mask, has_frame=has_frame)
        loss = softmax_cross_entropy(logits=logits[..., has_frame, :, :], labels=true_bins)
        denom = self.eps + torch.sum(pair_mask, dim=(-1, -2))
        loss = loss * pair_mask.unsqueeze(dim=-3)
        loss = torch.sum(loss, dim=(-1, -2))
        loss = loss / denom.unsqueeze(dim=-1)
        loss = loss.mean(dim=-1)
        return loss_reduction(loss, self.reduction)


class ExperimentallyResolvedLoss(nn.Module):

    def __init__(self, eps: 'float'=1e-06, reduction: 'str'='mean') ->None:
        """
        Args:
            eps (float, optional): avoid nan. Defaults to 1e-6.
        """
        super(ExperimentallyResolvedLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: 'torch.Tensor', coordinate_mask: 'torch.Tensor', atom_mask: 'torch.Tensor'=None) ->torch.Tensor:
        """
        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_atom, no_bins:=2]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [..., N_atom] | [N_atom]
            atom_mask (torch.Tensor, optional): whether to conside the atom in the loss
                [..., N_atom]
        Returns:
            torch.Tensor: the experimentally resolved loss
        """
        is_resolved = F.one_hot(coordinate_mask.long(), 2)
        errors = softmax_cross_entropy(logits=logits, labels=is_resolved.unsqueeze(dim=-3))
        if atom_mask is None:
            loss = errors.mean(dim=-1)
        else:
            loss = torch.sum(errors * atom_mask[..., None, :], dim=-1)
            loss = loss / (self.eps + torch.sum(atom_mask[..., None, :], dim=-1))
        loss = loss.mean(dim=-1)
        return loss_reduction(loss, method=self.reduction)


def expand_at_dim(x: 'torch.Tensor', dim: 'int', n: 'int') ->torch.Tensor:
    """expand a tensor at specific dim by n times

    Args:
        x (torch.Tensor): input
        dim (int): dimension to expand
        n (int): expand size

    Returns:
        torch.Tensor: expanded tensor of shape [..., n, ...]
    """
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1:]
    return x.expand(*before_shape, n, *after_shape)


class MSELoss(nn.Module):
    """
    Implements Formula 2-4 [MSELoss] in AF3
    """

    def __init__(self, weight_mse: 'float'=1 / 3, weight_dna: 'float'=5.0, weight_rna=5.0, weight_ligand=10.0, eps=1e-06, reduction: 'str'='mean') ->None:
        super(MSELoss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_dna = weight_dna
        self.weight_rna = weight_rna
        self.weight_ligand = weight_ligand
        self.eps = eps
        self.reduction = reduction

    def weighted_rigid_align(self, pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', coordinate_mask: 'torch.Tensor', is_dna: 'torch.Tensor', is_rna: 'torch.Tensor', is_ligand: 'torch.Tensor') ->tuple[torch.Tensor, torch.Tensor]:
        """compute weighted rigid alignment results

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask
                [N_atom] or [..., N_atom]

        Returns:
            true_coordinate_aligned (torch.Tensor): aligned coordinates for each sample
                [..., N_sample, N_atom, 3]
            weight (torch.Tensor): weights for each atom
                [N_atom] or [..., N_sample, N_atom]
        """
        N_sample = pred_coordinate.size(-3)
        weight = 1 + self.weight_dna * is_dna + self.weight_rna * is_rna + self.weight_ligand * is_ligand
        weight = weight * coordinate_mask
        true_coordinate = true_coordinate * coordinate_mask.unsqueeze(dim=-1)
        pred_coordinate = pred_coordinate * coordinate_mask[..., None, :, None]
        true_coordinate = expand_at_dim(true_coordinate, dim=-3, n=N_sample)
        if len(weight.shape) > 1:
            weight = expand_at_dim(weight, dim=-2, n=N_sample)
        d = pred_coordinate.dtype
        with torch.amp.autocast(enabled=False):
            true_coordinate_aligned = weighted_rigid_align(x=true_coordinate, x_target=pred_coordinate, atom_weight=weight, stop_gradient=True)
            true_coordinate_aligned = true_coordinate_aligned
        return true_coordinate_aligned.detach(), weight.detach()

    def forward(self, pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', coordinate_mask: 'torch.Tensor', is_dna: 'torch.Tensor', is_rna: 'torch.Tensor', is_ligand: 'torch.Tensor', per_sample_scale: 'torch.Tensor'=None) ->torch.Tensor:
        """MSELoss

        Args:
            pred_coordinate (torch.Tensor): the denoised coordinates from diffusion module.
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): the ground truth coordinates.
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist.
                [N_atom] or [..., N_atom]
            is_dna / is_rna / is_ligand (torch.Tensor): mol type mask.
                [N_atom] or [..., N_atom]
            per_sample_scale (torch.Tensor, optional): whether to scale the loss by the per-sample noise-level.
                [..., N_sample]

        Returns:
            torch.Tensor: the weighted mse loss.
                [...] is self.reduction is None else []
        """
        with torch.no_grad():
            true_coordinate_aligned, weight = self.weighted_rigid_align(pred_coordinate=pred_coordinate, true_coordinate=true_coordinate, coordinate_mask=coordinate_mask, is_dna=is_dna, is_rna=is_rna, is_ligand=is_ligand)
        per_atom_se = ((pred_coordinate - true_coordinate_aligned) ** 2).sum(dim=-1)
        per_sample_weighted_mse = (weight * per_atom_se).sum(dim=-1) / (coordinate_mask.sum(dim=-1, keepdim=True) + self.eps)
        if per_sample_scale is not None:
            per_sample_weighted_mse = per_sample_weighted_mse * per_sample_scale
        weighted_align_mse_loss = self.weight_mse * per_sample_weighted_mse.mean(dim=-1)
        loss = loss_reduction(weighted_align_mse_loss, method=self.reduction)
        return loss


class PLDDTLoss(nn.Module):
    """
    Implements PLDDT Loss in AF3, different from the paper description.
    Main changes:
    1. use difference of distance instead of predicted distance when calculating plddt
    2. normalize each plddt score within 0-1
    """

    def __init__(self, min_bin: 'float'=0, max_bin: 'float'=1, no_bins: 'int'=50, is_nucleotide_threshold: 'float'=30.0, is_not_nucleotide_threshold: 'float'=15.0, eps: 'float'=1e-06, normalize: 'bool'=True, reduction: 'str'='mean') ->None:
        """PLDDT loss
        This loss are between atoms l and m (has some filters) in the mini-rollout prediction

        Args:
            min_bin (float, optional): min boundary of bins. Defaults to 0.
            max_bin (float, optional): max boundary of bins. Defaults to 1.
            no_bins (int, optional): number of bins. Defaults to 50.
            is_nucleotide_threshold (float, optional): threshold for nucleotide atoms. Defaults 30.0.
            is_not_nucleotide_threshold (float, optional): threshold for non-nucleotide atoms. Defaults 15.0
            eps (float, optional): small number added to denominator. Defaults to 1e-6.
            reduction (str, optional): reduction method for the batch dims. Defaults to mean.
        """
        super(PLDDTLoss, self).__init__()
        self.normalize = normalize
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.eps = eps
        self.reduction = reduction
        self.is_nucleotide_threshold = is_nucleotide_threshold
        self.is_not_nucleotide_threshold = is_not_nucleotide_threshold

    def calculate_label(self, pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', is_nucleotide: 'torch.Tensor', is_polymer: 'torch.Tensor', rep_atom_mask: 'torch.Tensor') ->torch.Tensor:
        """calculate the lddt as described in Sec 4.3.1.

        Args:
            pred_coordinate (torch.Tensor):
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor):
                [..., N_atom]
            is_nucleotide (torch.Tensor):
                [N_atom] or [..., N_atom]
            is_polymer (torch.Tensor):
                [N_atom]
            rep_atom_mask (torch.Tensor):
                [N_atom]

        Returns:
            torch.Tensor: per-atom lddt
                [..., N_sample, N_atom]
        """
        N_atom = true_coordinate.size(-2)
        atom_m_mask = (rep_atom_mask * is_polymer).bool()
        pred_d_lm = torch.cdist(pred_coordinate, pred_coordinate[..., atom_m_mask, :])
        true_d_lm = torch.cdist(true_coordinate, true_coordinate[..., atom_m_mask, :])
        delta_d_lm = torch.abs(pred_d_lm - true_d_lm.unsqueeze(dim=-3))
        thresholds = [0.5, 1, 2, 4]
        lddt_lm = torch.stack([(delta_d_lm < t) for t in thresholds], dim=-1).mean(dim=-1)
        is_nucleotide = is_nucleotide[..., atom_m_mask].bool()
        locality_mask = (true_d_lm < self.is_nucleotide_threshold) * is_nucleotide.unsqueeze(dim=-2) + (true_d_lm < self.is_not_nucleotide_threshold) * ~is_nucleotide.unsqueeze(dim=-2)
        diagonal_mask = (1 - torch.eye(n=N_atom)).bool()[..., atom_m_mask]
        pair_mask = (locality_mask * diagonal_mask).unsqueeze(dim=-3)
        per_atom_lddt = torch.sum(lddt_lm * pair_mask, dim=-1, keepdim=True)
        if self.normalize:
            per_atom_lddt = per_atom_lddt / (torch.sum(pair_mask, dim=-1, keepdim=True) + self.eps)
        boundaries = torch.linspace(start=self.min_bin, end=self.max_bin, steps=self.no_bins + 1, device=true_coordinate.device)
        true_bins = torch.sum(per_atom_lddt > boundaries, dim=-1)
        true_bins = torch.clamp(true_bins, min=1, max=self.no_bins)
        true_bins = F.one_hot(true_bins - 1, self.no_bins)
        return true_bins

    def forward(self, logits: 'torch.Tensor', pred_coordinate: 'torch.Tensor', true_coordinate: 'torch.Tensor', coordinate_mask: 'torch.Tensor', is_nucleotide: 'torch.Tensor', is_polymer: 'torch.Tensor', rep_atom_mask: 'torch.Tensor') ->torch.Tensor:
        """PLDDT loss

        Args:
            logits (torch.Tensor): logits
                [..., N_sample, N_atom, no_bins:=50]
            pred_coordinate (torch.Tensor): predicted coordinates
                [..., N_sample, N_atom, 3]
            true_coordinate (torch.Tensor): true coordinates
                [..., N_atom, 3]
            coordinate_mask (torch.Tensor): whether true coordinates exist
                [N_atom]
            is_nucleotide (torch.Tensor): "is_rna" or "is_dna"
                [N_atom]
            is_polymer (torch.Tensor): not "is_ligand"
                [N_atom]
            rep_atom_mask (torch.Tensor): representative atom of each token
                [N_atom]

        Returns:
            torch.Tensor: the return loss
                [...] if self.reduction is None else []
        """
        assert is_nucleotide.shape == is_polymer.shape == rep_atom_mask.shape == coordinate_mask.shape == coordinate_mask.view(-1).shape
        coordinate_mask = coordinate_mask.bool()
        rep_atom_mask = rep_atom_mask.bool()
        is_nucleotide = is_nucleotide.bool()
        is_polymer = is_polymer.bool()
        with torch.no_grad():
            true_bins = self.calculate_label(pred_coordinate=pred_coordinate[..., coordinate_mask, :], true_coordinate=true_coordinate[..., coordinate_mask, :], is_nucleotide=is_nucleotide[coordinate_mask], is_polymer=is_polymer[coordinate_mask], rep_atom_mask=rep_atom_mask[coordinate_mask]).detach()
        plddt_loss = softmax_cross_entropy(logits=logits[..., coordinate_mask, :], labels=true_bins)
        plddt_loss = plddt_loss.mean(dim=-1)
        plddt_loss = plddt_loss.mean(dim=-1)
        return loss_reduction(plddt_loss, method=self.reduction)


def compute_lddt_mask(true_distance: 'torch.Tensor', distance_mask: 'torch.Tensor', is_nucleotide: 'torch.Tensor', is_nucleotide_threshold: 'float'=30.0, is_not_nucleotide_threshold: 'float'=15.0) ->torch.Tensor:
    """calculate the atom pair mask with the bespoke radius

    Args:
        true_distance (torch.Tensor): the ground truth coordinates
            [..., N_atom, N_atom]
        distance_mask (torch.Tensor): whether true coordinates exist.
            [..., N_atom, N_atom] or [N_atom, N_atom]
        is_nucleotide (torch.Tensor): Indicator for nucleotide atoms.
            [..., N_atom] or [N_atom]
        is_nucleotide_threshold (float): Threshold distance for nucleotide atoms. Defaults to 30.0.
        is_not_nucleotide_threshold (float): Threshold distance for non-nucleotide atoms. Defaults to 15.0.

    Returns:
        c_lm (torch.Tenson): the atom pair mask c_lm, not symmetric
            [..., N_atom, N_atom]
    """
    is_nucleotide_mask = is_nucleotide.bool()
    c_lm = (true_distance < is_nucleotide_threshold) * is_nucleotide_mask[..., None] + (true_distance < is_not_nucleotide_threshold) * ~is_nucleotide_mask[..., None]
    c_lm = c_lm * (1 - torch.eye(n=c_lm.size(-1), device=c_lm.device, dtype=true_distance.dtype))
    c_lm = c_lm * distance_mask
    return c_lm


class ProtenixLoss(nn.Module):
    """Aggregation of the various losses"""

    def __init__(self, configs) ->None:
        super(ProtenixLoss, self).__init__()
        self.configs = configs
        self.alpha_confidence = self.configs.loss.weight.alpha_confidence
        self.alpha_pae = self.configs.loss.weight.alpha_pae
        self.alpha_except_pae = self.configs.loss.weight.alpha_except_pae
        self.alpha_diffusion = self.configs.loss.weight.alpha_diffusion
        self.alpha_distogram = self.configs.loss.weight.alpha_distogram
        self.alpha_bond = self.configs.loss.weight.alpha_bond
        self.weight_smooth_lddt = self.configs.loss.weight.smooth_lddt
        self.lddt_radius = {'is_nucleotide_threshold': 30.0, 'is_not_nucleotide_threshold': 15.0}
        self.loss_weight = {'plddt_loss': self.alpha_confidence * self.alpha_except_pae, 'pde_loss': self.alpha_confidence * self.alpha_except_pae, 'resolved_loss': self.alpha_confidence * self.alpha_except_pae, 'pae_loss': self.alpha_confidence * self.alpha_pae, 'mse_loss': self.alpha_diffusion, 'bond_loss': self.alpha_diffusion * self.alpha_bond, 'smooth_lddt_loss': self.alpha_diffusion * self.weight_smooth_lddt, 'distogram_loss': self.alpha_distogram}
        self.plddt_loss = PLDDTLoss(**configs.loss.plddt, **self.lddt_radius)
        self.pde_loss = PDELoss(**configs.loss.pde)
        self.resolved_loss = ExperimentallyResolvedLoss(**configs.loss.resolved)
        self.pae_loss = PAELoss(**configs.loss.pae)
        self.mse_loss = MSELoss(**configs.loss.diffusion.mse)
        self.bond_loss = BondLoss(**configs.loss.diffusion.bond)
        self.smooth_lddt_loss = SmoothLDDTLoss(**configs.loss.diffusion.smooth_lddt)
        self.distogram_loss = DistogramLoss(**configs.loss.distogram)

    def calculate_label(self, feat_dict: 'dict[str, Any]', label_dict: 'dict[str, Any]') ->dict[str, Any]:
        """calculate true distance, and atom pair mask

        Args:
            feat_dict (dict): Feature dictionary containing additional features.
            label_dict (dict): Label dictionary containing ground truth data.

        Returns:
            label_dict (dict): with the following updates:
                distance (torch.Tensor): true atom-atom distance.
                    [..., N_atom, N_atom]
                distance_mask (torch.Tensor): atom-atom mask indicating whether true distance exists.
                    [..., N_atom, N_atom]
        """
        distance_mask = label_dict['coordinate_mask'][..., None] * label_dict['coordinate_mask'][..., None, :]
        distance = cdist(label_dict['coordinate'], label_dict['coordinate']) * distance_mask
        lddt_mask = compute_lddt_mask(true_distance=distance, distance_mask=distance_mask, is_nucleotide=feat_dict['is_rna'].bool() + feat_dict['is_dna'].bool(), **self.lddt_radius)
        label_dict['lddt_mask'] = lddt_mask
        label_dict['distance_mask'] = distance_mask
        if not self.configs.loss_metrics_sparse_enable:
            label_dict['distance'] = distance
        del distance, distance_mask, lddt_mask
        return label_dict

    def calculate_prediction(self, pred_dict: 'dict[str, torch.Tensor]') ->dict[str, torch.Tensor]:
        """get more predictions used for calculating difference losses

        Args:
            pred_dict (dict[str, torch.Tensor]): raw prediction dict given by the model

        Returns:
            dict[str, torch.Tensor]: updated predictions
        """
        if not self.configs.loss_metrics_sparse_enable:
            pred_dict['distance'] = torch.cdist(pred_dict['coordinate'], pred_dict['coordinate'])
        return pred_dict

    def aggregate_losses(self, loss_fns: 'dict', has_valid_resolution: 'Optional[torch.Tensor]'=None) ->tuple[torch.Tensor, dict]:
        """
        Aggregates multiple loss functions and their respective metrics.

        Args:
            loss_fns (dict): Dictionary of loss functions to be aggregated.
            has_valid_resolution (Optional[torch.Tensor]): Tensor indicating valid resolutions. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - all_metrics (dict): Dictionary containing all metrics.
        """
        cum_loss = 0.0
        all_metrics = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.loss_weight[loss_name]
            loss_outputs = loss_fn()
            if isinstance(loss_outputs, tuple):
                loss, metrics = loss_outputs
            else:
                assert isinstance(loss_outputs, torch.Tensor)
                loss, metrics = loss_outputs, {}
            all_metrics.update({f'{loss_name}/{key}': val for key, val in metrics.items()})
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f'{loss_name} loss is NaN. Skipping...')
            if has_valid_resolution is not None and has_valid_resolution.sum() == 0 and loss_name in ['plddt_loss', 'pde_loss', 'resolved_loss', 'pae_loss']:
                loss = 0.0 * loss
            else:
                all_metrics[loss_name] = loss.detach().clone()
                all_metrics[f'weighted_{loss_name}'] = weight * loss.detach().clone()
            cum_loss = cum_loss + weight * loss
        all_metrics['loss'] = cum_loss.detach().clone()
        return cum_loss, all_metrics

    def calculate_losses(self, feat_dict: 'dict[str, Any]', pred_dict: 'dict[str, torch.Tensor]', label_dict: 'dict[str, Any]', mode: 'str'='train') ->tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculate the cumulative loss and aggregated metrics for the given predictions and labels.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - metrics (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        assert mode in ['train', 'eval', 'inference']
        if mode == 'train':
            confidence_coordinate = 'coordinate_mini'
            if not self.configs.train_confidence_only:
                diffusion_per_sample_scale = (pred_dict['noise_level'] ** 2 + self.configs.sigma_data ** 2) / (self.configs.sigma_data * pred_dict['noise_level']) ** 2
        else:
            confidence_coordinate = 'coordinate'
            diffusion_per_sample_scale = None
        if self.configs.train_confidence_only and mode == 'train':
            loss_fns = {}
        else:
            loss_fns = {}
            if self.configs.loss.diffusion_lddt_loss_dense:
                loss_fns.update({'smooth_lddt_loss': lambda : self.smooth_lddt_loss.dense_forward(pred_coordinate=pred_dict['coordinate'], true_coordinate=label_dict['coordinate'], lddt_mask=label_dict['lddt_mask'], diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size)})
            elif self.configs.loss.diffusion_sparse_loss_enable:
                loss_fns.update({'smooth_lddt_loss': lambda : self.smooth_lddt_loss.sparse_forward(pred_coordinate=pred_dict['coordinate'], true_coordinate=label_dict['coordinate'], lddt_mask=label_dict['lddt_mask'], diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size)})
            else:
                loss_fns.update({'smooth_lddt_loss': lambda : self.smooth_lddt_loss(pred_distance=pred_dict['distance'], true_distance=label_dict['distance'], distance_mask=label_dict['distance_mask'], lddt_mask=label_dict['lddt_mask'], diffusion_chunk_size=self.configs.loss.diffusion_lddt_chunk_size)})
            loss_fns.update({'bond_loss': lambda : self.bond_loss.sparse_forward(pred_coordinate=pred_dict['coordinate'], true_coordinate=label_dict['coordinate'], distance_mask=label_dict['distance_mask'], bond_mask=feat_dict['bond_mask'], per_sample_scale=diffusion_per_sample_scale) if self.configs.loss.diffusion_sparse_loss_enable else self.bond_loss(pred_distance=pred_dict['distance'], true_distance=label_dict['distance'], distance_mask=label_dict['distance_mask'], bond_mask=feat_dict['bond_mask'], per_sample_scale=diffusion_per_sample_scale, diffusion_chunk_size=self.configs.loss.diffusion_bond_chunk_size), 'mse_loss': lambda : self.mse_loss(pred_coordinate=pred_dict['coordinate'], true_coordinate=label_dict['coordinate'], coordinate_mask=label_dict['coordinate_mask'], is_rna=feat_dict['is_rna'], is_dna=feat_dict['is_dna'], is_ligand=feat_dict['is_ligand'], per_sample_scale=diffusion_per_sample_scale)})
            if 'distogram' in pred_dict:
                loss_fns.update({'distogram_loss': lambda : self.distogram_loss(logits=pred_dict['distogram'], true_coordinate=label_dict['coordinate'], coordinate_mask=label_dict['coordinate_mask'], rep_atom_mask=feat_dict['distogram_rep_atom_mask'])})
        resolution = feat_dict['resolution'].item()
        has_valid_resolution = (resolution >= self.configs.loss.resolution.min) & (resolution <= self.configs.loss.resolution.max)
        if has_valid_resolution:
            has_valid_resolution = torch.tensor([1.0], dtype=label_dict['coordinate'].dtype, device=label_dict['coordinate'].device)
        else:
            has_valid_resolution = torch.tensor([0.0], dtype=label_dict['coordinate'].dtype, device=label_dict['coordinate'].device)
        if all(x in pred_dict for x in ['plddt', 'pde', 'pae', 'resolved']):
            loss_fns.update({'plddt_loss': lambda : self.plddt_loss(logits=pred_dict['plddt'], pred_coordinate=pred_dict[confidence_coordinate].detach(), true_coordinate=label_dict['coordinate'], coordinate_mask=label_dict['coordinate_mask'], rep_atom_mask=feat_dict['plddt_m_rep_atom_mask'], is_nucleotide=feat_dict['is_rna'] + feat_dict['is_dna'], is_polymer=1 - feat_dict['is_ligand']), 'pde_loss': lambda : self.pde_loss(logits=pred_dict['pde'], pred_coordinate=pred_dict[confidence_coordinate].detach(), true_coordinate=label_dict['coordinate'], coordinate_mask=label_dict['coordinate_mask'], rep_atom_mask=feat_dict['distogram_rep_atom_mask']), 'resolved_loss': lambda : self.resolved_loss(logits=pred_dict['resolved'], coordinate_mask=label_dict['coordinate_mask']), 'pae_loss': lambda : self.pae_loss(logits=pred_dict['pae'], pred_coordinate=pred_dict[confidence_coordinate].detach(), true_coordinate=label_dict['coordinate'], coordinate_mask=label_dict['coordinate_mask'], frame_atom_index=feat_dict['frame_atom_index'], rep_atom_mask=feat_dict['pae_rep_atom_mask'], has_frame=feat_dict['has_frame'])})
        cum_loss, metrics = self.aggregate_losses(loss_fns, has_valid_resolution)
        return cum_loss, metrics

    def forward(self, feat_dict: 'dict[str, Any]', pred_dict: 'dict[str, torch.Tensor]', label_dict: 'dict[str, Any]', mode: 'str'='train') ->tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass for calculating the cumulative loss and aggregated metrics.

        Args:
            feat_dict (dict[str, Any]): Feature dictionary containing additional features.
            pred_dict (dict[str, torch.Tensor]): Prediction dictionary containing model outputs.
            label_dict (dict[str, Any]): Label dictionary containing ground truth data.
            mode (str): Mode of operation ('train', 'eval', 'inference'). Defaults to 'train'.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - losses (dict[str, torch.Tensor]): Dictionary containing aggregated metrics.
        """
        diffusion_chunk_size = self.configs.loss.diffusion_chunk_size_outer
        assert mode in ['train', 'eval', 'inference']
        with torch.no_grad():
            label_dict = self.calculate_label(feat_dict, label_dict)
        pred_dict = self.calculate_prediction(pred_dict)
        if diffusion_chunk_size <= 0:
            cum_loss, losses = self.calculate_losses(feat_dict=feat_dict, pred_dict=pred_dict, label_dict=label_dict, mode=mode)
        else:
            if 'coordinate' in pred_dict:
                N_sample = pred_dict['coordinate'].shape[-3]
            elif self.configs.train_confidence_only:
                N_sample = pred_dict['coordinate_mini'].shape[-3]
            else:
                raise KeyError('Missing key: coordinate (in pred_dict).')
            no_chunks = N_sample // diffusion_chunk_size + (N_sample % diffusion_chunk_size != 0)
            cum_loss = 0.0
            losses = {}
            for i in range(no_chunks):
                cur_sample_num = min(diffusion_chunk_size, N_sample - i * diffusion_chunk_size)
                pred_dict_i = {}
                for key, value in pred_dict.items():
                    if key in ['coordinate'] and mode == 'train':
                        pred_dict_i[key] = value[i * diffusion_chunk_size:(i + 1) * diffusion_chunk_size, :, :]
                    elif key in ['coordinate', 'plddt', 'pae', 'pde', 'resolved'] and mode != 'train':
                        pred_dict_i[key] = value[i * diffusion_chunk_size:(i + 1) * diffusion_chunk_size, :, :]
                    elif key == 'noise_level':
                        pred_dict_i[key] = value[i * diffusion_chunk_size:(i + 1) * diffusion_chunk_size]
                    else:
                        pred_dict_i[key] = value
                pred_dict_i = self.calculate_prediction(pred_dict_i)
                cum_loss_i, losses_i = self.calculate_losses(feat_dict=feat_dict, pred_dict=pred_dict_i, label_dict=label_dict, mode=mode)
                cum_loss += cum_loss_i * cur_sample_num
                for key, value in losses_i.items():
                    if key in losses:
                        losses[key] += value * cur_sample_num
                    else:
                        losses[key] = value * cur_sample_num
            cum_loss /= N_sample
            for key in losses.keys():
                losses[key] /= N_sample
        return cum_loss, losses


class OpenFoldLayerNorm(nn.Module):

    def __init__(self, c_in, eps=1e-05):
        super(OpenFoldLayerNorm, self).__init__()
        self.c_in = c_in,
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        deepspeed_is_initialized = deepspeed_is_installed and deepspeed.comm.comm.is_initialized()
        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(x, self.c_in, self.weight, self.bias, self.eps)
        else:
            out = nn.functional.layer_norm(x, self.c_in, self.weight, self.bias, self.eps)
        return out


def LayerNorm(c_in, eps: 'float'=1e-05):
    if fastln_is_installed:
        return FusedLayerNorm(c_in, eps)
    return OpenFoldLayerNorm(c_in, eps)


LinearNoBias = partial(Linear, bias=False)


class AdaptiveLayerNorm(nn.Module):
    """
    Implements Algorithm 26 in AF3
    """

    def __init__(self, c_a: 'int'=768, c_s: 'int'=384) ->None:
        """
        Args:
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        """
        super(AdaptiveLayerNorm, self).__init__()
        self.layernorm_a = nn.LayerNorm(c_a, elementwise_affine=False, bias=False)
        self.layernorm_s = nn.LayerNorm(c_s, bias=False)
        self.linear_s = Linear(in_features=c_s, out_features=c_a)
        self.linear_nobias_s = LinearNoBias(in_features=c_s, out_features=c_a)

    def zero_init(self):
        nn.init.zeros_(self.linear_s.weight)
        nn.init.zeros_(self.linear_s.bias)
        nn.init.zeros_(self.linear_nobias_s.weight)

    def forward(self, a: 'torch.Tensor', s: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]

        Returns:
            torch.Tensor: the updated a from AdaLN
                [..., N_token, c_a]
        """
        a = self.layernorm_a(a)
        s = self.layernorm_s(s)
        a = torch.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)
        return a


DEFAULT_LMA_KV_CHUNK_SIZE = 4096


DEFAULT_LMA_Q_CHUNK_SIZE = 1024


def permute_final_dims(tensor: 'torch.Tensor', inds: 'List[int]'):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [(zero_index + i) for i in inds])


@torch.jit.ignore
def softmax_no_cast(t: 'torch.Tensor', dim: 'int'=-1) ->torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of
    type bfloat16
    """
    d = t.dtype
    deepspeed_is_initialized = deepspeed_is_installed and deepspeed.comm.comm.is_initialized()
    if d is torch.bfloat16 and not deepspeed_is_initialized:
        with torch.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)
    return s


def _attention(query: 'torch.Tensor', key: 'torch.Tensor', value: 'torch.Tensor', biases: 'List[torch.Tensor]') ->torch.Tensor:
    key = permute_final_dims(key, (1, 0))
    a = torch.matmul(query, key)
    for b in biases:
        a += b
    a = softmax_no_cast(a, -1)
    a = torch.matmul(a, value)
    return a


@torch.jit.ignore
def _deepspeed_evo_attn(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', biases: 'List[torch.Tensor]'):
    """ ""
    Compute attention using the DeepSpeed DS4Sci_EvoformerAttention kernel.

    Args:
        q:
            [*, H, Q, C_hidden] query data
        k:
            [*, H, K, C_hidden] key data
        v:
            [*, H, V, C_hidden] value data
        biases:
            List of biases that broadcast to [*, H, Q, K]
    """
    if not ds4s_is_installed:
        raise ValueError('_deepspeed_evo_attn requires that DeepSpeed be installed and that the deepspeed.ops.deepspeed4science package exists')

    def reshape_dims(x):
        no_batch_dims = len(x.shape[:-3])
        if no_batch_dims < 2:
            return x.reshape(*((1,) * (2 - no_batch_dims) + x.shape))
        if no_batch_dims > 2:
            return x.reshape(*((x.shape[0], -1) + x.shape[-3:]))
        return x
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)
    orig_shape = q.shape
    if len(orig_shape[:-3]) != 2:
        q = reshape_dims(q)
        k = reshape_dims(k)
        v = reshape_dims(v)
        biases = [reshape_dims(b) for b in biases]
    orig_dtype = q.dtype
    if orig_dtype not in [torch.bfloat16, torch.float16]:
        o = DS4Sci_EvoformerAttention(q, k, v, [b for b in biases])
        o = o
    else:
        o = DS4Sci_EvoformerAttention(q, k, v, biases)
    o = o.reshape(orig_shape)
    return o


@torch.jit.ignore
def _flash_attn(q, k, v, kv_mask):
    if not fa_is_installed:
        raise ValueError('_flash_attn requires that FlashAttention be installed')
    batch_dims = q.shape[:-3]
    no_heads, n, c = q.shape[-3:]
    dtype = q.dtype
    q = q.half()
    k = k.half()
    v = v.half()
    kv_mask = kv_mask.half()
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])
    batch_size = q.shape[0]
    q = q.reshape(-1, *q.shape[-2:])
    q_max_s = n
    q_cu_seqlens = torch.arange(0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=q.device)
    kv = torch.stack([k, v], dim=-3)
    kv_shape = kv.shape
    kv = kv.reshape(*kv.shape[:-3], -1)
    kv_unpad, _, kv_cu_seqlens, kv_max_s = unpad_input(kv, kv_mask)
    kv_unpad = kv_unpad.reshape(-1, *kv_shape[-3:])
    out = flash_attn_unpadded_kvpacked_func(q, kv_unpad, q_cu_seqlens, kv_cu_seqlens, q_max_s, kv_max_s, dropout_p=0.0, softmax_scale=1.0)
    out = out.reshape(*batch_dims, n, no_heads, c)
    out = out
    return out


def _lma(q: 'torch.Tensor', k: 'torch.Tensor', v: 'torch.Tensor', biases: 'List[torch.Tensor]', q_chunk_size: 'int', kv_chunk_size: 'int'):
    no_q, no_kv = q.shape[-2], k.shape[-2]
    o = q.new_zeros(q.shape)
    for q_s in range(0, no_q, q_chunk_size):
        q_chunk = q[..., q_s:q_s + q_chunk_size, :]
        large_bias_chunks = [b[..., q_s:q_s + q_chunk_size, :] for b in biases]
        maxes = []
        weights = []
        values = []
        for kv_s in range(0, no_kv, kv_chunk_size):
            k_chunk = k[..., kv_s:kv_s + kv_chunk_size, :]
            v_chunk = v[..., kv_s:kv_s + kv_chunk_size, :]
            small_bias_chunks = [b[..., kv_s:kv_s + kv_chunk_size] for b in large_bias_chunks]
            a = torch.einsum('...hqd,...hkd->...hqk', q_chunk, k_chunk)
            for b in small_bias_chunks:
                a += b
            max_a = torch.max(a, dim=-1, keepdim=True)[0]
            exp_a = torch.exp(a - max_a)
            exp_v = torch.einsum('...hvf,...hqv->...hqf', v_chunk, exp_a)
            maxes.append(max_a.detach().squeeze(-1))
            weights.append(torch.sum(exp_a, dim=-1))
            values.append(exp_v)
        chunk_max = torch.stack(maxes, dim=-3)
        chunk_weights = torch.stack(weights, dim=-3)
        chunk_values = torch.stack(values, dim=-4)
        global_max = torch.max(chunk_max, dim=-3, keepdim=True)[0]
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values = chunk_values * max_diffs.unsqueeze(-1)
        chunk_weights = chunk_weights * max_diffs
        all_values = torch.sum(chunk_values, dim=-4)
        all_weights = torch.sum(chunk_weights.unsqueeze(-1), dim=-4)
        q_chunk_out = all_values / all_weights
        o[..., q_s:q_s + q_chunk_size, :] = q_chunk_out
    return o


SUPPORTED_DTYPES = [torch.float32, torch.bfloat16]


class AttentionCoreFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, bias_1=None, bias_2=None):
        if bias_1 is None and bias_2 is not None:
            raise ValueError('bias_1 must be specified before bias_2')
        if q.dtype not in SUPPORTED_DTYPES:
            raise ValueError('Unsupported datatype')
        q = q.contiguous()
        k = k.contiguous()
        attention_logits = torch.matmul(q, k.transpose(-1, -2))
        if bias_1 is not None:
            attention_logits += bias_1
        if bias_2 is not None:
            attention_logits += bias_2
        attn_core_inplace_cuda.forward_(attention_logits, reduce(mul, attention_logits.shape[:-1]), attention_logits.shape[-1])
        o = torch.matmul(attention_logits, v)
        ctx.bias_1_shape = bias_1.shape if bias_1 is not None else None
        ctx.bias_2_shape = bias_2.shape if bias_2 is not None else None
        ctx.save_for_backward(q, k, v, attention_logits)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, attention_logits = ctx.saved_tensors
        grad_q = grad_k = grad_v = grad_bias_1 = grad_bias_2 = None
        grad_v = torch.matmul(attention_logits.transpose(-1, -2), grad_output)
        attn_core_inplace_cuda.backward_(attention_logits, grad_output.contiguous(), v.contiguous(), reduce(mul, attention_logits.shape[:-1]), attention_logits.shape[-1], grad_output.shape[-1])
        if ctx.bias_1_shape is not None:
            grad_bias_1 = torch.sum(attention_logits, dim=tuple(i for i, d in enumerate(ctx.bias_1_shape) if d == 1), keepdim=True)
        if ctx.bias_2_shape is not None:
            grad_bias_2 = torch.sum(attention_logits, dim=tuple(i for i, d in enumerate(ctx.bias_2_shape) if d == 1), keepdim=True)
        grad_q = torch.matmul(attention_logits, k)
        grad_k = torch.matmul(q.transpose(-1, -2), attention_logits).transpose(-1, -2)
        return grad_q, grad_k, grad_v, grad_bias_1, grad_bias_2


attention_core = AttentionCoreFunction.apply


def flatten_final_dims(t: 'torch.Tensor', no_dims: 'int'):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def is_fp16_enabled():
    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()
    return fp16_enabled


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(self, c_q: 'int', c_k: 'int', c_v: 'int', c_hidden: 'int', no_heads: 'int', gating: 'bool'=True):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating
        self.linear_q = Linear(self.c_q, self.c_hidden * self.no_heads, bias=False, init='glorot')
        self.linear_k = Linear(self.c_k, self.c_hidden * self.no_heads, bias=False, init='glorot')
        self.linear_v = Linear(self.c_v, self.c_hidden * self.no_heads, bias=False, init='glorot')
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q, init='final')
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(self.c_q, self.c_hidden * self.no_heads, init='gating')
        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self, q_x: 'torch.Tensor', kv_x: 'torch.Tensor', apply_scale: 'bool'=True) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        if apply_scale:
            q /= math.sqrt(self.c_hidden)
        return q, k, v

    def _wrap_up(self, o: 'torch.Tensor', q_x: 'torch.Tensor') ->torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g
        o = flatten_final_dims(o, 2)
        o = self.linear_o(o)
        return o

    def forward(self, q_x: 'torch.Tensor', kv_x: 'torch.Tensor', biases: 'Optional[List[torch.Tensor]]'=None, use_memory_efficient_kernel: 'bool'=False, use_deepspeed_evo_attention: 'bool'=False, use_lma: 'bool'=False, lma_q_chunk_size: 'int'=DEFAULT_LMA_Q_CHUNK_SIZE, lma_kv_chunk_size: 'int'=DEFAULT_LMA_KV_CHUNK_SIZE, use_flash: 'bool'=False, flash_mask: 'Optional[torch.Tensor]'=None) ->torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_memory_efficient_kernel:
                Whether to use a custom memory-efficient attention kernel.
                This should be the default choice for most. If none of the
                "use_<...>" flags are True, a stock PyTorch implementation
                is used instead
            use_deepspeed_evo_attention:
                Whether to use DeepSpeed memory-efficient attention kernel.
                If none of the "use_<...>" flags are True, a stock PyTorch
                implementation is used instead
            use_lma:
                Whether to use low-memory attention (Staats & Rabe 2021). If
                none of the "use_<...>" flags are True, a stock PyTorch
                implementation is used instead
            lma_q_chunk_size:
                Query chunk size (for LMA)
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        if use_lma and (lma_q_chunk_size is None or lma_kv_chunk_size is None):
            raise ValueError('If use_lma is specified, lma_q_chunk_size and lma_kv_chunk_size must be provided')
        if use_flash and biases is not None:
            raise ValueError('use_flash is incompatible with the bias option. For masking, use flash_mask instead')
        attn_options = [use_memory_efficient_kernel, use_deepspeed_evo_attention, use_lma, use_flash]
        if sum(attn_options) > 1:
            raise ValueError('Choose at most one alternative attention algorithm')
        if biases is None:
            biases = []
        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=not use_deepspeed_evo_attention)
        if is_fp16_enabled():
            use_memory_efficient_kernel = False
        if use_memory_efficient_kernel:
            raise Exception(f'use_memory_efficient_kernel=True not supported!!!')
            if len(biases) > 2:
                raise ValueError('If use_memory_efficient_kernel is True, you may only provide up to two bias terms')
            o = attention_core(q, k, v, *(biases + [None] * 2)[:2])
            o = o.transpose(-2, -3)
        elif use_deepspeed_evo_attention:
            if len(biases) > 2:
                raise ValueError('If use_deepspeed_evo_attention is True, you may only provide up to two bias terms')
            o = _deepspeed_evo_attn(q, k, v, biases)
        elif use_lma:
            biases = [b.expand(b.shape[:-2] + (q_x.shape[-2],) + (kv_x.shape[-2],)) for b in biases]
            o = _lma(q, k, v, biases, lma_q_chunk_size, lma_kv_chunk_size)
            o = o.transpose(-2, -3)
        elif use_flash:
            o = _flash_attn(q, k, v, flash_mask)
        else:
            o = _attention(q, k, v, biases)
            o = o.transpose(-2, -3)
        o = self._wrap_up(o, q_x)
        return o


class BiasInitLinear(Linear):
    """Support biasinit for nn.Linear Called just like torch.nn.Linear."""

    def __init__(self, in_features: 'int', out_features: 'int', bias: 'bool'=True, biasinit: 'float'=0.0) ->None:
        """
        Args:
            in_features (int): in_features
            out_features (int): out_features
            bias (bool, optional): whether add bias. Defaults to True.
            biasinit (float, optional): the initial bias value. Defaults to 0.0.
        """
        super(BiasInitLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        nn.init.zeros_(tensor=self.weight)
        if bias:
            nn.init.constant_(tensor=self.bias, val=biasinit)


class AttentionPairBias(nn.Module):
    """
    Implements Algorithm 24 in AF3
    """

    def __init__(self, has_s: 'bool'=True, n_heads: 'int'=16, c_a: 'int'=768, c_s: 'int'=384, c_z: 'int'=128, biasinit: 'float'=-2.0) ->None:
        """
        Args:
            has_s (bool, optional):  whether s is None as stated in Algorithm 24 Line1. Defaults to True.
            n_heads (int, optional): number of attention-like head in AttentionPairBias. Defaults to 16.
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            biasinit (float, optional): biasinit for BiasInitLinear. Defaults to -2.0.
        """
        super(AttentionPairBias, self).__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        if has_s:
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
            self.linear_a_last = BiasInitLinear(in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit)
        else:
            self.layernorm_a = LayerNorm(c_a)
        self.local_attention_method = 'local_cross_attention'
        self.attention = Attention(c_q=c_a, c_k=c_a, c_v=c_a, c_hidden=c_a // n_heads, num_heads=n_heads, gating=True, q_linear_bias=True, local_attention_method=self.local_attention_method)
        self.layernorm_z = LayerNorm(c_z)
        self.linear_nobias_z = LinearNoBias(in_features=c_z, out_features=n_heads)

    def glorot_init(self):
        nn.init.xavier_uniform_(self.attention.linear_q.weight)
        nn.init.xavier_uniform_(self.attention.linear_k.weight)
        nn.init.xavier_uniform_(self.attention.linear_v.weight)
        nn.init.zeros_(self.attention.linear_q.bias)

    def local_multihead_attention(self, a: 'torch.Tensor', s: 'torch.Tensor', z: 'torch.Tensor', n_queries: 'int'=32, n_keys: 'int'=128, inplace_safe: 'bool'=False, chunk_size: 'Optional[int]'=None) ->torch.Tensor:
        """Used by Algorithm 24, with beta_ij being the local mask. Used in AtomTransformer.

        Args:
            a (torch.Tensor): atom embedding
                [..., N_atom, c_a]
            s (torch.Tensor): atom embedding
                [..., N_atom, c_s]
            z (torch.Tensor): atom-atom pair embedding, in trunked dense shape. Used for computing pair bias.
                [..., n_blocks, n_queries, n_keys, c_z]
            n_queries (int, optional): local window size of query tensor. Defaults to 32.
            n_keys (int, optional): local window size of key tensor. Defaults to 128.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_atom, c_a]
        """
        assert n_queries == z.size(-3)
        assert n_keys == z.size(-2)
        assert len(z.shape) == len(a.shape) + 2
        bias = self.linear_nobias_z(self.layernorm_z(z))
        bias = permute_final_dims(bias, [3, 0, 1, 2])
        a = self.attention(q_x=a, kv_x=a, trunked_attn_bias=bias, n_queries=n_queries, n_keys=n_keys, inplace_safe=inplace_safe, chunk_size=chunk_size)
        return a

    def standard_multihead_attention(self, a: 'torch.Tensor', s: 'torch.Tensor', z: 'torch.Tensor', inplace_safe: 'bool'=False) ->torch.Tensor:
        """Used by Algorithm 7/20

        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]
            z (torch.Tensor): pair embedding, used for computing pair bias.
                [..., N_token, N_token, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.

        Returns:
            torch.Tensor: the updated a from AttentionPairBias
                [..., N_token, c_a]
        """
        bias = self.linear_nobias_z(self.layernorm_z(z))
        bias = permute_final_dims(bias, [2, 0, 1])
        a = self.attention(q_x=a, kv_x=a, attn_bias=bias, inplace_safe=inplace_safe)
        return a

    def forward(self, a: 'torch.Tensor', s: 'torch.Tensor', z: 'torch.Tensor', n_queries: 'Optional[int]'=None, n_keys: 'Optional[int]'=None, inplace_safe: 'bool'=False, chunk_size: 'Optional[int]'=None) ->torch.Tensor:
        """Details are given in local_forward and standard_forward"""
        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)
        if n_queries and n_keys:
            a = self.local_multihead_attention(a, s, z, n_queries, n_keys, inplace_safe=inplace_safe, chunk_size=chunk_size)
        else:
            a = self.standard_multihead_attention(a, s, z, inplace_safe=inplace_safe)
        if self.has_s:
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a
        return a


class Dropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.

    If not in training mode, this module computes the identity function.
    """

    def __init__(self, r: 'float', batch_dim: 'Union[int, List[int]]'):
        """
        Args:
            r:
                Dropout rate
            batch_dim:
                Dimension(s) along which the dropout mask is shared
        """
        super(Dropout, self).__init__()
        self.r = r
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        self.dropout = nn.Dropout(self.r)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x:
                Tensor to which dropout is applied. Can have any shape
                compatible with self.batch_dim
        """
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = x.new_ones(shape)
        mask = self.dropout(mask)
        x *= mask
        return x


class DropoutRowwise(Dropout):
    """
    Convenience class for rowwise dropout as described in subsection
    1.11.6.
    """
    __init__ = partialmethod(Dropout.__init__, batch_dim=-3)


class Transition(nn.Module):
    """
    Implements Algorithm 11 in AF3
    """

    def __init__(self, c_in: 'int', n: 'int') ->None:
        """
        Args:
            c_in (int, optional): the input dimension.
            n (int, optional): factor by which c_in is multiplied to obtain hidden dimension.
        """
        super(Transition, self).__init__()
        self.n = n
        self.c_in = c_in
        self.layernorm1 = LayerNorm(c_in)
        self.linear_no_bias_a = LinearNoBias(in_features=c_in, out_features=n * c_in)
        self.linear_no_bias_b = LinearNoBias(in_features=c_in, out_features=n * c_in)
        self.linear_no_bias = LinearNoBias(in_features=n * c_in, out_features=c_in)
        self.zero_init()

    def zero_init(self):
        nn.init.zeros_(self.linear_no_bias.weight)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor
                [..., c]

        Returns:
            torch.Tensor: the output tensor as the same shape of x
                [..., c]
        """
        if self.training:
            x = self.layernorm1(x)
            a = self.linear_no_bias_a(x)
            b = self.linear_no_bias_b(x)
            x = self.linear_no_bias(F.silu(a) * b)
            return x
        else:
            other_dims = x.shape[:-1]
            dim_size = x.shape[-1]
            size = x.shape[-2]
            x = x.reshape(-1, dim_size)
            chunk_num = 1 if size < 3200 else 8
            chunks = torch.chunk(x, chunk_num, dim=-2)
            outputs = torch.empty((x.shape[0], self.c_in), dtype=x.dtype, device=x.device)
            start = 0
            for chunk in chunks:
                y = self.layernorm1(chunk)
                a = self.linear_no_bias_a(y)
                a = F.silu(a, True)
                b = self.linear_no_bias_b(y)
                del y
                b *= a
                del a
                b = self.linear_no_bias(b)
                outputs[start:start + b.shape[0]] = b
                start += b.shape[0]
                del b
            outputs = outputs.reshape(*other_dims, self.c_in)
            return outputs


@torch.jit.ignore
def _flat_idx_to_idx(flat_idx: 'int', dims: 'Tuple[int]') ->Tuple[int]:
    idx = []
    for d in reversed(dims):
        idx.append(flat_idx % d)
        flat_idx = flat_idx // d
    return tuple(reversed(idx))


@torch.jit.ignore
def _get_minimal_slice_set(start: 'Sequence[int]', end: 'Sequence[int]', dims: 'int', start_edges: 'Optional[Sequence[bool]]'=None, end_edges: 'Optional[Sequence[bool]]'=None) ->Sequence[Tuple[int]]:
    """
    Produces an ordered sequence of tensor slices that, when used in
    sequence on a tensor with shape dims, yields tensors that contain every
    leaf in the contiguous range [start, end]. Care is taken to yield a
    short sequence of slices, and perhaps even the shortest possible (I'm
    pretty sure it's the latter).

    end is INCLUSIVE.
    """

    def reduce_edge_list(l):
        tally = 1
        for i in range(len(l)):
            reversed_idx = -1 * (i + 1)
            l[reversed_idx] *= tally
            tally = l[reversed_idx]
    if start_edges is None:
        start_edges = [(s == 0) for s in start]
        reduce_edge_list(start_edges)
    if end_edges is None:
        end_edges = [(e == d - 1) for e, d in zip(end, dims)]
        reduce_edge_list(end_edges)
    if len(start) == 0:
        return [tuple()]
    elif len(start) == 1:
        return [(slice(start[0], end[0] + 1),)]
    slices = []
    path = []
    for s, e in zip(start, end):
        if s == e:
            path.append(slice(s, s + 1))
        else:
            break
    path = tuple(path)
    divergence_idx = len(path)
    if divergence_idx == len(dims):
        return [tuple(path)]

    def upper():
        sdi = start[divergence_idx]
        return [(path + (slice(sdi, sdi + 1),) + s) for s in _get_minimal_slice_set(start[divergence_idx + 1:], [(d - 1) for d in dims[divergence_idx + 1:]], dims[divergence_idx + 1:], start_edges=start_edges[divergence_idx + 1:], end_edges=[(1) for _ in end_edges[divergence_idx + 1:]])]

    def lower():
        edi = end[divergence_idx]
        return [(path + (slice(edi, edi + 1),) + s) for s in _get_minimal_slice_set([(0) for _ in start[divergence_idx + 1:]], end[divergence_idx + 1:], dims[divergence_idx + 1:], start_edges=[(1) for _ in start_edges[divergence_idx + 1:]], end_edges=end_edges[divergence_idx + 1:])]
    if start_edges[divergence_idx] and end_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx] + 1),))
    elif start_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx]),))
        slices.extend(lower())
    elif end_edges[divergence_idx]:
        slices.extend(upper())
        slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),))
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if middle_ground > 1:
            slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx]),))
        slices.extend(lower())
    return [tuple(s) for s in slices]


@torch.jit.ignore
def _chunk_slice(t: 'torch.Tensor', flat_start: 'int', flat_end: 'int', no_batch_dims: 'int') ->torch.Tensor:
    """
    Equivalent to

        t.reshape((-1,) + t.shape[no_batch_dims:])[flat_start:flat_end]

    but without the need for the initial reshape call, which can be
    memory-intensive in certain situations. The only reshape operations
    in this function are performed on sub-tensors that scale with
    (flat_end - flat_start), the chunk size.
    """
    batch_dims = t.shape[:no_batch_dims]
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))
    slices = _get_minimal_slice_set(start_idx, end_idx, batch_dims)
    sliced_tensors = [t[s] for s in slices]
    return torch.cat([s.view((-1,) + t.shape[no_batch_dims:]) for s in sliced_tensors])


def _fetch_dims(tree):
    shapes = []
    tree_type = type(tree)
    if tree_type is dict:
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif tree_type is list or tree_type is tuple:
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif tree_type is torch.Tensor:
        shapes.append(tree.shape)
    else:
        raise ValueError('Not supported')
    return shapes

