import sys
_module = sys.modules[__name__]
del sys
baselines = _module
data = _module
avazu = _module
custom = _module
dlrm_dataloader = _module
synth = _module
dlrm_main = _module
models = _module
deepfm = _module
dlrm = _module
benchmark_cache = _module
benchmark_fbgemm_uvm = _module
data_utils = _module
recsys = _module
datasets = _module
avazu = _module
criteo = _module
feature_counter = _module
utils = _module
dlrm_main = _module
dlrm = _module
dataloader = _module
base_dataiter = _module
cuda_stream_dataloader = _module
misc = _module
preprocess_synth = _module
npy_preproc_avazu = _module
npy_preproc_criteo = _module
split_criteo_kaggle = _module
csv_to_txt = _module
txt_to_npz = _module

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


import time


from typing import Any


from typing import Callable


from typing import Dict


from typing import Iterable


from typing import Iterator


from typing import List


from typing import Optional


from typing import Tuple


from typing import Union


import numpy as np


import torch


from torch.utils.data import IterableDataset


from torch import distributed as dist


from torch.utils.data import DataLoader


from torch.autograd.profiler import record_function


import itertools


from typing import cast


from torch import nn


from torch.profiler import profile


from torch.profiler import record_function


from torch.profiler import ProfilerActivity


from torch.profiler import schedule


from torch.profiler import tensorboard_trace_handler


import torch.utils.data.datapipes as dp


from torch.utils.data import IterDataPipe


import random


import torch.distributed as dist


import torch.nn as nn


from torch.nn.parallel import DistributedDataParallel as DDP


from abc import ABC


from abc import abstractmethod


from typing import TypeVar


from torch.utils.data import Sampler


from torch.utils.data import Dataset


from time import perf_counter


class SparseArch(nn.Module):
    """
    Processes the sparse features of DLRM. Does embedding lookups for all EmbeddingBag
    and embedding features of each collection.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): represents a collection of
            pooled embeddings.

    Example::

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(embedding_bag_collection)

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f2"],
           values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
           offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sparse_embeddings = sparse_arch(features)
    """

    def __init__(self, embedding_bag_collection: 'EmbeddingBagCollection') ->None:
        super().__init__()
        self.embedding_bag_collection: 'EmbeddingBagCollection' = embedding_bag_collection
        emb_config = self.embedding_bag_collection.embedding_bag_configs()
        assert emb_config, 'Embedding bag collection cannot be empty!'
        self.D: 'int' = emb_config[0].embedding_dim
        self._sparse_feature_names: 'List[str]' = [name for conf in embedding_bag_collection.embedding_bag_configs() for name in conf.feature_names]
        self.F: 'int' = len(self._sparse_feature_names)

    def forward(self, features: 'KeyedJaggedTensor') ->torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor): an input tensor of sparse features.

        Returns:
            torch.Tensor: tensor of shape B X F X D.
        """
        sparse_features: 'KeyedTensor' = self.embedding_bag_collection(features)
        B: 'int' = features.stride()
        sparse: 'Dict[str, torch.Tensor]' = sparse_features.to_dict()
        sparse_values: 'List[torch.Tensor]' = []
        for name in self.sparse_feature_names:
            sparse_values.append(sparse[name])
        return torch.cat(sparse_values, dim=1).view(B, self.F, self.D)

    @property
    def sparse_feature_names(self) ->List[str]:
        return self._sparse_feature_names


class DenseArch(nn.Module):
    """
    Processes the dense features of DLRM model.

    Args:
        in_features (int): dimensionality of the dense input features.
        layer_sizes (List[int]): list of layer sizes.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        dense_arch = DenseArch(10, layer_sizes=[15, D])
        dense_embedded = dense_arch(torch.rand((B, 10)))
    """

    def __init__(self, in_features: 'int', layer_sizes: 'List[int]', device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        self.model: 'nn.Module' = MLP(in_features, layer_sizes, bias=True, activation='relu', device=device)

    def forward(self, features: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            features (torch.Tensor): an input tensor of dense features.

        Returns:
            torch.Tensor: an output tensor of size B X D.
        """
        return self.model(features)


class FMInteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features) and apply the general DeepFM interaction according to the
    external source of DeepFM paper: https://arxiv.org/pdf/1703.04247.pdf
    The output dimension is expected to be a cat of `dense_features`, D.
    Args:
        fm_in_features (int): the input dimension of `dense_module` in DeepFM. For
            example, if the input embeddings is [randn(3, 2, 3), randn(3, 4, 5)], then
            the `fm_in_features` should be: 2 * 3 + 4 * 5.
        sparse_feature_names (List[str]): length of F.
        deep_fm_dimension (int): output of the deep interaction (DI) in the DeepFM arch.
    Example::
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        fm_inter_arch = FMInteractionArch(sparse_feature_names=keys)
        dense_features = torch.rand((B, D))
        sparse_features = KeyedTensor(
            keys=keys,
            length_per_key=[D, D],
            values=torch.rand((B, D * F)),
        )
        cat_fm_output = fm_inter_arch(dense_features, sparse_features)
    """

    def __init__(self, fm_in_features: 'int', sparse_feature_names: 'List[str]', deep_fm_dimension: 'int') ->None:
        super().__init__()
        self.sparse_feature_names: 'List[str]' = sparse_feature_names
        self.deep_fm = DeepFM(dense_module=nn.Sequential(nn.Linear(fm_in_features, deep_fm_dimension), nn.ReLU()))
        self.fm = FactorizationMachine()

    def forward(self, dense_features: 'torch.Tensor', sparse_features: 'KeyedTensor') ->torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): tensor of size B X D.
            sparse_features (KeyedJaggedTensor): KJT of size F * D X B.
        Returns:
            torch.Tensor: an output tensor of size B X (D + DI + 1).
        """
        if len(self.sparse_feature_names) == 0:
            return dense_features
        tensor_list: 'List[torch.Tensor]' = [dense_features]
        for feature_name in self.sparse_feature_names:
            tensor_list.append(sparse_features[feature_name])
        deep_interaction = self.deep_fm(tensor_list)
        fm_interaction = self.fm(tensor_list)
        return torch.cat([dense_features, deep_interaction, fm_interaction], dim=1)


class OverArch(nn.Module):
    """
    Final Arch of DLRM - simple MLP over OverArch.

    Args:
        in_features (int): size of the input.
        layer_sizes (List[int]): sizes of the layers of the `OverArch`.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        over_arch = OverArch(10, [5, 1])
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(self, in_features: 'int', layer_sizes: 'List[int]', device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        if len(layer_sizes) <= 1:
            raise ValueError('OverArch must have multiple layers.')
        self.model: 'nn.Module' = nn.Sequential(MLP(in_features, layer_sizes[:-1], bias=True, activation='relu', device=device), nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True, device=device))

    def forward(self, features: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            features (torch.Tensor):

        Returns:
            torch.Tensor: size B X layer_sizes[-1]
        """
        return self.model(features)


class SimpleDeepFMNN(nn.Module):
    """
    Basic recsys module with DeepFM arch. Processes sparse features by
    learning pooled embeddings for each feature. Learns the relationship between
    dense features and sparse features by projecting dense features into the same
    embedding space. Learns the interaction among those dense and sparse features
    by deep_fm proposed in this paper: https://arxiv.org/pdf/1703.04247.pdf
    The module assumes all sparse features have the same embedding dimension
    (i.e, each `EmbeddingBagConfig` uses the same embedding_dim)
    The following notation is used throughout the documentation for the models:
    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features
    Args:
        num_dense_features (int): the number of input dense features.
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        hidden_layer_size (int): the hidden layer size used in dense module.
        deep_fm_dimension (int): the output layer size used in `deep_fm`'s deep
            interaction module.
    Example::
        B = 2
        D = 8
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_nn = SimpleDeepFMNN(
            embedding_bag_collection=ebc, hidden_layer_size=20, over_embedding_dim=5
        )
        features = torch.rand((B, 100))
        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
            offsets=torch.tensor([0, 2, 4, 6, 8]),
        )
        logits = sparse_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )
    """

    def __init__(self, num_dense_features: 'int', embedding_bag_collection: 'EmbeddingBagCollection', hidden_layer_size: 'int', deep_fm_dimension: 'int') ->None:
        super().__init__()
        assert len(embedding_bag_collection.embedding_bag_configs()) > 0, 'At least one embedding bag is required'
        for i in range(1, len(embedding_bag_collection.embedding_bag_configs())):
            conf_prev = embedding_bag_collection.embedding_bag_configs()[i - 1]
            conf = embedding_bag_collection.embedding_bag_configs()[i]
            assert conf_prev.embedding_dim == conf.embedding_dim, 'All EmbeddingBagConfigs must have the same dimension'
        embedding_dim: 'int' = embedding_bag_collection.embedding_bag_configs()[0].embedding_dim
        feature_names = []
        fm_in_features = embedding_dim
        for conf in embedding_bag_collection.embedding_bag_configs():
            for feat in conf.feature_names:
                feature_names.append(feat)
                fm_in_features += conf.embedding_dim
        self.sparse_arch = SparseArch(embedding_bag_collection)
        self.dense_arch = DenseArch(in_features=num_dense_features, hidden_layer_size=hidden_layer_size, embedding_dim=embedding_dim)
        self.inter_arch = FMInteractionArch(fm_in_features=fm_in_features, sparse_feature_names=feature_names, deep_fm_dimension=deep_fm_dimension)
        over_in_features = embedding_dim + deep_fm_dimension + 1
        self.over_arch = OverArch(over_in_features)

    def forward(self, dense_features: 'torch.Tensor', sparse_features: 'KeyedJaggedTensor') ->torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.
        Returns:
            torch.Tensor: logits with size B X 1.
        """
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features)
        concatenated_dense = self.inter_arch(dense_features=embedded_dense, sparse_features=embedded_sparse)
        logits = self.over_arch(concatenated_dense)
        return logits


class InteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the pairwise dot product of each sparse feature pair,
    the dot product of each sparse features with the output of the dense layer,
    and the dense layer itself (all concatenated).

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.


    Args:
        num_sparse_features (int): F.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=len(keys))

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (D + F + F choose 2)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(self, num_sparse_features: 'int', num_dense_features: 'int'=1) ->None:
        super().__init__()
        self.F: 'int' = num_sparse_features
        self.num_dense_features = num_dense_features
        self.register_buffer('triu_indices', torch.triu_indices(self.F + self.num_dense_features, self.F + self.num_dense_features, offset=1).requires_grad_(False), False)

    def forward(self, dense_features: 'torch.Tensor', sparse_features: 'torch.Tensor') ->torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (D + F + F choose 2).
        """
        if self.F <= 0:
            return dense_features
        if self.num_dense_features <= 0:
            combined_values = sparse_features
        else:
            combined_values = torch.cat((dense_features.unsqueeze(1), sparse_features), dim=1)
        interactions = torch.bmm(combined_values, torch.transpose(combined_values, 1, 2))
        interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]
        return torch.cat((dense_features, interactions_flat), dim=1)


def choose(n: 'int', k: 'int') ->int:
    """
    Simple implementation of math.comb for Python 3.7 compatibility.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


class DLRM(nn.Module):
    """
    Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e. each EmbeddingBagConfig uses the same embedding_dim).

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the `DenseArch`.
        over_arch_layer_sizes (List[int]): the layer sizes for the `OverArch`.
            The output dimension of the `InteractionArch` should not be manually
            specified here.
        dense_device (Optional[torch.device]): default compute device.

    Example::

        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2",
           embedding_dim=D,
           num_embeddings=100,
           feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = DLRM(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20],
           over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f3"],
           values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
           offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
           dense_features=features,
           sparse_features=sparse_features,
        )
    """

    def __init__(self, embedding_bag_collection: 'EmbeddingBagCollection', dense_in_features: 'int', dense_arch_layer_sizes: 'List[int]', over_arch_layer_sizes: 'List[int]', dense_device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        emb_configs = embedding_bag_collection.embedding_bag_configs()
        assert len(emb_configs) > 0, 'At least one embedding bag is required'
        for i in range(1, len(emb_configs)):
            conf_prev = emb_configs[i - 1]
            conf = emb_configs[i]
            assert conf_prev.embedding_dim == conf.embedding_dim, 'All EmbeddingBagConfigs must have the same dimension'
        embedding_dim: 'int' = emb_configs[0].embedding_dim
        if dense_arch_layer_sizes[-1] != embedding_dim:
            raise ValueError(f'embedding_bag_collection dimension ({embedding_dim}) and final dense arch layer size ({{dense_arch_layer_sizes[-1]}}) must match.')
        self.sparse_arch: 'SparseArch' = SparseArch(embedding_bag_collection)
        num_sparse_features: 'int' = len(self.sparse_arch.sparse_feature_names)
        self.dense_arch = DenseArch(in_features=dense_in_features, layer_sizes=dense_arch_layer_sizes, device=dense_device)
        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features)
        over_in_features: 'int' = embedding_dim + choose(num_sparse_features, 2) + num_sparse_features
        self.over_arch = OverArch(in_features=over_in_features, layer_sizes=over_arch_layer_sizes, device=dense_device)

    def forward(self, dense_features: 'torch.Tensor', sparse_features: 'KeyedJaggedTensor') ->torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits.
        """
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features)
        concatenated_dense = self.inter_arch(dense_features=embedded_dense, sparse_features=embedded_sparse)
        logits = self.over_arch(concatenated_dense)
        return logits


class DLRMTrain(nn.Module):
    """
    nn.Module to wrap DLRM model to use with train_pipeline.

    DLRM Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e, each EmbeddingBagConfig uses the same embedding_dim)

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define SparseArch.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (list[int]): the layer sizes for the DenseArch.
        over_arch_layer_sizes (list[int]): the layer sizes for the OverArch. NOTE: The
            output dimension of the InteractionArch should not be manually specified
            here.
        dense_device: (Optional[torch.device]).

    Call Args:
        batch: batch used with criteo and random data from torchrec.datasets

    Returns:
        Tuple[loss, Tuple[loss, logits, labels]]

    Example::

        ebc = EmbeddingBagCollection(config=ebc_config)
        model = DLRMTrain(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20],
           over_arch_layer_sizes=[5, 1],
        )
    """

    def __init__(self, embedding_bag_collection: 'EmbeddingBagCollection', dense_in_features: 'int', dense_arch_layer_sizes: 'List[int]', over_arch_layer_sizes: 'List[int]', dense_device: 'Optional[torch.device]'=None) ->None:
        super().__init__()
        self.model = DLRM(embedding_bag_collection=embedding_bag_collection, dense_in_features=dense_in_features, dense_arch_layer_sizes=dense_arch_layer_sizes, over_arch_layer_sizes=over_arch_layer_sizes, dense_device=dense_device)
        self.loss_fn: 'nn.Module' = nn.BCEWithLogitsLoss()

    def forward(self, batch: 'Batch') ->Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        logits = self.model(batch.dense_features, batch.sparse_features)
        logits = logits.squeeze()
        loss = self.loss_fn(logits, batch.labels.float())
        return loss, (loss.detach(), logits.detach(), batch.labels.detach())


class KJTAllToAll:
    """
    Different from the module defined in torchrec.

    Basically, this class conducts all_gather with all_to_all collective.
    """

    def __init__(self, group):
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)

    @torch.no_grad()
    def all_to_all(self, kjt):
        if self.world_size == 1:
            return kjt
        values, lengths = kjt.values(), kjt.lengths()
        keys, batch_size = kjt.keys(), kjt.stride()
        length_list = [(lengths if i == self.rank else lengths.clone()) for i in range(self.world_size)]
        all_length_list = [torch.empty_like(lengths) for _ in range(self.world_size)]
        dist.all_to_all(all_length_list, length_list, group=self.group)
        intermediate_all_length_list = [_length.view(-1, batch_size) for _length in all_length_list]
        all_length_per_key_list = [torch.sum(_length, dim=1).cpu().tolist() for _length in intermediate_all_length_list]
        all_value_length = [torch.sum(each).item() for each in all_length_list]
        value_list = [(values if i == self.rank else values.clone()) for i in range(self.world_size)]
        all_value_list = [torch.empty(_length, dtype=values.dtype, device=values.device) for _length in all_value_length]
        dist.all_to_all(all_value_list, value_list, group=self.group)
        all_value_list = [torch.split(_values, _length_per_key) for _values, _length_per_key in zip(all_value_list, all_length_per_key_list)]
        all_values = torch.cat([torch.cat(values_per_key) for values_per_key in zip(*all_value_list)])
        all_lengths = torch.cat(intermediate_all_length_list, dim=1).view(-1)
        return KeyedJaggedTensor.from_lengths_sync(keys=keys, values=all_values, lengths=all_lengths)


def get_tablewise_rank_arrange(dataset=None, world_size=0):
    if 'criteo' in dataset and 'kaggle' in dataset:
        if world_size == 1:
            rank_arrange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif world_size == 2:
            rank_arrange = [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]
        elif world_size == 3:
            rank_arrange = [2, 1, 0, 1, 1, 2, 2, 1, 0, 0, 1, 1, 0, 1, 0, 2, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0]
        elif world_size == 4:
            rank_arrange = [3, 1, 0, 3, 1, 0, 2, 1, 0, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 0, 2, 0, 0, 2, 3, 2]
        elif world_size == 8:
            rank_arrange = [6, 6, 0, 4, 7, 2, 5, 7, 0, 5, 7, 1, 7, 3, 5, 3, 1, 6, 6, 0, 2, 2, 1, 4, 3, 4]
        else:
            raise NotImplementedError('Other Tablewise settings are under development')
    elif 'criteo' in dataset:
        if world_size == 1:
            rank_arrange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif world_size == 2:
            rank_arrange = [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
        elif world_size == 4:
            rank_arrange = [1, 3, 3, 3, 3, 0, 2, 2, 1, 2, 2, 2, 0, 1, 2, 1, 0, 1, 0, 0, 2, 3, 3, 3, 1, 0]
        else:
            raise NotImplementedError('Other Tablewise settings are under development')
    else:
        raise NotImplementedError('Other Tablewise settings are under development')
    return rank_arrange


def prepare_tablewise_config(num_embeddings_per_feature, cache_ratio, id_freq_map_total=None, dataset='criteo_kaggle', world_size=2):
    embedding_bag_config_list: 'List[TablewiseEmbeddingBagConfig]' = []
    rank_arrange = get_tablewise_rank_arrange(dataset, world_size)
    table_offsets = np.array([0, *np.cumsum(num_embeddings_per_feature)])
    for i, num_embeddings in enumerate(num_embeddings_per_feature):
        ids_freq_mapping = None
        if id_freq_map_total != None:
            ids_freq_mapping = id_freq_map_total[table_offsets[i]:table_offsets[i + 1]]
        cuda_row_num = int(cache_ratio * num_embeddings) + 2000
        if cuda_row_num > num_embeddings:
            cuda_row_num = num_embeddings
        embedding_bag_config_list.append(TablewiseEmbeddingBagConfig(num_embeddings=num_embeddings, cuda_row_num=cuda_row_num, assigned_rank=rank_arrange[i], ids_freq_mapping=ids_freq_mapping))
    return embedding_bag_config_list


def sparse_embedding_shape_hook(embeddings, feature_size, batch_size):
    return embeddings.view(feature_size, batch_size, -1).transpose(0, 1)


def sparse_embedding_shape_hook_for_tablewise(embeddings, feature_size, batch_size):
    return embeddings.view(embeddings.shape[0], feature_size, -1)


class FusedSparseModules(nn.Module):

    def __init__(self, num_embeddings_per_feature, embedding_dim, fused_op='all_to_all', reduction_mode='sum', sparse=False, output_device_type=None, use_cache=False, cache_ratio=0.01, id_freq_map=None, warmup_ratio=0.7, buffer_size=50000, is_dist_dataloader=True, use_lfu_eviction=False, use_tablewise_parallel=False, dataset: 'str'=None):
        super(FusedSparseModules, self).__init__()
        self.sparse_feature_num = len(num_embeddings_per_feature)
        if use_cache:
            if use_tablewise_parallel:
                world_size = torch.distributed.get_world_size()
                embedding_bag_config_list = prepare_tablewise_config(num_embeddings_per_feature, 0.01, id_freq_map, dataset, world_size)
                self.embed = ParallelCachedEmbeddingBagTablewise(embedding_bag_config_list, embedding_dim, sparse=sparse, mode=reduction_mode, include_last_offset=True, warmup_ratio=warmup_ratio, buffer_size=buffer_size, evict_strategy=EvictionStrategy.LFU if use_lfu_eviction else EvictionStrategy.DATASET)
                self.shape_hook = sparse_embedding_shape_hook_for_tablewise
            else:
                self.embed = ParallelCachedEmbeddingBag(sum(num_embeddings_per_feature), embedding_dim, sparse=sparse, mode=reduction_mode, include_last_offset=True, cache_ratio=cache_ratio, ids_freq_mapping=id_freq_map, warmup_ratio=warmup_ratio, buffer_size=buffer_size, evict_strategy=EvictionStrategy.LFU if use_lfu_eviction else EvictionStrategy.DATASET)
                self.shape_hook = sparse_embedding_shape_hook
        else:
            raise NotImplementedError('Other EmbeddingBags are under development')
        if is_dist_dataloader:
            self.kjt_collector = KJTAllToAll(gpc.get_group(ParallelMode.GLOBAL))
        else:
            self.kjt_collector = None

    def forward(self, sparse_features: 'Union[List, KeyedJaggedTensor]', cache_op: 'bool'=True):
        self.embed.set_cache_op(cache_op)
        if self.kjt_collector:
            with record_function('(zhg)KJT AllToAll collective'):
                sparse_features = self.kjt_collector.all_to_all(sparse_features)
        if isinstance(sparse_features, list):
            batch_size = sparse_features[2]
            flattened_sparse_embeddings = self.embed(sparse_features[0], sparse_features[1], shape_hook=lambda x: self.shape_hook(x, self.sparse_feature_num, batch_size))
        elif isinstance(sparse_features, KeyedJaggedTensor):
            batch_size = sparse_features.stride()
            flattened_sparse_embeddings = self.embed(sparse_features.values(), sparse_features.offsets(), shape_hook=lambda x: self.shape_hook(x, self.sparse_feature_num, batch_size))
        else:
            raise TypeError
        return flattened_sparse_embeddings


class FusedDenseModules(nn.Module):
    """
    Fusing dense operations of DLRM into a single module
    """

    def __init__(self, embedding_dim, num_sparse_features, dense_in_features, dense_arch_layer_sizes, over_arch_layer_sizes):
        super(FusedDenseModules, self).__init__()
        if dense_in_features <= 0:
            self.dense_arch = nn.Identity()
            over_in_features = choose(num_sparse_features, 2)
            num_dense = 0
        else:
            self.dense_arch = DenseArch(in_features=dense_in_features, layer_sizes=dense_arch_layer_sizes)
            over_in_features = embedding_dim + choose(num_sparse_features, 2) + num_sparse_features
            num_dense = 1
        self.inter_arch = InteractionArch(num_sparse_features=num_sparse_features, num_dense_features=num_dense)
        self.over_arch = OverArch(in_features=over_in_features, layer_sizes=over_arch_layer_sizes)

    def forward(self, dense_features, embedded_sparse_features):
        embedded_dense_features = self.dense_arch(dense_features)
        concat_dense = self.inter_arch(dense_features=embedded_dense_features, sparse_features=embedded_sparse_features)
        logits = self.over_arch(concat_dense)
        return logits


class Timer:
    """A timer object which helps to log the execution times, and provides different tools to assess the times.
    """

    def __init__(self):
        self._started = False
        self._start_time = time.time()
        self._elapsed = 0
        self._history = []

    @property
    def has_history(self):
        return len(self._history) != 0

    @property
    def current_time(self) ->float:
        torch.cuda.synchronize()
        return time.time()

    def start(self):
        """Firstly synchronize cuda, reset the clock and then start the timer.
        """
        self._elapsed = 0
        torch.cuda.synchronize()
        self._start_time = time.time()
        self._started = True

    def lap(self):
        """lap time and return elapsed time
        """
        return self.current_time - self._start_time

    def stop(self, keep_in_history: 'bool'=False):
        """Stop the timer and record the start-stop time interval.

        Args:
            keep_in_history (bool, optional): Whether does it record into history
                each start-stop interval, defaults to False.
        Returns:
            int: Start-stop interval.
        """
        torch.cuda.synchronize()
        end_time = time.time()
        elapsed = end_time - self._start_time
        if keep_in_history:
            self._history.append(elapsed)
        self._elapsed = elapsed
        self._started = False
        return elapsed

    def get_history_mean(self):
        """Mean of all history start-stop time intervals.

        Returns:
            int: Mean of time intervals
        """
        return sum(self._history) / len(self._history)

    def get_history_sum(self):
        """Add up all the start-stop time intervals.

        Returns:
            int: Sum of time intervals.
        """
        return sum(self._history)

    def get_elapsed_time(self):
        """Return the last start-stop time interval.

        Returns:
            int: The last time interval.

        Note:
            Use it only when timer is not in progress
        """
        assert not self._started, 'Timer is still in progress'
        return self._elapsed

    def reset(self):
        """Clear up the timer and its history
        """
        self._history = []
        self._started = False
        self._elapsed = 0


class HybridParallelDLRM(nn.Module):
    """
    Model parallelized Embedding followed by Data parallelized dense modules
    """

    def __init__(self, num_embeddings_per_feature, embedding_dim, num_sparse_features, dense_in_features, dense_arch_layer_sizes, over_arch_layer_sizes, dense_device, sparse_device, sparse=False, fused_op='all_to_all', use_cache=False, cache_ratio=0.01, id_freq_map=None, warmup_ratio=0.7, buffer_size=50000, is_dist_dataloader=True, use_lfu_eviction=False, use_tablewise=False, dataset: 'str'=None):
        super(HybridParallelDLRM, self).__init__()
        if use_cache and sparse_device.type != dense_device.type:
            raise ValueError(f'Sparse device must be the same as dense device, however we got {sparse_device.type} for sparse, {dense_device.type} for dense')
        self.dense_device = dense_device
        self.sparse_device = sparse_device
        self.sparse_modules = FusedSparseModules(num_embeddings_per_feature, embedding_dim, fused_op=fused_op, sparse=sparse, output_device_type=dense_device.type, use_cache=use_cache, cache_ratio=cache_ratio, id_freq_map=id_freq_map, warmup_ratio=warmup_ratio, buffer_size=buffer_size, is_dist_dataloader=is_dist_dataloader, use_lfu_eviction=use_lfu_eviction, use_tablewise_parallel=use_tablewise, dataset=dataset)
        self.dense_modules = DDP(module=FusedDenseModules(embedding_dim, num_sparse_features, dense_in_features, dense_arch_layer_sizes, over_arch_layer_sizes), device_ids=[0 if os.environ.get('NVT_TAG', None) else gpc.get_global_rank()], process_group=gpc.get_group(ParallelMode.GLOBAL), gradient_as_bucket_view=True, broadcast_buffers=False, static_graph=True)
        param_amount = sum(num_embeddings_per_feature) * embedding_dim
        param_storage = self.sparse_modules.embed.element_size() * param_amount
        param_amount += sum(p.numel() for p in self.dense_modules.parameters())
        param_storage += sum(p.numel() * p.element_size() for p in self.dense_modules.parameters())
        buffer_amount = sum(b.numel() for b in self.sparse_modules.buffers()) + sum(b.numel() for b in self.dense_modules.buffers())
        buffer_storage = sum(b.numel() * b.element_size() for b in self.sparse_modules.buffers()) + sum(b.numel() * b.element_size() for b in self.dense_modules.buffers())
        stat_str = f'Number of model parameters: {param_amount:,}, storage overhead: {param_storage / 1024 ** 3:.2f} GB. Number of model buffers: {buffer_amount:,}, storage overhead: {buffer_storage / 1024 ** 3:.2f} GB.'
        self.stat_str = stat_str

    def forward(self, dense_features, sparse_features, inspect_time=False, cache_op=True):
        ctx1 = get_time_elapsed(dist_logger, 'embedding lookup in forward pass') if inspect_time else nullcontext()
        with ctx1:
            with record_function('Embedding lookup:'):
                embedded_sparse = self.sparse_modules(sparse_features, cache_op)
        ctx2 = get_time_elapsed(dist_logger, 'dense operations in forward pass') if inspect_time else nullcontext()
        with ctx2:
            with record_function('Dense operations:'):
                logits = self.dense_modules(dense_features, embedded_sparse)
        return logits

    def model_stats(self, prefix=''):
        return f'{prefix}: {self.stat_str}'

