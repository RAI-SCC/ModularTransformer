import copy
import torch
from torch.nn import Module

from .qkv_maps import Qmap, KVmap, QKVmap
from .attention_mechanisms import AttentionModule
from .head_reductions import HeadReduction
from .output_modules import OutputModule

from torch import Tensor


__all__ = [
    'SelfAttentionModule',
    'CrossAttentionModule',
]


class SelfAttentionModule(Module):
    """
    Provides the basic structure for a self attention module

    Combines a `QKVmap`, `AttentionModule`, `HeadReduction`, and `OutputModule` into a self attention module.
    Also handles head creation and consistency checks between the components.

    Parameters:
        :param qkv_mapping QKVmap: mapping from input to query, key, and value
        :param attention_mechanism AttentionModule: performs attention with query, key, and value
        :param head_reduction HeadReduction: recombines the results of the heads
        :param output_module OutputModule: maps the recombined output to the output dimension
        :param nhead int: number of identical heads to create
    """
    def __init__(
            self,
            qkv_mapping: QKVmap,
            attention_mechanism: AttentionModule,
            head_reduction: HeadReduction,
            output_module: OutputModule,
            nhead: int = 1,
    ):
        super().__init__()
        self._nhead = nhead
        self.qkv_mappings = [copy.deepcopy(qkv_mapping) for _ in range(nhead)]
        self.attention_mechanisms = [copy.deepcopy(attention_mechanism) for _ in range(nhead)]
        self.head_reduction = head_reduction
        self.output_module = output_module

        self._check_validity()

    @property
    def input_features(self) -> int:
        return self.qkv_mappings[0].input_features

    @property
    def output_features(self) -> int:
        return self.output_module.output_features

    def _check_validity(self):
        """Checks consistency of the components"""
        assert self.qkv_mappings[0].q_features == self.attention_mechanisms[0].q_features
        assert self.head_reduction.nhead == self._nhead
        assert self.attention_mechanisms[0].output_features == self.head_reduction.attention_dimension
        assert self.head_reduction.attention_output_features == self.output_module.attention_output_features

    def forward(self, input_: Tensor) -> Tensor:
        """
        Accepts a Tensor of shape (*, S, I), where S is a sequence length and I the input_features, and returns a Tensor
        of shape (*, S, O), where O are the output_features
        """
        head_results = []
        for qkv_mapping, attention_mechanism in zip(self.qkv_mappings, self.attention_mechanisms):
            q, k, v = qkv_mapping(input_)
            head_results.append(attention_mechanism(q, k, v))

        attention_result = torch.stack(head_results)
        output = self.output_module(self.head_reduction(attention_result))
        return output


class CrossAttentionModule(Module):
    """
    Provides the basic structure for a self attention module

    Combines a `QKmap`, `KVmap`, `AttentionModule`, `HeadReduction`, and `OutputModule` into a cross attention module.
    Also handles head creation and consistency checks between the components.

    Parameters:
        :param q_mapping Qmap: mapping from input_ to query
        :param kv_mapping KVmap: mapping from other to key and value
        :param attention_mechanism AttentionModule: performs attention with query, key, and value
        :param head_reduction HeadReduction: recombines the results of the heads
        :param output_module OutputModule: maps the recombined output to the output dimension
        :param nhead int: number of identical heads to create
    """

    def __init__(
            self,
            q_mapping: Qmap,
            kv_mapping: KVmap,
            attention_mechanism: AttentionModule,
            head_reduction: HeadReduction,
            output_module: OutputModule,
            nhead: int = 1,
    ):
        super().__init__()
        self._nhead = nhead
        self.q_mappings = [copy.deepcopy(q_mapping) for _ in range(nhead)]
        self.kv_mappings = [copy.deepcopy(kv_mapping) for _ in range(nhead)]
        self.attention_mechanism = attention_mechanism
        self.head_reduction = head_reduction
        self.output_module = output_module

        self._check_validity()

    @property
    def input_features(self) -> int:
        return self.q_mappings[0].input_features

    @property
    def other_features(self) -> int:
        return self.kv_mappings[0].input_features

    @property
    def output_features(self) -> int:
        return self.output_module.output_features

    def _check_validity(self):
        """Checks consistency of the components"""
        assert self.q_mappings[0].q_features == self.attention_mechanism.q_features
        assert self.kv_mappings[0].k_features == self.attention_mechanism.k_features
        assert self.kv_mappings[0].v_features == self.attention_mechanism.v_features
        assert self.head_reduction.nhead == self._nhead
        assert self.attention_mechanism.output_features == self.head_reduction.attention_dimension
        assert self.head_reduction.attention_output_features == self.output_module.attention_output_features

    def forward(self, input_: Tensor, other: Tensor) -> Tensor:
        """
        Accepts Tensors input_ of shape (*, S, I) and other of shape (*, S, I*), where S is a sequence length,I the
        features ot input_, and I* the features of other, and returns a Tensor of shape (*, S, O), where O are the
        output_features
        """
        head_results = []
        for q_mapping, kv_mapping in zip(self.q_mappings, self.kv_mappings):
            q = q_mapping(input_)
            k, v = kv_mapping(other)
            head_results.append(self.attention_mechanism(q, k, v))

        attention_result = torch.stack(head_results)
        output = self.output_module(self.head_reduction(attention_result))
        return output
