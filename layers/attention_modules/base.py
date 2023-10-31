import torch
from torch.nn import Module

from .qkv_maps import Qmap, KVmap, QKVmap
from .attention_mechanisms import AttentionModule
from .head_reductions import HeadReduction
from .output_modules import OutputModule

from typing import Type, Optional, TypeVar
from torch import Tensor


__all__ = [
    'SelfAttentionModule',
    'CrossAttentionModule',
]

CLS = TypeVar('CLS')


class SelfAttentionModule(Module):
    def __init__(
            self,
            qkv_mapping: QKVmap,
            attention_mechanism: AttentionModule,
            head_reduction: HeadReduction,
            output_module: OutputModule,
    ):
        super().__init__()
        self.qkv_mapping = qkv_mapping
        self.attention_mechanism = attention_mechanism
        self.head_reduction = head_reduction
        self.output_module = output_module

        self._check_validity()

    @property
    def input_features(self) -> int:
        return self.qkv_mapping.input_features

    @property
    def output_features(self) -> int:
        return self.output_module.output_features

    def _check_validity(self):
        assert self.qkv_mapping.attention_dimension == self.attention_mechanism.attention_dimension
        assert self.qkv_mapping.nhead == self.head_reduction.nhead
        assert self.qkv_mapping.k_features == self.head_reduction.attention_dimension
        assert self.head_reduction.attention_output_features == self.output_module.attention_output_features

    def forward(self, input_: Tensor) -> Tensor:
        q, k, v = self.qkv_mapping(input_)
        attention_result = self.attention_mechanism(q, k, v)
        output = self.output_module(self.head_reduction(attention_result))
        return output


class CrossAttentionModule(Module):
    def __init__(
            self,
            q_mapping: Qmap,
            kv_mapping: KVmap,
            attention_mechanism: AttentionModule,
            head_reduction: HeadReduction,
            output_module: OutputModule,
    ):
        super().__init__()
        self.q_mapping = q_mapping
        self.kv_mapping = kv_mapping
        self.attention_mechanism = attention_mechanism
        self.head_reduction = head_reduction
        self.output_module = output_module

        self._check_validity()

    @property
    def input_features(self) -> int:
        return self.q_mapping.input_features

    @property
    def other_features(self) -> int:
        return self.kv_mapping.input_features

    @property
    def output_features(self) -> int:
        return self.output_module.output_features

    def _check_validity(self):
        assert self.q_mapping.attention_dimension == self.kv_mapping.attention_dimension
        assert self.q_mapping.attention_dimension == self.attention_mechanism.attention_dimension
        assert self.q_mapping.nhead == self.kv_mapping.nhead
        assert self.q_mapping.nhead == self.head_reduction.nhead
        assert self.kv_mapping.k_features == self.head_reduction.attention_dimension
        assert self.head_reduction.attention_output_features == self.output_module.attention_output_features

    def forward(self, input_: Tensor, other: Tensor) -> Tensor:
        q = self.q_mapping(input_)
        k, v = self.kv_mapping(other)
        attention_result = self.attention_mechanism(q, k, v)
        output = self.output_module(self.head_reduction(attention_result))
        return output


#class SelfAttentionModule(Module, ABC):
#    def __init__(
#            self,
#            input_features: int,
#            attention_dimension: int,
#            attention_module: Union[Type[AttentionModule], str],
#            attention_args: Optional[dict] = None,
#            nhead: int = 1,
#            activation: Optional[Union[Module, str]] = None,
#            device: Optional[torch.device] = None,
#            dtype: Optional[torch.dtype] = None
#    ) -> None:
#        factory_kwargs = {'device': device, 'dtype': dtype}
#        super().__init__()
#        if isinstance(attention_module, str):
#            attention_module = getattr(attention_mechanisms, attention_module)
#        attention_args = attention_args or {}
#
#        self.input_features = input_features
#        self.attention_dimension = attention_dimension
#        self.attention_module = attention_module(attention_dimension=attention_dimension, **attention_args, **factory_kwargs)
#        self.nhead = nhead
#        self.attention_output_dimension = nhead * attention_dimension
#
#        # finds torch default activation functions from str name (with default args) or just uses provided
#        self.activation = getattr(torch.nn, activation)() if isinstance(activation, str) else activation
#
#    @abstractmethod
#    def forward(self, input_: Tensor) -> Tensor:
#        raise NotImplementedError
#
#
#class CrossAttentionModule(Module, ABC):
#    def __init__(
#            self,
#            input_features: int,
#            other_features: int,
#            attention_dimension: int,
#            attention_module: Union[AttentionModule, str],
#            attention_args: Optional[dict] = None,
#            nhead: int = 1,
#            activation: Optional[Union[Module, str]] = None,
#            device: Optional[torch.device] = None,
#            dtype: Optional[torch.dtype] = None
#    ) -> None:
#        factory_kwargs = {'device': device, 'dtype': dtype}
#        super().__init__()
#        if isinstance(attention_module, str):
#            attention_module = getattr(attention_mechanisms, attention_module)
#        attention_args = attention_args or {}
#
#        self.input_features = input_features
#        self.other_features = other_features
#        self.attention_dimension = attention_dimension
#        self.attention_module = attention_module(attention_dimension=attention_dimension, **attention_args, **factory_kwargs)
#        self.nhead = nhead
#        self.attention_output_dimension = nhead * attention_dimension
#
#        # finds torch default activation functions from str name (with default args) or just uses provided
#        self.activation = getattr(torch.nn, activation)() if isinstance(activation, str) else activation
#
#    @abstractmethod
#    def forward(self, input_: Tensor, other: Tensor) -> Tensor:
#        raise NotImplementedError
