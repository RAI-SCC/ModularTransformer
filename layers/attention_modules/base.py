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
        assert self.qkv_mappings[0].q_features == self.attention_mechanism.q_features
        assert self.head_reduction.nhead == self._nhead
        assert self.attention_mechanism.output_features == self.head_reduction.attention_dimension
        assert self.head_reduction.attention_output_features == self.output_module.attention_output_features

    def forward(self, input_: Tensor) -> Tensor:
        head_results = []
        for qkv_mapping in self.qkv_mappings:
            q, k, v = qkv_mapping(input_)
            head_results.append(self.attention_mechanism(q, k, v))

        attention_result = torch.stack(head_results)
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
        return self.q_mapping.input_features

    @property
    def other_features(self) -> int:
        return self.kv_mapping.input_features

    @property
    def output_features(self) -> int:
        return self.output_module.output_features

    def _check_validity(self):
        assert self.q_mappings[0].q_features == self.attention_mechanism.q_features
        assert self.kv_mappings[0].k_features == self.attention_mechanism.k_features
        assert self.kv_mappings[0].v_features == self.attention_mechanism.v_features
        assert self.head_reduction.nhead == self._nhead
        assert self.attention_mechanism.output_features == self.head_reduction.attention_dimension
        assert self.head_reduction.attention_output_features == self.output_module.attention_output_features

    def forward(self, input_: Tensor, other: Tensor) -> Tensor:
        head_results = []
        for q_mapping, kv_mapping in zip(self.q_mappings, self.kv_mappings):
            q = q_mapping(input_)
            k, v = kv_mapping(other)
            head_results.append(self.attention_mechanism(q, k, v))

        attention_result = torch.stack(head_results)
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
