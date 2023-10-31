import torch

from .base import SelfAttentionModule, CrossAttentionModule
from .qkv_maps import LinearQmap, LinearKVmap, LinearQKVmap
from .attention_mechanisms import DotProductAttention
from .head_reductions import ConcatHeads
from .output_modules import LinearOutputModule

from typing import Optional
from torch import Tensor

__all__ = [
    'ClassicalSelfAttentionModule',
    'ClassicalCrossAttentionModule',
]


class ClassicalSelfAttentionModule(SelfAttentionModule):
    def __init__(
            self,
            input_features: int,
            attention_dimension: int,
            nhead: int,
            output_features: int,
            mask: Optional[Tensor] = None,
            bias: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        qkv_mapping = LinearQKVmap(
            input_features=input_features,
            attention_dimension=attention_dimension,
            nhead=nhead,
            activation=None,
            bias=bias,
            **factory_kwargs)

        attention_mechanism = DotProductAttention(
            attention_dimension=attention_dimension,
            mask=mask,
            **factory_kwargs)

        head_reduction = ConcatHeads(
            attention_dimension=attention_dimension,
            nhead=nhead,
            *factory_kwargs)
        attention_output_features = head_reduction.attention_output_features()

        output_module = LinearOutputModule(
            attention_output_features=attention_output_features,
            output_features=output_features,
            activation=None,
            bias=bias,
            **factory_kwargs)

        super().__init__(
            qkv_mapping=qkv_mapping,
            attention_mechanism=attention_mechanism,
            head_reduction=head_reduction,
            output_module=output_module
        )


class ClassicalCrossAttentionModule(CrossAttentionModule):
    def __init__(
            self,
            input_features: int,
            other_features: int,
            attention_dimension: int,
            nhead: int,
            output_features: int,
            mask: Optional[Tensor] = None,
            bias: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        q_mapping = LinearQmap(
            input_features=input_features,
            attention_dimension=attention_dimension,
            nhead=nhead,
            activation=None,
            bias=bias,
            **factory_kwargs)

        kv_mapping = LinearKVmap(
            input_features=other_features,
            attention_dimension=attention_dimension,
            nhead=nhead,
            activation=None,
            bias=bias,
            **factory_kwargs)

        attention_mechanism = DotProductAttention(
            attention_dimension=attention_dimension,
            mask=mask,
            **factory_kwargs)

        head_reduction = ConcatHeads(
            attention_dimension=attention_dimension,
            nhead=nhead,
            *factory_kwargs)
        attention_output_features = head_reduction.attention_output_features()

        output_module = LinearOutputModule(
            attention_output_features=attention_output_features,
            output_features=output_features,
            activation=None,
            bias=bias,
            **factory_kwargs)

        super().__init__(
            q_mapping=q_mapping,
            kv_mapping=kv_mapping,
            attention_mechanism=attention_mechanism,
            head_reduction=head_reduction,
            output_module=output_module
        )
