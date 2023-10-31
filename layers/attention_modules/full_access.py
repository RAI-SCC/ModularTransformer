import torch

import qkv_maps
import attention_mechanisms
import head_reductions
import output_modules

from .base import SelfAttentionModule, CrossAttentionModule
from .qkv_maps import Qmap, KVmap, QKVmap, LinearQmap, LinearKVmap, LinearQKVmap
from .attention_mechanisms import AttentionModule, DotProductAttention
from .head_reductions import HeadReduction, ConcatHeads
from .output_modules import OutputModule, LinearOutputModule

from typing import Type, Optional, Union
from torch import Tensor

__all__ = [
    'FullAccessSelfAttentionModule',
    'FullAccessCrossAttentionModule',
]


class FullAccessSelfAttentionModule(SelfAttentionModule):
    def __init__(
            self,
            input_features: Optional[int] = None,
            attention_dimension: Optional[int] = None,
            nhead: Optional[int] = None,
            output_features: Optional[int] = None,
            qkv_mapping: Union[str, QKVmap, Type[QKVmap]] = LinearQKVmap,
            attention_mechanism: Union[str, AttentionModule, Type[AttentionModule]] = DotProductAttention,
            head_reduction: Union[str, HeadReduction, Type[HeadReduction]] = ConcatHeads,
            output_module: Union[str, OutputModule, Type[OutputModule]] = LinearOutputModule,
            attention_mask: Optional[Tensor] = None,
            qkv_mapping_args: Optional[dict] = None,
            attention_mechanism_args: Optional[dict] = None,
            head_reduction_args: Optional[dict] = None,
            output_module_args: Optional[dict] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        if not isinstance(qkv_mapping, QKVmap):
            assert input_features is not None
            assert attention_dimension is not None
            assert nhead is not None
            qkv_mapping = getattr(qkv_maps, qkv_mapping) if isinstance(qkv_mapping, str) else qkv_mapping
            qkv_mapping = qkv_mapping(
                input_features=input_features,
                attention_dimension=attention_dimension,
                nhead=nhead,
                **(qkv_mapping_args or {}),
                **factory_kwargs)

        if not isinstance(attention_mechanism, AttentionModule):
            assert attention_dimension is not None
            attention_mechanism = getattr(attention_mechanisms, attention_mechanism) if isinstance(attention_mechanism, str) else attention_mechanism
            attention_mechanism = attention_mechanism(
                attention_dimension=attention_dimension,
                mask=attention_mask,
                **(attention_mechanism_args or {}),
                **factory_kwargs)

        if not isinstance(head_reduction, HeadReduction):
            assert attention_dimension is not None
            head_reduction = getattr(head_reductions, head_reduction) if isinstance(head_reduction, str) else head_reduction
            head_reduction = head_reduction(
                attention_dimension=attention_dimension,
                nhead=nhead,
                **(head_reduction_args or {}),
                **factory_kwargs)

        if not isinstance(output_module, OutputModule):
            output_module = getattr(output_modules, output_module) if isinstance(output_module, str) else output_module
            attention_output_features = head_reduction.attention_output_features
            output_module = output_module(
                attention_output_features=attention_output_features,
                output_features=output_features,
                **(output_module_args or {}),
                **factory_kwargs)

        super().__init__(
            qkv_mapping=qkv_mapping,
            attention_mechanism=attention_mechanism,
            head_reduction=head_reduction,
            output_module=output_module
        )


class FullAccessCrossAttentionModule(CrossAttentionModule):
    def __init__(
            self,
            input_features: Optional[int] = None,
            other_features: Optional[int] = None,
            attention_dimension: Optional[int] = None,
            nhead: Optional[int] = None,
            output_features: Optional[int] = None,
            q_mapping: Union[str, Qmap, Type[Qmap]] = LinearQmap,
            kv_mapping: Union[str, KVmap, Type[KVmap]] = LinearKVmap,
            attention_mechanism: Union[str, AttentionModule, Type[AttentionModule]] = DotProductAttention,
            head_reduction: Union[str, HeadReduction, Type[HeadReduction]] = ConcatHeads,
            output_module: Union[str, OutputModule, Type[OutputModule]] = LinearOutputModule,
            attention_mask: Optional[Tensor] = None,
            q_mapping_args: Optional[dict] = None,
            kv_mapping_args: Optional[dict] = None,
            attention_mechanism_args: Optional[dict] = None,
            head_reduction_args: Optional[dict] = None,
            output_module_args: Optional[dict] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        if not isinstance(q_mapping, Qmap):
            assert input_features is not None
            assert attention_dimension is not None
            assert nhead is not None
            q_mapping = getattr(qkv_maps, q_mapping) if isinstance(q_mapping, str) else q_mapping
            q_mapping = q_mapping(
                input_features=input_features,
                attention_dimension=attention_dimension,
                nhead=nhead,
                **(q_mapping_args or {}),
                **factory_kwargs)

        if not isinstance(kv_mapping, KVmap):
            assert other_features is not None
            assert attention_dimension is not None
            assert nhead is not None
            kv_mapping = getattr(qkv_maps, kv_mapping) if isinstance(kv_mapping, str) else kv_mapping
            kv_mapping = kv_mapping(
                input_features=other_features,
                attention_dimension=attention_dimension,
                nhead=nhead,
                **(kv_mapping_args or {}),
                **factory_kwargs)

        if not isinstance(attention_mechanism, AttentionModule):
            assert attention_dimension is not None
            attention_mechanism = getattr(attention_mechanisms, attention_mechanism) if isinstance(attention_mechanism, str) else attention_mechanism
            attention_mechanism = attention_mechanism(
                attention_dimension=attention_dimension,
                mask=attention_mask,
                **(attention_mechanism_args or {}),
                **factory_kwargs)

        if not isinstance(head_reduction, HeadReduction):
            assert attention_dimension is not None
            head_reduction = getattr(head_reductions, head_reduction) if isinstance(head_reduction, str) else head_reduction
            head_reduction = head_reduction(
                attention_dimension=attention_dimension,
                nhead=nhead,
                **(head_reduction_args or {}),
                **factory_kwargs)

        if not isinstance(output_module, OutputModule):
            output_module = getattr(output_modules, output_module) if isinstance(output_module, str) else output_module
            attention_output_features = head_reduction.attention_output_features
            output_module = output_module(
                attention_output_features=attention_output_features,
                output_features=output_features,
                **(output_module_args or {}),
                **factory_kwargs)

        super().__init__(
            q_mapping=q_mapping,
            kv_mapping=kv_mapping,
            attention_mechanism=attention_mechanism,
            head_reduction=head_reduction,
            output_module=output_module
        )
