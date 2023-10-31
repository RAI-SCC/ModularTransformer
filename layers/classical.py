import torch
from torch.nn import ReLU

from .base import TransformerEncoderLayer, TransformerDecoderLayer
from .attention_modules import ClassicalSelfAttentionModule, ClassicalCrossAttentionModule
from .attention_modules.output_modules import DoubleLinearOutputModule

from typing import Optional, Union, Callable
from torch import Tensor

__all__ = [
    'ClassicalTransformerEncoderLayer',
    'ClassicalTransformerDecoderLayer',
]


class ClassicalTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(
            self,
            input_features: int,
            attention_dimension: int,
            nhead: int,
            dim_feedforward: int,
            output_features: int,
            mask: Optional[Tensor] = None,
            bias: bool = True,
            layer_norm: bool = True,
            dropout: float = 0.,
            activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = ReLU(),
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        self_attention_layer = ClassicalSelfAttentionModule(
            input_features=input_features,
            attention_dimension=attention_dimension,
            nhead=nhead,
            output_features=input_features,
            mask=mask,
            bias=bias,
            **factory_kwargs)

        output_layer = DoubleLinearOutputModule(
            attention_output_features=input_features,
            dim_feedforward=dim_feedforward,
            output_features=output_features,
            activation=activation,
            bias=bias,
            **factory_kwargs)

        super().__init__(
            self_attention_layer=self_attention_layer,
            output_layer=output_layer,
            residual_connection=True,
            layer_norm=layer_norm,
            dropout=dropout,
            **factory_kwargs)


class ClassicalTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
            self,
            input_features: int,
            other_features: int,
            attention_dimension: int,
            nhead: int,
            dim_feedforward: int,
            output_features: int,
            mask: Optional[Tensor] = None,
            bias: bool = True,
            layer_norm: bool = True,
            dropout: float = 0.,
            activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = ReLU(),
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}

        self_attention_layer = ClassicalSelfAttentionModule(
            input_features=input_features,
            attention_dimension=attention_dimension,
            nhead=nhead,
            output_features=input_features,
            mask=None,
            bias=bias,
            **factory_kwargs)

        cross_attention_layer = ClassicalCrossAttentionModule(
            input_features=input_features,
            other_features=other_features,
            attention_dimension=attention_dimension,
            nhead=nhead,
            output_features=input_features,
            mask=mask,
            bias=bias,
            **factory_kwargs)

        output_layer = DoubleLinearOutputModule(
            attention_output_features=input_features,
            dim_feedforward=dim_feedforward,
            output_features=output_features,
            activation=activation,
            bias=bias,
            **factory_kwargs)

        super().__init__(
            self_attention_layer=self_attention_layer,
            cross_attention_layer=cross_attention_layer,
            output_layer=output_layer,
            residual_connection=True,
            layer_norm=layer_norm,
            dropout=dropout,
            **factory_kwargs)
