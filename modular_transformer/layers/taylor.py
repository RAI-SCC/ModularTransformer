from typing import Callable

import torch
from torch import Tensor
from torch.nn import ReLU

from modular_transformer.layers import TransformerDecoderLayer, TransformerEncoderLayer
from modular_transformer.layers.attention_modules.attention_mechanisms.masking import (
    AttentionMatrixMask,
)
from modular_transformer.layers.attention_modules.output_modules import DoubleLinearOutputModule
from modular_transformer.layers.attention_modules.taylor import (
    TaylorCrossAttentionModule,
    TaylorSelfAttentionModule,
)


class TaylorTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(
        self,
        input_features: int,
        d_model: int,
        sequence_length: int,
        nhead: int,
        dim_feedforward: int,
        output_features: int,
        mask: AttentionMatrixMask | str | None = None,
        bias: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] | None = ReLU(),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        self_attention_layer = TaylorSelfAttentionModule(
            input_features=input_features,
            d_model=d_model,
            sequence_length=sequence_length,
            nhead=nhead,
            output_features=input_features,
            mask=mask,
            bias=bias,
            **factory_kwargs,
        )

        output_layer = DoubleLinearOutputModule(
            attention_output_features=input_features,
            dim_feedforward=dim_feedforward,
            output_features=output_features,
            activation=activation,
            bias=bias,
            **factory_kwargs,
        )

        super().__init__(
            self_attention_layer=self_attention_layer,
            output_layer=output_layer,
            residual_connection=False,
            layer_norm=layer_norm,
            dropout=dropout,
            **factory_kwargs,
        )


class TaylorTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        input_features: int,
        other_features: int,
        sequence_length: int,
        sequence_length_other: int,
        attention_dimension: int,
        nhead: int,
        dim_feedforward: int,
        output_features: int,
        mask: AttentionMatrixMask | str | None = None,
        bias: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] | None = ReLU(),
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        self_attention_layer = TaylorSelfAttentionModule(
            input_features=input_features,
            d_model=attention_dimension,
            sequence_length=sequence_length,
            nhead=nhead,
            output_features=input_features,
            mask=mask,
            bias=bias,
            **factory_kwargs,
        )

        cross_attention_layer = TaylorCrossAttentionModule(
            input_features=input_features,
            other_features=other_features,
            sequence_length=sequence_length,
            sequence_length_other=sequence_length_other,
            d_model=attention_dimension,
            nhead=nhead,
            output_features=input_features,
            mask=None,
            bias=bias,
            **factory_kwargs,
        )

        output_layer = DoubleLinearOutputModule(
            attention_output_features=input_features,
            dim_feedforward=dim_feedforward,
            output_features=output_features,
            activation=activation,
            bias=bias,
            **factory_kwargs,
        )

        super().__init__(
            self_attention_layer=self_attention_layer,
            cross_attention_layer=cross_attention_layer,
            output_layer=output_layer,
            residual_connection=False,
            layer_norm=layer_norm,
            dropout=dropout,
            **factory_kwargs,
        )
