import torch
from torch.nn import Module, LayerNorm, Dropout

from .attention_modules import SelfAttentionModule, CrossAttentionModule
from .attention_modules.output_modules import OutputModule

from typing import Optional
from torch import Tensor

__all__ = [
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
]


class TransformerEncoderLayer(Module):
    def __init__(
            self,
            self_attention_layer: SelfAttentionModule,
            output_layer: OutputModule,
            residual_connection: bool = True,
            layer_norm: bool = True,
            dropout: float = 0.,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attention_layer = self_attention_layer
        self.output_layer = output_layer

        self.residual_connection = residual_connection

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_norm1 = LayerNorm(self_attention_layer.output_features, **factory_kwargs)
            self.layer_norm2 = LayerNorm(output_layer.output_features, **factory_kwargs)

        self.dropout = Dropout(p=dropout)

        self._check_validity()

    def _check_validity(self) -> None:
        assert self.self_attention_layer.output_features == self.output_layer.attention_output_features

        if self.residual_connection:
            assert self.self_attention_layer.input_features == self.self_attention_layer.output_features
            assert self.output_layer.attention_output_features == self.output_layer.output_features

    @property
    def input_features(self) -> int:
        return self.self_attention_layer.input_features

    @property
    def output_features(self) -> int:
        return self.output_layer.output_features

    def forward(self, input_: Tensor) -> Tensor:
        # Self attention
        x = self.self_attention_layer(input_)
        if self.residual_connection:
            x = x + input_
        if self.layer_norm:
            x = self.layer_norm1(x)
        x = self.dropout(x)

        # Output layer
        if self.residual_connection:
            output = x + self.output_layer(x)
        else:
            output = self.output_layer(x)
        if self.layer_norm:
            output = self.layer_norm2(output)
        output = self.dropout(output)

        return output


class TransformerDecoderLayer(Module):
    def __init__(
            self,
            self_attention_layer: SelfAttentionModule,
            cross_attention_layer: CrossAttentionModule,
            output_layer: OutputModule,
            residual_connection: bool = True,
            layer_norm: bool = True,
            dropout: float = 0.,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.output_layer = output_layer

        self.residual_connection = residual_connection

        self.layer_norm = layer_norm
        if self.layer_norm:
            self.layer_norm1 = LayerNorm(self_attention_layer.output_features, **factory_kwargs)
            self.layer_norm2 = LayerNorm(cross_attention_layer.output_features, **factory_kwargs)
            self.layer_norm3 = LayerNorm(output_layer.output_features, **factory_kwargs)

        self.dropout = Dropout(p=dropout)

        self._check_validity()

    def _check_validity(self) -> None:
        assert self.self_attention_layer.output_features == self.cross_attention_layer.input_features
        assert self.cross_attention_layer.output_features == self.output_layer.attention_output_features

        if self.residual_connection:
            assert self.self_attention_layer.input_features == self.self_attention_layer.output_features
            assert self.cross_attention_layer.input_features == self.cross_attention_layer.output_features
            assert self.output_layer.attention_output_features == self.output_layer.output_features

    @property
    def input_features(self) -> int:
        return self.self_attention_layer.input_features

    @property
    def other_features(self) -> int:
        return self.cross_attention_layer.other_features

    @property
    def output_features(self) -> int:
        return self.output_layer.output_features

    def forward(self, input_: Tensor, other: Tensor) -> Tensor:
        # Self-attention
        x = self.self_attention_layer(input_)
        if self.residual_connection:
            x = x + input_
        if self.layer_norm:
            x = self.layer_norm1(x)
        x = self.dropout(x)

        # Cross-attention
        if self.residual_connection:
            x = x + self.cross_attention_layer(x, other)
        else:
            x = self.cross_attention_layer(x, other)
        if self.layer_norm:
            x = self.layer_norm2(x)
        x = self.dropout(x)

        # Output layer
        if self.residual_connection:
            output = x + self.output_layer(x)
        else:
            output = self.output_layer(x)
        if self.layer_norm:
            output = self.layer_norm3(output)
        output = self.dropout(output)

        return output
