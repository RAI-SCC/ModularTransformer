from typing import Optional

import torch
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Module

from .attention_modules import CrossAttentionModule, SelfAttentionModule
from .attention_modules.output_modules import OutputModule

__all__ = [
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
]


class TransformerEncoderLayer(Module):
    """
    Provides the basic structure for a Transformer encoder layer

    Combines a `SelfAttentionModule` and an `OutputModule` into an encoder layer.
    Also includes and gives control over residual connections, layer norms and dropout.
    Also ensures consistency of the components (e.g. that input and output features of attention block and output module
    are equal if there are residual connections.

    Parameters
        :param self_attention_layer SelfAttentionModule: the self-attention block
        :param output_layer OutputModule: the output or feedforward layer
        :param residual_connection bool: If False there are no residual connections around attention block and output
            module (default: True)
        :param layer_norm bool: if False, no layer norm is applied after each sublayer (default: True)
        :param dropout float: dropout rate on the output of each sublayer (default: 0.)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    """

    def __init__(
        self,
        self_attention_layer: SelfAttentionModule,
        output_layer: OutputModule,
        residual_connection: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
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
        """Checks consistency of the model"""
        assert (
            self.self_attention_layer.output_features == self.output_layer.attention_output_features
        )

        if self.residual_connection:
            assert self.input_features == self.self_attention_layer.output_features
            assert self.output_layer.attention_output_features == self.output_layer.output_features

    @property
    def input_features(self) -> int:
        return self.self_attention_layer.input_features

    @property
    def output_features(self) -> int:
        return self.output_layer.output_features

    def forward(self, input_: Tensor) -> Tensor:
        """Applies the encoder layer"""
        # Self attention
        x = self.self_attention_layer(input_)
        if self.residual_connection:
            x += input_
        if self.layer_norm:
            x = self.layer_norm1(x)
        x = self.dropout(x)

        # Output layer
        output = self.output_layer(x)
        if self.residual_connection:
            output += x
        if self.layer_norm:
            output = self.layer_norm2(output)
        output = self.dropout(output)

        return output


class TransformerDecoderLayer(Module):
    """
    Provides the basic structure for a Transformer decoder layer

    Combines a `SelfAttentionModule`, a `CrossAttentionModule`, and an `OutputModule` into a decoder layer.
    Also includes and gives control over residual connections, layer norms and dropout.
    Also ensures consistency of the components (e.g. that input and output features of attention block and output module
    are equal if there are residual connections.

    Parameters
        :param self_attention_layer SelfAttentionModule: the self-attention block
        :param cross_attention_layer CrossAttentionModule: the cross-attention block
        :param output_layer OutputModule: the output or feedforward layer
        :param residual_connection bool: If False there are no residual connections around attention block and output
            module (default: True)
        :param layer_norm bool: if False, no layer norm is applied after each sublayer (default: True)
        :param dropout float: dropout rate on the output of each sublayer (default: 0.)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    """

    def __init__(
        self,
        self_attention_layer: SelfAttentionModule,
        cross_attention_layer: CrossAttentionModule,
        output_layer: OutputModule,
        residual_connection: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
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
        """Checks consistency of the model"""
        assert (
            self.self_attention_layer.output_features == self.cross_attention_layer.input_features
        )
        assert (
            self.cross_attention_layer.output_features
            == self.output_layer.attention_output_features
        )

        if self.residual_connection:
            assert (
                self.self_attention_layer.input_features
                == self.self_attention_layer.output_features
            )
            assert (
                self.cross_attention_layer.input_features
                == self.cross_attention_layer.output_features
            )
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
        """Performs a decoder layer with the decoder input `input_` and hidden state `other`"""
        # Self-attention
        x = self.self_attention_layer(input_)
        if self.residual_connection:
            x += input_
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
        output = self.output_layer(x)
        if self.residual_connection:
            output += x
        if self.layer_norm:
            output = self.layer_norm3(output)
        output = self.dropout(output)

        return output
