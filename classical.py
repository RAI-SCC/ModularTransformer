import torch
from torch.nn import Softmax, ReLU

from .base import Transformer
from .layers import ClassicalTransformerEncoderLayer, ClassicalTransformerDecoderLayer
from .layers.attention_modules.output_modules import LinearOutputModule
from .layers.attention_modules.attention_mechanisms.masking import AttentionMatrixMask

from typing import Optional, Union, Callable
from torch import Tensor


class ClassicalTransformer(Transformer):
    def __init__(
            self,
            input_features: int,
            attention_dimension: int,
            nhead: int,
            dim_feedforward: int,
            num_encoder_layers: int = 1,
            num_decoder_layers: int = 1,
            hidden_features: Optional[int] = None,
            output_features: Optional[int] = None,
            inter_activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = ReLU(),
            final_activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = Softmax(),
            encoder_mask: Optional[Union[AttentionMatrixMask, str]] = None,
            decoder_mask: Optional[Union[AttentionMatrixMask, str]] = None,
            bias: bool = True,
            layer_norm: bool = True,
            dropout: float = 0.,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        hidden_features = hidden_features or input_features
        output_features = output_features or input_features

        encoder_layer = ClassicalTransformerEncoderLayer(
            input_features=input_features,
            d_model=attention_dimension,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            output_features=hidden_features,
            mask=encoder_mask,
            bias=bias,
            layer_norm=layer_norm,
            dropout=dropout,
            activation=inter_activation,
            **factory_kwargs)

        decoder_layer = ClassicalTransformerDecoderLayer(
            input_features=input_features,
            other_features=hidden_features,
            attention_dimension=attention_dimension,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            output_features=input_features,
            mask=decoder_mask,
            bias=bias,
            layer_norm=layer_norm,
            dropout=dropout,
            activation=inter_activation,
            **factory_kwargs)
        attention_output_features = decoder_layer.output_features

        output_layer = LinearOutputModule(
            attention_output_features=attention_output_features,
            output_features=output_features,
            activation=final_activation,
            bias=bias,
            **factory_kwargs)

        super().__init__(
            encoder_layer=encoder_layer,
            decoder_layer=decoder_layer,
            output_layer=output_layer,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )
