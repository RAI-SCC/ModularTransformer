import torch
from torch.nn import Softmax, ReLU

from .base import Transformer
from .layers import ClassicalTransformerEncoderLayer, ClassicalTransformerDecoderLayer
from .layers.attention_modules.output_modules import LinearOutputModule
from .layers.attention_modules.attention_mechanisms.masking import AttentionMatrixMask

from typing import Optional, Union, Callable
from torch import Tensor


class ClassicalTransformer(Transformer):
    """
    The classical Transformer architecture as propose by Vaswani et 17

    Parameters:
        :param input_features int: size of the input feature dimension
        :param d_model int: internal number of features for the attention mechanism
        :param nhead int: number of attention heads
        :param dim_feedforward: dimension of the `DoubleLinearOutputModule`s (feedforward layers) in each layer and on
            the final output
        :param num_encoder_layers int: number of encoder layers
        :param num_decoder_layers int: number of decoder layers
        :param hidden_features int: size of the encoder output feature dimension (default: input_features)
        :param output_features int: size of the decoder output feature dimension (default: input_features)
        :param inter_activation Optional[str or Callable[[Tensor], Tensor]]: activation of the `DoubleLinearOutputModule`s
            in each layer (default: ReLU())
        :param final_activation Optional[str or Callable[[Tensor], Tensor]]: activation of the `DoubleLinearOutputModule`
            in the final output layer (default: ReLU())
        :param encoder_mask Optional[AttentionMatrixMask or str]: mask for encoder attention (default: None)
        :param decoder_mask Optional[AttentionMatrixMask or str]: mask for decoder attention (default: None)
        :param bias bool: If set to False, all Linear layers will not learn an additive bias (default: True)
        :param layer_norm bool: if False not layer norm will be applied after attention and output module (default: True)
        :param dropout float: dropout rate applied on the output of attention and output module (default: 0.)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    """
    def __init__(
            self,
            input_features: int,
            d_model: int,
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
            d_model=d_model,
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
            attention_dimension=d_model,
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
