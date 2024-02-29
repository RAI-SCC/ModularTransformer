from typing import Callable

import torch
from torch import Tensor
from torch.nn import ReLU, Softmax

from modular_transformer import Transformer
from modular_transformer.layers import ClassicalTransformerDecoderLayer
from modular_transformer.layers.attention_modules.attention_mechanisms.masking import AttentionMatrixMask
from modular_transformer.layers.attention_modules.output_modules import LinearOutputModule
from modular_transformer.layers.taylor import TaylorTransformerEncoderLayer, TaylorTransformerDecoderLayer


class TaylorTransformer(Transformer):
    def __init__(
        self,
        input_features: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        hidden_features: int | None = None,
        output_features: int | None = None,
        sequence_length: int | None = None,
        sequence_length_decoder: int | None = None,
        inter_activation: str | Callable[[Tensor], Tensor] | None = ReLU(),
        final_activation: str | Callable[[Tensor], Tensor] = Softmax(),
        encoder_mask: AttentionMatrixMask | str | None = None,
        decoder_mask: AttentionMatrixMask | str | None = None,
        bias: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        use_classical_decoder = False

        hidden_features = hidden_features or input_features
        output_features = output_features or input_features

        encoder_layer = TaylorTransformerEncoderLayer(
            input_features=input_features,
            d_model=d_model,
            sequence_length=sequence_length,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            output_features=hidden_features,
            mask=encoder_mask,
            bias=bias,
            layer_norm=layer_norm,
            dropout=dropout,
            activation=inter_activation,
            **factory_kwargs,
        )

        if use_classical_decoder:
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
                **factory_kwargs,
            )
        else:
            decoder_layer = TaylorTransformerDecoderLayer(
                input_features=input_features,
                other_features=hidden_features,
                sequence_length=sequence_length_decoder,
                sequence_length_other=sequence_length,
                attention_dimension=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                output_features=input_features,
                mask=decoder_mask,
                bias=bias,
                layer_norm=layer_norm,
                dropout=dropout,
                activation=inter_activation,
                **factory_kwargs,
            )
        attention_output_features = decoder_layer.output_features

        output_layer = LinearOutputModule(
            attention_output_features=attention_output_features,
            output_features=output_features,
            activation=final_activation,
            bias=bias,
            **factory_kwargs,
        )

        super().__init__(
            encoder_layer=encoder_layer,
            decoder_layer=decoder_layer,
            output_layer=output_layer,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

    def forward(self, encoder_input: Tensor, decoder_input: Tensor) -> Tensor:
        x = encoder_input
        for layer in self.encoder_layers:
            x = layer(x)

        y = decoder_input
        for layer in self.decoder_layers:
            y = layer(y, x)

        # output = x
        output = self.output_module(y)

        make_consistent = False
        if make_consistent:
            last_timepoint = encoder_input[:, -1, 0]
            # add last timepoint to output
            for batch in range(0, output.shape[0]):
                output[batch] = output[batch] + last_timepoint[batch]

        return output
