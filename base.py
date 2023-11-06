import copy
import torch
from torch.nn import Module, ModuleList

from .layers import TransformerEncoderLayer, TransformerDecoderLayer
from .layers.attention_modules.output_modules import OutputModule

from torch import Tensor

__all__ = [
    'Transformer',
    'ParallelTransformer',
]


class Transformer(Module):
    def __init__(
            self,
            encoder_layer: TransformerEncoderLayer,
            decoder_layer: TransformerDecoderLayer,
            output_layer: OutputModule,
            num_encoder_layers: int = 1,
            num_decoder_layers: int = 1
    ) -> None:
        super().__init__()

        self.encoder_layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(num_encoder_layers)])
        self.decoder_layers = ModuleList([copy.deepcopy(decoder_layer) for i in range(num_decoder_layers)])
        self.output_module = output_layer

        self._check_validity()

    def _check_validity(self) -> None:
        assert self.encoder_layers[-1].output_features == self.decoder_layers[0].other_features
        assert self.decoder_layers[-1].output_features == self.output_module.attention_output_features

    @property
    def encoder_features(self) -> int:
        return self.encoder_layers[0].input_features

    @property
    def decoder_features(self) -> int:
        return self.decoder_layers[0].input_features

    @property
    def output_features(self) -> int:
        return self.output_module.output_features

    def forward(self, encoder_input: Tensor, decoder_input: Tensor) -> Tensor:
        x = encoder_input
        for layer in self.encoder_layers:
            x = layer(x)

        y = decoder_input
        for layer in self.decoder_layers:
            y = layer(y, x)

        output = self.output_module(y)

        return output


class ParallelTransformer(Transformer):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        decoder_layer: TransformerDecoderLayer,
        output_layer: OutputModule,
        num_layers: int = 1,
    ) -> None:
        super().__init__(
            encoder_layer=encoder_layer,
            decoder_layer=decoder_layer,
            output_layer=output_layer,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )

    def forward(self, encoder_input: Tensor, decoder_input: Tensor) -> Tensor:
        x = encoder_input
        y = decoder_input
        for encoder_layer, decoder_layer in zip(self.encoder_layers, self.decoder_layers):
            x = encoder_layer(x)
            y = decoder_layer(y, x)

        output = self.output_module(y)

        return output