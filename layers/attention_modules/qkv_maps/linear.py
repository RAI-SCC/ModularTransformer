import torch
from torch.nn import Module, Linear
from .base import Qmap, KVmap, QKVmap

from typing import Optional, Union, Tuple
from torch import Tensor

__all__ = [
    'LinearQmap',
    'LinearKVmap',
    'LinearQKVmap',
]


class LinearQmap(Qmap):
    def __init__(
            self,
            input_features: int,
            attention_dimension: int,
            nhead: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            activation: Optional[Union[Module, str]] = None,
            bias: bool = True
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            input_features=input_features,
            attention_dimension=attention_dimension,
            nhead=nhead
        )

        self.linear = Linear(self.input_features, torch.tensor(self.output_shape).prod(), bias=bias, **factory_kwargs)
        self.activation = getattr(torch.nn, activation)() if isinstance(activation, str) else activation

    def forward(self, input_: Tensor) -> Tensor:
        output = self.linear(input_).reshape(self.output_shape)
        if self.activation is not None:
            output = self.activation(output)
        return output[..., 0]


class LinearKVmap(KVmap):
    __init__ = LinearQmap.__init__

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.linear(input_).reshape(self.output_shape)
        if self.activation is not None:
            output = self.activation(output)
        return output[..., 0], output[..., 1]


class LinearQKVmap(QKVmap):
    __init__ = LinearQmap.__init__

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        output = self.linear(input_).reshape(self.output_shape)
        if self.activation is not None:
            output = self.activation(output)
        return output[..., 0], output[..., 1], output[..., 2]
