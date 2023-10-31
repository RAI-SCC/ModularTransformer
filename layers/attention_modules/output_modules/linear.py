import torch
from torch.nn import Module, Linear, ReLU
from .base import OutputModule

from typing import Optional, Union, Callable
from torch import Tensor

__all__ = [
    'LinearOutputModule',
    'DoubleLinearOutputModule',
]


class LinearOutputModule(OutputModule):
    def __init__(
            self,
            attention_output_features: int,
            output_features: Optional[int] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
            bias: bool = True
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            attention_output_features=attention_output_features,
            output_features=output_features
        )

        self.linear = Linear(self.attention_output_features, self.output_features, bias=bias, **factory_kwargs)
        self.activation = getattr(torch.nn, activation)() if isinstance(activation, str) else activation

    def forward(self, input_: Tensor) -> Tensor:
        if self.activation is not None:
            output = self.activation(self.linear(input_))
        else:
            output = self.linear(input_)
        return output


class DoubleLinearOutputModule(OutputModule):
    def __init__(
            self,
            attention_output_features: int,
            output_features: Optional[int] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            dim_feedforward: int = 1024,
            activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = ReLU(),
            bias: bool = True
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            attention_output_features=attention_output_features,
            output_features=output_features
        )

        self.linear1 = Linear(self.attention_output_features, dim_feedforward, bias=bias, **factory_kwargs)
        self.linear2 = Linear(dim_feedforward, self.output_features, bias=bias, **factory_kwargs)
        self.activation = getattr(torch.nn, activation)() if isinstance(activation, str) else activation

    def forward(self, input_: Tensor) -> Tensor:
        if self.activation is not None:
            output = self.linear2(self.activation(self.linear1(input_)))
        else:
            output = self.linear2(self.linear1(input_))
        return output
