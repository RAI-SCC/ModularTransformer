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
            q_features: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            activation: Optional[Union[Module, str]] = None,
            bias: bool = True
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            input_features=input_features,
            q_features=q_features,
            **factory_kwargs)

        self.linear = Linear(self.input_features, self.q_features, bias=bias, **factory_kwargs)
        self.activation = getattr(torch.nn, activation)() if isinstance(activation, str) else activation

    def forward(self, input_: Tensor) -> Tensor:
        output = self.linear(input_)
        if self.activation is not None:
            output = self.activation(output)
        return output


class LinearKVmap(KVmap):
    def __init__(
            self,
            input_features: int,
            k_features: int,
            v_features: Optional[int] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            activation: Optional[Union[Module, str]] = None,
            bias: bool = True
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            input_features=input_features,
            k_features=k_features,
            v_features=v_features,
            **factory_kwargs)

        total_output_features = self.k_features + self.v_features
        self.linear = Linear(self.input_features, total_output_features, bias=bias, **factory_kwargs)
        self.activation = getattr(torch.nn, activation)() if isinstance(activation, str) else activation

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.linear(input_)
        if self.activation is not None:
            output = self.activation(output)
        k, v = torch.split(output, [self.k_features, self.v_features], dim=-1)
        return k.clone(), v.clone()


class LinearQKVmap(QKVmap):
    def __init__(
            self,
            input_features: int,
            q_features: int,
            k_features: Optional[int] = None,
            v_features: Optional[int] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            activation: Optional[Union[Module, str]] = None,
            bias: bool = True
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            input_features=input_features,
            q_features=q_features,
            k_features=k_features,
            v_features=v_features,
            **factory_kwargs)

        total_output_features = self.q_features + self.k_features + self.v_features
        self.linear = Linear(self.input_features, total_output_features, bias=bias, **factory_kwargs)
        self.activation = getattr(torch.nn, activation)() if isinstance(activation, str) else activation

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        output = self.linear(input_)
        if self.activation is not None:
            output = self.activation(output)
        q, k, v = torch.split(output, [self.q_features, self.k_features, self.v_features], dim=-1)
        return q.clone(), k.clone(), v.clone()
