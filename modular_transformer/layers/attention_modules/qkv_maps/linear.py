from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Linear, Module

from .base import KVmap, QKVmap, Qmap

__all__ = [
    "LinearQmap",
    "LinearKVmap",
    "LinearQKVmap",
]


class LinearQmap(Qmap):
    """
    Default q_mapping using a single torch.nn.Linear layer

    Parameters
        :param input_features int: number of input nodes and size of the feature dimension of the intended input
        :param q_features int: number of output features
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
        :param activation Optional[Union[Module, str]]: output activation function (default: None)
        :param bias bool: If set to False, the layer will not learn an additive bias (default: True)

    :return Tensor:
    """

    def __init__(
        self,
        input_features: int,
        q_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        activation: Optional[Union[Module, str]] = None,
        bias: bool = True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_features=input_features, q_features=q_features, **factory_kwargs)

        self.linear = Linear(self.input_features, self.q_features, bias=bias, **factory_kwargs)
        self.activation = (
            getattr(torch.nn, activation)() if isinstance(activation, str) else activation
        )

    def forward(self, input_: Tensor) -> Tensor:
        output = self.linear(input_)
        if self.activation is not None:
            output = self.activation(output)
        return output


class LinearKVmap(KVmap):
    """
    Default kv_mapping using a single torch.nn.Linear layer

    Slight efficiency increase via obtaining k and v from one layer with their combined output features.

    Parameters
        :param input_features int: number of input nodes and size of the feature dimension of the intended input
        :param k_features int: number of output features of the k-component (i.e. first) output
        :param v_features Optional[int]: number of output features of the v-component (i.e. second) output (default: k_features)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
        :param activation Optional[Union[Module, str]]: output activation function (default: None)
        :param bias bool: If set to False, the layer will not learn an additive bias (default: True)

    :return Tensor, Tensor:
    """

    def __init__(
        self,
        input_features: int,
        k_features: int,
        v_features: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        activation: Optional[Union[Module, str]] = None,
        bias: bool = True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            input_features=input_features,
            k_features=k_features,
            v_features=v_features,
            **factory_kwargs,
        )

        total_output_features = self.k_features + self.v_features
        self.linear = Linear(
            self.input_features, total_output_features, bias=bias, **factory_kwargs
        )
        self.activation = (
            getattr(torch.nn, activation)() if isinstance(activation, str) else activation
        )

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        output = self.linear(input_)
        if self.activation is not None:
            output = self.activation(output)
        k, v = torch.split(output, [self.k_features, self.v_features], dim=-1)
        return k.clone(), v.clone()


class LinearQKVmap(QKVmap):
    """
    Default qkv_mapping using a single torch.nn.Linear layer

    Slight efficiency increase via obtaining q, k, and v from one layer with their combined output features.

    Parameters
        :param input_features int: number of input nodes and size of the feature dimension of the intended input
        :param q_features int: number of output features of the q-component (i.e. first) output
        :param k_features Optional[int]: number of output features of the k-component (i.e. second) output (default: q_features)
        :param v_features Optional[int]: number of output features of the v-component (i.e. third) output (default: k_features)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
        :param activation Optional[Union[Module, str]]: output activation function (default: None)
        :param bias bool: If set to False, the layer will not learn an additive bias (default: True)

    :return Tensor, Tensor, Tensor:
    """

    def __init__(
        self,
        input_features: int,
        q_features: int,
        k_features: Optional[int] = None,
        v_features: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        activation: Optional[Union[Module, str]] = None,
        bias: bool = True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            input_features=input_features,
            q_features=q_features,
            k_features=k_features,
            v_features=v_features,
            **factory_kwargs,
        )

        total_output_features = self.q_features + self.k_features + self.v_features
        self.linear = Linear(
            self.input_features, total_output_features, bias=bias, **factory_kwargs
        )
        self.activation = (
            getattr(torch.nn, activation)() if isinstance(activation, str) else activation
        )

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        output = self.linear(input_)
        if self.activation is not None:
            output = self.activation(output)
        q, k, v = torch.split(output, [self.q_features, self.k_features, self.v_features], dim=-1)
        return q.clone(), k.clone(), v.clone()
