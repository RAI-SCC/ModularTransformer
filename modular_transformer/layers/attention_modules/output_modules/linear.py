"""Linear or MLP output modules."""
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Linear, ReLU

from .base import OutputModule

__all__ = [
    "LinearOutputModule",
    "DoubleLinearOutputModule",
]


class LinearOutputModule(OutputModule):
    """
    A simple single layer output module with optional activation.

    Commonly used to reduce the nhead * dmodel output features of classical multihead attention back to dmodel

    Parameters
    ----------
        :param attention_output_features int: number of input nodes and size of the feature dimension of the intended input
        :param output_features int: number of output features (default: attention_output_features)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
        :param activation Optional[Union[Module, str]]: output activation function (default: None)
        :param bias bool: If set to False, the layer will not learn an additive bias (default: True)
    """

    def __init__(
        self,
        attention_output_features: int,
        output_features: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        bias: bool = True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            attention_output_features=attention_output_features, output_features=output_features
        )

        self.linear = Linear(
            self.attention_output_features, self.output_features, bias=bias, **factory_kwargs
        )
        self.activation = (
            getattr(torch.nn, activation)() if isinstance(activation, str) else activation
        )

    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass through the model."""
        if self.activation is not None:
            output = self.activation(self.linear(input_))
        else:
            output = self.linear(input_)
        return output


class DoubleLinearOutputModule(OutputModule):
    """
    A two layer output module with optional activation after the first layer.

    Commonly used as "feedforward layer" in the classical Transformer architecture

    Parameters
    ----------
        :param attention_output_features int: number of input nodes and size of the feature dimension of the intended input
        :param output_features int: number of output features (default: attention_output_features)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
        :param dim_feedforward int: dimension of the hidden layer (default: 1024)
        :param activation Optional[Union[Module, str]]: intermediate activation function (default: ReLU())
        :param bias bool: If set to False, the layer will not learn an additive bias (default: True)
    """

    def __init__(
        self,
        attention_output_features: int,
        output_features: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        dim_feedforward: int = 1024,
        activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = ReLU(),
        bias: bool = True,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            attention_output_features=attention_output_features, output_features=output_features
        )

        self.linear1 = Linear(
            self.attention_output_features, dim_feedforward, bias=bias, **factory_kwargs
        )
        self.linear2 = Linear(dim_feedforward, self.output_features, bias=bias, **factory_kwargs)
        self.activation = (
            getattr(torch.nn, activation)() if isinstance(activation, str) else activation
        )

    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass through the model."""
        if self.activation is not None:
            output = self.linear2(self.activation(self.linear1(input_)))
        else:
            output = self.linear2(self.linear1(input_))
        return output
