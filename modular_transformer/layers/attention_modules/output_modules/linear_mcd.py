from typing import Callable, Optional, Union

import torch
from torch import Tensor

from .base import OutputModule
from .mc_dropout import (
    BernoulliNodeMCDLayer,
    BernoulliWeightMCDLayer,
    GaussianNodeMCDLayer,
    GaussianWeightMCDLayer,
)

__all__ = [
    "LinearMCDOutputModule",
    "DoubleLinearMCDOutputModule",
]


class LinearMCDOutputModule(OutputModule):
    """
    A single layer output module with optional activation and 4 types of optional Monte Carlo Dropout

    Commonly used to reduce the nhead * dmodel output features of classical multihead attention back to dmodel

    Parameters
        :param attention_output_features int: number of input nodes and size of the feature dimension of the intended input
        :param output_features int: number of output features (default: attention_output_features)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
        :param activation Optional[Union[Module, str]]: output activation function (default: None)
        :param bias bool: If set to False, the layer will not learn an additive bias (default: True)
        :param gaussian bool: If set to True, the MC Dropout will be gaussian, otherwise it will be a bernoulli dropout(default: False)
        :param weight_drop bool: If set to True, the MC Dropout will be performed on the weights rather than the nodes (default: False)
        :param rate float: in case of a bernoulli dropout this rate will be used as a dropout rate (default: 0.5)
        :param std_dev float: in case of a gaussian dropout this value will be used as the standard deviation of the weights (default: 0.5)
    """

    def __init__(
        self,
        attention_output_features: int,
        output_features: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        bias: bool = True,
        gaussian: bool = False,
        weight_drop: bool = False,
        rate: float = 0.5,
        std_dev: float = 0.5,
        istrainablesigma: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            attention_output_features=attention_output_features, output_features=output_features
        )
        if gaussian and weight_drop:
            self.linear = GaussianWeightMCDLayer(
                attention_output_features,
                output_features,
                bias,
                std_dev,
                istrainablesigma,
                **factory_kwargs,
            )
        elif gaussian and not weight_drop:
            self.linear = GaussianNodeMCDLayer(
                attention_output_features, output_features, bias, std_dev, **factory_kwargs
            )
        elif weight_drop and not gaussian:
            self.linear = BernoulliWeightMCDLayer(
                attention_output_features, output_features, bias, rate, **factory_kwargs
            )
        else:
            self.linear = BernoulliNodeMCDLayer(
                attention_output_features, output_features, bias, rate, **factory_kwargs
            )
        # self.linear = Linear(self.attention_output_features, self.output_features, bias=bias, **factory_kwargs)
        self.activation = (
            getattr(torch.nn, activation)() if isinstance(activation, str) else activation
        )

    def forward(self, input_: Tensor) -> Tensor:
        if self.activation is not None:
            output = self.activation(self.linear(input_))
        else:
            output = self.linear(input_)
        return output


class DoubleLinearMCDOutputModule(OutputModule):
    """
    A two layer output module with optional activation after the first layer

    Commonly used as "feedforward layer" in the classical Transformer architecture

    Parameters
        :param attention_output_features int: number of input nodes and size of the feature dimension of the intended input
        :param output_features int: number of output features (default: attention_output_features)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
        :param activation Optional[Union[Module, str]]: output activation function (default: None)
        :param bias bool: If set to False, the layer will not learn an additive bias (default: True)
        :param gaussian bool: If set to True, the MC Dropout will be gaussian, otherwise it will be a bernoulli dropout(default: False)
        :param weight_drop bool: If set to True, the MC Dropout will be performed on the weights rather than the nodes (default: False)
        :param rate float: in case of a bernoulli dropout this rate will be used as a dropout rate (default: 0.5)
        :param std_dev float: in case of a gaussian dropout this value will be used as the standard deviation of the weights (default: 0.5)
        :param dim_feedforward int: dimension of the hidden layer (default: 1024)
        :param activation Optional[Union[Module, str]]: intermediate activation function (default
    """

    def __init__(
        self,
        attention_output_features: int,
        output_features: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        bias: bool = True,
        gaussian: bool = False,
        weight_drop: bool = False,
        rate: float = 0.5,
        std_dev: float = 0.5,
        istrainablesigma: bool = False,
        dim_feedforward: int = 1024,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            attention_output_features=attention_output_features, output_features=output_features
        )
        if gaussian and weight_drop:
            self.linear1 = GaussianWeightMCDLayer(
                attention_output_features,
                dim_feedforward,
                bias,
                std_dev,
                istrainablesigma,
                **factory_kwargs,
            )
            self.linear2 = GaussianWeightMCDLayer(
                dim_feedforward, output_features, bias, std_dev, istrainablesigma, **factory_kwargs
            )
        elif gaussian and not weight_drop:
            self.linear1 = GaussianNodeMCDLayer(
                attention_output_features, dim_feedforward, bias, std_dev, **factory_kwargs
            )
            self.linear2 = GaussianNodeMCDLayer(
                dim_feedforward, output_features, bias, std_dev, **factory_kwargs
            )
        elif weight_drop and not gaussian:
            self.linear1 = BernoulliWeightMCDLayer(
                attention_output_features, dim_feedforward, bias, rate, **factory_kwargs
            )
            self.linear2 = BernoulliWeightMCDLayer(
                dim_feedforward, output_features, bias, rate, **factory_kwargs
            )
        else:
            self.linear1 = BernoulliNodeMCDLayer(
                attention_output_features, dim_feedforward, bias, rate, **factory_kwargs
            )
            self.linear2 = BernoulliNodeMCDLayer(
                dim_feedforward, output_features, bias, rate, **factory_kwargs
            )

        # self.linear1 = Linear(self.attention_output_features, dim_feedforward, bias=bias, **factory_kwargs)
        # self.linear2 = Linear(dim_feedforward, self.output_features, bias=bias, **factory_kwargs)
        self.activation = (
            getattr(torch.nn, activation)() if isinstance(activation, str) else activation
        )

    def forward(self, input_: Tensor) -> Tensor:
        if self.activation is not None:
            output = self.linear2(self.activation(self.linear1(input_)))
        else:
            output = self.linear2(self.linear1(input_))
        return output
