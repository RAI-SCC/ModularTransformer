from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import ReLU

from .attention_modules import ClassicalMCDCrossAttentionModule, ClassicalMCDSelfAttentionModule
from .attention_modules.attention_mechanisms.masking import AttentionMatrixMask
from .attention_modules.output_modules import DoubleLinearMCDOutputModule
from .base import TransformerDecoderLayer, TransformerEncoderLayer

__all__ = [
    "ClassicalMCDTransformerEncoderLayer",
    "ClassicalMCDTransformerDecoderLayer",
]


class ClassicalMCDTransformerEncoderLayer(TransformerEncoderLayer):
    """
    A Transformer encoder layer as defined in Vasiwani et al 17

    Parameters
        :param input_features int: size of the input feature dimension
        :param d_model int: internal number of features for the attention mechanism
        :param nhead int: number of attention heads
        :param dim_feedforward int: size of the hidden layer of the `DoubleLinearOutputModule` (feedforward layer)
        :param output_features int: size of the output feature dimension
        :param mask Optional[AttentionMatrixMask or str]: mask for masked attention (default: None)
        :param bias bool: If set to False, all Linear layers will not learn an additive bias (default: True)
        :param layer_norm bool: if False not layer norm will be applied after attention and output module (default: True)
        :param dropout float: dropout rate applied on the output of attention and output module (default: 0.)
        :param activation Optional[str or Callable[[Tensor], Tensor]]: activation of the `DoubleLinearOutputModule` (default: ReLU())
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    """

    def __init__(
        self,
        input_features: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        output_features: int,
        mask: Optional[Union[AttentionMatrixMask, str]] = None,
        bias: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.0,
        gaussian: bool = False,
        weight_drop: bool = False,
        rate: float = 0.5,
        std_dev: float = 0.5,
        istrainablesigma: bool = False,
        activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = ReLU(),
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        self_attention_layer = ClassicalMCDSelfAttentionModule(
            input_features=input_features,
            d_model=d_model,
            nhead=nhead,
            output_features=input_features,
            mask=mask,
            bias=bias,
            gaussian=gaussian,
            weight_drop=weight_drop,
            rate=rate,
            std_dev=std_dev,
            istrainablesigma=istrainablesigma,
            **factory_kwargs,
        )

        output_layer = DoubleLinearMCDOutputModule(
            attention_output_features=input_features,
            dim_feedforward=dim_feedforward,
            output_features=output_features,
            activation=activation,
            bias=bias,
            gaussian=gaussian,
            weight_drop=weight_drop,
            rate=rate,
            std_dev=std_dev,
            istrainablesigma=istrainablesigma,
            **factory_kwargs,
        )

        super().__init__(
            self_attention_layer=self_attention_layer,
            output_layer=output_layer,
            residual_connection=True,
            layer_norm=layer_norm,
            dropout=dropout,
            **factory_kwargs,
        )


class ClassicalMCDTransformerDecoderLayer(TransformerDecoderLayer):
    """
    A Transformer decoder layer as defined in Vasiwani et al 17

    Parameters
        :param input_features int: size of the (first) input feature dimension
        :param other_features int: size of the other (second) input feature dimension
        :param d_model int: internal number of features for the attention mechanism
        :param nhead int: number of attention heads
        :param dim_feedforward int: size of the hidden layer of the `DoubleLinearOutputModule` (feedforward layer)
        :param output_features int: size of the output feature dimension
        :param mask Optional[AttentionMatrixMask or str]: mask for masked attention (default: None)
        :param bias bool: If set to False, all Linear layers will not learn an additive bias (default: True)
        :param layer_norm bool: if False not layer norm will be applied after attention and output module (default: True)
        :param dropout float: dropout rate applied on the output of attention and output module (default: 0.)
        :param activation Optional[str or Callable[[Tensor], Tensor]]: activation of the `DoubleLinearOutputModule` (default: ReLU())
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    """

    def __init__(
        self,
        input_features: int,
        other_features: int,
        attention_dimension: int,
        nhead: int,
        dim_feedforward: int,
        output_features: int,
        mask: Optional[Union[AttentionMatrixMask, str]] = None,
        bias: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.0,
        gaussian: bool = False,
        weight_drop: bool = False,
        rate: float = 0.5,
        std_dev: float = 0.5,
        istrainablesigma: bool = False,
        activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = ReLU(),
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        self_attention_layer = ClassicalMCDSelfAttentionModule(
            input_features=input_features,
            d_model=attention_dimension,
            nhead=nhead,
            output_features=input_features,
            mask=mask,
            bias=bias,
            gaussian=gaussian,
            weight_drop=weight_drop,
            rate=rate,
            std_dev=std_dev,
            istrainablesigma=istrainablesigma,
            **factory_kwargs,
        )

        cross_attention_layer = ClassicalMCDCrossAttentionModule(
            input_features=input_features,
            other_features=other_features,
            d_model=attention_dimension,
            nhead=nhead,
            output_features=input_features,
            mask=None,
            bias=bias,
            gaussian=gaussian,
            weight_drop=weight_drop,
            rate=rate,
            std_dev=std_dev,
            istrainablesigma=istrainablesigma,
            **factory_kwargs,
        )

        output_layer = DoubleLinearMCDOutputModule(
            attention_output_features=input_features,
            dim_feedforward=dim_feedforward,
            output_features=output_features,
            activation=activation,
            bias=bias,
            gaussian=gaussian,
            weight_drop=weight_drop,
            rate=rate,
            std_dev=std_dev,
            istrainablesigma=istrainablesigma,
            **factory_kwargs,
        )

        super().__init__(
            self_attention_layer=self_attention_layer,
            cross_attention_layer=cross_attention_layer,
            output_layer=output_layer,
            residual_connection=True,
            layer_norm=layer_norm,
            dropout=dropout,
            **factory_kwargs,
        )
