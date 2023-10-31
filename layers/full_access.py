import torch

import attention_modules
from .attention_modules import output_modules

from .attention_modules import SelfAttentionModule, CrossAttentionModule
from .attention_modules import FullAccessSelfAttentionModule, FullAccessCrossAttentionModule
from .attention_modules.output_modules import OutputModule, DoubleLinearOutputModule, LinearOutputModule
from .base import TransformerEncoderLayer, TransformerDecoderLayer

from typing import Type, Optional, Union
from torch import Tensor


class FullAccessTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(
            self,
            input_features: Optional[int] = None,
            attention_dimension: Optional[int] = None,
            nhead: Optional[int] = None,
            direct_attention_output_features: Optional[int] = None,
            output_features: Optional[int] = None,
            self_attention_layer: Optional[Union[str, SelfAttentionModule, Type[SelfAttentionModule]]] = None,
            output_layer: Optional[Union[str, OutputModule, Type[OutputModule]]] = None,
            self_attention_args: Optional[dict] = None,
            output_layer_args: Optional[dict] = None,
            residual_connection: bool = True,
            layer_norm: bool = True,
            dropout: float = 0.,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}

        if self_attention_layer is None:
            self_attention_layer = FullAccessSelfAttentionModule(
                input_features=input_features,
                attention_dimension=attention_dimension,
                nhead=nhead,
                output_features=direct_attention_output_features,
                **(self_attention_args or {}),
                **factory_kwargs)
        if not isinstance(self_attention_layer, SelfAttentionModule):
            self_attention_layer = getattr(attention_modules, self_attention_layer) if isinstance(self_attention_layer, str) else self_attention_layer
            self_attention_layer = self_attention_layer(**(self_attention_args or {}), **factory_kwargs)

        if output_layer is None:
            attention_output_features = self_attention_layer.output_features
            output_layer = DoubleLinearOutputModule(
                attention_output_features=attention_output_features,
                output_features=output_features,
                **(output_layer_args or {}),
                **factory_kwargs)
        if not isinstance(output_layer, OutputModule):
            output_layer = getattr(output_modules, output_layer) if isinstance(output_layer, str) else output_layer
            output_layer = output_layer(**(output_layer_args or {}), **factory_kwargs)

        super().__init__(
            self_attention_layer=self_attention_layer,
            output_layer=output_layer,
            residual_connection=residual_connection,
            layer_norm=layer_norm,
            dropout=dropout,
            **factory_kwargs)
