import torch
from torch.nn import Linear
from .base import SelfAttentionModule, CrossAttentionModule
from .attention_mechanisms import DotProductAttention

from nn.VI import MFVILinear

from typing import Union, Optional
from torch import Tensor
from torch.nn import Module
from .attention_mechanisms import AttentionModule


class ClassicalSelfAttentionModule(SelfAttentionModule):
    def __init__(
            self,
            input_features: int,
            attention_dimension: int,
            attention_module: Union[AttentionModule, str] = DotProductAttention(),
            nhead: int = 1,
            activation: Optional[Union[Module, str]] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(attention_module=attention_module)

        self.input_features = input_features
        self.attention_dimension = attention_dimension
        self.nhead = nhead
        self.output_dimension = nhead * attention_dimension

        # this mapping can be made more efficient by equivalently using one layer that maps to 3x the attention_dimension
        self.qkv_mapping = Linear(
            in_features=input_features,
            out_features=3*self.output.dimension,
            bias=True, **factory_kwargs
        )

        # finds torch default activation functions from str name or just uses provided
        self.activation = getattr(torch.nn, activation)() if isinstance(activation, str) else activation

    def forward(self, input_: Tensor) -> Tensor:
        # unflatten into heads, attn_dim, and qkv (for easy splitting) and switch heads to the front
        qkv = self.qkv_mapping(input_).unflatten(dim=-1, sizes=(self.nhead, self.attention_dimension, 3)).transpose(-3, -4)
        if self.activation is not None:
            qkv = self.activation(qkv)

        # attention and flatten away heads
        output = self.attention_module(qkv[..., 0], qkv[..., 1], qkv[..., 2]).transpose(-2, -3).flatten(dim=-2)

        return output


class ClassicalCrossAttentionModule(CrossAttentionModule):
    def __init__(
            self,
            input_features: int,
            other_features: int,
            attention_dimension: int,
            attention_module: AttentionModule = DotProductAttention,
            nhead: int = 1,
            activation: Optional[Union[Module, str]] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(attention_module=attention_module)

        self.q_mapping = Linear(
            in_features=input_features,
            out_features=self.attention_output_dimension,
            bias=True, **factory_kwargs
        )
        # this mapping can be made more efficient by equivalently using one layer that maps to 2x the attention_dimension
        self.kv_mapping = Linear(
            in_features=other_features,
            out_features=2*self.output.dimension,
            bias=True, **factory_kwargs
        )

    def forward(self, input_: Tensor, other: Tensor) -> Tensor:
        # unflatten into heads, attn_dim, and kv (for easy splitting) and switch heads to the front
        q = self.q_mapping(input_).unflatten(dim=-1, sizes=(self.nhead, self.attention_dimension)).transpose(-2, -3)
        kv = self.kv_mapping(other).unflatten(dim=-1, sizes=(self.nhead, self.attention_dimension, 2)).transpose(-3, -4)
        if self.activation is not None:
            q = self.activation(q)
            kv = self.activation(kv)

        # attention and flatten away heads
        output = self.attention_module(q, kv[..., 0], kv[..., 1]).transpose(-2, -3).flatten(dim=-2)

        return output


class MFVISelfAttentionModule(ClassicalSelfAttentionModule):
    def __init__(
            self,
            input_features: int,
            attention_dimension: int,
            attention_module: AttentionModule = DotProductAttention,
            nhead: int = 1,
            activation: Optional[Union[Module, str]] = None,
            prior_weight_std: float = 1.0,
            prior_bias_std: float = 1.0,
            init_std: float = 0.05,
            sqrt_width_scaling: bool = False,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        mfvi_kwargs = {'prior_weight_std': prior_weight_std, 'prior_bias_std': prior_bias_std,
                       'init_std': init_std, 'sqrt_width_scaling': sqrt_width_scaling}
        super().__init__(
            input_features=input_features,
            attention_dimension=attention_dimension,
            attention_module=attention_module,
            nhead=nhead,
            activation=activation,
            **factory_kwargs
        )

        self.qkv_mapping = MFVILinear(
            dim_in=self.input_features,
            dim_out=3*self.output_dimension,
            **mfvi_kwargs, **factory_kwargs
        )


class MFVICrossAttentionModule(ClassicalCrossAttentionModule):
    def __init__(
            self,
            input_features: int,
            other_features: int,
            attention_dimension: int,
            attention_module: AttentionModule = DotProductAttention,
            nhead: int = 1,
            activation: Optional[Union[Module, str]] = None,
            prior_weight_std: float = 1.0,
            prior_bias_std: float = 1.0,
            init_std: float = 0.05,
            sqrt_width_scaling: bool = False,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        mfvi_kwargs = {'prior_weight_std': prior_weight_std, 'prior_bias_std': prior_bias_std,
                       'init_std': init_std, 'sqrt_width_scaling': sqrt_width_scaling}
        super().__init__(
            input_features=input_features,
            other_features=other_features,
            attention_dimension=attention_dimension,
            attention_module=attention_module,
            nhead=nhead,
            activation=activation,
            **factory_kwargs
        )

        self.q_mapping = MFVILinear(
            dim_in=self.input_features,
            dim_out=self.attention_output_dimension,
            **mfvi_kwargs, **factory_kwargs
        )
        self.kv_mapping = MFVILinear(
            dim_in=self.other_features,
            dim_out=2*self.attention_output_dimension,
            **mfvi_kwargs, **factory_kwargs
        )


class ClassicalEncoderLayer(Module):
    def __init__(
            self,
            input_features: int,
            nhead: int = 1,
            residual_connection: bool = True,
            layer_norm: bool = True,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.attention_module = ClassicalSelfAttentionModule(
            input_features=input_features,
            attention_dimension=input_features,
            attention_module=DotProductAttention,
            nhead=nhead,

        )

