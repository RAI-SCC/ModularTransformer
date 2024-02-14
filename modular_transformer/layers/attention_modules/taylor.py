import copy

import torch
from torch import Tensor
from torch.nn import Module

from modular_transformer.layers.attention_modules import SelfAttentionModule
from modular_transformer.layers.attention_modules.attention_mechanisms.masking import AttentionMatrixMask
from modular_transformer.layers.attention_modules.head_reductions import ConcatHeads
from modular_transformer.layers.attention_modules.output_modules import LinearOutputModule
from modular_transformer.layers.attention_modules.qkv_maps import LinearQKVmap


class QuadraticScalarProductAttentionModule(Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        sequence_length: int,
    ):
        super().__init__()
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.output_features = output_features

        self.linear = torch.nn.Linear(self.sequence_length, self.output_features)

    def forward(self, input_: Tensor) -> Tensor:
        _batch_size, _sequence_length, _dimension = input_.shape
        assert _dimension == self.input_features
        assert _sequence_length == self.sequence_length

        x_squared = torch.empty(_batch_size, _sequence_length, _sequence_length)
        for j in range(_batch_size):
            x_squared[j] = input_[j] @ input_[j].transpose(0, 1)

        output = self.linear(x_squared)

        assert output.shape == (_batch_size, _sequence_length, self.output_features)
        assert output.isnan().sum() == 0
        return output


class QuadraticAttentionModule(Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        sequence_length: int,
    ):
        super().__init__()
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.output_features = output_features

        in_dim = self.sequence_length * self.input_features

        self.linear = torch.nn.Linear(in_dim + (in_dim + in_dim ** 2) // 2, self.output_features * self.sequence_length)

    def forward(self, input_: Tensor) -> Tensor:
        _batch_size, _sequence_length, _dimension = input_.shape
        assert _dimension == self.input_features

        in_dim = _sequence_length * self.input_features

        output = torch.empty(_batch_size, _sequence_length, self.output_features)

        x_squared = torch.empty(_batch_size, (in_dim + in_dim ** 2) // 2)
        for j in range(_batch_size):
            triu_indices = torch.triu_indices(in_dim, in_dim)
            assert triu_indices.shape[1] == x_squared.shape[1]
            x_squared[j] = torch.outer(input_[j, :].reshape(in_dim), input_[j, :].reshape(in_dim))[triu_indices[0], triu_indices[1]]
        x = torch.cat([input_.reshape(-1, in_dim), x_squared], dim=-1)

        output[:, :, :] = self.linear(x).reshape(-1, self.sequence_length, self.output_features)

        assert output.shape == (_batch_size, _sequence_length, self.output_features)
        assert output.isnan().sum() == 0
        return output


class TaylorSelfAttentionModule(Module):
    def __init__(
        self,
        input_features: int,
        d_model: int,
        nhead: int,
        output_features: int,
        sequence_length: int,
        mask: AttentionMatrixMask | str | None = None,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        use_scalar_product = True

        if use_scalar_product:
            attention_mechanism = QuadraticScalarProductAttentionModule(
                input_features=input_features,
                output_features=d_model,
                sequence_length=sequence_length,
            )
        else:
            attention_mechanism = QuadraticAttentionModule(
                input_features=input_features,
                output_features=d_model,
                sequence_length=sequence_length,
            )

        super().__init__()

        self.attention_mechanisms = [copy.deepcopy(attention_mechanism) for _ in range(nhead)]

        self.head_reduction = ConcatHeads(attention_dimension=d_model, nhead=nhead, **factory_kwargs)
        attention_output_features = self.head_reduction.attention_output_features

        self.output_module = LinearOutputModule(
            attention_output_features=attention_output_features,
            output_features=output_features,
            activation=None,
            bias=bias,
            **factory_kwargs,
        )

    @property
    def input_features(self) -> int:
        return self.attention_mechanisms[0].input_features

    @property
    def output_features(self) -> int:
        return self.output_module.output_features

    def forward(self, input_: Tensor) -> Tensor:
        head_results = []
        for layer in self.attention_mechanisms:
            head_results.append(layer(input_))
        attention_result = torch.stack(head_results)
        output = self.output_module(self.head_reduction(attention_result))
        return output
