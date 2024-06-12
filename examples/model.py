from math import ceil, sqrt

import torch
from torch import nn


def linear_hidden_size(input_size: int, output_size: int, hidden_size_quadratic) -> int:
    """
    Calculates the hidden size of a linear model with approximately the same number of parameters as the quadratic model.
    """
    return ceil(
        (
            (input_size + input_size**2 + 1) * hidden_size_quadratic
            + (hidden_size_quadratic + hidden_size_quadratic**2) * output_size
        )
        / (input_size + 1 + output_size)
    )


def get_hidden_size(size_function, quadratic_size: int) -> int:
    """
    Calculates the hidden size of a linear model with approximately the same number of parameters as the quadratic model.
    """
    for i in range(1000):
        if size_function(i) > quadratic_size:
            return i
    raise ValueError("Could not find hidden size")


def multiple_linear_hidden_size(
    input_size: int, output_size: int, hidden_size_quadratic: int
) -> int:
    """
    Calculates the hidden size of a linear model with hidden layers with approximately the same number of parameters as the quadratic model.
    """
    return ceil(
        (
            -(input_size + 2 + output_size)
            + sqrt(
                (input_size + 2 + output_size) ** 2
                + 4
                * hidden_size_quadratic
                * (
                    input_size
                    + input_size**2
                    + 1
                    + output_size
                    + output_size * hidden_size_quadratic
                )
            )
        )
        / 2
    )


def multiple_linear_hidden_size_multiple_quad(
    input_size: int, output_size: int, hidden_size_quadratic: int
) -> int:
    """
    Calculates the hidden size of a linear model with hidden layers with approximately the same number of parameters as the quadratic model.
    """
    return ceil(
        (
            -(input_size + 2 + output_size)
            + sqrt(
                (input_size + 2 + output_size) ** 2
                + 4
                * hidden_size_quadratic
                * (
                    input_size
                    + input_size**2
                    + 2
                    + hidden_size_quadratic
                    + hidden_size_quadratic**2
                    + output_size
                    + output_size * hidden_size_quadratic
                )
            )
        )
        / 2
    )


def size_quadratic(input_size: int, output_size: int, hidden_layers: list[int]) -> int:
    params = 0
    dim_in = input_size
    for dim_hidden in hidden_layers:
        params += (dim_in + (dim_in**2 + dim_in) // 2 + 1) * dim_hidden
        dim_in = dim_hidden
    params += (dim_in + (dim_in**2 + dim_in) // 2 + 1) * output_size
    return params


def get_linear_model(
    input_size: int, output_size: int, hidden_size: int = 10, device: torch.device | None = None
):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size, device=device),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size, device=device),
    )
    # n_params: (input_size + 1) * dim_hidden + (dim_hidden + 1) * output_size


def get_linear_model_multiple_layers(
    input_size: int, output_size: int, hidden_size: int = 10, device: torch.device | None = None
):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size, device=device),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size, device=device),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size, device=device),
    )
    # n_params: (input_size + 1) * dim_hidden
    #         + (dim_hidden + 1) * dim_hidden
    #         + (dim_hidden + 1) * output_size


class QuadraticModel(nn.Module):
    def __init__(
        self,
        dim_in: int,
        hidden_layers: list[int],
        dim_out: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_layers = hidden_layers
        self.flatten = nn.Flatten()
        for i, dim_hidden in enumerate(hidden_layers):
            setattr(
                self,
                f"lin{i}",
                nn.Linear(dim_in + (dim_in**2 + dim_in) // 2, dim_hidden, device=device),
            )
            dim_in = dim_hidden
        setattr(
            self,
            f"lin{len(hidden_layers)}",
            nn.Linear(dim_in + (dim_in**2 + dim_in) // 2, dim_out, device=device),
        )

    def n_params(self):
        n_params = 0
        for i in range(len(self.hidden_layers) + 1):
            n_params += getattr(self, f"lin{i}").weight.numel()
        return n_params

    def forward(self, x):
        _batch_size = x.shape[0]
        # x.shape: (batch_size, 1, dim_in, dim_in)
        x = self.flatten(x)
        # x.shape: (batch_size, dim_in)
        for i in range(len(self.hidden_layers) + 1):
            _dim_in = x.shape[1]
            layer = getattr(self, f"lin{i}")
            x_squared = torch.empty(_batch_size, (_dim_in**2 + _dim_in) // 2)
            for j in range(_batch_size):
                triu_indices = torch.triu_indices(_dim_in, _dim_in)
                assert triu_indices.shape[1] == x_squared.shape[1]
                x_squared[j] = torch.outer(x[j], x[j])[triu_indices[0], triu_indices[1]]
            x = torch.cat([x, x_squared], dim=-1)
            # x.shape = (batch_size, dim_in + dim_in ** 2)
            x = layer(x)
        # x.shape = (batch_size, dim_out)
        assert x.shape == (_batch_size, self.dim_out)
        return x


class CubicModel(nn.Module):
    @staticmethod
    def dim_cubed(dim_in: int) -> int:
        return dim_in + (dim_in**2 + dim_in) // 2 + dim_in

    def __init__(
        self,
        dim_in: int,
        hidden_layers: list[int],
        dim_out: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_layers = hidden_layers
        self.flatten = nn.Flatten()
        for i, dim_hidden in enumerate(hidden_layers):
            setattr(self, f"lin{i}", nn.Linear(self.dim_cubed(dim_in), dim_hidden, device=device))
            dim_in = dim_hidden
        setattr(
            self,
            f"lin{len(hidden_layers)}",
            nn.Linear(self.dim_cubed(dim_in), dim_out, device=device),
        )

    def n_params(self):
        n_params = 0
        for i in range(len(self.hidden_layers) + 1):
            n_params += getattr(self, f"lin{i}").weight.numel()
        return n_params

    def forward(self, x):
        _batch_size = x.shape[0]
        # x.shape: (batch_size, 1, dim_in, dim_in)
        x = self.flatten(x)
        # x.shape: (batch_size, dim_in)
        for i in range(len(self.hidden_layers) + 1):
            _dim_in = x.shape[1]
            layer = getattr(self, f"lin{i}")
            x_squared = torch.empty(_batch_size, self.dim_cubed(_dim_in) - 2 * _dim_in)
            x_cubed = torch.empty(_batch_size, _dim_in)
            for j in range(_batch_size):
                triu_indices = torch.triu_indices(_dim_in, _dim_in)
                assert triu_indices.shape[1] == x_squared.shape[1]
                x_squared[j] = torch.outer(x[j], x[j])[triu_indices[0], triu_indices[1]]
                x_cubed[j] = x[j] ** 3
            x = torch.cat([x, x_squared, x_cubed], dim=-1)
            # x.shape = (batch_size, dim_in + dim_in ** 2)
            x = layer(x)
        # x.shape = (batch_size, dim_out)
        assert x.shape == (_batch_size, self.dim_out)
        return x


class CubicModelLarge(nn.Module):
    @staticmethod
    def dim_cubed(dim_in: int) -> int:
        return dim_in + (dim_in**2 + dim_in) // 2 + (dim_in**2 + dim_in) // 2 * dim_in

    def __init__(
        self,
        dim_in: int,
        hidden_layers: list[int],
        dim_out: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_layers = hidden_layers
        self.flatten = nn.Flatten()
        for i, dim_hidden in enumerate(hidden_layers):
            setattr(self, f"lin{i}", nn.Linear(self.dim_cubed(dim_in), dim_hidden, device=device))
            dim_in = dim_hidden
        setattr(
            self,
            f"lin{len(hidden_layers)}",
            nn.Linear(self.dim_cubed(dim_in), dim_out, device=device),
        )

    def n_params(self):
        n_params = 0
        for i in range(len(self.hidden_layers) + 1):
            n_params += getattr(self, f"lin{i}").weight.numel()
        return n_params

    def forward(self, x):
        _batch_size = x.shape[0]
        # x.shape: (batch_size, 1, dim_in, dim_in)
        x = self.flatten(x)
        # x.shape: (batch_size, dim_in)
        for i in range(len(self.hidden_layers) + 1):
            _dim_in = x.shape[1]
            layer = getattr(self, f"lin{i}")
            x_squared = torch.empty(_batch_size, (_dim_in**2 + _dim_in) // 2)
            x_cubed = torch.empty(_batch_size, (_dim_in**2 + _dim_in) // 2 * _dim_in)
            for j in range(_batch_size):
                triu_indices = torch.triu_indices(_dim_in, _dim_in)
                assert triu_indices.shape[1] == x_squared.shape[1]
                x_squared[j] = torch.outer(x[j], x[j])[triu_indices[0], triu_indices[1]]
                cubed = torch.outer(x[j], x_squared[j]).flatten()
                assert cubed.shape[0] == x_cubed.shape[1]
                x_cubed[j] = cubed
            x = torch.cat([x, x_squared, x_cubed], dim=-1)
            # x.shape = (batch_size, dim_in + dim_in ** 2)
            x = layer(x)
        # x.shape = (batch_size, dim_out)
        assert x.shape == (_batch_size, self.dim_out)
        return x
