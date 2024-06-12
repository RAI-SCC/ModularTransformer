import torch
import torch.nn as nn
from torch.nn import functional as f

#self.linear = Linear(self.attention_output_features, self.output_features, bias=bias, **factory_kwargs)
class GaussianWeightMCDLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, bias, sigma, is_trainable_sigma, **factory_kwargs):
        super().__init__()
        self.lin_layer = nn.Linear(input_size, output_size, bias=bias, **factory_kwargs)

        if is_trainable_sigma:
            t_weight = torch.ones(self.lin_layer.weight.shape) * sigma
            self.sigma_w = nn.Parameter(t_weight)
            if bias:
                t_bias = torch.ones(self.lin_layer.bias.shape) * sigma
                self.sigma_b = nn.Parameter(t_bias)

        self.sigma = sigma
        self.is_trainable_sigma = is_trainable_sigma
        self.bias = bias

    def forward(self, x):
        weights = self.lin_layer.weight
        epsilon_w = torch.randn_like(weights)
        if self.bias:
            biases = self.lin_layer.bias
            epsilon_b = torch.randn_like(biases)
        if self.is_trainable_sigma:
            masked_w = weights + (self.sigma_w * epsilon_w)
            if self.bias:
                masked_b = biases + (self.sigma_b * epsilon_b)
        else:
            masked_w = weights + ((torch.ones(self.lin_layer.weight.shape) * self.sigma) * epsilon_w)
            if self.bias:
                masked_b = biases + ((torch.ones(self.lin_layer.bias.shape) * self.sigma) * epsilon_b)
        if self.bias:
            return f.linear(x, masked_w, masked_b)
        else:
            return f.linear(x, masked_w)


class GaussianNodeMCDLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, bias, sigma, **factory_kwargs):
        super().__init__()
        self.lin_layer = nn.Linear(input_size, output_size,  bias=bias, **factory_kwargs)
        self.sigma = sigma
        self.bias = bias

    def forward(self, x):
        epsilon = torch.randn_like(x)
        x = x + (self.sigma * epsilon)
        if self.bias:
            return f.linear(x, self.lin_layer.weight, self.lin_layer.bias)
        else:
            return f.linear(x, self.lin_layer.weight)


class BernoulliWeightMCDLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, bias, dropout_rate, **factory_kwargs):
        super().__init__()
        self.lin_layer = nn.Linear(input_size, output_size, bias=bias, **factory_kwargs)
        self.dropout_rate = dropout_rate
        self.bias = bias

    def forward(self, x):
        weights = self.lin_layer.weight
        bernoulli_mask_w = torch.where(torch.rand(size=weights.shape) > self.dropout_rate, 1., 0.)
        masked_w = weights * bernoulli_mask_w
        if self.bias:
            biases = self.lin_layer.bias
            bernoulli_mask_b = torch.where(torch.rand(size=biases.shape) > self.dropout_rate, 1., 0.)
            masked_b = biases * bernoulli_mask_b
            return f.linear(x, masked_w, masked_b)
        else:
            return f.linear(x, masked_w)


class BernoulliNodeMCDLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, bias, dropout_rate, **factory_kwargs):
        super().__init__()
        self.lin_layer = nn.Linear(input_size, output_size, bias=bias, **factory_kwargs)
        self.dropout_rate = dropout_rate
        self.bias = bias

    def forward(self, x):
        x = x * (torch.where(torch.rand(size=x.shape) > self.dropout_rate, 1., 0.))
        weights = self.lin_layer.weight
        if self.bias:
            biases = self.lin_layer.bias
            return f.linear(x, weights, biases)
        else:
            return f.linear(x, weights)
