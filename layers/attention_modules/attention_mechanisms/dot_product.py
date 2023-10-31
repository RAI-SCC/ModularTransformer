import torch

from .base import AttentionModule
from torch.nn.functional import softmax

from torch import Tensor

__all__ = [
    'DotProductAttention',
]


class DotProductAttention(AttentionModule):
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        scale = query.shape[-1]
        attention_matrix = softmax(torch.div(torch.matmul(query, key.transpose(-1, -2)), torch.sqrt(scale)))
        if self.mask is not None:
            attention_matrix = torch.mul(attention_matrix, self.mask)
        output = torch.matmul(attention_matrix, value)

        return output
