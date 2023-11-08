import torch
from torch import Tensor

from .base import AttentionMatrixMask

__all__ = [
    'TriangularMask'
]


class TriangularMask(AttentionMatrixMask):
    """
    Works as the standard mask blocking 'future' information
    """
    def apply_to(self, attention_matrix: Tensor) -> Tensor:
        return torch.tril(attention_matrix)
