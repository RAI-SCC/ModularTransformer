"""Mask base class."""
from abc import ABC, abstractmethod

from torch import Tensor

__all__ = ["AttentionMatrixMask"]


class AttentionMatrixMask(ABC):
    """
    Abstract base class for all `AttentionMatrixMask`s.

    `AttentionMatrixMask`s operate on a (possibly batched) two-dimensional attention matrix and proved a masked version
    of it through th `apply_to` method.
    """

    @abstractmethod
    def apply_to(self, attention_matrix: Tensor) -> Tensor:
        """
        Apply mask to given tensor.

        Accepts the attention matrix as input and return a masked version, which should have the same shape.
        """
        raise NotImplementedError
