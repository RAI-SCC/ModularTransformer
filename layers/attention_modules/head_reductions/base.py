import torch
from abc import ABC, abstractmethod

from typing import Optional
from torch import Tensor
from torch.nn import Module

__all__ = [
    'HeadReduction',
]


class HeadReduction(Module, ABC):
    def __init__(self, attention_dimension: int, nhead: int,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.attention_dimension = attention_dimension
        self.nhead = nhead

        self._check_validity()

    def _check_validity(self) -> None:
        pass

    @property
    @abstractmethod
    def attention_output_features(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_: Tensor) -> Tensor:
        """
        This accepts a Tensor of shape (*, H, S, A) and should output a Tensor of shape (*, S, F), where S is the
        input sequence length, A is self.attention_dimension, H is self.nhead, and F is self.attention_output_features
        """
        raise NotImplementedError

