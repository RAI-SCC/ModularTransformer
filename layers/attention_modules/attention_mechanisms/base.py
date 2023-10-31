import torch
from abc import ABC, abstractmethod

from typing import Optional
from torch import Tensor
from torch.nn import Module

__all__ = [
    'AttentionModule',
]


class AttentionModule(Module, ABC):
    def __init__(self, attention_dimension: int, mask: Optional[Tensor] = None,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.attention_dimension = attention_dimension
        self.mask = mask

        self._check_validity()

    def _check_validity(self) -> None:
        pass

    @abstractmethod
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        raise NotImplementedError
