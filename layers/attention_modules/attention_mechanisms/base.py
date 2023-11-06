import torch
from abc import ABC, abstractmethod

from typing import Optional
from torch import Tensor
from torch.nn import Module

__all__ = [
    'AttentionModule',
]


class AttentionModule(Module, ABC):
    def __init__(self, q_features: int, k_features: Optional[int] = None, v_features: Optional[int] = None,
                 mask: Optional[Tensor] = None, device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.q_features = q_features
        self.k_features = k_features or q_features
        self.v_features = v_features or self.k_features
        self.mask = mask

        self._check_validity()

    def _check_validity(self) -> None:
        pass

    @property
    @abstractmethod
    def output_features(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        raise NotImplementedError
