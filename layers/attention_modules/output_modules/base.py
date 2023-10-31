import torch
from abc import ABC, abstractmethod

from typing import Optional
from torch import Tensor
from torch.nn import Module

__all__ = [
    'OutputModule',
]


class OutputModule(Module, ABC):
    def __init__(self, attention_output_features: int, output_features: Optional[int] = None,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.attention_output_features = attention_output_features
        self.output_features = output_features or attention_output_features

        self._check_validity()

    def _check_validity(self) -> None:
        pass

    @abstractmethod
    def forward(self, input_: Tensor) -> Tensor:
        """
        This accepts a Tensor of shape (*, S, F) and should output a Tensor of shape (*, S, O), where S is the
        input sequence length, F is self.attention_output_features, and O is self.output_features
        """
        raise NotImplementedError
