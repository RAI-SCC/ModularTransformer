import torch
from abc import ABC, abstractmethod

from typing import Tuple, Optional
from torch import Tensor
from torch.nn import Module

__all__ = [
    'Qmap',
    'KVmap',
    'QKVmap',
]


class Qmap(Module, ABC):
    def __init__(self, input_features: int, attention_dimension: int, nhead: int = 1,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.input_features = input_features
        self.attention_dimension = attention_dimension
        self.nhead = nhead
        self.output_shape = torch.Size(nhead, attention_dimension, 1)

        self._check_validity()

    def _check_validity(self) -> None:
        pass

    @abstractmethod
    def forward(self, input_: Tensor) -> Tensor:
        """
        This accepts a Tensor of shape (*, S, F) and should output a Tensor of shape (*, H, S, A), where S is the
        input sequence length, F is self.attention_output_features, H is self.nhead, and A is self.attention_dimension
        """
        raise NotImplementedError


class KVmap(Module, ABC):
    def __init__(self, input_features: int, attention_dimension: int, nhead: int = 1,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.input_features = input_features
        self.attention_dimension = attention_dimension
        self.nhead = nhead
        self.output_shape = torch.Size(nhead, attention_dimension, 2)

        self._check_validity()

    def _check_validity(self) -> None:
        pass

    @property
    def k_features(self) -> int:
        return self.attention_dimension

    @abstractmethod
    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        """
        This accepts a Tensor of shape (*, S, F) and should output two Tensors of shape (*, H, S, A), where S is the
        input sequence length, F is self.attention_output_features, H is self.nhead, and A is self.attention_dimension
        """
        raise NotImplementedError


class QKVmap(Module, ABC):
    def __init__(self, input_features: int, attention_dimension: int, nhead: int = 1,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.input_features = input_features
        self.attention_dimension = attention_dimension
        self.nhead = nhead
        self.output_shape = torch.Size(nhead, attention_dimension, 3)

        self._check_validity()

    def _check_validity(self) -> None:
        pass

    @property
    def k_features(self) -> int:
        return self.attention_dimension

    @abstractmethod
    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        This accepts a Tensor of shape (*, S, F) and should output three Tensors of shape (*, H, S, A), where S is the
        input sequence length, F is self.attention_output_features, H is self.nhead, and A is self.attention_dimension
        """
        raise NotImplementedError
