import copy
import torch
from abc import ABC, abstractmethod

from typing import Tuple, Optional, overload
from torch import Tensor
from torch.nn import Module

__all__ = [
    'Qmap',
    'KVmap',
    'QKVmap',
    'CompositeKVmap',
    'CompositeQKVmap',
]


class Qmap(Module, ABC):
    def __init__(self, input_features: int, q_features: int,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.input_features = input_features
        self.q_features = q_features

        self._check_validity()

    def _check_validity(self) -> None:
        pass

    @abstractmethod
    def forward(self, input_: Tensor) -> Tensor:
        """
        This accepts a Tensor of shape (*, S, F) and should output a Tensor of shape (*, H, S, Q), where S is the
        input sequence length, F is input_features, and Q is q_features
        """
        raise NotImplementedError

    @overload
    def __add__(self, other: 'Qmap') -> 'KVmap':
        ...

    @overload
    def __add__(self, other: 'KVmap') -> 'QKVmap':
        ...

    def __add__(self, other):
        if isinstance(other, Qmap):
            return CompositeKVmap(self, other)
        if isinstance(other, KVmap):
            return CompositeQKVmap(self, other)
        else:
            raise TypeError

    def as_KVmap(self) -> 'KVmap':
        k_map = self
        v_map = copy.deepcopy(self)
        return CompositeKVmap(k_map, v_map)

    def as_QKVmap(self) -> 'QKVmap':
        q_map = copy.deepcopy(self)
        kv_map = self.as_KVmap()
        return CompositeQKVmap(q_map, kv_map)


class KVmap(Module, ABC):
    def __init__(self, input_features: int, k_features: int, v_features: Optional[int] = None,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.input_features = input_features
        self.k_features = k_features
        self.v_features = v_features or k_features

        self._check_validity()

    def _check_validity(self) -> None:
        pass

    def __add__(self, other):
        if isinstance(other, Qmap):
            raise TypeError('Use Qmap + KVmap instead! This prevents counterintuitive behaviour when adding 3 Qmaps')
        else:
            raise TypeError

    @abstractmethod
    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        """
        This accepts a Tensor of shape (*, S, F) and should output two Tensors of shape (*, S, K) and (*, S, V),
        where S is the input sequence length, F is input_features, K is k_features, and V is v_features
        """
        raise NotImplementedError


class QKVmap(Module, ABC):
    def __init__(self, input_features: int, q_features: int, k_features: Optional[int] = None, v_features: Optional[int] = None,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.input_features = input_features
        self.q_features = q_features
        self.k_features = k_features or q_features
        self.v_features = v_features or self.v_features

        self._check_validity()

    def _check_validity(self) -> None:
        pass

    @abstractmethod
    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        This accepts a Tensor of shape (*, S, F) and should output three Tensors of shape (*, H, S, A), where S is the
        input sequence length, F is self.attention_output_features, H is self.nhead, and A is self.attention_dimension
        """
        raise NotImplementedError


class CompositeKVmap(KVmap):
    def __init__(self, k_map: Qmap, v_map: Qmap) -> None:
        self.k_map = k_map
        self.v_map = v_map

        super().__init__(
            input_features=self.k_map.input_features,
            k_features=self.k_map.q_features,
            v_features=self.v_map.q_features
        )

    def _check_validity(self) -> None:
        super()._check_validity()
        assert self.k_map.input_features == self.v_map.input_features

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        k = self.k_map(input_)
        v = self.v_map(input_)
        return k, v


class CompositeQKVmap(QKVmap):
    def __init__(self, q_map: Qmap, kv_map: KVmap) -> None:
        self.q_map = q_map
        self.kv_map = kv_map

        super().__init__(
            input_features=self.q_map.input_features,
            q_features=self.q_map.q_features,
            k_features=self.kv_map.k_features,
            v_features=self.kv_map.v_features
        )

    def _check_validity(self) -> None:
        super()._check_validity()
        assert self.q_map.input_features == self.kv_map.input_features

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        q = self.q_map(input_)
        k, v = self.kv_map(input_)
        return q, k, v
