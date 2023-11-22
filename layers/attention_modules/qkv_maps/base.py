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
    """
    Abstract base class for all 'Qmap's

    'Qmap's are torch.modules with a consistent interface for use in mapping from an input to query, key or value of an
    attention_mechanism.
    All 'Qmap's should be initialized with the following arguments and any number of keyword arguments
        :param input_features int: number of input nodes and size of the feature dimension of the intended input
        :param q_features int: number of output features
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    The super.__init__ call should be done at the end of the __init__ method of child classes.

    Two 'Qmap's can be added yielding a KVmap, where the first argument becomes the K-component and the second the
    V-component. Similarly, Qmap + KVmap = QKVmap. KVmap + Qmap is intentionally disabled, since it would enable
    Qmap + Qmap + Qmap = QKVmap, which counterintuitively has the third input working as the Q-component. Use
    Qmap + (Qmap + Qmap) = QKVmap instead, if needed.

    Additionally, the methods as_KVmap()/as_QKVmap() create a KVmap/QKVmap that uses the Qmap for each component.
    Note that, while convenient, it is usually slightly more efficient to implement dedicated KV/QKVmaps

    The method _check_validity() is called at the end of the super.__init__ method (which is why it should be call last)
    and can be implemented to ensure consistency of the created module. By default there are no checks.
    """
    def __init__(self, input_features: int, q_features: int,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.input_features = input_features
        self.q_features = q_features

        self._check_validity()

    def _check_validity(self) -> None:
        """Checks the consistency of the module. Should be implemented by subclasses, if needed."""
        pass

    @abstractmethod
    def forward(self, input_: Tensor) -> Tensor:
        """
        This accepts a Tensor of shape (*, S, F) and should output a Tensor of shape (*, S, Q), where S is the
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
        """
        Two 'Qmap's can be added yielding a KVmap, where the first argument becomes the K-component and the second the
        V-component. Similarly, Qmap + KVmap = QKVmap. KVmap + Qmap is intentionally disabled, since it would enable
        Qmap + Qmap + Qmap = QKVmap, which counterintuitively has the third input working as the Q-component. Use
        Qmap + (Qmap + Qmap) = QKVmap instead, if needed.

        :param other: Qmap/KVmap to composite
        :return: CompositeKVmap/QKVmap
        """
        if isinstance(other, Qmap):
            return CompositeKVmap(self, other)
        if isinstance(other, KVmap):
            return CompositeQKVmap(self, other)
        else:
            raise TypeError

    def as_KVmap(self) -> 'CompositeKVmap':
        """
        Creates a KVmap that uses independent copies of self for both components
        :return CompositeKVmap:
        """
        k_map = self
        v_map = copy.deepcopy(self)
        return CompositeKVmap(k_map, v_map)

    def as_QKVmap(self) -> 'CompositeQKVmap':
        """
        Creates a QKVmap that uses independent copies of self for all components
        :return CompositeQKVmap:
        """
        q_map = copy.deepcopy(self)
        kv_map = self.as_KVmap()
        return CompositeQKVmap(q_map, kv_map)


class KVmap(Module, ABC):
    """
    Abstract base class for all 'KVmap's

    'KVmap's are torch.modules with a consistent interface for use in mapping from an input to key and value of an
    attention_mechanism.
    All 'KVmap's should be initialized with the following arguments and any number of keyword arguments
        :param input_features int: number of input nodes and size of the feature dimension of the intended input
        :param k_features int: number of output features of the k-component (i.e. first) output
        :param v_features Optional[int]: number of output features of the v-component (i.e. second) output (default: k_features)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    The super.__init__ call should be done at the end of the __init__ method of child classes.

    'KVmap's can be created from 'Qmap's via adding or the as_KVmap method. See the Qmap class for details.
    Note that, while convenient, it is usually slightly more efficient to implement dedicated KVmaps.

    The method _check_validity() is called at the end of the super.__init__ method (which is why it should be call last)
    and can be implemented to ensure consistency of the created module. By default there are no checks.
    """
    def __init__(self, input_features: int, k_features: int, v_features: Optional[int] = None,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.input_features = input_features
        self.k_features = k_features
        self.v_features = v_features or k_features

        self._check_validity()

    def _check_validity(self) -> None:
        """Checks the consistency of the module. Should be implemented by subclasses, if needed."""
        pass

    def __add__(self, other) -> None:
        """Raises an error, see Qmap for details"""
        if isinstance(other, Qmap):
            raise TypeError('Use Qmap + KVmap instead! This prevents counterintuitive behaviour when adding 3 Qmaps')
        else:
            raise TypeError

    @abstractmethod
    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        """
        This accepts a Tensor of shape (*, S, F) and should output two Tensors of shape (*, S, K), and (*, S, V),
        where S is the input sequence length, F is input_features, K is k_features, and V is v_features
        """
        raise NotImplementedError


class QKVmap(Module, ABC):
    """
    Abstract base class for all 'QKVmap's

    'KVmap's are torch.modules with a consistent interface for use in mapping from an input to query, key, and value of
    an attention_mechanism.
    All 'QKVmap's should be initialized with the following arguments and any number of keyword arguments
        :param input_features int: number of input nodes and size of the feature dimension of the intended input
        :param q_features int: number of output features of the q-component (i.e. first) output
        :param k_features Optional[int]: number of output features of the k-component (i.e. second) output (default: q_features)
        :param v_features Optional[int]: number of output features of the v-component (i.e. third) output (default: k_features)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    The super.__init__ call should be done at the end of the __init__ method of child classes.

    'QKVmap's can be created from the sum of a Qmap and a KVmap or the Qmap.as_QKVmap method. See the Qmap class for details.
    Note that, while convenient, it is usually slightly more efficient to implement dedicated QKVmaps.

    The method _check_validity() is called at the end of the super.__init__ method (which is why it should be call last)
    and can be implemented to ensure consistency of the created module. By default there are no checks.
    """

    def __init__(self, input_features: int, q_features: int, k_features: Optional[int] = None, v_features: Optional[int] = None,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.input_features = input_features
        self.q_features = q_features
        self.k_features = k_features or q_features
        self.v_features = v_features or self.k_features

        self._check_validity()

    def _check_validity(self) -> None:
        """Checks the consistency of the module. Should be implemented by subclasses, if needed."""
        pass

    @abstractmethod
    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        This accepts a Tensor of shape (*, S, F) and should output three Tensors of shape (*, S, Q), (*, S, K),
        and (*, S, V) where S is the input sequence length, F is input_features, Q is q_features, K is k_features,
        and V is v_features
        """
        raise NotImplementedError


class CompositeKVmap(KVmap):
    """Combines two 'Qmaps' into a KVmap, via deepcopying and forwarding the input."""
    def __init__(self, k_map: Qmap, v_map: Qmap) -> None:
        self.k_map = k_map
        self.v_map = v_map

        super().__init__(
            input_features=self.k_map.input_features,
            k_features=self.k_map.q_features,
            v_features=self.v_map.q_features
        )

    def _check_validity(self) -> None:
        """Checks the consistency of the module"""
        super()._check_validity()
        assert self.k_map.input_features == self.v_map.input_features

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor]:
        k = self.k_map(input_)
        v = self.v_map(input_)
        return k, v


class CompositeQKVmap(QKVmap):
    """Combines a Qmap and a KVmap into a QKVmap by forwarding the input."""
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
        """Checks the consistency of the module"""
        super()._check_validity()
        assert self.q_map.input_features == self.kv_map.input_features

    def forward(self, input_: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        q = self.q_map(input_)
        k, v = self.kv_map(input_)
        return q, k, v
