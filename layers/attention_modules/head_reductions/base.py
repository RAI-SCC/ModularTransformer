import torch
from abc import ABC, abstractmethod

from typing import Optional
from torch import Tensor
from torch.nn import Module

__all__ = [
    'HeadReduction',
]


class HeadReduction(Module, ABC):
    """
    Abstract base class for all `HeadReduction`s

    Responsible for combining the heads of multihead attention. The vast majority of models us ConcatHeads, but also
    allows to implement layers with head interaction.

    All `HeadReduction`s should be implemented with the following arguments and any number of keyword arguments
        :param attention_dimension int: size of the input feature dimension
        :param nhead int: number of attention heads and size of the input head dimension
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    The super.__init__ call should be done at the end of the __init__ method of child classes.

    Each child class needs to implement the attention_output_features property, which provides the number of output
    features for consistency checks.

    The method _check_validity() is called at the end of the super.__init__ method (which is why it should be call last)
    and can be implemented to ensure consistency of the created module. By default there are no checks.
    """
    def __init__(self, attention_dimension: int, nhead: int,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.attention_dimension = attention_dimension
        self.nhead = nhead

        self._check_validity()

    def _check_validity(self) -> None:
        """Checks the consistency of the module. Should be implemented by subclasses, if needed."""
        pass

    @property
    @abstractmethod
    def attention_output_features(self) -> int:
        """Provides the number of output_features for consistency checks"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, input_: Tensor) -> Tensor:
        """
        This accepts a Tensor of shape (*, H, S, A) and should output a Tensor of shape (*, S, F), where S is the
        input sequence length, A is self.attention_dimension, H is self.nhead, and F is self.attention_output_features
        """
        raise NotImplementedError

