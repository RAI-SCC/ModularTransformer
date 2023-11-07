import torch
from abc import ABC, abstractmethod

from typing import Optional
from torch import Tensor
from torch.nn import Module

__all__ = [
    'OutputModule',
]


class OutputModule(Module, ABC):
    """
    Abstract base class for all 'OutputModule's

    'OutputModule's can be any torch.nn.Module that takes one input Tensor and provides an output Tensor of the same
    shape except (possibly) in the last dimension.

    All 'OutputModules's should be initialized with the following arguments and any number of keyword arguments
        :param attention_output_features int: number of input nodes and size of the feature dimension of the intended input
        :param output_features int: number of output features (default: attention_output_features)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    The super.__init__ call should be done at the end of the __init__ method of child classes.

    The method _check_validity() is called at the end of the super.__init__ method (which is why it should be call last)
    and can be implemented to ensure consistency of the created module. By default there are no checks.
    """
    def __init__(self, attention_output_features: int, output_features: Optional[int] = None,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, **kwargs) -> None:
        super().__init__()
        self.attention_output_features = attention_output_features
        self.output_features = output_features or attention_output_features

        self._check_validity()

    def _check_validity(self) -> None:
        """Checks the consistency of the module. Should be implemented by subclasses, if needed."""
        pass

    @abstractmethod
    def forward(self, input_: Tensor) -> Tensor:
        """
        This accepts a Tensor of shape (*, S, F) and should output a Tensor of shape (*, S, O), where S is the
        input sequence length, F is self.attention_output_features, and O is self.output_features
        """
        raise NotImplementedError
