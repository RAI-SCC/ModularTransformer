from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

__all__ = [
    "AttentionModule",
]


class AttentionModule(Module, ABC):
    """
    Abstract base class for all `AttentionModule`s

    Responsible for implementing the attention mechanism.
    `AttentionModules` should be head agnostic for head interactions, see `HeadReduction`. They have access to the last
    two dimensions (i.e. sequence_length and feature dimension), but the output must have the same sequence_length
    dimension as the query q (first input).

    All 'AttentionModule's should be initialized with the following arguments and any number of keyword arguments
        :param q_features int: number of features of the q-component (i.e. first) input
        :param k_features Optional[int]: number of features of the k-component (i.e. second) input (default: q_features)
        :param v_features Optional[int]: number of features of the v-component (i.e. third) input (default: k_features)
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    The super.__init__ call should be done at the end of the __init__ method of child classes.

    The method _check_validity() is called at the end of the super.__init__ method (which is why it should be call last)
    and can be implemented to ensure consistency of the created module. By default there are no checks.
    """

    def __init__(
        self,
        q_features: int,
        k_features: Optional[int] = None,
        v_features: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.q_features = q_features
        self.k_features = k_features or q_features
        self.v_features = v_features or self.k_features

        self._check_validity()

    def _check_validity(self) -> None:
        """Checks the consistency of the module. Should be implemented by subclasses, if needed."""
        pass

    @property
    @abstractmethod
    def output_features(self) -> int:
        """Provides the number of output_features for consistency checks"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        This accepts the three Tensors query (q), key (k), and value (v) of shape (*, D), where D is q_features,
        k_features or v_features respectively and returns a Tensor of shape (*, S, O), where S is the size of the
        second to last dimension of q and O is output_features
        """
        raise NotImplementedError
