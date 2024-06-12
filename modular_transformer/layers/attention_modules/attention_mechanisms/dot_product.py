"""Dot product attention."""
from math import sqrt
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn.functional import softmax

from . import masking
from .base import AttentionModule
from .masking import AttentionMatrixMask, TriangularMask

__all__ = [
    "DotProductAttention",
    "MaskedDotProductAttention",
]


class DotProductAttention(AttentionModule):
    """
    Classical dot product attention mechanism (Vaswani et al 17) with optional mask.

    Parameters
    ----------
        :param q_features int: number of features of the q- and k-component (i.e. first and second) input
        :param v_features Optional[int]: number of features of the v-component (i.e. third) input (default: q_features)
        :param mask Optional[AttentionMatrixMask]: mask for masked attention (default: None)
    """

    def __init__(
        self,
        q_features: int,
        v_features: Optional[int] = None,
        mask: Optional[Union[AttentionMatrixMask, str]] = None,
        **kwargs,
    ) -> None:
        self.mask = getattr(masking, mask)() if isinstance(mask, str) else mask

        super().__init__(q_features=q_features, v_features=v_features)

    def _check_validity(self) -> None:
        # q_features and k_features are identical
        assert self.q_features == self.k_features
        super()._check_validity()

    @property
    def output_features(self) -> int:
        """Return output dimension."""
        return self.v_features

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Compute layer forward pass."""
        scale = query.shape[-1]
        attention_matrix = softmax(
            torch.div(torch.matmul(query, key.transpose(-1, -2)), sqrt(scale)), dim=-1
        )
        if self.mask is not None:
            attention_matrix = self.ma
        output = torch.matmul(attention_matrix, value)

        return output


class MaskedDotProductAttention(DotProductAttention):
    """
    Alternate version of `DotProductAttention`.

    The only difference to `DotProductAttention` is that `TriangularMask` is used as default for ease of use

    Parameters
    ----------
        :param q_features int: number of features of the q- and k-component (i.e. first and second) input
        :param v_features Optional[int]: number of features of the v-component (i.e. third) input (default: q_features)
        :param mask Optional[AttentionMatrixMask]: mask for masked attention (default: TriangularMask())
    """

    def __init__(
        self,
        q_features: int,
        v_features: Optional[int] = None,
        mask: Optional[Union[AttentionMatrixMask, str]] = None,
        **kwargs,
    ) -> None:
        self.mask = (
            getattr(masking, mask)()
            if isinstance(mask, str)
            else (mask if mask is not None else TriangularMask())
        )

        super().__init__(q_features=q_features, v_features=v_features, mask=mask)
