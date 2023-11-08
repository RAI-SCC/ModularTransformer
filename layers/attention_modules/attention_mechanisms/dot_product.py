import torch

from . import masking
from .base import AttentionModule
from torch.nn.functional import softmax

from typing import Optional, Union
from torch import Tensor
from .masking import AttentionMatrixMask

__all__ = [
    'DotProductAttention',
]


class DotProductAttention(AttentionModule):
    """
    Classical dot product attention mechanism (Vaswani et al 17) with optional mask

    Parameters
        :param q_features int: number of features of the q- and k-component (i.e. first and second) input
        :param v_features Optional[int]: number of features of the v-component (i.e. third) input (default: q_features)
        :param mask Optional[AttentionMatrixMask]: mask for masked attention (default
    """
    def __init__(
            self,
            q_features: int,
            v_features: Optional[int] = None,
            mask: Optional[Union[AttentionMatrixMask, str]] = None,
            **kwargs
    ) -> None:
        self.mask = getattr(masking, mask)() if isinstance(mask, str) else mask

        super().__init__(
            q_features=q_features,
            v_features=v_features
        )

    def _check_validity(self) -> None:
        # q_features and k_features are identical
        assert self.q_features == self.k_features
        super()._check_validity()

    def output_features(self) -> int:
        return self.v_features

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        scale = query.shape[-1]
        attention_matrix = softmax(torch.div(torch.matmul(query, key.transpose(-1, -2)), torch.sqrt(scale)))
        if self.mask is not None:
            attention_matrix = self.ma
        output = torch.matmul(attention_matrix, value)

        return output
