"""Layers used to combine the output of multiple attention heads input one output through concatenation."""
from torch import Tensor

from .base import HeadReduction

__all__ = [
    "ConcatHeads",
]


class ConcatHeads(HeadReduction):
    """
    Collapses the head dimension by concatenating all features.

    Default approach for most attention architectures using a sequence of input vectors

    Parameters
    ----------
        :param attention_dimension int: size of the input feature dimension
        :param nhead int: number of attention heads and size of the input head dimension
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    """

    @property
    def attention_output_features(self) -> int:
        """Compute the total output dimension."""
        return self.nhead * self.attention_dimension

    def forward(self, input_: Tensor) -> Tensor:
        """Compute forward pass of the module."""
        output = input_.movedim(0, -2).flatten(-2)
        return output
