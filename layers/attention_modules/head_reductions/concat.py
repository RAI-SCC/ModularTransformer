from .base import HeadReduction

from torch import Tensor

__all__ = [
    'ConcatHeads',
]


class ConcatHeads(HeadReduction):
    """
    Collapses the head dimension by concatenating all features

    Default approach for most attention architectures

    Parameters
        :param attention_dimension int: size of the input feature dimension
        :param nhead int: number of attention heads and size of the input head dimension
        :param device Optional[torch.device]: computation device the module is initialized on
        :param dtype Optional[torch.dtype]: data type of the module
    """
    def attention_output_features(self) -> int:
        return self.nhead * self.attention_dimension

    def forward(self, input_: Tensor) -> Tensor:
        output = input_.transpose(-2, -3).flatten(-2)
        return output
