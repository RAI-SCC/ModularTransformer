from .base import HeadReduction

from torch import Tensor

__all__ = [
    'ConcatHeads',
]


class ConcatHeads(HeadReduction):
    def attention_output_features(self) -> int:
        return self.nhead * self.attention_dimension

    def forward(self, input_: Tensor) -> Tensor:
        output = input_.transpose(-2, -3).flatten(-2)
        return output
