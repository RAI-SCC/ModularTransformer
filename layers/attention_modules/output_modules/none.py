from .base import OutputModule

from torch import Tensor

__all__ = [
    'NoModule'
]


class NoModule(OutputModule):
    def forward(self, input_: Tensor) -> Tensor:
        return input_

    def _check_validity(self) -> None:
        assert self.attention_output_features == self.output_features
        super()._check_validity()
