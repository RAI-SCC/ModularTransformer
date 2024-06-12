"""Identity output module."""
from torch import Tensor

from .base import OutputModule

__all__ = ["NoModule"]


class NoModule(OutputModule):
    """
    Used to not use an `OutputModule`, simply forwards the input.

    It's main function is automatically checking that the feature numbers work.
    """

    def forward(self, input_: Tensor) -> Tensor:
        """Forward pass through the module."""
        return input_

    def _check_validity(self) -> None:
        # Since the input is forwarded the output_features must equal the input_features
        assert self.attention_output_features == self.output_features
        super()._check_validity()
