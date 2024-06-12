"""Collection of neural network layers implemented as torch.nn.module."""
from .attention_modules import output_modules
from .base import TransformerDecoderLayer, TransformerEncoderLayer
from .classical import ClassicalTransformerDecoderLayer, ClassicalTransformerEncoderLayer

__all__ = [
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "ClassicalTransformerEncoderLayer",
    "ClassicalTransformerDecoderLayer",
    "output_modules",
]
