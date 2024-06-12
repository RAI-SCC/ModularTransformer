"""Layers used to combine the output of multiple attention heads input one output."""
from .base import HeadReduction
from .concat import ConcatHeads

__all__ = [
    "HeadReduction",
    "ConcatHeads",
]
