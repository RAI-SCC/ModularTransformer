"""Attention modules."""
from .base import CrossAttentionModule, SelfAttentionModule
from .classical import ClassicalCrossAttentionModule, ClassicalSelfAttentionModule

__all__ = [
    "SelfAttentionModule",
    "CrossAttentionModule",
    "ClassicalSelfAttentionModule",
    "ClassicalCrossAttentionModule",
]
