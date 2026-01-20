"""
Embedding backends for Vector-Verse.
"""

from .base import BaseEmbedder
from .openai_embedder import OpenAIEmbedder

__all__ = ["BaseEmbedder", "OpenAIEmbedder"]
