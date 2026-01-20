"""
Base class for embedding backends.
Defines the interface all embedders must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding backends.
    
    All embedders must:
    - Accept a list of texts and return normalized embeddings
    - Report their embedding dimension
    - Provide a unique name for cache keying
    """
    
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts into vectors.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            np.ndarray of shape (len(texts), dimension) with L2-normalized vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimensionality of the embeddings.
        
        Returns:
            Integer dimension of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this embedder (used for cache key).
        
        Returns:
            String identifier for the embedder
        """
        pass
    
    @staticmethod
    def normalize(vectors: np.ndarray) -> np.ndarray:
        """
        L2-normalize vectors for cosine similarity.
        
        Args:
            vectors: Array of shape (n, dim)
            
        Returns:
            Normalized array of same shape
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms


# Registry for available embedders
_EMBEDDER_REGISTRY: dict[str, type[BaseEmbedder]] = {}


def register_embedder(name: str):
    """
    Decorator to register an embedder class.
    
    Usage:
        @register_embedder("openai")
        class OpenAIEmbedder(BaseEmbedder):
            ...
    """
    def decorator(cls: type[BaseEmbedder]):
        _EMBEDDER_REGISTRY[name] = cls
        return cls
    return decorator


def get_embedder(name: str, **kwargs) -> BaseEmbedder:
    """
    Get an embedder instance by name.
    
    Args:
        name: Registered embedder name
        **kwargs: Arguments passed to embedder constructor
        
    Returns:
        Embedder instance
        
    Raises:
        ValueError: If embedder name not found
    """
    if name not in _EMBEDDER_REGISTRY:
        available = list(_EMBEDDER_REGISTRY.keys())
        raise ValueError(f"Unknown embedder '{name}'. Available: {available}")
    
    return _EMBEDDER_REGISTRY[name](**kwargs)


def list_embedders() -> list[str]:
    """Return list of registered embedder names."""
    return list(_EMBEDDER_REGISTRY.keys())
