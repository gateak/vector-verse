"""
Base class for dataset loaders.
Defines the interface all loaders must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class BaseDatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    All loaders must return a DataFrame with these required columns:
    - id: Unique identifier for each item
    - title: Display title
    - author: Creator/author name
    - text: Main text content (used for embedding)
    - source: Origin identifier (e.g., "poetry", "custom")
    
    Additional metadata columns are allowed and will be preserved.
    """
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load and return the dataset as a DataFrame.
        
        Returns:
            pd.DataFrame with required columns: id, title, author, text, source
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this dataset (used for cache key).
        
        Returns:
            String identifier for the dataset
        """
        pass
    
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = {"id", "title", "author", "text", "source"}
        missing = required_columns - set(df.columns)
        
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        
        # Ensure id is string type for consistent handling
        df["id"] = df["id"].astype(str)
        
        # Remove rows with empty text
        df = df[df["text"].notna() & (df["text"].str.strip() != "")]
        
        return df.reset_index(drop=True)


# Registry for available loaders
_LOADER_REGISTRY: dict[str, type[BaseDatasetLoader]] = {}


def register_loader(name: str):
    """
    Decorator to register a loader class.
    
    Usage:
        @register_loader("poetry")
        class PoetryLoader(BaseDatasetLoader):
            ...
    """
    def decorator(cls: type[BaseDatasetLoader]):
        _LOADER_REGISTRY[name] = cls
        return cls
    return decorator


def get_loader(name: str, **kwargs) -> BaseDatasetLoader:
    """
    Get a loader instance by name.
    
    Args:
        name: Registered loader name
        **kwargs: Arguments passed to loader constructor
        
    Returns:
        Loader instance
        
    Raises:
        ValueError: If loader name not found
    """
    if name not in _LOADER_REGISTRY:
        available = list(_LOADER_REGISTRY.keys())
        raise ValueError(f"Unknown loader '{name}'. Available: {available}")
    
    return _LOADER_REGISTRY[name](**kwargs)


def list_loaders() -> list[str]:
    """Return list of registered loader names."""
    return list(_LOADER_REGISTRY.keys())
