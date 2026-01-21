"""
Base class for dataset loaders.
Defines the interface all loaders must implement.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


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

        # Remove rows with empty text and log how many were dropped
        original_count = len(df)
        df = df[df["text"].notna() & (df["text"].str.strip() != "")]
        dropped_count = original_count - len(df)

        if dropped_count > 0:
            logger.warning(
                f"Dropped {dropped_count} rows with empty/missing text "
                f"({dropped_count / original_count * 100:.1f}% of {original_count} total)"
            )

        logger.info(f"Validated dataset: {len(df)} rows (from {original_count} original)")

        return df.reset_index(drop=True)

    def _auto_detect_columns(
        self,
        columns: list[str],
        column_presets: dict[str, dict[str, list[str]]]
    ) -> dict[str, str]:
        """
        Auto-detect column mapping from available columns using presets.

        This is a shared utility for loaders that need to auto-detect column mappings
        from various CSV formats. Each loader defines its own COLUMN_PRESETS.

        Args:
            columns: List of column names in the CSV
            column_presets: Dict of preset name -> {target_col: [candidate_names]}
                Example: {"default": {"text": ["tweet", "content", "text"]}}

        Returns:
            Mapping from normalized target names to actual column names
        """
        mapping = {}
        columns_lower = {c.lower(): c for c in columns}

        # Try each preset
        for preset_name, preset in column_presets.items():
            for target, candidates in preset.items():
                if target in mapping:
                    continue
                for candidate in candidates:
                    if candidate in columns:
                        mapping[target] = candidate
                        break
                    elif candidate.lower() in columns_lower:
                        mapping[target] = columns_lower[candidate.lower()]
                        break

        return mapping


# Registry for available loaders
_LOADER_REGISTRY: dict[str, type[BaseDatasetLoader]] = {}


def register_loader(name: str):
    """
    Decorator to register a loader class.

    Usage:
        @register_loader("poetry")
        class PoetryLoader(BaseDatasetLoader):
            ...

    Raises:
        TypeError: If class doesn't inherit from BaseDatasetLoader
        ValueError: If name is already registered
    """
    def decorator(cls: type[BaseDatasetLoader]):
        if not issubclass(cls, BaseDatasetLoader):
            raise TypeError(f"{cls.__name__} must inherit from BaseDatasetLoader")
        if name in _LOADER_REGISTRY:
            raise ValueError(
                f"Loader '{name}' already registered by {_LOADER_REGISTRY[name].__name__}"
            )
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
