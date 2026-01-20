"""
Custom items loader for user's own content.
Loads user-provided CSV with poems, notes, or any text items.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import BaseDatasetLoader, register_loader
import config


@register_loader("custom")
class CustomItemsLoader(BaseDatasetLoader):
    """
    Loader for user's custom items (poems, notes, etc.).
    
    Expected CSV format:
    - title: Item title
    - author: Author name
    - text: Main content
    - language: Optional language tag (e.g., "en", "tr")
    
    All columns except 'text' are optional and will use defaults.
    """
    
    def __init__(
        self,
        csv_path: Optional[Path] = None,
        default_author: str = "User"
    ):
        """
        Initialize the custom items loader.
        
        Args:
            csv_path: Path to the CSV file (defaults to config.CUSTOM_ITEMS_PATH)
            default_author: Default author name if not specified in CSV
        """
        self.csv_path = Path(csv_path) if csv_path else config.CUSTOM_ITEMS_PATH
        self.default_author = default_author
    
    @property
    def name(self) -> str:
        return "custom_items"
    
    def load(self) -> pd.DataFrame:
        """
        Load custom items CSV and normalize to standard format.
        
        Returns:
            DataFrame with columns: id, title, author, text, source, language
            Returns empty DataFrame if file doesn't exist.
        """
        # Return empty DataFrame if file doesn't exist
        if not self.csv_path.exists():
            return pd.DataFrame(columns=["id", "title", "author", "text", "source", "language"])
        
        # Load CSV
        df = pd.read_csv(self.csv_path)
        
        if df.empty:
            return pd.DataFrame(columns=["id", "title", "author", "text", "source", "language"])
        
        # Normalize columns
        df = self._normalize_columns(df)
        
        # Add source identifier
        df["source"] = config.SOURCE_CUSTOM
        
        # Generate unique IDs
        df["id"] = [f"custom_{i}" for i in range(len(df))]
        
        # Validate and return
        return self.validate(df)
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize columns and fill in defaults for missing fields.
        """
        # Ensure text column exists (required)
        text_candidates = ["text", "Text", "content", "Content", "poem", "Poem", "body", "Body"]
        text_col = None
        for col in text_candidates:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            raise ValueError(
                f"Custom items CSV must have a 'text' column. "
                f"Available columns: {list(df.columns)}"
            )
        
        if text_col != "text":
            df = df.rename(columns={text_col: "text"})
        
        # Handle title column
        if "title" not in df.columns:
            if "Title" in df.columns:
                df = df.rename(columns={"Title": "title"})
            else:
                # Generate titles from text preview
                df["title"] = df["text"].str[:50].str.strip() + "..."
        
        # Handle author column
        if "author" not in df.columns:
            if "Author" in df.columns:
                df = df.rename(columns={"Author": "author"})
            else:
                df["author"] = self.default_author
        
        # Handle language column (optional, preserve if present)
        if "language" not in df.columns:
            if "Language" in df.columns:
                df = df.rename(columns={"Language": "language"})
            else:
                df["language"] = "unknown"
        
        return df
    
    def exists(self) -> bool:
        """Check if custom items file exists."""
        return self.csv_path.exists()
