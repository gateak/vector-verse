"""
Poetry Foundation dataset loader.
Loads the Kaggle Poetry Foundation dataset (~14k poems).
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .base import BaseDatasetLoader, register_loader
import config


@register_loader("poetry")
class PoetryLoader(BaseDatasetLoader):
    """
    Loader for the Poetry Foundation dataset from Kaggle.
    
    Expected CSV format (auto-detected):
    - Title, Author, Poem columns (original Kaggle format)
    
    Download from: https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems
    """
    
    def __init__(
        self,
        csv_path: Optional[Path] = None,
        max_items: Optional[int] = None
    ):
        """
        Initialize the poetry loader.
        
        Args:
            csv_path: Path to the CSV file (defaults to config.POETRY_CSV_PATH)
            max_items: Optional limit on number of items to load (for testing)
        """
        self.csv_path = Path(csv_path) if csv_path else config.POETRY_CSV_PATH
        self.max_items = max_items
    
    @property
    def name(self) -> str:
        return "poetry_foundation"
    
    def load(self) -> pd.DataFrame:
        """
        Load the Poetry Foundation CSV and normalize to standard format.
        
        Returns:
            DataFrame with columns: id, title, author, text, source
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Poetry dataset not found at {self.csv_path}\n"
                f"Download from: https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems\n"
                f"Place the CSV file at: {self.csv_path}"
            )
        
        # Load CSV
        df = pd.read_csv(self.csv_path)
        
        # Auto-detect and normalize column names
        df = self._normalize_columns(df)
        
        # Add source identifier
        df["source"] = config.SOURCE_POETRY
        
        # Generate unique IDs
        df["id"] = [f"poetry_{i}" for i in range(len(df))]
        
        # Limit items if specified
        if self.max_items is not None:
            df = df.head(self.max_items)
        
        # Validate and return
        return self.validate(df)
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to standard format.
        
        Handles variations in Kaggle CSV format.
        """
        # Common column name mappings
        column_mapping = {
            # Title variations
            "Title": "title",
            "title": "title",
            "TITLE": "title",
            "poem_title": "title",
            # Author variations
            "Author": "author",
            "author": "author",
            "AUTHOR": "author",
            "poet": "author",
            "Poet": "author",
            # Text variations
            "Poem": "text",
            "poem": "text",
            "POEM": "text",
            "Content": "text",
            "content": "text",
            "text": "text",
            "poem_content": "text",
        }
        
        # Rename columns that exist
        rename_dict = {}
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in rename_dict.values():
                rename_dict[old_name] = new_name
        
        df = df.rename(columns=rename_dict)
        
        # Verify required columns exist
        required = {"title", "author", "text"}
        if not required.issubset(set(df.columns)):
            available = set(df.columns)
            missing = required - available
            raise ValueError(
                f"Could not find required columns in CSV. "
                f"Missing: {missing}. Available: {available}"
            )
        
        return df
