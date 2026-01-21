"""
Combined tweets loader for comparing multiple tweet sources.

Loads multiple datasets and labels them by source type for visualization.
"""

from pathlib import Path
from typing import Optional
import re

import pandas as pd

from .base import BaseDatasetLoader, register_loader
import config


@register_loader("combined_tweets")
class CombinedTweetsLoader(BaseDatasetLoader):
    """
    Loader that combines multiple tweet sources for comparison.

    Adds a 'tweet_type' column for coloring by source.
    """

    # Available tweet sources with their configs
    TWEET_SOURCES = {
        "russian_bot": {
            "file": "tweets.csv",
            "label": "Russian Bot",
            "text_col": "text",
            "user_col": "user_key",
            "date_col": "created_str",
        },
        "chatgpt_2023": {
            "file": "Twitter Jan Mar.csv",
            "label": "ChatGPT 2023",
            "text_col": "content",
            "user_col": "username",
            "date_col": "date",
        },
        "feb_2024": {
            "file": "feb2024_tweets.csv",
            "label": "Feb 2024",
            "text_col": "text",
            "user_col": "username",
            "date_col": "date",
        },
        "sentiment": {
            "file": "Tweets.csv",
            "label": "Sentiment",
            "text_col": "text",
            "user_col": None,
            "date_col": None,
            "extra_cols": {"sentiment": "sentiment"},
        },
    }

    DEFAULT_CLEAN_OPTIONS = {
        "strip_urls": True,
        "strip_mentions": False,
        "keep_emojis": True,
        "keep_hashtags": True,
        "strip_rt_prefix": True,
    }

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        max_items: Optional[int] = None,
        clean_options: Optional[dict] = None,
    ):
        """
        Initialize the combined tweets loader.

        Args:
            data_dir: Directory containing tweet CSVs (default: data/tweets/)
            max_items: Total max items (split among available sources, None for all)
            clean_options: Text cleaning options
        """
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR / "tweets"
        self.max_items = max_items
        self.clean_options = {**self.DEFAULT_CLEAN_OPTIONS, **(clean_options or {})}

    @property
    def name(self) -> str:
        return "combined_tweets"

    def load(self) -> pd.DataFrame:
        """
        Load and combine all available tweet datasets.

        Returns:
            DataFrame with columns: id, title, author, text, source, tweet_type
        """
        # Find which sources are available
        available_sources = []
        for key, cfg in self.TWEET_SOURCES.items():
            path = self.data_dir / cfg["file"]
            if path.exists():
                available_sources.append((key, cfg, path))

        if not available_sources:
            raise FileNotFoundError(
                f"No tweet files found in {self.data_dir}. "
                f"Expected one of: {[c['file'] for c in self.TWEET_SOURCES.values()]}"
            )

        # Calculate per-source limit
        if self.max_items:
            per_source = self.max_items // len(available_sources)
        else:
            per_source = None

        # Load each source
        dfs = []
        for key, cfg, path in available_sources:
            df = self._load_source(path, cfg, per_source)
            df["tweet_type"] = cfg["label"]
            dfs.append(df)

        # Combine
        df = pd.concat(dfs, ignore_index=True)

        # Shuffle to mix them up
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Add standard fields
        df["source"] = "combined_tweets"
        df["id"] = [f"tweet_{i}" for i in range(len(df))]

        # Title: truncated text
        df["title"] = df["text"].apply(lambda x: (x[:80] + "...") if len(x) > 80 else x)

        return self.validate(df)

    def _load_source(self, path: Path, cfg: dict, max_items: Optional[int]) -> pd.DataFrame:
        """Load and process a single tweet source."""
        df = pd.read_csv(path, low_memory=False)

        # Sample if needed
        if max_items and len(df) > max_items:
            df = df.sample(n=max_items, random_state=42)

        # Normalize columns
        result = pd.DataFrame()

        # Text column (required)
        text_col = cfg["text_col"]
        if text_col in df.columns:
            result["text"] = df[text_col].astype(str).apply(self._clean_tweet)
        else:
            raise ValueError(f"Text column '{text_col}' not found in {path}")

        # User column (optional)
        user_col = cfg.get("user_col")
        if user_col and user_col in df.columns:
            result["author"] = df[user_col].fillna("Unknown")
        else:
            result["author"] = "Unknown"

        # Date column (optional)
        date_col = cfg.get("date_col")
        if date_col and date_col in df.columns:
            result["date"] = pd.to_datetime(df[date_col], errors="coerce")

        # Extra columns (like sentiment)
        extra_cols = cfg.get("extra_cols", {})
        for target, source in extra_cols.items():
            if source in df.columns:
                result[target] = df[source]

        # Extract hashtags from text
        result["hashtags"] = result["text"].apply(self._extract_hashtags)

        # Remove empty texts
        result = result[result["text"].str.strip().astype(bool)]

        return result

    def _clean_tweet(self, text: str) -> str:
        """Clean tweet text."""
        if pd.isna(text):
            return ""

        text = str(text)

        # Strip RT prefix
        if self.clean_options.get("strip_rt_prefix"):
            text = re.sub(r'^RT @\w+:\s*', '', text)

        # Strip URLs
        if self.clean_options.get("strip_urls"):
            text = re.sub(r'https?://\S+', '', text)

        # Strip mentions
        if self.clean_options.get("strip_mentions"):
            text = re.sub(r'@\w+', '', text)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def _extract_hashtags(text: str) -> list[str]:
        """Extract hashtags from tweet text."""
        if pd.isna(text):
            return []
        return re.findall(r'#(\w+)', str(text))

    def get_color_dimensions(self) -> dict[str, str]:
        """Available color dimensions."""
        return {
            "tweet_type": "categorical",
            "author": "categorical",
            "hashtag": "categorical",
            "sentiment": "categorical",
        }
