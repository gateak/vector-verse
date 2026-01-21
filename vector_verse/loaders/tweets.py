"""
Tweets dataset loader.
Loads tweet CSV datasets from Kaggle and other sources.

Supported Kaggle fixtures:
- khalidryder777/500k-chatgpt-tweets-jan-mar-2023
- smmmmmmmmmmmm/twitter-data
"""

from pathlib import Path
from typing import Optional
import re
import json

import pandas as pd

from .base import BaseDatasetLoader, register_loader
import config


@register_loader("tweets")
class TweetsCsvLoader(BaseDatasetLoader):
    """
    Loader for tweet CSV datasets (Kaggle-compatible).
    
    Auto-detects CSVs in data/tweets/ or reads dataset.yaml manifest.
    Normalizes various tweet CSV formats to a standard schema.
    """
    
    # Column mapping presets for known Kaggle datasets
    COLUMN_PRESETS = {
        # khalidryder777/500k-chatgpt-tweets-jan-mar-2023
        "chatgpt_tweets": {
            "text": ["content", "Content"],
            "user": ["username", "Username", "user"],
            "date": ["date", "Date", "created_at"],
            "likes": ["like_count", "likes", "Likes"],
            "retweets": ["retweet_count", "retweets", "Retweets"],
            "id_col": ["id", "tweet_id", "ID"],
        },
        # smmmmmmmmmmmm/twitter-data
        "twitter_data": {
            "text": ["tweet", "Tweet", "text", "Text", "content"],
            "user": ["user", "User", "username", "screen_name"],
            "date": ["timestamp", "Timestamp", "date", "created_at"],
            "likes": ["likes", "Likes", "favorite_count", "like_count"],
            "retweets": ["retweets", "Retweets", "retweet_count"],
            "hashtags": ["hashtags", "Hashtags"],
            "mentions": ["mentions", "Mentions"],
            "location": ["location", "Location"],
        },
        # vikasg/russian-troll-tweets (FiveThirtyEight IRA dataset)
        "russian_troll_tweets": {
            "text": ["content", "Content", "text"],
            "user": ["author", "Author", "user", "handle"],
            "date": ["publish_date", "created_at", "date"],
            "followers": ["followers", "following"],
            "updates": ["updates"],
            "region": ["region", "Region"],
            "language": ["language", "Language"],
            "label": ["account_category", "account_type", "category"],
        },
        # Generic fallback
        "generic": {
            "text": ["text", "tweet", "content", "message", "body"],
            "user": ["user", "username", "author", "screen_name", "handle"],
            "date": ["date", "created_at", "timestamp", "time", "datetime", "publish_date"],
            "likes": ["likes", "like_count", "favorites", "favorite_count"],
            "retweets": ["retweets", "retweet_count", "rt_count"],
            "hashtags": ["hashtags", "tags"],
        },
    }
    
    # Text cleaning options
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
        csv_file: Optional[str] = None,
        column_map: Optional[dict] = None,
        max_items: Optional[int] = None,
        clean_options: Optional[dict] = None,
    ):
        """
        Initialize the tweets loader.
        
        Args:
            data_dir: Directory containing tweet CSVs (default: data/tweets/)
            csv_file: Specific CSV file to load (auto-detected if not provided)
            column_map: Custom column mapping override
            max_items: Maximum number of items to load
            clean_options: Text cleaning options
        """
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR / "tweets"
        self.csv_file = csv_file
        self.column_map = column_map
        self.max_items = max_items
        self.clean_options = {**self.DEFAULT_CLEAN_OPTIONS, **(clean_options or {})}
    
    @property
    def name(self) -> str:
        return "tweets"
    
    def load(self) -> pd.DataFrame:
        """
        Load and normalize tweet dataset.
        
        Returns:
            DataFrame with columns: id, title, author, text, source, plus metadata
        """
        # 1. Find CSV and column mapping
        csv_path, column_map = self._resolve_csv_and_mapping()
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Tweet CSV not found at {csv_path}\n"
                f"Download a tweet dataset from Kaggle and place it in: {self.data_dir}\n"
                f"Supported datasets:\n"
                f"  - https://www.kaggle.com/datasets/khalidryder777/500k-chatgpt-tweets-jan-mar-2023\n"
                f"  - https://www.kaggle.com/datasets/smmmmmmmmmmmm/twitter-data\n"
                f"  - https://www.kaggle.com/datasets/vikasg/russian-troll-tweets"
            )
        
        # 2. Load CSV
        df = pd.read_csv(csv_path, low_memory=False)
        
        # 3. Normalize columns
        df = self._normalize_columns(df, column_map)
        
        # 4. Clean text
        df["text"] = df["text"].apply(self._clean_tweet)
        
        # 5. Remove empty texts
        df = df[df["text"].str.strip().astype(bool)]
        
        # 6. Parse hashtags if present
        if "hashtags_raw" in df.columns:
            df["hashtags"] = df["hashtags_raw"].apply(self._parse_hashtags)
        elif "text" in df.columns:
            # Extract hashtags from text
            df["hashtags"] = df["text"].apply(self._extract_hashtags)
        
        # 7. Add standard fields
        df["source"] = "tweets"
        df["id"] = [f"tweet_{i}" for i in range(len(df))]
        
        # Title: truncated text
        df["title"] = df["text"].apply(lambda x: (x[:80] + "...") if len(x) > 80 else x)
        
        # Author: map user to author for base class compatibility
        df["author"] = df.get("user", pd.Series(["Unknown"] * len(df)))
        df["author"] = df["author"].fillna("Unknown")
        
        # 8. Limit items if specified
        if self.max_items:
            df = df.head(self.max_items)
        
        # 9. Validate and return
        return self.validate(df)
    
    def _resolve_csv_and_mapping(self) -> tuple[Path, dict]:
        """
        Find CSV file and column mapping.
        
        Checks:
        1. Specific csv_file if provided
        2. dataset.yaml manifest
        3. Auto-detect first CSV in directory
        
        Returns:
            Tuple of (csv_path, column_map)
        """
        # If specific file provided
        if self.csv_file:
            csv_path = self.data_dir / self.csv_file
            return csv_path, self.column_map or {}
        
        # Check for manifest
        manifest_path = self.data_dir / "dataset.yaml"
        if manifest_path.exists():
            try:
                import yaml
                with open(manifest_path) as f:
                    manifest = yaml.safe_load(f)
                csv_path = self.data_dir / manifest.get("file", "")
                column_map = manifest.get("column_map", {})
                return csv_path, column_map
            except ImportError:
                pass  # yaml not available, continue to auto-detect
        
        # Auto-detect: find first CSV
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Tweets data directory not found: {self.data_dir}\n"
                f"Create the directory and add a tweet CSV file."
            )
        
        csvs = list(self.data_dir.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}\n"
                f"Download a tweet dataset from Kaggle and place it here."
            )
        
        # Prioritize CSVs that likely contain tweets (not users/metadata)
        csv_path = self._find_best_tweet_csv(csvs)
        
        # Auto-detect column mapping from file
        df_sample = pd.read_csv(csv_path, nrows=5)
        column_map = self._auto_detect_columns(df_sample.columns.tolist())
        
        return csv_path, column_map
    
    def _find_best_tweet_csv(self, csvs: list[Path]) -> Path:
        """
        Find the CSV most likely to contain tweets.
        
        Prioritizes:
        1. Files with 'tweet' in the name
        2. Files with a text-like column
        3. Excludes files that look like user/metadata files
        
        Args:
            csvs: List of CSV paths
            
        Returns:
            Best candidate CSV path
        """
        # Priority 1: Files with 'tweet' in the name
        for csv in csvs:
            name_lower = csv.stem.lower()
            if 'tweet' in name_lower:
                return csv
        
        # Priority 2: Check each file for text-like columns, skip user/metadata files
        text_columns = {"text", "tweet", "content", "message", "body"}
        user_indicators = {"followers_count", "friends_count", "statuses_count", "screen_name"}
        
        for csv in csvs:
            try:
                df_sample = pd.read_csv(csv, nrows=1)
                cols_lower = {c.lower() for c in df_sample.columns}
                
                # Skip if this looks like a users file
                if len(user_indicators & cols_lower) >= 2:
                    continue
                
                # Check for text-like column
                if text_columns & cols_lower:
                    return csv
            except Exception:
                continue
        
        # Fallback: return first non-user CSV, or just the first CSV
        for csv in csvs:
            name_lower = csv.stem.lower()
            if 'user' not in name_lower:
                return csv
        
        return csvs[0]
    
    def _auto_detect_columns(self, columns: list[str]) -> dict:
        """
        Auto-detect column mapping from available columns.
        
        Args:
            columns: List of column names in the CSV
            
        Returns:
            Mapping from normalized names to actual column names
        """
        mapping = {}
        columns_lower = {c.lower(): c for c in columns}
        
        # Try each preset
        for preset_name, preset in self.COLUMN_PRESETS.items():
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
    
    def _normalize_columns(self, df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
        """
        Normalize column names to standard schema.
        
        Args:
            df: Raw DataFrame
            column_map: Mapping from target names to source column names
            
        Returns:
            DataFrame with normalized columns
        """
        # Auto-detect if no mapping provided
        if not column_map:
            column_map = self._auto_detect_columns(df.columns.tolist())
        
        # Store source metadata
        df["source_meta"] = df.apply(lambda row: row.to_dict(), axis=1)
        
        # Map columns
        normalized = pd.DataFrame()
        
        # Text (required)
        text_col = column_map.get("text")
        if text_col and text_col in df.columns:
            normalized["text"] = df[text_col].astype(str)
        else:
            # Try to find any text-like column
            for col in df.columns:
                if col.lower() in ["text", "tweet", "content", "message"]:
                    normalized["text"] = df[col].astype(str)
                    break
            else:
                raise ValueError(
                    f"Could not find text column in CSV. "
                    f"Available columns: {list(df.columns)}"
                )
        
        # Optional columns
        for target, default in [
            ("user", None),
            ("date", None),
            ("likes", None),
            ("retweets", None),
            ("hashtags_raw", None),
            ("location", None),
            ("label", None),
            ("region", None),
            ("language", None),
            ("followers", None),
        ]:
            source_col = column_map.get(target) or column_map.get(target.replace("_raw", ""))
            if source_col and source_col in df.columns:
                normalized[target] = df[source_col]
            elif target == "hashtags_raw" and "hashtags" in column_map and column_map["hashtags"] in df.columns:
                normalized["hashtags_raw"] = df[column_map["hashtags"]]
        
        # Convert numeric columns
        for col in ["likes", "retweets"]:
            if col in normalized.columns:
                normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        
        # Parse dates
        if "date" in normalized.columns:
            normalized["created_at"] = pd.to_datetime(normalized["date"], errors="coerce")
        
        # Keep source metadata
        normalized["source_meta"] = df["source_meta"]
        
        return normalized
    
    def _clean_tweet(self, text: str) -> str:
        """
        Clean tweet text based on options.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Strip RT prefix
        if self.clean_options.get("strip_rt_prefix"):
            text = re.sub(r'^RT @\w+:\s*', '', text)
        
        # Strip URLs
        if self.clean_options.get("strip_urls"):
            text = re.sub(r'https?://\S+', '', text)
            text = re.sub(r'http?://\S+', '', text)
        
        # Strip mentions
        if self.clean_options.get("strip_mentions"):
            text = re.sub(r'@\w+', '', text)
        
        # Strip hashtags (but keep the word)
        if not self.clean_options.get("keep_hashtags"):
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def _parse_hashtags(raw: str) -> list[str]:
        """
        Parse hashtags from various formats.
        
        Handles:
        - JSON arrays: ["tag1", "tag2"]
        - Comma-separated: tag1, tag2
        - Space-separated: #tag1 #tag2
        
        Args:
            raw: Raw hashtag string
            
        Returns:
            List of hashtag strings (without #)
        """
        if pd.isna(raw) or not raw:
            return []
        
        raw = str(raw).strip()
        
        # Try JSON array
        if raw.startswith("["):
            try:
                parsed = json.loads(raw.replace("'", '"'))
                if isinstance(parsed, list):
                    return [str(h).lstrip("#") for h in parsed]
            except json.JSONDecodeError:
                pass
        
        # Try comma-separated
        if "," in raw:
            return [h.strip().lstrip("#") for h in raw.split(",") if h.strip()]
        
        # Space-separated or single
        return [h.strip().lstrip("#") for h in re.split(r'\s+', raw) if h.strip()]
    
    @staticmethod
    def _extract_hashtags(text: str) -> list[str]:
        """
        Extract hashtags from tweet text.
        
        Args:
            text: Tweet text
            
        Returns:
            List of hashtag strings (without #)
        """
        if pd.isna(text):
            return []
        return re.findall(r'#(\w+)', str(text))
    
    def get_color_dimensions(self) -> dict[str, str]:
        """
        Get available color dimensions for this dataset.
        
        Returns:
            Dict of dimension name -> type (categorical/sequential)
        """
        return {
            "user": "categorical",
            "hashtag": "categorical",
            "date_bucket": "sequential",
            "label": "categorical",
        }
