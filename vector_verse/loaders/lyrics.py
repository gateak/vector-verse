"""
Song lyrics dataset loader.
Loads lyrics CSV datasets from Kaggle and other sources.

Supported Kaggle fixtures:
- neisse/scrapped-lyrics-from-6-genres (lyrics-data.csv + artists-data.csv)
- 380000 lyrics from MetroLyrics pattern (song, artist, year, genre, lyrics)
"""

from pathlib import Path
from typing import Optional
import re

import pandas as pd

from .base import BaseDatasetLoader, register_loader
import config


@register_loader("lyrics")
class LyricsCsvLoader(BaseDatasetLoader):
    """
    Loader for song lyrics CSV datasets.
    
    Supports:
    - Single CSV with all metadata (MetroLyrics format)
    - Multi-file datasets (artists-data.csv + lyrics-data.csv)
    
    Auto-detects CSVs in data/lyrics/ or reads dataset.yaml manifest.
    """
    
    # Column mapping presets for known datasets
    COLUMN_PRESETS = {
        # neisse/scrapped-lyrics-from-6-genres
        "scrapped_lyrics": {
            "text": ["Lyric", "lyric", "lyrics", "Lyrics"],
            "title": ["SName", "song_name", "song", "Song", "title", "Title"],
            "artist_id": ["ALink", "artist_link", "artist_id"],
            "language": ["language", "Language"],
        },
        # MetroLyrics / common format
        "metrolyrics": {
            "text": ["lyrics", "Lyrics", "lyric", "Lyric", "text"],
            "title": ["song", "Song", "title", "Title", "track", "Track"],
            "artist": ["artist", "Artist", "band", "Band"],
            "genre": ["genre", "Genre"],
            "year": ["year", "Year", "release_year"],
            "album": ["album", "Album"],
        },
        # Generic fallback
        "generic": {
            "text": ["lyrics", "lyric", "text", "content", "words"],
            "title": ["title", "song", "track", "name"],
            "artist": ["artist", "band", "singer", "performer"],
            "genre": ["genre", "style", "category"],
            "year": ["year", "release_year", "date"],
            "album": ["album", "record"],
        },
    }
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        csv_file: Optional[str] = None,
        column_map: Optional[dict] = None,
        max_items: Optional[int] = None,
        strip_section_markers: bool = True,
        min_lyrics_length: int = 50,
    ):
        """
        Initialize the lyrics loader.
        
        Args:
            data_dir: Directory containing lyrics CSVs (default: data/lyrics/)
            csv_file: Specific CSV file to load (auto-detected if not provided)
            column_map: Custom column mapping override
            max_items: Maximum number of items to load
            strip_section_markers: Remove [Chorus], [Verse], etc.
            min_lyrics_length: Minimum character length for lyrics
        """
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR / "lyrics"
        self.csv_file = csv_file
        self.column_map = column_map
        self.max_items = max_items
        self.strip_section_markers = strip_section_markers
        self.min_lyrics_length = min_lyrics_length
    
    @property
    def name(self) -> str:
        return "lyrics"
    
    def load(self) -> pd.DataFrame:
        """
        Load and normalize lyrics dataset.
        
        Returns:
            DataFrame with columns: id, title, author, text, source, plus metadata
        """
        # 1. Determine which format we're dealing with
        if self._is_multi_file_format():
            df = self._load_multi_file_format()
        else:
            df = self._load_single_file_format()
        
        # 2. Clean lyrics text
        if self.strip_section_markers:
            df["text"] = df["text"].apply(self._strip_markers)
        
        # 3. Clean whitespace
        df["text"] = df["text"].apply(self._clean_whitespace)
        
        # 4. Filter by minimum length
        df = df[df["text"].str.len() >= self.min_lyrics_length]
        
        # 5. Add standard fields
        df["source"] = "lyrics"
        df["id"] = [f"lyrics_{i}" for i in range(len(df))]
        
        # Author: map artist to author for base class compatibility
        if "artist" not in df.columns:
            df["artist"] = "Unknown"
        df["author"] = df["artist"].fillna("Unknown")
        
        # Ensure title exists
        if "title" not in df.columns:
            df["title"] = df["text"].apply(lambda x: (x[:50] + "...") if len(x) > 50 else x)
        df["title"] = df["title"].fillna("Untitled")
        
        # 6. Derive decade from year if available
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df["decade"] = (df["year"] // 10 * 10).astype("Int64").astype(str) + "s"
            df.loc[df["year"].isna(), "decade"] = None
        
        # 7. Limit items if specified
        if self.max_items:
            df = df.head(self.max_items)
        
        # 8. Validate and return
        return self.validate(df)
    
    def _is_multi_file_format(self) -> bool:
        """Check if this is a multi-file dataset (lyrics-data.csv + artists-data.csv)."""
        return (self.data_dir / "lyrics-data.csv").exists()
    
    def _load_multi_file_format(self) -> pd.DataFrame:
        """
        Load the 'scrapped-lyrics-from-6-genres' multi-file format.
        
        Files:
        - lyrics-data.csv: ALink, SName, SLink, Lyric, language
        - artists-data.csv: Artist, Genres, Songs, Popularity, Link
        """
        lyrics_path = self.data_dir / "lyrics-data.csv"
        artists_path = self.data_dir / "artists-data.csv"
        
        # Load lyrics
        df = pd.read_csv(lyrics_path, low_memory=False)
        
        # Normalize columns
        column_map = self._auto_detect_columns(df.columns.tolist())
        df = self._normalize_columns(df, column_map)
        
        # Load artists for metadata if available
        if artists_path.exists():
            artists_df = pd.read_csv(artists_path, low_memory=False)
            
            # Detect link column
            link_col = None
            for col in ["Link", "link", "ALink", "artist_link"]:
                if col in artists_df.columns:
                    link_col = col
                    break
            
            if link_col and "artist_id" in df.columns:
                # Create artist lookup
                artist_col = None
                for col in ["Artist", "artist", "Name", "name"]:
                    if col in artists_df.columns:
                        artist_col = col
                        break
                
                genre_col = None
                for col in ["Genres", "genres", "Genre", "genre"]:
                    if col in artists_df.columns:
                        genre_col = col
                        break
                
                if artist_col:
                    artist_lookup = dict(zip(artists_df[link_col], artists_df[artist_col]))
                    df["artist"] = df["artist_id"].map(artist_lookup)
                
                if genre_col:
                    genre_lookup = dict(zip(artists_df[link_col], artists_df[genre_col]))
                    df["genre"] = df["artist_id"].map(genre_lookup)
                    # Take first genre if comma-separated
                    df["genre"] = df["genre"].apply(
                        lambda x: x.split(",")[0].strip() if pd.notna(x) and isinstance(x, str) else x
                    )
        
        return df
    
    def _load_single_file_format(self) -> pd.DataFrame:
        """
        Load a single CSV file with all metadata.
        
        Supports MetroLyrics format and similar.
        """
        csv_path = self._find_csv()
        
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Normalize columns
        column_map = self.column_map or self._auto_detect_columns(df.columns.tolist())
        df = self._normalize_columns(df, column_map)
        
        return df
    
    def _find_csv(self) -> Path:
        """Find the lyrics CSV file to load."""
        # Specific file
        if self.csv_file:
            return self.data_dir / self.csv_file
        
        # Check manifest
        manifest_path = self.data_dir / "dataset.yaml"
        if manifest_path.exists():
            try:
                import yaml
                with open(manifest_path) as f:
                    manifest = yaml.safe_load(f)
                return self.data_dir / manifest.get("file", "")
            except ImportError:
                pass
        
        # Auto-detect
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Lyrics data directory not found: {self.data_dir}\n"
                f"Create the directory and add a lyrics CSV file."
            )
        
        # Skip multi-file format files
        csvs = [
            f for f in self.data_dir.glob("*.csv")
            if f.name not in ["artists-data.csv", "lyrics-data.csv"]
        ]
        
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}\n"
                f"Download a lyrics dataset from Kaggle and place it here.\n"
                f"Supported datasets:\n"
                f"  - https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres"
            )
        
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
        normalized = pd.DataFrame()
        
        # Store source metadata
        normalized["source_meta"] = df.apply(lambda row: row.to_dict(), axis=1)
        
        # Text (required) - lyrics content
        text_col = column_map.get("text")
        if text_col and text_col in df.columns:
            normalized["text"] = df[text_col].astype(str)
        else:
            # Try to find any lyrics-like column
            for col in df.columns:
                if col.lower() in ["lyrics", "lyric", "text", "content"]:
                    normalized["text"] = df[col].astype(str)
                    break
            else:
                raise ValueError(
                    f"Could not find lyrics column in CSV. "
                    f"Available columns: {list(df.columns)}"
                )
        
        # Map other columns
        for target in ["title", "artist", "artist_id", "album", "genre", "year", "language"]:
            source_col = column_map.get(target)
            if source_col and source_col in df.columns:
                normalized[target] = df[source_col]
        
        return normalized
    
    @staticmethod
    def _strip_markers(text: str) -> str:
        """
        Remove section markers like [Chorus], [Verse 1], etc.
        
        Args:
            text: Raw lyrics text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove bracketed markers
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove parenthesized markers like (Chorus), (x2)
        text = re.sub(r'\((?:Chorus|Verse|Bridge|Intro|Outro|Hook|x\d+)\)', '', text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """
        Clean up excessive whitespace while preserving line breaks.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines (more than 2 in a row)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove leading/trailing whitespace from whole text
        return text.strip()
    
    def get_color_dimensions(self) -> dict[str, str]:
        """
        Get available color dimensions for this dataset.
        
        Returns:
            Dict of dimension name -> type (categorical/sequential)
        """
        return {
            "genre": "categorical",
            "artist": "categorical",
            "decade": "sequential",
            "language": "categorical",
        }
