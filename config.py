"""
Vector-Verse Configuration
Central configuration for paths, defaults, and settings.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"

# Default dataset paths
POETRY_CSV_PATH = DATA_DIR / "PoetryFoundationData.csv"
CUSTOM_ITEMS_PATH = DATA_DIR / "custom_items.csv"
TWEETS_DATA_DIR = DATA_DIR / "tweets"
LYRICS_DATA_DIR = DATA_DIR / "lyrics"

# Embedding settings
DEFAULT_EMBEDDER = "openai"
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536
OPENAI_BATCH_SIZE = 2000  # Max texts per API call (actual batch size adapts to token limits)

# UMAP settings
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
UMAP_N_COMPONENTS = 2
UMAP_RANDOM_STATE = 42

# Search settings
DEFAULT_K_NEIGHBORS = 10

# Visualization settings
PLOT_HEIGHT = 600
PLOT_WIDTH = 800

# Source identifiers for different item types
SOURCE_POETRY = "poetry"
SOURCE_CUSTOM = "custom"
SOURCE_TWEETS = "tweets"
SOURCE_LYRICS = "lyrics"
SOURCE_QUERY = "query"

# Dataset registry
AVAILABLE_DATASETS = {
    "poetry": {
        "loader": "poetry",
        "label": "📜 Poetry",
        "description": "Poetry Foundation poems (~14k)",
        "data_check": lambda: POETRY_CSV_PATH.exists(),
        "color_dimensions": ["author", "source"],
    },
    "tweets": {
        "loader": "tweets",
        "label": "🐦 Tweets",
        "description": "Twitter/X posts from Kaggle datasets",
        "data_check": lambda: TWEETS_DATA_DIR.exists() and any(TWEETS_DATA_DIR.glob("*.csv")),
        "color_dimensions": ["user", "hashtag", "date_bucket", "label", "region", "language"],
    },
    "combined_tweets": {
        "loader": "combined_tweets",
        "label": "🔍 Tweet Sources",
        "description": "Compare tweets from multiple sources (bots, sentiment, etc.)",
        "data_check": lambda: TWEETS_DATA_DIR.exists() and len(list(TWEETS_DATA_DIR.glob("*.csv"))) >= 2,
        "color_dimensions": ["tweet_type", "sentiment", "author", "hashtag"],
    },
    "lyrics": {
        "loader": "lyrics",
        "label": "🎵 Lyrics",
        "description": "Song lyrics from various genres",
        "data_check": lambda: LYRICS_DATA_DIR.exists() and (
            any(LYRICS_DATA_DIR.glob("*.csv")) or 
            (LYRICS_DATA_DIR / "lyrics-data.csv").exists()
        ),
        "color_dimensions": ["genre", "artist", "decade", "language"],
    },
}

DEFAULT_DATASET = "poetry"
