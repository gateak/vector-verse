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

# Embedding settings
DEFAULT_EMBEDDER = "openai"
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536
OPENAI_BATCH_SIZE = 100  # Texts per API call

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
SOURCE_QUERY = "query"
