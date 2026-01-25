# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Vector-Verse is a Streamlit web application for exploring text datasets through semantic similarity. It embeds text using OpenAI's embedding models and visualizes them as interactive 2D/3D scatter plots using UMAP.

## Commands

```bash
# Run the application
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Setup environment (copy and edit with your API key)
cp .env.example .env
```

**Requirements**: Python 3.10+, OpenAI API key in `.env`

## Architecture

### Plugin-Based Design

The codebase uses a registry pattern for extensibility:

**Dataset Loaders** (`vector_verse/loaders/`):
- Inherit from `BaseDatasetLoader` and use `@register_loader("name")` decorator
- Must implement `load() -> pd.DataFrame` and `name` property
- DataFrame must have columns: `id`, `title`, `author`, `text`, `source`
- Additional metadata columns (genre, hashtags, etc.) are preserved for color dimensions

**Embedders** (`vector_verse/embedders/`):
- Inherit from `BaseEmbedder` and use `@register_embedder("name")` decorator
- Must return L2-normalized vectors for dot-product similarity search

### Core Components

```
VectorStore (core/vector_store.py)
├── Orchestrates: load → embed → project → search
├── Manages cache key as {dataset_name}_{embedder_name}
└── Provides similarity search scoped to zoom level

UMAPProjector (core/projector.py)
├── Fits UMAP on corpus embeddings
├── Projects new queries onto existing space
└── Configurable parameters (n_neighbors, min_dist, metric)

ZoomManager (core/zoom_manager.py)
├── Hierarchical exploration: re-runs UMAP on selected subsets
└── MIN_ITEMS_FOR_ZOOM = 3 (validates selection size)

CacheManager (vector_verse/cache/manager.py)
└── Persists: embeddings.npz, items.parquet, umap_coords.npz, umap_model.pkl

ScatterPlotBuilder (visualization/scatter.py)
├── Builds interactive Plotly 2D/3D scatter plots
├── Handles color dimension mapping and symbol assignment
└── SYMBOL_3D_MAP maps 2D symbols to 3D-compatible versions
```

### UI Components (`vector_verse/ui/`)

The Streamlit UI is modularized into focused components:

```
ui/
├── __init__.py      # Package exports
├── state.py         # AppState: centralized session state management
├── styles.py        # Theme dataclass + CSS injection with vv-* classes
├── sidebar.py       # Dataset switcher, color selector, view toggle, item browse
├── main_view.py     # Search bar, visualization, zoom controls
├── details.py       # Item cards, neighbors list, search results
└── docs.py          # Methodology and Architecture tab content
```

**Key classes:**
- `AppState` - Static methods for session state (init, reset_for_dataset_change, set_search_results, etc.)
- `Theme` - Frozen dataclass with color constants

### Data Flow

1. Loader produces DataFrame with required columns
2. Embedder converts text → normalized 1536-dim vectors
3. UMAPProjector reduces to 2D/3D coordinates
4. ScatterPlotBuilder renders Plotly visualization
5. ZoomManager handles subset re-projection

### Configuration

`config.py` defines:
- `AVAILABLE_DATASETS`: Registry of datasets with data_check lambdas and color_dimensions
- UMAP parameters (n_neighbors=15, min_dist=0.1, metric="cosine")
- Source identifiers: `SOURCE_POETRY`, `SOURCE_TWEETS`, `SOURCE_LYRICS`, `SOURCE_CUSTOM`

### Adding a New Dataset

1. Create `vector_verse/loaders/your_loader.py`:
```python
from .base import BaseDatasetLoader, register_loader

@register_loader("your_dataset")
class YourLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "your_dataset"

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(...)
        return self.validate(df)
```

2. Export in `vector_verse/loaders/__init__.py`

3. Add to `AVAILABLE_DATASETS` in `config.py`

### Cache Structure

```
cache/{dataset}_{embedder}/
├── embeddings.npz      # Vector embeddings
├── items.parquet       # Item metadata
├── umap_coords.npz     # 2D projection
├── umap_coords_3d.npz  # 3D projection (optional)
├── umap_model.pkl      # Fitted UMAP model
└── zoom_levels/        # Cached subset projections
```

Caches are isolated per dataset+embedder combination. Use sidebar "Rebuild Cache" to force re-embedding.
