"""Documentation tabs content (Methodology and Architecture)."""

import streamlit as st


def render_methodology_tab() -> None:
    """Render the Methodology explanation tab."""
    st.markdown("""
## How Vector-Verse Works

### Embeddings: Turning Text into Numbers

Each piece of text is converted into a high-dimensional vector (1536 dimensions)
using OpenAI's `text-embedding-3-small` model. These vectors capture **semantic meaning** —
texts about similar topics will have similar vectors, regardless of the exact words used.

For example:
- "I love sunny days" and "Bright weather makes me happy" → similar vectors
- "I love sunny days" and "Database optimization techniques" → very different vectors

### UMAP: Visualizing High Dimensions

We use **UMAP** (Uniform Manifold Approximation and Projection) to reduce 1536 dimensions
down to 2D or 3D for visualization. UMAP is chosen because it:

- Preserves **local neighborhoods** (similar items stay close)
- Maintains **global structure** (clusters remain separated)
- Works well with high-dimensional embedding spaces

**2D vs 3D**: The 3D projection can reveal cluster separations that are hidden in 2D.
Some clusters that overlap in 2D may be clearly separated in the third dimension!

### Similarity Search

When you search or find neighbors, we compute **cosine similarity** between vectors.
For normalized vectors (which ours are), this simplifies to a dot product:

```
similarity = embedding_a · embedding_b
```

- **1.0** = Identical meaning
- **0.7-0.9** = Very similar
- **0.5-0.7** = Related
- **< 0.5** = Probably unrelated

### Hierarchical Zoom

When you zoom into a cluster:

1. We take the subset of embeddings you selected
2. Re-run UMAP on just that subset
3. This reveals **finer structure** that's hidden at the global scale

Think of it like zooming in on a map — you see more detail at street level than from space.

### What Clusters Mean (and Don't Mean)

**Clusters suggest thematic groupings**, but they're not definitive categories:

- Items at cluster edges may belong to multiple conceptual groups
- The 2D projection is a lossy compression — some relationships are lost
- Different UMAP runs can produce different layouts (but preserve neighborhoods)
- Cluster size doesn't necessarily indicate importance

**Use clusters as exploration aids**, not ground truth categories.
""")


def render_architecture_tab() -> None:
    """Render the Architecture documentation tab."""
    st.markdown("""
## System Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Sources                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Poetry  │  │  Tweets  │  │  Lyrics  │  │  Custom  │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Loader Plugins                              │
│  BaseDatasetLoader → (PoetryLoader, TweetsLoader, LyricsLoader) │
│  • Normalize columns (id, title, author, text, source)          │
│  • Clean text (configurable per loader)                         │
│  • Extract metadata (hashtags, genre, etc.)                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Embedding Layer                             │
│  BaseEmbedder → OpenAIEmbedder (text-embedding-3-small)         │
│  • Batch processing with checkpoints                            │
│  • L2 normalization for cosine similarity                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Cache Layer                                 │
│  CacheManager: cache/{dataset}_{embedder}/                      │
│  • embeddings.npz      (vectors)                                │
│  • items.parquet       (metadata)                               │
│  • umap_coords.npz     (2D coords)                              │
│  • umap_coords_3d.npz  (3D coords)                              │
│  • umap_model.pkl      (fitted 2D UMAP)                         │
│  • umap_model_3d.pkl   (fitted 3D UMAP)                         │
│  • zoom_levels/        (subset projections)                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VectorStore                                 │
│  • Orchestrates load → embed → project                          │
│  • Similarity search (scoped to zoom level)                     │
│  • Neighbor lookup                                              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Visualization                               │
│  ScatterPlotBuilder + ZoomManager                               │
│  • Plotly scatter with lasso selection                          │
│  • Color by metadata dimensions                                 │
│  • Hierarchical zoom (re-UMAP on subset)                        │
└─────────────────────────────────────────────────────────────────┘
```

### Adding a New Dataset Loader

1. **Create loader file**: `vector_verse/loaders/your_loader.py`

2. **Extend BaseDatasetLoader**:
```python
from .base import BaseDatasetLoader, register_loader

@register_loader("your_dataset")
class YourLoader(BaseDatasetLoader):
    @property
    def name(self) -> str:
        return "your_dataset"

    def load(self) -> pd.DataFrame:
        # Return DataFrame with columns:
        # id, title, author, text, source
        # Plus any metadata columns
        return self.validate(df)
```

3. **Add to config.py**:
```python
AVAILABLE_DATASETS["your_dataset"] = {
    "loader": "your_dataset",
    "label": "Your Dataset",
    "data_check": lambda: (DATA_DIR / "your_data").exists(),
    "color_dimensions": ["category", "author"],
}
```

4. **Export in `__init__.py`**:
```python
from .your_loader import YourLoader
```

### Caching Strategy

**Cache isolation**: Each dataset+embedder gets its own folder:
```
cache/
├── poetry_foundation_openai_text-embedding-3-small/
├── tweets_openai_text-embedding-3-small/
└── lyrics_openai_text-embedding-3-small/
```

**Zoom caching**: Subset projections are cached by content hash:
```
zoom_levels/
└── {parent_id}_{indices_hash}.npz
```

### Key Classes

| Class | Responsibility |
|-------|---------------|
| `VectorStore` | Central orchestrator: load, embed, search |
| `BaseDatasetLoader` | Abstract loader interface |
| `BaseEmbedder` | Abstract embedding interface |
| `CacheManager` | Persistence layer |
| `UMAPProjector` | Dimensionality reduction |
| `ZoomManager` | Hierarchical exploration state |
| `ScatterPlotBuilder` | Plotly visualization |
""")
