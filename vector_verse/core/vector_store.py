"""
VectorStore: Central orchestrator for Vector-Verse.
Manages data loading, embedding, caching, and similarity search.
"""

from typing import Optional

import numpy as np
import pandas as pd

from vector_verse.loaders.base import BaseDatasetLoader
from vector_verse.loaders.poetry import PoetryLoader
from vector_verse.loaders.custom import CustomItemsLoader
from vector_verse.embedders.base import BaseEmbedder
from vector_verse.embedders.openai_embedder import OpenAIEmbedder
from vector_verse.cache.manager import CacheManager
from vector_verse.core.projector import UMAPProjector
import config


class VectorStore:
    """
    Central orchestrator for embedding-based similarity search.
    
    Responsibilities:
    - Load datasets via loaders
    - Embed texts via embedder
    - Manage caching of embeddings and UMAP
    - Provide similarity search and neighbor lookup
    """
    
    def __init__(
        self,
        dataset_loader: Optional[BaseDatasetLoader] = None,
        custom_loader: Optional[BaseDatasetLoader] = None,
        embedder: Optional[BaseEmbedder] = None,
        force_rebuild: bool = False
    ):
        """
        Initialize the VectorStore.
        
        Args:
            dataset_loader: Primary dataset loader (defaults to PoetryLoader)
            custom_loader: Custom items loader (defaults to CustomItemsLoader)
            embedder: Embedding backend (defaults to OpenAIEmbedder)
            force_rebuild: Whether to rebuild cache even if exists
        """
        self.dataset_loader = dataset_loader or PoetryLoader()
        self.custom_loader = custom_loader or CustomItemsLoader()
        self.embedder = embedder or OpenAIEmbedder()
        self.force_rebuild = force_rebuild
        
        # Will be populated after initialization
        self.items_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.umap_coords: Optional[np.ndarray] = None
        self.projector: Optional[UMAPProjector] = None
        
        # Cache manager (set during initialize)
        self._cache: Optional[CacheManager] = None
        
        # ID to index mapping for fast lookup
        self._id_to_idx: dict[str, int] = {}
    
    @property
    def cache_key(self) -> str:
        """Generate unique cache key for this dataset+embedder combination."""
        return f"{self.dataset_loader.name}_{self.embedder.name}"
    
    def initialize(self, progress_callback=None) -> None:
        """
        Initialize the vector store: load data, embed, and compute UMAP.
        
        Args:
            progress_callback: Optional callable(message: str) for progress updates
        
        This method:
        1. Checks for existing cache
        2. Loads or computes embeddings
        3. Fits or loads UMAP model
        4. Merges custom items (always recomputed)
        """
        def log(msg: str):
            if progress_callback:
                progress_callback(msg)
            print(msg)
        
        # Initialize cache manager
        self._cache = CacheManager(self.cache_key)
        
        # Check if we should use cache
        use_cache = not self.force_rebuild and self._cache.exists()
        
        if use_cache:
            log("Loading from cache...")
            self._load_from_cache()
        else:
            log("Building embeddings (this may take a few minutes)...")
            self._build_from_scratch(log)
        
        # Always reload and merge custom items
        self._merge_custom_items(log)
        
        # Build ID index
        self._build_id_index()
        
        log(f"Ready! {len(self.items_df)} items loaded.")
    
    def _load_from_cache(self) -> None:
        """Load embeddings and UMAP from cache."""
        # Load embeddings
        self.embeddings, cached_ids = self._cache.load_embeddings()
        
        # Load items
        self.items_df = self._cache.load_items()
        
        # Load UMAP
        self.projector = UMAPProjector()
        if self._cache.has_umap():
            self.umap_coords = self._cache.load_umap_coords()
            self.projector.set_model(self._cache.load_umap_model())
    
    def _build_from_scratch(self, log) -> None:
        """Compute embeddings and UMAP from scratch."""
        # Load primary dataset
        log(f"Loading {self.dataset_loader.name}...")
        self.items_df = self.dataset_loader.load()
        log(f"Loaded {len(self.items_df)} items")
        
        # Generate embeddings
        texts = self.items_df["text"].tolist()
        log(f"Generating embeddings for {len(texts)} texts...")
        self.embeddings = self.embedder.embed(texts)
        log(f"Embeddings shape: {self.embeddings.shape}")
        
        # Fit UMAP
        log("Fitting UMAP projection...")
        self.projector = UMAPProjector()
        self.umap_coords = self.projector.fit(self.embeddings)
        
        # Save to cache
        log("Saving to cache...")
        self._cache.save_embeddings(self.embeddings, self.items_df["id"].tolist())
        self._cache.save_items(self.items_df)
        self._cache.save_umap(self.umap_coords, self.projector.get_model())
    
    def _merge_custom_items(self, log) -> None:
        """Merge custom items into the dataset."""
        custom_df = self.custom_loader.load()
        
        if custom_df.empty:
            return
        
        log(f"Adding {len(custom_df)} custom items...")
        
        # Embed custom items
        custom_texts = custom_df["text"].tolist()
        custom_embeddings = self.embedder.embed(custom_texts, show_progress=False)
        
        # Project to UMAP space
        custom_coords = self.projector.transform(custom_embeddings)
        
        # Merge DataFrames
        self.items_df = pd.concat([self.items_df, custom_df], ignore_index=True)
        
        # Merge embeddings and coords
        self.embeddings = np.vstack([self.embeddings, custom_embeddings])
        self.umap_coords = np.vstack([self.umap_coords, custom_coords])
    
    def _build_id_index(self) -> None:
        """Build ID to index mapping."""
        self._id_to_idx = {
            id_: idx for idx, id_ in enumerate(self.items_df["id"])
        }
    
    # -------------------------------------------------------------------------
    # Search and retrieval
    # -------------------------------------------------------------------------
    
    def search(
        self,
        query: str,
        k: int = config.DEFAULT_K_NEIGHBORS
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Search for similar items given a text query.
        
        Args:
            query: Search text (any language)
            k: Number of results to return
            
        Returns:
            Tuple of:
            - DataFrame of top-k similar items with similarity scores
            - Query embedding (for potential reuse)
            - Query 2D coordinates (for visualization)
        """
        # Embed query
        query_embedding = self.embedder.embed_single(query)
        
        # Compute similarities (dot product since vectors are normalized)
        similarities = self.embeddings @ query_embedding
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Build results DataFrame
        results = self.items_df.iloc[top_indices].copy()
        results["similarity"] = similarities[top_indices]
        
        # Project query to UMAP space
        query_coords = self.projector.transform_single(query_embedding)
        
        return results, query_embedding, query_coords
    
    def get_neighbors(
        self,
        item_id: str,
        k: int = config.DEFAULT_K_NEIGHBORS
    ) -> pd.DataFrame:
        """
        Get nearest neighbors for a given item.
        
        Args:
            item_id: ID of the item
            k: Number of neighbors to return (excluding the item itself)
            
        Returns:
            DataFrame of k nearest neighbors with similarity scores
        """
        if item_id not in self._id_to_idx:
            raise ValueError(f"Item not found: {item_id}")
        
        idx = self._id_to_idx[item_id]
        item_embedding = self.embeddings[idx]
        
        # Compute similarities
        similarities = self.embeddings @ item_embedding
        
        # Get top k+1 (to exclude self)
        top_indices = np.argsort(similarities)[::-1][:k+1]
        
        # Filter out self
        top_indices = [i for i in top_indices if i != idx][:k]
        
        # Build results DataFrame
        results = self.items_df.iloc[top_indices].copy()
        results["similarity"] = similarities[top_indices]
        
        return results
    
    def get_item(self, item_id: str) -> pd.Series:
        """
        Get a single item by ID.
        
        Args:
            item_id: Item ID
            
        Returns:
            pandas Series with item data
            
        Raises:
            ValueError: If item not found
        """
        if item_id not in self._id_to_idx:
            raise ValueError(f"Item not found: {item_id}")
        
        idx = self._id_to_idx[item_id]
        return self.items_df.iloc[idx]
    
    def get_item_coords(self, item_id: str) -> np.ndarray:
        """Get UMAP coordinates for an item."""
        if item_id not in self._id_to_idx:
            raise ValueError(f"Item not found: {item_id}")
        
        idx = self._id_to_idx[item_id]
        return self.umap_coords[idx]
    
    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    
    def get_all_items(self) -> pd.DataFrame:
        """Get all items as a DataFrame."""
        return self.items_df.copy()
    
    def get_cache_info(self) -> dict:
        """Get information about the cache."""
        if self._cache is None:
            return {"status": "not initialized"}
        return self._cache.get_cache_info()
    
    def clear_cache(self) -> None:
        """Clear the cache for this dataset+embedder."""
        if self._cache:
            self._cache.clear()
    
    @property
    def n_items(self) -> int:
        """Number of items in the store."""
        return len(self.items_df) if self.items_df is not None else 0
    
    @property
    def n_custom_items(self) -> int:
        """Number of custom items in the store."""
        if self.items_df is None:
            return 0
        return len(self.items_df[self.items_df["source"] == config.SOURCE_CUSTOM])
