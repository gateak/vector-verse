"""
VectorStore: Central orchestrator for Vector-Verse.
Manages data loading, embedding, caching, and similarity search.
"""

from pathlib import Path
from typing import Optional, Callable, Union

import numpy as np
import pandas as pd

from vector_verse.loaders.base import BaseDatasetLoader, get_loader
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
        self.umap_coords: Optional[np.ndarray] = None  # 2D coords
        self.umap_coords_3d: Optional[np.ndarray] = None  # 3D coords
        self.projector: Optional[UMAPProjector] = None
        self.projector_3d: Optional[UMAPProjector] = None
        
        # Cache manager (set during initialize)
        self._cache: Optional[CacheManager] = None
        
        # ID to index mapping for fast lookup
        self._id_to_idx: dict[str, int] = {}
        
        # Track initialization state
        self._initialized = False
    
    @property
    def cache_key(self) -> str:
        """Generate unique cache key for this dataset+embedder+sample combination."""
        max_items = getattr(self.dataset_loader, 'max_items', None)
        # For combined_tweets, use max_per_type * 2
        if max_items is None:
            max_per_type = getattr(self.dataset_loader, 'max_per_type', None)
            if max_per_type:
                max_items = max_per_type * 2
        sample_suffix = f"_n{max_items}" if max_items else "_all"
        return f"{self.dataset_loader.name}_{self.embedder.name}{sample_suffix}"
    
    @property
    def is_initialized(self) -> bool:
        """Check if the store has been initialized."""
        return self._initialized
    
    def initialize(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
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
        if self._initialized:
            return
        
        def log(msg: str):
            if progress_callback:
                progress_callback(msg)
            print(msg)
        
        # Initialize cache manager
        self._cache = CacheManager(self.cache_key)
        
        # Check cache state
        if self.force_rebuild:
            log("Force rebuild requested, computing from scratch...")
            self._build_from_scratch(log)
        elif self._cache.is_complete():
            log("Loading from cache...")
            self._load_from_cache()
        elif self._cache.has_embeddings():
            # Partial cache: embeddings exist but UMAP missing
            log("Found embeddings cache, rebuilding UMAP...")
            self._rebuild_umap_from_cached_embeddings(log)
        else:
            log("Building embeddings (this may take a few minutes)...")
            self._build_from_scratch(log)
        
        # Always reload and merge custom items
        self._merge_custom_items(log)
        
        # Build ID index
        self._build_id_index()
        
        self._initialized = True
        log(f"Ready! {len(self.items_df)} items loaded.")
    
    def _load_from_cache(self) -> None:
        """Load embeddings and UMAP from cache."""
        # Load embeddings
        self.embeddings, cached_ids = self._cache.load_embeddings()

        # Load items
        self.items_df = self._cache.load_items()

        # Load 2D UMAP
        self.projector = UMAPProjector()
        self.umap_coords = self._cache.load_umap_coords()
        self.projector.set_model(self._cache.load_umap_model())

        # Load 3D UMAP if available
        self.projector_3d = UMAPProjector(n_components=3)
        if self._cache.has_umap_3d():
            self.umap_coords_3d = self._cache.load_umap_coords_3d()
            self.projector_3d.set_model(self._cache.load_umap_model_3d())

    def _rebuild_umap_from_cached_embeddings(self, log: Callable[[str], None]) -> None:
        """Rebuild UMAP from cached embeddings (handles partial cache state)."""
        # Load existing embeddings and items
        self.embeddings, cached_ids = self._cache.load_embeddings()
        self.items_df = self._cache.load_items()

        log(f"Loaded {len(self.items_df)} items from cache")
        log(f"Embeddings shape: {self.embeddings.shape}")

        # Fit UMAP
        log("Fitting UMAP projection (this takes a minute)...")
        self.projector = UMAPProjector()
        self.umap_coords = self.projector.fit(self.embeddings, show_progress=False)

        # Save UMAP to cache
        log("Saving UMAP to cache...")
        self._cache.save_umap(self.umap_coords, self.projector.get_model())
    
    def _build_from_scratch(self, log: Callable[[str], None]) -> None:
        """Compute embeddings and UMAP from scratch."""
        # Check for partial cache (embeddings done but not UMAP)
        partial_embeddings_path = self._cache.cache_path / "embeddings_partial.npz"
        
        # Load primary dataset
        log(f"Loading {self.dataset_loader.name}...")
        self.items_df = self.dataset_loader.load()
        log(f"Loaded {len(self.items_df)} items")
        
        # Check if we have partial progress
        if partial_embeddings_path.exists() and not self.force_rebuild:
            log("Found partial embeddings, resuming...")
            data = np.load(partial_embeddings_path)
            existing_embeddings = data["embeddings"]
            start_idx = len(existing_embeddings)
            
            if start_idx < len(self.items_df):
                # Resume from where we left off
                remaining_texts = self.items_df["text"].tolist()[start_idx:]
                log(f"Resuming embeddings from item {start_idx}...")
                new_embeddings = self.embedder.embed(remaining_texts)
                self.embeddings = np.vstack([existing_embeddings, new_embeddings])
            else:
                self.embeddings = existing_embeddings
        else:
            # Generate embeddings with incremental saving
            texts = self.items_df["text"].tolist()
            log(f"Generating embeddings for {len(texts)} texts...")
            self.embeddings = self._embed_with_checkpoints(texts, log)
        
        log(f"Embeddings shape: {self.embeddings.shape}")
        
        # Clean up partial file
        if partial_embeddings_path.exists():
            partial_embeddings_path.unlink()
        
        # Fit UMAP
        log("Fitting UMAP projection (this takes a minute)...")
        self.projector = UMAPProjector()
        self.umap_coords = self.projector.fit(self.embeddings, show_progress=False)
        
        # Save to cache
        log("Saving to cache...")
        self._cache.save_embeddings(self.embeddings, self.items_df["id"].tolist())
        self._cache.save_items(self.items_df)
        self._cache.save_umap(self.umap_coords, self.projector.get_model())
    
    def _embed_with_checkpoints(
        self,
        texts: list[str],
        log: Callable[[str], None],
        checkpoint_every: int = 5
    ) -> np.ndarray:
        """
        Embed texts with periodic checkpoints to allow resuming.
        
        Args:
            texts: List of texts to embed
            log: Logging function
            checkpoint_every: Save checkpoint every N batches
            
        Returns:
            Embeddings array
        """
        partial_path = self._cache.cache_path / "embeddings_partial.npz"
        self._cache._ensure_cache_dir()
        
        all_embeddings = []
        batch_size = self.embedder.batch_size
        n_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_num in range(n_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            log(f"Embedding batch {batch_num + 1}/{n_batches}...")
            
            # Embed this batch
            batch_embeddings = self.embedder.embed(batch_texts, show_progress=False)
            all_embeddings.append(batch_embeddings)
            
            # Save checkpoint periodically
            if (batch_num + 1) % checkpoint_every == 0:
                current_embeddings = np.vstack(all_embeddings)
                np.savez_compressed(partial_path, embeddings=current_embeddings)
                log(f"Checkpoint saved ({end_idx}/{len(texts)} items)")
        
        return np.vstack(all_embeddings)
    
    def _merge_custom_items(self, log: Callable[[str], None]) -> None:
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
    
    def ensure_3d_umap(self, progress_callback: Optional[Callable[[str], None]] = None) -> np.ndarray:
        """
        Ensure 3D UMAP coords exist, computing if necessary.
        
        Args:
            progress_callback: Optional callable for progress updates
            
        Returns:
            3D UMAP coordinates array of shape (n, 3)
        """
        if self.umap_coords_3d is not None:
            return self.umap_coords_3d
        
        def log(msg: str):
            if progress_callback:
                progress_callback(msg)
            print(msg)
        
        # Check cache first
        if self._cache.has_umap_3d():
            log("Loading 3D UMAP from cache...")
            self.umap_coords_3d = self._cache.load_umap_coords_3d()
            self.projector_3d = UMAPProjector(n_components=3)
            self.projector_3d.set_model(self._cache.load_umap_model_3d())
            return self.umap_coords_3d
        
        # Compute 3D UMAP
        log("Computing 3D UMAP projection (this may take a minute)...")
        self.projector_3d = UMAPProjector(n_components=3)
        self.umap_coords_3d = self.projector_3d.fit(self.embeddings, show_progress=False)
        
        # Save to cache
        log("Saving 3D UMAP to cache...")
        self._cache.save_umap_3d(self.umap_coords_3d, self.projector_3d.get_model())
        
        # If we have custom items, we need to project them too
        n_custom = self.n_custom_items
        if n_custom > 0:
            # Re-project custom items (they were already added to embeddings)
            # The 3D coords already include them from the fit above
            pass
        
        return self.umap_coords_3d
    
    @property
    def has_3d_umap(self) -> bool:
        """Check if 3D UMAP coords are available (loaded or computed)."""
        return self.umap_coords_3d is not None or self._cache.has_umap_3d()
    
    # -------------------------------------------------------------------------
    # Search and retrieval
    # -------------------------------------------------------------------------
    
    def search(
        self,
        query: str,
        k: int = config.DEFAULT_K_NEIGHBORS,
        index_mask: Optional[np.ndarray] = None
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Search for similar items given a text query.
        
        Args:
            query: Search text (any language)
            k: Number of results to return
            index_mask: Optional array of indices to scope search to (for zoom)
            
        Returns:
            Tuple of:
            - DataFrame of top-k similar items with similarity scores
            - Query embedding (for potential reuse)
            - Query 2D coordinates (for visualization)
        """
        # Embed query
        query_embedding = self.embedder.embed_single(query)
        
        # Determine which embeddings to search
        if index_mask is not None:
            search_embeddings = self.embeddings[index_mask]
            search_df = self.items_df.iloc[index_mask]
        else:
            search_embeddings = self.embeddings
            search_df = self.items_df
        
        # Compute similarities (dot product since vectors are normalized)
        similarities = search_embeddings @ query_embedding
        
        # Get top-k indices (within the search scope)
        k_actual = min(k, len(similarities))
        top_local_indices = np.argsort(similarities)[::-1][:k_actual]
        
        # Build results DataFrame
        results = search_df.iloc[top_local_indices].copy()
        results["similarity"] = similarities[top_local_indices]
        
        # Map back to full dataset indices if needed
        if index_mask is not None:
            results["_full_index"] = index_mask[top_local_indices]
        
        # Project query to UMAP space
        query_coords = self.projector.transform_single(query_embedding)
        
        return results, query_embedding, query_coords
    
    def get_neighbors(
        self,
        item_id: str,
        k: int = config.DEFAULT_K_NEIGHBORS,
        index_mask: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Get nearest neighbors for a given item.
        
        Args:
            item_id: ID of the item
            k: Number of neighbors to return (excluding the item itself)
            index_mask: Optional array of indices to scope search to (for zoom)
            
        Returns:
            DataFrame of k nearest neighbors with similarity scores
        """
        if item_id not in self._id_to_idx:
            raise ValueError(f"Item not found: {item_id}")
        
        idx = self._id_to_idx[item_id]
        item_embedding = self.embeddings[idx]
        
        # Determine which embeddings to search
        if index_mask is not None:
            search_embeddings = self.embeddings[index_mask]
            search_df = self.items_df.iloc[index_mask]
            # Find the item's position in the masked view
            try:
                local_idx = np.where(index_mask == idx)[0][0]
            except IndexError:
                local_idx = -1  # Item not in current scope
        else:
            search_embeddings = self.embeddings
            search_df = self.items_df
            local_idx = idx
        
        # Compute similarities
        similarities = search_embeddings @ item_embedding
        
        # Get top k+1 (to exclude self if present)
        k_fetch = min(k + 1, len(similarities))
        top_local_indices = np.argsort(similarities)[::-1][:k_fetch]
        
        # Filter out self
        if local_idx >= 0:
            top_local_indices = [i for i in top_local_indices if i != local_idx][:k]
        else:
            top_local_indices = top_local_indices[:k]
        
        # Build results DataFrame
        results = search_df.iloc[top_local_indices].copy()
        results["similarity"] = similarities[top_local_indices]
        
        # Map back to full dataset indices if needed
        if index_mask is not None:
            results["_full_index"] = index_mask[top_local_indices]
        
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
