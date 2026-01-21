"""
ZoomManager: Hierarchical zoom into embedding subsets.
Manages zoom state, caches subset UMAP projections, and provides breadcrumb navigation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import hashlib

import numpy as np

from vector_verse.cache.manager import CacheManager
from vector_verse.core.projector import UMAPProjector
import config


@dataclass
class ZoomLevel:
    """Represents one level in the zoom hierarchy."""
    level_id: str                    # Hash of parent_id + index_mask
    parent_id: Optional[str]         # None for root
    index_mask: np.ndarray           # Boolean mask or integer indices into full embeddings
    umap_coords: np.ndarray          # Re-computed 2D coords for this subset
    label: str                       # Human-readable: "All", "Zoom 1", etc.
    depth: int = 0                   # Depth in hierarchy (0 = root)


class ZoomManager:
    """
    Manages hierarchical zoom state and caches subset UMAP projections.
    
    Features:
    - Zoom into lasso-selected regions
    - Re-run UMAP on subset for better local structure
    - Cache zoom projections for instant re-access
    - Breadcrumb navigation (back/reset)
    """
    
    def __init__(
        self,
        cache_manager: CacheManager,
        full_embeddings: np.ndarray,
        full_umap_coords: np.ndarray,
        umap_params: Optional[dict] = None
    ):
        """
        Initialize ZoomManager.
        
        Args:
            cache_manager: CacheManager for the current dataset
            full_embeddings: Full embedding matrix (n_items, embedding_dim)
            full_umap_coords: Full UMAP coordinates (n_items, 2)
            umap_params: Optional UMAP parameters override
        """
        self.cache = cache_manager
        self.full_embeddings = full_embeddings
        self.full_umap_coords = full_umap_coords
        self.umap_params = umap_params or {
            "n_neighbors": config.UMAP_N_NEIGHBORS,
            "min_dist": config.UMAP_MIN_DIST,
            "metric": config.UMAP_METRIC,
        }
        
        # Zoom level stack
        self._levels: list[ZoomLevel] = []
        
        # Initialize with root level
        self._init_root_level()
        
        # Cache directory for zoom projections
        self._zoom_cache_dir = cache_manager.cache_path / "zoom_levels"
        self._zoom_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_root_level(self) -> None:
        """Initialize the root (full dataset) level."""
        n_items = len(self.full_embeddings)
        root = ZoomLevel(
            level_id="root",
            parent_id=None,
            index_mask=np.arange(n_items),  # All indices
            umap_coords=self.full_umap_coords,
            label="All",
            depth=0
        )
        self._levels = [root]
    
    @property
    def levels(self) -> list[ZoomLevel]:
        """Get all zoom levels (read-only view)."""
        return list(self._levels)
    
    @property
    def current_level(self) -> ZoomLevel:
        """Get the current (deepest) zoom level."""
        return self._levels[-1]
    
    @property
    def depth(self) -> int:
        """Current zoom depth (0 = root)."""
        return len(self._levels) - 1
    
    @property
    def is_zoomed(self) -> bool:
        """Check if we're zoomed in (not at root)."""
        return len(self._levels) > 1
    
    @property
    def breadcrumb(self) -> str:
        """Human-readable breadcrumb trail."""
        return " → ".join(level.label for level in self._levels)
    
    def get_current_indices(self) -> np.ndarray:
        """Get indices (into full dataset) of items in current view."""
        return self.current_level.index_mask
    
    def get_current_coords(self) -> np.ndarray:
        """Get UMAP coordinates for current zoom level."""
        return self.current_level.umap_coords
    
    def zoom_into(
        self,
        selected_indices: np.ndarray,
        label: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> ZoomLevel:
        """
        Create new zoom level from selected indices.
        
        Args:
            selected_indices: Indices into CURRENT level's data (not full dataset)
            label: Human-readable label for this level
            progress_callback: Optional progress reporter
            
        Returns:
            The new ZoomLevel
        """
        def log(msg: str):
            if progress_callback:
                progress_callback(msg)
        
        # Convert from current-level indices to full-dataset indices
        current_mask = self.current_level.index_mask
        full_indices = current_mask[selected_indices]
        
        # Generate cache key
        parent_id = self.current_level.level_id
        level_id = self._compute_cache_key(parent_id, full_indices)
        
        # Auto-generate label if not provided
        if label is None:
            label = f"Zoom {self.depth + 1}"
        
        # Check cache first
        cache_path = self._zoom_cache_dir / f"{level_id}.npz"
        
        if cache_path.exists():
            log("Loading cached zoom projection...")
            data = np.load(cache_path)
            umap_coords = data["coords"]
        else:
            log(f"Computing UMAP for {len(full_indices)} items...")
            umap_coords = self._fit_subset_umap(full_indices)
            
            # Cache the result
            np.savez_compressed(cache_path, coords=umap_coords, indices=full_indices)
            log("Zoom projection cached.")
        
        # Create new level
        new_level = ZoomLevel(
            level_id=level_id,
            parent_id=parent_id,
            index_mask=full_indices,
            umap_coords=umap_coords,
            label=label,
            depth=self.depth + 1
        )
        
        self._levels.append(new_level)
        return new_level
    
    def zoom_out(self) -> Optional[ZoomLevel]:
        """
        Go back one zoom level.
        
        Returns:
            The previous level, or None if already at root
        """
        if len(self._levels) <= 1:
            return None
        
        self._levels.pop()
        return self.current_level
    
    def reset_to_root(self) -> ZoomLevel:
        """
        Reset to full dataset view.
        
        Returns:
            The root level
        """
        self._levels = self._levels[:1]  # Keep only root
        return self.current_level
    
    def _compute_cache_key(self, parent_id: str, indices: np.ndarray) -> str:
        """
        Compute unique cache key for a zoom level.
        
        Args:
            parent_id: ID of parent level
            indices: Full-dataset indices for this level
            
        Returns:
            Unique hash string
        """
        # Sort indices for consistent hashing
        sorted_indices = np.sort(indices)
        indices_hash = hashlib.md5(sorted_indices.tobytes()).hexdigest()[:12]
        return f"{parent_id}_{indices_hash}"
    
    def _fit_subset_umap(self, indices: np.ndarray) -> np.ndarray:
        """
        Fit UMAP on a subset of embeddings.
        
        Args:
            indices: Indices into full embedding matrix
            
        Returns:
            2D coordinates for the subset
        """
        subset_embeddings = self.full_embeddings[indices]
        
        # Adjust n_neighbors if subset is small
        n_items = len(subset_embeddings)
        n_neighbors = min(self.umap_params["n_neighbors"], max(2, n_items - 1))
        
        projector = UMAPProjector(
            n_neighbors=n_neighbors,
            min_dist=self.umap_params["min_dist"],
            metric=self.umap_params["metric"],
        )
        
        coords = projector.fit(subset_embeddings, show_progress=False)
        return coords
    
    def map_to_current_indices(self, full_indices: np.ndarray) -> np.ndarray:
        """
        Map full-dataset indices to current-level indices.
        
        Args:
            full_indices: Indices into full dataset
            
        Returns:
            Indices into current level's data, or -1 for items not in current level
        """
        current_mask = self.current_level.index_mask
        
        # Create mapping: full_idx -> current_idx
        full_to_current = {full_idx: i for i, full_idx in enumerate(current_mask)}
        
        return np.array([
            full_to_current.get(idx, -1) for idx in full_indices
        ])
    
    def get_subset_items(self, df):
        """
        Get DataFrame rows for current zoom level.
        
        Args:
            df: Full items DataFrame
            
        Returns:
            Subset DataFrame for current zoom level
        """
        indices = self.current_level.index_mask
        return df.iloc[indices].reset_index(drop=True)
    
    def clear_zoom_cache(self) -> None:
        """Clear all cached zoom projections."""
        import shutil
        if self._zoom_cache_dir.exists():
            shutil.rmtree(self._zoom_cache_dir)
            self._zoom_cache_dir.mkdir(parents=True, exist_ok=True)
