"""
UMAP projection for dimensionality reduction.
Handles fitting and transforming embeddings to 2D/3D space.
"""

import warnings
from typing import Optional, Any

import numpy as np
import umap

import config


class UMAPProjector:
    """
    UMAP-based dimensionality reduction for embedding visualization.
    
    Features:
    - Fits UMAP on corpus embeddings
    - Projects new queries onto existing space
    - Configurable parameters for tuning
    """
    
    def __init__(
        self,
        n_neighbors: int = config.UMAP_N_NEIGHBORS,
        min_dist: float = config.UMAP_MIN_DIST,
        metric: str = config.UMAP_METRIC,
        n_components: int = config.UMAP_N_COMPONENTS,
        random_state: int = config.UMAP_RANDOM_STATE
    ):
        """
        Initialize UMAP projector.
        
        Args:
            n_neighbors: Number of neighbors for local structure (default: 15)
            min_dist: Minimum distance between points (default: 0.1)
            metric: Distance metric (default: "cosine")
            n_components: Output dimensions (default: 2)
            random_state: Random seed for reproducibility
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.random_state = random_state
        
        self._model: Optional[umap.UMAP] = None
    
    def fit(self, embeddings: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """
        Fit UMAP on embeddings and return projected coordinates.
        
        Args:
            embeddings: Array of shape (n, embedding_dim)
            show_progress: Whether to print progress
            
        Returns:
            Array of shape (n, n_components) with projected coordinates
        """
        if show_progress:
            print(f"Fitting UMAP ({self.n_components}D) on {len(embeddings)} embeddings...")
        
        self._model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            n_components=self.n_components,
            random_state=self.random_state,
            verbose=show_progress
        )
        
        # Suppress spectral initialization warnings (common with large datasets)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Spectral initialisation failed.*")
            coords = self._model.fit_transform(embeddings)
        
        if show_progress:
            print("UMAP fitting complete.")
        
        return coords.astype(np.float32)
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project new embeddings onto existing UMAP space.
        
        Args:
            embeddings: Array of shape (n, embedding_dim)
            
        Returns:
            Array of shape (n, 2) with 2D coordinates
            
        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if self._model is None:
            raise RuntimeError("UMAP model not fitted. Call fit() first or load from cache.")
        
        coords = self._model.transform(embeddings)
        return coords.astype(np.float32)
    
    def transform_single(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project a single embedding to 2D.
        
        Args:
            embedding: Array of shape (embedding_dim,)
            
        Returns:
            Array of shape (2,) with 2D coordinates
        """
        return self.transform(embedding.reshape(1, -1))[0]
    
    def set_model(self, model: umap.UMAP) -> None:
        """
        Set the UMAP model (e.g., loaded from cache).
        
        Args:
            model: Pre-fitted UMAP model
        """
        self._model = model
    
    def get_model(self) -> Optional[umap.UMAP]:
        """
        Get the fitted UMAP model.
        
        Returns:
            Fitted UMAP model or None if not fitted
        """
        return self._model
    
    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._model is not None
