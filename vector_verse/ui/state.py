"""
Centralized session state management for Vector-Verse.
Provides typed accessors and clear state transition methods.
"""

from dataclasses import dataclass
from typing import Optional, List, Any
import streamlit as st
import numpy as np
import pandas as pd


@dataclass
class StateDefaults:
    """Default values for all session state variables."""
    selected_item_id: Optional[str] = None
    search_results: Optional[pd.DataFrame] = None
    search_query: str = ""
    query_coords: Optional[np.ndarray] = None
    query_embedding: Optional[np.ndarray] = None
    current_dataset: str = "poetry"
    zoom_manager: Optional[Any] = None
    color_by: Optional[str] = None
    lasso_selection: Optional[List[int]] = None
    view_3d: bool = False
    last_error: Optional[str] = None
    sample_size: Optional[int] = 5000  # None means all items


class AppState:
    """
    Wrapper around Streamlit session state with type hints and defaults.
    Provides clear API for state transitions.
    """

    @classmethod
    def init(cls, default_dataset: str = "poetry") -> None:
        """Initialize all session state with defaults."""
        defaults = StateDefaults(current_dataset=default_dataset)
        for field_name in defaults.__dataclass_fields__:
            if field_name not in st.session_state:
                st.session_state[field_name] = getattr(defaults, field_name)

    @classmethod
    def reset_for_dataset_change(cls) -> None:
        """Clear transient state when switching datasets."""
        st.session_state.selected_item_id = None
        st.session_state.search_results = None
        st.session_state.search_query = ""
        st.session_state.query_coords = None
        st.session_state.query_embedding = None
        st.session_state.zoom_manager = None
        st.session_state.color_by = None
        st.session_state.lasso_selection = None
        st.session_state.last_error = None

    @classmethod
    def reset_for_zoom_change(cls) -> None:
        """Clear state when zoom level changes."""
        st.session_state.selected_item_id = None
        st.session_state.search_results = None
        st.session_state.lasso_selection = None

    @classmethod
    def clear_selection(cls) -> None:
        """Clear current selection state."""
        st.session_state.selected_item_id = None
        st.session_state.search_results = None
        st.session_state.query_coords = None
        st.session_state.query_embedding = None

    @classmethod
    def set_error(cls, message: str) -> None:
        """Record an error for display."""
        st.session_state.last_error = message

    @classmethod
    def clear_error(cls) -> None:
        """Clear any recorded error."""
        st.session_state.last_error = None

    @classmethod
    def set_search_results(
        cls,
        results: pd.DataFrame,
        query: str,
        coords: np.ndarray,
        embedding: np.ndarray
    ) -> None:
        """Set search results with all associated data."""
        st.session_state.search_results = results
        st.session_state.search_query = query
        st.session_state.query_coords = coords
        st.session_state.query_embedding = embedding
        st.session_state.selected_item_id = None

    @classmethod
    def set_selected_item(cls, item_id: str) -> None:
        """Select an item and clear search results."""
        st.session_state.selected_item_id = item_id
        st.session_state.search_results = None
        st.session_state.query_coords = None

    @classmethod
    def set_lasso_selection(cls, indices: List[int]) -> None:
        """Set lasso selection indices."""
        st.session_state.lasso_selection = list(set(indices)) if indices else None

    @classmethod
    def clear_lasso_selection(cls) -> None:
        """Clear lasso selection."""
        st.session_state.lasso_selection = None

    # Property-style accessors for common checks
    @staticmethod
    def has_selection() -> bool:
        """Check if an item is selected."""
        return st.session_state.get("selected_item_id") is not None

    @staticmethod
    def has_search_results() -> bool:
        """Check if search results exist."""
        return st.session_state.get("search_results") is not None

    @staticmethod
    def has_lasso_selection() -> bool:
        """Check if lasso selection exists."""
        selection = st.session_state.get("lasso_selection")
        return selection is not None and len(selection) > 0

    @staticmethod
    def has_error() -> bool:
        """Check if there's an error to display."""
        return st.session_state.get("last_error") is not None

    @staticmethod
    def is_3d_view() -> bool:
        """Check if 3D view is enabled."""
        return st.session_state.get("view_3d", False)


def init_session_state(default_dataset: str = "poetry") -> None:
    """Convenience function to initialize session state."""
    AppState.init(default_dataset)
