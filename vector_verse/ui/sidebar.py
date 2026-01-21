"""Sidebar UI components for Vector-Verse."""

import logging
import streamlit as st
import pandas as pd
from typing import TYPE_CHECKING

from vector_verse.ui.state import AppState
import config

if TYPE_CHECKING:
    from vector_verse.core.vector_store import VectorStore
    from vector_verse.core.zoom_manager import ZoomManager

logger = logging.getLogger(__name__)

MAX_BROWSE_ITEMS = 500


def render_sidebar(vs: "VectorStore", zm: "ZoomManager") -> None:
    """Render the complete sidebar."""
    with st.sidebar:
        render_dataset_switcher()
        st.markdown("---")
        render_sample_size()
        st.markdown("---")
        render_dataset_info(vs)
        st.markdown("---")
        render_color_selector()
        st.markdown("---")
        render_view_toggle(vs)
        st.markdown("---")
        render_cache_controls(vs)
        st.markdown("---")
        render_browse_items(vs, zm)


def render_dataset_switcher() -> None:
    """Render dataset selector radio buttons."""
    st.markdown("### Dataset")

    available = []
    for key, cfg in config.AVAILABLE_DATASETS.items():
        try:
            if cfg["data_check"]():
                available.append((key, cfg["label"], cfg.get("description", "")))
        except Exception as e:
            logger.debug(f"Dataset {key} check failed: {e}")

    if not available:
        st.warning("No datasets found. Add data to the data/ folder.")
        return

    current = st.session_state.current_dataset
    valid_keys = [k for k, _, _ in available]

    if current not in valid_keys:
        current = available[0][0]
        st.session_state.current_dataset = current

    selected = st.radio(
        "Select dataset:",
        valid_keys,
        format_func=lambda x: dict((k, l) for k, l, _ in available)[x],
        index=valid_keys.index(current),
        key="dataset_radio",
        help="Switch between different text datasets"
    )

    if selected != current:
        st.session_state.current_dataset = selected
        AppState.reset_for_dataset_change()
        st.cache_resource.clear()
        st.rerun()


def render_sample_size() -> None:
    """Render sample size control."""
    st.markdown("### Sample Size")

    current_sample = st.session_state.sample_size
    is_load_all = current_sample is None

    load_all = st.checkbox(
        "Load all items",
        value=is_load_all,
        help="Load entire dataset (slower, uses more memory)"
    )

    # Handle checkbox change
    if load_all and not is_load_all:
        # User just checked "Load all"
        st.session_state.sample_size = None
        AppState.reset_for_dataset_change()
        st.cache_resource.clear()
        st.rerun()
    elif not load_all and is_load_all:
        # User just unchecked "Load all" - set default
        st.session_state.sample_size = 5000
        AppState.reset_for_dataset_change()
        st.cache_resource.clear()
        st.rerun()

    # Show slider only when not loading all
    if not load_all:
        new_sample_size = st.slider(
            "Max items to load:",
            min_value=1000,
            max_value=50000,
            value=current_sample or 5000,
            step=1000,
            help="Randomly sample this many items from the dataset"
        )

        if new_sample_size != current_sample:
            st.session_state.sample_size = new_sample_size
            AppState.reset_for_dataset_change()
            st.cache_resource.clear()
            st.rerun()


def render_dataset_info(vs: "VectorStore") -> None:
    """Render dataset info section."""
    st.markdown("### Dataset Info")
    st.markdown(f"**Items:** {vs.n_items:,}")
    if vs.n_custom_items > 0:
        st.markdown(f"**Custom items:** {vs.n_custom_items}")


def render_color_selector() -> None:
    """Render color dimension selector."""
    st.markdown("### Color By")

    current_dataset = st.session_state.current_dataset
    dataset_config = config.AVAILABLE_DATASETS.get(current_dataset, {})
    color_dims = dataset_config.get("color_dimensions", ["source"])

    options = ["source (default)"] + color_dims
    current_color = st.session_state.color_by or "source (default)"

    selected = st.selectbox(
        "Color dimension:",
        options,
        index=options.index(current_color) if current_color in options else 0,
        key="color_selector"
    )

    if selected == "source (default)":
        st.session_state.color_by = None
    else:
        st.session_state.color_by = selected


def render_view_toggle(vs: "VectorStore") -> None:
    """Render 2D/3D view toggle."""
    st.markdown("### View Mode")

    view_3d = st.toggle(
        "3D View",
        value=st.session_state.view_3d,
        help="Switch between 2D and 3D UMAP projection"
    )

    if view_3d != st.session_state.view_3d:
        st.session_state.view_3d = view_3d
        if view_3d and not vs.has_3d_umap:
            st.rerun()


def render_cache_controls(vs: "VectorStore") -> None:
    """Render cache management controls."""
    if st.button("Rebuild Cache", help="Re-embed all items and recompute UMAP"):
        vs.clear_cache()
        st.session_state.zoom_manager = None
        st.cache_resource.clear()
        st.rerun()


def render_browse_items(vs: "VectorStore", zm: "ZoomManager") -> None:
    """Render item browser dropdown."""
    st.markdown("### Browse Items")

    if zm.is_zoomed:
        items_df = zm.get_subset_items(vs.items_df)
    else:
        items_df = vs.get_all_items()

    total_items = len(items_df)
    display_col = "title" if "title" in items_df.columns else "text"
    author_col = "author" if "author" in items_df.columns else None

    # Show limit warning for large datasets
    if total_items > MAX_BROWSE_ITEMS:
        st.caption(f"Showing first {MAX_BROWSE_ITEMS} of {total_items:,} items")

    # Build options list
    browse_df = items_df.head(MAX_BROWSE_ITEMS)
    options = ["-- Select an item --"] + [
        _format_item_label(row, display_col, author_col)
        for _, row in browse_df.iterrows()
    ]

    selected_idx = st.selectbox(
        "Select item:",
        range(len(options)),
        format_func=lambda x: options[x],
        key="item_selector"
    )

    if selected_idx > 0:
        item_id = browse_df.iloc[selected_idx - 1]["id"]
        if item_id != st.session_state.selected_item_id:
            AppState.set_selected_item(item_id)
            st.rerun()


def _format_item_label(row: pd.Series, display_col: str, author_col: str | None) -> str:
    """Format a single item for the dropdown."""
    title = str(row[display_col])[:50]
    if len(str(row[display_col])) > 50:
        title += "..."

    if author_col and pd.notna(row.get(author_col)):
        author = str(row[author_col])[:20]
        return f"{title} ({author})"
    return title
