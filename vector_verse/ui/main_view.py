"""Main view UI components (search, visualization, zoom)."""

import logging
import streamlit as st
import numpy as np
from typing import TYPE_CHECKING, Optional

from vector_verse.visualization.scatter import ScatterPlotBuilder
from vector_verse.ui.state import AppState
from vector_verse.ui.styles import render_error
import config

if TYPE_CHECKING:
    from vector_verse.core.vector_store import VectorStore
    from vector_verse.core.zoom_manager import ZoomManager

logger = logging.getLogger(__name__)


def render_search_bar(vs: "VectorStore", zm: "ZoomManager") -> None:
    """Render search input and handle search action."""
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            "Search by meaning",
            placeholder="Enter text in any language to find similar items...",
            key="search_input",
            label_visibility="collapsed"
        )

    with col2:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    if search_clicked and query.strip():
        _perform_search(vs, zm, query.strip())


def _perform_search(vs: "VectorStore", zm: "ZoomManager", query: str) -> None:
    """Execute search and update state."""
    index_mask = zm.get_current_indices() if zm.is_zoomed else None

    try:
        with st.spinner("Searching..."):
            results, query_embedding, query_coords = vs.search(
                query,
                k=config.DEFAULT_K_NEIGHBORS,
                index_mask=index_mask
            )
            AppState.set_search_results(results, query, query_coords, query_embedding)
    except Exception as e:
        logger.exception("Search failed")
        st.error(f"Search failed: {e}")


def render_zoom_controls(zm: "ZoomManager") -> None:
    """Render zoom breadcrumb and navigation controls."""
    if not zm.is_zoomed:
        return

    st.markdown(
        f'<div class="vv-breadcrumb">📍 {zm.breadcrumb}</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Back", use_container_width=True):
            zm.zoom_out()
            AppState.reset_for_zoom_change()
            st.rerun()

    with col2:
        if st.button("Reset", use_container_width=True):
            zm.reset_to_root()
            AppState.reset_for_zoom_change()
            st.rerun()

    with col3:
        items_in_view = len(zm.get_current_indices())
        st.markdown(f"*{items_in_view:,} items in view*")


def render_zoom_button(vs: "VectorStore", zm: "ZoomManager") -> None:
    """Render zoom-into-selection button if there's a selection."""
    if not AppState.has_lasso_selection():
        return

    n_selected = len(st.session_state.lasso_selection)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"🔍 {n_selected} items selected")
    with col2:
        if st.button("Zoom Into Selection", type="primary", use_container_width=True):
            _perform_zoom(vs, zm)

    if st.button("Clear Selection", use_container_width=True):
        AppState.clear_lasso_selection()
        st.rerun()


def _perform_zoom(vs: "VectorStore", zm: "ZoomManager") -> None:
    """Execute zoom into selection."""
    indices = np.array(st.session_state.lasso_selection)

    try:
        with st.spinner("Computing zoomed view..."):
            zm.zoom_into(indices)
        AppState.clear_lasso_selection()
        AppState.reset_for_zoom_change()
        st.rerun()
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        logger.exception("Zoom failed")
        st.error(f"Zoom failed: {e}")


def render_visualization(vs: "VectorStore", zm: "ZoomManager") -> None:
    """Render the UMAP scatter plot."""
    builder = ScatterPlotBuilder()
    is_3d = st.session_state.view_3d

    # Get data for current view
    if zm.is_zoomed:
        current_df = zm.get_subset_items(vs.items_df)
        current_coords = zm.get_current_coords()
        if is_3d:
            st.info("3D view not available when zoomed. Showing 2D.")
            is_3d = False
    else:
        current_df = vs.items_df
        if is_3d:
            if vs.umap_coords_3d is None:
                with st.spinner("Computing 3D projection..."):
                    vs.ensure_3d_umap()
            current_coords = vs.umap_coords_3d
        else:
            current_coords = vs.umap_coords

    # Get neighbor highlighting
    neighbor_ids = _get_neighbor_ids(vs, zm)

    # Get query coordinates (with 3D projection)
    query_coords = _get_query_coords(vs, zm, is_3d)

    # Determine color settings
    color_by = st.session_state.color_by
    color_type = "sequential" if color_by in ["decade", "year", "date_bucket"] else "categorical"

    # Build plot
    fig = builder.build(
        df=current_df,
        coords=current_coords,
        selected_id=st.session_state.selected_item_id,
        neighbor_ids=neighbor_ids,
        query_coords=query_coords,
        query_text=st.session_state.search_query if query_coords is not None else None,
        color_by=color_by,
        color_type=color_type,
        enable_lasso=not is_3d,
        mode_3d=is_3d,
    )

    # Render
    if is_3d:
        st.plotly_chart(fig, use_container_width=True, key="umap_plot_3d")
    else:
        _render_2d_with_selection(fig)


def _get_neighbor_ids(vs: "VectorStore", zm: "ZoomManager") -> Optional[list]:
    """Get neighbor IDs for highlighting."""
    if st.session_state.selected_item_id:
        index_mask = zm.get_current_indices() if zm.is_zoomed else None
        neighbors = vs.get_neighbors(st.session_state.selected_item_id, index_mask=index_mask)
        return neighbors["id"].tolist()
    elif st.session_state.search_results is not None:
        return st.session_state.search_results["id"].tolist()
    return None


def _get_query_coords(vs: "VectorStore", zm: "ZoomManager", is_3d: bool) -> Optional[np.ndarray]:
    """Get query coordinates, projecting to 3D if needed."""
    query_embedding = st.session_state.get("query_embedding")
    if query_embedding is None:
        return None

    if zm.is_zoomed:
        # Use stored 2D coords for zoomed view
        return st.session_state.query_coords

    if is_3d:
        if vs.projector_3d is not None and vs.projector_3d.is_fitted:
            return vs.projector_3d.transform_single(query_embedding)
        return None

    return st.session_state.query_coords


def _render_2d_with_selection(fig) -> None:
    """Render 2D plot with lasso selection handling."""
    selection = st.plotly_chart(
        fig,
        use_container_width=True,
        key="umap_plot",
        on_select="rerun",
        selection_mode="lasso"
    )

    # Handle lasso selection using customdata[1] which is global index
    if selection and "selection" in selection:
        points = selection["selection"].get("points", [])
        if points:
            selected_indices = []
            for point in points:
                customdata = point.get("customdata")
                if customdata is not None and len(customdata) > 1:
                    try:
                        selected_indices.append(int(customdata[1]))
                    except (ValueError, TypeError):
                        pass

            if selected_indices:
                AppState.set_lasso_selection(selected_indices)


def render_loading_screen(vs: "VectorStore") -> None:
    """Render the loading/embedding progress screen."""
    st.markdown("### Building Embedding Index")
    st.markdown("""
    This is a one-time process. The embeddings will be cached for future use.

    **Please don't refresh the page** - progress is saved periodically.
    """)

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(msg: str):
        status_text.text(msg)
        if "batch" in msg.lower():
            try:
                parts = msg.split()
                for p in parts:
                    if "/" in p:
                        current, total = p.split("/")
                        current = int(current)
                        total = int(total.rstrip("..."))
                        progress_bar.progress(current / total)
                        break
            except (ValueError, IndexError):
                pass

    try:
        vs.initialize(progress_callback=update_progress)
        progress_bar.progress(1.0)
        status_text.text("Done! Refreshing...")
        st.rerun()
    except Exception as e:
        logger.exception("Initialization failed")
        st.error(f"Error during initialization: {e}")
        st.exception(e)
