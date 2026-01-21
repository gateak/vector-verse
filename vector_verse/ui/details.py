"""Item details panel components."""

import streamlit as st
import pandas as pd
from typing import TYPE_CHECKING

from vector_verse.ui.state import AppState
import config

if TYPE_CHECKING:
    from vector_verse.core.vector_store import VectorStore
    from vector_verse.core.zoom_manager import ZoomManager


def render_item_details(vs: "VectorStore", zm: "ZoomManager") -> None:
    """Render the selected item details panel."""
    index_mask = zm.get_current_indices() if zm.is_zoomed else None

    if st.session_state.selected_item_id:
        item = vs.get_item(st.session_state.selected_item_id)
        neighbors = vs.get_neighbors(st.session_state.selected_item_id, index_mask=index_mask)
        render_item_card(item)
        render_neighbors_list(neighbors, "Similar Items")

    elif st.session_state.search_results is not None:
        render_search_results()

    else:
        render_getting_started()


def render_search_results() -> None:
    """Render search results list."""
    st.markdown("### Search Results")
    st.markdown(f'*Query: "{st.session_state.search_query}"*')

    for _, result in st.session_state.search_results.iterrows():
        title = result.get("title", result.get("text", "Untitled")[:60])
        title_display = str(title)[:60]
        if len(str(title)) > 60:
            title_display += "..."

        with st.expander(f"📄 {title_display} ({result['similarity']:.3f})"):
            if "author" in result:
                st.markdown(f"**Author:** {result['author']}")
            st.markdown("---")
            text = result.get("text", "")
            st.markdown(text[:1000] + ("..." if len(text) > 1000 else ""))

            if st.button(f"Select", key=f"select_{result['id']}"):
                AppState.set_selected_item(result["id"])
                st.rerun()


def render_item_card(item: pd.Series) -> None:
    """Render a single item card."""
    source = item.get("source", "")
    source_class = ""

    if source == config.SOURCE_CUSTOM:
        source_class = "vv-source-custom"
        source_badge = " <span class='vv-source-custom'>(custom)</span>"
    elif source == config.SOURCE_TWEETS:
        source_class = "vv-source-tweets"
        source_badge = " <span class='vv-source-tweets'>(tweet)</span>"
    elif source == config.SOURCE_LYRICS:
        source_class = "vv-source-lyrics"
        source_badge = " <span class='vv-source-lyrics'>(lyrics)</span>"
    else:
        source_badge = ""

    title = item.get("title", "Untitled")
    author = item.get("author", "Unknown")
    text = item.get("text", "")

    # Build extra metadata
    extra_info = _build_extra_info(item, source)

    st.markdown(f"""
    <div class="vv-card">
        <div class="vv-card-title">{_escape_html(str(title))}{source_badge}</div>
        <div class="vv-card-author">by {_escape_html(str(author))}</div>
        {f'<div class="vv-card-meta">{extra_info}</div>' if extra_info else ''}
        <div class="vv-card-text">{_escape_html(str(text))}</div>
    </div>
    """, unsafe_allow_html=True)


def _build_extra_info(item: pd.Series, source: str) -> str:
    """Build extra metadata string for item card."""
    parts = []

    if source == "tweets":
        if pd.notna(item.get("likes")):
            parts.append(f"❤️ {int(item['likes']):,}")
        if pd.notna(item.get("retweets")):
            parts.append(f"🔄 {int(item['retweets']):,}")
        hashtags = item.get("hashtags")
        if hashtags and isinstance(hashtags, list):
            parts.append("#" + " #".join(str(h) for h in hashtags[:5]))
    elif source == "lyrics":
        if pd.notna(item.get("genre")):
            parts.append(f"🎵 {item['genre']}")
        if pd.notna(item.get("year")):
            parts.append(f"📅 {int(item['year'])}")

    return " ".join(parts)


def render_neighbors_list(neighbors: pd.DataFrame, title: str = "Similar Items") -> None:
    """Render a list of neighbor items."""
    st.markdown(f"### {title}")

    for idx, (_, neighbor) in enumerate(neighbors.iterrows()):
        similarity = neighbor.get("similarity", 0)
        title_text = str(neighbor.get("title", neighbor.get("text", "Untitled")))[:50]
        if len(str(neighbor.get("title", ""))) > 50:
            title_text += "..."

        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                f"📄 {title_text}",
                key=f"neighbor_{idx}_{neighbor['id']}",
                width="stretch"
            ):
                AppState.set_selected_item(neighbor["id"])
                st.rerun()

        with col2:
            st.markdown(
                f"<span class='vv-badge'>{similarity:.3f}</span>",
                unsafe_allow_html=True
            )


def render_getting_started() -> None:
    """Render getting started guide."""
    current_dataset = st.session_state.current_dataset
    dataset_label = config.AVAILABLE_DATASETS.get(current_dataset, {}).get("label", current_dataset)

    st.markdown(f"""
    ### Getting Started

    **Current dataset:** {dataset_label}

    **Browse:** Select an item from the sidebar dropdown

    **Search:** Enter any text above to find similar items

    **Explore:** Use lasso selection on the plot to zoom into clusters

    **Zoom:** Select points with the lasso tool, then click "Zoom Into Selection"

    **3D View:** Toggle in the sidebar to see clusters in 3D space

    The visualization shows all items projected into 2D/3D space using UMAP.
    Items that are semantically similar appear close together.
    """)


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
