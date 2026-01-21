"""
Vector-Verse: Semantic Similarity Explorer
Main Streamlit application entry point.

Run with: streamlit run app.py
"""

# Fix Numba threading conflict with Streamlit reruns
# Must be set before importing any library that uses Numba (e.g., umap)
import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import streamlit as st
from dotenv import load_dotenv

from vector_verse.core.vector_store import VectorStore
from vector_verse.core.zoom_manager import ZoomManager
from vector_verse.cache.manager import CacheManager
from vector_verse.loaders.base import get_loader

from vector_verse.ui import (
    AppState,
    inject_styles,
    render_header,
    sidebar,
    main_view,
    details,
    docs,
)
import config

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Vector-Verse",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Application Setup
# -----------------------------------------------------------------------------


def setup_environment() -> bool:
    """Validate environment and return True if ready."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        st.error("""
        **OpenAI API key not found!**

        Please create a `.env` file with your API key:
        ```
        OPENAI_API_KEY=sk-your-key-here
        ```

        Get your API key at: https://platform.openai.com/api-keys
        """)
        return False

    # Check for any dataset
    for key, cfg in config.AVAILABLE_DATASETS.items():
        try:
            if cfg["data_check"]():
                return True
        except Exception:
            continue

    st.warning("""
    **No datasets found!**

    Add at least one dataset to the data/ folder.
    See README.md for setup instructions.
    """)
    return False


@st.cache_resource(show_spinner=False)
def get_vector_store(dataset_key: str) -> VectorStore:
    """Get or create cached VectorStore."""
    loader = get_loader(dataset_key)
    return VectorStore(dataset_loader=loader)


def get_zoom_manager(vs: VectorStore) -> ZoomManager:
    """Get or create ZoomManager for current session."""
    if st.session_state.zoom_manager is None:
        cache = CacheManager(vs.cache_key)
        st.session_state.zoom_manager = ZoomManager(
            cache_manager=cache,
            full_embeddings=vs.embeddings,
            full_umap_coords=vs.umap_coords
        )
    return st.session_state.zoom_manager


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------


def run_explore_tab() -> None:
    """Run the main exploration interface."""
    # Get vector store
    vs = get_vector_store(st.session_state.current_dataset)

    # Initialize if needed
    if not vs.is_initialized:
        cache = CacheManager(vs.cache_key)
        if cache.is_complete():
            with st.spinner("Loading from cache..."):
                vs.initialize()
        else:
            main_view.render_loading_screen(vs)
            st.stop()

    # Get zoom manager
    zm = get_zoom_manager(vs)

    # Render sidebar
    sidebar.render_sidebar(vs, zm)

    # Main content
    main_view.render_zoom_controls(zm)
    main_view.render_search_bar(vs, zm)
    main_view.render_zoom_button(vs, zm)

    # Two-column layout
    col_viz, col_details = st.columns([3, 2])

    with col_viz:
        st.markdown("### Embedding Space")
        main_view.render_visualization(vs, zm)

    with col_details:
        details.render_item_details(vs, zm)


def main() -> None:
    """Main application entry point."""
    # Initialize
    AppState.init(default_dataset=config.DEFAULT_DATASET)
    inject_styles()
    render_header()

    # Environment check
    if not setup_environment():
        st.stop()

    # Main tabs
    tab_explore, tab_methodology, tab_architecture = st.tabs([
        "Explore", "Methodology", "Architecture"
    ])

    with tab_explore:
        run_explore_tab()

    with tab_methodology:
        docs.render_methodology_tab()

    with tab_architecture:
        docs.render_architecture_tab()


if __name__ == "__main__":
    main()
