"""
Vector-Verse: Semantic Similarity Explorer
Main Streamlit application.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

from vector_verse.core.vector_store import VectorStore
from vector_verse.core.zoom_manager import ZoomManager
from vector_verse.visualization.scatter import ScatterPlotBuilder
from vector_verse.loaders.base import get_loader, list_loaders
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

# Custom CSS for dark, modern aesthetic
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    
    .sub-header {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0;
    }
    
    /* Card styling */
    .item-card {
        background: rgba(30, 30, 46, 0.8);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .item-title {
        color: #e2e8f0;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .item-author {
        color: #667eea;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    .item-text {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Similarity badge */
    .similarity-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Neighbor list */
    .neighbor-item {
        background: rgba(30, 30, 46, 0.6);
        border-left: 3px solid #667eea;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .neighbor-item:hover {
        background: rgba(102, 126, 234, 0.2);
        border-left-color: #764ba2;
    }
    
    /* Zoom breadcrumb */
    .zoom-breadcrumb {
        background: rgba(102, 126, 234, 0.2);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 15, 35, 0.95);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 30, 46, 0.8);
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 46, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------

def init_session_state():
    """Initialize session state variables."""
    if "selected_item_id" not in st.session_state:
        st.session_state.selected_item_id = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "query_coords" not in st.session_state:
        st.session_state.query_coords = None
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = config.DEFAULT_DATASET
    if "zoom_manager" not in st.session_state:
        st.session_state.zoom_manager = None
    if "color_by" not in st.session_state:
        st.session_state.color_by = None
    if "lasso_selection" not in st.session_state:
        st.session_state.lasso_selection = None


init_session_state()


# -----------------------------------------------------------------------------
# Data Loading - Cached to survive refreshes
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_vector_store(dataset_key: str) -> VectorStore:
    """
    Get or create the VectorStore for a specific dataset.
    Cached so it survives page refreshes.
    """
    loader = get_loader(dataset_key)
    vs = VectorStore(dataset_loader=loader)
    return vs


def initialize_store(vs: VectorStore) -> bool:
    """
    Initialize the vector store if needed.
    Returns True if ready, False if still loading.
    """
    if vs.is_initialized:
        return True
    
    # Check if cache exists - if so, load quickly
    from vector_verse.cache.manager import CacheManager
    cache = CacheManager(vs.cache_key)
    
    if cache.exists():
        # Cache exists, load it (fast)
        with st.spinner("Loading from cache..."):
            vs.initialize()
        return True
    else:
        # Need to build - show progress
        return False


def get_or_create_zoom_manager(vs: VectorStore) -> ZoomManager:
    """Get or create ZoomManager for current dataset."""
    if st.session_state.zoom_manager is None:
        from vector_verse.cache.manager import CacheManager
        cache = CacheManager(vs.cache_key)
        st.session_state.zoom_manager = ZoomManager(
            cache_manager=cache,
            full_embeddings=vs.embeddings,
            full_umap_coords=vs.umap_coords
        )
    return st.session_state.zoom_manager


# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

def render_header():
    """Render the app header."""
    st.markdown('<h1 class="main-header">Vector-Verse</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Semantic Similarity Explorer for Text</p>', unsafe_allow_html=True)


def render_loading_screen(vs: VectorStore):
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
        # Parse batch progress if available
        if "batch" in msg.lower():
            try:
                parts = msg.split()
                for i, p in enumerate(parts):
                    if "/" in p:
                        current, total = p.split("/")
                        current = int(current)
                        total = int(total.rstrip("..."))
                        progress_bar.progress(current / total)
                        break
            except:
                pass
    
    try:
        vs.initialize(progress_callback=update_progress)
        progress_bar.progress(1.0)
        status_text.text("Done! Refreshing...")
        st.rerun()
    except Exception as e:
        st.error(f"Error during initialization: {str(e)}")
        st.exception(e)


def render_sidebar(vs: VectorStore):
    """Render the sidebar controls."""
    with st.sidebar:
        # Dataset switcher
        render_dataset_switcher()
        
        st.markdown("---")
        
        # Dataset info
        st.markdown("### Dataset Info")
        st.markdown(f"**Items:** {vs.n_items:,}")
        if vs.n_custom_items > 0:
            st.markdown(f"**Custom items:** {vs.n_custom_items}")
        
        st.markdown("---")
        
        # Color dimension selector
        render_color_selector()
        
        st.markdown("---")
        
        # Rebuild cache option
        if st.button("🔄 Rebuild Cache", help="Re-embed all items and recompute UMAP"):
            vs.clear_cache()
            st.session_state.zoom_manager = None
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Browse items
        st.markdown("### Browse Items")
        
        # Get items for current zoom level
        zm = get_or_create_zoom_manager(vs)
        if zm.is_zoomed:
            items_df = zm.get_subset_items(vs.items_df)
        else:
            items_df = vs.get_all_items()
        
        # Create display labels for dropdown
        display_col = "title" if "title" in items_df.columns else "text"
        author_col = "author" if "author" in items_df.columns else None
        
        options = ["-- Select an item --"] + [
            f"{str(row[display_col])[:50]}{'...' if len(str(row[display_col])) > 50 else ''}"
            + (f" ({str(row[author_col])[:20]})" if author_col and pd.notna(row.get(author_col)) else "")
            for _, row in items_df.head(500).iterrows()
        ]
        
        # Item selector
        selected_idx = st.selectbox(
            "Select item:",
            range(len(options)),
            format_func=lambda x: options[x],
            key="item_selector"
        )
        
        if selected_idx > 0:
            item_id = items_df.iloc[selected_idx - 1]["id"]
            if item_id != st.session_state.selected_item_id:
                st.session_state.selected_item_id = item_id
                st.session_state.search_results = None
                st.session_state.query_coords = None
                st.rerun()


def render_dataset_switcher():
    """Render dataset selector in sidebar."""
    st.markdown("### 📊 Dataset")
    
    # Check which datasets are available
    available = []
    for key, cfg in config.AVAILABLE_DATASETS.items():
        try:
            if cfg["data_check"]():
                available.append((key, cfg["label"], cfg.get("description", "")))
        except Exception:
            pass
    
    if not available:
        st.warning("No datasets found. Add data to the data/ folder.")
        return
    
    current = st.session_state.current_dataset
    
    # Ensure current is valid
    if current not in [k for k, _, _ in available]:
        current = available[0][0]
        st.session_state.current_dataset = current
    
    selected = st.radio(
        "Select dataset:",
        [k for k, _, _ in available],
        format_func=lambda x: dict((k, l) for k, l, _ in available)[x],
        index=[k for k, _, _ in available].index(current),
        key="dataset_radio",
        help="Switch between different text datasets"
    )
    
    if selected != current:
        st.session_state.current_dataset = selected
        st.session_state.selected_item_id = None
        st.session_state.search_results = None
        st.session_state.query_coords = None
        st.session_state.zoom_manager = None
        st.session_state.color_by = None
        st.cache_resource.clear()
        st.rerun()


def render_color_selector():
    """Render color dimension selector."""
    st.markdown("### 🎨 Color By")
    
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


def render_zoom_controls(vs: VectorStore):
    """Render zoom breadcrumb and controls."""
    zm = get_or_create_zoom_manager(vs)
    
    if not zm.is_zoomed:
        return
    
    # Breadcrumb
    st.markdown(f'<div class="zoom-breadcrumb">📍 {zm.breadcrumb}</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            zm.zoom_out()
            st.session_state.selected_item_id = None
            st.session_state.search_results = None
            st.rerun()
    
    with col2:
        if st.button("🏠 Reset", use_container_width=True):
            zm.reset_to_root()
            st.session_state.selected_item_id = None
            st.session_state.search_results = None
            st.rerun()
    
    with col3:
        items_in_view = len(zm.get_current_indices())
        st.markdown(f"*{items_in_view:,} items in view*")


def render_zoom_button(vs: VectorStore):
    """Render zoom-into-selection button if there's a selection."""
    if st.session_state.lasso_selection and len(st.session_state.lasso_selection) > 1:
        n_selected = len(st.session_state.lasso_selection)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"🔍 {n_selected} items selected")
        with col2:
            if st.button("Zoom Into Selection", type="primary", use_container_width=True):
                zm = get_or_create_zoom_manager(vs)
                indices = np.array(st.session_state.lasso_selection)
                
                with st.spinner("Computing zoomed view..."):
                    zm.zoom_into(indices)
                
                st.session_state.lasso_selection = None
                st.session_state.selected_item_id = None
                st.session_state.search_results = None
                st.rerun()
        
        if st.button("Clear Selection", use_container_width=True):
            st.session_state.lasso_selection = None
            st.rerun()


def render_search(vs: VectorStore):
    """Render the search interface."""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Search by meaning",
            placeholder="Enter text in any language to find similar items...",
            key="search_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("Search", type="primary", use_container_width=True)
    
    if search_clicked and query.strip():
        zm = get_or_create_zoom_manager(vs)
        index_mask = zm.get_current_indices() if zm.is_zoomed else None
        
        with st.spinner("Searching..."):
            results, _, query_coords = vs.search(
                query.strip(),
                k=config.DEFAULT_K_NEIGHBORS,
                index_mask=index_mask
            )
            st.session_state.search_results = results
            st.session_state.search_query = query.strip()
            st.session_state.query_coords = query_coords
            st.session_state.selected_item_id = None


def render_visualization(vs: VectorStore):
    """Render the UMAP scatter plot."""
    builder = ScatterPlotBuilder()
    
    # Get current zoom level data
    zm = get_or_create_zoom_manager(vs)
    
    if zm.is_zoomed:
        current_df = zm.get_subset_items(vs.items_df)
        current_coords = zm.get_current_coords()
    else:
        current_df = vs.items_df
        current_coords = vs.umap_coords
    
    # Get neighbor IDs if item is selected
    neighbor_ids = None
    if st.session_state.selected_item_id:
        index_mask = zm.get_current_indices() if zm.is_zoomed else None
        neighbors = vs.get_neighbors(st.session_state.selected_item_id, index_mask=index_mask)
        neighbor_ids = neighbors["id"].tolist()
    elif st.session_state.search_results is not None:
        neighbor_ids = st.session_state.search_results["id"].tolist()
    
    # Determine color settings
    color_by = st.session_state.color_by
    color_type = "categorical"
    
    if color_by in ["decade", "year", "date_bucket"]:
        color_type = "sequential"
    
    # Build plot
    fig = builder.build(
        df=current_df,
        coords=current_coords,
        selected_id=st.session_state.selected_item_id,
        neighbor_ids=neighbor_ids,
        query_coords=st.session_state.query_coords if not zm.is_zoomed else None,
        query_text=st.session_state.search_query if st.session_state.query_coords is not None else None,
        color_by=color_by,
        color_type=color_type,
        enable_lasso=True
    )
    
    # Render with selection callback
    selection = st.plotly_chart(
        fig,
        use_container_width=True,
        key="umap_plot",
        on_select="rerun",
        selection_mode="lasso"
    )
    
    # Handle lasso selection
    if selection and "selection" in selection and selection["selection"].get("points"):
        points = selection["selection"]["points"]
        if points:
            # Extract indices from selection
            selected_indices = []
            for point in points:
                if "point_indices" in point:
                    selected_indices.extend(point["point_indices"])
            
            if selected_indices:
                st.session_state.lasso_selection = list(set(selected_indices))


def render_item_details(vs: VectorStore):
    """Render the selected item details panel."""
    zm = get_or_create_zoom_manager(vs)
    index_mask = zm.get_current_indices() if zm.is_zoomed else None
    
    if st.session_state.selected_item_id:
        item = vs.get_item(st.session_state.selected_item_id)
        neighbors = vs.get_neighbors(st.session_state.selected_item_id, index_mask=index_mask)
        
        render_item_card(item)
        render_neighbors_list(neighbors, "Similar Items")
        
    elif st.session_state.search_results is not None:
        st.markdown("### Search Results")
        st.markdown(f'*Query: "{st.session_state.search_query}"*')
        
        for _, result in st.session_state.search_results.iterrows():
            title = result.get('title', result.get('text', 'Untitled')[:60])
            with st.expander(f"📄 {str(title)[:60]}... ({result['similarity']:.3f})"):
                if 'author' in result:
                    st.markdown(f"**Author:** {result['author']}")
                st.markdown("---")
                text = result.get('text', '')
                st.markdown(text[:1000] + ("..." if len(text) > 1000 else ""))
                
                if st.button(f"Select", key=f"select_{result['id']}"):
                    st.session_state.selected_item_id = result['id']
                    st.rerun()
    else:
        render_getting_started()


def render_getting_started():
    """Render getting started guide."""
    st.markdown("### Getting Started")
    
    current_dataset = st.session_state.current_dataset
    dataset_label = config.AVAILABLE_DATASETS.get(current_dataset, {}).get("label", current_dataset)
    
    st.markdown(f"""
    **Current dataset:** {dataset_label}
    
    **Browse:** Select an item from the sidebar dropdown
    
    **Search:** Enter any text above to find similar items
    
    **Explore:** Use lasso selection on the plot to zoom into clusters
    
    **Zoom:** Select points with the lasso tool, then click "Zoom Into Selection" to re-run UMAP on the subset for finer detail.
    
    The visualization shows all items projected into 2D space using UMAP.
    Items that are semantically similar appear close together.
    """)
    
    # Show legend based on current dataset
    st.markdown("#### Legend")
    if current_dataset == "poetry":
        st.markdown("""
        - **Blue circles:** Poetry items
        - **Orange diamonds:** Custom items  
        - **Green:** Selected item
        - **Purple:** Similar items
        - **Red star:** Search query
        """)
    elif current_dataset == "tweets":
        st.markdown("""
        - **Blue circles:** Tweet items
        - **Green:** Selected item
        - **Purple:** Similar items
        - **Red star:** Search query
        
        *Tip: Use "Color By" in sidebar to visualize by user, hashtag, or date*
        """)
    elif current_dataset == "lyrics":
        st.markdown("""
        - **Green circles:** Lyrics items
        - **Green:** Selected item
        - **Purple:** Similar items
        - **Red star:** Search query
        
        *Tip: Use "Color By" in sidebar to visualize by genre, artist, or decade*
        """)


def render_item_card(item: pd.Series):
    """Render a single item card."""
    source = item.get("source", "")
    source_badge = ""
    
    if source == config.SOURCE_CUSTOM:
        source_badge = " <span style='color: #f59e0b;'>(custom)</span>"
    elif source == config.SOURCE_TWEETS:
        source_badge = " <span style='color: #1da1f2;'>(tweet)</span>"
    elif source == config.SOURCE_LYRICS:
        source_badge = " <span style='color: #1db954;'>(lyrics)</span>"
    
    title = item.get('title', 'Untitled')
    author = item.get('author', 'Unknown')
    text = item.get('text', '')
    
    # Add metadata for tweets/lyrics
    extra_info = ""
    if source == "tweets":
        if pd.notna(item.get('likes')):
            extra_info += f"❤️ {int(item['likes']):,} "
        if pd.notna(item.get('retweets')):
            extra_info += f"🔄 {int(item['retweets']):,}"
        if item.get('hashtags'):
            hashtags = item['hashtags']
            if isinstance(hashtags, list):
                extra_info += f"<br>#{' #'.join(hashtags[:5])}"
    elif source == "lyrics":
        if pd.notna(item.get('genre')):
            extra_info += f"🎵 {item['genre']} "
        if pd.notna(item.get('year')):
            extra_info += f"📅 {int(item['year'])}"
    
    st.markdown(f"""
    <div class="item-card">
        <div class="item-title">{title}{source_badge}</div>
        <div class="item-author">by {author}</div>
        {f'<div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem;">{extra_info}</div>' if extra_info else ''}
        <div class="item-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_neighbors_list(neighbors: pd.DataFrame, title: str = "Similar Items"):
    """Render a list of neighbor items."""
    st.markdown(f"### {title}")
    
    for idx, (_, neighbor) in enumerate(neighbors.iterrows()):
        similarity = neighbor.get("similarity", 0)
        title_text = str(neighbor.get('title', neighbor.get('text', 'Untitled')))[:50]
        if len(str(neighbor.get('title', ''))) > 50:
            title_text += "..."
        
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                f"📄 {title_text}",
                key=f"neighbor_{idx}_{neighbor['id']}",
                use_container_width=True
            ):
                st.session_state.selected_item_id = neighbor['id']
                st.session_state.search_results = None
                st.session_state.query_coords = None
                st.rerun()
        
        with col2:
            st.markdown(f"<span class='similarity-badge'>{similarity:.3f}</span>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Methodology & Architecture Tabs
# -----------------------------------------------------------------------------

def render_methodology_tab():
    """Render the Methodology explanation tab."""
    st.markdown("""
    ## How Vector-Verse Works
    
    ### 🧠 Embeddings: Turning Text into Numbers
    
    Each piece of text is converted into a high-dimensional vector (1536 dimensions) 
    using OpenAI's `text-embedding-3-small` model. These vectors capture **semantic meaning** —
    texts about similar topics will have similar vectors, regardless of the exact words used.
    
    For example:
    - "I love sunny days" and "Bright weather makes me happy" → similar vectors
    - "I love sunny days" and "Database optimization techniques" → very different vectors
    
    ### 🗺️ UMAP: Visualizing High Dimensions
    
    We use **UMAP** (Uniform Manifold Approximation and Projection) to reduce 1536 dimensions 
    down to 2 for visualization. UMAP is chosen because it:
    
    - Preserves **local neighborhoods** (similar items stay close)
    - Maintains **global structure** (clusters remain separated)
    - Works well with high-dimensional embedding spaces
    
    ### 🔍 Similarity Search
    
    When you search or find neighbors, we compute **cosine similarity** between vectors.
    For normalized vectors (which ours are), this simplifies to a dot product:
    
    ```
    similarity = embedding_a · embedding_b
    ```
    
    - **1.0** = Identical meaning
    - **0.7-0.9** = Very similar
    - **0.5-0.7** = Related
    - **< 0.5** = Probably unrelated
    
    ### 🔬 Hierarchical Zoom
    
    When you zoom into a cluster:
    
    1. We take the subset of embeddings you selected
    2. Re-run UMAP on just that subset
    3. This reveals **finer structure** that's hidden at the global scale
    
    Think of it like zooming in on a map — you see more detail at street level than from space.
    
    ### ⚠️ What Clusters Mean (and Don't Mean)
    
    **Clusters suggest thematic groupings**, but they're not definitive categories:
    
    - Items at cluster edges may belong to multiple conceptual groups
    - The 2D projection is a lossy compression — some relationships are lost
    - Different UMAP runs can produce different layouts (but preserve neighborhoods)
    - Cluster size doesn't necessarily indicate importance
    
    **Use clusters as exploration aids**, not ground truth categories.
    """)


def render_architecture_tab():
    """Render the Architecture documentation tab."""
    st.markdown("""
    ## System Architecture
    
    ### 📊 Data Flow Diagram
    
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
    │  • embeddings.npz   (vectors)                                   │
    │  • items.parquet    (metadata)                                  │
    │  • umap_coords.npz  (2D coords)                                 │
    │  • umap_model.pkl   (fitted UMAP)                               │
    │  • zoom_levels/     (subset projections)                        │
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
    
    ### 🔌 Adding a New Dataset Loader
    
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
        "label": "🏷️ Your Dataset",
        "data_check": lambda: (DATA_DIR / "your_data").exists(),
        "color_dimensions": ["category", "author"],
    }
    ```
    
    4. **Export in `__init__.py`**:
    ```python
    from .your_loader import YourLoader
    ```
    
    ### 💾 Caching Strategy
    
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
    
    This means:
    - Adding new datasets never invalidates existing caches
    - Repeated zoom operations are instant
    - Cache can be cleared per-dataset
    
    ### 🔑 Key Classes
    
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


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------

def main():
    """Main application entry point."""
    render_header()
    
    # Check for API key
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("""
        **OpenAI API key not found!**
        
        Please create a `.env` file in the project root with your API key:
        ```
        OPENAI_API_KEY=sk-your-key-here
        ```
        
        Get your API key at: https://platform.openai.com/api-keys
        """)
        st.stop()
    
    # Check for any dataset
    has_any_dataset = False
    for key, cfg in config.AVAILABLE_DATASETS.items():
        try:
            if cfg["data_check"]():
                has_any_dataset = True
                break
        except Exception:
            pass
    
    if not has_any_dataset:
        st.warning("""
        **No datasets found!**
        
        Add at least one dataset:
        
        **Poetry (default):**
        1. Download from: https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems
        2. Place CSV at: `data/PoetryFoundationData.csv`
        
        **Tweets:**
        1. Download from: https://www.kaggle.com/datasets/khalidryder777/500k-chatgpt-tweets-jan-mar-2023
        2. Create folder: `data/tweets/`
        3. Place CSV inside
        
        **Lyrics:**
        1. Download from: https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres
        2. Create folder: `data/lyrics/`
        3. Place CSV(s) inside
        """)
        st.stop()
    
    # Main tabs
    tab_explore, tab_methodology, tab_architecture = st.tabs([
        "🔍 Explore", "📚 Methodology", "🏗️ Architecture"
    ])
    
    with tab_explore:
        # Get cached vector store for current dataset
        current_dataset = st.session_state.current_dataset
        vs = get_vector_store(current_dataset)
        
        # Initialize if needed
        if not vs.is_initialized:
            from vector_verse.cache.manager import CacheManager
            cache = CacheManager(vs.cache_key)
            
            if cache.exists():
                # Quick load from cache
                with st.spinner("Loading from cache..."):
                    vs.initialize()
            else:
                # Need to build - show loading screen
                render_loading_screen(vs)
                st.stop()
        
        # Render sidebar
        render_sidebar(vs)
        
        # Zoom controls
        render_zoom_controls(vs)
        
        # Main content area
        render_search(vs)
        
        # Zoom button if selection exists
        render_zoom_button(vs)
        
        # Two-column layout for visualization and details
        col_viz, col_details = st.columns([3, 2])
        
        with col_viz:
            st.markdown("### Embedding Space")
            render_visualization(vs)
        
        with col_details:
            render_item_details(vs)
    
    with tab_methodology:
        render_methodology_tab()
    
    with tab_architecture:
        render_architecture_tab()


if __name__ == "__main__":
    main()
