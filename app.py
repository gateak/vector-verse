"""
Vector-Verse: Semantic Similarity Explorer
Main Streamlit application.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

from vector_verse.core.vector_store import VectorStore
from vector_verse.visualization.scatter import ScatterPlotBuilder
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
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 15, 35, 0.95);
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
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "selected_item_id" not in st.session_state:
        st.session_state.selected_item_id = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "query_coords" not in st.session_state:
        st.session_state.query_coords = None


init_session_state()


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

@st.cache_resource
def load_vector_store(force_rebuild: bool = False):
    """Load and cache the vector store."""
    vs = VectorStore(force_rebuild=force_rebuild)
    return vs


def initialize_app(force_rebuild: bool = False):
    """Initialize the application with data."""
    progress_container = st.empty()
    
    with progress_container:
        with st.spinner("Initializing Vector-Verse..."):
            progress_text = st.empty()
            
            def update_progress(msg: str):
                progress_text.text(msg)
            
            # Load vector store
            vs = load_vector_store(force_rebuild)
            
            if not st.session_state.initialized or force_rebuild:
                vs.initialize(progress_callback=update_progress)
                st.session_state.vector_store = vs
                st.session_state.initialized = True
            
            progress_text.empty()
    
    progress_container.empty()
    return st.session_state.vector_store


# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

def render_header():
    """Render the app header."""
    st.markdown('<h1 class="main-header">Vector-Verse</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Semantic Similarity Explorer for Text</p>', unsafe_allow_html=True)


def render_sidebar(vs: VectorStore):
    """Render the sidebar controls."""
    with st.sidebar:
        st.markdown("### Settings")
        
        # Cache info
        cache_info = vs.get_cache_info()
        st.markdown(f"**Items:** {vs.n_items:,}")
        if vs.n_custom_items > 0:
            st.markdown(f"**Custom items:** {vs.n_custom_items}")
        
        st.markdown("---")
        
        # Rebuild cache option
        if st.button("🔄 Rebuild Cache", help="Re-embed all items and recompute UMAP"):
            st.session_state.initialized = False
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Browse items
        st.markdown("### Browse Items")
        
        items_df = vs.get_all_items()
        
        # Create display labels for dropdown
        options = ["-- Select an item --"] + [
            f"{row['title'][:50]}{'...' if len(row['title']) > 50 else ''} ({row['author'][:20]})"
            for _, row in items_df.iterrows()
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
        with st.spinner("Searching..."):
            results, _, query_coords = vs.search(query.strip(), k=config.DEFAULT_K_NEIGHBORS)
            st.session_state.search_results = results
            st.session_state.search_query = query.strip()
            st.session_state.query_coords = query_coords
            st.session_state.selected_item_id = None


def render_visualization(vs: VectorStore):
    """Render the UMAP scatter plot."""
    builder = ScatterPlotBuilder()
    
    # Get neighbor IDs if item is selected
    neighbor_ids = None
    if st.session_state.selected_item_id:
        neighbors = vs.get_neighbors(st.session_state.selected_item_id)
        neighbor_ids = neighbors["id"].tolist()
    elif st.session_state.search_results is not None:
        neighbor_ids = st.session_state.search_results["id"].tolist()
    
    # Build plot
    fig = builder.build(
        df=vs.items_df,
        coords=vs.umap_coords,
        selected_id=st.session_state.selected_item_id,
        neighbor_ids=neighbor_ids,
        query_coords=st.session_state.query_coords,
        query_text=st.session_state.search_query if st.session_state.query_coords is not None else None
    )
    
    st.plotly_chart(fig, use_container_width=True, key="umap_plot")


def render_item_details(vs: VectorStore):
    """Render the selected item details panel."""
    if st.session_state.selected_item_id:
        item = vs.get_item(st.session_state.selected_item_id)
        neighbors = vs.get_neighbors(st.session_state.selected_item_id)
        
        render_item_card(item)
        render_neighbors_list(neighbors, "Similar Items")
        
    elif st.session_state.search_results is not None:
        st.markdown("### Search Results")
        st.markdown(f'*Query: "{st.session_state.search_query}"*')
        
        for _, result in st.session_state.search_results.iterrows():
            with st.expander(f"📄 {result['title'][:60]}... ({result['similarity']:.3f})"):
                st.markdown(f"**Author:** {result['author']}")
                st.markdown("---")
                st.markdown(result['text'][:1000] + ("..." if len(result['text']) > 1000 else ""))
                
                if st.button(f"Select", key=f"select_{result['id']}"):
                    st.session_state.selected_item_id = result['id']
                    st.rerun()
    else:
        st.markdown("### Getting Started")
        st.markdown("""
        **Browse:** Select an item from the sidebar dropdown
        
        **Search:** Enter any text above to find similar items
        
        **Explore:** Click on points in the scatter plot to discover connections
        
        The visualization shows all items projected into 2D space using UMAP.
        Items that are semantically similar appear close together.
        """)
        
        # Show legend
        st.markdown("#### Legend")
        st.markdown("""
        - **Blue circles:** Poetry dataset items
        - **Orange diamonds:** Your custom items  
        - **Green:** Selected item
        - **Purple:** Similar items
        - **Red star:** Search query
        """)


def render_item_card(item: pd.Series):
    """Render a single item card."""
    source_badge = ""
    if item.get("source") == config.SOURCE_CUSTOM:
        source_badge = " <span style='color: #f59e0b;'>(custom)</span>"
    
    st.markdown(f"""
    <div class="item-card">
        <div class="item-title">{item['title']}{source_badge}</div>
        <div class="item-author">by {item['author']}</div>
        <div class="item-text">{item['text']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_neighbors_list(neighbors: pd.DataFrame, title: str = "Similar Items"):
    """Render a list of neighbor items."""
    st.markdown(f"### {title}")
    
    for idx, (_, neighbor) in enumerate(neighbors.iterrows()):
        similarity = neighbor.get("similarity", 0)
        title_text = neighbor['title'][:50] + ("..." if len(neighbor['title']) > 50 else "")
        
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
    
    # Check for dataset
    if not config.POETRY_CSV_PATH.exists():
        st.warning(f"""
        **Poetry dataset not found!**
        
        Please download the Poetry Foundation dataset from Kaggle:
        1. Go to: https://www.kaggle.com/datasets/tgdivy/poetry-foundation-poems
        2. Download and extract the CSV file
        3. Place it at: `{config.POETRY_CSV_PATH}`
        
        The app will still work if you add custom items in `{config.CUSTOM_ITEMS_PATH}`
        """)
        
        if not config.CUSTOM_ITEMS_PATH.exists():
            st.error("No data available. Please add a dataset or custom items.")
            st.stop()
    
    # Initialize
    try:
        vs = initialize_app()
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        st.exception(e)
        st.stop()
    
    # Render sidebar
    render_sidebar(vs)
    
    # Main content area
    render_search(vs)
    
    # Two-column layout for visualization and details
    col_viz, col_details = st.columns([3, 2])
    
    with col_viz:
        st.markdown("### Embedding Space")
        render_visualization(vs)
    
    with col_details:
        render_item_details(vs)


if __name__ == "__main__":
    main()
