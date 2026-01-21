"""
Theme constants and CSS injection for Vector-Verse.
Centralizes all styling in one place for easy customization.
"""

import streamlit as st
from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Central theme configuration - all colors in one place."""
    # Primary palette (gradient)
    primary_start: str = "#667eea"
    primary_end: str = "#764ba2"

    # Backgrounds
    bg_dark: str = "#0f0f23"
    bg_medium: str = "#1a1a2e"
    bg_light: str = "#16213e"
    bg_card: str = "rgba(30, 30, 46, 0.8)"
    bg_card_hover: str = "rgba(102, 126, 234, 0.2)"

    # Text
    text_primary: str = "#e2e8f0"
    text_secondary: str = "#94a3b8"
    text_muted: str = "#cbd5e1"

    # Accents
    accent_selected: str = "#10b981"
    accent_neighbor: str = "#8b5cf6"
    accent_query: str = "#ef4444"
    accent_custom: str = "#f59e0b"
    accent_twitter: str = "#1da1f2"
    accent_spotify: str = "#1db954"

    # Borders
    border_subtle: str = "rgba(102, 126, 234, 0.3)"
    border_focus: str = "rgba(102, 126, 234, 0.6)"


THEME = Theme()


def get_css() -> str:
    """Generate CSS using theme constants."""
    return f"""
<style>
    /* CSS Custom Properties for theming */
    :root {{
        --vv-primary-start: {THEME.primary_start};
        --vv-primary-end: {THEME.primary_end};
        --vv-bg-dark: {THEME.bg_dark};
        --vv-bg-medium: {THEME.bg_medium};
        --vv-bg-light: {THEME.bg_light};
        --vv-text-primary: {THEME.text_primary};
        --vv-text-secondary: {THEME.text_secondary};
        --vv-text-muted: {THEME.text_muted};
    }}

    /* Main app background - using documented Streamlit selectors */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(135deg, {THEME.bg_dark} 0%, {THEME.bg_medium} 50%, {THEME.bg_light} 100%);
    }}

    [data-testid="stSidebar"] {{
        background: rgba(15, 15, 35, 0.95);
    }}

    /* Header styling */
    .vv-header {{
        font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
        background: linear-gradient(90deg, var(--vv-primary-start) 0%, var(--vv-primary-end) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }}

    .vv-subheader {{
        color: var(--vv-text-secondary);
        font-size: 1rem;
        margin-top: 0.25rem;
        font-weight: 400;
    }}

    /* Item card styling */
    .vv-card {{
        background: {THEME.bg_card};
        border: 1px solid {THEME.border_subtle};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    .vv-card:hover {{
        border-color: {THEME.border_focus};
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.1);
    }}

    .vv-card-title {{
        color: {THEME.text_primary};
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }}

    .vv-card-author {{
        color: {THEME.primary_start};
        font-size: 0.9rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }}

    .vv-card-text {{
        color: {THEME.text_muted};
        font-size: 0.95rem;
        line-height: 1.7;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }}

    .vv-card-meta {{
        color: {THEME.text_secondary};
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
    }}

    /* Source badges */
    .vv-source-custom {{
        color: {THEME.accent_custom};
    }}

    .vv-source-tweets {{
        color: {THEME.accent_twitter};
    }}

    .vv-source-lyrics {{
        color: {THEME.accent_spotify};
    }}

    /* Similarity badge */
    .vv-badge {{
        background: linear-gradient(90deg, var(--vv-primary-start) 0%, var(--vv-primary-end) 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }}

    /* Neighbor list item */
    .vv-neighbor {{
        background: rgba(30, 30, 46, 0.6);
        border-left: 3px solid {THEME.primary_start};
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }}

    .vv-neighbor:hover {{
        background: {THEME.bg_card_hover};
        border-left-color: {THEME.primary_end};
        transform: translateX(4px);
    }}

    /* Zoom breadcrumb */
    .vv-breadcrumb {{
        background: {THEME.bg_card_hover};
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: {THEME.text_primary};
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* Error message styling */
    .vv-error {{
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #fca5a5;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }}

    /* Warning message styling */
    .vv-warning {{
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: #fcd34d;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }}

    /* Info message styling */
    .vv-info {{
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: {THEME.text_primary};
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }}

    /* Tab styling - using documented Streamlit selectors */
    [data-baseweb="tab-list"] {{
        gap: 8px;
        background: transparent;
    }}

    [data-baseweb="tab"] {{
        background-color: {THEME.bg_card};
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: {THEME.text_secondary};
        border: none;
        transition: all 0.2s ease;
    }}

    [data-baseweb="tab"]:hover {{
        background-color: rgba(102, 126, 234, 0.2);
    }}

    [data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(90deg, var(--vv-primary-start) 0%, var(--vv-primary-end) 100%);
        color: white;
    }}

    /* Button improvements */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, var(--vv-primary-start) 0%, var(--vv-primary-end) 100%);
        border: none;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    .stButton > button[kind="primary"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }}

    /* Search input styling */
    [data-testid="stTextInput"] input {{
        background: {THEME.bg_card};
        border: 1px solid {THEME.border_subtle};
        border-radius: 8px;
        color: {THEME.text_primary};
        transition: border-color 0.2s ease;
    }}

    [data-testid="stTextInput"] input:focus {{
        border-color: {THEME.primary_start};
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }}

    /* Select box styling */
    [data-testid="stSelectbox"] > div > div {{
        background: {THEME.bg_card};
        border: 1px solid {THEME.border_subtle};
        border-radius: 8px;
    }}

    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: rgba(30, 30, 46, 0.5);
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb {{
        background: {THEME.primary_start};
        border-radius: 4px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {THEME.primary_end};
    }}

    /* Firefox scrollbar */
    * {{
        scrollbar-width: thin;
        scrollbar-color: {THEME.primary_start} rgba(30, 30, 46, 0.5);
    }}

    /* Expander styling */
    [data-testid="stExpander"] {{
        background: {THEME.bg_card};
        border: 1px solid {THEME.border_subtle};
        border-radius: 8px;
    }}

    /* Metric styling */
    [data-testid="stMetric"] {{
        background: {THEME.bg_card};
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid {THEME.border_subtle};
    }}

    /* Caption styling */
    .vv-caption {{
        color: {THEME.text_secondary};
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }}

    /* Loading animation */
    @keyframes vv-pulse {{
        0%, 100% {{ opacity: 0.6; }}
        50% {{ opacity: 1; }}
    }}

    .vv-loading {{
        animation: vv-pulse 1.5s ease-in-out infinite;
    }}
</style>
"""


def inject_styles() -> None:
    """Inject CSS styles into the Streamlit app."""
    st.markdown(get_css(), unsafe_allow_html=True)


def render_header() -> None:
    """Render the styled application header."""
    st.markdown('<h1 class="vv-header">Vector-Verse</h1>', unsafe_allow_html=True)
    st.markdown('<p class="vv-subheader">Semantic Similarity Explorer</p>', unsafe_allow_html=True)


def render_error(message: str) -> None:
    """Render a styled error message."""
    st.markdown(f'<div class="vv-error">{message}</div>', unsafe_allow_html=True)


def render_warning(message: str) -> None:
    """Render a styled warning message."""
    st.markdown(f'<div class="vv-warning">{message}</div>', unsafe_allow_html=True)


def render_info(message: str) -> None:
    """Render a styled info message."""
    st.markdown(f'<div class="vv-info">{message}</div>', unsafe_allow_html=True)
