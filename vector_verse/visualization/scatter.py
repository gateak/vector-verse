"""
Interactive scatter plot visualization for embedding space.
Uses Plotly for interactivity.
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config


class ScatterPlotBuilder:
    """
    Builds interactive Plotly scatter plots for embedding visualization.
    
    Features:
    - Different markers for dataset vs custom items
    - Highlight selected item and neighbors
    - Add query projection with distinct marker
    """
    
    # Color palette (distinctive, accessible)
    COLORS = {
        "poetry": "#6366f1",      # Indigo
        "custom": "#f59e0b",      # Amber
        "query": "#ef4444",       # Red
        "selected": "#10b981",    # Emerald
        "neighbor": "#8b5cf6",    # Purple
        "default": "#94a3b8",     # Slate
    }
    
    # Marker settings
    MARKERS = {
        "poetry": dict(symbol="circle", size=6, opacity=0.6),
        "custom": dict(symbol="diamond", size=10, opacity=0.9),
        "query": dict(symbol="star", size=16, opacity=1.0),
        "selected": dict(symbol="circle", size=14, opacity=1.0),
        "neighbor": dict(symbol="circle", size=10, opacity=0.8),
    }
    
    def __init__(
        self,
        height: int = config.PLOT_HEIGHT,
        width: int = config.PLOT_WIDTH
    ):
        """
        Initialize the scatter plot builder.
        
        Args:
            height: Plot height in pixels
            width: Plot width in pixels
        """
        self.height = height
        self.width = width
    
    def build(
        self,
        df: pd.DataFrame,
        coords: np.ndarray,
        selected_id: Optional[str] = None,
        neighbor_ids: Optional[list[str]] = None,
        query_coords: Optional[np.ndarray] = None,
        query_text: Optional[str] = None
    ) -> go.Figure:
        """
        Build an interactive scatter plot.
        
        Args:
            df: DataFrame with item metadata (must have 'id', 'title', 'author', 'source')
            coords: Array of shape (n, 2) with UMAP coordinates
            selected_id: ID of currently selected item (optional)
            neighbor_ids: List of neighbor IDs to highlight (optional)
            query_coords: 2D coordinates of search query (optional)
            query_text: Text of search query for tooltip (optional)
            
        Returns:
            Plotly Figure object
        """
        # Ensure we have coordinate columns
        df = df.copy()
        df["x"] = coords[:, 0]
        df["y"] = coords[:, 1]
        
        # Initialize figure
        fig = go.Figure()
        
        # Track which items to exclude from base layer
        special_ids = set()
        if selected_id:
            special_ids.add(selected_id)
        if neighbor_ids:
            special_ids.update(neighbor_ids)
        
        # Add base layer for each source type
        for source in df["source"].unique():
            source_df = df[(df["source"] == source) & (~df["id"].isin(special_ids))]
            
            if source_df.empty:
                continue
            
            marker_settings = self.MARKERS.get(source, self.MARKERS["poetry"])
            color = self.COLORS.get(source, self.COLORS["default"])
            
            fig.add_trace(go.Scatter(
                x=source_df["x"],
                y=source_df["y"],
                mode="markers",
                marker=dict(
                    color=color,
                    **marker_settings
                ),
                text=self._build_hover_text(source_df),
                hovertemplate="%{text}<extra></extra>",
                name=source.title(),
                customdata=source_df["id"].values
            ))
        
        # Add neighbor highlights (before selected, so selected is on top)
        if neighbor_ids:
            neighbor_df = df[df["id"].isin(neighbor_ids)]
            if not neighbor_df.empty:
                fig.add_trace(go.Scatter(
                    x=neighbor_df["x"],
                    y=neighbor_df["y"],
                    mode="markers",
                    marker=dict(
                        color=self.COLORS["neighbor"],
                        line=dict(color="white", width=2),
                        **self.MARKERS["neighbor"]
                    ),
                    text=self._build_hover_text(neighbor_df),
                    hovertemplate="%{text}<extra></extra>",
                    name="Similar",
                    customdata=neighbor_df["id"].values
                ))
        
        # Add selected item highlight
        if selected_id:
            selected_df = df[df["id"] == selected_id]
            if not selected_df.empty:
                fig.add_trace(go.Scatter(
                    x=selected_df["x"],
                    y=selected_df["y"],
                    mode="markers",
                    marker=dict(
                        color=self.COLORS["selected"],
                        line=dict(color="white", width=2),
                        **self.MARKERS["selected"]
                    ),
                    text=self._build_hover_text(selected_df),
                    hovertemplate="%{text}<extra></extra>",
                    name="Selected",
                    customdata=selected_df["id"].values
                ))
        
        # Add query point
        if query_coords is not None:
            query_text_display = (query_text[:100] + "...") if query_text and len(query_text) > 100 else query_text
            fig.add_trace(go.Scatter(
                x=[query_coords[0]],
                y=[query_coords[1]],
                mode="markers",
                marker=dict(
                    color=self.COLORS["query"],
                    line=dict(color="white", width=2),
                    **self.MARKERS["query"]
                ),
                text=[f"<b>Search Query</b><br>{query_text_display}" if query_text else "Search Query"],
                hovertemplate="%{text}<extra></extra>",
                name="Query"
            ))
        
        # Update layout
        fig.update_layout(
            height=self.height,
            width=self.width,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17,17,17,0.8)",
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                title=""
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                title=""
            ),
            hovermode="closest"
        )
        
        return fig
    
    def _build_hover_text(self, df: pd.DataFrame) -> list[str]:
        """Build hover text for items."""
        texts = []
        for _, row in df.iterrows():
            title = row.get("title", "Untitled")
            author = row.get("author", "Unknown")
            source = row.get("source", "")
            
            # Truncate long titles
            if len(title) > 60:
                title = title[:60] + "..."
            
            text = f"<b>{title}</b><br>{author}"
            if source == config.SOURCE_CUSTOM:
                text += "<br><i>(custom)</i>"
            
            texts.append(text)
        
        return texts
    
    def get_color_for_source(self, source: str) -> str:
        """Get the color associated with a source type."""
        return self.COLORS.get(source, self.COLORS["default"])
