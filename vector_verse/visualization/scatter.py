"""
Interactive scatter plot visualization for embedding space.
Uses Plotly for interactivity with lasso selection support.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import config


class ScatterPlotBuilder:
    """
    Builds interactive Plotly scatter plots for embedding visualization.
    
    Features:
    - Different markers for dataset vs custom items
    - Highlight selected item and neighbors
    - Add query projection with distinct marker
    - Color by metadata dimensions (genre, artist, user, etc.)
    - Lasso selection for zoom functionality
    """
    
    # Color palette (distinctive, accessible)
    COLORS = {
        "poetry": "#6366f1",      # Indigo
        "custom": "#f59e0b",      # Amber
        "tweets": "#1da1f2",      # Twitter blue
        "lyrics": "#1db954",      # Spotify green
        "query": "#ef4444",       # Red
        "selected": "#10b981",    # Emerald
        "neighbor": "#8b5cf6",    # Purple
        "default": "#94a3b8",     # Slate
    }
    
    # Categorical color palette for color_by dimensions
    CATEGORICAL_COLORS = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1
    
    # Sequential color palette
    SEQUENTIAL_COLORS = px.colors.sequential.Viridis
    
    # Marker settings
    MARKERS = {
        "poetry": dict(symbol="circle", size=6, opacity=0.6),
        "custom": dict(symbol="diamond", size=10, opacity=0.9),
        "tweets": dict(symbol="circle", size=5, opacity=0.5),
        "lyrics": dict(symbol="circle", size=5, opacity=0.5),
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
        query_text: Optional[str] = None,
        color_by: Optional[str] = None,
        color_type: str = "categorical",
        enable_lasso: bool = True,
        top_n_categories: int = 10,
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
            color_by: Column name to color by (e.g., 'genre', 'user', 'hashtag')
            color_type: 'categorical' or 'sequential'
            enable_lasso: Enable lasso selection tool
            top_n_categories: For categorical coloring, group smaller categories as "Other"
            
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
        
        # Determine coloring strategy
        if color_by and color_by in df.columns:
            fig = self._build_colored_by_field(
                df, special_ids, color_by, color_type, top_n_categories
            )
        else:
            fig = self._build_by_source(df, special_ids)
        
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
        
        # Configure layout
        dragmode = "lasso" if enable_lasso else "pan"
        
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
                bgcolor="rgba(0,0,0,0.5)",
                font=dict(size=10)
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
            hovermode="closest",
            dragmode=dragmode,
            # Lasso selection styling
            newselection=dict(
                line=dict(color="#667eea", width=2, dash="dot")
            ),
            activeselection=dict(
                fillcolor="rgba(102, 126, 234, 0.2)",
                opacity=0.8
            ),
        )
        
        return fig
    
    def _build_by_source(
        self,
        df: pd.DataFrame,
        special_ids: set
    ) -> go.Figure:
        """Build plot colored by source type (original behavior)."""
        fig = go.Figure()
        
        for source in df["source"].unique():
            source_df = df[(df["source"] == source) & (~df["id"].isin(special_ids))]
            
            if source_df.empty:
                continue
            
            marker_settings = self.MARKERS.get(source, self.MARKERS.get("poetry"))
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
        
        return fig
    
    def _build_colored_by_field(
        self,
        df: pd.DataFrame,
        special_ids: set,
        color_by: str,
        color_type: str,
        top_n: int
    ) -> go.Figure:
        """Build plot colored by a metadata field."""
        fig = go.Figure()
        
        # Filter out special items
        plot_df = df[~df["id"].isin(special_ids)].copy()
        
        if plot_df.empty:
            return fig
        
        # Get the first source type for marker settings
        first_source = plot_df["source"].iloc[0] if "source" in plot_df.columns else "poetry"
        marker_settings = self.MARKERS.get(first_source, self.MARKERS.get("poetry")).copy()
        
        if color_type == "categorical":
            fig = self._add_categorical_traces(plot_df, color_by, top_n, marker_settings)
        else:
            fig = self._add_sequential_trace(plot_df, color_by, marker_settings)
        
        return fig
    
    def _add_categorical_traces(
        self,
        df: pd.DataFrame,
        color_by: str,
        top_n: int,
        marker_settings: dict
    ) -> go.Figure:
        """Add traces for categorical coloring (one trace per category)."""
        fig = go.Figure()
        
        # Handle list columns (like hashtags)
        if df[color_by].apply(lambda x: isinstance(x, list)).any():
            # Explode list column and take first item
            df = df.copy()
            df[color_by] = df[color_by].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if not isinstance(x, list) else None)
            )
        
        # Get value counts and identify top N
        value_counts = df[color_by].value_counts()
        top_values = set(value_counts.head(top_n).index)
        
        # Create "Other" category
        df = df.copy()
        df["_color_group"] = df[color_by].apply(
            lambda x: x if x in top_values else "Other" if pd.notna(x) else "Unknown"
        )
        
        # Sort categories: top values by count, then Other, then Unknown
        categories = (
            [v for v in value_counts.index if v in top_values] +
            (["Other"] if (df["_color_group"] == "Other").any() else []) +
            (["Unknown"] if (df["_color_group"] == "Unknown").any() else [])
        )
        
        # Add trace for each category
        for i, category in enumerate(categories):
            cat_df = df[df["_color_group"] == category]
            
            if cat_df.empty:
                continue
            
            # Select color
            if category == "Other":
                color = "#6b7280"  # Gray
            elif category == "Unknown":
                color = "#4b5563"  # Darker gray
            else:
                color = self.CATEGORICAL_COLORS[i % len(self.CATEGORICAL_COLORS)]
            
            # Truncate legend label
            legend_label = str(category)[:20] + ("..." if len(str(category)) > 20 else "")
            
            fig.add_trace(go.Scatter(
                x=cat_df["x"],
                y=cat_df["y"],
                mode="markers",
                marker=dict(
                    color=color,
                    **marker_settings
                ),
                text=self._build_hover_text(cat_df, extra_field=color_by),
                hovertemplate="%{text}<extra></extra>",
                name=legend_label,
                customdata=cat_df["id"].values,
                legendgroup=category,
            ))
        
        return fig
    
    def _add_sequential_trace(
        self,
        df: pd.DataFrame,
        color_by: str,
        marker_settings: dict
    ) -> go.Figure:
        """Add single trace with sequential coloring."""
        fig = go.Figure()
        
        # Convert to numeric for colorscale
        color_values = pd.to_numeric(df[color_by], errors="coerce")
        
        # Handle NaN values
        has_values = ~color_values.isna()
        
        if has_values.any():
            plot_df = df[has_values]
            plot_values = color_values[has_values]
            
            fig.add_trace(go.Scatter(
                x=plot_df["x"],
                y=plot_df["y"],
                mode="markers",
                marker=dict(
                    color=plot_values,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title=color_by.title(),
                        thickness=15,
                        len=0.5,
                        y=0.5
                    ),
                    **marker_settings
                ),
                text=self._build_hover_text(plot_df, extra_field=color_by),
                hovertemplate="%{text}<extra></extra>",
                name=color_by.title(),
                customdata=plot_df["id"].values
            ))
        
        # Add NaN values separately
        if (~has_values).any():
            nan_df = df[~has_values]
            fig.add_trace(go.Scatter(
                x=nan_df["x"],
                y=nan_df["y"],
                mode="markers",
                marker=dict(
                    color="#4b5563",
                    **marker_settings
                ),
                text=self._build_hover_text(nan_df),
                hovertemplate="%{text}<extra></extra>",
                name="Unknown",
                customdata=nan_df["id"].values
            ))
        
        return fig
    
    def _build_hover_text(
        self,
        df: pd.DataFrame,
        extra_field: Optional[str] = None
    ) -> list[str]:
        """Build hover text for items."""
        texts = []
        for _, row in df.iterrows():
            title = row.get("title", "Untitled")
            author = row.get("author", "Unknown")
            source = row.get("source", "")
            
            # Truncate long titles
            if len(str(title)) > 60:
                title = str(title)[:60] + "..."
            
            text = f"<b>{title}</b><br>{author}"
            
            if source == config.SOURCE_CUSTOM:
                text += "<br><i>(custom)</i>"
            
            # Add extra field if specified
            if extra_field and extra_field in row.index:
                value = row[extra_field]
                if pd.notna(value):
                    if isinstance(value, list):
                        value = ", ".join(str(v) for v in value[:3])
                        if len(row[extra_field]) > 3:
                            value += "..."
                    text += f"<br><i>{extra_field}: {value}</i>"
            
            texts.append(text)
        
        return texts
    
    def get_color_for_source(self, source: str) -> str:
        """Get the color associated with a source type."""
        return self.COLORS.get(source, self.COLORS["default"])
    
    @staticmethod
    def get_selected_indices_from_selection(selection_data: dict) -> Optional[list[int]]:
        """
        Extract selected point indices from Plotly selection data.
        
        Args:
            selection_data: The 'selectedData' from Plotly chart
            
        Returns:
            List of selected point indices, or None if no selection
        """
        if not selection_data or "points" not in selection_data:
            return None
        
        indices = []
        for point in selection_data["points"]:
            if "pointIndex" in point:
                indices.append(point["pointIndex"])
            elif "customdata" in point:
                # customdata contains the item ID
                indices.append(point["customdata"])
        
        return indices if indices else None