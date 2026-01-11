"""
Chart factory for creating various types of visualizations.

This module provides a unified interface for creating different chart types.
"""

from typing import Optional, List
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Install with: pip install plotly")


class ChartFactory:
    """Factory for creating various chart types."""

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize chart factory.

        Args:
            theme: Plotly theme (plotly, plotly_white, plotly_dark, etc.)
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is required for visualization. Install with: pip install plotly"
            )

        self.theme = theme
        self.default_config = {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }

    def create_line_chart(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        colors: Optional[List[str]] = None,
        show_legend: bool = True,
    ) -> go.Figure:
        """
        Create a line chart.

        Args:
            data: DataFrame with data
            x_col: Column name for x-axis
            y_cols: List of column names for y-axis
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            colors: Optional list of colors for each line
            show_legend: Whether to show legend

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for i, col in enumerate(y_cols):
            color = colors[i] if colors and i < len(colors) else None
            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data[col],
                    mode="lines",
                    name=col,
                    line=dict(color=color) if color else None,
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.theme,
            showlegend=show_legend,
            hovermode="x unified",
        )

        return fig

    def create_area_chart(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        fill_color: str = "rgba(0, 100, 200, 0.3)",
        line_color: str = "rgb(0, 100, 200)",
    ) -> go.Figure:
        """
        Create an area chart.

        Args:
            data: DataFrame with data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            fill_color: Fill color (RGBA)
            line_color: Line color (RGB)

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode="lines",
                fill="tozeroy",
                fillcolor=fill_color,
                line=dict(color=line_color, width=2),
                name=y_col,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.theme,
            hovermode="x unified",
        )

        return fig

    def create_bar_chart(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        color: Optional[str] = None,
        orientation: str = "v",
    ) -> go.Figure:
        """
        Create a bar chart.

        Args:
            data: DataFrame with data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            color: Bar color
            orientation: 'v' for vertical, 'h' for horizontal

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=data[x_col] if orientation == "v" else data[y_col],
                y=data[y_col] if orientation == "v" else data[x_col],
                marker_color=color,
                orientation=orientation,
            )
        )

        fig.update_layout(
            title=title, xaxis_title=x_label, yaxis_title=y_label, template=self.theme
        )

        return fig

    def create_scatter_plot(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        color_col: Optional[str] = None,
        size_col: Optional[str] = None,
        text_col: Optional[str] = None,
    ) -> go.Figure:
        """
        Create a scatter plot.

        Args:
            data: DataFrame with data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            color_col: Optional column for color coding
            size_col: Optional column for marker size
            text_col: Optional column for hover text

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        marker_dict = {}
        if color_col:
            marker_dict["color"] = data[color_col]
            marker_dict["colorscale"] = "Viridis"
            marker_dict["showscale"] = True
        if size_col:
            marker_dict["size"] = data[size_col]

        fig.add_trace(
            go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode="markers",
                marker=marker_dict if marker_dict else None,
                text=data[text_col] if text_col else None,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + f"{x_label}: %{{x}}<br>"
                    + f"{y_label}: %{{y}}<extra></extra>"
                    if text_col
                    else None
                ),
            )
        )

        fig.update_layout(
            title=title, xaxis_title=x_label, yaxis_title=y_label, template=self.theme
        )

        return fig

    def create_heatmap(
        self,
        data: np.ndarray,
        x_labels: List[str],
        y_labels: List[str],
        title: str = "",
        colorscale: str = "RdBu_r",
        show_values: bool = True,
        value_format: str = ".2f",
    ) -> go.Figure:
        """
        Create a heatmap.

        Args:
            data: 2D array of values
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            title: Chart title
            colorscale: Plotly colorscale name
            show_values: Whether to show values in cells
            value_format: Format string for values

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        text = None
        if show_values:
            text = [[f"{val:{value_format}}" for val in row] for row in data]

        fig.add_trace(
            go.Heatmap(
                z=data,
                x=x_labels,
                y=y_labels,
                colorscale=colorscale,
                text=text,
                texttemplate="%{text}" if show_values else None,
                textfont={"size": 10},
                hovertemplate="%{y} - %{x}<br>Value: %{z}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            template=self.theme,
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),
        )

        return fig

    def create_subplot_figure(
        self,
        rows: int,
        cols: int,
        subplot_titles: Optional[List[str]] = None,
        row_heights: Optional[List[float]] = None,
        vertical_spacing: float = 0.1,
        horizontal_spacing: float = 0.1,
        shared_xaxes: bool = False,
        shared_yaxes: bool = False,
    ) -> go.Figure:
        """
        Create a figure with subplots.

        Args:
            rows: Number of rows
            cols: Number of columns
            subplot_titles: Optional titles for each subplot
            row_heights: Optional relative heights for rows
            vertical_spacing: Vertical spacing between subplots
            horizontal_spacing: Horizontal spacing between subplots
            shared_xaxes: Whether to share x-axes
            shared_yaxes: Whether to share y-axes

        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            row_heights=row_heights,
            vertical_spacing=vertical_spacing,
            horizontal_spacing=horizontal_spacing,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
        )

        fig.update_layout(template=self.theme)

        return fig
