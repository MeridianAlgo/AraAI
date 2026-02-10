"""
Candlestick chart visualization with technical indicators.

This module provides candlestick charts with overlaid technical indicators.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class CandlestickChart:
    """Create candlestick charts with technical indicators."""

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize candlestick chart.

        Args:
            theme: Plotly theme
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required. Install with: pip install plotly")

        self.theme = theme

    def create_chart(
        self,
        data: pd.DataFrame,
        symbol: str,
        indicators: Optional[Dict[str, pd.Series]] = None,
        volume: bool = True,
        height: int = 800,
    ) -> go.Figure:
        """
        Create candlestick chart with indicators.

        Args:
            data: DataFrame with OHLCV data (columns: date, open, high, low, close, volume)
            symbol: Asset symbol
            indicators: Dictionary of indicator name -> Series
            volume: Whether to show volume subplot
            height: Chart height in pixels

        Returns:
            Plotly figure
        """
        # Determine number of subplots
        num_subplots = 1
        if volume:
            num_subplots += 1

        # Create subplots
        row_heights = [0.7, 0.3] if volume else [1.0]

        fig = make_subplots(
            rows=num_subplots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=([f"{symbol} Price", "Volume"] if volume else [f"{symbol} Price"]),
        )

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=data["date"] if "date" in data.columns else data.index,
                open=data["open"],
                high=data["high"],
                low=data["low"],
                close=data["close"],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # Add indicators
        if indicators:
            for name, series in indicators.items():
                self._add_indicator(fig, data, name, series, row=1, col=1)

        # Add volume
        if volume and "volume" in data.columns:
            colors = [
                "green" if close >= open else "red"
                for close, open in zip(data["close"], data["open"])
            ]

            fig.add_trace(
                go.Bar(
                    x=data["date"] if "date" in data.columns else data.index,
                    y=data["volume"],
                    name="Volume",
                    marker_color=colors,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # Update layout
        fig.update_layout(
            title=f"{symbol} Candlestick Chart",
            template=self.theme,
            xaxis_rangeslider_visible=False,
            height=height,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=num_subplots, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        if volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig

    def _add_indicator(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
        name: str,
        series: pd.Series,
        row: int,
        col: int,
    ):
        """Add technical indicator to chart."""
        x_data = data["date"] if "date" in data.columns else data.index

        # Determine line style based on indicator type
        line_style = self._get_indicator_style(name)

        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=series,
                mode="lines",
                name=name,
                line=line_style,
                opacity=0.8,
            ),
            row=row,
            col=col,
        )

    def _get_indicator_style(self, name: str) -> Dict[str, Any]:
        """Get line style for indicator based on name."""
        name_lower = name.lower()

        # Moving averages
        if "sma" in name_lower or "ema" in name_lower:
            if "20" in name or "50" in name:
                return {"color": "blue", "width": 1.5}
            elif "200" in name:
                return {"color": "purple", "width": 2}
            else:
                return {"color": "orange", "width": 1}

        # Bollinger Bands
        elif "bb" in name_lower or "bollinger" in name_lower:
            if "upper" in name_lower:
                return {"color": "gray", "width": 1, "dash": "dash"}
            elif "lower" in name_lower:
                return {"color": "gray", "width": 1, "dash": "dash"}
            else:
                return {"color": "blue", "width": 1}

        # VWAP
        elif "vwap" in name_lower:
            return {"color": "brown", "width": 2}

        # Support/Resistance
        elif "support" in name_lower or "resistance" in name_lower:
            return {
                "color": "red" if "resistance" in name_lower else "green",
                "width": 1.5,
                "dash": "dot",
            }

        # Default
        else:
            return {"width": 1.5}

    def create_multi_timeframe_chart(
        self, data_dict: Dict[str, pd.DataFrame], symbol: str, height: int = 1000
    ) -> go.Figure:
        """
        Create multi-timeframe candlestick chart.

        Args:
            data_dict: Dictionary of timeframe -> DataFrame
            symbol: Asset symbol
            height: Chart height in pixels

        Returns:
            Plotly figure with multiple timeframes
        """
        num_timeframes = len(data_dict)

        fig = make_subplots(
            rows=num_timeframes,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.05,
            subplot_titles=[f"{symbol} - {tf}" for tf in data_dict.keys()],
        )

        for i, (timeframe, data) in enumerate(data_dict.items(), 1):
            fig.add_trace(
                go.Candlestick(
                    x=data["date"] if "date" in data.columns else data.index,
                    open=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    name=timeframe,
                    increasing_line_color="green",
                    decreasing_line_color="red",
                    showlegend=False,
                ),
                row=i,
                col=1,
            )

            fig.update_yaxes(title_text="Price", row=i, col=1)

        fig.update_layout(
            title=f"{symbol} Multi-Timeframe Analysis",
            template=self.theme,
            xaxis_rangeslider_visible=False,
            height=height,
            hovermode="x unified",
        )

        return fig

    def add_pattern_annotations(
        self, fig: go.Figure, patterns: List[Dict[str, Any]], data: pd.DataFrame
    ) -> go.Figure:
        """
        Add pattern annotations to chart.

        Args:
            fig: Existing figure
            patterns: List of pattern dictionaries with keys:
                     - date: Pattern date
                     - type: Pattern type (e.g., 'Head and Shoulders')
                     - price: Price level
                     - description: Optional description
            data: DataFrame with price data

        Returns:
            Updated figure
        """
        for pattern in patterns:
            fig.add_annotation(
                x=pattern["date"],
                y=pattern["price"],
                text=pattern["type"],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="purple",
                ax=0,
                ay=-40,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="purple",
                borderwidth=2,
            )

        return fig

    def add_support_resistance_lines(
        self, fig: go.Figure, levels: Dict[str, List[float]], data: pd.DataFrame
    ) -> go.Figure:
        """
        Add support and resistance lines.

        Args:
            fig: Existing figure
            levels: Dictionary with 'support' and 'resistance' keys,
                   each containing list of price levels
            data: DataFrame with price data

        Returns:
            Updated figure
        """
        x_data = data["date"] if "date" in data.columns else data.index
        x_min, x_max = x_data.iloc[0], x_data.iloc[-1]

        # Add support lines
        for level in levels.get("support", []):
            fig.add_shape(
                type="line",
                x0=x_min,
                x1=x_max,
                y0=level,
                y1=level,
                line=dict(color="green", width=2, dash="dot"),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=x_max,
                y=level,
                text=f"Support: {level:.2f}",
                showarrow=False,
                xanchor="left",
                bgcolor="rgba(0, 255, 0, 0.2)",
            )

        # Add resistance lines
        for level in levels.get("resistance", []):
            fig.add_shape(
                type="line",
                x0=x_min,
                x1=x_max,
                y0=level,
                y1=level,
                line=dict(color="red", width=2, dash="dot"),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=x_max,
                y=level,
                text=f"Resistance: {level:.2f}",
                showarrow=False,
                xanchor="left",
                bgcolor="rgba(255, 0, 0, 0.2)",
            )

        return fig
