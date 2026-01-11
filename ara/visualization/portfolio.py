"""
Portfolio visualization including equity curves and efficient frontier.

This module provides portfolio-related visualizations.
"""

from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class PortfolioChart:
    """Create portfolio visualizations."""

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize portfolio chart.

        Args:
            theme: Plotly theme
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required. Install with: pip install plotly")

        self.theme = theme

    def create_equity_curve(
        self,
        returns: np.ndarray,
        dates: List,
        benchmark_returns: Optional[np.ndarray] = None,
        title: str = "Portfolio Equity Curve",
        height: int = 800,
    ) -> go.Figure:
        """
        Create equity curve with drawdown.

        Args:
            returns: Array of portfolio returns
            dates: List of dates
            benchmark_returns: Optional benchmark returns for comparison
            title: Chart title
            height: Chart height in pixels

        Returns:
            Plotly figure
        """
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)

        # Calculate drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Equity Curve", "Drawdown"),
            row_heights=[0.7, 0.3],
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode="lines",
                name="Portfolio",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        # Benchmark if provided
        if benchmark_returns is not None:
            benchmark_cumulative = np.cumprod(1 + benchmark_returns)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=benchmark_cumulative,
                    mode="lines",
                    name="Benchmark",
                    line=dict(color="gray", width=2, dash="dash"),
                ),
                row=1,
                col=1,
            )

        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown * 100,
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.2)",
                line=dict(color="red", width=1),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        fig.update_layout(
            title=title, template=self.theme, height=height, hovermode="x unified"
        )

        return fig

    def create_efficient_frontier(
        self,
        frontier_points: List[Tuple[float, float]],
        optimal_portfolio: Tuple[float, float],
        individual_assets: Optional[Dict[str, Tuple[float, float]]] = None,
        title: str = "Efficient Frontier",
        height: int = 600,
    ) -> go.Figure:
        """
        Create efficient frontier visualization.

        Args:
            frontier_points: List of (risk, return) tuples for frontier
            optimal_portfolio: (risk, return) tuple for optimal portfolio
            individual_assets: Optional dict of asset_name -> (risk, return)
            title: Chart title
            height: Chart height in pixels

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Efficient frontier
        risks, returns = zip(*frontier_points)
        fig.add_trace(
            go.Scatter(
                x=risks,
                y=returns,
                mode="lines",
                name="Efficient Frontier",
                line=dict(color="blue", width=3),
            )
        )

        # Optimal portfolio
        fig.add_trace(
            go.Scatter(
                x=[optimal_portfolio[0]],
                y=[optimal_portfolio[1]],
                mode="markers",
                name="Optimal Portfolio",
                marker=dict(
                    size=15,
                    color="red",
                    symbol="star",
                    line=dict(color="darkred", width=2),
                ),
            )
        )

        # Individual assets
        if individual_assets:
            asset_risks = [v[0] for v in individual_assets.values()]
            asset_returns = [v[1] for v in individual_assets.values()]
            asset_names = list(individual_assets.keys())

            fig.add_trace(
                go.Scatter(
                    x=asset_risks,
                    y=asset_returns,
                    mode="markers+text",
                    name="Individual Assets",
                    text=asset_names,
                    textposition="top center",
                    marker=dict(size=10, color="green", symbol="circle"),
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return",
            template=self.theme,
            height=height,
            hovermode="closest",
        )

        return fig

    def create_allocation_pie(
        self,
        weights: Dict[str, float],
        title: str = "Portfolio Allocation",
        height: int = 500,
    ) -> go.Figure:
        """
        Create portfolio allocation pie chart.

        Args:
            weights: Dictionary of asset -> weight
            title: Chart title
            height: Chart height in pixels

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Sort by weight
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        assets, values = zip(*sorted_weights)

        fig.add_trace(
            go.Pie(
                labels=assets,
                values=values,
                textinfo="label+percent",
                textposition="auto",
                hovertemplate="<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>",
            )
        )

        fig.update_layout(title=title, template=self.theme, height=height)

        return fig

    def create_risk_contribution_chart(
        self,
        risk_contributions: Dict[str, float],
        title: str = "Risk Contribution by Asset",
        height: int = 500,
    ) -> go.Figure:
        """
        Create risk contribution chart.

        Args:
            risk_contributions: Dictionary of asset -> risk contribution
            title: Chart title
            height: Chart height in pixels

        Returns:
            Plotly figure
        """
        # Sort by contribution
        sorted_contrib = sorted(
            risk_contributions.items(), key=lambda x: x[1], reverse=True
        )
        assets, contributions = zip(*sorted_contrib)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=list(assets),
                y=list(contributions),
                marker_color="steelblue",
                text=[f"{c:.2%}" for c in contributions],
                textposition="outside",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Asset",
            yaxis_title="Risk Contribution",
            template=self.theme,
            height=height,
            yaxis_tickformat=".0%",
        )

        return fig

    def create_rolling_metrics_chart(
        self,
        dates: List,
        metrics: Dict[str, np.ndarray],
        title: str = "Rolling Portfolio Metrics",
        height: int = 800,
    ) -> go.Figure:
        """
        Create chart with rolling portfolio metrics.

        Args:
            dates: List of dates
            metrics: Dictionary of metric_name -> values array
            title: Chart title
            height: Chart height in pixels

        Returns:
            Plotly figure
        """
        num_metrics = len(metrics)

        fig = make_subplots(
            rows=num_metrics,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=list(metrics.keys()),
        )

        colors = ["blue", "green", "red", "purple", "orange"]

        for i, (metric_name, values) in enumerate(metrics.items(), 1):
            color = colors[(i - 1) % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=values,
                    mode="lines",
                    name=metric_name,
                    line=dict(color=color, width=2),
                    showlegend=False,
                ),
                row=i,
                col=1,
            )

            fig.update_yaxes(title_text=metric_name, row=i, col=1)

        fig.update_xaxes(title_text="Date", row=num_metrics, col=1)

        fig.update_layout(
            title=title, template=self.theme, height=height, hovermode="x unified"
        )

        return fig

    def create_monthly_returns_heatmap(
        self,
        returns: np.ndarray,
        dates: List,
        title: str = "Monthly Returns Heatmap",
        height: int = 500,
    ) -> go.Figure:
        """
        Create monthly returns heatmap.

        Args:
            returns: Array of returns
            dates: List of dates
            title: Chart title
            height: Chart height in pixels

        Returns:
            Plotly figure
        """
        # Convert to DataFrame
        df = pd.DataFrame({"date": dates, "return": returns})
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["month"] = pd.to_datetime(df["date"]).dt.month

        # Calculate monthly returns
        monthly = (
            df.groupby(["year", "month"])["return"]
            .apply(lambda x: (1 + x).prod() - 1)
            .reset_index()
        )

        # Pivot to create matrix
        pivot = monthly.pivot(index="year", columns="month", values="return")

        # Create heatmap
        fig = go.Figure()

        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        text = [
            [f"{val*100:.1f}%" if not pd.isna(val) else "" for val in row]
            for row in pivot.values
        ]

        fig.add_trace(
            go.Heatmap(
                z=pivot.values * 100,
                x=month_names,
                y=pivot.index,
                colorscale="RdYlGn",
                zmid=0,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Return (%)"),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Year",
            template=self.theme,
            height=height,
        )

        return fig

    def create_performance_comparison(
        self,
        portfolios: Dict[str, Dict[str, float]],
        title: str = "Portfolio Performance Comparison",
        height: int = 600,
    ) -> go.Figure:
        """
        Create performance comparison chart for multiple portfolios.

        Args:
            portfolios: Dictionary of portfolio_name -> metrics_dict
            title: Chart title
            height: Chart height in pixels

        Returns:
            Plotly figure
        """
        # Extract metrics
        portfolio_names = list(portfolios.keys())
        metrics = list(portfolios[portfolio_names[0]].keys())

        fig = go.Figure()

        for metric in metrics:
            values = [portfolios[p][metric] for p in portfolio_names]

            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=portfolio_names,
                    y=values,
                    text=[f"{v:.2f}" for v in values],
                    textposition="outside",
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Portfolio",
            yaxis_title="Value",
            template=self.theme,
            height=height,
            barmode="group",
        )

        return fig
