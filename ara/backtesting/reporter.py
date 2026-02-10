"""
Backtest reporting and visualization.

This module provides comprehensive reporting capabilities including:
- Equity curve generation
- Monthly/yearly return tables
- Trade statistics
- Regime-specific performance
- Model comparison charts
- PDF report generation
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ara.backtesting.metrics import MetricsResult


class BacktestReporter:
    """Generate comprehensive backtest reports and visualizations."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize backtest reporter.

        Args:
            output_dir: Directory for saving reports and charts
        """
        self.output_dir = output_dir or Path("backtest_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        metrics: MetricsResult,
        returns: np.ndarray,
        dates: List[datetime],
        predictions: np.ndarray,
        actuals: np.ndarray,
        symbol: str,
        regime_performance: Optional[Dict[str, Dict[str, float]]] = None,
        model_comparison: Optional[Dict[str, MetricsResult]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive backtest report.

        Args:
            metrics: Performance metrics
            returns: Strategy returns
            dates: Dates for each return
            predictions: Predicted values
            actuals: Actual values
            symbol: Asset symbol
            regime_performance: Optional regime-specific metrics
            model_comparison: Optional model comparison data

        Returns:
            Dictionary containing all report data
        """
        report = {
            "symbol": symbol,
            "generated_at": datetime.now().isoformat(),
            "metrics": metrics.to_dict(),
            "equity_curve": self._generate_equity_curve_data(returns, dates),
            "monthly_returns": self._calculate_monthly_returns(returns, dates),
            "yearly_returns": self._calculate_yearly_returns(returns, dates),
            "trade_statistics": self._generate_trade_statistics(predictions, actuals, returns),
            "drawdown_analysis": self._analyze_drawdowns(returns, dates),
        }

        if regime_performance:
            report["regime_performance"] = regime_performance

        if model_comparison:
            report["model_comparison"] = self._format_model_comparison(model_comparison)

        return report

    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """
        Save report to JSON file.

        Args:
            report: Report data
            filename: Optional filename (default: backtest_report_TIMESTAMP.json)

        Returns:
            Path to saved report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_report_{timestamp}.json"

        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return filepath

    def _generate_equity_curve_data(
        self, returns: np.ndarray, dates: List[datetime]
    ) -> Dict[str, List]:
        """Generate equity curve data."""
        cumulative_returns = np.cumprod(1 + returns)

        # Calculate drawdown
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        return {
            "dates": [d.isoformat() for d in dates],
            "equity": cumulative_returns.tolist(),
            "drawdown": drawdown.tolist(),
        }

    def _calculate_monthly_returns(
        self, returns: np.ndarray, dates: List[datetime]
    ) -> Dict[str, Any]:
        """Calculate monthly return table."""
        df = pd.DataFrame({"date": dates, "return": returns})
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        monthly = (
            df.groupby(["year", "month"])["return"]
            .apply(lambda x: (1 + x).prod() - 1)
            .reset_index()
        )

        # Pivot to create table
        pivot = monthly.pivot(index="year", columns="month", values="return")

        # Add yearly totals
        pivot["Year"] = pivot.apply(lambda x: (1 + x.dropna()).prod() - 1, axis=1)

        return {
            "data": pivot.to_dict(),
            "summary": {
                "best_month": float(monthly["return"].max()),
                "worst_month": float(monthly["return"].min()),
                "avg_month": float(monthly["return"].mean()),
                "positive_months": int((monthly["return"] > 0).sum()),
                "negative_months": int((monthly["return"] < 0).sum()),
            },
        }

    def _calculate_yearly_returns(
        self, returns: np.ndarray, dates: List[datetime]
    ) -> Dict[str, Any]:
        """Calculate yearly return table."""
        df = pd.DataFrame({"date": dates, "return": returns})
        df["year"] = df["date"].dt.year

        yearly = df.groupby("year")["return"].apply(lambda x: (1 + x).prod() - 1).reset_index()

        return {
            "data": yearly.to_dict("records"),
            "summary": {
                "best_year": float(yearly["return"].max()),
                "worst_year": float(yearly["return"].min()),
                "avg_year": float(yearly["return"].mean()),
                "positive_years": int((yearly["return"] > 0).sum()),
                "negative_years": int((yearly["return"] < 0).sum()),
            },
        }

    def _generate_trade_statistics(
        self, predictions: np.ndarray, actuals: np.ndarray, returns: np.ndarray
    ) -> Dict[str, Any]:
        """Generate detailed trade statistics."""
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]

        return {
            "total_trades": len(returns),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "avg_win": (float(np.mean(winning_trades)) if len(winning_trades) > 0 else 0.0),
            "avg_loss": (float(np.mean(losing_trades)) if len(losing_trades) > 0 else 0.0),
            "largest_win": (float(np.max(winning_trades)) if len(winning_trades) > 0 else 0.0),
            "largest_loss": (float(np.min(losing_trades)) if len(losing_trades) > 0 else 0.0),
            "avg_trade": float(np.mean(returns)),
            "median_trade": float(np.median(returns)),
            "std_trade": float(np.std(returns)),
        }

    def _analyze_drawdowns(self, returns: np.ndarray, dates: List[datetime]) -> Dict[str, Any]:
        """Analyze drawdown periods."""
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []

        if np.any(in_drawdown):
            start_idx = None
            for i, is_dd in enumerate(in_drawdown):
                if is_dd and start_idx is None:
                    start_idx = i
                elif not is_dd and start_idx is not None:
                    drawdown_periods.append(
                        {
                            "start": dates[start_idx].isoformat(),
                            "end": dates[i - 1].isoformat(),
                            "duration_days": (dates[i - 1] - dates[start_idx]).days,
                            "depth": float(np.min(drawdown[start_idx:i])),
                        }
                    )
                    start_idx = None

            # Handle ongoing drawdown
            if start_idx is not None:
                drawdown_periods.append(
                    {
                        "start": dates[start_idx].isoformat(),
                        "end": dates[-1].isoformat(),
                        "duration_days": (dates[-1] - dates[start_idx]).days,
                        "depth": float(np.min(drawdown[start_idx:])),
                    }
                )

        return {
            "max_drawdown": float(np.min(drawdown)),
            "avg_drawdown": (
                float(np.mean(drawdown[drawdown < 0])) if np.any(drawdown < 0) else 0.0
            ),
            "num_drawdowns": len(drawdown_periods),
            "drawdown_periods": drawdown_periods[:5],  # Top 5 worst
        }

    def _format_model_comparison(
        self, model_comparison: Dict[str, MetricsResult]
    ) -> Dict[str, Any]:
        """Format model comparison data."""
        comparison = {}
        for model_name, metrics in model_comparison.items():
            comparison[model_name] = {
                "accuracy": metrics.accuracy,
                "directional_accuracy": metrics.directional_accuracy,
                "sharpe_ratio": metrics.sharpe_ratio,
                "max_drawdown": metrics.max_drawdown,
                "total_return": metrics.total_return,
            }
        return comparison

    def plot_equity_curve(
        self,
        returns: np.ndarray,
        dates: List[datetime],
        symbol: str,
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Plot equity curve with drawdowns.

        Args:
            returns: Strategy returns
            dates: Dates for each return
            symbol: Asset symbol
            save_path: Optional path to save plot

        Returns:
            Path to saved plot if save_path provided
        """
        if not PLOTLY_AVAILABLE and not MATPLOTLIB_AVAILABLE:
            print("Warning: Neither plotly nor matplotlib available for plotting")
            return None

        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max

        if PLOTLY_AVAILABLE:
            return self._plot_equity_curve_plotly(
                dates, cumulative_returns, drawdown, symbol, save_path
            )
        else:
            return self._plot_equity_curve_matplotlib(
                dates, cumulative_returns, drawdown, symbol, save_path
            )

    def _plot_equity_curve_plotly(
        self,
        dates: List[datetime],
        equity: np.ndarray,
        drawdown: np.ndarray,
        symbol: str,
        save_path: Optional[Path],
    ) -> Optional[Path]:
        """Plot equity curve using Plotly."""
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
                y=equity,
                mode="lines",
                name="Equity",
                line=dict(color="blue", width=2),
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
                line=dict(color="red", width=1),
            ),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        fig.update_layout(title=f"Backtest Results - {symbol}", height=800, showlegend=True)

        if save_path:
            fig.write_html(str(save_path))
            return save_path
        else:
            fig.show()
            return None

    def _plot_equity_curve_matplotlib(
        self,
        dates: List[datetime],
        equity: np.ndarray,
        drawdown: np.ndarray,
        symbol: str,
        save_path: Optional[Path],
    ) -> Optional[Path]:
        """Plot equity curve using Matplotlib."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Equity curve
        ax1.plot(dates, equity, "b-", linewidth=2, label="Equity")
        ax1.set_ylabel("Cumulative Return")
        ax1.set_title(f"Backtest Results - {symbol}")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Drawdown
        ax2.fill_between(dates, drawdown * 100, 0, color="red", alpha=0.3)
        ax2.plot(dates, drawdown * 100, "r-", linewidth=1)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            plt.show()
            return None

    def plot_monthly_returns_heatmap(
        self,
        returns: np.ndarray,
        dates: List[datetime],
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Plot monthly returns as heatmap.

        Args:
            returns: Strategy returns
            dates: Dates for each return
            save_path: Optional path to save plot

        Returns:
            Path to saved plot if save_path provided
        """
        if not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available for heatmap plotting")
            return None

        # Calculate monthly returns
        df = pd.DataFrame({"date": dates, "return": returns})
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        monthly = (
            df.groupby(["year", "month"])["return"]
            .apply(lambda x: (1 + x).prod() - 1)
            .reset_index()
        )

        pivot = monthly.pivot(index="year", columns="month", values="return")

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values * 100,
                x=[
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
                ],
                y=pivot.index,
                colorscale="RdYlGn",
                zmid=0,
                text=np.round(pivot.values * 100, 2),
                texttemplate="%{text}%",
                textfont={"size": 10},
                colorbar=dict(title="Return (%)"),
            )
        )

        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            height=400 + len(pivot) * 30,
        )

        if save_path:
            fig.write_html(str(save_path))
            return save_path
        else:
            fig.show()
            return None
