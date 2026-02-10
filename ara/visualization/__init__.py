"""
Advanced visualization system for ARA AI.

This module provides comprehensive visualization capabilities including:
- Interactive charts with Plotly
- Candlestick charts with overlaid indicators
- Prediction visualization with confidence intervals
- Equity curve and drawdown charts
- Correlation heatmaps
- Efficient frontier visualization
- Chart export to multiple formats (PNG, SVG, PDF, HTML)
- Comprehensive PDF reports
- Data export (CSV, Excel, JSON)
"""

from ara.visualization.candlestick import CandlestickChart
from ara.visualization.chart_factory import ChartFactory
from ara.visualization.correlation import CorrelationChart
from ara.visualization.exporter import ChartExporter, ReportGenerator
from ara.visualization.portfolio import PortfolioChart
from ara.visualization.predictions import PredictionChart

__all__ = [
    "ChartFactory",
    "CandlestickChart",
    "PredictionChart",
    "PortfolioChart",
    "CorrelationChart",
    "ChartExporter",
    "ReportGenerator",
]
