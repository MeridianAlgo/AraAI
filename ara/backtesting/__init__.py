"""
Backtesting module for ARA AI prediction system.

This module provides comprehensive backtesting capabilities including:
- Walk-forward validation
- Performance metrics calculation
- Reporting and visualization
- Automated model validation
"""

from ara.backtesting.metrics import PerformanceMetrics
from ara.backtesting.engine import BacktestEngine
from ara.backtesting.reporter import BacktestReporter
from ara.backtesting.validator import ModelValidator

__all__ = ["PerformanceMetrics", "BacktestEngine", "BacktestReporter", "ModelValidator"]
