"""
Multi-Asset Correlation Analysis Module

This module provides comprehensive correlation analysis capabilities for cross-asset
trading, including rolling correlations, correlation breakdown detection, lead-lag
relationships, and pairs trading opportunities.
"""

from ara.correlation.analyzer import CorrelationAnalyzer
from ara.correlation.cross_asset import CrossAssetPredictor
from ara.correlation.pairs_trading import PairsTradingAnalyzer

__all__ = [
    'CorrelationAnalyzer',
    'CrossAssetPredictor',
    'PairsTradingAnalyzer',
]
