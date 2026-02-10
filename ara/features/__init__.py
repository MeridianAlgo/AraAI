"""
Feature Engineering Module

This module provides technical indicators, pattern recognition, and feature calculation
for financial time series data.
"""

from ara.features.calculator import IndicatorCalculator
from ara.features.indicator_registry import IndicatorRegistry, get_registry
from ara.features.momentum import MomentumIndicators, register_momentum_indicators
from ara.features.patterns import PatternRecognition
from ara.features.support_resistance import SupportResistance

# Register all indicators
from ara.features.trend import TrendIndicators, register_trend_indicators
from ara.features.volatility import VolatilityIndicators, register_volatility_indicators
from ara.features.volume import VolumeIndicators, register_volume_indicators

register_trend_indicators()
register_momentum_indicators()
register_volatility_indicators()
register_volume_indicators()

__all__ = [
    "IndicatorRegistry",
    "get_registry",
    "IndicatorCalculator",
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    "VolumeIndicators",
    "PatternRecognition",
    "SupportResistance",
]
