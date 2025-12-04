"""
Feature Engineering Module

This module provides technical indicators, pattern recognition, and feature calculation
for financial time series data.
"""

from ara.features.indicator_registry import IndicatorRegistry, get_registry
from ara.features.calculator import IndicatorCalculator
from ara.features.trend import TrendIndicators
from ara.features.momentum import MomentumIndicators
from ara.features.volatility import VolatilityIndicators
from ara.features.volume import VolumeIndicators
from ara.features.patterns import PatternRecognition
from ara.features.support_resistance import SupportResistance

__all__ = [
    'IndicatorRegistry',
    'get_registry',
    'IndicatorCalculator',
    'TrendIndicators',
    'MomentumIndicators',
    'VolatilityIndicators',
    'VolumeIndicators',
    'PatternRecognition',
    'SupportResistance',
]
