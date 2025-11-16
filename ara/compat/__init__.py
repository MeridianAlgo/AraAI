"""
Backward Compatibility Layer for ARA AI

This module provides compatibility wrappers for the old MeridianAlgo API,
allowing existing code to work with the new ARA architecture.

Deprecated: This compatibility layer will be removed in version 5.0.0
Please migrate to the new ara.api module.
"""

from .wrappers import (
    AraAI,
    StockPredictor,
    predict_stock,
    analyze_stock
)
from .migration import ModelMigrator, DataMigrator
from .deprecation import deprecated, DeprecationLevel

__all__ = [
    'AraAI',
    'StockPredictor',
    'predict_stock',
    'analyze_stock',
    'ModelMigrator',
    'DataMigrator',
    'deprecated',
    'DeprecationLevel'
]

# Version info
__version__ = "4.0.0"
__compat_version__ = "3.1.1"  # Compatible with MeridianAlgo 3.1.1
