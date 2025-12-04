"""
Ensemble ML System for MeridianAlgo
"""
from .unified_ml import UnifiedStockML

# Alias for backward compatibility
EnsembleMLSystem = UnifiedStockML
UnifiedMLSystem = UnifiedStockML

__all__ = ['EnsembleMLSystem', 'UnifiedMLSystem', 'UnifiedStockML']
