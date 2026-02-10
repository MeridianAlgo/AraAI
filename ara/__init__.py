"""
ARA AI - World-Class Financial Prediction System
Supports stocks, cryptocurrencies, and forex with advanced ML models
"""

__version__ = "4.0.0"
__author__ = "MeridianAlgo"

# Core interfaces
# CLI entry point
from ara.cli import main
from ara.core.exceptions import (
    APIError,
    AraAIException,
    BacktestError,
    CacheError,
    ConfigurationError,
    DataProviderError,
    FeatureEngineeringError,
    ModelError,
    PortfolioError,
    ValidationError,
)
from ara.core.interfaces import AssetType, IDataProvider, IFeatureEngine, IMLModel

# Currency support
from ara.currency import (
    Currency,
    CurrencyConverter,
    CurrencyPreferenceManager,
    CurrencyRiskAnalyzer,
)
from ara.models.base_model import BaseModel

# Risk management
from ara.risk import PortfolioMetrics, RiskCalculator

__all__ = [
    # Interfaces
    "IDataProvider",
    "IMLModel",
    "IFeatureEngine",
    "AssetType",
    # Base classes
    "BaseModel",
    # Exceptions
    "AraAIException",
    "DataProviderError",
    "ModelError",
    "ValidationError",
    "CacheError",
    "APIError",
    "ConfigurationError",
    "FeatureEngineeringError",
    "BacktestError",
    "PortfolioError",
    # Risk Management
    "RiskCalculator",
    "PortfolioMetrics",
    # Currency Support
    "CurrencyConverter",
    "CurrencyRiskAnalyzer",
    "CurrencyPreferenceManager",
    "Currency",
    # CLI
    "main",
]
