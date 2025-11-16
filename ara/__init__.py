"""
ARA AI - World-Class Financial Prediction System
Supports stocks, cryptocurrencies, and forex with advanced ML models
"""

__version__ = "4.0.0"
__author__ = "MeridianAlgo"

# Core interfaces
from ara.core.interfaces import IDataProvider, IMLModel, IFeatureEngine, AssetType
from ara.models.base_model import BaseModel
from ara.core.exceptions import (
    AraAIException,
    DataProviderError,
    ModelError,
    ValidationError,
    CacheError,
    APIError,
    ConfigurationError,
    FeatureEngineeringError,
    BacktestError,
    PortfolioError
)

# Risk management
from ara.risk import RiskCalculator, PortfolioMetrics

# Currency support
from ara.currency import (
    CurrencyConverter,
    CurrencyRiskAnalyzer,
    CurrencyPreferenceManager,
    Currency
)

# CLI entry point
from ara.cli import main

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
