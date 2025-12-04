"""
Core interfaces for ARA AI system
Defines abstract base classes for all major components
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Callable, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum


class AssetType(Enum):
    """Asset type enumeration"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"


class IDataProvider(ABC):
    """
    Abstract interface for data providers
    All data sources must implement this interface
    """
    
    @abstractmethod
    async def fetch_historical(
        self, 
        symbol: str, 
        period: str = "2y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            symbol: Asset symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max)
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo)
            
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        pass
    
    @abstractmethod
    async def fetch_realtime(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch real-time price data
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dict with current price, volume, timestamp, etc.
        """
        pass
    
    @abstractmethod
    async def stream_data(
        self, 
        symbol: str, 
        callback: Callable[[Dict], None]
    ) -> None:
        """
        Stream real-time data via WebSocket
        
        Args:
            symbol: Asset symbol
            callback: Function to call with each data update
        """
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """
        Return list of supported symbols
        
        Returns:
            List of symbol strings
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider name"""
        pass
    
    @abstractmethod
    def get_asset_type(self) -> AssetType:
        """Return asset type this provider handles"""
        pass


class IMLModel(ABC):
    """
    Abstract interface for ML models
    All prediction models must implement this interface
    """
    
    @abstractmethod
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        validation_split: float = 0.2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            validation_split: Fraction of data for validation
            **kwargs: Additional training parameters
            
        Returns:
            Dict with training metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        pass
    
    @abstractmethod
    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Generate prediction explanations
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Dict with feature importance, SHAP values, etc.
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model to disk
        
        Args:
            path: Path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load model from disk
        
        Args:
            path: Path to load model from
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dict with model name, version, parameters, etc.
        """
        pass


class IFeatureEngine(ABC):
    """
    Abstract interface for feature engineering
    All feature calculators must implement this interface
    """
    
    @abstractmethod
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated features
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Return list of feature names
        
        Returns:
            List of feature name strings
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance scores
        
        Returns:
            Dict mapping feature names to importance scores
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        pass
