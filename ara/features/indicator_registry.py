"""
Indicator Registry for managing all technical indicators.

This module provides a centralized registry for all technical indicators,
allowing dynamic registration and retrieval of indicator functions.
"""

from typing import Dict, Callable, List, Any, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class IndicatorMetadata:
    """Metadata for a registered indicator."""
    name: str
    category: str  # trend, momentum, volatility, volume, pattern
    description: str
    parameters: Dict[str, Any]
    required_columns: List[str]  # e.g., ['close'], ['high', 'low', 'close']
    output_columns: List[str]  # Names of columns this indicator produces


class IndicatorRegistry:
    """
    Central registry for all technical indicators.
    
    Provides a plugin system for indicators with metadata tracking,
    caching support, and multi-timeframe analysis.
    """
    
    def __init__(self):
        self._indicators: Dict[str, Callable] = {}
        self._metadata: Dict[str, IndicatorMetadata] = {}
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_enabled = True
    
    def register(
        self,
        name: str,
        func: Callable,
        category: str,
        description: str,
        parameters: Dict[str, Any],
        required_columns: List[str],
        output_columns: List[str]
    ) -> None:
        """
        Register a new indicator function.
        
        Args:
            name: Unique identifier for the indicator
            func: Callable that calculates the indicator
            category: Category (trend, momentum, volatility, volume, pattern)
            description: Human-readable description
            parameters: Default parameters for the indicator
            required_columns: Required DataFrame columns
            output_columns: Names of output columns
        """
        self._indicators[name] = func
        self._metadata[name] = IndicatorMetadata(
            name=name,
            category=category,
            description=description,
            parameters=parameters,
            required_columns=required_columns,
            output_columns=output_columns
        )
    
    def get(self, name: str) -> Optional[Callable]:
        """Get an indicator function by name."""
        return self._indicators.get(name)
    
    def get_metadata(self, name: str) -> Optional[IndicatorMetadata]:
        """Get metadata for an indicator."""
        return self._metadata.get(name)
    
    def list_indicators(self, category: Optional[str] = None) -> List[str]:
        """
        List all registered indicators, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of indicator names
        """
        if category:
            return [
                name for name, meta in self._metadata.items()
                if meta.category == category
            ]
        return list(self._indicators.keys())
    
    def list_categories(self) -> List[str]:
        """List all indicator categories."""
        return list(set(meta.category for meta in self._metadata.values()))
    
    def calculate(
        self,
        name: str,
        data: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate an indicator with caching support.
        
        Args:
            name: Indicator name
            data: Input DataFrame with OHLCV data
            **kwargs: Override default parameters
            
        Returns:
            DataFrame with indicator columns added
        """
        if name not in self._indicators:
            raise ValueError(f"Indicator '{name}' not registered")
        
        # Generate cache key
        cache_key = self._generate_cache_key(name, data, kwargs)
        
        # Check cache
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Calculate indicator
        func = self._indicators[name]
        result = func(data, **kwargs)
        
        # Cache result
        if self._cache_enabled:
            self._cache[cache_key] = result.copy()
        
        return result
    
    def calculate_multiple(
        self,
        indicators: List[str],
        data: pd.DataFrame,
        params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Calculate multiple indicators efficiently.
        
        Args:
            indicators: List of indicator names
            data: Input DataFrame
            params: Optional dict mapping indicator names to parameters
            
        Returns:
            DataFrame with all indicator columns
        """
        result = data.copy()
        params = params or {}
        
        for indicator_name in indicators:
            indicator_params = params.get(indicator_name, {})
            result = self.calculate(indicator_name, result, **indicator_params)
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the indicator cache."""
        self._cache.clear()
    
    def enable_cache(self, enabled: bool = True) -> None:
        """Enable or disable caching."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()
    
    def _generate_cache_key(
        self,
        name: str,
        data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> str:
        """Generate a cache key for an indicator calculation."""
        # Use data hash and parameters to create unique key
        data_hash = hash(tuple(data.index) + tuple(data.columns))
        params_str = str(sorted(params.items()))
        return f"{name}_{data_hash}_{hash(params_str)}"


# Global registry instance
_global_registry = IndicatorRegistry()


def get_registry() -> IndicatorRegistry:
    """Get the global indicator registry instance."""
    return _global_registry
