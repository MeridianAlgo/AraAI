"""
Compatibility Wrappers for Old API

Provides backward-compatible interfaces that wrap the new ARA architecture.
All functions and classes in this module are deprecated and will be removed in v5.0.0
"""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

try:
    from ara.api.prediction_engine import PredictionEngine
    from ara.core.interfaces import AssetType

    PREDICTION_ENGINE_AVAILABLE = True
except ImportError:
    # Fallback if prediction engine not available
    PREDICTION_ENGINE_AVAILABLE = False
    PredictionEngine = None
    AssetType = None

from .deprecation import deprecated, DeprecationLevel, get_warning_manager


class AraAI:
    """
    Backward compatibility wrapper for old AraAI class

    Deprecated: Use ara.api.PredictionEngine instead
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize AraAI with backward compatibility

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self._warning_manager = get_warning_manager()

        # Initialize engine if available
        if PREDICTION_ENGINE_AVAILABLE:
            self._engine = PredictionEngine()
        else:
            self._engine = None
            if verbose:
                print(
                    "Warning: New prediction engine not available. Limited functionality."
                )

        # Show migration guide on first use
        if verbose:
            self._warning_manager.show_migration_guide()
        else:
            # Issue warning
            self._warning_manager.warn_once(
                "AraAI.__init__",
                "AraAI class is deprecated. Use ara.api.PredictionEngine instead. "
                "This compatibility layer will be removed in version 5.0.0",
            )

    @deprecated(
        reason="Old API structure, synchronous interface",
        version="4.0.0",
        removal_version="5.0.0",
        alternative="ara.api.PredictionEngine.predict()",
        level=DeprecationLevel.WARNING,
    )
    def predict(
        self,
        symbol: str,
        days: int = 5,
        use_cache: bool = True,
        include_analysis: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Predict stock prices (backward compatible)

        Args:
            symbol: Stock symbol
            days: Number of days to predict
            use_cache: Ignored (caching handled automatically)
            include_analysis: Whether to include explanations

        Returns:
            Prediction results in old format
        """
        if not PREDICTION_ENGINE_AVAILABLE or self._engine is None:
            return {
                "error": "Prediction engine not available",
                "message": "Please ensure ara.api.prediction_engine is properly configured",
                "symbol": symbol,
            }

        try:
            # Run async prediction in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    self._engine.predict(
                        symbol=symbol, days=days, include_explanations=include_analysis
                    )
                )
            finally:
                loop.close()

            # Convert to old format
            return self._convert_to_old_format(result)

        except Exception as e:
            if self.verbose:
                print(f"Error: Prediction failed for {symbol}: {e}")
            return None

    def _convert_to_old_format(self, new_result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert new API format to old format"""
        try:
            # Extract data
            symbol = new_result.get("symbol", "")
            current_price = new_result.get("current_price", 0)
            predictions = new_result.get("predictions", [])
            confidence = new_result.get("confidence", {})
            explanations = new_result.get("explanations", {})
            regime = new_result.get("regime", {})

            # Build old format
            old_format = {
                "symbol": symbol,
                "current_price": current_price,
                "predictions": [],
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "version": new_result.get("model_version", "4.0.0"),
                    "type": "ensemble",
                },
            }

            # Convert predictions
            for pred in predictions:
                old_format["predictions"].append(
                    {
                        "day": pred.get("day", 0),
                        "date": (
                            pred.get("date", datetime.now()).isoformat()
                            if isinstance(pred.get("date"), datetime)
                            else pred.get("date")
                        ),
                        "predicted_price": pred.get("predicted_price", 0),
                        "change": pred.get("predicted_price", 0) - current_price,
                        "change_pct": pred.get("predicted_return", 0),
                        "confidence": pred.get("confidence", 0.75),
                    }
                )

            # Add optional fields if available
            if explanations:
                old_format["explanations"] = {
                    "top_factors": explanations.get("top_factors", []),
                    "natural_language": explanations.get("natural_language", ""),
                }

            if regime:
                old_format["market_regime"] = {
                    "regime": regime.get("current_regime", "unknown"),
                    "confidence": regime.get("confidence", 0.5),
                }

            if confidence:
                old_format["confidence_score"] = confidence.get("overall", 0.75)

            return old_format

        except Exception as e:
            if self.verbose:
                print(f"Warning: Format conversion failed: {e}")
            return new_result

    @deprecated(
        reason="Old API structure",
        version="4.0.0",
        removal_version="5.0.0",
        alternative="Use predict() with include_analysis=True",
        level=DeprecationLevel.WARNING,
    )
    def predict_with_ai(
        self, symbol: str, days: int = 5, use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Predict with AI analysis (backward compatible)

        This is now equivalent to predict() with include_analysis=True
        """
        return self.predict(symbol, days, use_cache, include_analysis=True)

    @deprecated(
        reason="Old API structure",
        version="4.0.0",
        removal_version="5.0.0",
        alternative="Use ara.data providers directly",
        level=DeprecationLevel.WARNING,
    )
    def analyze_accuracy(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze prediction accuracy (backward compatible)

        Note: This feature is not fully implemented in the new architecture
        """
        return {
            "message": "Accuracy tracking has been redesigned in v4.0",
            "recommendation": "Use ara.backtesting.BacktestEngine for accuracy analysis",
        }

    @deprecated(
        reason="Old API structure",
        version="4.0.0",
        removal_version="5.0.0",
        alternative="Use ara.backtesting.BacktestEngine",
        level=DeprecationLevel.WARNING,
    )
    def validate_predictions(self) -> Dict[str, Any]:
        """
        Validate predictions (backward compatible)

        Note: This feature is not fully implemented in the new architecture
        """
        return {
            "message": "Validation has been redesigned in v4.0",
            "recommendation": "Use ara.backtesting.BacktestEngine for validation",
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "version": "4.0.0",
            "compatibility_mode": True,
            "warning": "Using backward compatibility layer",
            "migration_guide": "https://docs.ara-ai.com/migration-guide",
        }


class StockPredictor:
    """
    Simplified interface for stock prediction (backward compatibility)

    Deprecated: Use ara.api.PredictionEngine instead
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize StockPredictor

        Args:
            verbose: Enable verbose output
        """
        self.ara = AraAI(verbose=verbose)

        # Issue deprecation warning
        get_warning_manager().warn_once(
            "StockPredictor.__init__",
            "StockPredictor class is deprecated. Use ara.api.PredictionEngine instead. "
            "This compatibility layer will be removed in version 5.0.0",
        )

    @deprecated(
        reason="Old API structure",
        version="4.0.0",
        removal_version="5.0.0",
        alternative="ara.api.PredictionEngine.predict()",
        level=DeprecationLevel.WARNING,
    )
    def predict(self, symbol: str, days: int = 5) -> Optional[Dict[str, Any]]:
        """Predict stock prices"""
        return self.ara.predict(symbol, days=days)

    @deprecated(
        reason="Old API structure",
        version="4.0.0",
        removal_version="5.0.0",
        alternative="Use ara.data providers directly",
        level=DeprecationLevel.WARNING,
    )
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze stock with technical indicators"""
        return {
            "message": "Analysis has been redesigned in v4.0",
            "recommendation": "Use ara.features.FeatureCalculator for technical analysis",
        }


# Convenience functions for backward compatibility
@deprecated(
    reason="Old API structure, synchronous interface",
    version="4.0.0",
    removal_version="5.0.0",
    alternative="ara.api.PredictionEngine.predict()",
    level=DeprecationLevel.WARNING,
)
def predict_stock(
    symbol: str, days: int = 5, verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Predict stock prices using Ara AI (backward compatible)

    Args:
        symbol: Stock symbol
        days: Number of days to predict
        verbose: Enable verbose output

    Returns:
        Prediction results in old format
    """
    ara = AraAI(verbose=verbose)
    return ara.predict(symbol, days=days)


@deprecated(
    reason="Old API structure",
    version="4.0.0",
    removal_version="5.0.0",
    alternative="Use ara.data providers and ara.features.FeatureCalculator",
    level=DeprecationLevel.WARNING,
)
def analyze_stock(symbol: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze stock with technical indicators (backward compatible)

    Args:
        symbol: Stock symbol
        verbose: Enable verbose output

    Returns:
        Analysis results
    """
    return {
        "message": "Analysis has been redesigned in v4.0",
        "recommendation": "Use ara.features.FeatureCalculator for technical analysis",
        "symbol": symbol,
    }
