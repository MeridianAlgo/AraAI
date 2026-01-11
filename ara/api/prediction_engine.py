"""
Prediction Engine for API
Orchestrates data fetching, feature calculation, and model predictions
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ara.core.interfaces import AssetType
from ara.core.exceptions import DataProviderError, ValidationError
from ara.data.base_provider import BaseDataProvider
from ara.data.crypto_provider import CryptoExchangeProvider
from ara.features.calculator import IndicatorCalculator
from ara.models.ensemble import EnhancedEnsemble
from ara.models.regime_detector import RegimeDetector
from ara.explainability.explanation_generator import ExplanationGenerator


class PredictionEngine:
    """
    Main prediction engine that orchestrates the prediction workflow
    """

    def __init__(self):
        """Initialize prediction engine with providers and models"""
        self.stock_provider = BaseDataProvider()
        self.crypto_provider = CryptoExchangeProvider()
        self.feature_calculator = IndicatorCalculator()
        self.regime_detector = RegimeDetector()
        self.explanation_generator = ExplanationGenerator()

        # Model cache
        self._models: Dict[str, EnhancedEnsemble] = {}
        self._model_version = "4.0.0"

    def _detect_asset_type(self, symbol: str) -> AssetType:
        """Detect asset type from symbol"""
        symbol_upper = symbol.upper()

        # Crypto patterns
        if "-USD" in symbol_upper or "BTC" in symbol_upper or "ETH" in symbol_upper:
            return AssetType.CRYPTO

        # Forex patterns (6 character pairs)
        if len(symbol_upper) == 6 and symbol_upper.isalpha():
            return AssetType.FOREX

        # Default to stock
        return AssetType.STOCK

    def _get_provider(self, asset_type: AssetType):
        """Get appropriate data provider for asset type"""
        if asset_type == AssetType.CRYPTO:
            return self.crypto_provider
        else:
            return self.stock_provider

    async def _fetch_data(self, symbol: str, asset_type: AssetType) -> pd.DataFrame:
        """Fetch historical data for symbol"""
        provider = self._get_provider(asset_type)

        try:
            data = await provider.fetch_historical(symbol, period="2y", interval="1d")

            if data is None or len(data) < 100:
                raise DataProviderError(
                    f"Insufficient data for {symbol}",
                    {"symbol": symbol, "rows": len(data) if data is not None else 0},
                )

            return data
        except Exception as e:
            raise DataProviderError(
                f"Failed to fetch data for {symbol}: {str(e)}",
                {"symbol": symbol, "asset_type": asset_type.value},
            )

    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and features"""
        try:
            features = self.feature_calculator.calculate_features(data)
            return features
        except Exception as e:
            raise ValidationError(
                f"Feature calculation failed: {str(e)}", {"data_shape": data.shape}
            )

    def _get_or_create_model(self, symbol: str) -> EnhancedEnsemble:
        """Get cached model or create new one"""
        if symbol not in self._models:
            self._models[symbol] = EnhancedEnsemble(symbol=symbol)
        return self._models[symbol]

    def _prepare_features(self, features: pd.DataFrame) -> np.ndarray:
        """Prepare features for model input"""
        # Select relevant features and handle missing values
        feature_cols = [
            col
            for col in features.columns
            if col not in ["Date", "Open", "High", "Low", "Close", "Volume"]
        ]

        if not feature_cols:
            raise ValidationError("No features available for prediction")

        X = (
            features[feature_cols]
            .fillna(method="ffill")
            .fillna(method="bfill")
            .fillna(0)
        )
        return X.values[-60:]  # Use last 60 days for prediction

    async def predict(
        self,
        symbol: str,
        days: int = 5,
        asset_type: Optional[AssetType] = None,
        include_explanations: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate prediction for a symbol

        Args:
            symbol: Asset symbol
            days: Number of days to predict
            asset_type: Asset type (auto-detected if None)
            include_explanations: Include prediction explanations

        Returns:
            Dict with prediction results
        """
        # Detect asset type if not provided
        if asset_type is None:
            asset_type = self._detect_asset_type(symbol)

        # Fetch data
        data = await self._fetch_data(symbol, asset_type)

        # Get current price
        current_price = float(data["Close"].iloc[-1])

        # Calculate features
        features = self._calculate_features(data)

        # Detect market regime
        regime_info = self.regime_detector.detect_regime(data)

        # Get or create model
        model = self._get_or_create_model(symbol)

        # Prepare features for prediction
        X = self._prepare_features(features)

        # Train model if not trained
        if not model.is_trained():
            # Prepare training data
            y = data["Close"].pct_change().fillna(0).values[-len(X) :]
            model.train(X, y)

        # Generate predictions
        predictions = []
        last_price = current_price

        for day in range(1, days + 1):
            # Predict next day return
            pred_return, confidence = model.predict(X[-1:])
            pred_return = float(pred_return[0])
            confidence = float(confidence[0])

            # Calculate predicted price
            pred_price = last_price * (1 + pred_return)

            # Calculate confidence interval (95%)
            volatility = data["Close"].pct_change().std()
            margin = 1.96 * volatility * last_price

            predictions.append(
                {
                    "day": day,
                    "date": datetime.now() + timedelta(days=day),
                    "predicted_price": round(pred_price, 2),
                    "predicted_return": round(pred_return * 100, 2),
                    "confidence": round(confidence, 3),
                    "lower_bound": round(pred_price - margin, 2),
                    "upper_bound": round(pred_price + margin, 2),
                }
            )

            last_price = pred_price

        # Calculate confidence scores
        confidence_score = {
            "overall": round(np.mean([p["confidence"] for p in predictions]), 3),
            "model_agreement": 0.85,  # Placeholder
            "data_quality": 0.90,  # Placeholder
            "regime_stability": 0.80,  # Placeholder
            "historical_accuracy": 0.75,  # Placeholder
        }

        # Generate explanations if requested
        explanations = None
        if include_explanations:
            try:
                explanations = self.explanation_generator.generate_explanation(
                    model=model,
                    features=X[-1:],
                    feature_names=self.feature_calculator.get_feature_names(),
                )
            except Exception as e:
                # Don't fail prediction if explanation fails
                explanations = {
                    "top_factors": [],
                    "feature_importance": {},
                    "natural_language": f"Explanation generation failed: {str(e)}",
                }

        # Build response
        result = {
            "symbol": symbol,
            "asset_type": asset_type.value,
            "current_price": round(current_price, 2),
            "predictions": predictions,
            "confidence": confidence_score,
            "explanations": explanations,
            "regime": {
                "current_regime": regime_info.get("regime", "unknown"),
                "confidence": regime_info.get("confidence", 0.5),
                "transition_probabilities": regime_info.get(
                    "transition_probabilities", {}
                ),
                "duration_in_regime": regime_info.get("duration", 0),
                "expected_duration": regime_info.get("expected_duration", 0),
            },
            "timestamp": datetime.now(),
            "model_version": self._model_version,
        }

        return result

    async def batch_predict(
        self, symbols: List[str], days: int = 5, asset_type: Optional[AssetType] = None
    ) -> Dict[str, Any]:
        """
        Generate predictions for multiple symbols

        Args:
            symbols: List of symbols
            days: Number of days to predict
            asset_type: Asset type for all symbols

        Returns:
            Dict with batch prediction results
        """
        results = []
        failed_symbols = []

        # Process predictions concurrently
        tasks = [
            self.predict(symbol, days, asset_type, include_explanations=False)
            for symbol in symbols
        ]

        # Gather results
        for symbol, task in zip(symbols, asyncio.as_completed(tasks)):
            try:
                result = await task
                results.append(result)
            except Exception:
                failed_symbols.append(symbol)

        return {
            "predictions": results,
            "total_count": len(symbols),
            "successful_count": len(results),
            "failed_count": len(failed_symbols),
            "failed_symbols": failed_symbols,
            "timestamp": datetime.now(),
        }
