"""
Cross-Asset Prediction Integration

Implements inter-market relationship modeling and correlation-based
prediction adjustments for improved multi-asset forecasting.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from ara.correlation.analyzer import CorrelationAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class CrossAssetFeature:
    """Cross-asset feature for prediction enhancement"""

    source_asset: str
    target_asset: str
    feature_type: str  # 'price', 'return', 'volatility', 'momentum'
    correlation: float
    lag_days: int
    importance: float


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity between assets"""

    asset1: str
    asset2: str
    expected_price1: float
    current_price1: float
    expected_price2: float
    current_price2: float
    mispricing_pct: float
    confidence: float
    opportunity_type: str  # 'statistical', 'triangular', 'cross_market'


class CrossAssetPredictor:
    """
    Enhances predictions using cross-asset relationships:
    - Inter-market relationship modeling
    - Cross-asset features (e.g., BTC price as feature for altcoins)
    - Correlation-based prediction adjustments
    - Arbitrage opportunity detection
    """

    def __init__(
        self,
        correlation_analyzer: Optional[CorrelationAnalyzer] = None,
        min_correlation: float = 0.5,
        feature_lookback: int = 30,
    ):
        """
        Initialize CrossAssetPredictor

        Args:
            correlation_analyzer: CorrelationAnalyzer instance
            min_correlation: Minimum correlation for feature inclusion
            feature_lookback: Days of historical data for feature calculation
        """
        self.correlation_analyzer = correlation_analyzer or CorrelationAnalyzer()
        self.min_correlation = min_correlation
        self.feature_lookback = feature_lookback

        # Cache for cross-asset relationships
        self._relationship_cache: Dict[Tuple[str, str], float] = {}

        logger.info("Initialized CrossAssetPredictor")

    def identify_related_assets(
        self, target_asset: str, available_assets: Dict[str, pd.Series], top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Identify assets most correlated with target asset

        Args:
            target_asset: Asset to find relationships for
            available_assets: Dictionary of available asset price series
            top_n: Number of top correlated assets to return

        Returns:
            List of (asset_name, correlation) tuples
        """
        if target_asset not in available_assets:
            raise ValueError(f"Target asset {target_asset} not in available assets")

        target_data = available_assets[target_asset]
        correlations = []

        for asset_name, asset_data in available_assets.items():
            if asset_name == target_asset:
                continue

            try:
                # Calculate correlation
                aligned = pd.DataFrame({"target": target_data, "other": asset_data}).dropna()

                if len(aligned) < self.feature_lookback:
                    continue

                corr = aligned["target"].corr(aligned["other"])

                if abs(corr) >= self.min_correlation:
                    correlations.append((asset_name, corr))

            except Exception as e:
                logger.warning(f"Failed to calculate correlation with {asset_name}: {e}")
                continue

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        return correlations[:top_n]

    def create_cross_asset_features(
        self,
        target_asset: str,
        available_assets: Dict[str, pd.Series],
        max_features: int = 10,
    ) -> List[CrossAssetFeature]:
        """
        Create cross-asset features for prediction enhancement

        Args:
            target_asset: Asset to create features for
            available_assets: Dictionary of available asset price series
            max_features: Maximum number of features to create

        Returns:
            List of CrossAssetFeature objects
        """
        features = []

        # Identify related assets
        related_assets = self.identify_related_assets(
            target_asset, available_assets, top_n=max_features
        )

        target_data = available_assets[target_asset]

        for related_asset, correlation in related_assets:
            related_data = available_assets[related_asset]

            # Detect lead-lag relationship
            lead_lag = self.correlation_analyzer.detect_lead_lag_relationship(
                target_data, related_data, target_asset, related_asset, max_lag=5
            )

            lag_days = lead_lag.optimal_lag_days if lead_lag else 0

            # Create features for different types
            feature_types = ["price", "return", "volatility"]

            for feature_type in feature_types:
                if len(features) >= max_features:
                    break

                feature = CrossAssetFeature(
                    source_asset=related_asset,
                    target_asset=target_asset,
                    feature_type=feature_type,
                    correlation=correlation,
                    lag_days=lag_days,
                    importance=abs(correlation),
                )
                features.append(feature)

        # Sort by importance
        features.sort(key=lambda x: x.importance, reverse=True)

        logger.info(f"Created {len(features)} cross-asset features for {target_asset}")

        return features[:max_features]

    def calculate_cross_asset_feature_values(
        self,
        features: List[CrossAssetFeature],
        asset_data: Dict[str, pd.Series],
        date: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Calculate actual feature values for a given date

        Args:
            features: List of CrossAssetFeature objects
            asset_data: Dictionary of asset price series
            date: Date to calculate features for (None = most recent)

        Returns:
            Dictionary mapping feature names to values
        """
        feature_values = {}

        for feature in features:
            if feature.source_asset not in asset_data:
                continue

            source_data = asset_data[feature.source_asset]

            # Get data up to specified date
            if date:
                source_data = source_data[source_data.index <= date]

            if len(source_data) < feature.lag_days + 10:
                continue

            # Apply lag
            if feature.lag_days > 0:
                lagged_data = source_data.shift(feature.lag_days)
            else:
                lagged_data = source_data

            # Calculate feature based on type
            try:
                if feature.feature_type == "price":
                    value = lagged_data.iloc[-1]
                elif feature.feature_type == "return":
                    value = lagged_data.pct_change(periods=1).iloc[-1]
                elif feature.feature_type == "volatility":
                    value = lagged_data.pct_change().rolling(20).std().iloc[-1]
                else:
                    continue

                feature_name = (
                    f"{feature.source_asset}_{feature.feature_type}_lag{feature.lag_days}"
                )
                feature_values[feature_name] = value

            except Exception as e:
                logger.warning(f"Failed to calculate feature {feature.feature_type}: {e}")
                continue

        return feature_values

    def adjust_prediction_with_correlations(
        self,
        target_asset: str,
        base_prediction: float,
        related_predictions: Dict[str, float],
        asset_data: Dict[str, pd.Series],
        adjustment_strength: float = 0.3,
    ) -> float:
        """
        Adjust prediction based on correlated asset predictions

        Args:
            target_asset: Asset being predicted
            base_prediction: Base prediction without correlation adjustment
            related_predictions: Predictions for related assets
            asset_data: Historical price data for all assets
            adjustment_strength: How much to adjust (0-1)

        Returns:
            Adjusted prediction
        """
        if target_asset not in asset_data:
            return base_prediction

        target_data = asset_data[target_asset]
        current_price = target_data.iloc[-1]

        # Calculate implied predictions from related assets
        implied_predictions = []
        weights = []

        for related_asset, related_pred in related_predictions.items():
            if related_asset not in asset_data:
                continue

            related_data = asset_data[related_asset]

            # Calculate correlation
            aligned = pd.DataFrame({"target": target_data, "related": related_data}).dropna()

            if len(aligned) < 30:
                continue

            correlation = aligned["target"].corr(aligned["related"])

            if abs(correlation) < self.min_correlation:
                continue

            # Calculate implied target prediction based on related asset movement
            related_current = related_data.iloc[-1]
            related_return = (related_pred - related_current) / related_current

            # Apply correlation to estimate target return
            implied_return = related_return * correlation
            implied_pred = current_price * (1 + implied_return)

            implied_predictions.append(implied_pred)
            weights.append(abs(correlation))

        if not implied_predictions:
            return base_prediction

        # Calculate weighted average of implied predictions
        total_weight = sum(weights)
        weighted_implied = sum(p * w for p, w in zip(implied_predictions, weights)) / total_weight

        # Blend base prediction with correlation-adjusted prediction
        adjusted_prediction = (
            base_prediction * (1 - adjustment_strength) + weighted_implied * adjustment_strength
        )

        logger.info(
            f"Adjusted {target_asset} prediction from {base_prediction:.2f} "
            f"to {adjusted_prediction:.2f} using {len(implied_predictions)} related assets"
        )

        return adjusted_prediction

    def detect_arbitrage_opportunities(
        self,
        asset_data: Dict[str, pd.Series],
        predictions: Dict[str, float],
        min_mispricing: float = 0.02,
    ) -> List[ArbitrageOpportunity]:
        """
        Detect arbitrage opportunities based on correlation and predictions

        Args:
            asset_data: Historical price data
            predictions: Predicted prices for assets
            min_mispricing: Minimum mispricing percentage to flag

        Returns:
            List of detected arbitrage opportunities
        """
        opportunities = []
        assets = list(asset_data.keys())

        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1 = assets[i]
                asset2 = assets[j]

                if asset1 not in predictions or asset2 not in predictions:
                    continue

                try:
                    opportunity = self._check_pair_arbitrage(
                        asset1,
                        asset2,
                        asset_data[asset1],
                        asset_data[asset2],
                        predictions[asset1],
                        predictions[asset2],
                        min_mispricing,
                    )

                    if opportunity:
                        opportunities.append(opportunity)

                except Exception as e:
                    logger.warning(f"Failed to check arbitrage for {asset1}-{asset2}: {e}")
                    continue

        # Sort by mispricing percentage
        opportunities.sort(key=lambda x: abs(x.mispricing_pct), reverse=True)

        logger.info(f"Detected {len(opportunities)} arbitrage opportunities")

        return opportunities

    def _check_pair_arbitrage(
        self,
        asset1: str,
        asset2: str,
        data1: pd.Series,
        data2: pd.Series,
        pred1: float,
        pred2: float,
        min_mispricing: float,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check for arbitrage opportunity between a pair of assets

        Args:
            asset1: First asset name
            asset2: Second asset name
            data1: First asset price series
            data2: Second asset price series
            pred1: Predicted price for asset1
            pred2: Predicted price for asset2
            min_mispricing: Minimum mispricing threshold

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        # Align data
        aligned = pd.DataFrame({"asset1": data1, "asset2": data2}).dropna()

        if len(aligned) < 30:
            return None

        # Calculate correlation
        correlation = aligned["asset1"].corr(aligned["asset2"])

        if abs(correlation) < 0.7:
            return None

        # Get current prices
        current1 = data1.iloc[-1]
        current2 = data2.iloc[-1]

        # Calculate expected returns
        expected_return1 = (pred1 - current1) / current1
        expected_return2 = (pred2 - current2) / current2

        # For highly correlated assets, returns should be similar
        # If they diverge significantly, there may be an arbitrage opportunity
        return_divergence = abs(expected_return1 - expected_return2)

        if return_divergence < min_mispricing:
            return None

        # Calculate expected prices based on correlation
        avg_return = (expected_return1 + expected_return2) / 2
        expected_price1 = current1 * (1 + avg_return)
        expected_price2 = current2 * (1 + avg_return)

        # Calculate mispricing
        mispricing1 = (pred1 - expected_price1) / expected_price1
        mispricing2 = (pred2 - expected_price2) / expected_price2

        # Use the larger mispricing
        if abs(mispricing1) > abs(mispricing2):
            mispricing_pct = mispricing1
        else:
            mispricing_pct = mispricing2

        # Calculate confidence
        confidence = min(abs(correlation) * (abs(mispricing_pct) / 0.1), 1.0)

        opportunity = ArbitrageOpportunity(
            asset1=asset1,
            asset2=asset2,
            expected_price1=expected_price1,
            current_price1=current1,
            expected_price2=expected_price2,
            current_price2=current2,
            mispricing_pct=mispricing_pct,
            confidence=confidence,
            opportunity_type="statistical",
        )

        return opportunity

    def build_inter_market_model(
        self,
        asset_relationships: Dict[str, List[str]],
        asset_data: Dict[str, pd.Series],
    ) -> Dict[str, Dict[str, float]]:
        """
        Build inter-market relationship model

        Args:
            asset_relationships: Dictionary mapping assets to their related assets
            asset_data: Historical price data

        Returns:
            Dictionary of relationship strengths
        """
        model = {}

        for target_asset, related_assets in asset_relationships.items():
            if target_asset not in asset_data:
                continue

            target_data = asset_data[target_asset]
            relationships = {}

            for related_asset in related_assets:
                if related_asset not in asset_data:
                    continue

                related_data = asset_data[related_asset]

                # Calculate relationship strength
                aligned = pd.DataFrame({"target": target_data, "related": related_data}).dropna()

                if len(aligned) < 30:
                    continue

                correlation = aligned["target"].corr(aligned["related"])
                relationships[related_asset] = correlation

            model[target_asset] = relationships

        logger.info(f"Built inter-market model for {len(model)} assets")

        return model
