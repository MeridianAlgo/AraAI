"""
Tests for Multi-Asset Correlation Analysis Module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ara.correlation import (
    CorrelationAnalyzer,
    PairsTradingAnalyzer,
    CrossAssetPredictor
)


@pytest.fixture
def sample_data():
    """Generate sample correlated price data"""
    n_days = 365
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate base series
    returns1 = np.random.normal(0.001, 0.02, n_days)
    prices1 = 100 * np.exp(np.cumsum(returns1))
    
    # Generate correlated series
    correlation = 0.8
    noise = np.random.normal(0, 0.01, n_days)
    returns2 = correlation * returns1 + np.sqrt(1 - correlation**2) * noise
    prices2 = 100 * np.exp(np.cumsum(returns2))
    
    series1 = pd.Series(prices1, index=dates)
    series2 = pd.Series(prices2, index=dates)
    
    return series1, series2


@pytest.fixture
def correlation_analyzer():
    """Create CorrelationAnalyzer instance"""
    return CorrelationAnalyzer(
        min_window=7,
        max_window=365,
        breakdown_threshold=0.3
    )


@pytest.fixture
def pairs_analyzer():
    """Create PairsTradingAnalyzer instance"""
    return PairsTradingAnalyzer(
        correlation_threshold=0.8,
        entry_z_score=2.0,
        exit_z_score=0.5
    )


@pytest.fixture
def cross_asset_predictor():
    """Create CrossAssetPredictor instance"""
    return CrossAssetPredictor(
        min_correlation=0.5,
        feature_lookback=30
    )


class TestCorrelationAnalyzer:
    """Tests for CorrelationAnalyzer"""
    
    def test_initialization(self, correlation_analyzer):
        """Test analyzer initialization"""
        assert correlation_analyzer.min_window == 7
        assert correlation_analyzer.max_window == 365
        assert correlation_analyzer.breakdown_threshold == 0.3
    
    def test_rolling_correlation(self, correlation_analyzer, sample_data):
        """Test rolling correlation calculation"""
        data1, data2 = sample_data
        
        rolling_corr = correlation_analyzer.calculate_rolling_correlation(
            data1, data2, window=30
        )
        
        assert isinstance(rolling_corr, pd.Series)
        assert len(rolling_corr) > 0
        assert rolling_corr.iloc[-1] >= -1 and rolling_corr.iloc[-1] <= 1
    
    def test_rolling_correlation_invalid_window(self, correlation_analyzer, sample_data):
        """Test rolling correlation with invalid window"""
        data1, data2 = sample_data
        
        with pytest.raises(ValueError):
            correlation_analyzer.calculate_rolling_correlation(
                data1, data2, window=5  # Too small
            )
        
        with pytest.raises(ValueError):
            correlation_analyzer.calculate_rolling_correlation(
                data1, data2, window=400  # Too large
            )
    
    def test_correlation_matrix(self, correlation_analyzer, sample_data):
        """Test correlation matrix calculation"""
        data1, data2 = sample_data
        
        asset_data = {
            'Asset1': data1,
            'Asset2': data2
        }
        
        corr_matrix = correlation_analyzer.calculate_correlation_matrix(asset_data)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (2, 2)
        assert corr_matrix.loc['Asset1', 'Asset1'] == 1.0
        assert corr_matrix.loc['Asset2', 'Asset2'] == 1.0
    
    def test_correlation_breakdown_detection(self, correlation_analyzer, sample_data):
        """Test correlation breakdown detection"""
        data1, data2 = sample_data
        
        breakdowns = correlation_analyzer.detect_correlation_breakdowns(
            data1, data2, 'Asset1', 'Asset2',
            short_window=30, long_window=90
        )
        
        assert isinstance(breakdowns, list)
        
        for breakdown in breakdowns:
            assert breakdown.asset1 == 'Asset1'
            assert breakdown.asset2 == 'Asset2'
            assert abs(breakdown.change) > correlation_analyzer.breakdown_threshold
    
    def test_lead_lag_detection(self, correlation_analyzer):
        """Test lead-lag relationship detection"""
        # Create data with known lead-lag
        n_days = 365
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        returns = np.random.normal(0.001, 0.02, n_days)
        leading = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)
        
        # Lagging series with 3-day lag
        lagging_returns = np.roll(returns, 3)
        lagging_returns[:3] = 0
        lagging = pd.Series(100 * np.exp(np.cumsum(lagging_returns)), index=dates)
        
        lead_lag = correlation_analyzer.detect_lead_lag_relationship(
            leading, lagging, 'Leading', 'Lagging', max_lag=10
        )
        
        # May or may not detect depending on noise, but should not crash
        if lead_lag:
            assert lead_lag.leading_asset in ['Leading', 'Lagging']
            assert lead_lag.optimal_lag_days >= 0
            assert lead_lag.confidence >= 0 and lead_lag.confidence <= 1
    
    def test_correlation_stability(self, correlation_analyzer, sample_data):
        """Test correlation stability calculation"""
        data1, data2 = sample_data
        
        stability = correlation_analyzer.calculate_correlation_stability(
            data1, data2, window=30, lookback_periods=12
        )
        
        assert 'mean_correlation' in stability
        assert 'std_correlation' in stability
        assert 'stability_score' in stability
        assert stability['stability_score'] >= 0
        assert stability['stability_score'] <= 1
    
    def test_correlation_regime(self, correlation_analyzer, sample_data):
        """Test correlation regime classification"""
        data1, data2 = sample_data
        
        regime = correlation_analyzer.analyze_correlation_regime(
            data1, data2, window=30
        )
        
        assert regime in [
            'strong_positive',
            'moderate_positive',
            'weak_or_no_correlation',
            'moderate_negative',
            'strong_negative',
            'unknown'
        ]


class TestPairsTradingAnalyzer:
    """Tests for PairsTradingAnalyzer"""
    
    def test_initialization(self, pairs_analyzer):
        """Test analyzer initialization"""
        assert pairs_analyzer.correlation_threshold == 0.8
        assert pairs_analyzer.entry_z_score == 2.0
        assert pairs_analyzer.exit_z_score == 0.5
    
    def test_analyze_pair(self, pairs_analyzer, sample_data):
        """Test pair analysis"""
        data1, data2 = sample_data
        
        opportunity = pairs_analyzer.analyze_pair(
            data1, data2, 'Asset1', 'Asset2'
        )
        
        # May or may not find opportunity depending on correlation
        if opportunity:
            assert opportunity.asset1 == 'Asset1'
            assert opportunity.asset2 == 'Asset2'
            assert opportunity.correlation >= -1 and opportunity.correlation <= 1
            assert opportunity.signal in ['long_spread', 'short_spread', 'neutral']
            assert opportunity.confidence >= 0 and opportunity.confidence <= 1
    
    def test_identify_opportunities(self, pairs_analyzer, sample_data):
        """Test identifying multiple pairs opportunities"""
        data1, data2 = sample_data
        
        asset_data = {
            'Asset1': data1,
            'Asset2': data2
        }
        
        opportunities = pairs_analyzer.identify_pairs_opportunities(
            asset_data,
            min_correlation=0.5
        )
        
        assert isinstance(opportunities, list)
    
    def test_position_sizing(self, pairs_analyzer, sample_data):
        """Test position size calculation"""
        data1, data2 = sample_data
        
        opportunity = pairs_analyzer.analyze_pair(
            data1, data2, 'Asset1', 'Asset2'
        )
        
        if opportunity:
            capital = 100000
            pos1, pos2 = pairs_analyzer.calculate_position_sizes(
                opportunity, capital, risk_per_trade=0.02
            )
            
            assert pos1 > 0
            assert pos2 > 0
            assert pos1 + pos2 <= capital * 0.02


class TestCrossAssetPredictor:
    """Tests for CrossAssetPredictor"""
    
    def test_initialization(self, cross_asset_predictor):
        """Test predictor initialization"""
        assert cross_asset_predictor.min_correlation == 0.5
        assert cross_asset_predictor.feature_lookback == 30
    
    def test_identify_related_assets(self, cross_asset_predictor, sample_data):
        """Test related asset identification"""
        data1, data2 = sample_data
        
        asset_data = {
            'Target': data1,
            'Related': data2
        }
        
        related = cross_asset_predictor.identify_related_assets(
            'Target', asset_data, top_n=5
        )
        
        assert isinstance(related, list)
        
        for asset, corr in related:
            assert asset != 'Target'
            assert corr >= -1 and corr <= 1
    
    def test_create_cross_asset_features(self, cross_asset_predictor, sample_data):
        """Test cross-asset feature creation"""
        data1, data2 = sample_data
        
        asset_data = {
            'Target': data1,
            'Related': data2
        }
        
        features = cross_asset_predictor.create_cross_asset_features(
            'Target', asset_data, max_features=10
        )
        
        assert isinstance(features, list)
        
        for feature in features:
            assert feature.target_asset == 'Target'
            assert feature.feature_type in ['price', 'return', 'volatility']
            assert feature.importance >= 0
    
    def test_calculate_feature_values(self, cross_asset_predictor, sample_data):
        """Test feature value calculation"""
        data1, data2 = sample_data
        
        asset_data = {
            'Target': data1,
            'Related': data2
        }
        
        features = cross_asset_predictor.create_cross_asset_features(
            'Target', asset_data, max_features=5
        )
        
        if features:
            feature_values = cross_asset_predictor.calculate_cross_asset_feature_values(
                features, asset_data
            )
            
            assert isinstance(feature_values, dict)
    
    def test_adjust_prediction(self, cross_asset_predictor, sample_data):
        """Test prediction adjustment with correlations"""
        data1, data2 = sample_data
        
        asset_data = {
            'Target': data1,
            'Related': data2
        }
        
        base_prediction = data1.iloc[-1] * 1.05
        related_predictions = {
            'Related': data2.iloc[-1] * 1.08
        }
        
        adjusted = cross_asset_predictor.adjust_prediction_with_correlations(
            'Target',
            base_prediction,
            related_predictions,
            asset_data,
            adjustment_strength=0.3
        )
        
        assert isinstance(adjusted, float)
        assert adjusted > 0
    
    def test_detect_arbitrage(self, cross_asset_predictor, sample_data):
        """Test arbitrage opportunity detection"""
        data1, data2 = sample_data
        
        asset_data = {
            'Asset1': data1,
            'Asset2': data2
        }
        
        predictions = {
            'Asset1': data1.iloc[-1] * 1.10,
            'Asset2': data2.iloc[-1] * 1.02
        }
        
        opportunities = cross_asset_predictor.detect_arbitrage_opportunities(
            asset_data,
            predictions,
            min_mispricing=0.02
        )
        
        assert isinstance(opportunities, list)
        
        for opp in opportunities:
            assert opp.asset1 in asset_data
            assert opp.asset2 in asset_data
            assert opp.confidence >= 0 and opp.confidence <= 1
    
    def test_build_inter_market_model(self, cross_asset_predictor, sample_data):
        """Test inter-market model building"""
        data1, data2 = sample_data
        
        asset_relationships = {
            'Asset1': ['Asset2']
        }
        
        asset_data = {
            'Asset1': data1,
            'Asset2': data2
        }
        
        model = cross_asset_predictor.build_inter_market_model(
            asset_relationships,
            asset_data
        )
        
        assert isinstance(model, dict)
        assert 'Asset1' in model
        assert 'Asset2' in model['Asset1']


def test_integration_workflow(sample_data):
    """Test complete workflow integration"""
    data1, data2 = sample_data
    
    # Create asset data
    asset_data = {
        'BTC': data1,
        'ETH': data2
    }
    
    # 1. Analyze correlations
    analyzer = CorrelationAnalyzer()
    corr_matrix = analyzer.calculate_correlation_matrix(asset_data)
    assert corr_matrix is not None
    
    # 2. Find pairs trading opportunities
    pairs_analyzer = PairsTradingAnalyzer()
    opportunities = pairs_analyzer.identify_pairs_opportunities(asset_data)
    assert isinstance(opportunities, list)
    
    # 3. Create cross-asset features
    predictor = CrossAssetPredictor()
    features = predictor.create_cross_asset_features('ETH', asset_data)
    assert isinstance(features, list)
    
    # 4. Detect arbitrage
    predictions = {
        'BTC': data1.iloc[-1] * 1.05,
        'ETH': data2.iloc[-1] * 1.03
    }
    arbitrage = predictor.detect_arbitrage_opportunities(
        asset_data, predictions
    )
    assert isinstance(arbitrage, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
