"""
Tests for Risk Management Module

Tests for RiskCalculator and PortfolioMetrics classes.
"""

import pytest
import numpy as np
import pandas as pd
from ara.risk import RiskCalculator, PortfolioMetrics


class TestRiskCalculator:
    """Tests for RiskCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = RiskCalculator()
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 1000)
    
    def test_calculate_var_historical(self):
        """Test VaR calculation using historical method."""
        var_95 = self.calculator.calculate_var(self.returns, 0.95, 'historical')
        var_99 = self.calculator.calculate_var(self.returns, 0.99, 'historical')
        
        assert var_95 > 0, "VaR should be positive"
        assert var_99 > var_95, "VaR at 99% should be higher than at 95%"
        assert 0 < var_95 < 0.1, "VaR should be reasonable"
    
    def test_calculate_var_parametric(self):
        """Test VaR calculation using parametric method."""
        var = self.calculator.calculate_var(self.returns, 0.95, 'parametric')
        
        assert var > 0, "VaR should be positive"
        assert 0 < var < 0.1, "VaR should be reasonable"
    
    def test_calculate_var_monte_carlo(self):
        """Test VaR calculation using Monte Carlo method."""
        var = self.calculator.calculate_var(self.returns, 0.95, 'monte_carlo')
        
        assert var > 0, "VaR should be positive"
        assert 0 < var < 0.1, "VaR should be reasonable"
    
    def test_calculate_var_invalid_confidence(self):
        """Test VaR with invalid confidence level."""
        with pytest.raises(ValueError):
            self.calculator.calculate_var(self.returns, 1.5)
        
        with pytest.raises(ValueError):
            self.calculator.calculate_var(self.returns, -0.1)
    
    def test_calculate_var_empty_returns(self):
        """Test VaR with empty returns."""
        with pytest.raises(ValueError):
            self.calculator.calculate_var(np.array([]), 0.95)
    
    def test_calculate_cvar_historical(self):
        """Test CVaR calculation using historical method."""
        cvar_95 = self.calculator.calculate_cvar(self.returns, 0.95, 'historical')
        cvar_99 = self.calculator.calculate_cvar(self.returns, 0.99, 'historical')
        var_95 = self.calculator.calculate_var(self.returns, 0.95, 'historical')
        
        assert cvar_95 > 0, "CVaR should be positive"
        assert cvar_99 > cvar_95, "CVaR at 99% should be higher than at 95%"
        assert cvar_95 >= var_95, "CVaR should be >= VaR"
    
    def test_calculate_cvar_parametric(self):
        """Test CVaR calculation using parametric method."""
        cvar = self.calculator.calculate_cvar(self.returns, 0.95, 'parametric')
        var = self.calculator.calculate_var(self.returns, 0.95, 'parametric')
        
        assert cvar > 0, "CVaR should be positive"
        assert cvar >= var, "CVaR should be >= VaR"
    
    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        returns_dict = {
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.02, 100),
            'GOOGL': np.random.normal(0.001, 0.02, 100)
        }
        
        corr_matrix = self.calculator.calculate_correlation_matrix(returns_dict)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (3, 3)
        assert all(corr_matrix.index == ['AAPL', 'MSFT', 'GOOGL'])
        
        # Diagonal should be 1
        for i in range(3):
            assert abs(corr_matrix.iloc[i, i] - 1.0) < 0.001
        
        # Correlation should be between -1 and 1
        assert (corr_matrix >= -1).all().all()
        assert (corr_matrix <= 1).all().all()
    
    def test_calculate_correlation_matrix_empty(self):
        """Test correlation matrix with empty dict."""
        with pytest.raises(ValueError):
            self.calculator.calculate_correlation_matrix({})
    
    def test_calculate_risk_decomposition(self):
        """Test risk decomposition analysis."""
        returns_dict = {
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.02, 100)
        }
        weights = {'AAPL': 0.6, 'MSFT': 0.4}
        
        decomp = self.calculator.calculate_risk_decomposition(returns_dict, weights)
        
        assert 'portfolio_volatility' in decomp
        assert 'assets' in decomp
        assert decomp['portfolio_volatility'] > 0
        
        # Check asset-level metrics
        for asset in ['AAPL', 'MSFT']:
            assert asset in decomp['assets']
            assert 'weight' in decomp['assets'][asset]
            assert 'volatility' in decomp['assets'][asset]
            assert 'marginal_risk' in decomp['assets'][asset]
            assert 'component_risk' in decomp['assets'][asset]
            assert 'percent_contribution' in decomp['assets'][asset]
        
        # Contributions should sum to 100%
        total_contribution = sum(
            decomp['assets'][asset]['percent_contribution']
            for asset in ['AAPL', 'MSFT']
        )
        assert abs(total_contribution - 100.0) < 0.01
    
    def test_calculate_risk_decomposition_equal_weights(self):
        """Test risk decomposition with equal weights."""
        returns_dict = {
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.02, 100)
        }
        
        # No weights provided, should use equal weights
        decomp = self.calculator.calculate_risk_decomposition(returns_dict)
        
        assert decomp['assets']['AAPL']['weight'] == 0.5
        assert decomp['assets']['MSFT']['weight'] == 0.5
    
    def test_calculate_risk_decomposition_invalid_weights(self):
        """Test risk decomposition with invalid weights."""
        returns_dict = {
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.02, 100)
        }
        
        # Weights don't sum to 1
        with pytest.raises(ValueError):
            self.calculator.calculate_risk_decomposition(
                returns_dict,
                {'AAPL': 0.6, 'MSFT': 0.5}
            )
        
        # Missing weight
        with pytest.raises(ValueError):
            self.calculator.calculate_risk_decomposition(
                returns_dict,
                {'AAPL': 1.0}
            )
    
    def test_calculate_portfolio_var(self):
        """Test portfolio VaR calculation."""
        returns_dict = {
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.02, 100)
        }
        weights = {'AAPL': 0.6, 'MSFT': 0.4}
        
        portfolio_var = self.calculator.calculate_portfolio_var(
            returns_dict, weights, 0.95
        )
        
        assert portfolio_var > 0
        assert 0 < portfolio_var < 0.2
    
    def test_calculate_portfolio_cvar(self):
        """Test portfolio CVaR calculation."""
        returns_dict = {
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.001, 0.02, 100)
        }
        weights = {'AAPL': 0.6, 'MSFT': 0.4}
        
        portfolio_cvar = self.calculator.calculate_portfolio_cvar(
            returns_dict, weights, 0.95
        )
        portfolio_var = self.calculator.calculate_portfolio_var(
            returns_dict, weights, 0.95
        )
        
        assert portfolio_cvar > 0
        assert portfolio_cvar >= portfolio_var


class TestPortfolioMetrics:
    """Tests for PortfolioMetrics class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = PortfolioMetrics(risk_free_rate=0.02)
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 252)
        self.benchmark_returns = np.random.normal(0.0008, 0.015, 252)
    
    def test_calculate_portfolio_volatility(self):
        """Test portfolio volatility calculation."""
        vol = self.metrics.calculate_portfolio_volatility(self.returns)
        
        assert vol > 0
        assert 0.1 < vol < 0.5  # Reasonable annual volatility
    
    def test_calculate_portfolio_volatility_no_annualize(self):
        """Test volatility without annualization."""
        vol_annual = self.metrics.calculate_portfolio_volatility(self.returns, annualize=True)
        vol_daily = self.metrics.calculate_portfolio_volatility(self.returns, annualize=False)
        
        assert vol_annual > vol_daily
        assert abs(vol_annual - vol_daily * np.sqrt(252)) < 0.001
    
    def test_calculate_beta(self):
        """Test beta calculation."""
        beta = self.metrics.calculate_beta(self.returns, self.benchmark_returns)
        
        assert isinstance(beta, float)
        # Beta can be any value, but should be reasonable
        assert -5 < beta < 5
    
    def test_calculate_beta_mismatched_length(self):
        """Test beta with mismatched return lengths."""
        short_returns = self.returns[:100]
        
        with pytest.raises(ValueError):
            self.metrics.calculate_beta(short_returns, self.benchmark_returns)
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        sharpe = self.metrics.calculate_sharpe_ratio(self.returns)
        
        assert isinstance(sharpe, float)
        assert -2 < sharpe < 5  # Reasonable Sharpe range
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        sortino = self.metrics.calculate_sortino_ratio(self.returns)
        
        assert isinstance(sortino, float)
        # Sortino should be higher than Sharpe for positive returns
        sharpe = self.metrics.calculate_sharpe_ratio(self.returns)
        if np.mean(self.returns) > 0:
            assert sortino >= sharpe
    
    def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        calmar = self.metrics.calculate_calmar_ratio(self.returns)
        
        assert isinstance(calmar, float)
    
    def test_calculate_maximum_drawdown(self):
        """Test maximum drawdown calculation."""
        max_dd = self.metrics.calculate_maximum_drawdown(self.returns)
        
        assert max_dd >= 0
        assert max_dd <= 1.0  # Drawdown should be between 0 and 100%
    
    def test_calculate_drawdown_series(self):
        """Test drawdown series calculation."""
        dd_series = self.metrics.calculate_drawdown_series(self.returns)
        
        assert len(dd_series) == len(self.returns)
        assert all(dd_series <= 0)  # All drawdowns should be negative or zero
        assert dd_series[0] == 0  # First value should be zero
    
    def test_calculate_recovery_time(self):
        """Test recovery time calculation."""
        recovery_info = self.metrics.calculate_recovery_time(self.returns)
        
        assert 'max_drawdown' in recovery_info
        assert 'drawdown_start' in recovery_info
        assert 'drawdown_end' in recovery_info
        assert 'recovery_end' in recovery_info
        assert 'recovery_time' in recovery_info
        
        assert recovery_info['max_drawdown'] >= 0
        assert recovery_info['drawdown_start'] >= 0
        assert recovery_info['drawdown_end'] >= recovery_info['drawdown_start']
    
    def test_calculate_tracking_error(self):
        """Test tracking error calculation."""
        te = self.metrics.calculate_tracking_error(self.returns, self.benchmark_returns)
        
        assert te > 0
        assert 0 < te < 0.5  # Reasonable tracking error
    
    def test_calculate_information_ratio(self):
        """Test information ratio calculation."""
        ir = self.metrics.calculate_information_ratio(self.returns, self.benchmark_returns)
        
        assert isinstance(ir, float)
        assert -5 < ir < 5  # Reasonable information ratio range
    
    def test_calculate_downside_deviation(self):
        """Test downside deviation calculation."""
        dd = self.metrics.calculate_downside_deviation(self.returns)
        
        assert dd >= 0
        
        # Downside deviation should be less than total volatility
        vol = self.metrics.calculate_portfolio_volatility(self.returns)
        assert dd <= vol
    
    def test_calculate_downside_deviation_custom_target(self):
        """Test downside deviation with custom target."""
        dd_zero = self.metrics.calculate_downside_deviation(self.returns, target_return=0.0)
        dd_high = self.metrics.calculate_downside_deviation(self.returns, target_return=0.01)
        
        # Higher target should result in higher downside deviation
        assert dd_high >= dd_zero
    
    def test_calculate_all_metrics(self):
        """Test calculating all metrics at once."""
        all_metrics = self.metrics.calculate_all_metrics(
            self.returns,
            self.benchmark_returns
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'volatility', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'max_drawdown', 'downside_deviation', 'recovery_time',
            'beta', 'tracking_error', 'information_ratio'
        ]
        
        for metric in expected_metrics:
            assert metric in all_metrics
            if all_metrics[metric] is not None:
                assert isinstance(all_metrics[metric], (int, float))
    
    def test_calculate_all_metrics_no_benchmark(self):
        """Test calculating metrics without benchmark."""
        all_metrics = self.metrics.calculate_all_metrics(self.returns)
        
        # Should not have benchmark-relative metrics
        assert 'beta' not in all_metrics
        assert 'tracking_error' not in all_metrics
        assert 'information_ratio' not in all_metrics
        
        # Should have other metrics
        assert 'volatility' in all_metrics
        assert 'sharpe_ratio' in all_metrics
    
    def test_pandas_series_input(self):
        """Test that pandas Series input works correctly."""
        returns_series = pd.Series(self.returns)
        benchmark_series = pd.Series(self.benchmark_returns)
        
        # Should work with Series input
        vol = self.metrics.calculate_portfolio_volatility(returns_series)
        beta = self.metrics.calculate_beta(returns_series, benchmark_series)
        
        assert vol > 0
        assert beta > 0


class TestRiskCalculatorIntegration:
    """Integration tests for risk calculator."""
    
    def test_diversification_benefit(self):
        """Test that portfolio VaR shows diversification benefit."""
        np.random.seed(42)
        
        # Create two uncorrelated assets
        returns_dict = {
            'Asset1': np.random.normal(0.001, 0.02, 252),
            'Asset2': np.random.normal(0.001, 0.02, 252)
        }
        weights = {'Asset1': 0.5, 'Asset2': 0.5}
        
        calculator = RiskCalculator()
        
        # Calculate individual VaRs
        var1 = calculator.calculate_var(returns_dict['Asset1'], 0.95)
        var2 = calculator.calculate_var(returns_dict['Asset2'], 0.95)
        weighted_avg_var = 0.5 * var1 + 0.5 * var2
        
        # Calculate portfolio VaR
        portfolio_var = calculator.calculate_portfolio_var(returns_dict, weights, 0.95)
        
        # Portfolio VaR should be less than weighted average (diversification benefit)
        assert portfolio_var < weighted_avg_var
    
    def test_risk_metrics_consistency(self):
        """Test consistency between different risk metrics."""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        
        calculator = RiskCalculator()
        metrics = PortfolioMetrics()
        
        # VaR and CVaR relationship
        var_95 = calculator.calculate_var(returns, 0.95)
        cvar_95 = calculator.calculate_cvar(returns, 0.95)
        assert cvar_95 >= var_95
        
        # Sharpe and Sortino relationship (for positive returns)
        sharpe = metrics.calculate_sharpe_ratio(returns)
        sortino = metrics.calculate_sortino_ratio(returns)
        if np.mean(returns) > 0:
            assert sortino >= sharpe


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
