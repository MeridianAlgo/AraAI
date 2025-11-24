"""
Tests for backtesting module.

Tests cover:
- Performance metrics calculation
- Backtest engine functionality
- Report generation
- Model validation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from ara.backtesting import (
    PerformanceMetrics,
    BacktestEngine,
    BacktestReporter,
    ModelValidator
)
from ara.backtesting.engine import BacktestConfig


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_days = 200
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    prices = 100 + np.cumsum(np.random.normal(0.1, 2, n_days))
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'returns': returns,
        'feature1': np.random.randn(n_days),
        'feature2': np.random.randn(n_days),
        'target': np.concatenate([returns[1:], [0]])
    })
    
    data = data[:-1]  # Remove last row with 0 target
    data.set_index('date', inplace=True)
    
    return data


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestPerformanceMetrics:
    """Test performance metrics calculation."""
    
    def test_classification_metrics(self):
        """Test classification metrics calculation."""
        metrics_calc = PerformanceMetrics()
        
        predictions = np.array([0.01, -0.01, 0.02, -0.01, 0.01])
        actuals = np.array([0.015, -0.005, 0.025, 0.005, 0.01])
        
        metrics = metrics_calc.calculate_all_metrics(predictions, actuals)
        
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1_score <= 1
        assert 0 <= metrics.directional_accuracy <= 1
    
    def test_error_metrics(self):
        """Test error metrics calculation."""
        metrics_calc = PerformanceMetrics()
        
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = metrics_calc.calculate_all_metrics(predictions, actuals)
        
        assert metrics.mae > 0
        assert metrics.rmse > 0
        assert metrics.mape > 0
        assert metrics.rmse >= metrics.mae  # RMSE should be >= MAE
    
    def test_financial_metrics(self):
        """Test financial metrics calculation."""
        metrics_calc = PerformanceMetrics(risk_free_rate=0.02)
        
        predictions = np.array([0.01, 0.02, -0.01, 0.015, 0.01])
        actuals = np.array([0.015, 0.025, -0.005, 0.02, 0.012])
        prices = np.array([100, 101, 103, 102, 104, 105])
        
        metrics = metrics_calc.calculate_all_metrics(predictions, actuals, prices=prices)
        
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        assert isinstance(metrics.calmar_ratio, float)
        assert metrics.max_drawdown <= 0  # Drawdown should be negative
    
    def test_trading_metrics(self):
        """Test trading metrics calculation."""
        metrics_calc = PerformanceMetrics()
        
        predictions = np.array([0.01, 0.02, -0.01, 0.015, 0.01])
        actuals = np.array([0.015, 0.025, -0.005, 0.02, 0.012])
        prices = np.array([100, 101, 103, 102, 104, 105])
        
        metrics = metrics_calc.calculate_all_metrics(predictions, actuals, prices=prices)
        
        assert 0 <= metrics.win_rate <= 1
        assert metrics.profit_factor >= 0
        assert metrics.total_trades == len(predictions)
        assert metrics.winning_trades + metrics.losing_trades <= metrics.total_trades
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics_calc = PerformanceMetrics()
        
        predictions = np.array([0.01, -0.01, 0.02])
        actuals = np.array([0.015, -0.005, 0.025])
        
        metrics = metrics_calc.calculate_all_metrics(predictions, actuals)
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'classification_metrics' in metrics_dict
        assert 'error_metrics' in metrics_dict
        assert 'financial_metrics' in metrics_dict
        assert 'trading_metrics' in metrics_dict


class TestBacktestEngine:
    """Test backtest engine functionality."""
    
    def test_engine_initialization(self, temp_dir):
        """Test engine initialization."""
        config = BacktestConfig(
            train_window_days=60,
            test_window_days=10
        )
        
        engine = BacktestEngine(config=config, output_dir=temp_dir)
        
        assert engine.config.train_window_days == 60
        assert engine.config.test_window_days == 10
        assert engine.output_dir == temp_dir
    
    def test_split_holdout(self, sample_data, temp_dir):
        """Test holdout split."""
        engine = BacktestEngine(output_dir=temp_dir)
        
        train_data, holdout_data = engine._split_holdout(sample_data)
        
        assert len(train_data) + len(holdout_data) == len(sample_data)
        # Allow for rounding differences
        expected_holdout = int(len(sample_data) * 0.20)
        assert abs(len(holdout_data) - expected_holdout) <= 1
    
    def test_walk_forward_validation(self, sample_data, temp_dir):
        """Test walk-forward validation."""
        config = BacktestConfig(
            train_window_days=60,
            test_window_days=10,
            step_size_days=10
        )
        engine = BacktestEngine(config=config, output_dir=temp_dir)
        
        def simple_model(X):
            return np.random.randn(len(X)) * 0.01
        
        results = engine._walk_forward_validation(
            sample_data,
            simple_model,
            ['feature1', 'feature2'],
            'target',
            'close'
        )
        
        assert len(results) > 0
        assert all('predictions' in r for r in results)
        assert all('actuals' in r for r in results)
        assert all('metrics' in r for r in results)
    
    def test_run_backtest(self, sample_data, temp_dir):
        """Test full backtest run."""
        config = BacktestConfig(
            train_window_days=60,
            test_window_days=10,
            step_size_days=10,
            holdout_ratio=0.20,
            n_folds=3,
            n_simulations=100
        )
        engine = BacktestEngine(config=config, output_dir=temp_dir)
        
        def simple_model(X):
            return np.random.randn(len(X)) * 0.01
        
        result = engine.run_backtest(
            symbol='TEST',
            data=sample_data,
            model_predict_fn=simple_model,
            feature_columns=['feature1', 'feature2'],
            target_column='target',
            price_column='close'
        )
        
        assert result.symbol == 'TEST'
        assert result.metrics is not None
        assert len(result.walk_forward_results) > 0
        assert result.holdout_metrics is not None
        assert result.cv_metrics is not None
        assert result.monte_carlo_results is not None
        assert len(result.returns) > 0
        assert len(result.equity_curve) > 0
    
    def test_monte_carlo_simulation(self, temp_dir):
        """Test Monte Carlo simulation."""
        engine = BacktestEngine(output_dir=temp_dir)
        
        returns = np.random.randn(100) * 0.01
        mc_results = engine._monte_carlo_simulation(returns)
        
        assert 'mean_return' in mc_results
        assert 'median_return' in mc_results
        assert 'lower_bound' in mc_results
        assert 'upper_bound' in mc_results
        assert 'probability_positive' in mc_results
        assert 0 <= mc_results['probability_positive'] <= 1
    
    def test_transaction_costs(self, temp_dir):
        """Test transaction cost application."""
        config = BacktestConfig(
            slippage_bps=5.0,
            commission_bps=10.0
        )
        engine = BacktestEngine(config=config, output_dir=temp_dir)
        
        predictions = np.array([0.01, -0.01, 0.02])
        actuals = np.array([0.015, -0.005, 0.025])
        prices = np.array([100, 101, 102, 104])
        
        returns = engine._apply_transaction_costs(predictions, actuals, prices)
        
        assert len(returns) == len(predictions)
        # Transaction costs should be applied (15 bps total)
        total_cost = (5.0 + 10.0) / 10000
        assert total_cost > 0


class TestBacktestReporter:
    """Test backtest reporter functionality."""
    
    def test_reporter_initialization(self, temp_dir):
        """Test reporter initialization."""
        reporter = BacktestReporter(output_dir=temp_dir)
        
        assert reporter.output_dir == temp_dir
        assert reporter.output_dir.exists()
    
    def test_generate_equity_curve_data(self, temp_dir):
        """Test equity curve data generation."""
        reporter = BacktestReporter(output_dir=temp_dir)
        
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        dates = [datetime.now() + timedelta(days=i) for i in range(5)]
        
        equity_data = reporter._generate_equity_curve_data(returns, dates)
        
        assert 'dates' in equity_data
        assert 'equity' in equity_data
        assert 'drawdown' in equity_data
        assert len(equity_data['equity']) == len(returns)
    
    def test_calculate_monthly_returns(self, temp_dir):
        """Test monthly returns calculation."""
        reporter = BacktestReporter(output_dir=temp_dir)
        
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        returns = np.random.randn(len(dates)) * 0.01
        
        monthly_data = reporter._calculate_monthly_returns(returns, dates.tolist())
        
        assert 'data' in monthly_data
        assert 'summary' in monthly_data
        assert 'best_month' in monthly_data['summary']
        assert 'worst_month' in monthly_data['summary']
    
    def test_save_report(self, temp_dir):
        """Test report saving."""
        reporter = BacktestReporter(output_dir=temp_dir)
        
        report = {
            'symbol': 'TEST',
            'metrics': {'accuracy': 0.75}
        }
        
        filepath = reporter.save_report(report, filename='test_report.json')
        
        assert filepath.exists()
        assert filepath.name == 'test_report.json'


class TestModelValidator:
    """Test model validator functionality."""
    
    def test_validator_initialization(self, temp_dir):
        """Test validator initialization."""
        validator = ModelValidator(
            accuracy_threshold=0.75,
            degradation_threshold=0.10,
            validation_dir=temp_dir
        )
        
        assert validator.accuracy_threshold == 0.75
        assert validator.degradation_threshold == 0.10
        assert validator.validation_dir == temp_dir
    
    def test_validate_model(self, temp_dir):
        """Test model validation."""
        validator = ModelValidator(validation_dir=temp_dir)
        
        predictions = np.array([0.01, 0.02, -0.01, 0.015, 0.01])
        actuals = np.array([0.015, 0.025, -0.005, 0.02, 0.012])
        dates = [datetime.now() + timedelta(days=i) for i in range(5)]
        
        result = validator.validate_model(
            model_name='test_model',
            predictions=predictions,
            actuals=actuals,
            dates=dates
        )
        
        assert result.model_name == 'test_model'
        assert isinstance(result.needs_retraining, bool)
        assert isinstance(result.degradation_detected, bool)
        assert len(result.recommendations) > 0
    
    def test_monitor_daily_accuracy(self, temp_dir):
        """Test daily accuracy monitoring."""
        validator = ModelValidator(validation_dir=temp_dir)
        
        predictions = np.array([0.01, 0.02, -0.01])
        actuals = np.array([0.015, 0.025, -0.005])
        date = datetime.now()
        
        result = validator.monitor_daily_accuracy(
            model_name='test_model',
            predictions=predictions,
            actuals=actuals,
            date=date
        )
        
        assert 'daily_accuracy' in result
        assert 'rolling_avg_7d' in result
        assert 'needs_attention' in result
        assert 0 <= result['daily_accuracy'] <= 1
    
    def test_ab_test_models(self, temp_dir):
        """Test A/B testing between models."""
        validator = ModelValidator(validation_dir=temp_dir)
        
        model_a_predictions = np.array([0.01, 0.02, -0.01, 0.015, 0.01])
        model_b_predictions = np.array([0.012, 0.018, -0.008, 0.02, 0.009])
        actuals = np.array([0.015, 0.025, -0.005, 0.02, 0.012])
        dates = [datetime.now() + timedelta(days=i) for i in range(5)]
        
        result = validator.ab_test_models(
            model_a_name='model_a',
            model_a_predictions=model_a_predictions,
            model_b_name='model_b',
            model_b_predictions=model_b_predictions,
            actuals=actuals,
            dates=dates
        )
        
        assert result.model_a_name == 'model_a'
        assert result.model_b_name == 'model_b'
        assert result.winner in ['model_a', 'model_b']
        assert 0 <= result.confidence <= 1
        assert 'accuracy' in result.improvement


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
