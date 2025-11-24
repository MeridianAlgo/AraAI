"""
Comprehensive backtesting demo.

This example demonstrates:
1. Walk-forward validation
2. Out-of-sample testing
3. Cross-validation
4. Monte Carlo simulation
5. Performance metrics calculation
6. Report generation
7. Model validation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara.backtesting import BacktestEngine, PerformanceMetrics, BacktestReporter, ModelValidator
from ara.backtesting.engine import BacktestConfig


def generate_sample_data(n_days: int = 500) -> pd.DataFrame:
    """Generate sample financial data for demonstration."""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate synthetic price data with trend and noise
    trend = np.linspace(100, 150, n_days)
    noise = np.random.normal(0, 5, n_days)
    prices = trend + noise + np.cumsum(np.random.normal(0, 2, n_days))
    
    # Calculate returns
    returns = np.diff(prices) / prices[:-1]
    returns = np.concatenate([[0], returns])
    
    # Generate features
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'returns': returns,
        'sma_20': pd.Series(prices).rolling(20).mean(),
        'sma_50': pd.Series(prices).rolling(50).mean(),
        'volatility': pd.Series(returns).rolling(20).std(),
        'momentum': pd.Series(prices).pct_change(10),
        'rsi': 50 + np.random.normal(0, 15, n_days)  # Simplified RSI
    })
    
    # Target: next day return
    data['target'] = data['returns'].shift(-1)
    
    # Add regime classification
    data['regime'] = pd.cut(
        data['volatility'],
        bins=3,
        labels=['low_vol', 'medium_vol', 'high_vol']
    )
    
    # Drop NaN values
    data = data.dropna()
    data.set_index('date', inplace=True)
    
    return data


def simple_prediction_model(X: np.ndarray) -> np.ndarray:
    """
    Simple prediction model for demonstration.
    
    Uses momentum and moving average crossover signals.
    """
    # Extract features (assuming order: sma_20, sma_50, volatility, momentum, rsi)
    sma_20 = X[:, 0]
    sma_50 = X[:, 1]
    momentum = X[:, 3]
    
    # Simple strategy: predict positive return if SMA20 > SMA50 and momentum > 0
    predictions = np.where(
        (sma_20 > sma_50) & (momentum > 0),
        0.01,  # Predict 1% gain
        -0.005  # Predict 0.5% loss
    )
    
    return predictions


def main():
    """Run comprehensive backtesting demo."""
    print("=" * 80)
    print("ARA AI Backtesting Demo")
    print("=" * 80)
    print()
    
    # Generate sample data
    print("1. Generating sample data...")
    data = generate_sample_data(n_days=500)
    print(f"   Generated {len(data)} days of data")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print()
    
    # Define features and target
    feature_columns = ['sma_20', 'sma_50', 'volatility', 'momentum', 'rsi']
    target_column = 'target'
    price_column = 'close'
    regime_column = 'regime'
    
    # Configure backtest
    print("2. Configuring backtest engine...")
    config = BacktestConfig(
        train_window_days=120,  # 4 months training
        test_window_days=20,    # 20 days testing
        step_size_days=20,      # Move forward 20 days
        holdout_ratio=0.20,     # 20% holdout
        n_folds=5,              # 5-fold CV
        n_simulations=1000,     # Monte Carlo simulations
        slippage_bps=5.0,       # 5 bps slippage
        commission_bps=10.0     # 10 bps commission
    )
    
    engine = BacktestEngine(config=config)
    print("   Backtest engine configured")
    print()
    
    # Run backtest
    print("3. Running comprehensive backtest...")
    print("   This includes:")
    print("   - Walk-forward validation")
    print("   - Out-of-sample testing (20% holdout)")
    print("   - 5-fold cross-validation")
    print("   - Monte Carlo simulation (1000 runs)")
    print("   - Transaction cost modeling")
    print()
    
    result = engine.run_backtest(
        symbol='DEMO',
        data=data,
        model_predict_fn=simple_prediction_model,
        feature_columns=feature_columns,
        target_column=target_column,
        price_column=price_column,
        regime_column=regime_column
    )
    
    print()
    print("=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print()
    
    # Display overall metrics
    print("Overall Performance Metrics:")
    print("-" * 80)
    metrics = result.metrics
    print(f"  Directional Accuracy:  {metrics.directional_accuracy:.2%}")
    print(f"  Precision:             {metrics.precision:.2%}")
    print(f"  Recall:                {metrics.recall:.2%}")
    print(f"  F1 Score:              {metrics.f1_score:.2%}")
    print()
    print(f"  MAE:                   {metrics.mae:.4f}")
    print(f"  RMSE:                  {metrics.rmse:.4f}")
    print(f"  MAPE:                  {metrics.mape:.2f}%")
    print()
    print(f"  Sharpe Ratio:          {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:         {metrics.sortino_ratio:.2f}")
    print(f"  Calmar Ratio:          {metrics.calmar_ratio:.2f}")
    print(f"  Max Drawdown:          {metrics.max_drawdown:.2%}")
    print()
    print(f"  Win Rate:              {metrics.win_rate:.2%}")
    print(f"  Profit Factor:         {metrics.profit_factor:.2f}")
    print(f"  Total Trades:          {metrics.total_trades}")
    print()
    print(f"  Total Return:          {metrics.total_return:.2%}")
    print(f"  Annualized Return:     {metrics.annualized_return:.2%}")
    print(f"  Volatility:            {metrics.volatility:.2%}")
    print()
    
    # Holdout results
    if result.holdout_metrics:
        print("Holdout Test Results (20% out-of-sample):")
        print("-" * 80)
        holdout = result.holdout_metrics
        print(f"  Directional Accuracy:  {holdout.directional_accuracy:.2%}")
        print(f"  Sharpe Ratio:          {holdout.sharpe_ratio:.2f}")
        print(f"  Max Drawdown:          {holdout.max_drawdown:.2%}")
        print()
    
    # Cross-validation results
    if result.cv_metrics:
        print("Cross-Validation Results (5 folds):")
        print("-" * 80)
        cv_accuracies = [m.directional_accuracy for m in result.cv_metrics]
        cv_sharpes = [m.sharpe_ratio for m in result.cv_metrics]
        print(f"  Avg Accuracy:          {np.mean(cv_accuracies):.2%} ± {np.std(cv_accuracies):.2%}")
        print(f"  Avg Sharpe Ratio:      {np.mean(cv_sharpes):.2f} ± {np.std(cv_sharpes):.2f}")
        print()
    
    # Monte Carlo results
    if result.monte_carlo_results:
        print("Monte Carlo Simulation Results (1000 runs):")
        print("-" * 80)
        mc = result.monte_carlo_results
        print(f"  Mean Return:           {mc['mean_return']:.2%}")
        print(f"  Median Return:         {mc['median_return']:.2%}")
        print(f"  Std Deviation:         {mc['std_return']:.2%}")
        print(f"  95% Confidence:        [{mc['lower_bound']:.2%}, {mc['upper_bound']:.2%}]")
        print(f"  Prob. Positive:        {mc['probability_positive']:.2%}")
        print()
    
    # Regime-specific performance
    if result.regime_performance:
        print("Regime-Specific Performance:")
        print("-" * 80)
        for regime, regime_metrics in result.regime_performance.items():
            print(f"  {regime}:")
            print(f"    Accuracy:            {regime_metrics.directional_accuracy:.2%}")
            print(f"    Sharpe Ratio:        {regime_metrics.sharpe_ratio:.2f}")
            print(f"    Max Drawdown:        {regime_metrics.max_drawdown:.2%}")
        print()
    
    # Save results
    print("4. Saving results...")
    saved_files = engine.save_results(result, generate_plots=True)
    print(f"   Results saved to: {engine.output_dir}")
    for file_type, filepath in saved_files.items():
        print(f"   - {file_type}: {filepath.name}")
    print()
    
    # Model validation
    print("5. Running model validation...")
    validator = ModelValidator(
        accuracy_threshold=0.75,
        degradation_threshold=0.10
    )
    
    # Get all predictions and actuals
    all_predictions = np.concatenate([r['predictions'] for r in result.walk_forward_results])
    all_actuals = np.concatenate([r['actuals'] for r in result.walk_forward_results])
    all_dates = []
    for r in result.walk_forward_results:
        all_dates.extend(r['dates'])
    
    validation_result = validator.validate_model(
        model_name='simple_momentum',
        predictions=all_predictions,
        actuals=all_actuals,
        dates=all_dates
    )
    
    print(f"   Model: {validation_result.model_name}")
    print(f"   Needs Retraining: {validation_result.needs_retraining}")
    print(f"   Degradation Detected: {validation_result.degradation_detected}")
    print()
    print("   Recommendations:")
    for rec in validation_result.recommendations:
        print(f"   - {rec}")
    print()
    
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
