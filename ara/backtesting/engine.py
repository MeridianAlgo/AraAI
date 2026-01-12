"""
Comprehensive backtesting engine.

This module provides:
- Walk-forward validation
- Out-of-sample testing with 20% holdout
- Cross-validation across time periods
- Monte Carlo simulation for robustness testing
- Slippage and transaction cost modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import logging

from ara.backtesting.metrics import PerformanceMetrics, MetricsResult
from ara.backtesting.reporter import BacktestReporter
from ara.backtesting.validator import ModelValidator

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Walk-forward validation
    train_window_days: int = 252  # 1 year
    test_window_days: int = 30  # 1 month
    step_size_days: int = 30  # Move forward 1 month

    # Holdout testing
    holdout_ratio: float = 0.20  # 20% holdout

    # Cross-validation
    n_folds: int = 5

    # Monte Carlo
    n_simulations: int = 1000
    confidence_level: float = 0.95

    # Transaction costs
    slippage_bps: float = 5.0  # 5 basis points
    commission_bps: float = 10.0  # 10 basis points

    # Risk-free rate for Sharpe calculation
    risk_free_rate: float = 0.02


@dataclass
class BacktestResult:
    """Complete backtest result."""

    symbol: str
    start_date: datetime
    end_date: datetime
    config: BacktestConfig

    # Overall metrics
    metrics: MetricsResult

    # Walk-forward results
    walk_forward_results: List[Dict[str, Any]]

    # Holdout results
    holdout_metrics: Optional[MetricsResult]

    # Cross-validation results
    cv_metrics: Optional[List[MetricsResult]]

    # Monte Carlo results
    monte_carlo_results: Optional[Dict[str, Any]]

    # Returns and equity
    returns: np.ndarray
    equity_curve: np.ndarray
    dates: List[datetime]

    # Regime-specific performance
    regime_performance: Optional[Dict[str, MetricsResult]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "config": {
                "train_window_days": self.config.train_window_days,
                "test_window_days": self.config.test_window_days,
                "holdout_ratio": self.config.holdout_ratio,
                "n_folds": self.config.n_folds,
                "slippage_bps": self.config.slippage_bps,
                "commission_bps": self.config.commission_bps,
            },
            "metrics": self.metrics.to_dict(),
            "walk_forward_results": self.walk_forward_results,
            "holdout_metrics": (
                self.holdout_metrics.to_dict() if self.holdout_metrics else None
            ),
            "cv_metrics": (
                [m.to_dict() for m in self.cv_metrics] if self.cv_metrics else None
            ),
            "monte_carlo_results": self.monte_carlo_results,
        }


class BacktestEngine:
    """Comprehensive backtesting engine with multiple validation methods."""

    def __init__(
        self, config: Optional[BacktestConfig] = None, output_dir: Optional[Path] = None
    ):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            output_dir: Directory for saving results
        """
        self.config = config or BacktestConfig()
        self.output_dir = output_dir or Path("backtest_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_calculator = PerformanceMetrics(
            risk_free_rate=self.config.risk_free_rate
        )
        self.reporter = BacktestReporter(output_dir=self.output_dir)
        self.validator = ModelValidator()

        logger.info(f"BacktestEngine initialized with config: {self.config}")

    def run_backtest(
        self,
        symbol: str,
        data: pd.DataFrame,
        model_predict_fn: Callable,
        feature_columns: List[str],
        target_column: str = "target",
        price_column: str = "close",
        regime_column: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run comprehensive backtest with all validation methods.

        Args:
            symbol: Asset symbol
            data: Historical data with features and targets
            model_predict_fn: Function that takes features and returns predictions
            feature_columns: List of feature column names
            target_column: Name of target column
            price_column: Name of price column
            regime_column: Optional regime classification column

        Returns:
            BacktestResult with all metrics and analysis
        """
        logger.info(f"Starting backtest for {symbol}")

        # Split data into train and holdout
        train_data, holdout_data = self._split_holdout(data)

        # Walk-forward validation on training data
        logger.info("Running walk-forward validation...")
        walk_forward_results = self._walk_forward_validation(
            train_data, model_predict_fn, feature_columns, target_column, price_column
        )

        # Aggregate walk-forward predictions
        all_predictions = []
        all_actuals = []
        all_prices = []
        all_dates = []

        for result in walk_forward_results:
            all_predictions.extend(result["predictions"])
            all_actuals.extend(result["actuals"])
            all_prices.extend(result["prices"])
            all_dates.extend(result["dates"])

        all_predictions = np.array(all_predictions)
        all_actuals = np.array(all_actuals)
        all_prices = np.array(all_prices)

        # Apply transaction costs
        returns = self._apply_transaction_costs(
            all_predictions, all_actuals, all_prices
        )

        # Calculate overall metrics
        logger.info("Calculating performance metrics...")
        metrics = self.metrics_calculator.calculate_all_metrics(
            all_predictions, all_actuals, all_dates, all_prices
        )

        # Test on holdout data
        logger.info("Testing on holdout data...")
        holdout_metrics = None
        if len(holdout_data) > 0:
            holdout_metrics = self._test_on_holdout(
                holdout_data,
                model_predict_fn,
                feature_columns,
                target_column,
                price_column,
            )

        # Cross-validation
        logger.info("Running cross-validation...")
        cv_metrics = self._cross_validation(
            train_data, model_predict_fn, feature_columns, target_column, price_column
        )

        # Monte Carlo simulation
        logger.info("Running Monte Carlo simulation...")
        monte_carlo_results = self._monte_carlo_simulation(returns)

        # Regime-specific performance
        regime_performance = None
        if regime_column and regime_column in data.columns:
            logger.info("Analyzing regime-specific performance...")
            regime_performance = self._analyze_regime_performance(
                data, all_predictions, all_actuals, all_dates, regime_column
            )

        # Calculate equity curve
        equity_curve = np.cumprod(1 + returns)

        # Create result
        result = BacktestResult(
            symbol=symbol,
            start_date=all_dates[0],
            end_date=all_dates[-1],
            config=self.config,
            metrics=metrics,
            walk_forward_results=walk_forward_results,
            holdout_metrics=holdout_metrics,
            cv_metrics=cv_metrics,
            monte_carlo_results=monte_carlo_results,
            returns=returns,
            equity_curve=equity_curve,
            dates=all_dates,
            regime_performance=regime_performance,
        )

        logger.info(f"Backtest completed for {symbol}")
        logger.info(f"Directional Accuracy: {metrics.directional_accuracy:.2%}")
        logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {metrics.max_drawdown:.2%}")

        return result

    def _split_holdout(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and holdout sets."""
        split_idx = int(len(data) * (1 - self.config.holdout_ratio))
        train_data = data.iloc[:split_idx].copy()
        holdout_data = data.iloc[split_idx:].copy()

        logger.info(f"Split data: {len(train_data)} train, {len(holdout_data)} holdout")
        return train_data, holdout_data

    def _walk_forward_validation(
        self,
        data: pd.DataFrame,
        model_predict_fn: Callable,
        feature_columns: List[str],
        target_column: str,
        price_column: str,
    ) -> List[Dict[str, Any]]:
        """
        Perform walk-forward validation.

        This prevents look-ahead bias by training on past data and testing on future data.
        """
        results = []

        train_window = self.config.train_window_days
        test_window = self.config.test_window_days
        step_size = self.config.step_size_days

        start_idx = train_window

        while start_idx + test_window <= len(data):
            # Training window
            train_start = max(0, start_idx - train_window)
            train_end = start_idx
            data.iloc[train_start:train_end]

            # Test window
            test_start = start_idx
            test_end = min(start_idx + test_window, len(data))
            test_subset = data.iloc[test_start:test_end]

            # Generate predictions
            X_test = test_subset[feature_columns].values
            predictions = model_predict_fn(X_test)

            # Get actuals and prices
            actuals = test_subset[target_column].values
            prices = test_subset[price_column].values
            dates = (
                test_subset.index.tolist()
                if isinstance(test_subset.index, pd.DatetimeIndex)
                else list(range(len(test_subset)))
            )

            # Calculate metrics for this window
            window_metrics = self.metrics_calculator.calculate_all_metrics(
                predictions, actuals, dates, prices
            )

            results.append(
                {
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "predictions": predictions.tolist(),
                    "actuals": actuals.tolist(),
                    "prices": prices.tolist(),
                    "dates": dates,
                    "metrics": window_metrics.to_dict(),
                }
            )

            # Move forward
            start_idx += step_size

        logger.info(f"Walk-forward validation completed: {len(results)} windows")
        return results

    def _test_on_holdout(
        self,
        holdout_data: pd.DataFrame,
        model_predict_fn: Callable,
        feature_columns: List[str],
        target_column: str,
        price_column: str,
    ) -> MetricsResult:
        """Test model on holdout data."""
        X_holdout = holdout_data[feature_columns].values
        predictions = model_predict_fn(X_holdout)

        actuals = holdout_data[target_column].values
        prices = holdout_data[price_column].values
        dates = (
            holdout_data.index.tolist()
            if isinstance(holdout_data.index, pd.DatetimeIndex)
            else None
        )

        metrics = self.metrics_calculator.calculate_all_metrics(
            predictions, actuals, dates, prices
        )

        logger.info(f"Holdout test - Accuracy: {metrics.directional_accuracy:.2%}")
        return metrics

    def _cross_validation(
        self,
        data: pd.DataFrame,
        model_predict_fn: Callable,
        feature_columns: List[str],
        target_column: str,
        price_column: str,
    ) -> List[MetricsResult]:
        """
        Perform time-series cross-validation.

        Uses expanding window approach to respect temporal ordering.
        """
        n_folds = self.config.n_folds
        fold_size = len(data) // (n_folds + 1)

        cv_results = []

        for i in range(n_folds):
            # Training data: all data up to this fold
            train_end = fold_size * (i + 1)
            data.iloc[:train_end]

            # Test data: next fold
            test_start = train_end
            test_end = min(train_end + fold_size, len(data))
            test_data = data.iloc[test_start:test_end]

            if len(test_data) == 0:
                continue

            # Generate predictions
            X_test = test_data[feature_columns].values
            predictions = model_predict_fn(X_test)

            actuals = test_data[target_column].values
            prices = test_data[price_column].values
            dates = (
                test_data.index.tolist()
                if isinstance(test_data.index, pd.DatetimeIndex)
                else None
            )

            # Calculate metrics
            fold_metrics = self.metrics_calculator.calculate_all_metrics(
                predictions, actuals, dates, prices
            )

            cv_results.append(fold_metrics)

        logger.info(f"Cross-validation completed: {len(cv_results)} folds")
        return cv_results

    def _monte_carlo_simulation(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for robustness testing.

        Simulates random resampling of returns to estimate confidence intervals.
        """
        n_sims = self.config.n_simulations
        n_periods = len(returns)

        simulated_returns = []

        for _ in range(n_sims):
            # Bootstrap resample returns
            sim_returns = np.random.choice(returns, size=n_periods, replace=True)
            total_return = np.prod(1 + sim_returns) - 1
            simulated_returns.append(total_return)

        simulated_returns = np.array(simulated_returns)

        # Calculate confidence intervals
        confidence = self.config.confidence_level
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100

        return {
            "n_simulations": n_sims,
            "mean_return": float(np.mean(simulated_returns)),
            "median_return": float(np.median(simulated_returns)),
            "std_return": float(np.std(simulated_returns)),
            "confidence_level": confidence,
            "lower_bound": float(np.percentile(simulated_returns, lower_percentile)),
            "upper_bound": float(np.percentile(simulated_returns, upper_percentile)),
            "probability_positive": float(np.mean(simulated_returns > 0)),
        }

    def _apply_transaction_costs(
        self, predictions: np.ndarray, actuals: np.ndarray, prices: np.ndarray
    ) -> np.ndarray:
        """
        Apply slippage and commission costs to returns.

        Args:
            predictions: Predicted values
            actuals: Actual values
            prices: Actual prices

        Returns:
            Returns after transaction costs
        """
        # Calculate base returns
        actual_returns = (
            actuals / prices[:-1] if len(prices) > len(actuals) else actuals / prices
        )

        # Strategy returns based on predictions
        pred_direction = np.sign(predictions)
        strategy_returns = pred_direction * actual_returns

        # Apply transaction costs
        total_cost_bps = self.config.slippage_bps + self.config.commission_bps
        cost_per_trade = total_cost_bps / 10000  # Convert basis points to decimal

        # Subtract costs from returns
        returns_after_costs = strategy_returns - cost_per_trade

        return returns_after_costs

    def _analyze_regime_performance(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: List[datetime],
        regime_column: str,
    ) -> Dict[str, MetricsResult]:
        """Analyze performance by market regime."""
        regime_performance = {}

        # Get regime labels for prediction dates
        date_to_regime = dict(zip(data.index, data[regime_column]))

        # Group by regime
        regime_groups = {}
        for i, date in enumerate(dates):
            regime = date_to_regime.get(date, "unknown")
            if regime not in regime_groups:
                regime_groups[regime] = {"predictions": [], "actuals": [], "dates": []}

            regime_groups[regime]["predictions"].append(predictions[i])
            regime_groups[regime]["actuals"].append(actuals[i])
            regime_groups[regime]["dates"].append(date)

        # Calculate metrics for each regime
        for regime, group_data in regime_groups.items():
            if len(group_data["predictions"]) > 0:
                regime_metrics = self.metrics_calculator.calculate_all_metrics(
                    np.array(group_data["predictions"]),
                    np.array(group_data["actuals"]),
                    group_data["dates"],
                )
                regime_performance[regime] = regime_metrics

        return regime_performance

    def save_results(
        self, result: BacktestResult, generate_plots: bool = True
    ) -> Dict[str, Path]:
        """
        Save backtest results to disk.

        Args:
            result: Backtest result to save
            generate_plots: Whether to generate visualization plots

        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}

        # Save JSON report
        report = self.reporter.generate_full_report(
            metrics=result.metrics,
            returns=result.returns,
            dates=result.dates,
            predictions=np.concatenate(
                [r["predictions"] for r in result.walk_forward_results]
            ),
            actuals=np.concatenate([r["actuals"] for r in result.walk_forward_results]),
            symbol=result.symbol,
            regime_performance=(
                {k: v.to_dict() for k, v in result.regime_performance.items()}
                if result.regime_performance
                else None
            ),
        )

        report_path = self.reporter.save_report(
            report,
            filename=f"backtest_{result.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        saved_files["report"] = report_path

        # Generate plots
        if generate_plots:
            equity_plot = self.reporter.plot_equity_curve(
                result.returns,
                result.dates,
                result.symbol,
                save_path=self.output_dir / f"equity_curve_{result.symbol}.html",
            )
            if equity_plot:
                saved_files["equity_plot"] = equity_plot

            monthly_plot = self.reporter.plot_monthly_returns_heatmap(
                result.returns,
                result.dates,
                save_path=self.output_dir / f"monthly_returns_{result.symbol}.html",
            )
            if monthly_plot:
                saved_files["monthly_plot"] = monthly_plot

        logger.info(f"Results saved to {self.output_dir}")
        return saved_files
