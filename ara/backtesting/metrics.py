"""
Performance metrics calculation for backtesting.

This module implements comprehensive performance metrics including:
- Classification metrics (accuracy, precision, recall, F1)
- Directional accuracy
- Error metrics (MAE, RMSE, MAPE)
- Financial metrics (Sharpe, Sortino, Calmar ratios)
- Maximum drawdown
- Win rate and profit factor
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MetricsResult:
    """Container for all performance metrics."""

    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    directional_accuracy: float

    # Error metrics
    mae: float
    rmse: float
    mape: float

    # Financial metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float

    # Trading metrics
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Additional info
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    volatility: float

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "classification_metrics": {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score,
                "directional_accuracy": self.directional_accuracy,
            },
            "error_metrics": {"mae": self.mae, "rmse": self.rmse, "mape": self.mape},
            "financial_metrics": {
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "calmar_ratio": self.calmar_ratio,
                "max_drawdown": self.max_drawdown,
            },
            "trading_metrics": {
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
            },
            "summary": {
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
                "total_return": self.total_return,
                "annualized_return": self.annualized_return,
                "volatility": self.volatility,
            },
        }


class PerformanceMetrics:
    """Calculate comprehensive performance metrics for backtesting."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations
        """
        self.risk_free_rate = risk_free_rate

    def calculate_all_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: Optional[List[datetime]] = None,
        prices: Optional[np.ndarray] = None,
    ) -> MetricsResult:
        """
        Calculate all performance metrics.

        Args:
            predictions: Predicted values
            actuals: Actual values
            dates: Optional dates for each prediction
            prices: Optional actual prices for financial metrics

        Returns:
            MetricsResult containing all metrics
        """
        # Classification metrics
        classification = self._calculate_classification_metrics(predictions, actuals)

        # Error metrics
        errors = self._calculate_error_metrics(predictions, actuals)

        # Financial metrics
        if prices is not None:
            financial = self._calculate_financial_metrics(predictions, actuals, prices)
            trading = self._calculate_trading_metrics(predictions, actuals, prices)
        else:
            # Use default values if prices not provided
            financial = {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
            }
            trading = {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
            }

        # Date range
        start_date = dates[0] if dates else datetime.now()
        end_date = dates[-1] if dates else datetime.now()

        return MetricsResult(
            accuracy=classification["accuracy"],
            precision=classification["precision"],
            recall=classification["recall"],
            f1_score=classification["f1_score"],
            directional_accuracy=classification["directional_accuracy"],
            mae=errors["mae"],
            rmse=errors["rmse"],
            mape=errors["mape"],
            sharpe_ratio=financial["sharpe_ratio"],
            sortino_ratio=financial["sortino_ratio"],
            calmar_ratio=financial["calmar_ratio"],
            max_drawdown=financial["max_drawdown"],
            win_rate=trading["win_rate"],
            profit_factor=trading["profit_factor"],
            total_trades=trading["total_trades"],
            winning_trades=trading["winning_trades"],
            losing_trades=trading["losing_trades"],
            start_date=start_date,
            end_date=end_date,
            total_return=financial["total_return"],
            annualized_return=financial["annualized_return"],
            volatility=financial["volatility"],
        )

    def _calculate_classification_metrics(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        # Convert to binary classification (up/down)
        pred_direction = (predictions > 0).astype(int)
        actual_direction = (actuals > 0).astype(int)

        # True positives, false positives, etc.
        tp = np.sum((pred_direction == 1) & (actual_direction == 1))
        fp = np.sum((pred_direction == 1) & (actual_direction == 0))
        tn = np.sum((pred_direction == 0) & (actual_direction == 0))
        fn = np.sum((pred_direction == 0) & (actual_direction == 1))

        # Calculate metrics
        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        )
        directional_accuracy = np.mean(pred_direction == actual_direction)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "directional_accuracy": directional_accuracy,
        }

    def _calculate_error_metrics(
        self, predictions: np.ndarray, actuals: np.ndarray
    ) -> Dict[str, float]:
        """Calculate error metrics."""
        # Mean Absolute Error
        mae = np.mean(np.abs(predictions - actuals))

        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

        # Mean Absolute Percentage Error
        # Avoid division by zero
        mask = actuals != 0
        if np.any(mask):
            mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
        else:
            mape = 0.0

        return {"mae": mae, "rmse": rmse, "mape": mape}

    def _calculate_financial_metrics(
        self, predictions: np.ndarray, actuals: np.ndarray, prices: np.ndarray
    ) -> Dict[str, float]:
        """Calculate financial metrics."""
        # Calculate returns based on predictions
        returns = self._calculate_strategy_returns(predictions, actuals, prices)

        if len(returns) == 0:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
            }

        # Total and annualized return
        total_return = np.sum(returns)
        days = len(returns)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0.0

        # Volatility
        volatility = np.std(returns) * np.sqrt(252)

        # Sharpe Ratio
        excess_returns = returns - (self.risk_free_rate / 252)
        sharpe_ratio = (
            np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        )

        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        sortino_ratio = (
            np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
        )

        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
        }

    def _calculate_trading_metrics(
        self, predictions: np.ndarray, actuals: np.ndarray, prices: np.ndarray
    ) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        # Calculate returns for each trade
        returns = self._calculate_strategy_returns(predictions, actuals, prices)

        if len(returns) == 0:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
            }

        # Win/Loss statistics
        winning_trades = np.sum(returns > 0)
        losing_trades = np.sum(returns < 0)
        total_trades = len(returns)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Profit Factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
        }

    def _calculate_strategy_returns(
        self, predictions: np.ndarray, actuals: np.ndarray, prices: np.ndarray
    ) -> np.ndarray:
        """
        Calculate returns based on trading strategy.

        Strategy: Go long if prediction is positive, short if negative.
        """
        # Calculate actual returns
        actual_returns = actuals / prices[:-1] if len(prices) > len(actuals) else actuals / prices

        # Strategy returns: multiply by prediction direction
        pred_direction = np.sign(predictions)
        strategy_returns = pred_direction * actual_returns

        return strategy_returns

    def calculate_drawdown_series(self, returns: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown series from returns.

        Args:
            returns: Array of returns

        Returns:
            Array of drawdown values
        """
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown

    def calculate_rolling_sharpe(self, returns: np.ndarray, window: int = 252) -> np.ndarray:
        """
        Calculate rolling Sharpe ratio.

        Args:
            returns: Array of returns
            window: Rolling window size

        Returns:
            Array of rolling Sharpe ratios
        """
        rolling_sharpe = []
        for i in range(window, len(returns) + 1):
            window_returns = returns[i - window : i]
            excess_returns = window_returns - (self.risk_free_rate / 252)
            sharpe = (
                np.mean(excess_returns) / np.std(window_returns) * np.sqrt(252)
                if np.std(window_returns) > 0
                else 0.0
            )
            rolling_sharpe.append(sharpe)

        return np.array(rolling_sharpe)
