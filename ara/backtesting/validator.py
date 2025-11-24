"""
Automated model validation and monitoring.

This module provides:
- Daily accuracy monitoring
- Automatic retraining triggers
- Model performance degradation detection
- Validation against holdout data
- A/B testing for model comparison
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import json

from ara.backtesting.metrics import PerformanceMetrics, MetricsResult


@dataclass
class ValidationResult:
    """Result of model validation."""
    
    model_name: str
    validation_date: datetime
    accuracy: float
    directional_accuracy: float
    mae: float
    rmse: float
    sharpe_ratio: float
    needs_retraining: bool
    degradation_detected: bool
    degradation_score: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'validation_date': self.validation_date.isoformat(),
            'accuracy': float(self.accuracy),
            'directional_accuracy': float(self.directional_accuracy),
            'mae': float(self.mae),
            'rmse': float(self.rmse),
            'sharpe_ratio': float(self.sharpe_ratio),
            'needs_retraining': bool(self.needs_retraining),
            'degradation_detected': bool(self.degradation_detected),
            'degradation_score': float(self.degradation_score),
            'recommendations': self.recommendations
        }


@dataclass
class ABTestResult:
    """Result of A/B test between two models."""
    
    model_a_name: str
    model_b_name: str
    test_period_start: datetime
    test_period_end: datetime
    model_a_metrics: MetricsResult
    model_b_metrics: MetricsResult
    winner: str
    confidence: float
    improvement: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_a_name': self.model_a_name,
            'model_b_name': self.model_b_name,
            'test_period_start': self.test_period_start.isoformat(),
            'test_period_end': self.test_period_end.isoformat(),
            'model_a_metrics': self.model_a_metrics.to_dict(),
            'model_b_metrics': self.model_b_metrics.to_dict(),
            'winner': self.winner,
            'confidence': self.confidence,
            'improvement': self.improvement
        }


class ModelValidator:
    """Automated model validation and monitoring."""
    
    def __init__(
        self,
        accuracy_threshold: float = 0.75,
        degradation_threshold: float = 0.10,
        monitoring_window: int = 30,
        validation_dir: Optional[Path] = None
    ):
        """
        Initialize model validator.
        
        Args:
            accuracy_threshold: Minimum accuracy before retraining trigger
            degradation_threshold: Maximum allowed performance degradation (10%)
            monitoring_window: Days to monitor for degradation
            validation_dir: Directory for storing validation results
        """
        self.accuracy_threshold = accuracy_threshold
        self.degradation_threshold = degradation_threshold
        self.monitoring_window = monitoring_window
        self.validation_dir = validation_dir or Path("validation_results")
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = PerformanceMetrics()
        self.validation_history: List[ValidationResult] = []
    
    def validate_model(
        self,
        model_name: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        dates: List[datetime],
        prices: Optional[np.ndarray] = None,
        baseline_metrics: Optional[MetricsResult] = None
    ) -> ValidationResult:
        """
        Validate model performance and check for degradation.
        
        Args:
            model_name: Name of the model
            predictions: Model predictions
            actuals: Actual values
            dates: Dates for each prediction
            prices: Optional actual prices
            baseline_metrics: Optional baseline metrics for comparison
            
        Returns:
            ValidationResult with recommendations
        """
        # Calculate current metrics
        current_metrics = self.metrics_calculator.calculate_all_metrics(
            predictions, actuals, dates, prices
        )
        
        # Check if retraining is needed
        needs_retraining = bool(current_metrics.directional_accuracy < self.accuracy_threshold)
        
        # Check for degradation
        degradation_detected = False
        degradation_score = 0.0
        
        if baseline_metrics:
            degradation_score = self._calculate_degradation_score(
                current_metrics, baseline_metrics
            )
            degradation_detected = bool(degradation_score > self.degradation_threshold)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_metrics,
            needs_retraining,
            degradation_detected,
            degradation_score
        )
        
        # Create validation result
        result = ValidationResult(
            model_name=model_name,
            validation_date=datetime.now(),
            accuracy=current_metrics.accuracy,
            directional_accuracy=current_metrics.directional_accuracy,
            mae=current_metrics.mae,
            rmse=current_metrics.rmse,
            sharpe_ratio=current_metrics.sharpe_ratio,
            needs_retraining=needs_retraining,
            degradation_detected=degradation_detected,
            degradation_score=degradation_score,
            recommendations=recommendations
        )
        
        # Store in history
        self.validation_history.append(result)
        
        # Save to disk
        self._save_validation_result(result)
        
        return result
    
    def monitor_daily_accuracy(
        self,
        model_name: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        date: datetime
    ) -> Dict[str, Any]:
        """
        Monitor daily accuracy for a model.
        
        Args:
            model_name: Name of the model
            predictions: Daily predictions
            actuals: Actual values
            date: Date of predictions
            
        Returns:
            Dictionary with daily metrics
        """
        # Calculate daily metrics
        pred_direction = (predictions > 0).astype(int)
        actual_direction = (actuals > 0).astype(int)
        
        daily_accuracy = np.mean(pred_direction == actual_direction)
        mae = np.mean(np.abs(predictions - actuals))
        
        # Load historical accuracy
        history = self._load_accuracy_history(model_name)
        
        # Add today's accuracy
        history.append({
            'date': date.isoformat(),
            'accuracy': float(daily_accuracy),
            'mae': float(mae)
        })
        
        # Keep only recent history
        if len(history) > self.monitoring_window:
            history = history[-self.monitoring_window:]
        
        # Save updated history
        self._save_accuracy_history(model_name, history)
        
        # Calculate rolling average
        recent_accuracies = [h['accuracy'] for h in history[-7:]]  # Last 7 days
        rolling_avg = np.mean(recent_accuracies) if recent_accuracies else daily_accuracy
        
        # Check if intervention needed
        needs_attention = rolling_avg < self.accuracy_threshold
        
        return {
            'date': date.isoformat(),
            'daily_accuracy': daily_accuracy,
            'rolling_avg_7d': rolling_avg,
            'mae': mae,
            'needs_attention': needs_attention,
            'history_length': len(history)
        }
    
    def detect_performance_degradation(
        self,
        model_name: str,
        current_metrics: MetricsResult,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Detect performance degradation over time.
        
        Args:
            model_name: Name of the model
            current_metrics: Current performance metrics
            lookback_days: Days to look back for comparison
            
        Returns:
            Dictionary with degradation analysis
        """
        # Load historical validation results
        historical_results = self._load_historical_validations(
            model_name, lookback_days
        )
        
        if not historical_results:
            return {
                'degradation_detected': False,
                'message': 'Insufficient historical data for comparison'
            }
        
        # Calculate average historical performance
        avg_historical_accuracy = np.mean([r['accuracy'] for r in historical_results])
        avg_historical_sharpe = np.mean([r['sharpe_ratio'] for r in historical_results])
        
        # Calculate degradation
        accuracy_degradation = (avg_historical_accuracy - current_metrics.accuracy) / avg_historical_accuracy
        sharpe_degradation = (avg_historical_sharpe - current_metrics.sharpe_ratio) / avg_historical_sharpe if avg_historical_sharpe != 0 else 0
        
        # Overall degradation score
        degradation_score = (accuracy_degradation + sharpe_degradation) / 2
        
        degradation_detected = degradation_score > self.degradation_threshold
        
        return {
            'degradation_detected': degradation_detected,
            'degradation_score': float(degradation_score),
            'accuracy_degradation': float(accuracy_degradation),
            'sharpe_degradation': float(sharpe_degradation),
            'current_accuracy': current_metrics.accuracy,
            'historical_avg_accuracy': float(avg_historical_accuracy),
            'current_sharpe': current_metrics.sharpe_ratio,
            'historical_avg_sharpe': float(avg_historical_sharpe),
            'recommendation': 'Retrain model' if degradation_detected else 'Continue monitoring'
        }
    
    def validate_on_holdout(
        self,
        model_name: str,
        model_predict_fn: Callable,
        holdout_data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str
    ) -> MetricsResult:
        """
        Validate model on holdout dataset.
        
        Args:
            model_name: Name of the model
            model_predict_fn: Function to generate predictions
            holdout_data: Holdout dataset
            feature_columns: Feature column names
            target_column: Target column name
            
        Returns:
            MetricsResult on holdout data
        """
        # Extract features and targets
        X_holdout = holdout_data[feature_columns].values
        y_holdout = holdout_data[target_column].values
        
        # Generate predictions
        predictions = model_predict_fn(X_holdout)
        
        # Calculate metrics
        dates = holdout_data.index.tolist() if isinstance(holdout_data.index, pd.DatetimeIndex) else None
        
        metrics = self.metrics_calculator.calculate_all_metrics(
            predictions, y_holdout, dates
        )
        
        return metrics
    
    def ab_test_models(
        self,
        model_a_name: str,
        model_a_predictions: np.ndarray,
        model_b_name: str,
        model_b_predictions: np.ndarray,
        actuals: np.ndarray,
        dates: List[datetime],
        prices: Optional[np.ndarray] = None
    ) -> ABTestResult:
        """
        Perform A/B test between two models.
        
        Args:
            model_a_name: Name of first model
            model_a_predictions: Predictions from first model
            model_b_name: Name of second model
            model_b_predictions: Predictions from second model
            actuals: Actual values
            dates: Dates for each prediction
            prices: Optional actual prices
            
        Returns:
            ABTestResult with winner and improvement metrics
        """
        # Calculate metrics for both models
        metrics_a = self.metrics_calculator.calculate_all_metrics(
            model_a_predictions, actuals, dates, prices
        )
        metrics_b = self.metrics_calculator.calculate_all_metrics(
            model_b_predictions, actuals, dates, prices
        )
        
        # Determine winner based on multiple criteria
        score_a = self._calculate_model_score(metrics_a)
        score_b = self._calculate_model_score(metrics_b)
        
        winner = model_a_name if score_a > score_b else model_b_name
        confidence = abs(score_a - score_b) / max(score_a, score_b)
        
        # Calculate improvements
        improvement = {
            'accuracy': (metrics_b.accuracy - metrics_a.accuracy) / metrics_a.accuracy if metrics_a.accuracy != 0 else 0,
            'sharpe_ratio': (metrics_b.sharpe_ratio - metrics_a.sharpe_ratio) / abs(metrics_a.sharpe_ratio) if metrics_a.sharpe_ratio != 0 else 0,
            'max_drawdown': (metrics_a.max_drawdown - metrics_b.max_drawdown) / abs(metrics_a.max_drawdown) if metrics_a.max_drawdown != 0 else 0
        }
        
        result = ABTestResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            test_period_start=dates[0],
            test_period_end=dates[-1],
            model_a_metrics=metrics_a,
            model_b_metrics=metrics_b,
            winner=winner,
            confidence=confidence,
            improvement=improvement
        )
        
        # Save A/B test result
        self._save_ab_test_result(result)
        
        return result
    
    def _calculate_degradation_score(
        self,
        current: MetricsResult,
        baseline: MetricsResult
    ) -> float:
        """Calculate overall degradation score."""
        # Weight different metrics
        accuracy_weight = 0.4
        sharpe_weight = 0.3
        drawdown_weight = 0.3
        
        # Calculate relative changes
        accuracy_change = (baseline.directional_accuracy - current.directional_accuracy) / baseline.directional_accuracy if baseline.directional_accuracy != 0 else 0
        sharpe_change = (baseline.sharpe_ratio - current.sharpe_ratio) / abs(baseline.sharpe_ratio) if baseline.sharpe_ratio != 0 else 0
        drawdown_change = (current.max_drawdown - baseline.max_drawdown) / abs(baseline.max_drawdown) if baseline.max_drawdown != 0 else 0
        
        # Weighted average (only count negative changes)
        degradation = (
            max(0, accuracy_change) * accuracy_weight +
            max(0, sharpe_change) * sharpe_weight +
            max(0, drawdown_change) * drawdown_weight
        )
        
        return degradation
    
    def _calculate_model_score(self, metrics: MetricsResult) -> float:
        """Calculate overall model score for comparison."""
        # Weighted combination of key metrics
        score = (
            metrics.directional_accuracy * 0.4 +
            (metrics.sharpe_ratio / 3.0) * 0.3 +  # Normalize Sharpe (assume max ~3)
            (1 + metrics.max_drawdown) * 0.3  # Convert drawdown to positive
        )
        return score
    
    def _generate_recommendations(
        self,
        metrics: MetricsResult,
        needs_retraining: bool,
        degradation_detected: bool,
        degradation_score: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if needs_retraining:
            recommendations.append(
                f"Model accuracy ({metrics.directional_accuracy:.2%}) is below threshold "
                f"({self.accuracy_threshold:.2%}). Recommend retraining with recent data."
            )
        
        if degradation_detected:
            recommendations.append(
                f"Performance degradation detected (score: {degradation_score:.2%}). "
                "Consider retraining or adjusting model parameters."
            )
        
        if metrics.sharpe_ratio < 1.0:
            recommendations.append(
                f"Sharpe ratio ({metrics.sharpe_ratio:.2f}) is below 1.0. "
                "Review risk management and position sizing."
            )
        
        if abs(metrics.max_drawdown) > 0.20:
            recommendations.append(
                f"Maximum drawdown ({metrics.max_drawdown:.2%}) exceeds 20%. "
                "Consider implementing stricter risk controls."
            )
        
        if metrics.win_rate < 0.50:
            recommendations.append(
                f"Win rate ({metrics.win_rate:.2%}) is below 50%. "
                "Review prediction thresholds and entry criteria."
            )
        
        if not recommendations:
            recommendations.append("Model performance is within acceptable parameters. Continue monitoring.")
        
        return recommendations
    
    def _save_validation_result(self, result: ValidationResult) -> None:
        """Save validation result to disk."""
        filename = f"validation_{result.model_name}_{result.validation_date.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.validation_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _save_ab_test_result(self, result: ABTestResult) -> None:
        """Save A/B test result to disk."""
        filename = f"abtest_{result.model_a_name}_vs_{result.model_b_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.validation_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _load_accuracy_history(self, model_name: str) -> List[Dict]:
        """Load accuracy history for a model."""
        filepath = self.validation_dir / f"accuracy_history_{model_name}.json"
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return []
    
    def _save_accuracy_history(self, model_name: str, history: List[Dict]) -> None:
        """Save accuracy history for a model."""
        filepath = self.validation_dir / f"accuracy_history_{model_name}.json"
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _load_historical_validations(
        self,
        model_name: str,
        lookback_days: int
    ) -> List[Dict]:
        """Load historical validation results."""
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        historical_results = []
        
        # Find all validation files for this model
        pattern = f"validation_{model_name}_*.json"
        for filepath in self.validation_dir.glob(pattern):
            with open(filepath, 'r') as f:
                result = json.load(f)
                result_date = datetime.fromisoformat(result['validation_date'])
                
                if result_date >= cutoff_date:
                    historical_results.append(result)
        
        return historical_results
