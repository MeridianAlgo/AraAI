"""
Comprehensive Performance Metrics and Error Tracking System
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict
import json

try:
    from models import PredictionResult, PerformanceMetrics, ModelDiagnostics
    from data_manager import prediction_manager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from models import PredictionResult, PerformanceMetrics, ModelDiagnostics
    from data_manager import prediction_manager

logger = logging.getLogger(__name__)

class ErrorTracker:
    """Real-time error calculation and tracking"""
    
    def __init__(self):
        self.error_history = defaultdict(list)
        self.rolling_windows = [1, 7, 30]  # days
    
    def calculate_prediction_error(self, prediction: PredictionResult, actual_price: float) -> Dict[str, float]:
        """Calculate error metrics for a single prediction"""
        
        # Absolute error
        absolute_error = abs(prediction.predicted_price - actual_price)
        
        # Percentage error
        percentage_error = (absolute_error / actual_price) * 100
        
        # Squared error
        squared_error = (prediction.predicted_price - actual_price) ** 2
        
        # Directional accuracy
        predicted_direction = prediction.direction
        actual_direction = "UP" if actual_price > prediction.current_price else "DOWN"
        directional_correct = 1.0 if predicted_direction == actual_direction else 0.0
        
        return {
            'absolute_error': absolute_error,
            'percentage_error': percentage_error,
            'squared_error': squared_error,
            'directional_correct': directional_correct,
            'predicted_price': prediction.predicted_price,
            'actual_price': actual_price,
            'current_price': prediction.current_price
        }
    
    def update_error_history(self, symbol: str, error_metrics: Dict[str, float], timestamp: datetime):
        """Update error history for a symbol"""
        error_entry = {
            'timestamp': timestamp,
            'metrics': error_metrics
        }
        self.error_history[symbol].append(error_entry)
        
        # Keep only last 100 entries per symbol
        if len(self.error_history[symbol]) > 100:
            self.error_history[symbol] = self.error_history[symbol][-100:]
    
    def calculate_mae(self, errors: List[float]) -> float:
        """Calculate Mean Absolute Error"""
        return np.mean(errors) if errors else 0.0
    
    def calculate_mse(self, errors: List[float]) -> float:
        """Calculate Mean Squared Error"""
        return np.mean([e**2 for e in errors]) if errors else 0.0
    
    def calculate_rmse(self, errors: List[float]) -> float:
        """Calculate Root Mean Squared Error"""
        mse = self.calculate_mse(errors)
        return np.sqrt(mse)
    
    def get_rolling_metrics(self, symbol: str, days: int) -> Dict[str, float]:
        """Get rolling metrics for specified days"""
        if symbol not in self.error_history:
            return self._empty_metrics()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_errors = [
            entry for entry in self.error_history[symbol]
            if entry['timestamp'] >= cutoff_date
        ]
        
        if not recent_errors:
            return self._empty_metrics()
        
        # Extract error values
        absolute_errors = [e['metrics']['absolute_error'] for e in recent_errors]
        percentage_errors = [e['metrics']['percentage_error'] for e in recent_errors]
        directional_correct = [e['metrics']['directional_correct'] for e in recent_errors]
        
        return {
            'mae': self.calculate_mae(absolute_errors),
            'mse': self.calculate_mse(absolute_errors),
            'rmse': self.calculate_rmse(absolute_errors),
            'mape': np.mean(percentage_errors) if percentage_errors else 0.0,
            'directional_accuracy': np.mean(directional_correct) if directional_correct else 0.0,
            'sample_count': len(recent_errors)
        }
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics structure"""
        return {
            'mae': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'mape': 0.0,
            'directional_accuracy': 0.0,
            'sample_count': 0
        }

class PerformanceAnalyzer:
    """Comprehensive performance analysis and reporting"""
    
    def __init__(self):
        self.error_tracker = ErrorTracker()
    
    def analyze_predictions(self, symbol: str, days: int = 30) -> Dict:
        """Comprehensive analysis of recent predictions"""
        
        # Get recent predictions
        predictions = prediction_manager.get_recent_predictions(symbol, days)
        
        if not predictions:
            return {'error': f'No predictions found for {symbol} in last {days} days'}
        
        # Filter predictions with actual prices
        predictions_with_actuals = [p for p in predictions if p.actual_price is not None]
        
        analysis = {
            'symbol': symbol,
            'analysis_period_days': days,
            'total_predictions': len(predictions),
            'predictions_with_actuals': len(predictions_with_actuals),
            'timestamp': datetime.now().isoformat()
        }
        
        if not predictions_with_actuals:
            analysis['error'] = 'No predictions with actual prices available'
            return analysis
        
        # Calculate comprehensive metrics
        analysis.update(self._calculate_comprehensive_metrics(predictions_with_actuals))
        
        # Confidence analysis
        analysis['confidence_analysis'] = self._analyze_confidence(predictions)
        
        # Error distribution
        analysis['error_distribution'] = self._analyze_error_distribution(predictions_with_actuals)
        
        # Trend analysis
        analysis['trend_analysis'] = self._analyze_trends(predictions_with_actuals)
        
        return analysis
    
    def _calculate_comprehensive_metrics(self, predictions: List[PredictionResult]) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Extract values
        predicted_prices = [p.predicted_price for p in predictions]
        actual_prices = [p.actual_price for p in predictions]
        current_prices = [p.current_price for p in predictions]
        
        # Error calculations
        absolute_errors = [abs(pred - actual) for pred, actual in zip(predicted_prices, actual_prices)]
        percentage_errors = [abs(pred - actual) / actual * 100 for pred, actual in zip(predicted_prices, actual_prices)]
        squared_errors = [(pred - actual) ** 2 for pred, actual in zip(predicted_prices, actual_prices)]
        
        # Directional accuracy
        predicted_directions = [p.direction for p in predictions]
        actual_directions = ["UP" if actual > current else "DOWN" 
                           for actual, current in zip(actual_prices, current_prices)]
        directional_correct = [1 if pred == actual else 0 
                             for pred, actual in zip(predicted_directions, actual_directions)]
        
        return {
            'error_metrics': {
                'mae': np.mean(absolute_errors),
                'mse': np.mean(squared_errors),
                'rmse': np.sqrt(np.mean(squared_errors)),
                'mape': np.mean(percentage_errors),
                'max_error': max(absolute_errors),
                'min_error': min(absolute_errors),
                'std_error': np.std(absolute_errors)
            },
            'directional_accuracy': np.mean(directional_correct),
            'accuracy_by_direction': {
                'up_predictions': predicted_directions.count('UP'),
                'down_predictions': predicted_directions.count('DOWN'),
                'up_correct': sum(1 for i, pred in enumerate(predicted_directions) 
                                if pred == 'UP' and directional_correct[i] == 1),
                'down_correct': sum(1 for i, pred in enumerate(predicted_directions) 
                                  if pred == 'DOWN' and directional_correct[i] == 1)
            }
        }
    
    def _analyze_confidence(self, predictions: List[PredictionResult]) -> Dict:
        """Analyze confidence scores and calibration"""
        confidences = [p.confidence for p in predictions]
        
        # Confidence distribution
        confidence_bins = {
            'low (0.0-0.3)': sum(1 for c in confidences if 0.0 <= c < 0.3),
            'medium (0.3-0.7)': sum(1 for c in confidences if 0.3 <= c < 0.7),
            'high (0.7-1.0)': sum(1 for c in confidences if 0.7 <= c <= 1.0)
        }
        
        return {
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'confidence_distribution': confidence_bins
        }
    
    def _analyze_error_distribution(self, predictions: List[PredictionResult]) -> Dict:
        """Analyze error distribution patterns"""
        errors = [abs(p.predicted_price - p.actual_price) for p in predictions]
        
        # Error percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        error_percentiles = {f'p{p}': np.percentile(errors, p) for p in percentiles}
        
        # Error bins
        max_error = max(errors)
        bin_size = max_error / 5
        error_bins = {
            f'0-{bin_size:.2f}': sum(1 for e in errors if 0 <= e < bin_size),
            f'{bin_size:.2f}-{2*bin_size:.2f}': sum(1 for e in errors if bin_size <= e < 2*bin_size),
            f'{2*bin_size:.2f}-{3*bin_size:.2f}': sum(1 for e in errors if 2*bin_size <= e < 3*bin_size),
            f'{3*bin_size:.2f}-{4*bin_size:.2f}': sum(1 for e in errors if 3*bin_size <= e < 4*bin_size),
            f'{4*bin_size:.2f}+': sum(1 for e in errors if e >= 4*bin_size)
        }
        
        return {
            'error_percentiles': error_percentiles,
            'error_bins': error_bins,
            'outliers_count': sum(1 for e in errors if e > np.percentile(errors, 95))
        }
    
    def _analyze_trends(self, predictions: List[PredictionResult]) -> Dict:
        """Analyze performance trends over time"""
        # Sort by timestamp
        sorted_predictions = sorted(predictions, key=lambda p: p.timestamp)
        
        if len(sorted_predictions) < 5:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Split into early and recent halves
        mid_point = len(sorted_predictions) // 2
        early_predictions = sorted_predictions[:mid_point]
        recent_predictions = sorted_predictions[mid_point:]
        
        # Calculate metrics for each half
        early_metrics = self._calculate_basic_metrics(early_predictions)
        recent_metrics = self._calculate_basic_metrics(recent_predictions)
        
        return {
            'early_period': early_metrics,
            'recent_period': recent_metrics,
            'improvement': {
                'mae_change': recent_metrics['mae'] - early_metrics['mae'],
                'directional_accuracy_change': recent_metrics['directional_accuracy'] - early_metrics['directional_accuracy'],
                'confidence_change': recent_metrics['avg_confidence'] - early_metrics['avg_confidence']
            }
        }
    
    def _calculate_basic_metrics(self, predictions: List[PredictionResult]) -> Dict:
        """Calculate basic metrics for a set of predictions"""
        if not predictions:
            return {'mae': 0.0, 'directional_accuracy': 0.0, 'avg_confidence': 0.0}
        
        # Filter predictions with actual prices
        valid_predictions = [p for p in predictions if p.actual_price is not None]
        
        if not valid_predictions:
            return {'mae': 0.0, 'directional_accuracy': 0.0, 'avg_confidence': np.mean([p.confidence for p in predictions])}
        
        # Calculate metrics
        absolute_errors = [abs(p.predicted_price - p.actual_price) for p in valid_predictions]
        mae = np.mean(absolute_errors)
        
        directional_correct = []
        for p in valid_predictions:
            predicted_dir = p.direction
            actual_dir = "UP" if p.actual_price > p.current_price else "DOWN"
            directional_correct.append(1 if predicted_dir == actual_dir else 0)
        
        directional_accuracy = np.mean(directional_correct)
        avg_confidence = np.mean([p.confidence for p in predictions])
        
        return {
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'avg_confidence': avg_confidence
        }
    
    def generate_performance_report(self, symbol: str, days: int = 30) -> str:
        """Generate a formatted performance report"""
        analysis = self.analyze_predictions(symbol, days)
        
        if 'error' in analysis:
            return f"Performance Report for {symbol}: {analysis['error']}"
        
        report = f"""
📊 PERFORMANCE REPORT FOR {symbol}
{'='*50}

📈 OVERVIEW
• Analysis Period: {days} days
• Total Predictions: {analysis['total_predictions']}
• Predictions with Actuals: {analysis['predictions_with_actuals']}

🎯 ACCURACY METRICS
• Directional Accuracy: {analysis['directional_accuracy']:.1%}
• Mean Absolute Error: ${analysis['error_metrics']['mae']:.2f}
• Root Mean Squared Error: ${analysis['error_metrics']['rmse']:.2f}
• Mean Absolute Percentage Error: {analysis['error_metrics']['mape']:.1f}%

📊 CONFIDENCE ANALYSIS
• Average Confidence: {analysis['confidence_analysis']['avg_confidence']:.2f}
• High Confidence Predictions: {analysis['confidence_analysis']['confidence_distribution']['high (0.7-1.0)']}
• Medium Confidence Predictions: {analysis['confidence_analysis']['confidence_distribution']['medium (0.3-0.7)']}
• Low Confidence Predictions: {analysis['confidence_analysis']['confidence_distribution']['low (0.0-0.3)']}

🔄 DIRECTION BREAKDOWN
• UP Predictions: {analysis['accuracy_by_direction']['up_predictions']} (Correct: {analysis['accuracy_by_direction']['up_correct']})
• DOWN Predictions: {analysis['accuracy_by_direction']['down_predictions']} (Correct: {analysis['accuracy_by_direction']['down_correct']})
"""
        
        if 'trend_analysis' in analysis and 'error' not in analysis['trend_analysis']:
            trend = analysis['trend_analysis']
            report += f"""
📈 TREND ANALYSIS
• MAE Change: ${trend['improvement']['mae_change']:+.2f}
• Directional Accuracy Change: {trend['improvement']['directional_accuracy_change']:+.1%}
• Confidence Change: {trend['improvement']['confidence_change']:+.2f}
"""
        
        return report

class ProfitLossSimulator:
    """Simulate trading performance based on predictions"""
    
    def __init__(self, initial_capital: float = 10000.0, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost  # 0.1% per trade
    
    def simulate_trading(self, predictions: List[PredictionResult], 
                        confidence_threshold: float = 0.6) -> Dict:
        """Simulate trading based on predictions"""
        
        capital = self.initial_capital
        positions = []
        trades = []
        
        for prediction in predictions:
            if prediction.actual_price is None or prediction.confidence < confidence_threshold:
                continue
            
            # Calculate position size (simple: use 10% of capital per trade)
            position_size = capital * 0.1
            shares = position_size / prediction.current_price
            
            # Transaction cost
            transaction_cost = position_size * self.transaction_cost
            
            # Calculate profit/loss
            if prediction.direction == "UP":
                # Long position
                profit_loss = shares * (prediction.actual_price - prediction.current_price) - transaction_cost
            else:
                # Short position (simplified)
                profit_loss = shares * (prediction.current_price - prediction.actual_price) - transaction_cost
            
            capital += profit_loss
            
            trades.append({
                'timestamp': prediction.timestamp,
                'direction': prediction.direction,
                'entry_price': prediction.current_price,
                'exit_price': prediction.actual_price,
                'shares': shares,
                'profit_loss': profit_loss,
                'capital_after': capital,
                'confidence': prediction.confidence
            })
        
        if not trades:
            return {'error': 'No trades executed'}
        
        # Calculate performance metrics
        total_return = capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        profitable_trades = [t for t in trades if t['profit_loss'] > 0]
        win_rate = len(profitable_trades) / len(trades) if trades else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = [t['profit_loss'] / self.initial_capital for t in trades]
        avg_return = np.mean(returns)
        return_std = np.std(returns)
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        
        # Maximum drawdown
        capital_curve = [self.initial_capital]
        for trade in trades:
            capital_curve.append(trade['capital_after'])
        
        peak = capital_curve[0]
        max_drawdown = 0
        for value in capital_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': len(trades),
            'profitable_trades': len(profitable_trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'confidence_threshold': confidence_threshold,
            'trades': trades[-10:]  # Last 10 trades for review
        }

# Global instances
error_tracker = ErrorTracker()
performance_analyzer = PerformanceAnalyzer()
profit_loss_simulator = ProfitLossSimulator()

if __name__ == "__main__":
    print("Performance Metrics Engine - Setting up...")
    print("Available: Error tracking, Performance analysis, P&L simulation")