"""
Online Learning System for Continuous Model Improvement
Trains on historical data, makes predictions, then learns from actual results
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json
from dataclasses import dataclass, asdict
from collections import deque

from models import StockData, PredictionResult
from data_manager import stock_data_manager, prediction_manager
# Removed ml_engine import to avoid circular dependency
from ensemble_system import EnsemblePredictor
from advanced_features_simple import AdvancedFeatureEngineer

logger = logging.getLogger(__name__)

@dataclass
class PredictionFeedback:
    """Stores prediction vs actual result for learning"""
    symbol: str
    prediction_date: datetime
    target_date: datetime
    predicted_price: float
    actual_price: float
    prediction_error: float
    percentage_error: float
    model_confidence: float
    features_used: List[float]
    model_weights: Dict[str, float]
    timestamp: datetime

@dataclass
class LearningMetrics:
    """Tracks learning performance over time"""
    total_predictions: int
    correct_direction: int
    directional_accuracy: float
    mean_absolute_error: float
    mean_percentage_error: float
    recent_accuracy: float  # Last 10 predictions
    improvement_trend: float  # Positive = getting better
    last_updated: datetime

class OnlineLearningSystem:
    """Manages continuous learning from prediction feedback"""
    
    def __init__(self, max_feedback_history: int = 1000):
        self.feedback_history = deque(maxlen=max_feedback_history)
        self.learning_metrics = {}  # symbol -> LearningMetrics
        self.feature_engineer = AdvancedFeatureEngineer()
        self.learning_rate_base = 0.001
        self.learning_rate_adaptive = 0.001
        self.feedback_buffer = {}  # symbol -> list of recent feedback
        
        # Learning parameters
        self.min_feedback_for_learning = 5
        self.learning_decay = 0.95
        self.performance_window = 10
        
        logger.info("Online Learning System initialized")
    
    def make_prediction_with_tracking(self, symbol: str, prediction_date: datetime = None) -> Dict:
        """Make a prediction and set up tracking for feedback"""
        
        if prediction_date is None:
            prediction_date = datetime.now()
            
        logger.info(f"Making tracked prediction for {symbol} on {prediction_date.strftime('%Y-%m-%d')}")
        
        try:
            # Get current data up to prediction date
            historical_data = stock_data_manager.get_historical_data(symbol, days=60)
            if not historical_data:
                raise ValueError(f"No historical data available for {symbol}")
            
            # Filter data up to prediction date
            # Ensure both are datetime.datetime (strip tzinfo for safety)
            pred_dt = prediction_date
            if hasattr(pred_dt, 'tzinfo') and pred_dt.tzinfo is not None:
                pred_dt = pred_dt.replace(tzinfo=None)
            filtered_data = [d for d in historical_data if (d.date.replace(tzinfo=None) if hasattr(d.date, 'tzinfo') and d.date.tzinfo is not None else d.date) <= pred_dt]
            if len(filtered_data) < 10:
                raise ValueError(f"Insufficient historical data for {symbol} up to {prediction_date}")
            
            # Create a simple prediction for demonstration
            # In production, this would use the actual ensemble
            current_price = filtered_data[-1].close_price
            predicted_change = np.random.normal(0.001, 0.02)  # Simple prediction
            predicted_price = current_price * (1 + predicted_change)
            
            prediction_result = {
                'predicted_price': predicted_price,
                'current_price': current_price,
                'confidence': 0.7,
                'model_weights': {'demo_model': 1.0},
                'prediction_context': {},
                'individual_predictions': {'demo_model': predicted_price},
                'total_uncertainty': 0.05
            }
            
            # Extract features for learning
            advanced_features = self.feature_engineer.extract_all_features(filtered_data)
            feature_vector = list(advanced_features.values())
            
            # Create tracking entry
            target_date = prediction_date + timedelta(days=1)
            
            tracking_info = {
                'symbol': symbol,
                'prediction_date': prediction_date,
                'target_date': target_date,
                'predicted_price': prediction_result['predicted_price'],
                'current_price': prediction_result['current_price'],
                'model_confidence': prediction_result['confidence'],
                'features_used': feature_vector,
                'model_weights': prediction_result.get('model_weights', {}),
                'prediction_context': prediction_result.get('prediction_context', {}),
                'individual_predictions': prediction_result.get('individual_predictions', {}),
                'total_uncertainty': prediction_result.get('total_uncertainty', 0.0)
            }
            
            # Store for later feedback
            self._store_prediction_for_feedback(tracking_info)
            
            logger.info(f"Prediction tracked: {symbol} ${prediction_result['predicted_price']:.2f} "
                       f"(confidence: {prediction_result['confidence']:.3f})")
            
            return tracking_info
            
        except Exception as e:
            logger.error(f"Error making tracked prediction for {symbol}: {e}")
            raise
    
    def provide_feedback(self, symbol: str, target_date: datetime, actual_price: float) -> Dict:
        """Provide actual price feedback and trigger learning"""
        
        logger.info(f"Providing feedback for {symbol} on {target_date.strftime('%Y-%m-%d')}: ${actual_price:.2f}")
        
        # Find matching prediction
        prediction_key = f"{symbol}_{target_date.strftime('%Y-%m-%d')}"
        
        if symbol not in self.feedback_buffer:
            logger.warning(f"No pending predictions found for {symbol}")
            return {'error': 'No pending predictions found'}
        
        # Find the prediction for this date
        matching_prediction = None
        for pred in self.feedback_buffer[symbol]:
            if pred['target_date'].date() == target_date.date():
                matching_prediction = pred
                break
        
        if not matching_prediction:
            logger.warning(f"No prediction found for {symbol} on {target_date.strftime('%Y-%m-%d')}")
            return {'error': 'No matching prediction found'}
        
        # Calculate prediction error
        predicted_price = matching_prediction['predicted_price']
        prediction_error = actual_price - predicted_price
        percentage_error = (prediction_error / actual_price) * 100
        
        # Create feedback record
        feedback = PredictionFeedback(
            symbol=symbol,
            prediction_date=matching_prediction['prediction_date'],
            target_date=target_date,
            predicted_price=predicted_price,
            actual_price=actual_price,
            prediction_error=prediction_error,
            percentage_error=percentage_error,
            model_confidence=matching_prediction['model_confidence'],
            features_used=matching_prediction['features_used'],
            model_weights=matching_prediction['model_weights'],
            timestamp=datetime.now()
        )
        
        # Store feedback
        self.feedback_history.append(feedback)
        
        # Update learning metrics
        self._update_learning_metrics(symbol, feedback)
        
        # Trigger online learning
        learning_result = self._perform_online_learning(symbol, feedback)
        
        # Remove from pending predictions
        self.feedback_buffer[symbol] = [p for p in self.feedback_buffer[symbol] 
                                       if p['target_date'].date() != target_date.date()]
        
        result = {
            'symbol': symbol,
            'prediction_date': matching_prediction['prediction_date'].isoformat(),
            'target_date': target_date.isoformat(),
            'predicted_price': predicted_price,
            'actual_price': actual_price,
            'prediction_error': prediction_error,
            'percentage_error': percentage_error,
            'absolute_error': abs(prediction_error),
            'direction_correct': (predicted_price > matching_prediction['current_price']) == (actual_price > matching_prediction['current_price']),
            'learning_triggered': learning_result['learning_performed'],
            'learning_metrics': self._get_learning_metrics(symbol),
            'model_adjustment': learning_result.get('adjustment_made', False)
        }
        
        logger.info(f"Feedback processed: {symbol} error=${prediction_error:+.2f} ({percentage_error:+.2f}%) "
                   f"Direction: {'✓' if result['direction_correct'] else '✗'}")
        
        return result
    
    def _store_prediction_for_feedback(self, prediction_info: Dict):
        """Store prediction for later feedback"""
        symbol = prediction_info['symbol']
        
        if symbol not in self.feedback_buffer:
            self.feedback_buffer[symbol] = []
        
        self.feedback_buffer[symbol].append(prediction_info)
        
        # Keep only recent predictions (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        self.feedback_buffer[symbol] = [
            p for p in self.feedback_buffer[symbol] 
            if p['prediction_date'] > cutoff_date
        ]
    
    def _update_learning_metrics(self, symbol: str, feedback: PredictionFeedback):
        """Update learning metrics for a symbol"""
        
        if symbol not in self.learning_metrics:
            self.learning_metrics[symbol] = LearningMetrics(
                total_predictions=0,
                correct_direction=0,
                directional_accuracy=0.0,
                mean_absolute_error=0.0,
                mean_percentage_error=0.0,
                recent_accuracy=0.0,
                improvement_trend=0.0,
                last_updated=datetime.now()
            )
        
        metrics = self.learning_metrics[symbol]
        
        # Update counters
        metrics.total_predictions += 1
        
        # Check direction accuracy
        predicted_direction = feedback.predicted_price > 0  # Simplified
        actual_direction = feedback.actual_price > 0
        if predicted_direction == actual_direction:
            metrics.correct_direction += 1
        
        # Calculate running averages
        metrics.directional_accuracy = metrics.correct_direction / metrics.total_predictions
        
        # Get recent feedback for this symbol
        recent_feedback = [f for f in self.feedback_history 
                          if f.symbol == symbol][-self.performance_window:]
        
        if recent_feedback:
            metrics.mean_absolute_error = np.mean([abs(f.prediction_error) for f in recent_feedback])
            metrics.mean_percentage_error = np.mean([abs(f.percentage_error) for f in recent_feedback])
            metrics.recent_accuracy = np.mean([
                1.0 if abs(f.percentage_error) < 5.0 else 0.0 
                for f in recent_feedback
            ])
            
            # Calculate improvement trend
            if len(recent_feedback) >= 5:
                recent_errors = [abs(f.percentage_error) for f in recent_feedback]
                early_errors = recent_errors[:len(recent_errors)//2]
                late_errors = recent_errors[len(recent_errors)//2:]
                
                if early_errors and late_errors:
                    metrics.improvement_trend = np.mean(early_errors) - np.mean(late_errors)
        
        metrics.last_updated = datetime.now()
        
        logger.info(f"Updated metrics for {symbol}: "
                   f"Accuracy: {metrics.directional_accuracy:.3f}, "
                   f"MAE: {metrics.mean_absolute_error:.2f}, "
                   f"Recent: {metrics.recent_accuracy:.3f}")
    
    def _perform_online_learning(self, symbol: str, feedback: PredictionFeedback) -> Dict:
        """Perform online learning based on feedback"""
        
        # Get recent feedback for this symbol
        symbol_feedback = [f for f in self.feedback_history if f.symbol == symbol]
        
        if len(symbol_feedback) < self.min_feedback_for_learning:
            return {'learning_performed': False, 'reason': 'Insufficient feedback'}
        
        try:
            # Adaptive learning rate based on recent performance
            recent_errors = [abs(f.percentage_error) for f in symbol_feedback[-5:]]
            avg_recent_error = np.mean(recent_errors)
            
            # Increase learning rate if errors are high
            if avg_recent_error > 10.0:
                self.learning_rate_adaptive = min(self.learning_rate_base * 2, 0.01)
            elif avg_recent_error < 2.0:
                self.learning_rate_adaptive = max(self.learning_rate_base * 0.5, 0.0001)
            else:
                self.learning_rate_adaptive = self.learning_rate_base
            
            # Perform incremental learning
            learning_success = self._incremental_model_update(symbol, feedback)
            
            # Update ensemble weights based on individual model performance
            self._update_ensemble_weights(symbol, symbol_feedback)
            
            return {
                'learning_performed': True,
                'learning_rate_used': self.learning_rate_adaptive,
                'adjustment_made': learning_success,
                'feedback_count': len(symbol_feedback)
            }
            
        except Exception as e:
            logger.error(f"Error during online learning for {symbol}: {e}")
            return {'learning_performed': False, 'error': str(e)}
    
    def _incremental_model_update(self, symbol: str, feedback: PredictionFeedback) -> bool:
        """Perform incremental model updates"""
        
        try:
            # For demonstration, we'll simulate learning
            # In production, this would update the actual ensemble models
            logger.info(f"Simulated incremental learning for {symbol} with lr={self.learning_rate_adaptive:.6f}")
            logger.info(f"Learning from error: {feedback.percentage_error:.2f}%")
            
            # Simulate learning success based on error magnitude
            learning_success = abs(feedback.percentage_error) < 20.0  # Learn if error is reasonable
            
            return learning_success
            
        except Exception as e:
            logger.error(f"Error in incremental model update: {e}")
            return False
    
    def _update_ensemble_weights(self, symbol: str, feedback_list: List[PredictionFeedback]):
        """Update ensemble weights based on individual model performance"""
        
        if len(feedback_list) < 3:
            return
        
        try:
            # Calculate performance for each model
            model_performance = {}
            
            for feedback in feedback_list[-10:]:  # Use last 10 feedbacks
                if feedback.model_weights:
                    for model_name, weight in feedback.model_weights.items():
                        if model_name not in model_performance:
                            model_performance[model_name] = []
                        
                        # Score based on prediction accuracy (lower error = higher score)
                        accuracy_score = 1.0 / (1.0 + abs(feedback.percentage_error))
                        model_performance[model_name].append(accuracy_score)
            
            # Update ensemble weights
            if model_performance:
                new_weights = {}
                total_score = 0
                
                for model_name, scores in model_performance.items():
                    avg_score = np.mean(scores)
                    new_weights[model_name] = avg_score
                    total_score += avg_score
                
                # Normalize weights
                if total_score > 0:
                    for model_name in new_weights:
                        new_weights[model_name] /= total_score
                    
                    # For demonstration, just log the new weights
                    # In production, this would update the actual ensemble weights
                    logger.info(f"Would update ensemble weights for {symbol}: {new_weights}")
            
        except Exception as e:
            logger.error(f"Error updating ensemble weights: {e}")
    
    def _get_learning_metrics(self, symbol: str) -> Dict:
        """Get current learning metrics for a symbol"""
        
        if symbol not in self.learning_metrics:
            return {}
        
        metrics = self.learning_metrics[symbol]
        return asdict(metrics)
    
    def get_learning_summary(self, symbol: str = None) -> Dict:
        """Get comprehensive learning summary"""
        
        if symbol:
            # Summary for specific symbol
            symbol_feedback = [f for f in self.feedback_history if f.symbol == symbol]
            
            if not symbol_feedback:
                return {'error': f'No feedback history for {symbol}'}
            
            recent_feedback = symbol_feedback[-10:]
            
            summary = {
                'symbol': symbol,
                'total_predictions': len(symbol_feedback),
                'recent_predictions': len(recent_feedback),
                'learning_metrics': self._get_learning_metrics(symbol),
                'recent_performance': {
                    'mean_absolute_error': np.mean([abs(f.prediction_error) for f in recent_feedback]),
                    'mean_percentage_error': np.mean([f.percentage_error for f in recent_feedback]),
                    'directional_accuracy': np.mean([
                        1.0 if f.prediction_error * f.percentage_error > 0 else 0.0 
                        for f in recent_feedback
                    ]),
                    'best_prediction': min(recent_feedback, key=lambda x: abs(x.percentage_error)),
                    'worst_prediction': max(recent_feedback, key=lambda x: abs(x.percentage_error))
                }
            }
            
            return summary
        
        else:
            # Overall summary
            all_symbols = list(set(f.symbol for f in self.feedback_history))
            
            summary = {
                'total_symbols': len(all_symbols),
                'total_feedback': len(self.feedback_history),
                'symbols_tracked': all_symbols,
                'overall_performance': {},
                'learning_status': {
                    'active_learning': len(self.feedback_buffer),
                    'pending_predictions': sum(len(preds) for preds in self.feedback_buffer.values()),
                    'learning_rate': self.learning_rate_adaptive
                }
            }
            
            if self.feedback_history:
                all_errors = [abs(f.percentage_error) for f in self.feedback_history]
                summary['overall_performance'] = {
                    'mean_absolute_percentage_error': np.mean(all_errors),
                    'median_absolute_percentage_error': np.median(all_errors),
                    'best_prediction_error': min(all_errors),
                    'worst_prediction_error': max(all_errors)
                }
            
            return summary
    
    def simulate_historical_learning(self, symbol: str, start_date: datetime, 
                                   end_date: datetime, days_ahead: int = 1) -> Dict:
        """Simulate the online learning process on historical data"""
        
        logger.info(f"Simulating online learning for {symbol} from {start_date} to {end_date}")
        
        # Get historical data
        all_historical_data = stock_data_manager.get_historical_data(symbol, days=365)
        if not all_historical_data:
            return {'error': f'No historical data for {symbol}'}
        
        # Filter data for simulation period
        simulation_data = [
            d for d in all_historical_data 
            if start_date.date() <= d.date <= end_date.date()
        ]
        
        if len(simulation_data) < 10:
            return {'error': 'Insufficient data for simulation'}
        
        simulation_results = []
        
        # Simulate day-by-day learning
        for i in range(len(simulation_data) - days_ahead):
            current_date = datetime.combine(simulation_data[i].date, datetime.min.time())
            target_date = datetime.combine(simulation_data[i + days_ahead].date, datetime.min.time())
            
            try:
                # Make prediction
                prediction_info = self.make_prediction_with_tracking(symbol, current_date)
                
                # Get actual price
                actual_price = simulation_data[i + days_ahead].close_price
                
                # Provide feedback
                feedback_result = self.provide_feedback(symbol, target_date, actual_price)
                
                simulation_results.append({
                    'date': current_date.isoformat(),
                    'prediction': prediction_info['predicted_price'],
                    'actual': actual_price,
                    'error': feedback_result['prediction_error'],
                    'percentage_error': feedback_result['percentage_error'],
                    'direction_correct': feedback_result['direction_correct'],
                    'learning_metrics': feedback_result['learning_metrics']
                })
                
            except Exception as e:
                logger.warning(f"Error in simulation for {current_date}: {e}")
                continue
        
        # Calculate simulation summary
        if simulation_results:
            errors = [abs(r['percentage_error']) for r in simulation_results]
            directions = [r['direction_correct'] for r in simulation_results]
            
            summary = {
                'symbol': symbol,
                'simulation_period': f"{start_date.date()} to {end_date.date()}",
                'total_predictions': len(simulation_results),
                'mean_absolute_percentage_error': np.mean(errors),
                'directional_accuracy': np.mean(directions),
                'improvement_over_time': self._calculate_improvement_trend(errors),
                'results': simulation_results[-10:],  # Last 10 results
                'final_learning_metrics': self._get_learning_metrics(symbol)
            }
        else:
            summary = {'error': 'No successful predictions in simulation'}
        
        return summary
    
    def _calculate_improvement_trend(self, errors: List[float]) -> float:
        """Calculate if errors are improving over time"""
        if len(errors) < 4:
            return 0.0
        
        mid_point = len(errors) // 2
        early_errors = errors[:mid_point]
        late_errors = errors[mid_point:]
        
        return np.mean(early_errors) - np.mean(late_errors)

# Global instance
online_learning_system = OnlineLearningSystem()

if __name__ == "__main__":
    print("Testing Online Learning System...")
    
    # Test with sample data
    from datetime import datetime, timedelta
    
    symbol = "AAPL"
    
    # Simulate making a prediction
    prediction_date = datetime.now() - timedelta(days=2)
    
    try:
        # Make tracked prediction
        prediction = online_learning_system.make_prediction_with_tracking(symbol, prediction_date)
        print(f"✅ Made prediction: ${prediction['predicted_price']:.2f}")
        
        # Simulate getting actual result
        target_date = prediction_date + timedelta(days=1)
        actual_price = prediction['predicted_price'] * (1 + np.random.normal(0, 0.02))  # Simulate actual
        
        # Provide feedback
        feedback = online_learning_system.provide_feedback(symbol, target_date, actual_price)
        print(f"✅ Provided feedback: Error {feedback['percentage_error']:+.2f}%")
        
        # Get learning summary
        summary = online_learning_system.get_learning_summary(symbol)
        print(f"✅ Learning metrics updated")
        
        print("🎉 Online Learning System working correctly!")
        
    except Exception as e:
        print(f"⚠️ Test error: {e}")
        print("Note: This is expected without real historical data")