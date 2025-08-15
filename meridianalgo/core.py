"""
Core Ara AI functionality - main prediction engine with ensemble ML system
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from pathlib import Path

from .models import EnsembleMLSystem
from .data import MarketDataManager, TechnicalIndicators
from .utils import GPUManager, CacheManager, AccuracyTracker
from .console import ConsoleManager

class AraAI:
    """
    Main Ara AI class with enhanced ensemble ML system and intelligent caching
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.console = ConsoleManager(verbose=verbose)
        self.gpu_manager = GPUManager()
        self.cache_manager = CacheManager()
        self.accuracy_tracker = AccuracyTracker()
        self.data_manager = MarketDataManager()
        self.indicators = TechnicalIndicators()
        self.ml_system = EnsembleMLSystem(device=self.gpu_manager.get_device())
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the Ara AI system"""
        if self.verbose:
            self.console.print_system_info()
            self.console.print_gpu_info(self.gpu_manager.detect_gpu_vendor())
    
    def predict(self, symbol, days=5, use_cache=True):
        """
        Main prediction function with intelligent caching and enhanced error handling
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to predict
            use_cache (bool): Whether to use cached predictions
            
        Returns:
            dict: Prediction results
        """
        try:
            # Input validation
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Symbol must be a non-empty string")
            
            if not isinstance(days, int) or days < 1 or days > 30:
                raise ValueError("Days must be an integer between 1 and 30")
            
            symbol = symbol.upper().strip()
            
            # Check for existing predictions
            if use_cache:
                try:
                    cached_result = self.cache_manager.check_cached_predictions(symbol, days)
                    if cached_result:
                        choice = self.cache_manager.ask_user_choice(symbol, cached_result)
                        if choice == "use_cached":
                            return self._format_cached_result(cached_result)
                except Exception as cache_error:
                    if self.verbose:
                        self.console.print_warning(f"Cache check failed: {cache_error}")
            
            # Generate new predictions
            return self._generate_new_predictions(symbol, days)
            
        except ValueError as e:
            self.console.print_error(f"Invalid input: {e}")
            return None
        except Exception as e:
            self.console.print_error(f"Prediction failed for {symbol}: {e}")
            if self.verbose:
                import traceback
                self.console.print_error(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def _generate_new_predictions(self, symbol, days):
        """Generate new predictions using ensemble ML system with enhanced error handling"""
        try:
            # Get market data with retry logic
            data = None
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    data = self.data_manager.get_stock_data(symbol)
                    if data is not None and not data.empty:
                        break
                except Exception as data_error:
                    if attempt == max_retries - 1:
                        raise ValueError(f"Failed to fetch data for {symbol} after {max_retries} attempts: {data_error}")
                    if self.verbose:
                        self.console.print_warning(f"Data fetch attempt {attempt + 1} failed, retrying...")
            
            if data is None or data.empty:
                raise ValueError(f"No market data available for {symbol}")
            
            if len(data) < 50:
                raise ValueError(f"Insufficient data for {symbol} (need at least 50 data points, got {len(data)})")
            
            # Calculate technical indicators with error handling
            try:
                enhanced_data = self.indicators.calculate_all_indicators(data)
            except Exception as indicator_error:
                if self.verbose:
                    self.console.print_warning(f"Technical indicator calculation failed: {indicator_error}")
                # Use basic data if indicators fail
                enhanced_data = data
            
            # Prepare features for ML models
            features = self._prepare_features(enhanced_data)
            if features is None:
                raise ValueError("Failed to prepare features for ML models")
            
            # Generate predictions using ensemble system with fallback
            try:
                predictions = self.ml_system.predict(features, days=days)
            except Exception as ml_error:
                if self.verbose:
                    self.console.print_warning(f"ML prediction failed: {ml_error}")
                # Fallback to simple trend prediction
                predictions = self._fallback_prediction(enhanced_data, days)
            
            if not predictions or len(predictions) != days:
                raise ValueError(f"Invalid prediction result: expected {days} predictions, got {len(predictions) if predictions else 0}")
            
            # Format results
            result = self._format_prediction_result(symbol, predictions, data)
            if not result:
                raise ValueError("Failed to format prediction results")
            
            # Save predictions to cache (non-critical)
            try:
                self.cache_manager.save_predictions(symbol, result)
            except Exception as cache_error:
                if self.verbose:
                    self.console.print_warning(f"Failed to save to cache: {cache_error}")
            
            # Update online learning (non-critical)
            try:
                self._update_online_learning(symbol, predictions, data)
            except Exception as learning_error:
                if self.verbose:
                    self.console.print_warning(f"Online learning update failed: {learning_error}")
            
            return result
            
        except ValueError as e:
            self.console.print_error(str(e))
            return None
        except Exception as e:
            self.console.print_error(f"Unexpected error generating predictions for {symbol}: {e}")
            if self.verbose:
                import traceback
                self.console.print_error(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def _prepare_features(self, data):
        """Prepare features for ML models with enhanced error handling"""
        try:
            # Select relevant features for prediction
            feature_columns = [
                'Close', 'Volume', 'High', 'Low', 'Open',
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
                'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI',
                'ATR', 'OBV', 'Price_Change', 'Volume_Change'
            ]
            
            # Filter available columns
            available_columns = [col for col in feature_columns if col in data.columns]
            
            # Ensure we have at least basic OHLCV data
            required_columns = ['Close', 'Volume', 'High', 'Low', 'Open']
            missing_required = [col for col in required_columns if col not in available_columns]
            
            if missing_required:
                raise ValueError(f"Missing required columns: {missing_required}")
            
            if len(available_columns) < 5:
                if self.verbose:
                    self.console.print_warning(f"Limited features available: {len(available_columns)}")
            
            # Prepare features with robust handling
            features_df = data[available_columns].copy()
            
            # Handle missing values more robustly
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Check for infinite values
            features_df = features_df.replace([float('inf'), float('-inf')], 0)
            
            # Validate feature matrix
            features_array = features_df.values
            
            if features_array.shape[0] < 10:
                raise ValueError(f"Insufficient data points: {features_array.shape[0]} (need at least 10)")
            
            if features_array.shape[1] < 5:
                raise ValueError(f"Insufficient features: {features_array.shape[1]} (need at least 5)")
            
            # Check for all-zero columns
            zero_columns = (features_array == 0).all(axis=0)
            if zero_columns.any():
                if self.verbose:
                    zero_col_names = [available_columns[i] for i, is_zero in enumerate(zero_columns) if is_zero]
                    self.console.print_warning(f"Zero-value columns detected: {zero_col_names}")
            
            return features_array
            
        except ValueError as e:
            self.console.print_error(f"Feature preparation error: {e}")
            return None
        except Exception as e:
            self.console.print_error(f"Unexpected error preparing features: {e}")
            return None
    
    def _fallback_prediction(self, data, days):
        """Fallback prediction method when ML models fail"""
        try:
            if self.verbose:
                self.console.print_warning("Using fallback prediction method")
            
            # Simple trend-based prediction
            prices = data['Close'].values
            
            if len(prices) < 5:
                # Not enough data for trend analysis
                current_price = prices[-1]
                return [current_price] * days
            
            # Calculate recent trend (last 10 days or available data)
            trend_period = min(10, len(prices))
            recent_prices = prices[-trend_period:]
            
            # Linear trend calculation
            x = range(trend_period)
            coeffs = np.polyfit(x, recent_prices, 1)
            trend_slope = coeffs[0]
            
            # Generate predictions
            current_price = prices[-1]
            predictions = []
            
            for i in range(days):
                # Apply trend with some dampening to avoid extreme predictions
                dampening_factor = 0.8 ** i  # Reduce trend impact over time
                predicted_price = current_price + (trend_slope * (i + 1) * dampening_factor)
                
                # Ensure prediction is reasonable (within 20% of current price)
                max_change = current_price * 0.2
                predicted_price = max(current_price - max_change, 
                                    min(current_price + max_change, predicted_price))
                
                predictions.append(predicted_price)
            
            return predictions
            
        except Exception as e:
            if self.verbose:
                self.console.print_error(f"Fallback prediction failed: {e}")
            
            # Ultimate fallback - return current price
            try:
                current_price = data['Close'].iloc[-1]
                return [current_price] * days
            except:
                return [100.0] * days  # Last resort
    
    def _format_prediction_result(self, symbol, predictions, data):
        """Format prediction results"""
        try:
            current_price = data['Close'].iloc[-1]
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'predictions': [],
                'timestamp': datetime.now().isoformat(),
                'model_info': self.ml_system.get_model_info()
            }
            
            for i, pred_price in enumerate(predictions):
                pred_date = datetime.now() + timedelta(days=i+1)
                change = pred_price - current_price
                change_pct = (change / current_price) * 100
                
                result['predictions'].append({
                    'day': i + 1,
                    'date': pred_date.isoformat(),
                    'predicted_price': float(pred_price),
                    'change': float(change),
                    'change_pct': float(change_pct)
                })
            
            return result
            
        except Exception as e:
            self.console.print_error(f"Error formatting results: {e}")
            return None
    
    def _format_cached_result(self, cached_data):
        """Format cached prediction results"""
        try:
            # Convert cached data to standard format
            return {
                'symbol': cached_data['symbol'],
                'current_price': cached_data.get('current_price', 0),
                'predictions': cached_data.get('predictions', []),
                'timestamp': cached_data.get('timestamp'),
                'cached': True,
                'cache_age': self._calculate_cache_age(cached_data.get('timestamp'))
            }
        except Exception as e:
            self.console.print_error(f"Error formatting cached result: {e}")
            return None
    
    def _calculate_cache_age(self, timestamp_str):
        """Calculate age of cached data"""
        try:
            if not timestamp_str:
                return "Unknown"
            
            timestamp = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - timestamp
            
            if age.days > 0:
                return f"{age.days}d {age.seconds // 3600}h"
            else:
                return f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m"
                
        except Exception:
            return "Unknown"
    
    def _update_online_learning(self, symbol, predictions, data):
        """Update online learning system"""
        try:
            current_price = data['Close'].iloc[-1]
            validation_summary = self.accuracy_tracker.validate_predictions()
            
            learning_data = {
                'symbol': symbol,
                'prediction': predictions[0] if predictions else current_price,
                'actual_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'validation_summary': validation_summary
            }
            
            # Update ML system with learning data
            self.ml_system.update_online_learning(learning_data)
            
        except Exception as e:
            if self.verbose:
                self.console.print_error(f"Online learning update failed: {e}")
    
    def analyze_accuracy(self, symbol=None):
        """Analyze prediction accuracy"""
        return self.accuracy_tracker.analyze_accuracy(symbol)
    
    def validate_predictions(self):
        """Validate and cleanup old predictions"""
        return self.accuracy_tracker.validate_predictions()
    
    def get_system_info(self):
        """Get system information"""
        return {
            'gpu_info': self.gpu_manager.detect_gpu_vendor(),
            'device': str(self.gpu_manager.get_device()),
            'model_info': self.ml_system.get_model_info(),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'accuracy_stats': self.accuracy_tracker.get_accuracy_stats()
        }

class StockPredictor:
    """Simplified interface for stock prediction (backward compatibility)"""
    
    def __init__(self, verbose=False):
        self.ara = AraAI(verbose=verbose)
    
    def predict(self, symbol, days=5):
        """Predict stock prices"""
        return self.ara.predict(symbol, days=days)
    
    def analyze(self, symbol):
        """Analyze stock with technical indicators"""
        return self.ara.data_manager.get_stock_analysis(symbol)

# Convenience functions for backward compatibility
def predict_stock(symbol, days=5, verbose=False):
    """
    Predict stock prices using Ara AI ensemble system
    
    Args:
        symbol (str): Stock symbol
        days (int): Number of days to predict
        verbose (bool): Enable verbose output
        
    Returns:
        dict: Prediction results
    """
    ara = AraAI(verbose=verbose)
    return ara.predict(symbol, days=days)

def analyze_stock(symbol, verbose=False):
    """
    Analyze stock with technical indicators
    
    Args:
        symbol (str): Stock symbol
        verbose (bool): Enable verbose output
        
    Returns:
        dict: Analysis results
    """
    ara = AraAI(verbose=verbose)
    return ara.data_manager.get_stock_analysis(symbol)