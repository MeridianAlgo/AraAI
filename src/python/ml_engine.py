"""
ML Stock Predictor Python Engine
Main orchestrator for training and prediction
"""

import torch
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
import json
import os
import sys

try:
    from models import StockData, PredictionResult, PerformanceMetrics
    from model import StockPredictor, ModelManager, ModelTrainer
    from data_pipeline import TrainingDataPipeline
    from data_manager import stock_data_manager, prediction_manager
    from indicators import TechnicalIndicators
    from volatility_analyzer import volatility_analyzer
    from ensemble_system import EnsemblePredictor
    from advanced_features_simple import AdvancedFeatureEngineer, AutoFeatureSelector
    from online_learning import online_learning_system
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from models import StockData, PredictionResult, PerformanceMetrics
    from model import StockPredictor, ModelManager, ModelTrainer
    from data_pipeline import TrainingDataPipeline
    from data_manager import stock_data_manager, prediction_manager
    from indicators import TechnicalIndicators
    from volatility_analyzer import volatility_analyzer
    from ensemble_system import EnsemblePredictor
    from advanced_features_simple import AdvancedFeatureEngineer, AutoFeatureSelector
    from online_learning import online_learning_system

logger = logging.getLogger(__name__)

class MLStockEngine:
    """Main ML engine for stock prediction with advanced ensemble capabilities"""
    
    def __init__(self, use_ensemble=True):
        self.model_manager = ModelManager()
        self.training_pipeline = TrainingDataPipeline()
        self.technical_indicators = TechnicalIndicators()
        self.current_model = None
        self.current_symbol = None
        
        # Advanced components
        self.use_ensemble = use_ensemble
        self.ensemble_predictor = None
        self.advanced_feature_engineer = AdvancedFeatureEngineer()
        self.feature_selector = AutoFeatureSelector()
        
        # Multi-timeframe predictions
        self.prediction_horizons = [1, 3, 5, 10]  # days
        
        # Performance tracking
        self.performance_history = {}
        
        logger.info(f"MLStockEngine initialized with ensemble: {use_ensemble}")
        
    def train_model(self, symbol: str, days_history: int = 60, 
                   epochs: int = 100, learning_rate: float = 0.001) -> Dict:
        """Train model(s) for the given symbol - ensemble or single model"""
        
        logger.info(f"Starting {'ensemble' if self.use_ensemble else 'single'} model training for {symbol}")
        
        # Get historical data
        historical_data = stock_data_manager.get_historical_data(symbol, days_history)
        
        if len(historical_data) < 20:
            raise ValueError(f"Insufficient historical data for {symbol}. Need at least 20 days.")
        
        logger.info(f"Retrieved {len(historical_data)} historical data points")
        
        # Extract advanced features
        advanced_features = self.advanced_feature_engineer.extract_all_features(historical_data)
        logger.info(f"Extracted {len(advanced_features)} advanced features")
        
        # Prepare training data with enhanced features
        data_loaders = self.training_pipeline.prepare_training_data(
            historical_data, 
            additional_features=advanced_features
        )
        
        if self.use_ensemble:
            return self._train_ensemble_model(symbol, data_loaders, epochs, learning_rate, days_history)
        else:
            return self._train_single_model(symbol, data_loaders, epochs, learning_rate, days_history)
    
    def _train_ensemble_model(self, symbol: str, data_loaders: Dict, epochs: int, 
                             learning_rate: float, days_history: int) -> Dict:
        """Train ensemble model"""
        
        # Initialize ensemble predictor with correct input size
        input_size = data_loaders['train'].dataset.features.shape[1]
        logger.info(f"Initializing ensemble with input size: {input_size}")
        self.ensemble_predictor = EnsemblePredictor(
            input_size=input_size, 
            device=str(self.model_manager.device)
        )
        
        # Train ensemble
        ensemble_results = self.ensemble_predictor.train_ensemble(
            data_loaders['train'],
            data_loaders['val'],
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        # Evaluate ensemble on test set
        test_metrics = self._evaluate_ensemble(data_loaders['test'])
        
        # Save ensemble
        ensemble_path = f"./data/models/{symbol}_ensemble_v1.0.pth"
        self.ensemble_predictor.save_ensemble(ensemble_path)
        
        # Update current model info
        self.current_symbol = symbol
        
        logger.info(f"Ensemble training completed for {symbol}")
        
        return {
            'symbol': symbol,
            'model_type': 'ensemble',
            'model_path': ensemble_path,
            'ensemble_results': ensemble_results,
            'test_metrics': test_metrics,
            'model_weights': self.ensemble_predictor.weights,
            'training_params': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'days_history': days_history
            }
        }
    
    def _train_single_model(self, symbol: str, data_loaders: Dict, epochs: int, 
                           learning_rate: float, days_history: int) -> Dict:
        """Train single model (original functionality)"""
        
        # Create model
        input_size = data_loaders['train'].dataset.features.shape[1]
        model = self.model_manager.create_model(
            input_size=input_size,
            hidden_sizes=[128, 64, 32],  # Slightly larger for advanced features
            dropout_rate=0.3
        )
        
        # Train model
        trainer = ModelTrainer(model, self.model_manager.device)
        training_results = trainer.train(
            data_loaders['train'], 
            data_loaders['val'],
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        # Evaluate on test set
        test_metrics = self.evaluate_model(model, data_loaders['test'])
        
        # Save model
        metadata = {
            'training_results': training_results,
            'test_metrics': test_metrics,
            'data_stats': self.training_pipeline.get_data_stats(historical_data),
            'training_params': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'days_history': days_history
            }
        }
        
        model_path = self.model_manager.save_model(model, symbol, "1.0", metadata)
        
        # Set as current model
        self.current_model = model
        self.current_symbol = symbol
        
        logger.info(f"Single model training completed for {symbol}")
        
        return {
            'symbol': symbol,
            'model_type': 'single',
            'model_path': model_path,
            'training_results': training_results,
            'test_metrics': test_metrics,
            'model_info': model.get_model_info()
        }
    
    def load_model(self, symbol: str) -> bool:
        """Load the latest model for a symbol"""
        try:
            model_data = self.model_manager.get_latest_model(symbol)
            if model_data:
                self.current_model, metadata = model_data
                self.current_symbol = symbol
                logger.info(f"Loaded model for {symbol}")
                return True
            else:
                logger.warning(f"No model found for {symbol}")
                return False
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            return False
    
    def predict_next_day_price(self, symbol: str, current_data: Optional[StockData] = None) -> PredictionResult:
        """Predict next day price for a symbol"""
        
        # Load model if not current
        if self.current_symbol != symbol or self.current_model is None:
            if not self.load_model(symbol):
                raise ValueError(f"No trained model available for {symbol}")
        
        # Get current data if not provided
        if current_data is None:
            current_data = stock_data_manager.get_stock_data(symbol)
            if current_data is None:
                raise ValueError(f"No current data available for {symbol}")
        
        # Get recent historical data for context
        historical_data = stock_data_manager.get_historical_data(symbol, 30)
        if not historical_data:
            historical_data = [current_data]
        else:
            # Add current data if it's not already the latest
            if historical_data[-1].date < current_data.date:
                historical_data.append(current_data)
        
        # Prepare prediction data
        prediction_features = self.training_pipeline.prepare_prediction_data(historical_data)
        
        # Make prediction with confidence
        predicted_price, confidence = self.model_manager.predict_with_confidence(
            self.current_model, prediction_features
        )
        
        # Convert percentage change prediction to actual price
        if self.training_pipeline.preprocessor.is_fitted:
            predicted_change_array = np.array([[predicted_price]])
            try:
                # Inverse transform to get percentage change
                raw_predicted_change_pct = self.training_pipeline.preprocessor.inverse_transform_targets(
                    predicted_change_array
                )[0][0]
                
            except Exception as e:
                logger.warning(f"Inverse transform failed: {e}. Using raw prediction as percentage.")
                # Fallback: treat raw prediction as percentage change
                raw_predicted_change_pct = predicted_price
        else:
            # No scaling available, treat as percentage change
            raw_predicted_change_pct = predicted_price
        
        # APPLY HISTORICAL VOLATILITY CONSTRAINTS
        # This is the key fix - constrain predictions based on actual historical behavior
        constrained_change_pct, constraint_level = volatility_analyzer.constrain_prediction(
            raw_predicted_change_pct, symbol, confidence
        )
        
        # Get prediction context for additional info
        prediction_context = volatility_analyzer.get_prediction_context(symbol, constrained_change_pct)
        
        # Convert constrained percentage change to actual price
        predicted_price = current_data.close_price * (1 + constrained_change_pct)
        
        logger.info(f"Prediction constrained using {constraint_level}: "
                   f"Raw: {raw_predicted_change_pct*100:.2f}% -> "
                   f"Constrained: {constrained_change_pct*100:.2f}% "
                   f"(Rarity: {prediction_context['rarity']})")
        
        # Determine direction
        direction = "UP" if predicted_price > current_data.close_price else "DOWN"
        
        # Calculate risk level based on confidence and volatility
        price_change_pct = abs(predicted_price - current_data.close_price) / current_data.close_price
        if confidence > 0.7 and price_change_pct < 0.02:
            risk_level = "LOW"
        elif confidence > 0.5 and price_change_pct < 0.05:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Create prediction result
        prediction = PredictionResult(
            symbol=symbol,
            current_price=current_data.close_price,
            predicted_price=predicted_price,
            direction=direction,
            confidence=confidence,
            risk_level=risk_level,
            timestamp=datetime.now(),
            model_version="1.0"
        )
        
        # Save prediction
        prediction_manager.save_prediction(prediction)
        
        logger.info(f"Prediction for {symbol}: ${predicted_price:.2f} ({direction}) - Confidence: {confidence:.2f}")
        
        return prediction
    
    def evaluate_model(self, model: StockPredictor, test_loader) -> Dict:
        """Evaluate model performance"""
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_features, batch_targets in test_loader:
                batch_features = batch_features.to(self.model_manager.device)
                outputs = model(batch_features)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_targets.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        pred_directions = np.sign(np.diff(predictions.flatten()))
        actual_directions = np.sign(np.diff(actuals.flatten()))
        directional_accuracy = np.mean(pred_directions == actual_directions) if len(pred_directions) > 0 else 0.0
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'directional_accuracy': float(directional_accuracy),
            'num_samples': len(predictions)
        }
    
    def get_model_diagnostics(self, symbol: str) -> Dict:
        """Get comprehensive model diagnostics"""
        if self.current_symbol != symbol or self.current_model is None:
            if not self.load_model(symbol):
                return {'error': f'No model available for {symbol}'}
        
        model_info = self.current_model.get_model_info()
        
        # Get recent predictions for analysis
        recent_predictions = prediction_manager.get_recent_predictions(symbol, 30)
        
        diagnostics = {
            'model_info': model_info,
            'symbol': symbol,
            'recent_predictions_count': len(recent_predictions),
            'model_loaded': True
        }
        
        if recent_predictions:
            # Calculate recent performance
            confidences = [p.confidence for p in recent_predictions]
            diagnostics['avg_confidence'] = np.mean(confidences)
            diagnostics['confidence_std'] = np.std(confidences)
            
            # Direction distribution
            directions = [p.direction for p in recent_predictions]
            diagnostics['direction_distribution'] = {
                'UP': directions.count('UP'),
                'DOWN': directions.count('DOWN')
            }
        
        return diagnostics
    
    def retrain_if_needed(self, symbol: str, performance_threshold: float = 0.6) -> bool:
        """Retrain model if performance degrades"""
        recent_predictions = prediction_manager.get_recent_predictions(symbol, 10)
        
        if len(recent_predictions) < 5:
            return False  # Not enough data to assess
        
        # Calculate recent accuracy
        correct_predictions = 0
        total_predictions = 0
        
        for pred in recent_predictions:
            if pred.actual_price is not None:
                predicted_direction = pred.direction
                actual_direction = "UP" if pred.actual_price > pred.current_price else "DOWN"
                
                if predicted_direction == actual_direction:
                    correct_predictions += 1
                total_predictions += 1
        
        if total_predictions == 0:
            return False
        
        accuracy = correct_predictions / total_predictions
        
        if accuracy < performance_threshold:
            logger.info(f"Model performance degraded for {symbol} (accuracy: {accuracy:.2f}). Retraining...")
            try:
                self.train_model(symbol)
                return True
            except Exception as e:
                logger.error(f"Retraining failed for {symbol}: {e}")
                return False
        
        return False
    
    def _evaluate_ensemble(self, test_loader) -> Dict:
        """Evaluate ensemble performance"""
        if not self.ensemble_predictor:
            return {'error': 'No ensemble model available'}
            
        predictions = []
        actuals = []
        uncertainties = []
        
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(self.model_manager.device)
            
            # Get ensemble prediction for each sample in batch
            for i in range(batch_features.shape[0]):
                sample_features = batch_features[i:i+1]
                result = self.ensemble_predictor.predict_with_ensemble(sample_features)
                
                predictions.append(result['ensemble_prediction'])
                uncertainties.append(result['total_uncertainty'])
                actuals.append(batch_targets[i].item())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        uncertainties = np.array(uncertainties)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        
        # Directional accuracy
        pred_directions = np.sign(predictions)
        actual_directions = np.sign(actuals)
        directional_accuracy = np.mean(pred_directions == actual_directions)
        
        # Uncertainty calibration
        avg_uncertainty = np.mean(uncertainties)
        uncertainty_correlation = np.corrcoef(uncertainties, np.abs(predictions - actuals))[0, 1]
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'directional_accuracy': float(directional_accuracy),
            'avg_uncertainty': float(avg_uncertainty),
            'uncertainty_correlation': float(uncertainty_correlation) if not np.isnan(uncertainty_correlation) else 0.0,
            'num_samples': len(predictions)
        }
    
    def predict_with_ensemble(self, symbol: str, current_data: Optional[StockData] = None) -> Dict:
        # Load ensemble if not available
        ensemble_path = f"./data/models/{symbol}_ensemble_v1.0.pth"
        current_input_size = None
        # Get recent historical data for context (to determine input size)
        historical_data = stock_data_manager.get_historical_data(symbol, 30)
        if not historical_data:
            if current_data is not None:
                historical_data = [current_data]
            else:
                raise ValueError(f"No current data available for {symbol}")
        advanced_features = self.advanced_feature_engineer.extract_all_features(historical_data)
        prediction_features = self.training_pipeline.prepare_prediction_data(historical_data, additional_features=advanced_features, skip_scaler=True)
        current_input_size = prediction_features.shape[1]
        # Check if ensemble exists and input size matches
        retrain_ensemble = False
        if os.path.exists(ensemble_path):
            try:
                import torch
                save_data = torch.load(ensemble_path, map_location='cpu')
                saved_input_size = save_data.get('input_size', None)
                if saved_input_size is not None and saved_input_size != current_input_size:
                    logger.warning(f"Ensemble model input size ({saved_input_size}) does not match current features ({current_input_size}). Deleting old model and retraining.")
                    os.remove(ensemble_path)
                    retrain_ensemble = True
            except Exception as e:
                logger.warning(f"Could not check ensemble input size: {e}. Will retrain ensemble.")
                os.remove(ensemble_path)
                retrain_ensemble = True
        if not os.path.exists(ensemble_path) or retrain_ensemble or not self.ensemble_predictor or self.current_symbol != symbol:
            try:
                logger.warning(f"Ensemble model for {symbol} not found or mismatched. Training a new ensemble model...")
                # Auto-train ensemble if missing or mismatched
                historical_data = stock_data_manager.get_historical_data(symbol, 60)
                if len(historical_data) < 20:
                    raise ValueError(f"Insufficient historical data for {symbol}. Need at least 20 days to train ensemble.")
                advanced_features = self.advanced_feature_engineer.extract_all_features(historical_data)
                data_loaders = self.training_pipeline.prepare_training_data(
                    historical_data,
                    additional_features=advanced_features
                )
                self._train_ensemble_model(symbol, data_loaders, epochs=100, learning_rate=0.001, days_history=60)
                logger.info(f"Ensemble model for {symbol} trained and saved.")
                # Save the latest data point as current data
                if len(historical_data) > 0:
                    stock_data_manager.save_stock_data(historical_data[-1])
                # Re-initialize EnsemblePredictor with correct input size and do NOT load old state dicts
                self.ensemble_predictor = EnsemblePredictor(input_size=prediction_features.shape[1], device=str(self.model_manager.device))
                self.current_symbol = symbol
            except Exception as e:
                logger.warning(f"[Prediction Warning] Could not load or train ensemble for {symbol}: {e}. Falling back to single model prediction.")
                return self._convert_single_to_ensemble_format(
                    self.predict_next_day_price(symbol, current_data)
                )
        # Use the already prepared prediction_features for prediction
        ensemble_result = self.ensemble_predictor.predict_with_ensemble(prediction_features)
        
        # Apply volatility constraints to ensemble prediction
        raw_prediction = ensemble_result['ensemble_prediction']
        
        # Robust price fallback logic
        price = getattr(current_data, 'close_price', None)
        if price is None or price == 0:
            for field in ['open_price', 'high_price', 'low_price']:
                price = getattr(current_data, field, None)
                if price is not None and price != 0:
                    logger.warning(f"close_price missing, using {field}={price} as fallback.")
                    break
            else:
                logger.error("No valid price found in current_data for prediction.")
                raise ValueError("No valid price found in current_data for prediction.")
        raw_change_pct = (raw_prediction - price) / price
        
        # Apply constraints
        constrained_change_pct, constraint_level = volatility_analyzer.constrain_prediction(
            raw_change_pct, symbol, ensemble_result['ensemble_confidence']
        )
        
        # Get prediction context
        prediction_context = volatility_analyzer.get_prediction_context(symbol, constrained_change_pct)
        
        # Calculate final constrained price
        constrained_price = price * (1 + constrained_change_pct)
        
        # Determine direction and risk level
        direction = "UP" if constrained_price > price else "DOWN"
        
        # Enhanced risk assessment using ensemble uncertainty
        uncertainty = ensemble_result['total_uncertainty']
        confidence = ensemble_result['ensemble_confidence']
        price_change_pct = abs(constrained_change_pct)
        
        if confidence > 0.8 and uncertainty < 0.02 and price_change_pct < 0.02:
            risk_level = "LOW"
        elif confidence > 0.6 and uncertainty < 0.05 and price_change_pct < 0.05:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Create enhanced prediction result
        enhanced_result = {
            'symbol': symbol,
            'current_price': current_data.close_price,
            'predicted_price': constrained_price,
            'raw_prediction': raw_prediction,
            'direction': direction,
            'confidence': confidence,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            
            # Ensemble-specific information
            'ensemble_prediction': ensemble_result['ensemble_prediction'],
            'individual_predictions': ensemble_result['individual_predictions'],
            'model_weights': ensemble_result['model_weights'],
            'total_uncertainty': uncertainty,
            'individual_uncertainties': ensemble_result['individual_uncertainties'],
            'prediction_intervals': ensemble_result['prediction_intervals'],
            'model_agreement': ensemble_result['model_agreement'],
            
            # Volatility analysis
            'constraint_level': constraint_level,
            'prediction_context': prediction_context,
            'raw_change_pct': raw_change_pct,
            'constrained_change_pct': constrained_change_pct,
            
            # Advanced features used
            'num_features': len(advanced_features),
            'feature_names': list(advanced_features.keys())[:10]  # Top 10 for display
        }
        
        # Save enhanced prediction
        prediction = PredictionResult(
            symbol=symbol,
            current_price=current_data.close_price,
            predicted_price=constrained_price,
            direction=direction,
            confidence=confidence,
            risk_level=risk_level,
            timestamp=datetime.now(),
            model_version="ensemble_v1.0"
        )
        prediction_manager.save_prediction(prediction)
        
        logger.info(f"Ensemble prediction for {symbol}: ${constrained_price:.2f} ({direction}) - "
                   f"Confidence: {confidence:.3f}, Uncertainty: {uncertainty:.3f}")
        
        return enhanced_result
    
    def _convert_single_to_ensemble_format(self, single_prediction: PredictionResult) -> Dict:
        """Convert single model prediction to ensemble format for consistency"""
        
        return {
            'symbol': single_prediction.symbol,
            'current_price': single_prediction.current_price,
            'predicted_price': single_prediction.predicted_price,
            'raw_prediction': single_prediction.predicted_price,
            'direction': single_prediction.direction,
            'confidence': single_prediction.confidence,
            'risk_level': single_prediction.risk_level,
            'timestamp': single_prediction.timestamp.isoformat(),
            
            # Mock ensemble information
            'ensemble_prediction': single_prediction.predicted_price,
            'individual_predictions': {'single_model': single_prediction.predicted_price},
            'model_weights': {'single_model': 1.0},
            'total_uncertainty': 0.05,  # Default uncertainty
            'individual_uncertainties': {'single_model': 0.05},
            'prediction_intervals': {
                '95%': {
                    'lower': single_prediction.predicted_price * 0.95,
                    'upper': single_prediction.predicted_price * 1.05,
                    'width': single_prediction.predicted_price * 0.1
                }
            },
            'model_agreement': {'agreement_score': 1.0, 'coefficient_of_variation': 0.0},
            
            # Volatility analysis (simplified)
            'constraint_level': 'single_model',
            'prediction_context': {'rarity': 'TYPICAL'},
            'raw_change_pct': (single_prediction.predicted_price - single_prediction.current_price) / single_prediction.current_price,
            'constrained_change_pct': (single_prediction.predicted_price - single_prediction.current_price) / single_prediction.current_price,
            
            'num_features': 22,  # Default feature count
            'feature_names': ['basic_indicators']
        }
    
    def get_enhanced_diagnostics(self, symbol: str) -> Dict:
        """Get comprehensive diagnostics including ensemble information"""
        
        diagnostics = self.get_model_diagnostics(symbol)
        
        if self.ensemble_predictor and self.current_symbol == symbol:
            # Add ensemble diagnostics
            ensemble_diag = self.ensemble_predictor.get_ensemble_diagnostics()
            diagnostics.update({
                'model_type': 'ensemble',
                'ensemble_info': ensemble_diag,
                'use_ensemble': True
            })
        else:
            diagnostics.update({
                'model_type': 'single',
                'use_ensemble': False
            })
        
        # Add advanced feature information
        try:
            historical_data = stock_data_manager.get_historical_data(symbol, 30)
            if historical_data:
                advanced_features = self.advanced_feature_engineer.extract_all_features(historical_data)
                diagnostics['advanced_features'] = {
                    'count': len(advanced_features),
                    'feature_names': list(advanced_features.keys())[:20],  # Top 20
                    'feature_categories': self._categorize_features(advanced_features)
                }
        except Exception as e:
            logger.warning(f"Could not extract advanced features for diagnostics: {e}")
            
        return diagnostics
    
    def _categorize_features(self, features: Dict) -> Dict:
        """Categorize features by type"""
        categories = {
            'ichimoku': [],
            'fibonacci': [],
            'volume_profile': [],
            'support_resistance': [],
            'chart_patterns': [],
            'market_structure': [],
            'basic_indicators': []
        }
        
        for feature_name in features.keys():
            if 'ichimoku' in feature_name:
                categories['ichimoku'].append(feature_name)
            elif 'fib' in feature_name:
                categories['fibonacci'].append(feature_name)
            elif 'vp_' in feature_name:
                categories['volume_profile'].append(feature_name)
            elif 'sr_' in feature_name:
                categories['support_resistance'].append(feature_name)
            elif any(pattern in feature_name for pattern in ['trend', 'volatility', 'doji', 'hammer']):
                categories['chart_patterns'].append(feature_name)
            elif 'regime' in feature_name or 'volume_trend' in feature_name:
                categories['market_structure'].append(feature_name)
            else:
                categories['basic_indicators'].append(feature_name)
        
        # Return counts
        return {cat: len(features) for cat, features in categories.items()}
    
    def predict_with_tracking(self, symbol: str, prediction_date: datetime = None) -> Dict:
        """Make a prediction with online learning tracking"""
        return online_learning_system.make_prediction_with_tracking(symbol, prediction_date)
    
    def provide_prediction_feedback(self, symbol: str, target_date: datetime, actual_price: float) -> Dict:
        """Provide feedback for a prediction to enable learning"""
        return online_learning_system.provide_feedback(symbol, target_date, actual_price)
    
    def get_learning_summary(self, symbol: str = None) -> Dict:
        """Get online learning performance summary"""
        return online_learning_system.get_learning_summary(symbol)
    
    def simulate_historical_learning(self, symbol: str, start_date: datetime, 
                                   end_date: datetime, days_ahead: int = 1) -> Dict:
        """Simulate online learning on historical data"""
        return online_learning_system.simulate_historical_learning(
            symbol, start_date, end_date, days_ahead
        )

# Global instance - default to ensemble mode
ml_engine = MLStockEngine(use_ensemble=True)

if __name__ == "__main__":
    print("ML Stock Predictor Python Engine - Setting up...")
    print("Engine ready for training and prediction")