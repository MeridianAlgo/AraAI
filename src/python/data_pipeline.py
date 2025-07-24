"""
Training Data Preparation Pipeline for ML Stock Predictor
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime, timedelta
import logging

try:
    from models import StockData
    from indicators import TechnicalIndicators
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from models import StockData
    from indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    """PyTorch Dataset for stock data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DataPreprocessor:
    """Handles data preprocessing and feature engineering"""
    
    def __init__(self):
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.technical_indicators = TechnicalIndicators()
        self.is_fitted = False
    
    def prepare_features(self, stock_data_list: List[StockData], additional_features: Dict = None) -> np.ndarray:
        """Prepare feature matrix from stock data with optional advanced features"""
        if not stock_data_list:
            return np.array([])
        
        features = []
        
        for stock_data in stock_data_list:
            # OHLCV features (5 features)
            ohlcv = [
                stock_data.open_price,
                stock_data.high_price,
                stock_data.low_price,
                stock_data.close_price,
                float(stock_data.volume)
            ]
            
            # Technical indicators (17 features)
            indicators = stock_data.indicators
            technical_features = [
                indicators.get('rsi', 50.0),
                indicators.get('macd', 0.0),
                indicators.get('macd_signal', 0.0),
                indicators.get('macd_histogram', 0.0),
                indicators.get('sma_5', stock_data.close_price),
                indicators.get('sma_10', stock_data.close_price),
                indicators.get('sma_20', stock_data.close_price),
                indicators.get('sma_50', stock_data.close_price),
                indicators.get('ema_12', stock_data.close_price),
                indicators.get('ema_26', stock_data.close_price),
                indicators.get('bollinger_upper', stock_data.close_price * 1.02),
                indicators.get('bollinger_middle', stock_data.close_price),
                indicators.get('bollinger_lower', stock_data.close_price * 0.98),
                indicators.get('stochastic_k', 50.0),
                indicators.get('stochastic_d', 50.0),
                indicators.get('williams_r', -50.0),
                indicators.get('cci', 0.0)
            ]
            
            # Combine basic features
            feature_vector = ohlcv + technical_features
            
            # Add advanced features if provided
            if additional_features:
                advanced_feature_values = []
                for feature_name in sorted(additional_features.keys()):  # Sort for consistency
                    advanced_feature_values.append(additional_features[feature_name])
                feature_vector.extend(advanced_feature_values)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def prepare_targets(self, stock_data_list: List[StockData], prediction_days: int = 1) -> np.ndarray:
        """Prepare target values (next day price changes as percentages)"""
        if len(stock_data_list) < prediction_days + 1:
            return np.array([])
        
        targets = []
        
        # For each data point, the target is the percentage change N days ahead
        for i in range(len(stock_data_list) - prediction_days):
            current_price = stock_data_list[i].close_price
            future_price = stock_data_list[i + prediction_days].close_price
            # Use percentage change instead of absolute price
            price_change_pct = (future_price - current_price) / current_price
            targets.append(price_change_pct)
        
        return np.array(targets).reshape(-1, 1)
    
    def add_technical_indicators(self, stock_data_list: List[StockData]) -> List[StockData]:
        """Add technical indicators to stock data"""
        if not stock_data_list:
            return []
        
        # Convert to format expected by technical indicators
        ohlcv_data = []
        for stock_data in stock_data_list:
            ohlcv_data.append({
                'symbol': stock_data.symbol,
                'date': stock_data.date.isoformat(),
                'open_price': stock_data.open_price,
                'high_price': stock_data.high_price,
                'low_price': stock_data.low_price,
                'close_price': stock_data.close_price,
                'volume': stock_data.volume
            })
        
        # Calculate indicators
        enhanced_data = self.technical_indicators.calculate_all_indicators(ohlcv_data)
        
        # Update stock data with indicators
        for i, stock_data in enumerate(stock_data_list):
            stock_data.indicators = enhanced_data[i]['indicators']
        
        return stock_data_list
    
    def fit_scalers(self, features: np.ndarray, targets: np.ndarray):
        """Fit the scalers on training data"""
        self.feature_scaler.fit(features)
        self.target_scaler.fit(targets)
        self.is_fitted = True
        logger.info("Scalers fitted on training data")
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        return self.feature_scaler.transform(features)
    
    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Transform targets using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        return self.target_scaler.transform(targets)
    
    def inverse_transform_targets(self, targets: np.ndarray) -> np.ndarray:
        """Inverse transform targets to original scale"""
        if not self.is_fitted:
            raise ValueError("Scalers not fitted. Call fit_scalers first.")
        return self.target_scaler.inverse_transform(targets)
    
    def handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing values in features"""
        # Replace NaN with column means
        col_means = np.nanmean(features, axis=0)
        inds = np.where(np.isnan(features))
        features[inds] = np.take(col_means, inds[1])
        
        # Replace any remaining NaN with 0
        features = np.nan_to_num(features)
        
        return features

class TrainingDataPipeline:
    """Complete pipeline for preparing training data"""
    
    def __init__(self, validation_split: float = 0.2, test_split: float = 0.1):
        self.validation_split = validation_split
        self.test_split = test_split
        self.preprocessor = DataPreprocessor()
    
    def prepare_training_data(self, stock_data_list: List[StockData], 
                            prediction_days: int = 1,
                            additional_features: Dict = None) -> Dict[str, DataLoader]:
        """Prepare complete training pipeline with optional advanced features"""
        
        if len(stock_data_list) < 10:
            raise ValueError("Insufficient data for training. Need at least 10 data points.")
        
        logger.info(f"Preparing training data from {len(stock_data_list)} data points")
        
        # Add technical indicators
        stock_data_list = self.preprocessor.add_technical_indicators(stock_data_list)
        
        # Prepare features and targets
        features = self.preprocessor.prepare_features(stock_data_list, additional_features)
        targets = self.preprocessor.prepare_targets(stock_data_list, prediction_days)
        
        # Handle missing values
        features = self.preprocessor.handle_missing_values(features)
        
        # Ensure we have matching samples
        min_samples = min(len(features), len(targets))
        features = features[:min_samples]
        targets = targets[:min_samples]
        
        logger.info(f"Features shape: {features.shape}, Targets shape: {targets.shape}")
        
        # Split data temporally (important for time series)
        train_size = int(min_samples * (1 - self.validation_split - self.test_split))
        val_size = int(min_samples * self.validation_split)
        
        # Training data (oldest)
        train_features = features[:train_size]
        train_targets = targets[:train_size]
        
        # Validation data (middle)
        val_features = features[train_size:train_size + val_size]
        val_targets = targets[train_size:train_size + val_size]
        
        # Test data (newest)
        test_features = features[train_size + val_size:]
        test_targets = targets[train_size + val_size:]
        
        # Fit scalers on training data only
        self.preprocessor.fit_scalers(train_features, train_targets)
        
        # Transform all data
        train_features = self.preprocessor.transform_features(train_features)
        train_targets = self.preprocessor.transform_targets(train_targets)
        
        val_features = self.preprocessor.transform_features(val_features)
        val_targets = self.preprocessor.transform_targets(val_targets)
        
        test_features = self.preprocessor.transform_features(test_features)
        test_targets = self.preprocessor.transform_targets(test_targets)
        
        # Create datasets - ensure targets are properly shaped
        train_dataset = StockDataset(train_features, train_targets.reshape(-1))
        val_dataset = StockDataset(val_features, val_targets.reshape(-1))
        test_dataset = StockDataset(test_features, test_targets.reshape(-1))
        
        # Create data loaders
        batch_size = min(32, len(train_dataset) // 4)  # Adaptive batch size
        
        data_loaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }
        
        logger.info(f"Data prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return data_loaders
    
    def prepare_prediction_data(self, stock_data_list: List[StockData], 
                               additional_features: Dict = None,
                               skip_scaler: bool = False) -> torch.Tensor:
        """Prepare data for prediction (single sample) with optional advanced features"""
        if not stock_data_list:
            raise ValueError("No data provided for prediction")
        
        # Add technical indicators
        stock_data_list = self.preprocessor.add_technical_indicators(stock_data_list)
        
        # Use the latest data point for prediction
        latest_data = stock_data_list[-1:]
        
        # Prepare features
        features = self.preprocessor.prepare_features(latest_data, additional_features)
        features = self.preprocessor.handle_missing_values(features)
        
        # Transform features (scalers should already be fitted)
        if self.preprocessor.is_fitted:
            if skip_scaler:
                return torch.FloatTensor(features)
            features = self.preprocessor.transform_features(features)
        else:
            logger.warning("Scalers not fitted. Using raw features.")
        
        return torch.FloatTensor(features)
    
    def create_sequences(self, stock_data_list: List[StockData], 
                        sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM-style training (future enhancement)"""
        # Add technical indicators
        stock_data_list = self.preprocessor.add_technical_indicators(stock_data_list)
        
        # Prepare features
        features = self.preprocessor.prepare_features(stock_data_list)
        features = self.preprocessor.handle_missing_values(features)
        
        sequences = []
        targets = []
        
        for i in range(len(features) - sequence_length):
            seq = features[i:i + sequence_length]
            target = features[i + sequence_length, 3]  # Close price
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        return [
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'stochastic_k', 'stochastic_d',
            'williams_r', 'cci'
        ]
    
    def get_data_stats(self, stock_data_list: List[StockData]) -> Dict:
        """Get statistics about the data"""
        if not stock_data_list:
            return {}
        
        prices = [data.close_price for data in stock_data_list]
        volumes = [data.volume for data in stock_data_list]
        
        return {
            'num_samples': len(stock_data_list),
            'date_range': {
                'start': min(data.date for data in stock_data_list).isoformat(),
                'end': max(data.date for data in stock_data_list).isoformat()
            },
            'price_stats': {
                'min': min(prices),
                'max': max(prices),
                'mean': np.mean(prices),
                'std': np.std(prices)
            },
            'volume_stats': {
                'min': min(volumes),
                'max': max(volumes),
                'mean': np.mean(volumes),
                'std': np.std(volumes)
            }
        }

# Global instance
training_pipeline = TrainingDataPipeline()

if __name__ == "__main__":
    print("Training Data Preparation Pipeline - Setting up...")
    print("Features: OHLCV (5) + Technical Indicators (17) = 22 total features")