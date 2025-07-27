"""
ML Predictor Module for MeridianAlgo
Provides machine learning-based stock price prediction capabilities
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MLPredictor:
    """
    Machine Learning predictor for stock price forecasting
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the ML predictor
        
        Args:
            device: Device to use for computation ("auto", "cpu", "cuda", "mps")
        """
        self.device = self._get_device(device) if TORCH_AVAILABLE else "cpu"
        self.scaler_features = MinMaxScaler() if TORCH_AVAILABLE else None
        self.scaler_target = MinMaxScaler() if TORCH_AVAILABLE else None
        self.model = None
        
    def _get_device(self, device: str) -> str:
        """Get the best available device for computation"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def fetch_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """
        Fetch stock data for the given symbol
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            days: Number of days of historical data
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 30)  # Extra buffer
            
            data = ticker.history(start=start_date, end=end_date)
            if len(data) < days:
                raise ValueError(f"Insufficient data: only {len(data)} days available")
            
            return data
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Data with additional technical indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
        bb_std_dev = df['Close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std_dev * bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std_dev * bb_std)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Momentum'] = df['Close'].pct_change(periods=5)
        
        return df.dropna()
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for ML training
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and target arrays
        """
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Position', 'Volume_Ratio', 'Price_Change', 'Price_Momentum'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in data.columns]
        X = data[available_features].values
        
        # Target is next day's close price
        y = data['Close'].shift(-1).dropna().values
        X = X[:-1]  # Remove last row to match target length
        
        return X, y
    
    def predict_simple(self, symbol: str, days: int = 60, forecast_days: int = 5) -> Dict:
        """
        Simple prediction using statistical methods (no ML dependencies required)
        
        Args:
            symbol: Stock symbol
            days: Historical data days
            forecast_days: Number of days to forecast
            
        Returns:
            Dict: Prediction results
        """
        try:
            # Fetch data
            data = self.fetch_data(symbol, days)
            
            # Calculate technical indicators
            data_with_indicators = self.calculate_technical_indicators(data)
            
            # Simple ensemble prediction using multiple methods
            current_price = data_with_indicators['Close'].iloc[-1]
            predictions = []
            
            # Method 1: Trend-based prediction
            short_trend = data_with_indicators['Close'].tail(5).pct_change().mean()
            medium_trend = data_with_indicators['Close'].tail(10).pct_change().mean()
            long_trend = data_with_indicators['Close'].tail(20).pct_change().mean()
            
            # Method 2: Moving average convergence
            sma_10 = data_with_indicators['SMA_10'].iloc[-1]
            sma_20 = data_with_indicators['SMA_20'].iloc[-1]
            ma_signal = (sma_10 - sma_20) / sma_20
            
            # Method 3: RSI-based adjustment
            rsi = data_with_indicators['RSI'].iloc[-1]
            rsi_adjustment = 0
            if rsi > 70:  # Overbought
                rsi_adjustment = -0.01
            elif rsi < 30:  # Oversold
                rsi_adjustment = 0.01
            
            # Generate predictions
            for i in range(forecast_days):
                day_ahead = i + 1
                
                # Weighted trend combination
                trend_signal = (short_trend * 0.5 + medium_trend * 0.3 + long_trend * 0.2)
                
                # Apply moving average signal
                ma_influence = ma_signal * 0.005 * (1 / day_ahead)
                
                # Apply RSI adjustment
                rsi_influence = rsi_adjustment * (1 / day_ahead)
                
                # Random walk component (small)
                random_component = np.random.normal(0, 0.005)
                
                # Combine signals
                total_change = trend_signal + ma_influence + rsi_influence + random_component
                
                # Apply to current price
                predicted_price = current_price * (1 + total_change * day_ahead)
                
                # Ensure reasonable bounds
                max_change = 0.1 * day_ahead  # Max 10% change per day
                predicted_price = max(min(predicted_price, 
                                        current_price * (1 + max_change)),
                                    current_price * (1 - max_change))
                
                predictions.append(predicted_price)
            
            # Calculate confidence based on data quality and volatility
            volatility = data_with_indicators['Price_Change'].std()
            data_quality = min(len(data_with_indicators) / 60.0, 1.0)
            confidence = max(60, 85 - (volatility * 100)) * data_quality
            
            return {
                'symbol': symbol.upper(),
                'current_price': current_price,
                'predictions': predictions,
                'confidence': confidence,
                'method': 'Statistical Ensemble',
                'technical_indicators': {
                    'rsi': rsi,
                    'sma_10': sma_10,
                    'sma_20': sma_20,
                    'trend_short': short_trend,
                    'trend_medium': medium_trend,
                    'trend_long': long_trend
                }
            }
            
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def predict_ml(self, symbol: str, days: int = 60, epochs: int = 10, 
                   forecast_days: int = 5) -> Dict:
        """
        Advanced ML-based prediction (requires PyTorch)
        
        Args:
            symbol: Stock symbol
            days: Historical data days
            epochs: Training epochs
            forecast_days: Number of days to forecast
            
        Returns:
            Dict: Prediction results
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Falling back to simple prediction.")
            return self.predict_simple(symbol, days, forecast_days)
        
        try:
            # Fetch and prepare data
            data = self.fetch_data(symbol, days)
            data_with_indicators = self.calculate_technical_indicators(data)
            X, y = self.prepare_features(data_with_indicators)
            
            # Scale features
            X_scaled = self.scaler_features.fit_transform(X)
            y_scaled = self.scaler_target.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Create simple neural network
            input_size = X_scaled.shape[1]
            model = SimpleNN(input_size).to(self.device)
            
            # Train model
            model = self._train_model(model, X_scaled, y_scaled, epochs)
            
            # Make predictions
            predictions_scaled = self._make_predictions(model, X_scaled, forecast_days)
            predictions = self.scaler_target.inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).flatten()
            
            # Calculate confidence
            current_price = data_with_indicators['Close'].iloc[-1]
            confidence = self._calculate_confidence(X, y, predictions)
            
            return {
                'symbol': symbol.upper(),
                'current_price': current_price,
                'predictions': predictions.tolist(),
                'confidence': confidence,
                'method': 'Neural Network',
                'device': self.device,
                'epochs': epochs
            }
            
        except Exception as e:
            print(f"ML prediction failed: {str(e)}. Falling back to simple prediction.")
            return self.predict_simple(symbol, days, forecast_days)
    
    def _train_model(self, model, X, y, epochs: int):
        """Train the neural network model"""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
        
        return model
    
    def _make_predictions(self, model, X, forecast_days: int) -> np.ndarray:
        """Make predictions using the trained model"""
        model.eval()
        predictions = []
        
        # Use last sequence for prediction
        last_sequence = torch.FloatTensor(X[-1:]).to(self.device)
        
        with torch.no_grad():
            for _ in range(forecast_days):
                pred = model(last_sequence)
                predictions.append(pred.cpu().numpy()[0])
                
                # Update sequence for next prediction (simple approach)
                # In practice, you'd want a more sophisticated sequence update
        
        return np.array(predictions)
    
    def _calculate_confidence(self, X, y, predictions) -> float:
        """Calculate prediction confidence"""
        # Simple confidence based on data quality and prediction consistency
        data_quality = min(len(X) / 60.0, 1.0) * 100
        
        if len(predictions) > 1:
            pred_std = np.std(predictions[:3])  # First 3 predictions
            consistency = max(50, 90 - (pred_std * 10))
        else:
            consistency = 75
        
        return min(max((data_quality + consistency) / 2, 65), 92)


class SimpleNN(nn.Module):
    """Simple neural network for stock prediction"""
    
    def __init__(self, input_size: int):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)