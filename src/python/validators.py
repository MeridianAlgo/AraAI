"""
Data validation functions for ML Stock Predictor
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class StockDataValidator:
    """Validates stock data integrity and quality"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate stock symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic symbol validation (1-5 uppercase letters)
        pattern = r'^[A-Z]{1,5}$'
        return bool(re.match(pattern, symbol.upper()))
    
    @staticmethod
    def validate_price(price: float, field_name: str) -> bool:
        """Validate price values"""
        if not isinstance(price, (int, float)):
            logger.warning(f"Invalid {field_name}: not a number")
            return False
        
        if price <= 0:
            logger.warning(f"Invalid {field_name}: must be positive")
            return False
        
        if price > 1000000:  # Sanity check for extremely high prices
            logger.warning(f"Suspicious {field_name}: extremely high value")
            return False
        
        return True
    
    @staticmethod
    def validate_volume(volume: int) -> bool:
        """Validate volume values"""
        if not isinstance(volume, (int, float)):
            logger.warning("Invalid volume: not a number")
            return False
        
        if volume < 0:
            logger.warning("Invalid volume: cannot be negative")
            return False
        
        return True
    
    @staticmethod
    def validate_ohlcv_consistency(open_price: float, high_price: float, 
                                 low_price: float, close_price: float) -> bool:
        """Validate OHLC price consistency"""
        # High should be the highest
        if high_price < max(open_price, close_price, low_price):
            logger.warning("Invalid OHLC: high price is not the highest")
            return False
        
        # Low should be the lowest
        if low_price > min(open_price, close_price, high_price):
            logger.warning("Invalid OHLC: low price is not the lowest")
            return False
        
        return True
    
    @staticmethod
    def validate_date(date: datetime) -> bool:
        """Validate date values"""
        if not isinstance(date, datetime):
            logger.warning("Invalid date: not a datetime object")
            return False
        # Make both dates offset-naive for comparison
        if date.tzinfo is not None:
            date = date.replace(tzinfo=None)
        now = datetime.now()
        if now.tzinfo is not None:
            now = now.replace(tzinfo=None)
        # Check if date is not in the future (with some tolerance)
        if date > now:
            logger.warning("Invalid date: cannot be in the future")
            return False
        # Check if date is not too old (more than 10 years)
        if (now - date).days > 3650:
            logger.warning("Invalid date: too old (more than 10 years)")
            return False
        return True
    
    @classmethod
    def validate_stock_data(cls, symbol: str, date: datetime, open_price: float,
                          high_price: float, low_price: float, close_price: float,
                          volume: int, indicators: Optional[Dict[str, float]] = None) -> Tuple[bool, List[str]]:
        """Comprehensive stock data validation"""
        errors = []
        
        # Validate symbol
        if not cls.validate_symbol(symbol):
            errors.append(f"Invalid symbol: {symbol}")
        
        # Validate date
        if not cls.validate_date(date):
            errors.append(f"Invalid date: {date}")
        
        # Validate prices
        if not cls.validate_price(open_price, "open_price"):
            errors.append(f"Invalid open_price: {open_price}")
        
        if not cls.validate_price(high_price, "high_price"):
            errors.append(f"Invalid high_price: {high_price}")
        
        if not cls.validate_price(low_price, "low_price"):
            errors.append(f"Invalid low_price: {low_price}")
        
        if not cls.validate_price(close_price, "close_price"):
            errors.append(f"Invalid close_price: {close_price}")
        
        # Validate volume
        if not cls.validate_volume(volume):
            errors.append(f"Invalid volume: {volume}")
        
        # Validate OHLC consistency (only if individual prices are valid)
        if (cls.validate_price(open_price, "open") and 
            cls.validate_price(high_price, "high") and
            cls.validate_price(low_price, "low") and 
            cls.validate_price(close_price, "close")):
            
            if not cls.validate_ohlcv_consistency(open_price, high_price, low_price, close_price):
                errors.append("OHLC prices are inconsistent")
        
        # Validate indicators if provided
        if indicators:
            for name, value in indicators.items():
                if not isinstance(value, (int, float)):
                    errors.append(f"Invalid indicator {name}: not a number")
                elif not (-1000 <= value <= 1000):  # Reasonable range for most indicators
                    errors.append(f"Invalid indicator {name}: value out of range")
        
        return len(errors) == 0, errors

class PredictionValidator:
    """Validates prediction results and confidence metrics"""
    
    @staticmethod
    def validate_confidence(confidence: float) -> bool:
        """Validate confidence score (should be between 0 and 1)"""
        if not isinstance(confidence, (int, float)):
            return False
        return 0.0 <= confidence <= 1.0
    
    @staticmethod
    def validate_direction(direction: str) -> bool:
        """Validate prediction direction"""
        return direction in ['UP', 'DOWN']
    
    @staticmethod
    def validate_risk_level(risk_level: str) -> bool:
        """Validate risk level"""
        return risk_level in ['LOW', 'MEDIUM', 'HIGH']
    
    @staticmethod
    def validate_prediction_result(symbol: str, current_price: float, 
                                 predicted_price: float, direction: str,
                                 confidence: float, risk_level: str) -> Tuple[bool, List[str]]:
        """Comprehensive prediction result validation"""
        errors = []
        
        # Validate symbol
        if not StockDataValidator.validate_symbol(symbol):
            errors.append(f"Invalid symbol: {symbol}")
        
        # Validate prices
        if not StockDataValidator.validate_price(current_price, "current_price"):
            errors.append(f"Invalid current_price: {current_price}")
        
        if not StockDataValidator.validate_price(predicted_price, "predicted_price"):
            errors.append(f"Invalid predicted_price: {predicted_price}")
        
        # Validate direction
        if not PredictionValidator.validate_direction(direction):
            errors.append(f"Invalid direction: {direction}")
        
        # Validate confidence
        if not PredictionValidator.validate_confidence(confidence):
            errors.append(f"Invalid confidence: {confidence}")
        
        # Validate risk level
        if not PredictionValidator.validate_risk_level(risk_level):
            errors.append(f"Invalid risk_level: {risk_level}")
        
        # Validate direction consistency with price prediction
        price_change = predicted_price - current_price
        if direction == 'UP' and price_change <= 0:
            errors.append("Direction 'UP' inconsistent with predicted price decrease")
        elif direction == 'DOWN' and price_change >= 0:
            errors.append("Direction 'DOWN' inconsistent with predicted price increase")
        
        return len(errors) == 0, errors

class IndicatorValidator:
    """Validates technical indicator values"""
    
    # Define reasonable ranges for common technical indicators
    INDICATOR_RANGES = {
        'rsi': (0, 100),
        'macd': (-10, 10),
        'macd_signal': (-10, 10),
        'macd_histogram': (-5, 5),
        'sma_5': (0, float('inf')),
        'sma_10': (0, float('inf')),
        'sma_20': (0, float('inf')),
        'sma_50': (0, float('inf')),
        'ema_12': (0, float('inf')),
        'ema_26': (0, float('inf')),
        'bollinger_upper': (0, float('inf')),
        'bollinger_lower': (0, float('inf')),
        'bollinger_middle': (0, float('inf')),
        'stochastic_k': (0, 100),
        'stochastic_d': (0, 100),
        'williams_r': (-100, 0),
        'cci': (-300, 300)
    }
    
    @staticmethod
    def validate_indicator(name: str, value: float) -> bool:
        """Validate individual technical indicator"""
        if not isinstance(value, (int, float)):
            return False
        
        if name.lower() in IndicatorValidator.INDICATOR_RANGES:
            min_val, max_val = IndicatorValidator.INDICATOR_RANGES[name.lower()]
            return min_val <= value <= max_val
        
        # Default range for unknown indicators
        return -1000 <= value <= 1000
    
    @staticmethod
    def validate_indicators(indicators: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate all technical indicators"""
        errors = []
        
        for name, value in indicators.items():
            if not IndicatorValidator.validate_indicator(name, value):
                errors.append(f"Invalid indicator {name}: {value}")
        
        return len(errors) == 0, errors

def clean_and_validate_data(raw_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    """Clean and validate raw stock data"""
    errors = []
    cleaned_data = {}
    
    try:
        # Clean and validate basic fields
        symbol = str(raw_data.get('symbol', '')).upper().strip()
        
        # Handle date conversion
        date_str = raw_data.get('date')
        if isinstance(date_str, str):
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        elif isinstance(date_str, datetime):
            date = date_str
        else:
            raise ValueError("Invalid date format")
        
        # Clean price data
        open_price = float(raw_data.get('open', 0))
        high_price = float(raw_data.get('high', 0))
        low_price = float(raw_data.get('low', 0))
        close_price = float(raw_data.get('close', 0))
        volume = int(raw_data.get('volume', 0))
        
        # Clean indicators
        indicators = raw_data.get('indicators', {})
        if isinstance(indicators, str):
            import json
            indicators = json.loads(indicators)
        
        # Validate cleaned data
        is_valid, validation_errors = StockDataValidator.validate_stock_data(
            symbol, date, open_price, high_price, low_price, close_price, volume, indicators
        )
        
        if not is_valid:
            errors.extend(validation_errors)
        
        cleaned_data = {
            'symbol': symbol,
            'date': date,
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price,
            'close_price': close_price,
            'volume': volume,
            'indicators': indicators
        }
        
    except (ValueError, TypeError, KeyError) as e:
        errors.append(f"Data cleaning error: {str(e)}")
        return False, {}, errors
    
    return len(errors) == 0, cleaned_data, errors