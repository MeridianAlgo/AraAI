"""
Data models for ML Stock Predictor
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

@dataclass
class StockData:
    """Stock price and volume data with technical indicators"""
    symbol: str
    date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    indicators: Dict[str, float] = field(default_factory=dict)
    
    def __init__(self, symbol, date, open_price, high_price, low_price, close_price, volume, indicators=None):
        self.symbol = symbol
        self.date = date
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.indicators = indicators or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'date': self.date.isoformat(),
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume,
            'indicators': self.indicators
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockData':
        """Create instance from dictionary"""
        return cls(
            symbol=data['symbol'],
            date=datetime.fromisoformat(data['date']),
            open_price=data['open_price'],
            high_price=data['high_price'],
            low_price=data['low_price'],
            close_price=data['close_price'],
            volume=data['volume'],
            indicators=data.get('indicators', {})
        )
    
    def get_ohlcv_array(self) -> List[float]:
        """Get OHLCV data as array for ML model input"""
        return [
            self.open_price,
            self.high_price,
            self.low_price,
            self.close_price,
            float(self.volume)
        ]
    
    def get_indicators_array(self, indicator_names: List[str]) -> List[float]:
        """Get technical indicators as array for ML model input"""
        return [self.indicators.get(name, 0.0) for name in indicator_names]

    @staticmethod
    def from_row(row):
        # Convert sqlite3.Row or tuple to dict if needed
        if not isinstance(row, dict):
            row = dict(row)
        # Extract and convert fields
        symbol = row.get('symbol')
        date = row.get('date')
        from datetime import datetime
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        open_price = float(row.get('open_price', 0) or 0)
        high_price = float(row.get('high_price', 0) or 0)
        low_price = float(row.get('low_price', 0) or 0)
        close_price = float(row.get('close_price', 0) or 0)
        volume = int(row.get('volume', 0) or 0)
        indicators = row.get('indicators')
        import json
        if isinstance(indicators, str):
            indicators = json.loads(indicators) if indicators else {}
        return StockData(symbol, date, open_price, high_price, low_price, close_price, volume, indicators)

@dataclass
class PredictionResult:
    """Stock price prediction result with confidence and metrics"""
    symbol: str
    current_price: float
    predicted_price: float
    direction: str  # 'UP' or 'DOWN'
    confidence: float
    risk_level: str
    timestamp: datetime
    error_metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    prediction_stability: float = 0.0
    model_version: str = "1.0"
    actual_price: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'predicted_price': self.predicted_price,
            'direction': self.direction,
            'confidence': self.confidence,
            'risk_level': self.risk_level,
            'timestamp': self.timestamp.isoformat(),
            'error_metrics': self.error_metrics,
            'feature_importance': self.feature_importance,
            'prediction_stability': self.prediction_stability,
            'model_version': self.model_version,
            'actual_price': self.actual_price
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """Create instance from dictionary"""
        return cls(
            symbol=data['symbol'],
            current_price=data['current_price'],
            predicted_price=data['predicted_price'],
            direction=data['direction'],
            confidence=data['confidence'],
            risk_level=data['risk_level'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            error_metrics=data.get('error_metrics', {}),
            feature_importance=data.get('feature_importance', {}),
            prediction_stability=data.get('prediction_stability', 0.0),
            model_version=data.get('model_version', '1.0'),
            actual_price=data.get('actual_price')
        )
    
    def get_price_change_percent(self) -> float:
        """Calculate predicted price change percentage"""
        return ((self.predicted_price - self.current_price) / self.current_price) * 100
    
    def get_accuracy_if_actual_known(self) -> Optional[float]:
        """Calculate prediction accuracy if actual price is known"""
        if self.actual_price is None:
            return None
        
        predicted_change = self.predicted_price - self.current_price
        actual_change = self.actual_price - self.current_price
        
        # Directional accuracy
        if (predicted_change > 0 and actual_change > 0) or (predicted_change < 0 and actual_change < 0):
            return 1.0
        return 0.0

@dataclass
class PerformanceMetrics:
    """Model performance metrics and analytics"""
    directional_accuracy: float
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    profit_loss_simulation: float
    rolling_accuracy_1d: float
    rolling_accuracy_7d: float
    rolling_accuracy_30d: float
    confidence_distribution: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'directional_accuracy': self.directional_accuracy,
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'profit_loss_simulation': self.profit_loss_simulation,
            'rolling_accuracy_1d': self.rolling_accuracy_1d,
            'rolling_accuracy_7d': self.rolling_accuracy_7d,
            'rolling_accuracy_30d': self.rolling_accuracy_30d,
            'confidence_distribution': self.confidence_distribution
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create instance from dictionary"""
        return cls(
            directional_accuracy=data['directional_accuracy'],
            mae=data['mae'],
            mse=data['mse'],
            rmse=data['rmse'],
            profit_loss_simulation=data['profit_loss_simulation'],
            rolling_accuracy_1d=data['rolling_accuracy_1d'],
            rolling_accuracy_7d=data['rolling_accuracy_7d'],
            rolling_accuracy_30d=data['rolling_accuracy_30d'],
            confidence_distribution=data.get('confidence_distribution', {})
        )

@dataclass
class ModelDiagnostics:
    """Model training diagnostics and analysis"""
    training_loss: List[float]
    validation_loss: List[float]
    convergence_metrics: Dict[str, float]
    overfitting_score: float
    feature_importance_scores: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    stability_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'convergence_metrics': self.convergence_metrics,
            'overfitting_score': self.overfitting_score,
            'feature_importance_scores': self.feature_importance_scores,
            'uncertainty_metrics': self.uncertainty_metrics,
            'stability_score': self.stability_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelDiagnostics':
        """Create instance from dictionary"""
        return cls(
            training_loss=data['training_loss'],
            validation_loss=data['validation_loss'],
            convergence_metrics=data['convergence_metrics'],
            overfitting_score=data['overfitting_score'],
            feature_importance_scores=data['feature_importance_scores'],
            uncertainty_metrics=data['uncertainty_metrics'],
            stability_score=data['stability_score']
        )
    
    def is_overfitting(self, threshold: float = 0.1) -> bool:
        """Check if model is overfitting based on threshold"""
        return self.overfitting_score > threshold
    
    def has_converged(self, patience: int = 10) -> bool:
        """Check if training has converged"""
        if len(self.training_loss) < patience:
            return False
        
        recent_losses = self.training_loss[-patience:]
        return max(recent_losses) - min(recent_losses) < 0.001

@dataclass
class ModelMetadata:
    """Model version and metadata information"""
    version: str
    symbol: str
    model_path: str
    performance_metrics: PerformanceMetrics
    training_params: Dict[str, Any]
    created_at: datetime
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'version': self.version,
            'symbol': self.symbol,
            'model_path': self.model_path,
            'performance_metrics': self.performance_metrics.to_dict(),
            'training_params': self.training_params,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create instance from dictionary"""
        return cls(
            version=data['version'],
            symbol=data['symbol'],
            model_path=data['model_path'],
            performance_metrics=PerformanceMetrics.from_dict(data['performance_metrics']),
            training_params=data['training_params'],
            created_at=datetime.fromisoformat(data['created_at']),
            is_active=data.get('is_active', True)
        )