"""
Data management and persistence for ML Stock Predictor
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import sys
import numpy as np
try:
    from database import db_manager
    from models import StockData, PredictionResult, PerformanceMetrics, ModelMetadata
    from validators import StockDataValidator, PredictionValidator, clean_and_validate_data
except ImportError:
    # Handle relative imports when run as module
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from database import db_manager
    from models import StockData, PredictionResult, PerformanceMetrics, ModelMetadata
    from validators import StockDataValidator, PredictionValidator, clean_and_validate_data

logger = logging.getLogger(__name__)

class StockDataManager:
    """Manages stock data storage and retrieval"""
    
    def __init__(self):
        self.db = db_manager
    
    def save_stock_data(self, stock_data: StockData) -> bool:
        """Save stock data to database"""
        try:
            symbol = stock_data.symbol.upper()
            date = stock_data.date
            # Always use only the date part for saving
            if hasattr(date, 'date'):
                date = date.date()
            # Convert indicators to JSON-serializable
            indicators = stock_data.indicators
            if indicators is not None:
                import numpy as np
                indicators = {k: float(v) if isinstance(v, np.generic) else v for k, v in indicators.items()}
                import json
                indicators = json.dumps(indicators)
            query = """
                INSERT OR REPLACE INTO stock_data (symbol, date, open_price, high_price, low_price, close_price, volume, indicators)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (symbol, date.isoformat(), stock_data.open_price, stock_data.high_price, stock_data.low_price, stock_data.close_price, stock_data.volume, indicators)
            self.db.execute_insert(query, params)
            logger.info(f"Saved stock data for {symbol} on {date}")
            return True
        except Exception as e:
            logger.error(f"Error saving stock data: {e}")
            return False
    
    def get_stock_data(self, symbol: str, date: Optional[datetime] = None) -> Optional[StockData]:
        try:
            symbol = symbol.upper()
            if date:
                if hasattr(date, 'tzinfo') and date.tzinfo is not None:
                    date = date.replace(tzinfo=None)
                query = "SELECT * FROM stock_data WHERE symbol = ? AND date = ?"
                params = (symbol, date.isoformat())
            else:
                query = "SELECT * FROM stock_data WHERE symbol = ? ORDER BY date DESC LIMIT 1"
                params = (symbol,)
            
            rows = self.db.execute_query(query, params)
            if not rows:
                return None
            try:
                stock_data = StockData.from_row(row)
                return stock_data
            except Exception as e:
                import traceback
                print(f"Exception converting row to StockData: {e}")
                traceback.print_exc()
                return None
            
        except Exception as e:
            logger.error(f"Error retrieving stock data: {e}")
            return None
    
    def get_historical_data(self, symbol: str, days: int = 30) -> List[StockData]:
        """Retrieve historical stock data for training"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            # Use only the date part for filtering
            query = '''
                SELECT * FROM stock_data 
                WHERE symbol = ? AND DATE(date) >= ? AND DATE(date) <= ?
                ORDER BY date ASC
            '''
            params = (symbol, start_date.isoformat(), end_date.isoformat())
            rows = self.db.execute_query(query, params)
            stock_data_list = []
            for row in rows:
                indicators = json.loads(row['indicators']) if row['indicators'] else {}
                stock_data = StockData(
                    symbol=row['symbol'],
                    date=datetime.fromisoformat(row['date']),
                    open_price=row['open_price'],
                    high_price=row['high_price'],
                    low_price=row['low_price'],
                    close_price=row['close_price'],
                    volume=row['volume'],
                    indicators=indicators
                )
                stock_data_list.append(stock_data)
            return stock_data_list
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return []

class PredictionManager:
    """Manages prediction storage and retrieval"""
    
    def __init__(self):
        self.db = db_manager
    
    def save_prediction(self, prediction: PredictionResult) -> bool:
        """Save prediction result to database"""
        try:
            # Validate prediction before saving
            is_valid, errors = PredictionValidator.validate_prediction_result(
                prediction.symbol, prediction.current_price, prediction.predicted_price,
                prediction.direction, prediction.confidence, prediction.risk_level
            )
            
            if not is_valid:
                logger.error(f"Invalid prediction: {errors}")
                return False
            
            query = '''
                INSERT INTO predictions 
                (symbol, prediction_date, current_price, predicted_price, direction, 
                 confidence, risk_level, error_metrics, feature_importance, 
                 prediction_stability, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                prediction.symbol,
                prediction.timestamp.isoformat(),
                prediction.current_price,
                prediction.predicted_price,
                prediction.direction,
                prediction.confidence,
                prediction.risk_level,
                json.dumps(prediction.error_metrics),
                json.dumps(prediction.feature_importance),
                prediction.prediction_stability,
                prediction.model_version
            )
            
            self.db.execute_insert(query, params)
            logger.info(f"Saved prediction for {prediction.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return False
    
    def get_recent_predictions(self, symbol: str, days: int = 7) -> List[PredictionResult]:
        """Retrieve recent predictions for analysis"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            query = '''
                SELECT * FROM predictions 
                WHERE symbol = ? AND prediction_date >= ?
                ORDER BY prediction_date DESC
            '''
            
            params = (symbol, start_date.isoformat())
            rows = self.db.execute_query(query, params)
            
            predictions = []
            for row in rows:
                error_metrics = json.loads(row['error_metrics']) if row['error_metrics'] else {}
                feature_importance = json.loads(row['feature_importance']) if row['feature_importance'] else {}
                
                prediction = PredictionResult(
                    symbol=row['symbol'],
                    current_price=row['current_price'],
                    predicted_price=row['predicted_price'],
                    direction=row['direction'],
                    confidence=row['confidence'],
                    risk_level=row['risk_level'],
                    timestamp=datetime.fromisoformat(row['prediction_date']),
                    error_metrics=error_metrics,
                    feature_importance=feature_importance,
                    prediction_stability=row['prediction_stability'],
                    model_version=row['model_version'],
                    actual_price=row['actual_price']
                )
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error retrieving predictions: {e}")
            return []
    
    def update_prediction_with_actual(self, prediction_id: int, actual_price: float) -> bool:
        """Update prediction with actual price for accuracy calculation"""
        try:
            query = "UPDATE predictions SET actual_price = ? WHERE id = ?"
            params = (actual_price, prediction_id)
            
            affected_rows = self.db.execute_update(query, params)
            return affected_rows > 0
            
        except Exception as e:
            logger.error(f"Error updating prediction with actual price: {e}")
            return False

class ModelMetadataManager:
    """Manages model metadata and versioning"""
    
    def __init__(self):
        self.db = db_manager
    
    def save_model_metadata(self, metadata: ModelMetadata) -> bool:
        """Save model metadata to database"""
        try:
            query = '''
                INSERT INTO model_metadata 
                (version, symbol, model_path, performance_metrics, training_params, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                metadata.version,
                metadata.symbol,
                metadata.model_path,
                json.dumps(metadata.performance_metrics.to_dict()),
                json.dumps(metadata.training_params),
                metadata.is_active
            )
            
            self.db.execute_insert(query, params)
            logger.info(f"Saved model metadata for {metadata.symbol} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model metadata: {e}")
            return False
    
    def get_active_model(self, symbol: str) -> Optional[ModelMetadata]:
        """Get the active model for a symbol"""
        try:
            query = '''
                SELECT * FROM model_metadata 
                WHERE symbol = ? AND is_active = 1 
                ORDER BY created_at DESC LIMIT 1
            '''
            
            params = (symbol,)
            rows = self.db.execute_query(query, params)
            
            if not rows:
                return None
            
            row = rows[0]
            performance_metrics = PerformanceMetrics.from_dict(
                json.loads(row['performance_metrics'])
            )
            
            return ModelMetadata(
                version=row['version'],
                symbol=row['symbol'],
                model_path=row['model_path'],
                performance_metrics=performance_metrics,
                training_params=json.loads(row['training_params']),
                created_at=datetime.fromisoformat(row['created_at']),
                is_active=bool(row['is_active'])
            )
            
        except Exception as e:
            logger.error(f"Error retrieving active model: {e}")
            return None

# Global instances
stock_data_manager = StockDataManager()
prediction_manager = PredictionManager()
model_metadata_manager = ModelMetadataManager()

if __name__ == "__main__":
    print("Data Manager - Setting up...")
    print("Database initialized and managers ready")