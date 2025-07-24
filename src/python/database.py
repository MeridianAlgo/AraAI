"""
Database connection and schema management for ML Stock Predictor
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages SQLite database connections and operations"""
    
    def __init__(self, db_path: str = "./data/stock_data.db"):
        self.db_path = db_path
        self.ensure_database_directory()
        self.init_database()
    
    def ensure_database_directory(self):
        """Ensure the database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Stock data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    indicators TEXT,  -- JSON string of technical indicators
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            ''')
            
            # Model metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    performance_metrics TEXT,  -- JSON string
                    training_params TEXT,      -- JSON string
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # Training history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    epoch INTEGER NOT NULL,
                    training_loss REAL NOT NULL,
                    validation_loss REAL,
                    accuracy REAL,
                    mae REAL,
                    mse REAL,
                    rmse REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES model_metadata (id)
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    actual_price REAL,  -- Filled later for accuracy calculation
                    error_metrics TEXT,  -- JSON string
                    feature_importance TEXT,  -- JSON string
                    prediction_stability REAL,
                    model_version TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    metric_date TEXT NOT NULL,
                    directional_accuracy REAL,
                    mae REAL,
                    mse REAL,
                    rmse REAL,
                    profit_loss_simulation REAL,
                    rolling_accuracy_1d REAL,
                    rolling_accuracy_7d REAL,
                    rolling_accuracy_30d REAL,
                    confidence_distribution TEXT,  -- JSON string
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_symbol_date ON stock_data(symbol, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON predictions(symbol, prediction_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_symbol_active ON model_metadata(symbol, is_active)')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT query and return the last row ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an UPDATE/DELETE query and return affected rows"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

# Global database instance
db_manager = DatabaseManager()