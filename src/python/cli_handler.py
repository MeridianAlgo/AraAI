#!/usr/bin/env python3
"""
CLI Handler for ML Stock Predictor
Handles commands from Node.js CLI interface
"""

import sys
import json
import logging
import torch
from datetime import datetime

# Add current directory to path for imports
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append('.')

try:
    from ml_engine import ml_engine
    from metrics import performance_analyzer, profit_loss_simulator
    from data_manager import stock_data_manager
    from models import StockData
except ImportError as e:
    # Fallback for import issues
    print(json.dumps({
        'success': False,
        'error': f'Import error: {str(e)}. Please ensure all Python dependencies are installed.'
    }))
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise for CLI

def handle_predict(params):
    """Handle prediction request"""
    try:
        symbol = params['symbol']
        verbose = params.get('verbose', False)
        
        # Make prediction
        prediction = ml_engine.predict_next_day_price(symbol)
        
        return {
            'success': True,
            'symbol': prediction.symbol,
            'current_price': prediction.current_price,
            'predicted_price': prediction.predicted_price,
            'direction': prediction.direction,
            'confidence': prediction.confidence,
            'risk_level': prediction.risk_level,
            'timestamp': prediction.timestamp.isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def handle_train(params):
    """Handle training request"""
    try:
        symbol = params['symbol']
        days = params.get('days', 60)
        epochs = params.get('epochs', 100)
        learning_rate = params.get('learning_rate', 0.001)
        
        # Train model
        result = ml_engine.train_model(
            symbol=symbol,
            days_history=days,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        return {
            'success': True,
            'symbol': result['symbol'],
            'training_results': result['training_results'],
            'test_metrics': result['test_metrics'],
            'model_info': result['model_info']
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def handle_analytics(params):
    """Handle analytics request"""
    try:
        symbol = params['symbol']
        days = params.get('days', 30)
        detailed = params.get('detailed', False)
        
        # Get analytics
        analysis = performance_analyzer.analyze_predictions(symbol, days)
        
        if detailed and 'error' not in analysis:
            # Generate detailed report
            report = performance_analyzer.generate_performance_report(symbol, days)
            analysis['detailed_report'] = report
        
        return {
            'success': True,
            **analysis
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def handle_performance(params):
    """Handle performance metrics request"""
    try:
        symbol = params['symbol']
        days = params.get('days', 30)
        
        # Get performance analysis
        analysis = performance_analyzer.analyze_predictions(symbol, days)
        
        # Get profit/loss simulation
        from data_manager import prediction_manager
        predictions = prediction_manager.get_recent_predictions(symbol, days)
        
        if predictions:
            pnl_simulation = profit_loss_simulator.simulate_trading(predictions)
            analysis['profit_loss_simulation'] = pnl_simulation
        
        return {
            'success': True,
            **analysis
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def handle_status(params):
    """Handle status check request"""
    try:
        return {
            'success': True,
            'pytorch_version': torch.__version__,
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat(),
            'ml_engine_ready': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Main CLI handler"""
    if len(sys.argv) < 3:
        print(json.dumps({
            'success': False,
            'error': 'Usage: cli_handler.py <action> <params_json>'
        }))
        sys.exit(1)
    
    action = sys.argv[1]
    try:
        params = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(json.dumps({
            'success': False,
            'error': f'Invalid JSON parameters: {e}'
        }))
        sys.exit(1)
    
    # Route to appropriate handler
    handlers = {
        'predict': handle_predict,
        'train': handle_train,
        'analytics': handle_analytics,
        'performance': handle_performance,
        'status': handle_status
    }
    
    if action not in handlers:
        print(json.dumps({
            'success': False,
            'error': f'Unknown action: {action}'
        }))
        sys.exit(1)
    
    # Execute handler
    try:
        result = handlers[action](params)
        print(json.dumps(result))
        
        if not result.get('success', False):
            sys.exit(1)
            
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': f'Handler error: {str(e)}'
        }))
        sys.exit(1)

if __name__ == '__main__':
    main()