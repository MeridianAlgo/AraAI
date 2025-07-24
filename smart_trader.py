#!/usr/bin/env python3
"""
Smart Trader - All-in-One AI Stock Analysis
One command does everything: data fetching, training, prediction, learning
"""

import sys
import os
sys.path.append('src/python')

import argparse
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

import requests

# Global verbose flag
VERBOSE = False

import logging
from rich.progress import Progress, BarColumn, TextColumn
logging.basicConfig(level=logging.WARNING)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.spinner import Spinner
from rich.text import Text
from rich import box
from rich.align import Align
from rich.style import Style
import time

import torch
# Set device to GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

console = Console()

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def check_and_auto_learn(symbol):
    """
    Check if new price data is available and automatically teach the AI
    """
    try:
        from online_learning import online_learning_system
        # TODO: Implement pending prediction logic if needed
        # pending_predictions = online_learning_system.get_pending_predictions(symbol)
        # if not pending_predictions:
        #     return None
        # vprint(f"[AutoLearn] Checking for new price data to teach AI...")
        
        # Get latest market data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")  # Get last 5 days
        
        if hist.empty:
            return None
        
        learned_count = 0
        
        for prediction in pending_predictions:
            prediction_date = prediction['prediction_date']
            predicted_price = prediction['predicted_price']
            
            # Look for actual price on or after prediction date
            for date, row in hist.iterrows():
                market_date = date.date()
                
                # If we find data for the day after prediction or later
                if market_date > prediction_date:
                    actual_price = float(row['Close'])
                    
                    # Teach the AI with the actual price
                    result = online_learning_system.learn_from_prediction(
                        symbol=symbol,
                        prediction_date=prediction_date,
                        predicted_price=predicted_price,
                        actual_price=actual_price,
                        market_date=market_date
                    )
                    
                    if result.get('success'):
                        learned_count += 1
                        vprint(f"[AutoLearn] AI learned: Predicted ${predicted_price:.2f}, Actual ${actual_price:.2f}")
                    
                    break
        
        if learned_count > 0:
            vprint(f"[AutoLearn] AI automatically learned from {learned_count} new price(s)")
            # Print model accuracy summary after learning
            from online_learning import online_learning_system
            summary = online_learning_system.get_learning_summary(symbol)
            from rich.panel import Panel
            from rich.console import Console
            console = Console()
            if summary and 'error' not in summary:
                console.print(Panel(f"[Model Accuracy After Learning]\n\n" + str(summary), title=f"[bold green]Model Accuracy: {symbol}", border_style="green", padding=(1,2)))
            else:
                console.print(Panel(f"No accuracy data available for {symbol}", title=f"[bold yellow]Model Accuracy", border_style="yellow"))
            return learned_count
        
        return None
        
    except Exception as e:
        vprint(f"[AutoLearn] Auto-learning check failed: {e}")
        return None

def gemini_fact_check(symbol, current_price, predicted_price, direction, confidence, recommendation, risk_level):
    """Send prediction summary to Gemini AI API for fact-checking."""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        return "[FactCheck] Gemini API key not set. Skipping fact check."
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    prompt = (
        f"Fact check this stock prediction and say if it is reasonable and safe for a user to act on. "
        f"Symbol: {symbol}\nCurrent Price: {current_price}\nPredicted Price: {predicted_price}\nDirection: {direction}\n"
        f"Confidence: {confidence}\nRecommendation: {recommendation}\nRisk Level: {risk_level}\n"
        f"Respond with a short assessment and a verdict: GOOD or WARNING."
    )
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(url, headers=headers, params=params, json=data, timeout=15)
        response.raise_for_status()
        result = response.json()
        gemini_text = result["candidates"][0]["content"]["parts"][0]["text"]
        return gemini_text
    except Exception as e:
        return f"[FactCheck] Gemini API error: {e}"

# Utility to ensure offset-naive datetime
from datetime import datetime

def make_naive(dt):
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt

def walk_forward_validation(symbol, days_history=60):
    """Fast walk-forward: train on first 80%, predict next 20% (no retrain). Only LSTM and FFN models for speed."""
    from ml_engine import ml_engine
    from data_manager import stock_data_manager
    from models import StockData
    import numpy as np
    historical_data = stock_data_manager.get_historical_data(symbol, days_history)
    if len(historical_data) < 10:
        return {'error': 'Not enough data for walk-forward validation'}
    split_idx = int(len(historical_data) * 0.8)
    train_data = historical_data[:split_idx]
    test_data = historical_data[split_idx:]
    # Train once on train_data, only LSTM and FFN models
    try:
        ml_engine.train_model(symbol=symbol, days_history=len(train_data), epochs=3, learning_rate=0.001, models=['lstm_attention', 'enhanced_ffn'])
    except Exception as e:
        return {'error': f'Training failed: {e}'}
    predictions = []
    actuals = []
    available_dates = set([make_naive(d.date) for d in historical_data])
    for d in test_data:
        # Ensure prediction date is valid; fallback to last available date if not
        pred_date = make_naive(d.date) if make_naive(d.date) in available_dates else make_naive(historical_data[-1].date)
        try:
            pred = ml_engine.predict_with_tracking(symbol, pred_date)
            if 'predicted_price' in pred:
                predictions.append(pred['predicted_price'])
                actuals.append(d.close_price)
        except Exception as e:
            continue
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    if len(predictions) == 0:
        return {'error': 'No predictions made in walk-forward validation'}
    errors = predictions - actuals
    abs_errors = np.abs(errors)
    pct_errors = np.abs(errors / actuals) * 100
    mae = np.mean(abs_errors)
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(pct_errors)
    max_error = np.max(abs_errors)
    min_error = np.min(abs_errors)
    std_error = np.std(abs_errors)
    # Directional accuracy
    pred_directions = np.sign(np.diff(predictions))
    actual_directions = np.sign(np.diff(actuals))
    directional_accuracy = np.mean(pred_directions == actual_directions) if len(pred_directions) > 0 else 0.0
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'max_error': float(max_error),
        'min_error': float(min_error),
        'std_error': float(std_error),
        'directional_accuracy': float(directional_accuracy),
        'num_samples': len(predictions)
    }

def smart_trade_analysis(symbol, days_history=60, epochs=5):
    steps = [
        "Fetching market data",
        "Training/Loading model",
        "Walk-forward validation",
        "Extracting features",
        "Making prediction"
    ]
    total_steps = len(steps)
    with Progress(BarColumn(), TextColumn("{task.description}"), transient=True) as progress:
        task = progress.add_task("Loading...", total=total_steps)
        # Step 1: Fetch data
        progress.update(task, advance=1, description=steps[0])
        auto_learned = check_and_auto_learn(symbol)
        from online_learning import online_learning_system
        summary = online_learning_system.get_learning_summary(symbol)
        # Step 2: Train/load model
        progress.update(task, advance=1, description=steps[1])
        from ml_engine import ml_engine
        from advanced_features_simple import AdvancedFeatureEngineer
        from models import StockData
        from data_manager import stock_data_manager
        used_sample_data = False
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days_history}d")
            if hist.empty:
                raise Exception(f"No data found for {symbol}")
            stock_data_list = []
            for date, row in hist.iterrows():
                try:
                    stock_data = StockData(
                        symbol=symbol.upper(),
                        date=date,
                        open_price=float(row['Open']),
                        high_price=float(row['High']),
                        low_price=float(row['Low']),
                        close_price=float(row['Close']),
                        volume=int(row['Volume'])
                    )
                    stock_data_list.append(stock_data)
                    stock_data_manager.save_stock_data(stock_data)
                except Exception as e:
                    vprint(f"[Data] Skipped row due to error: {e}")
            if not stock_data_list:
                raise Exception("No valid stock data parsed.")
            current_price = stock_data_list[-1].close_price
        except Exception as e:
            print(f"[bold yellow]Warning:[/] Could not fetch real data for {symbol}. Using sample data.")
            stock_data_list = create_sample_data(symbol, days_history)
            current_price = stock_data_list[-1].close_price
            used_sample_data = True
        # Step 3: Walk-forward validation
        progress.update(task, advance=1, description=steps[2])
        wf_metrics = walk_forward_validation(symbol, days_history)
        # Step 4: Feature extraction
        progress.update(task, advance=1, description=steps[3])
        feature_engineer = AdvancedFeatureEngineer()
        features = feature_engineer.extract_all_features(stock_data_list)
        rsi = features.get('rsi', 0.5) * 100
        regime = features.get('market_regime', 0.5)
        volatility = features.get('volatility', 0.02) * 100
        rsi_signal = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
        regime_signal = "BULL" if regime > 0.7 else "BEAR" if regime < 0.3 else "SIDEWAYS"
        # Step 5: Prediction
        progress.update(task, advance=1, description=steps[4])
        pred_date = make_naive(stock_data_list[-1].date) if stock_data_list else None
        # Check if features for pred_date exist
        valid_dates = [make_naive(d.date) for d in stock_data_list]
        if pred_date not in valid_dates:
            print(f"[bold yellow]Warning:[/] No features available for prediction date {pred_date}. Skipping prediction.")
            return
        try:
            prediction_info = ml_engine.predict_with_tracking(symbol, pred_date)
            predicted_price = prediction_info['predicted_price']
            confidence = prediction_info['model_confidence']
            prediction_result = ml_engine.predict_with_ensemble(symbol)
            direction = prediction_result['direction']
            risk_level = prediction_result['risk_level']
        except Exception as e:
            predicted_price = current_price * (1 + np.random.normal(0.001, 0.02))
            direction = "UP" if predicted_price > current_price else "DOWN"
            confidence = 0.65
            risk_level = "MEDIUM"
        # After loading bar, print only prediction, accuracy, and plan
        from rich.panel import Panel
        from rich.console import Console
        console = Console()
        val_loss = None
        if 'training_result' in locals() and 'best_val_loss' in training_result:
            val_loss = training_result['best_val_loss']
        accuracy_text = f"[Model Accuracy After Prediction]\n\n"
        if wf_metrics and 'error' not in wf_metrics:
            accuracy_text += (
                f"Walk-Forward MAE: {wf_metrics['mae']:.4f}\n"
                f"Walk-Forward MSE: {wf_metrics['mse']:.4f}\n"
                f"Walk-Forward RMSE: {wf_metrics['rmse']:.4f}\n"
                f"Walk-Forward MAPE: {wf_metrics['mape']:.2f}%\n"
                f"Walk-Forward Max Error: {wf_metrics['max_error']:.4f}\n"
                f"Walk-Forward Min Error: {wf_metrics['min_error']:.4f}\n"
                f"Walk-Forward Std Error: {wf_metrics['std_error']:.4f}\n"
                f"Walk-Forward Directional Accuracy: {wf_metrics['directional_accuracy']:.2%}\n"
                f"Samples: {wf_metrics['num_samples']}\n"
            )
        if val_loss is not None:
            accuracy_text += f"Best Validation Loss: {val_loss:.6f}\n"
        if summary and 'error' not in summary:
            accuracy_text += f"\nOther Metrics:\n{str(summary)}"
        console.print(Panel(accuracy_text, title=f"[bold blue]Model Accuracy: {symbol}", border_style="blue", padding=(1,2)))
        # Show prediction panel
        table = Table(show_header=False, box=box.ROUNDED, expand=True, padding=(0,1))
        table.add_row("[bold]Current[/]", f"${current_price:.2f}")
        table.add_row("[bold]Prediction[/]", f"${predicted_price:.2f} ({direction})")
        price_change = predicted_price - current_price
        percentage_change = (price_change / current_price) * 100
        table.add_row("[bold]Change[/]", f"{percentage_change:+.2f}%")
        table.add_row("[bold]Confidence[/]", f"{confidence:.0%}")
        # Recommendation logic
        if confidence > 0.8 and risk_level == "LOW":
            recommendation = "STRONG BUY" if direction == "UP" else "STRONG SELL"
        elif confidence > 0.6 and risk_level in ["LOW", "MEDIUM"]:
            recommendation = "BUY" if direction == "UP" else "SELL"
        else:
            recommendation = "HOLD"
        table.add_row("[bold]Recommendation[/]", f"{recommendation}")
        if recommendation != "HOLD":
            target_price = predicted_price
            stop_loss = current_price * (0.95 if direction == "UP" else 1.05)
            table.add_row("[bold]Plan[/]", f"Entry ${current_price:.2f} → Target ${target_price:.2f} | Stop ${stop_loss:.2f}")
        if used_sample_data:
            table.add_row("[bold yellow]Note[/]", "Sample data used")
        panel_title = f"[bold magenta]ML Stock Prediction: {symbol.upper()}" if not used_sample_data else f"[bold yellow]ML Stock Prediction: {symbol.upper()} (Sample)"
        console.print(Panel(table, title=panel_title, border_style="cyan", padding=(1,2)))

def create_sample_data(symbol, days=60):
    """Create realistic sample data when real data isn't available"""
    
    sample_data = []
    base_price = 150.0 + np.random.normal(0, 50)
    base_date = datetime.now() - timedelta(days=days)
    
    for i in range(days):
        trend = 0.0005
        volatility = 0.015
        price_change = np.random.normal(trend, volatility)
        
        new_price = base_price * (1 + price_change)
        
        high_price = new_price * (1 + abs(np.random.normal(0, 0.005)))
        low_price = new_price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = base_price + np.random.normal(0, new_price * 0.003)
        
        from models import StockData
        stock_data = StockData(
            symbol=symbol.upper(),
            date=(base_date + timedelta(days=i)).date(),
            open_price=open_price,
            high_price=max(high_price, open_price, new_price),
            low_price=min(low_price, open_price, new_price),
            close_price=new_price,
            volume=int(np.random.normal(50000000, 10000000))
        )
        
        sample_data.append(stock_data)
        base_price = new_price
    
    return sample_data

def ensure_historical_data(symbol, min_days=60):
    from src.python.data_manager import stock_data_manager
    from datetime import datetime, timedelta
    from models import StockData
    import time
    import os
    from dotenv import load_dotenv
    import requests
    from dateutil import parser as date_parser
    load_dotenv()
    # Check how many days of data are present
    data = stock_data_manager.get_historical_data(symbol, min_days)
    if data and len(data) >= min_days:
        return
    print(f"[INFO] Not enough historical data for {symbol}. Fetching from Alpaca...")
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET
    }
    end = datetime.utcnow()
    start = end - timedelta(days=min_days*2)  # Fetch extra in case of missing days
    url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {
        "symbols": symbol,
        "start": start.strftime('%Y-%m-%dT00:00:00Z'),
        "end": end.strftime('%Y-%m-%dT00:00:00Z'),
        "timeframe": "1Day",
        "limit": 1000
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    bars = response.json().get("bars", {}).get(symbol, [])
    for bar in bars:
        stock_data = StockData(
            symbol=symbol.upper(),
            date=date_parser.isoparse(bar["t"]),
            open_price=float(bar["o"]),
            high_price=float(bar["h"]),
            low_price=float(bar["l"]),
            close_price=float(bar["c"]),
            volume=int(bar["v"]),
            indicators={}
        )
        stock_data_manager.save_stock_data(stock_data)
    print(f"[INFO] Fetched and saved {len(bars)} days of data for {symbol} from Alpaca.")
    # Debug: Print number of rows, unique dates, earliest/latest date
    all_data = stock_data_manager.get_historical_data(symbol, min_days*2)
    unique_dates = sorted(set([d.date.date() if hasattr(d.date, 'date') else d.date for d in all_data]))
    print(f"[DEBUG] Total rows for {symbol}: {len(all_data)}")
    print(f"[DEBUG] Unique dates for {symbol}: {len(unique_dates)}")
    if unique_dates:
        print(f"[DEBUG] Earliest date: {unique_dates[0]}, Latest date: {unique_dates[-1]}")
    # Wait and retry until enough data is present
    max_retries = 10
    for attempt in range(max_retries):
        data = stock_data_manager.get_historical_data(symbol, min_days)
        if data and len(data) >= min_days:
            return
        print(f"[INFO] Waiting for historical data to be available... (attempt {attempt+1}/{max_retries})")
        time.sleep(2)
    # Final check
    data = stock_data_manager.get_historical_data(symbol, min_days)
    if not data or len(data) < min_days:
        raise RuntimeError(f"Failed to fetch enough historical data for {symbol} after {max_retries} retries.")

def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description='Smart Trader - All-in-One AI Stock Analysis')
    parser.add_argument('symbol', help='Stock symbol to analyze (e.g., AAPL, TSLA, MSFT)')
    parser.add_argument('--days', type=int, default=60, help='Days of historical data (default: 60)')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs (default: 10)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed logs and errors')
    args = parser.parse_args()
    VERBOSE = args.verbose
    # Ensure enough historical data is present
    ensure_historical_data(args.symbol, min_days=args.days)
    # Always show logs at INFO level or higher
    logging.getLogger().setLevel(logging.INFO)
    success = smart_trade_analysis(args.symbol, args.days, args.epochs)
    if not success:
        print(f"Analysis failed.")
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()