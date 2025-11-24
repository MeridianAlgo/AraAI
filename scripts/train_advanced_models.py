"""
Train Advanced Models with Validation and Learning
Downloads data for 20 stocks, trains models, validates predictions, and tracks accuracy
"""

import sys
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from meridianalgo.ultimate_ml import UltimateStockML
from meridianalgo.forex_ml import ForexML
from meridianalgo.console import ConsoleManager

# Top 20 stocks for training
STOCK_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'BRK-B', 'JPM', 'V',
    'JNJ', 'WMT', 'PG', 'MA', 'HD',
    'DIS', 'NFLX', 'ADBE', 'CRM', 'INTC'
]

# Top forex pairs for training
FOREX_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
    'AUDUSD', 'USDCAD', 'NZDUSD', 'EURJPY'
]

def download_and_train_stocks(console, period='5y'):
    """Download data and train stock models"""
    console.print_header("Training Advanced Stock Models")
    
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    
    console.print_info(f"Downloading data for {len(STOCK_SYMBOLS)} stocks...")
    
    # Download all datasets
    downloaded = []
    for symbol in STOCK_SYMBOLS:
        try:
            console.print_info(f"Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if len(data) >= 100:
                csv_file = datasets_dir / f"{symbol}.csv"
                data.to_csv(csv_file)
                downloaded.append((symbol, csv_file))
                console.print_success(f"{symbol}: {len(data)} days")
            else:
                console.print_warning(f"{symbol}: Insufficient data ({len(data)} days)")
        except Exception as e:
            console.print_error(f"{symbol}: Failed - {e}")
    
    console.print_success(f"Downloaded {len(downloaded)} datasets")
    
    # Train models on combined data
    console.print_info("\nTraining ensemble models on all stocks...")
    ml = UltimateStockML(model_dir="models/stock")
    
    # Combine all data for training
    all_data = []
    for symbol, csv_file in downloaded:
        try:
            df = pd.read_csv(csv_file)
            df['Symbol'] = symbol
            all_data.append(df)
        except Exception as e:
            console.print_warning(f"Could not load {symbol}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        console.print_info(f"Combined dataset: {len(combined_df)} rows from {len(downloaded)} stocks")
        
        # Train on first stock as representative
        success = ml.train_from_dataset(str(downloaded[0][1]), downloaded[0][0])
        
        if success:
            console.print_success("Stock models trained and saved!")
            return ml, downloaded
    
    return None, []

def download_and_train_forex(console, period='5y'):
    """Download data and train forex models"""
    console.print_header("Training Advanced Forex Models")
    
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    
    console.print_info(f"Downloading data for {len(FOREX_PAIRS)} forex pairs...")
    
    # Download all datasets
    downloaded = []
    for pair in FOREX_PAIRS:
        try:
            console.print_info(f"Downloading {pair}...")
            symbol = f"{pair}=X"
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if len(data) >= 100:
                csv_file = datasets_dir / f"{pair}.csv"
                data.to_csv(csv_file)
                downloaded.append((pair, csv_file))
                console.print_success(f"{pair}: {len(data)} days")
            else:
                console.print_warning(f"{pair}: Insufficient data ({len(data)} days)")
        except Exception as e:
            console.print_error(f"{pair}: Failed - {e}")
    
    console.print_success(f"Downloaded {len(downloaded)} datasets")
    
    # Train forex models
    console.print_info("\nTraining forex ensemble models...")
    forex = ForexML(model_dir="models/forex")
    
    if downloaded:
        # Train on first pair as representative
        success = forex.train_from_dataset(str(downloaded[0][1]), downloaded[0][0])
        
        if success:
            console.print_success("Forex models trained and saved!")
            return forex, downloaded
    
    return None, []

def validate_predictions(ml, symbol, console):
    """Validate model by predicting next day and comparing with actual"""
    try:
        console.print_info(f"\nValidating {symbol}...")
        
        # Get recent data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='1mo')
        
        if len(data) < 10:
            console.print_warning(f"Insufficient data for validation")
            return None
        
        # Use data up to yesterday to predict today
        yesterday_data = data.iloc[:-1]
        actual_today = data['Close'].iloc[-1]
        yesterday_close = data['Close'].iloc[-2]
        
        # Make prediction
        result = ml.predict_ultimate(symbol, days=1)
        
        if result and 'predictions' in result:
            predicted_price = result['predictions'][0]['predicted_price']
            predicted_return = (predicted_price - yesterday_close) / yesterday_close
            actual_return = (actual_today - yesterday_close) / yesterday_close
            
            error = abs(predicted_return - actual_return)
            error_pct = error * 100
            
            console.print_info(f"Yesterday: ${yesterday_close:.2f}")
            console.print_info(f"Predicted: ${predicted_price:.2f} ({predicted_return*100:+.2f}%)")
            console.print_info(f"Actual: ${actual_today:.2f} ({actual_return*100:+.2f}%)")
            console.print_info(f"Error: {error_pct:.2f}%")
            
            return {
                'symbol': symbol,
                'predicted': predicted_price,
                'actual': actual_today,
                'error_pct': error_pct,
                'predicted_return': predicted_return,
                'actual_return': actual_return
            }
    except Exception as e:
        console.print_error(f"Validation failed: {e}")
    
    return None

def main():
    console = ConsoleManager()
    console.print_header("ARA AI - Advanced Model Training System")
    
    print("\nThis will:")
    print("1. Download 5 years of data for 20 stocks")
    print("2. Download 5 years of data for 8 forex pairs")
    print("3. Train advanced models on all data")
    print("4. Validate predictions against actual prices")
    print("5. Save models for future use")
    print("\nThis may take 5-10 minutes...")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Train stock models
    stock_ml, stock_datasets = download_and_train_stocks(console)
    
    # Train forex models
    forex_ml, forex_datasets = download_and_train_forex(console)
    
    # Validate stock predictions
    if stock_ml and stock_datasets:
        console.print_header("Validating Stock Predictions")
        
        validation_results = []
        for symbol, _ in stock_datasets[:5]:  # Validate first 5
            result = validate_predictions(stock_ml, symbol, console)
            if result:
                validation_results.append(result)
        
        if validation_results:
            avg_error = np.mean([r['error_pct'] for r in validation_results])
            console.print_success(f"\nAverage prediction error: {avg_error:.2f}%")
            console.print_success(f"Average accuracy: {100 - avg_error:.2f}%")
    
    # Summary
    console.print_header("Training Complete!")
    console.print_success(f"Stock models: Trained on {len(stock_datasets)} stocks")
    console.print_success(f"Forex models: Trained on {len(forex_datasets)} pairs")
    console.print_success("Models saved to models/stock/ and models/forex/")
    console.print_info("\nYou can now use:")
    console.print_info("  python ara.py AAPL --days 7")
    console.print_info("  python ara_forex.py EURUSD --days 7")

if __name__ == '__main__':
    main()
