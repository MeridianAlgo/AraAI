"""
Train on All Major Stocks
Dynamically fetches stock tickers from major indices
No need to hardcode - trains on hundreds of stocks automatically!
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.unified_ml import UnifiedStockML
from meridianalgo.forex_ml import ForexML
from meridianalgo.console import ConsoleManager
import time

# Predefined lists of major indices (top 100 from each)
SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'XOM', 'WMT', 'LLY', 'MA', 'PG', 'AVGO', 'HD', 'CVX',
    'MRK', 'ABBV', 'COST', 'PEP', 'KO', 'ADBE', 'CRM', 'MCD', 'CSCO', 'ACN',
    'TMO', 'ABT', 'LIN', 'NFLX', 'AMD', 'NKE', 'DHR', 'DIS', 'VZ', 'WFC',
    'TXN', 'PM', 'CMCSA', 'NEE', 'ORCL', 'COP', 'INTC', 'UNP', 'RTX', 'QCOM',
    'IBM', 'HON', 'INTU', 'LOW', 'UPS', 'CAT', 'BA', 'GS', 'SPGI', 'AXP',
    'DE', 'SBUX', 'BLK', 'ELV', 'GILD', 'MDT', 'SYK', 'BKNG', 'ADI', 'PLD',
    'TJX', 'MDLZ', 'VRTX', 'ADP', 'REGN', 'MMC', 'CVS', 'CI', 'ISRG', 'ZTS',
    'C', 'SO', 'CB', 'DUK', 'PGR', 'SLB', 'MO', 'BDX', 'NOC', 'SCHW',
    'BSX', 'ETN', 'ITW', 'USB', 'EOG', 'PNC', 'MMM', 'GE', 'FI', 'CL'
]

NASDAQ100_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST',
    'NFLX', 'AMD', 'PEP', 'ADBE', 'CSCO', 'TMUS', 'CMCSA', 'INTC', 'TXN', 'QCOM',
    'INTU', 'AMGN', 'HON', 'SBUX', 'AMAT', 'BKNG', 'ADI', 'GILD', 'VRTX', 'REGN',
    'MDLZ', 'ADP', 'ISRG', 'LRCX', 'PANW', 'MU', 'PYPL', 'SNPS', 'KLAC', 'CDNS',
    'ASML', 'MELI', 'ABNB', 'MAR', 'ORLY', 'CTAS', 'NXPI', 'MNST', 'CSX', 'FTNT'
]

DOW_TICKERS = [
    'AAPL', 'MSFT', 'UNH', 'GS', 'HD', 'MCD', 'CAT', 'V', 'AMGN', 'BA',
    'TRV', 'AXP', 'JPM', 'HON', 'IBM', 'CRM', 'JNJ', 'PG', 'CVX', 'MRK',
    'WMT', 'DIS', 'NKE', 'MMM', 'KO', 'DOW', 'CSCO', 'VZ', 'INTC', 'WBA'
]

def get_all_major_tickers(index='sp500'):
    """Get all major stock tickers from specified index"""
    tickers = []
    
    if index == 'sp500' or index == 'all':
        tickers.extend(SP500_TICKERS)
        print(f"✓ Loaded {len(SP500_TICKERS)} S&P 500 tickers")
    
    if index == 'nasdaq100' or index == 'all':
        tickers.extend(NASDAQ100_TICKERS)
        print(f"✓ Loaded {len(NASDAQ100_TICKERS)} NASDAQ 100 tickers")
    
    if index == 'dow' or index == 'all':
        tickers.extend(DOW_TICKERS)
        print(f"✓ Loaded {len(DOW_TICKERS)} Dow Jones tickers")
    
    # Remove duplicates and sort
    tickers = sorted(list(set(tickers)))
    return tickers

def get_custom_tickers_from_file(filepath):
    """Load custom ticker list from a text file (one ticker per line)"""
    try:
        with open(filepath, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
        return tickers
    except Exception as e:
        print(f"Error loading tickers from file: {e}")
        return []

# Major forex pairs
FOREX_PAIRS = {
    'Major': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'],
    'Minor': ['AUDUSD', 'USDCAD', 'NZDUSD'],
    'Cross': ['EURGBP', 'EURJPY', 'GBPJPY']
}

FOREX_TO_TRAIN = [pair for pairs in FOREX_PAIRS.values() for pair in pairs]

def print_stocks_info(tickers):
    """Print stock ticker information"""
    console = ConsoleManager()
    console.print_info(f"\n📊 Total Stocks: {len(tickers)}")
    if len(tickers) <= 50:
        print(f"  Tickers: {', '.join(tickers[:50])}")
        if len(tickers) > 50:
            print(f"  ... and {len(tickers) - 50} more")
    else:
        print(f"  Sample: {', '.join(tickers[:20])} ...")
        print(f"  (showing first 20 of {len(tickers)})")

def print_forex_info():
    """Print forex pairs organized by type"""
    console = ConsoleManager()
    console.print_info("\n💱 Forex Pairs by Type:")
    for pair_type, pairs in FOREX_PAIRS.items():
        print(f"  {pair_type}: {', '.join(pairs)}")
    print(f"  Total: {len(FOREX_TO_TRAIN)} pairs")

def train_all_stocks(stocks_list, epochs=1000, strict_mode=False, cpu_limit=80):
    """Train on all stocks in the list"""
    console = ConsoleManager()
    console.print_header("Training Stock Models")
    
    total_stocks = len(stocks_list)
    successful = 0
    failed = []
    
    console.print_info(f"Training on {total_stocks} stocks with {epochs} epochs each")
    console.print_info(f"Using Intelligent Model (1.6M parameters)")
    if strict_mode:
        console.print_warning("⚠️  STRICT MODE: Training will stop on first error!")
    
    start_time = time.time()
    
    for idx, symbol in enumerate(stocks_list, 1):
        console.print_info(f"\n[{idx}/{total_stocks}] Training {symbol}...")
        
        try:
            ml = UnifiedStockML()
            success = ml.train_ultimate_models(
                period='2y',
                use_parallel=False,
                target_symbol=symbol,
                epochs=epochs,
                cpu_limit=cpu_limit
            )
            
            if success:
                console.print_success(f"{symbol} training completed!")
                successful += 1
                
                # Quick test prediction
                result = ml.predict_ultimate(symbol, days=5)
                if result and 'predictions' in result:
                    current = result.get('current_price', 0)
                    predicted = result['predictions'][-1]['predicted_price']
                    change = ((predicted - current) / current) * 100
                    console.print_info(f"  Test: ${current:.2f} → ${predicted:.2f} ({change:+.2f}%)")
            else:
                error_msg = f"{symbol} training completed with warnings"
                console.print_warning(error_msg)
                failed.append(symbol)
                
                if strict_mode:
                    console.print_error("❌ STRICT MODE: Stopping due to training warning!")
                    raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"{symbol} training failed: {e}"
            console.print_error(error_msg)
            failed.append(symbol)
            
            if strict_mode:
                console.print_error("❌ STRICT MODE: Stopping all training due to error!")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Training stopped in strict mode due to error in {symbol}: {e}")
    
    elapsed = time.time() - start_time
    
    # Summary
    console.print_header("Stock Training Summary")
    console.print_success(f"Successfully trained: {successful}/{total_stocks}")
    if failed:
        console.print_warning(f"Failed: {len(failed)} - {', '.join(failed)}")
    console.print_info(f"Total time: {elapsed/60:.1f} minutes")
    if total_stocks > 0:
        console.print_info(f"Average time per stock: {elapsed/total_stocks:.1f} seconds")

def train_all_forex(forex_list, epochs=1000, strict_mode=False, cpu_limit=80):
    """Train on all forex pairs in the list"""
    console = ConsoleManager()
    console.print_header("Training Forex Models")
    
    total_pairs = len(forex_list)
    successful = 0
    failed = []
    
    console.print_info(f"Training on {total_pairs} forex pairs with {epochs} epochs each")
    if strict_mode:
        console.print_warning("⚠️  STRICT MODE: Training will stop on first error!")
    
    start_time = time.time()
    
    for idx, pair in enumerate(forex_list, 1):
        console.print_info(f"\n[{idx}/{total_pairs}] Training {pair}...")
        
        try:
            forex = ForexML()
            target_symbol = forex.get_forex_symbol(pair)
            success = forex.train_ultimate_models(
                target_symbol=target_symbol,
                period='2y',
                use_parallel=False,
                epochs=epochs,
                cpu_limit=cpu_limit
            )
            
            if success:
                console.print_success(f"{pair} training completed!")
                successful += 1
                
                # Quick test prediction
                result = forex.predict_forex(pair, days=5)
                if result and 'predictions' in result:
                    current = result['current_price']
                    predicted = result['predictions'][-1]['predicted_price']
                    change = ((predicted - current) / current) * 100
                    console.print_info(f"  Test: {current:.5f} → {predicted:.5f} ({change:+.2f}%)")
            else:
                error_msg = f"{pair} training completed with warnings"
                console.print_warning(error_msg)
                failed.append(pair)
                
                if strict_mode:
                    console.print_error("❌ STRICT MODE: Stopping due to training warning!")
                    raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"{pair} training failed: {e}"
            console.print_error(error_msg)
            failed.append(pair)
            
            if strict_mode:
                console.print_error("❌ STRICT MODE: Stopping all training due to error!")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Training stopped in strict mode due to error in {pair}: {e}")
    
    elapsed = time.time() - start_time
    
    # Summary
    console.print_header("Forex Training Summary")
    console.print_success(f"Successfully trained: {successful}/{total_pairs}")
    if failed:
        console.print_warning(f"Failed: {len(failed)} - {', '.join(failed)}")
    console.print_info(f"Total time: {elapsed/60:.1f} minutes")
    if total_pairs > 0:
        console.print_info(f"Average time per pair: {elapsed/total_pairs:.1f} seconds")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train on all major stocks and forex pairs')
    parser.add_argument('--stocks-only', action='store_true', help='Train only stocks')
    parser.add_argument('--forex-only', action='store_true', help='Train only forex')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs (default: 1000)')
    parser.add_argument('--quick', action='store_true', help='Quick training on subset (5 stocks, 2 forex)')
    parser.add_argument('--index', type=str, default='sp500', 
                        choices=['sp500', 'nasdaq100', 'dow', 'all'],
                        help='Stock index to train on (default: sp500)')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit number of stocks to train (for testing)')
    parser.add_argument('--file', type=str, default=None,
                        help='Load tickers from custom file (one per line)')
    parser.add_argument('--strict', action='store_true',
                        help='Stop all training immediately if any error occurs')
    parser.add_argument('--cpu-limit', type=int, default=80,
                        help='Limit CPU usage percentage (default: 80)')
    
    args = parser.parse_args()
    
    console = ConsoleManager()
    console.print_header("ARA AI - Train All Models")
    console.print_info(f"🚀 Optimized with Hugging Face Accelerate")
    console.print_info(f"⚡ CPU Limit set to {args.cpu_limit}%")
    
    # Get stock tickers dynamically
    stocks_to_train = None
    if not args.forex_only:
        if args.quick:
            stocks_to_train = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
            console.print_info(f"\n⚡ Quick mode: Training on {len(stocks_to_train)} stocks")
        elif args.file:
            console.print_info(f"\n📥 Loading tickers from {args.file}...")
            stocks_to_train = get_custom_tickers_from_file(args.file)
        else:
            console.print_info(f"\n📥 Loading {args.index.upper()} tickers...")
            stocks_to_train = get_all_major_tickers(args.index)
            
            if args.limit:
                stocks_to_train = stocks_to_train[:args.limit]
                console.print_info(f"   Limited to first {args.limit} stocks")
        
        print_stocks_info(stocks_to_train)
    
    # Forex pairs
    forex_to_train = None
    if not args.stocks_only:
        if args.quick:
            forex_to_train = ['EURUSD', 'GBPUSD']
            console.print_info(f"\n⚡ Quick mode: Training on {len(forex_to_train)} forex pairs")
        else:
            forex_to_train = FOREX_TO_TRAIN
        
        print_forex_info()
    
    total_start = time.time()
    
    try:
        # Train stocks
        if not args.forex_only and stocks_to_train:
            train_all_stocks(stocks_to_train, epochs=args.epochs, strict_mode=args.strict, cpu_limit=args.cpu_limit)
        
        # Train forex
        if not args.stocks_only and forex_to_train:
            train_all_forex(forex_to_train, epochs=args.epochs, strict_mode=args.strict, cpu_limit=args.cpu_limit)
        
        total_elapsed = time.time() - total_start
        
        # Final summary
        console.print_header("Complete Training Summary")
        console.print_success(f"Total training time: {total_elapsed/60:.1f} minutes")
        console.print_info(f"Models saved to: models/")
        console.print_info("Ready for predictions!")
        
    except RuntimeError as e:
        console.print_error(f"\n❌ Training stopped: {e}")
        console.print_info("Fix the error and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
