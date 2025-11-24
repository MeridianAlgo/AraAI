"""
Prepare training data - Download AAPL 60 days 1-day bars
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime

def download_aapl_data():
    """Download AAPL 60 days of 1-day bars"""
    print("Downloading AAPL 60 days of 1-day bars...")
    
    # Download data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="60d", interval="1d")
    
    print(f"Downloaded {len(data)} days of data")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    # Save to datasets folder
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    output_file = datasets_dir / "AAPL_60d_1d.csv"
    data.to_csv(output_file)
    
    print(f"Saved to: {output_file}")
    print(f"\nData preview:")
    print(data.head())
    print(f"\nData shape: {data.shape}")
    
    return output_file

if __name__ == "__main__":
    download_aapl_data()
