# Training Datasets

This folder contains CSV datasets for training the ML models.

## Dataset Format

All datasets should be in CSV format with the following columns:

```csv
Date,Open,High,Low,Close,Volume
2023-01-01,100.0,105.0,99.0,104.0,1000000
2023-01-02,104.0,106.0,103.0,105.0,1200000
2023-01-03,106.0,108.0,105.5,107.5,1300000
```

### Required Columns
- **Date**: Date in YYYY-MM-DD format
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price

### Optional Columns
- **Volume**: Trading volume (will use default if missing)

## Training from Datasets

### Stock Data
```bash
python train_from_dataset.py datasets/AAPL_historical.csv --type stock --name AAPL
```

### Forex Data
```bash
python train_from_dataset.py datasets/EURUSD_historical.csv --type forex --name EURUSD
```

## Dataset Sources

You can obtain historical data from:
- Yahoo Finance (yfinance library)
- Alpha Vantage
- Quandl
- Your own trading data

## Example: Download Dataset

```python
import yfinance as yf
import pandas as pd

# Download stock data
ticker = yf.Ticker("AAPL")
data = ticker.history(period="5y")
data.to_csv("datasets/AAPL_historical.csv")

# Download forex data
ticker = yf.Ticker("EURUSD=X")
data = ticker.history(period="5y")
data.to_csv("datasets/EURUSD_historical.csv")
```

## Notes

- Minimum 100 days of data required for training
- More data generally leads to better predictions
- Data should be sorted by date (oldest first)
- Missing values will be forward-filled
