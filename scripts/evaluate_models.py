
import sqlite3
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.unified_ml import UnifiedStockML
from meridianalgo.forex_ml import ForexML
from meridianalgo.direction_loss import calculate_direction_metrics

def load_data_from_db(db_path, asset_type, symbol, limit=300):
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT date, open, high, low, close, volume
        FROM market_data
        WHERE asset_type = ? AND symbol = ?
        ORDER BY date ASC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=[asset_type, symbol, limit], parse_dates=['date'])
    conn.close()
    
    if df.empty:
        return None
        
    df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df.set_index("Date", inplace=True)
    return df

def evaluate_model(model, asset_type, db_path, symbols_to_test=5):
    print(f"\nEvaluating {asset_type.upper()} Model...")
    
    # Get symbols
    conn = sqlite3.connect(db_path)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data WHERE asset_type = ?", 
        conn, params=[asset_type]
    )["symbol"].tolist()
    conn.close()
    
    if not symbols:
        print(f"No symbols found for {asset_type}")
        return
        
    import random
    selected_symbols = random.sample(symbols, min(symbols_to_test, len(symbols)))
    print(f"Testing on: {', '.join(selected_symbols)}")
    
    all_preds_list = []
    all_trues_list = []
    
    lookback = 30
    
    extractor = model if asset_type == "stock" else model._unified_ml

    for symbol in selected_symbols:
        df = load_data_from_db(db_path, asset_type, symbol)
        if df is None or len(df) < lookback + 10:
            continue
            
        # Add indicators once
        df = extractor._add_indicators(df)
        
        # Pre-calculate all features for each point in time
        features_precalc = []
        for i in range(len(df)):
            features_precalc.append(extractor._extract_features(df.iloc[:i+1]))
        features_precalc = np.array(features_precalc)
        
        # Prepare windows
        X_batches = []
        trues = []
        for i in range(lookback, len(df) - 1):
            window = features_precalc[i - lookback + 1 : i + 1]
            X_batches.append(window)
            
            actual_return = (df['Close'].iloc[i+1] - df['Close'].iloc[i]) / df['Close'].iloc[i]
            trues.append(actual_return)
            
        if not X_batches:
            continue
            
        # Predict in large batch
        X_tensor = np.array(X_batches)
        pred_returns, _ = model.ml_system.predict(X_tensor)
        
        all_preds_list.extend(pred_returns.flatten().tolist())
        all_trues_list.extend(trues)
        print(f"  Processed {symbol}: {len(trues)} samples")
        
    if not all_preds_list:
        print("No predictions made.")
        return
        
    preds_tensor = torch.FloatTensor(all_preds_list)
    trues_tensor = torch.FloatTensor(all_trues_list)
    
    metrics = calculate_direction_metrics(preds_tensor, trues_tensor)
    
    print(f"\nFINISH: Results for {asset_type}:")
    print(f"  Directional Accuracy: {metrics['direction_accuracy']:.2f}%")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall: {metrics['recall']:.2f}%")
    print(f"  F1 Score: {metrics['f1_score']:.2f}%")
    print(f"  Total Samples: {len(all_preds_list)}")
    
    return metrics

if __name__ == "__main__":
    db_path = "data/training.db"
    
    # Stock Evaluation
    if Path("models/MeridianAlgo_Stocks.pt").exists():
        stock_model = UnifiedStockML(model_path="models/MeridianAlgo_Stocks.pt", model_type="stock")
        if stock_model.ml_system.is_trained():
            evaluate_model(stock_model, "stock", db_path)
    
    # Forex Evaluation
    if Path("models/MeridianAlgo_Forex.pt").exists():
        forex_model = ForexML(model_path="models/MeridianAlgo_Forex.pt")
        if forex_model.ml_system.is_trained():
            evaluate_model(forex_model, "forex", db_path)
