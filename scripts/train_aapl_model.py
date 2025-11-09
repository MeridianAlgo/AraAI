"""
Train existing stock_model.pt with AAPL 60d data - 2500 steps
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.unified_ml import UnifiedStockML

def train_aapl():
    # Use existing stock_model.pt
    ml = UnifiedStockML(model_path="models/stock_model.pt")
    
    # Train with AAPL dataset - 2500 steps
    ml.train_from_dataset(
        "datasets/AAPL_60d_1d.csv", 
        "AAPL",
        epochs=2500,
        batch_size=16,
        lr=0.0001,
        validation_split=0.2,
        metadata={
            'symbol': 'AAPL',
            'timeframe': '1d',
            'asset_type': 'stock',
            'bars': 60
        }
    )
    
    # Test prediction
    result = ml.predict_ultimate('AAPL', days=5)
    
    if result and 'predictions' in result:
        print(f"\nCurrent: ${result['current_price']:.2f}")
        for pred in result['predictions']:
            print(f"Day {pred['day']}: ${pred['predicted_price']:.2f}")

if __name__ == "__main__":
    train_aapl()
