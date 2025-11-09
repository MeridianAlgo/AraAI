"""
Train ML models from dataset CSV files
Supports both stock and forex datasets
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from meridianalgo.ultimate_ml import UltimateStockML
from meridianalgo.forex_ml import ForexML
from meridianalgo.console import ConsoleManager

def main():
    parser = argparse.ArgumentParser(description='Train ML models from dataset CSV files')
    parser.add_argument('dataset', help='Path to dataset CSV file')
    parser.add_argument('--type', choices=['stock', 'forex'], default='stock', help='Dataset type (default: stock)')
    parser.add_argument('--name', help='Symbol name (e.g., AAPL, EURUSD)')
    
    args = parser.parse_args()
    
    console = ConsoleManager()
    
    # Print header
    console.print_header(f"ARA AI - Training from Dataset")
    
    # Validate dataset file
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        console.print_error(f"Dataset file not found: {args.dataset}")
        return
    
    # Extract symbol name from filename if not provided
    symbol_name = args.name
    if not symbol_name:
        symbol_name = dataset_path.stem.upper()
    
    console.print_info(f"Dataset: {args.dataset}")
    console.print_info(f"Type: {args.type}")
    console.print_info(f"Symbol: {symbol_name}")
    
    # Initialize appropriate ML system
    if args.type == 'forex':
        console.print_info("Initializing Forex ML System...")
        ml = ForexML()
    else:
        console.print_info("Initializing Stock ML System...")
        ml = UltimateStockML()
    
    # Train from dataset
    console.print_info("Starting training...")
    console.print_info("This may take 1-2 minutes...")
    
    try:
        success = ml.train_from_dataset(args.dataset, symbol_name)
        
        if success:
            console.print_success("Training completed successfully!")
            console.print_success(f"Models saved to: {ml.model_dir}")
            
            # Show model status
            status = ml.get_model_status()
            console.print_info(f"Models trained: {status['model_count']}")
            console.print_info(f"Features: {status['feature_count']}")
            console.print_info(f"Trained on: {status['training_metadata'].get('symbol', 'Unknown')}")
            console.print_info(f"Data points: {status['training_metadata'].get('data_points', 'Unknown')}")
            
            # Test prediction
            console.print_info("\nTesting prediction...")
            if args.type == 'forex':
                result = ml.predict_forex(symbol_name, days=3)
            else:
                result = ml.predict_ultimate(symbol_name, days=3)
            
            if result and 'predictions' in result:
                console.print_success("Prediction test successful!")
                print(f"\nCurrent Price: {result['current_price']:.4f}")
                print("\n3-Day Forecast:")
                for pred in result['predictions']:
                    print(f"  Day {pred['day']}: {pred['predicted_price']:.4f} (Confidence: {pred['confidence']:.1%})")
            else:
                console.print_warning("Prediction test failed - but models are trained")
        else:
            console.print_error("Training failed!")
            
    except Exception as e:
        console.print_error(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
