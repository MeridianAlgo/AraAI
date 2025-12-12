#!/usr/bin/env python3
"""
Evaluate trained models and generate performance metrics
"""

import argparse
import json
import sqlite3
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

def evaluate_models(model_dir, db_file):
    """Evaluate all trained models"""
    model_dir = Path(model_dir)
    
    # Find all model files
    model_files = list(model_dir.rglob('*.pt'))
    
    if not model_files:
        print("Warning: No model files found")
        return {}
    
    print(f"Found {len(model_files)} model files")
    
    # Load model metadata from database
    conn = sqlite3.connect(db_file)
    metadata_df = pd.read_sql_query('''
        SELECT symbol, model_type, training_date, accuracy, loss, epochs
        FROM model_metadata
        ORDER BY training_date DESC
    ''', conn)
    conn.close()
    
    # Aggregate results
    results = {
        'evaluation_date': datetime.now().isoformat(),
        'total_models': len(model_files),
        'models': [],
        'summary': {
            'avg_accuracy': 0.0,
            'avg_loss': 0.0,
            'total_epochs': 0
        }
    }
    
    total_accuracy = 0
    total_loss = 0
    total_epochs = 0
    
    for _, row in metadata_df.iterrows():
        model_info = {
            'symbol': row['symbol'],
            'model_type': row['model_type'],
            'training_date': row['training_date'],
            'accuracy': float(row['accuracy']) if pd.notna(row['accuracy']) else 0.0,
            'loss': float(row['loss']) if pd.notna(row['loss']) else 0.0,
            'epochs': int(row['epochs']) if pd.notna(row['epochs']) else 0
        }
        
        results['models'].append(model_info)
        total_accuracy += model_info['accuracy']
        total_loss += model_info['loss']
        total_epochs += model_info['epochs']
    
    # Calculate averages
    if len(results['models']) > 0:
        results['summary']['avg_accuracy'] = total_accuracy / len(results['models'])
        results['summary']['avg_loss'] = total_loss / len(results['models'])
        results['summary']['total_epochs'] = total_epochs
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model-dir', required=True, help='Directory containing models')
    parser.add_argument('--db-file', required=True, help='SQLite database file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    print("Evaluating models...")
    results = evaluate_models(args.model_dir, args.db_file)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation complete")
    print(f"  Total models: {results['total_models']}")
    print(f"  Average accuracy: {results['summary']['avg_accuracy']:.2%}")
    print(f"  Average loss: {results['summary']['avg_loss']:.4f}")
    print(f"  Results saved to: {args.output}")

if __name__ == '__main__':
    main()
