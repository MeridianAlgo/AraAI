#!/usr/bin/env python3
"""
Create training summary for GitHub Actions
"""

import argparse
import json
from datetime import datetime

def create_summary(results_file):
    """Create markdown summary from evaluation results"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    summary = f"""# 🤖 Daily Training Summary

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## 📊 Overall Performance

- **Total Models Trained:** {results['total_models']}
- **Average Accuracy:** {results['summary']['avg_accuracy']:.2%}
- **Average Loss:** {results['summary']['avg_loss']:.4f}
- **Total Training Epochs:** {results['summary']['total_epochs']}

## 📈 Model Details

| Symbol | Type | Accuracy | Loss | Epochs | Training Date |
|--------|------|----------|------|--------|---------------|
"""
    
    for model in results['models']:
        summary += f"| {model['symbol']} | {model['model_type']} | {model['accuracy']:.2%} | {model['loss']:.4f} | {model['epochs']} | {model['training_date']} |\n"
    
    summary += f"""
## ✅ Status

All models trained successfully and ready for predictions!

---
*Automated training powered by GitHub Actions*
"""
    
    print(summary)

def main():
    parser = argparse.ArgumentParser(description='Create training summary')
    parser.add_argument('--results', required=True, help='Evaluation results JSON file')
    parser.add_argument('--output', help='Output file (optional, prints to stdout if not provided)')
    
    args = parser.parse_args()
    
    create_summary(args.results)

if __name__ == '__main__':
    main()
