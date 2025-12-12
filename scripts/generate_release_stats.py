#!/usr/bin/env python3
"""
Generate comprehensive statistics for weekly release
"""

import argparse
import json
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def generate_stats(db_file):
    """Generate comprehensive statistics"""
    
    if not Path(db_file).exists():
        print("Warning: Database not found, generating empty stats")
        return {
            'generated_at': datetime.now().isoformat(),
            'database_found': False,
            'message': 'No training data available yet'
        }
    
    conn = sqlite3.connect(db_file)
    
    stats = {
        'generated_at': datetime.now().isoformat(),
        'database_found': True,
        'period': 'last_7_days'
    }
    
    # Training runs statistics
    try:
        runs_df = pd.read_sql_query('''
            SELECT 
                mode,
                COUNT(*) as run_count,
                AVG(symbols_count) as avg_symbols,
                SUM(rows_processed) as total_rows,
                MIN(run_date) as first_run,
                MAX(run_date) as last_run
            FROM training_runs
            WHERE run_date >= datetime('now', '-7 days')
            GROUP BY mode
        ''', conn)
        
        stats['training_runs'] = {
            'total_runs': int(runs_df['run_count'].sum()),
            'hourly_runs': int(runs_df[runs_df['mode'] == 'hourly']['run_count'].sum()) if 'hourly' in runs_df['mode'].values else 0,
            'full_runs': int(runs_df[runs_df['mode'] == 'full']['run_count'].sum()) if 'full' in runs_df['mode'].values else 0,
            'total_rows_processed': int(runs_df['total_rows'].sum()),
            'by_mode': runs_df.to_dict('records')
        }
    except Exception as e:
        print(f"Warning: Could not get training runs stats: {e}")
        stats['training_runs'] = {'error': str(e)}
    
    # Model performance statistics
    try:
        perf_df = pd.read_sql_query('''
            SELECT 
                symbol,
                timeframe,
                training_mode,
                COUNT(*) as training_count,
                AVG(accuracy) as avg_accuracy,
                MAX(accuracy) as max_accuracy,
                MIN(accuracy) as min_accuracy,
                AVG(loss) as avg_loss,
                MIN(loss) as min_loss,
                MAX(training_date) as last_trained
            FROM model_metadata
            WHERE training_date >= datetime('now', '-7 days')
            GROUP BY symbol, timeframe, training_mode
        ''', conn)
        
        stats['model_performance'] = {
            'total_models': len(perf_df['symbol'].unique()),
            'total_training_sessions': int(perf_df['training_count'].sum()),
            'overall_avg_accuracy': float(perf_df['avg_accuracy'].mean()),
            'best_accuracy': float(perf_df['max_accuracy'].max()),
            'overall_avg_loss': float(perf_df['avg_loss'].mean()),
            'best_loss': float(perf_df['min_loss'].min()),
            'by_symbol': perf_df.to_dict('records')
        }
    except Exception as e:
        print(f"Warning: Could not get model performance stats: {e}")
        stats['model_performance'] = {'error': str(e)}
    
    # Data collection statistics
    try:
        data_df = pd.read_sql_query('''
            SELECT 
                asset_type,
                interval,
                COUNT(*) as row_count,
                COUNT(DISTINCT symbol) as symbol_count,
                MIN(date) as earliest_date,
                MAX(date) as latest_date
            FROM market_data
            WHERE fetch_date >= datetime('now', '-7 days')
            GROUP BY asset_type, interval
        ''', conn)
        
        stats['data_collection'] = {
            'total_rows': int(data_df['row_count'].sum()),
            'total_symbols': int(data_df['symbol_count'].sum()),
            'by_type': data_df.to_dict('records')
        }
    except Exception as e:
        print(f"Warning: Could not get data collection stats: {e}")
        stats['data_collection'] = {'error': str(e)}
    
    # Top performing models
    try:
        top_models = pd.read_sql_query('''
            SELECT 
                symbol,
                timeframe,
                MAX(accuracy) as best_accuracy,
                MIN(loss) as best_loss,
                MAX(training_date) as last_trained
            FROM model_metadata
            WHERE training_date >= datetime('now', '-7 days')
            GROUP BY symbol, timeframe
            ORDER BY best_accuracy DESC
            LIMIT 10
        ''', conn)
        
        stats['top_performers'] = top_models.to_dict('records')
    except Exception as e:
        print(f"Warning: Could not get top performers: {e}")
        stats['top_performers'] = []
    
    # Training by hour statistics
    try:
        hourly_stats = pd.read_sql_query('''
            SELECT 
                hour,
                COUNT(*) as run_count,
                AVG(rows_processed) as avg_rows
            FROM training_runs
            WHERE mode = 'hourly'
            AND run_date >= datetime('now', '-7 days')
            GROUP BY hour
            ORDER BY hour
        ''', conn)
        
        stats['hourly_distribution'] = hourly_stats.to_dict('records')
    except Exception as e:
        print(f"Warning: Could not get hourly distribution: {e}")
        stats['hourly_distribution'] = []
    
    conn.close()
    return stats

def format_markdown(stats):
    """Format statistics as markdown"""
    
    md = f"""# 📊 ARA AI Weekly Statistics

**Generated**: {stats['generated_at']}  
**Period**: Last 7 days

---

"""
    
    if not stats.get('database_found'):
        md += "⚠️ No training data available yet. Start training to see statistics!\n"
        return md
    
    # Training Runs
    if 'training_runs' in stats and 'error' not in stats['training_runs']:
        tr = stats['training_runs']
        md += f"""## 🔄 Training Runs

- **Total Runs**: {tr['total_runs']}
- **Hourly Runs**: {tr['hourly_runs']}
- **Full Runs**: {tr['full_runs']}
- **Total Rows Processed**: {tr['total_rows_processed']:,}

"""
    
    # Model Performance
    if 'model_performance' in stats and 'error' not in stats['model_performance']:
        mp = stats['model_performance']
        md += f"""## 🎯 Model Performance

- **Total Models**: {mp['total_models']}
- **Training Sessions**: {mp['total_training_sessions']}
- **Average Accuracy**: {mp['overall_avg_accuracy']:.2%}
- **Best Accuracy**: {mp['best_accuracy']:.2%}
- **Average Loss**: {mp['overall_avg_loss']:.4f}
- **Best Loss**: {mp['best_loss']:.4f}

"""
    
    # Top Performers
    if stats.get('top_performers'):
        md += "## 🏆 Top Performing Models\n\n"
        md += "| Symbol | Timeframe | Accuracy | Loss | Last Trained |\n"
        md += "|--------|-----------|----------|------|-------------|\n"
        for model in stats['top_performers'][:5]:
            md += f"| {model['symbol']} | {model['timeframe']} | {model['best_accuracy']:.2%} | {model['best_loss']:.4f} | {model['last_trained']} |\n"
        md += "\n"
    
    # Data Collection
    if 'data_collection' in stats and 'error' not in stats['data_collection']:
        dc = stats['data_collection']
        md += f"""## 📈 Data Collection

- **Total Data Points**: {dc['total_rows']:,}
- **Unique Symbols**: {dc['total_symbols']}

"""
    
    # Hourly Distribution
    if stats.get('hourly_distribution'):
        md += "## ⏰ Training Distribution by Hour\n\n"
        md += "| Hour (UTC) | Runs | Avg Rows |\n"
        md += "|------------|------|----------|\n"
        for hour_stat in stats['hourly_distribution']:
            md += f"| {hour_stat['hour']:02d}:00 | {int(hour_stat['run_count'])} | {int(hour_stat['avg_rows'])} |\n"
        md += "\n"
    
    md += """---

*Generated automatically by ARA AI Weekly Release*
"""
    
    return md

def main():
    parser = argparse.ArgumentParser(description='Generate release statistics')
    parser.add_argument('--db-file', default='training_data.db', help='Database file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--output-md', help='Output Markdown file')
    
    args = parser.parse_args()
    
    print("Generating release statistics...")
    stats = generate_stats(args.db_file)
    
    # Save JSON
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"✓ Saved JSON to {args.output}")
    
    # Save Markdown
    if args.output_md:
        md = format_markdown(stats)
        with open(args.output_md, 'w') as f:
            f.write(md)
        print(f"✓ Saved Markdown to {args.output_md}")
    
    # Print summary
    print("\n" + "="*60)
    print("STATISTICS SUMMARY")
    print("="*60)
    if stats.get('database_found'):
        if 'training_runs' in stats and 'error' not in stats['training_runs']:
            print(f"Training Runs: {stats['training_runs']['total_runs']}")
        if 'model_performance' in stats and 'error' not in stats['model_performance']:
            print(f"Models Trained: {stats['model_performance']['total_models']}")
            print(f"Avg Accuracy: {stats['model_performance']['overall_avg_accuracy']:.2%}")
    else:
        print("No training data available yet")
    print("="*60)

if __name__ == '__main__':
    main()
