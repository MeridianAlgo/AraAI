#!/usr/bin/env python3
"""
Query and visualize training data from the database
Useful for monitoring hourly training progress
"""

import argparse
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def show_recent_runs(db_file, limit=10):
    """Show recent training runs"""
    conn = sqlite3.connect(db_file)
    
    df = pd.read_sql_query(f'''
        SELECT 
            run_date,
            mode,
            timeframe,
            hour,
            symbols_count,
            rows_processed,
            status
        FROM training_runs
        ORDER BY run_date DESC
        LIMIT {limit}
    ''', conn)
    
    conn.close()
    
    print("\n" + "="*80)
    print("RECENT TRAINING RUNS")
    print("="*80)
    print(df.to_string(index=False))
    print()

def show_model_performance(db_file, symbol=None):
    """Show model performance by timeframe"""
    conn = sqlite3.connect(db_file)
    
    if symbol:
        query = f'''
            SELECT 
                symbol,
                timeframe,
                training_mode,
                AVG(accuracy) as avg_accuracy,
                AVG(loss) as avg_loss,
                COUNT(*) as training_count,
                MAX(training_date) as last_trained
            FROM model_metadata
            WHERE symbol = '{symbol}'
            GROUP BY symbol, timeframe, training_mode
            ORDER BY last_trained DESC
        '''
    else:
        query = '''
            SELECT 
                symbol,
                timeframe,
                training_mode,
                AVG(accuracy) as avg_accuracy,
                AVG(loss) as avg_loss,
                COUNT(*) as training_count,
                MAX(training_date) as last_trained
            FROM model_metadata
            GROUP BY symbol, timeframe, training_mode
            ORDER BY symbol, timeframe
        '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("\n" + "="*80)
    print(f"MODEL PERFORMANCE{' - ' + symbol if symbol else ''}")
    print("="*80)
    print(df.to_string(index=False))
    print()

def show_hourly_data(db_file, symbol, hours=24):
    """Show recent hourly data for a symbol"""
    conn = sqlite3.connect(db_file)
    
    df = pd.read_sql_query(f'''
        SELECT 
            date,
            hour,
            open,
            high,
            low,
            close,
            volume,
            timeframe,
            interval
        FROM market_data
        WHERE symbol = '{symbol}'
        AND interval = '1h'
        AND date >= datetime('now', '-{hours} hours')
        ORDER BY date DESC
    ''', conn)
    
    conn.close()
    
    print("\n" + "="*80)
    print(f"HOURLY DATA - {symbol} (Last {hours} hours)")
    print("="*80)
    print(df.to_string(index=False))
    print()

def show_training_by_hour(db_file):
    """Show training statistics by hour of day"""
    conn = sqlite3.connect(db_file)
    
    df = pd.read_sql_query('''
        SELECT 
            hour,
            COUNT(*) as run_count,
            AVG(symbols_count) as avg_symbols,
            AVG(rows_processed) as avg_rows,
            MAX(run_date) as last_run
        FROM training_runs
        WHERE hour IS NOT NULL
        GROUP BY hour
        ORDER BY hour
    ''', conn)
    
    conn.close()
    
    print("\n" + "="*80)
    print("TRAINING STATISTICS BY HOUR")
    print("="*80)
    print(df.to_string(index=False))
    print()

def show_database_stats(db_file):
    """Show overall database statistics"""
    conn = sqlite3.connect(db_file)
    
    # Market data stats
    market_stats = pd.read_sql_query('''
        SELECT 
            asset_type,
            interval,
            COUNT(*) as row_count,
            COUNT(DISTINCT symbol) as symbol_count,
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM market_data
        GROUP BY asset_type, interval
    ''', conn)
    
    # Training runs stats
    run_stats = pd.read_sql_query('''
        SELECT 
            mode,
            COUNT(*) as run_count,
            SUM(rows_processed) as total_rows,
            MIN(run_date) as first_run,
            MAX(run_date) as last_run
        FROM training_runs
        GROUP BY mode
    ''', conn)
    
    # Model stats
    model_stats = pd.read_sql_query('''
        SELECT 
            model_type,
            timeframe,
            COUNT(*) as model_count,
            AVG(accuracy) as avg_accuracy,
            AVG(loss) as avg_loss
        FROM model_metadata
        GROUP BY model_type, timeframe
    ''', conn)
    
    conn.close()
    
    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80)
    
    print("\nMarket Data:")
    print(market_stats.to_string(index=False))
    
    print("\nTraining Runs:")
    print(run_stats.to_string(index=False))
    
    print("\nModel Performance:")
    print(model_stats.to_string(index=False))
    print()

def export_to_csv(db_file, table, output_file):
    """Export table to CSV"""
    conn = sqlite3.connect(db_file)
    
    df = pd.read_sql_query(f'SELECT * FROM {table}', conn)
    conn.close()
    
    df.to_csv(output_file, index=False)
    print(f"\n✓ Exported {len(df)} rows from {table} to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Query training database')
    parser.add_argument('--db-file', default='training_data.db', help='Database file')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Recent runs
    runs_parser = subparsers.add_parser('runs', help='Show recent training runs')
    runs_parser.add_argument('--limit', type=int, default=10, help='Number of runs to show')
    
    # Model performance
    perf_parser = subparsers.add_parser('performance', help='Show model performance')
    perf_parser.add_argument('--symbol', help='Filter by symbol')
    
    # Hourly data
    hourly_parser = subparsers.add_parser('hourly', help='Show hourly data')
    hourly_parser.add_argument('--symbol', required=True, help='Symbol to query')
    hourly_parser.add_argument('--hours', type=int, default=24, help='Hours to show')
    
    # Training by hour
    subparsers.add_parser('by-hour', help='Show training stats by hour')
    
    # Database stats
    subparsers.add_parser('stats', help='Show database statistics')
    
    # Export
    export_parser = subparsers.add_parser('export', help='Export table to CSV')
    export_parser.add_argument('--table', required=True, 
                               choices=['market_data', 'training_runs', 'model_metadata'])
    export_parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    if not Path(args.db_file).exists():
        print(f"Error: Database file '{args.db_file}' not found")
        return
    
    if args.command == 'runs':
        show_recent_runs(args.db_file, args.limit)
    elif args.command == 'performance':
        show_model_performance(args.db_file, args.symbol)
    elif args.command == 'hourly':
        show_hourly_data(args.db_file, args.symbol, args.hours)
    elif args.command == 'by-hour':
        show_training_by_hour(args.db_file)
    elif args.command == 'stats':
        show_database_stats(args.db_file)
    elif args.command == 'export':
        export_to_csv(args.db_file, args.table, args.output)
    else:
        # Default: show stats
        show_database_stats(args.db_file)

if __name__ == '__main__':
    main()
