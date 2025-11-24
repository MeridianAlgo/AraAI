"""
Comprehensive demonstration of ARA AI visualization capabilities.

This script demonstrates all visualization features including:
- Candlestick charts with indicators
- Prediction visualizations
- Portfolio analysis
- Correlation analysis
- Chart export
- Report generation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Import visualization modules
from ara.visualization import (
    ChartFactory,
    CandlestickChart,
    PredictionChart,
    PortfolioChart,
    CorrelationChart,
    ChartExporter,
    ReportGenerator
)


def generate_sample_ohlcv_data(days=100, start_price=100.0):
    """Generate sample OHLCV data for demonstration."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate random walk for close prices
    returns = np.random.normal(0.001, 0.02, days)
    close = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, days)))
    open_price = np.roll(close, 1)
    open_price[0] = start_price
    
    # Generate volume
    volume = np.random.randint(1000000, 10000000, days)
    
    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


def calculate_sample_indicators(data):
    """Calculate sample technical indicators."""
    close = data['close']
    
    # Simple Moving Averages
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    
    # Bollinger Bands
    bb_middle = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    
    return {
        'SMA_20': sma_20,
        'SMA_50': sma_50,
        'BB_Upper': bb_upper,
        'BB_Lower': bb_lower
    }


def demo_candlestick_chart():
    """Demonstrate candlestick chart creation."""
    print("\n=== Candlestick Chart Demo ===")
    
    # Generate sample data
    data = generate_sample_ohlcv_data(days=100, start_price=150.0)
    indicators = calculate_sample_indicators(data)
    
    # Create chart
    chart = CandlestickChart()
    fig = chart.create_chart(
        data=data,
        symbol='AAPL',
        indicators=indicators,
        volume=True,
        height=800
    )
    
    # Add support/resistance levels
    levels = {
        'support': [145.0, 140.0],
        'resistance': [160.0, 165.0]
    }
    fig = chart.add_support_resistance_lines(fig, levels, data)
    
    print("✓ Candlestick chart created")
    return fig


def demo_prediction_chart():
    """Demonstrate prediction visualization."""
    print("\n=== Prediction Chart Demo ===")
    
    # Generate historical data
    historical = generate_sample_ohlcv_data(days=100, start_price=150.0)
    
    # Generate predictions
    last_price = historical['close'].iloc[-1]
    pred_days = 10
    pred_dates = pd.date_range(
        start=historical['date'].iloc[-1] + timedelta(days=1),
        periods=pred_days,
        freq='D'
    )
    
    # Simulate predictions with confidence intervals
    pred_returns = np.random.normal(0.002, 0.015, pred_days)
    predicted_prices = last_price * np.exp(np.cumsum(pred_returns))
    
    predictions = pd.DataFrame({
        'date': pred_dates,
        'predicted_price': predicted_prices,
        'lower_bound': predicted_prices * 0.95,
        'upper_bound': predicted_prices * 1.05,
        'confidence': np.random.uniform(0.7, 0.9, pred_days)
    })
    
    # Create chart
    chart = PredictionChart()
    fig = chart.create_prediction_chart(
        historical_data=historical,
        predictions=predictions,
        symbol='AAPL',
        confidence_level=0.95
    )
    
    print("✓ Prediction chart created")
    return fig


def demo_portfolio_charts():
    """Demonstrate portfolio visualizations."""
    print("\n=== Portfolio Charts Demo ===")
    
    # Generate portfolio returns
    days = 252  # One year
    returns = np.random.normal(0.0005, 0.01, days)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Create equity curve
    chart = PortfolioChart()
    equity_fig = chart.create_equity_curve(
        returns=returns,
        dates=dates,
        title="Portfolio Performance"
    )
    
    # Create efficient frontier
    frontier_points = [
        (0.10, 0.08), (0.12, 0.10), (0.14, 0.11),
        (0.16, 0.12), (0.18, 0.13), (0.20, 0.13),
        (0.22, 0.13), (0.25, 0.12)
    ]
    optimal = (0.16, 0.12)
    assets = {
        'AAPL': (0.20, 0.15),
        'MSFT': (0.18, 0.12),
        'GOOGL': (0.22, 0.14),
        'BTC': (0.50, 0.30),
        'ETH': (0.45, 0.25)
    }
    
    frontier_fig = chart.create_efficient_frontier(
        frontier_points=frontier_points,
        optimal_portfolio=optimal,
        individual_assets=assets
    )
    
    # Create allocation pie chart
    allocation = {
        'AAPL': 0.30,
        'MSFT': 0.25,
        'GOOGL': 0.20,
        'BTC': 0.15,
        'ETH': 0.10
    }
    
    allocation_fig = chart.create_allocation_pie(
        weights=allocation,
        title="Portfolio Allocation"
    )
    
    # Create monthly returns heatmap
    heatmap_fig = chart.create_monthly_returns_heatmap(
        returns=returns,
        dates=dates
    )
    
    print("✓ Portfolio charts created")
    return {
        'equity_curve': equity_fig,
        'efficient_frontier': frontier_fig,
        'allocation': allocation_fig,
        'monthly_returns': heatmap_fig
    }


def demo_correlation_charts():
    """Demonstrate correlation visualizations."""
    print("\n=== Correlation Charts Demo ===")
    
    # Generate correlation matrix
    assets = ['AAPL', 'MSFT', 'GOOGL', 'BTC', 'ETH']
    n = len(assets)
    
    # Create random correlation matrix
    random_matrix = np.random.rand(n, n)
    corr_matrix = (random_matrix + random_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = (corr_matrix - 0.5) * 2  # Scale to [-1, 1]
    
    corr_df = pd.DataFrame(corr_matrix, columns=assets, index=assets)
    
    # Create correlation heatmap
    chart = CorrelationChart()
    heatmap_fig = chart.create_correlation_heatmap(
        correlation_matrix=corr_df,
        title="Asset Correlation Matrix"
    )
    
    # Create rolling correlation chart
    days = 100
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    rolling_corr = {
        'AAPL-MSFT': np.random.uniform(0.5, 0.9, days),
        'BTC-ETH': np.random.uniform(0.7, 0.95, days),
        'AAPL-BTC': np.random.uniform(-0.2, 0.3, days)
    }
    
    rolling_fig = chart.create_rolling_correlation_chart(
        dates=dates,
        correlations=rolling_corr,
        title="Rolling 30-Day Correlations"
    )
    
    # Create correlation network
    network_fig = chart.create_correlation_network(
        correlation_matrix=corr_df,
        threshold=0.5,
        title="Correlation Network"
    )
    
    print("✓ Correlation charts created")
    return {
        'heatmap': heatmap_fig,
        'rolling': rolling_fig,
        'network': network_fig
    }


def demo_chart_export():
    """Demonstrate chart export capabilities."""
    print("\n=== Chart Export Demo ===")
    
    # Create a sample chart
    data = generate_sample_ohlcv_data(days=50, start_price=150.0)
    chart = CandlestickChart()
    fig = chart.create_chart(data=data, symbol='DEMO', volume=True)
    
    # Initialize exporter
    exporter = ChartExporter(output_dir=Path('exports'))
    
    # Export to HTML
    html_path = exporter.export_to_html(fig, 'demo_chart')
    print(f"✓ Exported to HTML: {html_path}")
    
    # Export to PNG (requires kaleido)
    try:
        png_path = exporter.export_to_png(fig, 'demo_chart')
        if png_path:
            print(f"✓ Exported to PNG: {png_path}")
    except Exception as e:
        print(f"⚠ PNG export skipped: {e}")
    
    # Export to JSON
    json_path = exporter.export_to_json(fig, 'demo_chart')
    print(f"✓ Exported to JSON: {json_path}")
    
    # Export data to CSV
    csv_path = exporter.export_data_to_csv(data, 'demo_data')
    print(f"✓ Exported data to CSV: {csv_path}")
    
    # Export data to Excel (requires openpyxl)
    try:
        excel_path = exporter.export_data_to_excel(data, 'demo_data')
        if excel_path:
            print(f"✓ Exported data to Excel: {excel_path}")
    except Exception as e:
        print(f"⚠ Excel export skipped: {e}")


def demo_report_generation():
    """Demonstrate PDF report generation."""
    print("\n=== Report Generation Demo ===")
    
    try:
        # Initialize report generator
        generator = ReportGenerator(output_dir=Path('reports'))
        
        # Generate prediction report
        prediction_data = {
            'metrics': {
                'accuracy': 0.85,
                'confidence': 0.78,
                'mae': 2.34,
                'rmse': 3.12
            },
            'predictions': [
                {
                    'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                    'predicted_price': 150.0 + i * 0.5,
                    'confidence': 0.85 - i * 0.02,
                    'lower_bound': 145.0 + i * 0.5,
                    'upper_bound': 155.0 + i * 0.5
                }
                for i in range(10)
            ]
        }
        
        # Create sample charts
        data = generate_sample_ohlcv_data(days=50)
        chart = CandlestickChart()
        candlestick_fig = chart.create_chart(data, 'DEMO')
        
        pred_chart = PredictionChart()
        historical = generate_sample_ohlcv_data(days=100)
        predictions = pd.DataFrame(prediction_data['predictions'])
        pred_fig = pred_chart.create_prediction_chart(historical, predictions, 'DEMO')
        
        charts = {
            'candlestick': candlestick_fig,
            'predictions': pred_fig
        }
        
        # Generate report
        report_path = generator.generate_prediction_report(
            symbol='DEMO',
            prediction_data=prediction_data,
            charts=charts
        )
        
        print(f"✓ Prediction report generated: {report_path}")
        
    except ImportError as e:
        print(f"⚠ Report generation skipped: {e}")
        print("  Install reportlab: pip install reportlab")


def main():
    """Run all visualization demos."""
    print("=" * 60)
    print("ARA AI Visualization System Demo")
    print("=" * 60)
    
    # Run demos
    demo_candlestick_chart()
    demo_prediction_chart()
    demo_portfolio_charts()
    demo_correlation_charts()
    demo_chart_export()
    demo_report_generation()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nCheck the following directories for outputs:")
    print("  - exports/  : Exported charts and data")
    print("  - reports/  : Generated PDF reports")
    print("\nNote: Some features require additional packages:")
    print("  - kaleido   : For PNG/SVG/PDF chart export")
    print("  - reportlab : For PDF report generation")
    print("  - openpyxl  : For Excel export")


if __name__ == '__main__':
    main()
