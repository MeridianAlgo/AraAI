# ARA AI Visualization System

Comprehensive visualization system for financial data analysis, predictions, and portfolio management.

## Features

### Interactive Charts
- **Candlestick Charts**: OHLCV data with technical indicators overlay
- **Prediction Charts**: Price predictions with confidence intervals
- **Portfolio Charts**: Equity curves, efficient frontier, allocation
- **Correlation Charts**: Heatmaps, rolling correlations, network graphs

### Chart Export
- **HTML**: Interactive web-based charts
- **PNG**: High-resolution raster images
- **SVG**: Scalable vector graphics
- **PDF**: Portable document format
- **JSON**: Chart data and configuration

### Data Export
- **CSV**: Comma-separated values
- **Excel**: Multi-sheet workbooks
- **JSON**: Structured data format

### Report Generation
- **Prediction Reports**: Comprehensive prediction analysis with charts
- **Backtest Reports**: Performance metrics and visualizations
- **Portfolio Reports**: Allocation, risk metrics, and performance

## Installation

### Required Dependencies
```bash
pip install plotly pandas numpy
```

### Optional Dependencies
```bash
# For image export (PNG, SVG, PDF)
pip install kaleido

# For PDF reports
pip install reportlab

# For Excel export
pip install openpyxl
```

## Quick Start

### 1. Candlestick Chart with Indicators

```python
from ara.visualization import CandlestickChart
import pandas as pd

# Create chart
chart = CandlestickChart()

# Prepare data
data = pd.DataFrame({
    'date': dates,
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volumes
})

# Add indicators
indicators = {
    'SMA_20': sma_20_series,
    'SMA_50': sma_50_series,
    'BB_Upper': bb_upper_series,
    'BB_Lower': bb_lower_series
}

# Create chart
fig = chart.create_chart(
    data=data,
    symbol='AAPL',
    indicators=indicators,
    volume=True
)

# Show chart
fig.show()
```

### 2. Prediction Visualization

```python
from ara.visualization import PredictionChart

chart = PredictionChart()

# Historical data
historical = pd.DataFrame({
    'date': historical_dates,
    'close': historical_prices
})

# Predictions
predictions = pd.DataFrame({
    'date': prediction_dates,
    'predicted_price': predicted_prices,
    'lower_bound': lower_bounds,
    'upper_bound': upper_bounds,
    'confidence': confidence_scores
})

# Create chart
fig = chart.create_prediction_chart(
    historical_data=historical,
    predictions=predictions,
    symbol='BTC-USD',
    confidence_level=0.95
)

fig.show()
```

### 3. Portfolio Equity Curve

```python
from ara.visualization import PortfolioChart
import numpy as np

chart = PortfolioChart()

# Portfolio returns
returns = np.array([0.01, -0.005, 0.02, ...])
dates = pd.date_range('2023-01-01', periods=len(returns))

# Create equity curve
fig = chart.create_equity_curve(
    returns=returns,
    dates=dates,
    title="My Portfolio Performance"
)

fig.show()
```

### 4. Correlation Heatmap

```python
from ara.visualization import CorrelationChart
import pandas as pd

chart = CorrelationChart()

# Correlation matrix
corr_matrix = pd.DataFrame(
    correlation_data,
    columns=['AAPL', 'MSFT', 'GOOGL', 'BTC', 'ETH'],
    index=['AAPL', 'MSFT', 'GOOGL', 'BTC', 'ETH']
)

# Create heatmap
fig = chart.create_correlation_heatmap(
    correlation_matrix=corr_matrix,
    title="Asset Correlation Matrix"
)

fig.show()
```

### 5. Efficient Frontier

```python
from ara.visualization import PortfolioChart

chart = PortfolioChart()

# Efficient frontier points
frontier = [(0.10, 0.08), (0.12, 0.10), (0.15, 0.12), ...]

# Optimal portfolio
optimal = (0.12, 0.10)

# Individual assets
assets = {
    'AAPL': (0.20, 0.15),
    'MSFT': (0.18, 0.12),
    'BTC': (0.50, 0.30)
}

# Create chart
fig = chart.create_efficient_frontier(
    frontier_points=frontier,
    optimal_portfolio=optimal,
    individual_assets=assets
)

fig.show()
```

## Chart Export

### Export to Multiple Formats

```python
from ara.visualization import ChartExporter

exporter = ChartExporter(output_dir='exports')

# Export to HTML
exporter.export_to_html(fig, 'my_chart')

# Export to PNG (requires kaleido)
exporter.export_to_png(fig, 'my_chart', width=1200, height=800)

# Export to SVG (requires kaleido)
exporter.export_to_svg(fig, 'my_chart')

# Export to PDF (requires kaleido)
exporter.export_to_pdf(fig, 'my_chart')

# Export to JSON
exporter.export_to_json(fig, 'my_chart')
```

### Export Data

```python
# Export to CSV
exporter.export_data_to_csv(data_df, 'my_data')

# Export to Excel (requires openpyxl)
exporter.export_data_to_excel(data_df, 'my_data')

# Export multiple sheets
sheets = {
    'Predictions': predictions_df,
    'Metrics': metrics_df,
    'Analysis': analysis_df
}
exporter.export_data_to_excel(sheets, 'comprehensive_data')
```

## Report Generation

### Prediction Report

```python
from ara.visualization import ReportGenerator

generator = ReportGenerator(output_dir='reports')

# Prepare data
prediction_data = {
    'metrics': {
        'accuracy': 0.85,
        'confidence': 0.78,
        'mae': 2.34
    },
    'predictions': [
        {'date': '2024-01-01', 'predicted_price': 150.0, 'confidence': 0.85},
        # ... more predictions
    ]
}

# Prepare charts
charts = {
    'prediction_chart': prediction_fig,
    'confidence_chart': confidence_fig
}

# Generate report
report_path = generator.generate_prediction_report(
    symbol='AAPL',
    prediction_data=prediction_data,
    charts=charts
)

print(f"Report saved to: {report_path}")
```

### Backtest Report

```python
backtest_results = {
    'start_date': '2020-01-01',
    'end_date': '2023-12-31',
    'metrics': {
        'total_return': 0.45,
        'sharpe_ratio': 1.8,
        'max_drawdown': -0.15
    },
    'trade_statistics': {
        'total_trades': 150,
        'win_rate': 0.62,
        'avg_win': 0.025
    }
}

charts = {
    'equity_curve': equity_fig,
    'monthly_returns': monthly_fig
}

report_path = generator.generate_backtest_report(
    symbol='AAPL',
    backtest_results=backtest_results,
    charts=charts
)
```

### Portfolio Report

```python
portfolio_data = {
    'allocation': {
        'AAPL': 0.30,
        'MSFT': 0.25,
        'GOOGL': 0.20,
        'BTC': 0.15,
        'ETH': 0.10
    },
    'metrics': {
        'total_return': 0.35,
        'sharpe_ratio': 1.5,
        'volatility': 0.18
    },
    'risk_metrics': {
        'var_95': -0.05,
        'cvar_95': -0.08,
        'max_drawdown': -0.20
    }
}

charts = {
    'allocation_pie': allocation_fig,
    'equity_curve': equity_fig,
    'efficient_frontier': frontier_fig
}

report_path = generator.generate_portfolio_report(
    portfolio_data=portfolio_data,
    charts=charts
)
```

## Advanced Features

### Multi-Timeframe Analysis

```python
from ara.visualization import CandlestickChart

chart = CandlestickChart()

# Data for different timeframes
data_dict = {
    '1D': daily_data,
    '4H': four_hour_data,
    '1H': hourly_data
}

fig = chart.create_multi_timeframe_chart(
    data_dict=data_dict,
    symbol='BTC-USD'
)

fig.show()
```

### Pattern Annotations

```python
# Add pattern annotations to candlestick chart
patterns = [
    {
        'date': '2024-01-15',
        'type': 'Head and Shoulders',
        'price': 150.0,
        'description': 'Bearish reversal pattern'
    },
    # ... more patterns
]

fig = chart.create_chart(data, 'AAPL')
fig = chart.add_pattern_annotations(fig, patterns, data)
fig.show()
```

### Support/Resistance Lines

```python
levels = {
    'support': [145.0, 140.0],
    'resistance': [155.0, 160.0]
}

fig = chart.create_chart(data, 'AAPL')
fig = chart.add_support_resistance_lines(fig, levels, data)
fig.show()
```

### Rolling Correlations

```python
from ara.visualization import CorrelationChart

chart = CorrelationChart()

correlations = {
    'AAPL-MSFT': correlation_series_1,
    'BTC-ETH': correlation_series_2,
    'AAPL-BTC': correlation_series_3
}

fig = chart.create_rolling_correlation_chart(
    dates=dates,
    correlations=correlations,
    title="Rolling 30-Day Correlations"
)

fig.show()
```

### Correlation Network

```python
# Create network graph of correlations
fig = chart.create_correlation_network(
    correlation_matrix=corr_matrix,
    threshold=0.5,  # Only show correlations > 0.5
    title="Asset Correlation Network"
)

fig.show()
```

## Customization

### Themes

```python
# Available themes
themes = [
    'plotly',
    'plotly_white',
    'plotly_dark',
    'ggplot2',
    'seaborn',
    'simple_white'
]

# Use custom theme
chart = CandlestickChart(theme='plotly_dark')
```

### Chart Dimensions

```python
# Custom height
fig = chart.create_chart(data, 'AAPL', height=1000)

# Custom export dimensions
exporter.export_to_png(
    fig,
    'my_chart',
    width=1920,
    height=1080,
    scale=3.0  # Higher resolution
)
```

## Best Practices

1. **Use appropriate chart types**: Choose the right visualization for your data
2. **Include confidence intervals**: Always show uncertainty in predictions
3. **Export to multiple formats**: Provide HTML for interactivity and PNG for reports
4. **Generate comprehensive reports**: Include both charts and metrics
5. **Optimize for performance**: Use appropriate data sampling for large datasets
6. **Maintain consistency**: Use consistent colors and styles across charts

## Troubleshooting

### Plotly Not Installed
```bash
pip install plotly
```

### Image Export Not Working
```bash
# Install kaleido for static image export
pip install kaleido
```

### PDF Reports Not Working
```bash
# Install reportlab
pip install reportlab
```

### Excel Export Not Working
```bash
# Install openpyxl
pip install openpyxl
```

## Examples

See the `examples/visualization_demo.py` file for comprehensive examples of all visualization features.

## API Reference

For detailed API documentation, see the docstrings in each module:
- `chart_factory.py`: Basic chart creation
- `candlestick.py`: Candlestick charts with indicators
- `predictions.py`: Prediction visualizations
- `portfolio.py`: Portfolio analysis charts
- `correlation.py`: Correlation analysis
- `exporter.py`: Chart and data export
