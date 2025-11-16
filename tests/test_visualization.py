"""
Tests for visualization module.

This module tests the visualization capabilities including:
- Chart creation
- Data export
- Report generation
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

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

# Check if plotly is available
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data."""
    days = 50
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    close = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, days)))
    high = close * (1 + np.abs(np.random.normal(0, 0.01, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, days)))
    open_price = np.roll(close, 1)
    open_price[0] = 100.0
    volume = np.random.randint(1000000, 10000000, days)
    
    return pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


@pytest.fixture
def sample_predictions():
    """Generate sample predictions."""
    pred_days = 10
    pred_dates = pd.date_range(start=datetime.now(), periods=pred_days, freq='D')
    
    predicted_prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.002, 0.015, pred_days)))
    
    return pd.DataFrame({
        'date': pred_dates,
        'predicted_price': predicted_prices,
        'lower_bound': predicted_prices * 0.95,
        'upper_bound': predicted_prices * 1.05,
        'confidence': np.random.uniform(0.7, 0.9, pred_days)
    })


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestChartFactory:
    """Test ChartFactory class."""
    
    def test_initialization(self):
        """Test chart factory initialization."""
        factory = ChartFactory()
        assert factory.theme == "plotly_white"
    
    def test_create_line_chart(self):
        """Test line chart creation."""
        factory = ChartFactory()
        
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'value1': np.random.rand(10),
            'value2': np.random.rand(10)
        })
        
        fig = factory.create_line_chart(
            data=data,
            x_col='date',
            y_cols=['value1', 'value2'],
            title='Test Chart'
        )
        
        assert fig is not None
        assert len(fig.data) == 2
    
    def test_create_heatmap(self):
        """Test heatmap creation."""
        factory = ChartFactory()
        
        data = np.random.rand(5, 5)
        labels = ['A', 'B', 'C', 'D', 'E']
        
        fig = factory.create_heatmap(
            data=data,
            x_labels=labels,
            y_labels=labels,
            title='Test Heatmap'
        )
        
        assert fig is not None
        assert len(fig.data) == 1


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestCandlestickChart:
    """Test CandlestickChart class."""
    
    def test_initialization(self):
        """Test candlestick chart initialization."""
        chart = CandlestickChart()
        assert chart.theme == "plotly_white"
    
    def test_create_chart(self, sample_ohlcv_data):
        """Test candlestick chart creation."""
        chart = CandlestickChart()
        
        fig = chart.create_chart(
            data=sample_ohlcv_data,
            symbol='TEST',
            volume=True
        )
        
        assert fig is not None
        assert len(fig.data) >= 2  # Candlestick + volume
    
    def test_create_chart_with_indicators(self, sample_ohlcv_data):
        """Test candlestick chart with indicators."""
        chart = CandlestickChart()
        
        indicators = {
            'SMA_20': sample_ohlcv_data['close'].rolling(20).mean()
        }
        
        fig = chart.create_chart(
            data=sample_ohlcv_data,
            symbol='TEST',
            indicators=indicators,
            volume=True
        )
        
        assert fig is not None
        assert len(fig.data) >= 3  # Candlestick + indicator + volume


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestPredictionChart:
    """Test PredictionChart class."""
    
    def test_initialization(self):
        """Test prediction chart initialization."""
        chart = PredictionChart()
        assert chart.theme == "plotly_white"
    
    def test_create_prediction_chart(self, sample_ohlcv_data, sample_predictions):
        """Test prediction chart creation."""
        chart = PredictionChart()
        
        fig = chart.create_prediction_chart(
            historical_data=sample_ohlcv_data,
            predictions=sample_predictions,
            symbol='TEST'
        )
        
        assert fig is not None
        assert len(fig.data) >= 2  # Historical + predictions
    
    def test_create_confidence_evolution_chart(self, sample_predictions):
        """Test confidence evolution chart."""
        chart = PredictionChart()
        
        fig = chart.create_confidence_evolution_chart(
            predictions=sample_predictions,
            symbol='TEST'
        )
        
        assert fig is not None


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestPortfolioChart:
    """Test PortfolioChart class."""
    
    def test_initialization(self):
        """Test portfolio chart initialization."""
        chart = PortfolioChart()
        assert chart.theme == "plotly_white"
    
    def test_create_equity_curve(self):
        """Test equity curve creation."""
        chart = PortfolioChart()
        
        returns = np.random.normal(0.001, 0.01, 100)
        dates = pd.date_range('2024-01-01', periods=100)
        
        fig = chart.create_equity_curve(
            returns=returns,
            dates=dates
        )
        
        assert fig is not None
        assert len(fig.data) >= 2  # Equity + drawdown
    
    def test_create_efficient_frontier(self):
        """Test efficient frontier creation."""
        chart = PortfolioChart()
        
        frontier = [(0.1 + i*0.02, 0.08 + i*0.01) for i in range(10)]
        optimal = (0.15, 0.12)
        
        fig = chart.create_efficient_frontier(
            frontier_points=frontier,
            optimal_portfolio=optimal
        )
        
        assert fig is not None
        assert len(fig.data) >= 2  # Frontier + optimal
    
    def test_create_allocation_pie(self):
        """Test allocation pie chart."""
        chart = PortfolioChart()
        
        weights = {
            'AAPL': 0.30,
            'MSFT': 0.25,
            'GOOGL': 0.25,
            'BTC': 0.20
        }
        
        fig = chart.create_allocation_pie(weights=weights)
        
        assert fig is not None
        assert len(fig.data) == 1


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestCorrelationChart:
    """Test CorrelationChart class."""
    
    def test_initialization(self):
        """Test correlation chart initialization."""
        chart = CorrelationChart()
        assert chart.theme == "plotly_white"
    
    def test_create_correlation_heatmap(self):
        """Test correlation heatmap creation."""
        chart = CorrelationChart()
        
        assets = ['AAPL', 'MSFT', 'GOOGL']
        corr_matrix = pd.DataFrame(
            np.random.rand(3, 3),
            columns=assets,
            index=assets
        )
        
        fig = chart.create_correlation_heatmap(
            correlation_matrix=corr_matrix
        )
        
        assert fig is not None
        assert len(fig.data) == 1
    
    def test_create_rolling_correlation_chart(self):
        """Test rolling correlation chart."""
        chart = CorrelationChart()
        
        dates = pd.date_range('2024-01-01', periods=50)
        correlations = {
            'AAPL-MSFT': np.random.uniform(0.5, 0.9, 50)
        }
        
        fig = chart.create_rolling_correlation_chart(
            dates=dates,
            correlations=correlations
        )
        
        assert fig is not None
        assert len(fig.data) >= 1


@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
class TestChartExporter:
    """Test ChartExporter class."""
    
    def test_initialization(self, temp_output_dir):
        """Test exporter initialization."""
        exporter = ChartExporter(output_dir=temp_output_dir)
        assert exporter.output_dir == temp_output_dir
        assert temp_output_dir.exists()
    
    def test_export_to_html(self, temp_output_dir, sample_ohlcv_data):
        """Test HTML export."""
        exporter = ChartExporter(output_dir=temp_output_dir)
        
        chart = CandlestickChart()
        fig = chart.create_chart(data=sample_ohlcv_data, symbol='TEST')
        
        filepath = exporter.export_to_html(fig, 'test_chart')
        
        assert filepath.exists()
        assert filepath.suffix == '.html'
    
    def test_export_to_json(self, temp_output_dir, sample_ohlcv_data):
        """Test JSON export."""
        exporter = ChartExporter(output_dir=temp_output_dir)
        
        chart = CandlestickChart()
        fig = chart.create_chart(data=sample_ohlcv_data, symbol='TEST')
        
        filepath = exporter.export_to_json(fig, 'test_chart')
        
        assert filepath.exists()
        assert filepath.suffix == '.json'
    
    def test_export_data_to_csv(self, temp_output_dir, sample_ohlcv_data):
        """Test CSV export."""
        exporter = ChartExporter(output_dir=temp_output_dir)
        
        filepath = exporter.export_data_to_csv(sample_ohlcv_data, 'test_data')
        
        assert filepath.exists()
        assert filepath.suffix == '.csv'


def test_module_imports():
    """Test that all modules can be imported."""
    from ara.visualization import (
        ChartFactory,
        CandlestickChart,
        PredictionChart,
        PortfolioChart,
        CorrelationChart,
        ChartExporter,
        ReportGenerator
    )
    
    assert ChartFactory is not None
    assert CandlestickChart is not None
    assert PredictionChart is not None
    assert PortfolioChart is not None
    assert CorrelationChart is not None
    assert ChartExporter is not None
    assert ReportGenerator is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
