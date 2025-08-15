# MeridianAlgo Python Package

**Advanced AI Stock Analysis with Ensemble ML Models and Intelligent Caching**

## 🚀 Quick Installation

```bash
pip install meridianalgo
```

## 🏗️ Package Architecture

MeridianAlgo is a comprehensive stock analysis platform that includes multiple components working together:

### 📦 Core Components

1. **AraAI Engine** (`meridianalgo.core`) - Main prediction system
2. **Ensemble ML Models** (`meridianalgo.models`) - Random Forest + Gradient Boosting + LSTM
3. **Market Data Manager** (`meridianalgo.data`) - Real-time data fetching and technical indicators
4. **GPU Manager** (`meridianalgo.utils`) - Multi-vendor GPU acceleration
5. **Cache System** (`meridianalgo.utils`) - Intelligent prediction caching
6. **Accuracy Tracker** (`meridianalgo.utils`) - Performance monitoring and validation
7. **Console Manager** (`meridianalgo.console`) - Rich output formatting
8. **CLI Interface** (`meridianalgo.cli`) - Command-line tools

### 🔄 How It All Works Together

When you install MeridianAlgo, you get access to both the **`ara` CLI command** and the **full Python API**. Here's how they connect:

```
User Input → CLI/Python API → AraAI Engine → ML Models → Predictions
     ↓              ↓              ↓           ↓           ↓
Cache Check → GPU Detection → Data Fetching → Processing → Output
```

## 📖 Complete Usage Guide

### 1. Quick Start Functions

```python
from meridianalgo import quick_predict, analyze_accuracy, get_version_info, check_gpu_support

# Quick stock prediction (uses full AraAI system internally)
result = quick_predict('AAPL', days=5)
print(f"AAPL 5-day predictions: {result}")

# Analyze prediction accuracy across all models
accuracy = analyze_accuracy('AAPL')
print(f"AAPL accuracy: {accuracy['accuracy_rate']:.1f}%")

# Check system capabilities
version_info = get_version_info()
print(f"Features: {version_info['features']}")

# Check GPU support
gpu_info = check_gpu_support()
print(f"Available GPUs: {gpu_info['details']}")
```

### 2. Full AraAI System Usage

```python
from meridianalgo import AraAI, StockPredictor

# Initialize the complete Ara AI system (this is what powers the 'ara' command)
ara = AraAI(verbose=True)

# The AraAI system automatically:
# - Detects best GPU (NVIDIA/AMD/Intel/Apple)
# - Initializes ensemble ML models
# - Sets up intelligent caching
# - Configures accuracy tracking

# Make detailed predictions with full system
result = ara.predict('TSLA', days=7, use_cache=True)
print(f"Predictions: {result['predictions']}")
print(f"Model info: {result.get('model_info', {})}")

# Get comprehensive system information
system_info = ara.get_system_info()
print(f"Device: {system_info['device']}")
print(f"GPU Info: {system_info['gpu_info']}")
print(f"Cache Stats: {system_info['cache_stats']}")

# Validate and analyze all previous predictions
validation = ara.validate_predictions()
print(f"Validation accuracy: {validation['accuracy_rate']:.1f}%")

# Analyze accuracy for specific symbol
accuracy = ara.analyze_accuracy('TSLA')
print(f"TSLA historical accuracy: {accuracy}")
```

### 3. Using Individual Components

The package is modular - you can use individual components independently:

#### A. Ensemble ML Models

```python
from meridianalgo.models import EnsembleMLSystem, LSTMModel
from meridianalgo.utils import GPUManager
import torch

# Initialize GPU manager
gpu_manager = GPUManager()
device = gpu_manager.get_best_device()
print(f"Using device: {device}")

# Create ensemble ML system (what powers predictions)
ml_system = EnsembleMLSystem(device=device)

# The ensemble includes:
# - Random Forest (robust tree-based predictions)
# - Gradient Boosting (advanced boosting algorithm)  
# - LSTM Neural Network (deep learning with attention)

# Get model information
model_info = ml_system.get_model_info()
print(f"Available models: {model_info['models']}")
print(f"Ensemble weights: {model_info['ensemble_weights']}")

# You can also use individual LSTM model
lstm_model = LSTMModel(input_size=22, hidden_size=128, num_layers=3)
print(f"LSTM architecture: {lstm_model}")
```

#### B. Market Data and Technical Analysis

```python
from meridianalgo.data import MarketDataManager, TechnicalIndicators
import pandas as pd

# Initialize data manager (handles Yahoo Finance integration)
data_manager = MarketDataManager()

# Get stock data with intelligent caching
data = data_manager.get_stock_data('AAPL', period='1y')
print(f"Data points: {len(data)}")

# Get current price
current_price = data_manager.get_current_price('AAPL')
print(f"Current AAPL price: ${current_price:.2f}")

# Get comprehensive stock information
stock_info = data_manager.get_stock_info('AAPL')
print(f"Company: {stock_info['name']}")
print(f"Sector: {stock_info['sector']}")
print(f"Market Cap: ${stock_info['market_cap']:,}")

# Get full analysis
analysis = data_manager.get_stock_analysis('AAPL')
print(f"Price change: {analysis['price_change_pct']:.1f}%")
print(f"Volume ratio: {analysis['volume_ratio']:.2f}")

# Calculate technical indicators (20+ indicators)
indicators = TechnicalIndicators()
enhanced_data = indicators.calculate_all_indicators(data)

# Available indicators include:
print("Technical Indicators calculated:")
indicator_columns = [col for col in enhanced_data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
for indicator in indicator_columns[:10]:  # Show first 10
    print(f"  - {indicator}: {enhanced_data[indicator].iloc[-1]:.2f}")
```

#### C. GPU Management and Performance

```python
from meridianalgo.utils import GPUManager
import torch

# Initialize GPU manager (auto-detects all GPU types)
gpu_manager = GPUManager()

# Detect all available GPUs
gpu_info = gpu_manager.detect_gpu_vendor()
print("GPU Detection Results:")
print(f"NVIDIA CUDA: {gpu_info['nvidia']}")
print(f"AMD ROCm/DirectML: {gpu_info['amd']}")
print(f"Intel XPU: {gpu_info['intel']}")
print(f"Apple MPS: {gpu_info['apple']}")

# Get detailed GPU information
for detail in gpu_info['details']:
    print(f"  Available: {detail}")

# Get the best device for computation
device = gpu_manager.get_best_device()
device_name = gpu_manager.get_device_name()
print(f"Selected device: {device}")
print(f"Device name: {device_name}")

# The GPU manager automatically prioritizes:
# 1. NVIDIA CUDA (best ML performance)
# 2. AMD ROCm/DirectML 
# 3. Intel XPU (Arc GPUs)
# 4. Apple MPS (Apple Silicon)
# 5. Optimized CPU (fallback)
```

#### D. Intelligent Caching System

```python
from meridianalgo.utils import CacheManager
from datetime import datetime

# Initialize cache manager
cache_manager = CacheManager()

# Check for cached predictions
cached_data = cache_manager.check_cached_predictions('AAPL', days=5)
if cached_data:
    print("Found cached predictions:")
    for pred in cached_data['predictions']:
        print(f"  Day {pred['day']}: ${pred['predicted_price']:.2f}")
    
    # The cache system automatically:
    # - Checks prediction age
    # - Validates accuracy
    # - Prompts user for choice
    choice = cache_manager.ask_user_choice('AAPL', cached_data)
    print(f"User choice: {choice}")

# Get cache statistics
cache_stats = cache_manager.get_cache_stats()
print(f"Cache stats: {cache_stats}")
```

#### E. Accuracy Tracking and Validation

```python
from meridianalgo.utils import AccuracyTracker

# Initialize accuracy tracker
tracker = AccuracyTracker()

# Validate all previous predictions against actual prices
validation_result = tracker.validate_predictions()
if validation_result:
    print("Validation Results:")
    print(f"  Total validated: {validation_result['validated']}")
    print(f"  Accuracy rate: {validation_result['accuracy_rate']:.1f}%")
    print(f"  Excellent predictions: {validation_result['excellent_rate']:.1f}%")
    print(f"  Average error: {validation_result['avg_error']:.2f}%")

# Analyze accuracy for specific symbol
accuracy_stats = tracker.analyze_accuracy('AAPL')
if accuracy_stats:
    print(f"\nAAPL Accuracy Analysis:")
    print(f"  Total predictions: {accuracy_stats['total_predictions']}")
    print(f"  Overall accuracy: {accuracy_stats['accuracy_rate']:.1f}%")
    print(f"  Recent performance: {accuracy_stats.get('recent_stats', {})}")

# Get overall system accuracy
overall_stats = tracker.get_accuracy_stats()
print(f"System-wide accuracy: {overall_stats}")
```

#### F. Rich Console Output

```python
from meridianalgo.console import ConsoleManager

# Initialize console manager (used by CLI and can be used independently)
console = ConsoleManager(verbose=True)

# The console manager provides rich formatting for:
console.print_system_info()  # System initialization
console.print_gpu_info({'nvidia': True, 'details': ['NVIDIA RTX 4090']})  # GPU info

# Format prediction results beautifully
sample_result = {
    'symbol': 'AAPL',
    'current_price': 202.38,
    'predictions': [
        {'day': 1, 'date': '2025-08-03', 'predicted_price': 199.46, 'change': -2.92, 'change_pct': -1.4},
        {'day': 2, 'date': '2025-08-04', 'predicted_price': 196.55, 'change': -5.83, 'change_pct': -2.9}
    ]
}
console.print_prediction_results(sample_result)

# Display accuracy summaries
console.print_accuracy_summary({
    'symbol': 'AAPL',
    'total_predictions': 150,
    'accuracy_rate': 82.5,
    'excellent_rate': 28.3,
    'avg_error': 1.8
})
```

### 4. How the `ara` CLI Command Works

When you run `ara AAPL --days 5`, here's what happens internally:

```python
# This is essentially what the 'ara' command does:

from meridianalgo.cli import main
from meridianalgo.core import AraAI
from meridianalgo.console import ConsoleManager

# 1. Parse command line arguments
# 2. Initialize console manager for rich output
console = ConsoleManager(verbose=args.verbose)

# 3. Initialize the full AraAI system
ara = AraAI(verbose=args.verbose)
# This automatically:
# - Detects and configures GPU
# - Initializes ensemble ML models
# - Sets up caching system
# - Configures accuracy tracking

# 4. Make prediction with intelligent caching
result = ara.predict(symbol='AAPL', days=5, use_cache=True)
# This process:
# - Checks for cached predictions
# - Asks user preference (cached vs fresh)
# - Fetches market data if needed
# - Calculates technical indicators
# - Runs ensemble ML prediction
# - Saves results to cache
# - Updates accuracy tracking

# 5. Display results with rich formatting
console.print_prediction_results(result)
```

### 5. Complete Integration Example

Here's how all components work together in a real application:

```python
from meridianalgo import AraAI
from meridianalgo.utils import GPUManager, CacheManager, AccuracyTracker
from meridianalgo.data import MarketDataManager, TechnicalIndicators
from meridianalgo.models import EnsembleMLSystem
from meridianalgo.console import ConsoleManager

class StockAnalysisApp:
    def __init__(self):
        # Initialize all components (this is what AraAI does internally)
        self.console = ConsoleManager(verbose=True)
        self.gpu_manager = GPUManager()
        self.cache_manager = CacheManager()
        self.accuracy_tracker = AccuracyTracker()
        self.data_manager = MarketDataManager()
        self.indicators = TechnicalIndicators()
        
        # Get best device and initialize ML system
        device = self.gpu_manager.get_best_device()
        self.ml_system = EnsembleMLSystem(device=device)
        
        self.console.print_system_info()
        
    def analyze_portfolio(self, symbols, days=5):
        """Analyze multiple stocks using all components"""
        results = {}
        
        for symbol in symbols:
            print(f"\n🔍 Analyzing {symbol}...")
            
            # 1. Check cache first
            cached = self.cache_manager.check_cached_predictions(symbol, days)
            
            # 2. Get market data and technical indicators
            data = self.data_manager.get_stock_data(symbol)
            enhanced_data = self.indicators.calculate_all_indicators(data)
            
            # 3. Prepare features for ML models
            features = self._prepare_features(enhanced_data)
            
            # 4. Generate predictions using ensemble
            predictions = self.ml_system.predict(features, days=days)
            
            # 5. Format and cache results
            result = self._format_results(symbol, predictions, data)
            self.cache_manager.save_predictions(symbol, result)
            
            # 6. Display with rich formatting
            self.console.print_prediction_results(result)
            
            results[symbol] = result
            
        return results
    
    def validate_system_accuracy(self):
        """Validate accuracy across all components"""
        validation = self.accuracy_tracker.validate_predictions()
        self.console.print_validation_summary(validation)
        return validation
    
    def get_system_status(self):
        """Get comprehensive system status"""
        return {
            'gpu_info': self.gpu_manager.detect_gpu_vendor(),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'accuracy_stats': self.accuracy_tracker.get_accuracy_stats(),
            'ml_model_info': self.ml_system.get_model_info()
        }

# Usage
app = StockAnalysisApp()
portfolio_results = app.analyze_portfolio(['AAPL', 'TSLA', 'MSFT'])
system_accuracy = app.validate_system_accuracy()
system_status = app.get_system_status()
```

## 🖥️ Command Line Interface

The package includes a powerful CLI tool that provides access to all functionality:

### Basic Commands
```bash
# Basic prediction (uses full ensemble ML system)
ara AAPL

# Predict for specific number of days
ara TSLA --days 7

# Show detailed output with system information
ara MSFT --verbose

# Skip cache and generate fresh predictions
ara GOOGL --no-cache

# Show accuracy statistics for specific symbol
ara --accuracy AAPL

# Validate all previous predictions across all models
ara --validate

# Show comprehensive system and GPU information
ara --system-info
```

### Advanced CLI Usage
```bash
# Analyze multiple aspects
ara AAPL --days 10 --verbose    # Detailed 10-day prediction
ara --accuracy                  # Overall system accuracy
ara --validate                  # Validate all cached predictions

# The CLI automatically:
# - Detects best GPU (NVIDIA/AMD/Intel/Apple)
# - Uses ensemble ML models (RF + GB + LSTM)
# - Applies intelligent caching
# - Tracks prediction accuracy
# - Provides rich console output
```

## 🔄 Relationship: Standalone vs Package

### Original Standalone System (`ara.py`)
The original `ara.py` file contains the complete Ara AI system in a single file. When you install MeridianAlgo, you get the same functionality but organized into a professional package structure:

```
Original ara.py (2800+ lines) → MeridianAlgo Package Structure:
├── meridianalgo/core.py      # Main AraAI class
├── meridianalgo/models.py    # ML ensemble system  
├── meridianalgo/data.py      # Market data & indicators
├── meridianalgo/utils.py     # GPU, cache, accuracy tracking
├── meridianalgo/console.py   # Rich output formatting
└── meridianalgo/cli.py       # Command-line interface
```

### Key Differences:

| Feature | Standalone `ara.py` | MeridianAlgo Package |
|---------|-------------------|---------------------|
| **Installation** | Manual setup | `pip install meridianalgo` |
| **Usage** | `python ara.py AAPL` | `ara AAPL` (global command) |
| **Import** | Not importable | `from meridianalgo import AraAI` |
| **Modularity** | Single file | Modular components |
| **Distribution** | Manual sharing | PyPI distribution |
| **Updates** | Manual download | `pip install --upgrade` |

### Migration from Standalone
If you were using the standalone `ara.py`, you can easily migrate:

```python
# Old way (standalone ara.py)
# python ara.py AAPL --days 5

# New way (MeridianAlgo package)
ara AAPL --days 5

# Or in Python:
from meridianalgo import quick_predict
result = quick_predict('AAPL', days=5)
```

## 🎯 Key Features

### Ensemble ML Models
- **Random Forest**: Robust tree-based ensemble
- **Gradient Boosting**: Advanced boosting algorithm
- **LSTM Neural Network**: Deep learning with attention mechanism

### Multi-GPU Support
- **NVIDIA CUDA**: Full CUDA acceleration
- **AMD ROCm/DirectML**: AMD GPU support
- **Intel XPU**: Intel Arc GPU acceleration
- **Apple MPS**: Apple Silicon optimization

### Intelligent Caching
- **7-day Prediction Cycles**: Efficient caching system
- **Accuracy-based Validation**: Smart cache invalidation
- **User Choice Prompts**: Interactive cache management

### Technical Indicators (20+)
- Moving Averages (SMA, EMA)
- Momentum Indicators (RSI, MACD, Stochastic)
- Volatility Indicators (Bollinger Bands, ATR)
- Volume Indicators (OBV, MFI)

## 📊 Accuracy Performance

Based on extensive backtesting:

- **Overall Accuracy**: 78-85% (within 3% of actual price)
- **Excellent Predictions**: 25-35% (within 1% of actual price)
- **Good Predictions**: 45-55% (within 2% of actual price)
- **Average Error**: 1.8-2.3%

## 🔧 Configuration Options

### GPU Acceleration

```python
from meridianalgo.utils import GPUManager

# Check available GPUs
gpu_manager = GPUManager()
gpu_info = gpu_manager.detect_gpu_vendor()

# Get best device
device = gpu_manager.get_best_device()
print(f"Using: {device}")
```

### Cache Management

```python
from meridianalgo.utils import CacheManager

# Initialize cache manager
cache = CacheManager()

# Get cache statistics
stats = cache.get_cache_stats()
print(f"Cached predictions: {stats['total_predictions']}")
```

### Accuracy Tracking

```python
from meridianalgo.utils import AccuracyTracker

# Initialize accuracy tracker
tracker = AccuracyTracker()

# Analyze accuracy for specific symbol
accuracy = tracker.analyze_accuracy('AAPL')
print(f"AAPL accuracy: {accuracy}")

# Validate all predictions
validation = tracker.validate_predictions()
print(f"Validation results: {validation}")
```

## 🛠️ Advanced Configuration

### Custom Model Parameters

```python
from meridianalgo.models import EnsembleMLSystem
import torch

# Initialize with custom device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ml_system = EnsembleMLSystem(device=device)

# Get model information
model_info = ml_system.get_model_info()
print(f"Models: {model_info['models']}")
```

### Data Management

```python
from meridianalgo.data import MarketDataManager

# Initialize data manager
data_manager = MarketDataManager()

# Get comprehensive stock analysis
analysis = data_manager.get_stock_analysis('AAPL')
print(f"Current price: ${analysis['current_price']:.2f}")
print(f"Price change: {analysis['price_change_pct']:.1f}%")
```

## 📈 Real-World Integration Examples

### 1. Jupyter Notebook Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from meridianalgo import AraAI
from meridianalgo.data import MarketDataManager, TechnicalIndicators
from meridianalgo.utils import GPUManager

# Initialize components
ara = AraAI(verbose=True)
data_manager = MarketDataManager()
indicators = TechnicalIndicators()
gpu_manager = GPUManager()

print(f"Using device: {gpu_manager.get_device_name()}")

# Get comprehensive analysis
symbol = 'AAPL'
result = ara.predict(symbol, days=7)
data = data_manager.get_stock_data(symbol, period='6mo')
enhanced_data = indicators.calculate_all_indicators(data)

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 1. Price predictions
dates = [pred['date'][:10] for pred in result['predictions']]
prices = [pred['predicted_price'] for pred in result['predictions']]
ax1.plot(dates, prices, marker='o', linewidth=2, markersize=8)
ax1.set_title(f'{symbol} Price Predictions (Ensemble ML)')
ax1.set_ylabel('Price ($)')
ax1.tick_params(axis='x', rotation=45)

# 2. Technical indicators
ax2.plot(enhanced_data.index[-60:], enhanced_data['Close'][-60:], label='Price')
ax2.plot(enhanced_data.index[-60:], enhanced_data['SMA_20'][-60:], label='SMA 20')
ax2.plot(enhanced_data.index[-60:], enhanced_data['EMA_12'][-60:], label='EMA 12')
ax2.set_title('Technical Indicators')
ax2.legend()

# 3. RSI
ax3.plot(enhanced_data.index[-60:], enhanced_data['RSI'][-60:])
ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7)
ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7)
ax3.set_title('RSI (Relative Strength Index)')
ax3.set_ylabel('RSI')

# 4. Volume analysis
ax4.bar(enhanced_data.index[-30:], enhanced_data['Volume'][-30:])
ax4.set_title('Volume Analysis')
ax4.set_ylabel('Volume')

plt.tight_layout()
plt.show()

# Display prediction accuracy
accuracy = ara.analyze_accuracy(symbol)
if accuracy:
    print(f"\n{symbol} Prediction Accuracy:")
    print(f"  Overall: {accuracy['accuracy_rate']:.1f}%")
    print(f"  Excellent (<1% error): {accuracy['excellent_rate']:.1f}%")
    print(f"  Average error: {accuracy['avg_error']:.2f}%")
```

### 2. Portfolio Analysis Dashboard

```python
from meridianalgo import AraAI
from meridianalgo.utils import AccuracyTracker
import pandas as pd

class PortfolioDashboard:
    def __init__(self, symbols):
        self.symbols = symbols
        self.ara = AraAI(verbose=False)
        self.tracker = AccuracyTracker()
        
    def analyze_portfolio(self, days=5):
        """Analyze entire portfolio with ensemble ML"""
        portfolio_results = {}
        
        print("🔍 Portfolio Analysis Using Ensemble ML Models")
        print("=" * 60)
        
        for symbol in self.symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            # Get prediction using full ensemble system
            result = self.ara.predict(symbol, days=days, use_cache=True)
            
            if result:
                current_price = result['current_price']
                predictions = result['predictions']
                
                # Calculate portfolio metrics
                avg_predicted_return = sum(p['change_pct'] for p in predictions) / len(predictions)
                max_predicted_price = max(p['predicted_price'] for p in predictions)
                min_predicted_price = min(p['predicted_price'] for p in predictions)
                
                portfolio_results[symbol] = {
                    'current_price': current_price,
                    'avg_return': avg_predicted_return,
                    'price_range': (min_predicted_price, max_predicted_price),
                    'predictions': predictions,
                    'model_info': result.get('model_info', {})
                }
                
                print(f"  Current: ${current_price:.2f}")
                print(f"  Avg predicted return: {avg_predicted_return:+.1f}%")
                print(f"  Price range: ${min_predicted_price:.2f} - ${max_predicted_price:.2f}")
                
                # Show model information
                model_info = result.get('model_info', {})
                if model_info:
                    print(f"  Models used: {', '.join(model_info.get('models', []))}")
        
        return portfolio_results
    
    def get_portfolio_accuracy(self):
        """Get accuracy statistics for entire portfolio"""
        print("\n📈 Portfolio Accuracy Analysis")
        print("=" * 40)
        
        for symbol in self.symbols:
            accuracy = self.tracker.analyze_accuracy(symbol)
            if accuracy:
                print(f"\n{symbol}:")
                print(f"  Predictions: {accuracy['total_predictions']}")
                print(f"  Accuracy: {accuracy['accuracy_rate']:.1f}%")
                print(f"  Avg Error: {accuracy['avg_error']:.2f}%")
    
    def compare_models(self, symbol):
        """Compare individual model performance"""
        print(f"\n🤖 Model Comparison for {symbol}")
        print("=" * 40)
        
        # Get system info to see model details
        system_info = self.ara.get_system_info()
        model_info = system_info.get('model_info', {})
        
        print(f"Ensemble Models: {model_info.get('models', [])}")
        print(f"Model Weights: {model_info.get('ensemble_weights', {})}")
        print(f"Device: {system_info.get('device', 'Unknown')}")

# Usage
portfolio = PortfolioDashboard(['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN'])
results = portfolio.analyze_portfolio(days=7)
portfolio.get_portfolio_accuracy()
portfolio.compare_models('AAPL')
```

### 3. Automated Trading Signal Generator

```python
from meridianalgo import AraAI
from meridianalgo.data import MarketDataManager
from meridianalgo.utils import AccuracyTracker
import schedule
import time
from datetime import datetime

class TradingSignalGenerator:
    def __init__(self, watchlist, accuracy_threshold=75.0):
        self.watchlist = watchlist
        self.accuracy_threshold = accuracy_threshold
        self.ara = AraAI(verbose=False)
        self.data_manager = MarketDataManager()
        self.tracker = AccuracyTracker()
        
    def generate_signals(self):
        """Generate trading signals using ensemble ML predictions"""
        signals = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n🚨 Trading Signals Generated: {timestamp}")
        print("=" * 60)
        
        for symbol in self.watchlist:
            try:
                # Check historical accuracy first
                accuracy_stats = self.tracker.analyze_accuracy(symbol)
                
                # Only generate signals for stocks with good accuracy
                if accuracy_stats and accuracy_stats['accuracy_rate'] >= self.accuracy_threshold:
                    
                    # Get prediction using ensemble ML
                    result = self.ara.predict(symbol, days=3, use_cache=True)
                    
                    if result:
                        predictions = result['predictions']
                        current_price = result['current_price']
                        
                        # Calculate signal strength
                        day1_change = predictions[0]['change_pct']
                        day3_change = predictions[2]['change_pct']
                        
                        # Generate signal
                        if day1_change > 2.0 and day3_change > 3.0:
                            signal = "STRONG BUY"
                            confidence = min(95, accuracy_stats['accuracy_rate'])
                        elif day1_change > 1.0 and day3_change > 1.5:
                            signal = "BUY"
                            confidence = accuracy_stats['accuracy_rate']
                        elif day1_change < -2.0 and day3_change < -3.0:
                            signal = "STRONG SELL"
                            confidence = min(95, accuracy_stats['accuracy_rate'])
                        elif day1_change < -1.0 and day3_change < -1.5:
                            signal = "SELL"
                            confidence = accuracy_stats['accuracy_rate']
                        else:
                            signal = "HOLD"
                            confidence = accuracy_stats['accuracy_rate']
                        
                        signals.append({
                            'symbol': symbol,
                            'signal': signal,
                            'confidence': confidence,
                            'current_price': current_price,
                            'day1_prediction': predictions[0]['predicted_price'],
                            'day3_prediction': predictions[2]['predicted_price'],
                            'expected_return_3d': day3_change,
                            'model_accuracy': accuracy_stats['accuracy_rate']
                        })
                        
                        print(f"{symbol:6} | {signal:12} | Confidence: {confidence:5.1f}% | "
                              f"3D Return: {day3_change:+6.1f}% | Price: ${current_price:7.2f}")
                
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
        
        return signals
    
    def validate_signals(self):
        """Validate previous signals against actual performance"""
        validation = self.tracker.validate_predictions()
        if validation:
            print(f"\n✅ Signal Validation Results:")
            print(f"   Accuracy: {validation['accuracy_rate']:.1f}%")
            print(f"   Validated: {validation['validated']} signals")
            print(f"   Avg Error: {validation['avg_error']:.2f}%")
    
    def run_daily_analysis(self):
        """Daily analysis routine"""
        print(f"\n🔄 Daily Analysis - {datetime.now().strftime('%Y-%m-%d')}")
        
        # Generate new signals
        signals = self.generate_signals()
        
        # Validate previous signals
        self.validate_signals()
        
        # Show system status
        system_info = self.ara.get_system_info()
        print(f"\n💻 System Status:")
        print(f"   Device: {system_info['device']}")
        print(f"   Cache: {system_info['cache_stats']['total_predictions']} predictions")
        
        return signals

# Usage
watchlist = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']
signal_generator = TradingSignalGenerator(watchlist, accuracy_threshold=75.0)

# Run once
signals = signal_generator.run_daily_analysis()

# Schedule daily runs
schedule.every().day.at("09:30").do(signal_generator.run_daily_analysis)

# Keep running
# while True:
#     schedule.run_pending()
#     time.sleep(60)
```

### Flask Web Application

```python
from flask import Flask, jsonify, request
from meridianalgo import quick_predict

app = Flask(__name__)

@app.route('/predict/<symbol>')
def predict_stock(symbol):
    days = request.args.get('days', 5, type=int)
    result = quick_predict(symbol.upper(), days=days)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
```

### Automated Trading Bot

```python
import schedule
import time
from meridianalgo import AraAI

def daily_analysis():
    ara = AraAI()
    symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL']
    
    for symbol in symbols:
        result = ara.predict(symbol, days=1)
        if result:
            change_pct = result['predictions'][0]['change_pct']
            print(f"{symbol}: {change_pct:+.1f}% predicted change")

# Schedule daily analysis
schedule.every().day.at("09:00").do(daily_analysis)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 🔍 Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```python
   from meridianalgo.utils import GPUManager
   gpu_manager = GPUManager()
   print(gpu_manager.detect_gpu_vendor())
   ```

2. **Cache Issues**
   ```bash
   # Clear cache files
   rm predictions.csv prediction_accuracy.csv
   ```

3. **Data Fetching Errors**
   ```python
   # Check internet connection and symbol validity
   from meridianalgo.data import MarketDataManager
   data_manager = MarketDataManager()
   data = data_manager.get_stock_data('AAPL')
   print(f"Data points: {len(data) if data is not None else 0}")
   ```

## 🎯 All Ways to Use MeridianAlgo

### 1. Command Line Interface (CLI)
```bash
# Global 'ara' command (installed automatically)
ara AAPL --days 5
ara --system-info
ara --accuracy AAPL
ara --validate
```

### 2. Quick Functions (Easiest)
```python
from meridianalgo import quick_predict, analyze_accuracy
result = quick_predict('AAPL', days=5)
accuracy = analyze_accuracy('AAPL')
```

### 3. Full AraAI System (Most Powerful)
```python
from meridianalgo import AraAI
ara = AraAI(verbose=True)
result = ara.predict('AAPL', days=5)
system_info = ara.get_system_info()
```

### 4. Individual Components (Most Flexible)
```python
from meridianalgo.models import EnsembleMLSystem
from meridianalgo.data import MarketDataManager
from meridianalgo.utils import GPUManager, CacheManager
# Use components independently
```

### 5. Backward Compatibility Functions
```python
from meridianalgo import predict_stock, analyze_stock
result = predict_stock('AAPL', days=5, verbose=True)
analysis = analyze_stock('AAPL', verbose=True)
```

### 6. Alternative Entry Points
```python
# These all work the same way:
from meridianalgo.core import AraAI
from meridianalgo import AraAI
from meridianalgo.core import predict_stock
from meridianalgo import predict_stock
```

## 📚 Complete API Reference

### Core Classes

#### `AraAI` - Main Prediction Engine
```python
ara = AraAI(verbose=False)
ara.predict(symbol, days=5, use_cache=True)
ara.validate_predictions()
ara.analyze_accuracy(symbol=None)
ara.get_system_info()
```

#### `StockPredictor` - Simplified Interface
```python
predictor = StockPredictor(verbose=False)
predictor.predict(symbol, days=5)
predictor.analyze(symbol)
```

#### `EnsembleMLSystem` - ML Models
```python
ml_system = EnsembleMLSystem(device=device)
ml_system.train(features, targets)
ml_system.predict(features, days=5)
ml_system.get_model_info()
```

#### `MarketDataManager` - Data Management
```python
data_manager = MarketDataManager()
data_manager.get_stock_data(symbol, period='2y')
data_manager.get_current_price(symbol)
data_manager.get_stock_info(symbol)
data_manager.get_stock_analysis(symbol)
```

#### `TechnicalIndicators` - Technical Analysis
```python
indicators = TechnicalIndicators()
indicators.calculate_all_indicators(data)
indicators.add_moving_averages(data)
indicators.add_momentum_indicators(data)
indicators.calculate_rsi(prices)
```

#### `GPUManager` - GPU Management
```python
gpu_manager = GPUManager()
gpu_manager.detect_gpu_vendor()
gpu_manager.get_best_device()
gpu_manager.get_device_name()
```

#### `CacheManager` - Caching System
```python
cache_manager = CacheManager()
cache_manager.check_cached_predictions(symbol, days)
cache_manager.save_predictions(symbol, result)
cache_manager.get_cache_stats()
```

#### `AccuracyTracker` - Accuracy Tracking
```python
tracker = AccuracyTracker()
tracker.validate_predictions()
tracker.analyze_accuracy(symbol)
tracker.get_accuracy_stats()
```

#### `ConsoleManager` - Rich Output
```python
console = ConsoleManager(verbose=True)
console.print_prediction_results(result)
console.print_accuracy_summary(stats)
console.print_system_info()
```

### Utility Functions

#### Quick Access Functions
```python
quick_predict(symbol, days=5, use_cache=True)
analyze_accuracy(symbol=None)
get_version_info()
check_gpu_support()
```

#### Backward Compatibility
```python
predict_stock(symbol, days=5, verbose=False)
analyze_stock(symbol, verbose=False)
```

### CLI Commands

#### Basic Usage
```bash
ara SYMBOL [--days N] [--verbose] [--no-cache]
ara --accuracy [SYMBOL]
ara --validate
ara --system-info
ara --version
```

#### Advanced Options
```bash
ara AAPL --days 10 --verbose    # Detailed prediction
ara --accuracy                  # Overall accuracy
ara TSLA --no-cache            # Force fresh prediction
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **PyPI Package**: https://pypi.org/project/meridianalgo/
- **GitHub Repository**: https://github.com/MeridianAlgo/Ara
- **Documentation**: https://github.com/MeridianAlgo/Ara/blob/main/README.md
- **Issues**: https://github.com/MeridianAlgo/Ara/issues

## 📞 Support

For support, please open an issue on GitHub or contact us at support@meridianalgo.com.