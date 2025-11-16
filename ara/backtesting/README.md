# Backtesting Module

Comprehensive backtesting engine for validating prediction models with institutional-grade features.

## Features

### 1. Walk-Forward Validation
- Prevents look-ahead bias by training on past data only
- Configurable training and testing windows
- Rolling window approach for realistic performance estimation

### 2. Out-of-Sample Testing
- 20% holdout dataset for unbiased performance evaluation
- Ensures model generalizes to unseen data

### 3. Cross-Validation
- Time-series aware cross-validation
- Expanding window approach respects temporal ordering
- Multiple folds for robust performance estimation

### 4. Monte Carlo Simulation
- 1000+ simulations for robustness testing
- Confidence interval estimation
- Probability of positive returns

### 5. Performance Metrics

#### Classification Metrics
- Accuracy, Precision, Recall, F1 Score
- Directional accuracy (up/down predictions)

#### Error Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

#### Financial Metrics
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown

#### Trading Metrics
- Win Rate
- Profit Factor
- Total/Winning/Losing Trades

### 6. Transaction Cost Modeling
- Slippage modeling (default: 5 bps)
- Commission costs (default: 10 bps)
- Realistic return estimation

### 7. Automated Model Validation
- Daily accuracy monitoring
- Performance degradation detection
- Automatic retraining triggers
- A/B testing between models

### 8. Comprehensive Reporting
- Equity curve visualization
- Monthly/yearly return tables
- Drawdown analysis
- Regime-specific performance
- Model comparison charts
- PDF report generation

## Quick Start

```python
from ara.backtesting import BacktestEngine
from ara.backtesting.engine import BacktestConfig
import pandas as pd

# Load your data
data = pd.read_csv('historical_data.csv')
data.set_index('date', inplace=True)

# Configure backtest
config = BacktestConfig(
    train_window_days=252,  # 1 year training
    test_window_days=30,    # 1 month testing
    holdout_ratio=0.20,     # 20% holdout
    n_folds=5,              # 5-fold CV
    slippage_bps=5.0,       # 5 bps slippage
    commission_bps=10.0     # 10 bps commission
)

# Initialize engine
engine = BacktestEngine(config=config)

# Define your prediction model
def my_model(X):
    # Your model logic here
    return predictions

# Run backtest
result = engine.run_backtest(
    symbol='AAPL',
    data=data,
    model_predict_fn=my_model,
    feature_columns=['feature1', 'feature2', 'feature3'],
    target_column='target',
    price_column='close'
)

# View results
print(f"Directional Accuracy: {result.metrics.directional_accuracy:.2%}")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.2%}")

# Save results and generate reports
engine.save_results(result, generate_plots=True)
```

## Performance Metrics Example

```python
from ara.backtesting import PerformanceMetrics
import numpy as np

# Initialize metrics calculator
metrics_calc = PerformanceMetrics(risk_free_rate=0.02)

# Your predictions and actuals
predictions = np.array([0.01, 0.02, -0.01, 0.015, 0.01])
actuals = np.array([0.015, 0.025, -0.005, 0.02, 0.012])
prices = np.array([100, 101, 103, 102, 104, 105])

# Calculate all metrics
metrics = metrics_calc.calculate_all_metrics(
    predictions, actuals, prices=prices
)

# Access metrics
print(f"Accuracy: {metrics.accuracy:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Win Rate: {metrics.win_rate:.2%}")
```

## Model Validation Example

```python
from ara.backtesting import ModelValidator

# Initialize validator
validator = ModelValidator(
    accuracy_threshold=0.75,
    degradation_threshold=0.10
)

# Validate model
result = validator.validate_model(
    model_name='my_model',
    predictions=predictions,
    actuals=actuals,
    dates=dates
)

# Check if retraining needed
if result.needs_retraining:
    print("Model needs retraining!")
    print("Recommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")

# Monitor daily accuracy
daily_result = validator.monitor_daily_accuracy(
    model_name='my_model',
    predictions=today_predictions,
    actuals=today_actuals,
    date=datetime.now()
)

print(f"Daily Accuracy: {daily_result['daily_accuracy']:.2%}")
print(f"7-Day Rolling Avg: {daily_result['rolling_avg_7d']:.2%}")
```

## A/B Testing Example

```python
from ara.backtesting import ModelValidator

validator = ModelValidator()

# Compare two models
result = validator.ab_test_models(
    model_a_name='model_v1',
    model_a_predictions=predictions_v1,
    model_b_name='model_v2',
    model_b_predictions=predictions_v2,
    actuals=actuals,
    dates=dates
)

print(f"Winner: {result.winner}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Accuracy Improvement: {result.improvement['accuracy']:.2%}")
```

## Configuration Options

### BacktestConfig

```python
BacktestConfig(
    # Walk-forward validation
    train_window_days=252,    # Training window size
    test_window_days=30,      # Testing window size
    step_size_days=30,        # Step size for rolling window
    
    # Holdout testing
    holdout_ratio=0.20,       # Holdout percentage (20%)
    
    # Cross-validation
    n_folds=5,                # Number of CV folds
    
    # Monte Carlo
    n_simulations=1000,       # Number of simulations
    confidence_level=0.95,    # Confidence level (95%)
    
    # Transaction costs
    slippage_bps=5.0,         # Slippage in basis points
    commission_bps=10.0,      # Commission in basis points
    
    # Risk-free rate
    risk_free_rate=0.02       # Annual risk-free rate (2%)
)
```

## Output Files

The backtesting engine generates several output files:

1. **JSON Report** (`backtest_SYMBOL_TIMESTAMP.json`)
   - Complete metrics and analysis
   - Walk-forward results
   - Holdout and CV results
   - Monte Carlo results

2. **Equity Curve** (`equity_curve_SYMBOL.html`)
   - Interactive Plotly chart
   - Equity curve with drawdowns

3. **Monthly Returns** (`monthly_returns_SYMBOL.html`)
   - Heatmap of monthly returns
   - Year-over-year comparison

4. **Validation Results** (`validation_MODEL_TIMESTAMP.json`)
   - Model validation metrics
   - Recommendations

## Best Practices

1. **Use Walk-Forward Validation**: Prevents look-ahead bias and provides realistic performance estimates.

2. **Always Test on Holdout**: Ensure your model generalizes to completely unseen data.

3. **Include Transaction Costs**: Real-world trading has costs. Model them accurately.

4. **Monitor Performance**: Use automated validation to detect degradation early.

5. **Compare Multiple Models**: Use A/B testing to select the best model.

6. **Analyze by Regime**: Different models perform better in different market conditions.

7. **Run Monte Carlo**: Understand the range of possible outcomes and confidence intervals.

## Requirements

- numpy >= 1.24.0
- pandas >= 2.0.0
- plotly >= 5.15.0 (optional, for visualizations)
- matplotlib >= 3.7.0 (optional, for visualizations)

## Integration with ARA AI

The backtesting module integrates seamlessly with other ARA AI components:

```python
from ara.models import AdaptiveEnsemble
from ara.features import FeatureCalculator
from ara.backtesting import BacktestEngine

# Initialize components
ensemble = AdaptiveEnsemble()
feature_calc = FeatureCalculator()
backtest_engine = BacktestEngine()

# Prepare data with features
data = feature_calc.calculate_all_features(raw_data)

# Define prediction function
def predict_fn(X):
    return ensemble.predict(X)

# Run backtest
result = backtest_engine.run_backtest(
    symbol='AAPL',
    data=data,
    model_predict_fn=predict_fn,
    feature_columns=feature_calc.get_feature_names(),
    target_column='target'
)
```

## Advanced Usage

### Custom Metrics

```python
from ara.backtesting.metrics import PerformanceMetrics

class CustomMetrics(PerformanceMetrics):
    def calculate_custom_metric(self, predictions, actuals):
        # Your custom metric logic
        return custom_value

metrics_calc = CustomMetrics()
```

### Custom Validation Logic

```python
from ara.backtesting.validator import ModelValidator

class CustomValidator(ModelValidator):
    def custom_validation_check(self, metrics):
        # Your custom validation logic
        return needs_action

validator = CustomValidator()
```

## Troubleshooting

### Issue: Backtest takes too long

**Solution**: Reduce the number of walk-forward windows or Monte Carlo simulations:
```python
config = BacktestConfig(
    step_size_days=60,      # Larger steps = fewer windows
    n_simulations=100       # Fewer simulations
)
```

### Issue: Memory errors with large datasets

**Solution**: Process data in chunks or reduce the training window:
```python
config = BacktestConfig(
    train_window_days=120   # Smaller training window
)
```

### Issue: Plots not generating

**Solution**: Install visualization libraries:
```bash
pip install plotly matplotlib
```

## References

- Walk-Forward Analysis: [Pardo, R. (2008). The Evaluation and Optimization of Trading Strategies](https://www.wiley.com/en-us/The+Evaluation+and+Optimization+of+Trading+Strategies%2C+2nd+Edition-p-9780470128015)
- Performance Metrics: [Bailey, D. H., & López de Prado, M. (2014). The Sharpe Ratio Efficient Frontier](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643)
- Cross-Validation: [Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation](https://www.sciencedirect.com/science/article/abs/pii/S0020025511006773)

## License

Part of the ARA AI prediction system. See main LICENSE file for details.
