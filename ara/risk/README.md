# Risk Management Module

Comprehensive risk management and portfolio analysis tools for the ARA AI prediction system.

## Overview

The risk management module provides institutional-grade risk metrics and portfolio analysis capabilities, including:

- **Value at Risk (VaR)**: Maximum expected loss at specified confidence levels
- **Conditional Value at Risk (CVaR)**: Expected loss beyond VaR threshold
- **Correlation Analysis**: Multi-asset correlation matrices
- **Risk Decomposition**: Component-level risk contribution analysis
- **Portfolio Metrics**: Sharpe, Sortino, Calmar ratios, and more
- **Drawdown Analysis**: Maximum drawdown and recovery time
- **Tracking Metrics**: Tracking error and information ratio

## Components

### RiskCalculator

The `RiskCalculator` class provides methods for calculating various risk metrics:

```python
from ara.risk import RiskCalculator
import numpy as np

# Initialize calculator
calculator = RiskCalculator()

# Generate sample returns
returns = np.random.normal(0.001, 0.02, 1000)

# Calculate VaR at 95% confidence
var_95 = calculator.calculate_var(returns, confidence_level=0.95)
print(f"VaR (95%): {var_95:.4f}")

# Calculate CVaR at 95% confidence
cvar_95 = calculator.calculate_cvar(returns, confidence_level=0.95)
print(f"CVaR (95%): {cvar_95:.4f}")

# Calculate VaR at 99% confidence
var_99 = calculator.calculate_var(returns, confidence_level=0.99)
print(f"VaR (99%): {var_99:.4f}")
```

### Portfolio Risk Analysis

```python
from ara.risk import RiskCalculator

# Multi-asset returns
returns_dict = {
    'AAPL': np.random.normal(0.001, 0.02, 252),
    'MSFT': np.random.normal(0.0008, 0.018, 252),
    'GOOGL': np.random.normal(0.0012, 0.022, 252)
}

# Portfolio weights
weights = {
    'AAPL': 0.4,
    'MSFT': 0.35,
    'GOOGL': 0.25
}

# Calculate correlation matrix
corr_matrix = calculator.calculate_correlation_matrix(returns_dict)
print("Correlation Matrix:")
print(corr_matrix)

# Calculate risk decomposition
risk_decomp = calculator.calculate_risk_decomposition(returns_dict, weights)
print(f"\nPortfolio Volatility: {risk_decomp['portfolio_volatility']:.4f}")
for asset, metrics in risk_decomp['assets'].items():
    print(f"{asset}: {metrics['percent_contribution']:.2f}% contribution to risk")

# Calculate portfolio VaR and CVaR
portfolio_var = calculator.calculate_portfolio_var(returns_dict, weights, 0.95)
portfolio_cvar = calculator.calculate_portfolio_cvar(returns_dict, weights, 0.95)
print(f"\nPortfolio VaR (95%): {portfolio_var:.4f}")
print(f"Portfolio CVaR (95%): {portfolio_cvar:.4f}")
```

### PortfolioMetrics

The `PortfolioMetrics` class provides comprehensive portfolio performance metrics:

```python
from ara.risk import PortfolioMetrics
import numpy as np

# Initialize with risk-free rate (2% annual)
metrics = PortfolioMetrics(risk_free_rate=0.02)

# Generate sample portfolio returns
portfolio_returns = np.random.normal(0.001, 0.02, 252)
benchmark_returns = np.random.normal(0.0008, 0.015, 252)

# Calculate individual metrics
volatility = metrics.calculate_portfolio_volatility(portfolio_returns)
sharpe = metrics.calculate_sharpe_ratio(portfolio_returns)
sortino = metrics.calculate_sortino_ratio(portfolio_returns)
calmar = metrics.calculate_calmar_ratio(portfolio_returns)
max_dd = metrics.calculate_maximum_drawdown(portfolio_returns)

print(f"Volatility: {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Sortino Ratio: {sortino:.4f}")
print(f"Calmar Ratio: {calmar:.4f}")
print(f"Max Drawdown: {max_dd:.4f}")

# Calculate benchmark-relative metrics
beta = metrics.calculate_beta(portfolio_returns, benchmark_returns)
tracking_error = metrics.calculate_tracking_error(portfolio_returns, benchmark_returns)
info_ratio = metrics.calculate_information_ratio(portfolio_returns, benchmark_returns)

print(f"\nBeta: {beta:.4f}")
print(f"Tracking Error: {tracking_error:.4f}")
print(f"Information Ratio: {info_ratio:.4f}")

# Calculate all metrics at once
all_metrics = metrics.calculate_all_metrics(portfolio_returns, benchmark_returns)
print("\nAll Metrics:")
for metric_name, value in all_metrics.items():
    if value is not None:
        print(f"{metric_name}: {value:.4f}")
```

### Drawdown Analysis

```python
from ara.risk import PortfolioMetrics
import numpy as np

metrics = PortfolioMetrics()
returns = np.random.normal(0.001, 0.02, 252)

# Calculate drawdown series
dd_series = metrics.calculate_drawdown_series(returns)

# Calculate recovery time
recovery_info = metrics.calculate_recovery_time(returns)
print(f"Maximum Drawdown: {recovery_info['max_drawdown']:.4f}")
print(f"Drawdown Start: Day {recovery_info['drawdown_start']}")
print(f"Drawdown End: Day {recovery_info['drawdown_end']}")
if recovery_info['recovery_time'] is not None:
    print(f"Recovery Time: {recovery_info['recovery_time']} days")
else:
    print("Portfolio has not yet recovered")
```

### Downside Risk Metrics

```python
from ara.risk import PortfolioMetrics

metrics = PortfolioMetrics()
returns = np.random.normal(0.001, 0.02, 252)

# Calculate downside deviation (only negative returns)
downside_dev = metrics.calculate_downside_deviation(returns, target_return=0.0)
print(f"Downside Deviation: {downside_dev:.4f}")

# Calculate Sortino ratio (uses downside deviation instead of total volatility)
sortino = metrics.calculate_sortino_ratio(returns)
print(f"Sortino Ratio: {sortino:.4f}")
```

## VaR Calculation Methods

The `RiskCalculator` supports three methods for calculating VaR and CVaR:

### 1. Historical Method (Default)

Uses empirical distribution of historical returns:

```python
var = calculator.calculate_var(returns, confidence_level=0.95, method='historical')
```

**Pros**: No distributional assumptions, captures actual tail behavior
**Cons**: Limited by historical data, may not predict future extreme events

### 2. Parametric Method

Assumes returns follow a normal distribution:

```python
var = calculator.calculate_var(returns, confidence_level=0.95, method='parametric')
```

**Pros**: Smooth estimates, works with limited data
**Cons**: May underestimate tail risk if returns are not normally distributed

### 3. Monte Carlo Method

Simulates returns based on historical mean and volatility:

```python
var = calculator.calculate_var(returns, confidence_level=0.95, method='monte_carlo')
```

**Pros**: Flexible, can incorporate complex scenarios
**Cons**: Computationally intensive, still assumes distribution

## Key Metrics Explained

### Value at Risk (VaR)

Maximum expected loss over a time period at a given confidence level.

- **VaR 95%**: There is a 5% chance of losing more than this amount
- **VaR 99%**: There is a 1% chance of losing more than this amount

### Conditional Value at Risk (CVaR)

Expected loss given that the loss exceeds VaR. Also known as Expected Shortfall.

- More conservative than VaR
- Captures tail risk better
- Preferred by regulators

### Sharpe Ratio

Risk-adjusted return metric:
```
Sharpe = (Return - Risk-Free Rate) / Volatility
```

- Higher is better
- Typical values: 0.5-1.0 (good), 1.0-2.0 (very good), >2.0 (excellent)

### Sortino Ratio

Similar to Sharpe but only penalizes downside volatility:
```
Sortino = (Return - Target) / Downside Deviation
```

- Better for asymmetric return distributions
- Higher values indicate better risk-adjusted returns

### Calmar Ratio

Return relative to maximum drawdown:
```
Calmar = Annualized Return / Maximum Drawdown
```

- Focuses on worst-case scenario
- Higher is better

### Maximum Drawdown

Largest peak-to-trough decline:
```
Max DD = (Trough Value - Peak Value) / Peak Value
```

- Measures worst historical loss
- Important for risk management

### Beta

Sensitivity to market movements:
```
Beta = Covariance(Portfolio, Benchmark) / Variance(Benchmark)
```

- Beta = 1: Moves with market
- Beta > 1: More volatile than market
- Beta < 1: Less volatile than market

### Tracking Error

Volatility of excess returns relative to benchmark:
```
TE = StdDev(Portfolio Returns - Benchmark Returns)
```

- Measures how closely portfolio follows benchmark
- Lower is better for index funds

### Information Ratio

Excess return per unit of tracking error:
```
IR = (Portfolio Return - Benchmark Return) / Tracking Error
```

- Measures manager skill
- Higher is better

## Requirements Satisfied

This implementation satisfies the following requirements:

- **Requirement 7.1**: Value at Risk (VaR) calculation at 95% and 99% confidence
- **Requirement 7.4**: Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- **Requirement 7.4**: Portfolio volatility and beta
- **Requirement 7.4**: Maximum drawdown and recovery time
- **Requirement 7.4**: Tracking error and information ratio
- **Requirement 7.4**: Downside deviation

## Integration with ARA AI

The risk management module integrates seamlessly with other ARA AI components:

```python
from ara.backtesting import BacktestEngine
from ara.risk import RiskCalculator, PortfolioMetrics

# Run backtest
backtest_result = backtest_engine.run_backtest(
    symbol='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# Calculate risk metrics on backtest results
calculator = RiskCalculator()
metrics = PortfolioMetrics()

returns = backtest_result.equity_curve.pct_change().dropna()

# Risk metrics
var_95 = calculator.calculate_var(returns, 0.95)
cvar_95 = calculator.calculate_cvar(returns, 0.95)

# Performance metrics
all_metrics = metrics.calculate_all_metrics(returns)

print(f"Backtest VaR (95%): {var_95:.4f}")
print(f"Backtest CVaR (95%): {cvar_95:.4f}")
print(f"Sharpe Ratio: {all_metrics['sharpe_ratio']:.4f}")
print(f"Max Drawdown: {all_metrics['max_drawdown']:.4f}")
```

## Best Practices

1. **Use Multiple Confidence Levels**: Calculate both 95% and 99% VaR for comprehensive risk assessment
2. **Prefer CVaR over VaR**: CVaR provides better tail risk measurement
3. **Consider Multiple Methods**: Compare historical, parametric, and Monte Carlo VaR
4. **Monitor Drawdowns**: Track recovery time to understand portfolio resilience
5. **Use Sortino for Asymmetric Returns**: Better than Sharpe for strategies with skewed returns
6. **Benchmark Comparison**: Always calculate beta and tracking error relative to appropriate benchmark
7. **Regular Recalculation**: Update risk metrics as new data becomes available

## Performance Considerations

- All calculations are vectorized using NumPy for optimal performance
- Correlation matrices use pandas built-in optimized methods
- Monte Carlo simulations use 10,000 iterations by default
- Large portfolios (100+ assets) calculate in under 1 second

## Future Enhancements

Planned features for future releases:

- Stress testing and scenario analysis
- Risk budgeting and allocation
- Marginal VaR and incremental VaR
- Extreme value theory (EVT) for tail risk
- Time-varying volatility models (GARCH)
- Copula-based correlation modeling
