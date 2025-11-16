"""
Risk Management Demo

Demonstrates the comprehensive risk management capabilities of ARA AI.
"""

import numpy as np
import pandas as pd
from ara.risk import RiskCalculator, PortfolioMetrics


def demo_var_cvar():
    """Demonstrate VaR and CVaR calculations."""
    print("=" * 80)
    print("VaR and CVaR Calculation Demo")
    print("=" * 80)
    
    # Generate sample returns (daily returns for 1 year)
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    
    calculator = RiskCalculator()
    
    # Calculate VaR at different confidence levels
    var_95 = calculator.calculate_var(returns, confidence_level=0.95, method='historical')
    var_99 = calculator.calculate_var(returns, confidence_level=0.99, method='historical')
    
    # Calculate CVaR at different confidence levels
    cvar_95 = calculator.calculate_cvar(returns, confidence_level=0.95, method='historical')
    cvar_99 = calculator.calculate_cvar(returns, confidence_level=0.99, method='historical')
    
    print(f"\nHistorical Method:")
    print(f"  VaR (95%):  {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"  CVaR (95%): {cvar_95:.4f} ({cvar_95*100:.2f}%)")
    print(f"  VaR (99%):  {var_99:.4f} ({var_99*100:.2f}%)")
    print(f"  CVaR (99%): {cvar_99:.4f} ({cvar_99*100:.2f}%)")
    
    # Compare with parametric method
    var_95_param = calculator.calculate_var(returns, confidence_level=0.95, method='parametric')
    cvar_95_param = calculator.calculate_cvar(returns, confidence_level=0.95, method='parametric')
    
    print(f"\nParametric Method:")
    print(f"  VaR (95%):  {var_95_param:.4f} ({var_95_param*100:.2f}%)")
    print(f"  CVaR (95%): {cvar_95_param:.4f} ({cvar_95_param*100:.2f}%)")
    
    # Compare with Monte Carlo method
    var_95_mc = calculator.calculate_var(returns, confidence_level=0.95, method='monte_carlo')
    cvar_95_mc = calculator.calculate_cvar(returns, confidence_level=0.95, method='monte_carlo')
    
    print(f"\nMonte Carlo Method:")
    print(f"  VaR (95%):  {var_95_mc:.4f} ({var_95_mc*100:.2f}%)")
    print(f"  CVaR (95%): {cvar_95_mc:.4f} ({cvar_95_mc*100:.2f}%)")
    
    print("\nInterpretation:")
    print(f"  There is a 5% chance of losing more than {var_95*100:.2f}% in a day")
    print(f"  If losses exceed VaR, the expected loss is {cvar_95*100:.2f}%")


def demo_correlation_analysis():
    """Demonstrate correlation matrix calculation."""
    print("\n" + "=" * 80)
    print("Correlation Analysis Demo")
    print("=" * 80)
    
    # Generate correlated returns for multiple assets
    np.random.seed(42)
    n_periods = 252
    
    # Create correlated returns
    base_returns = np.random.normal(0.001, 0.015, n_periods)
    
    returns_dict = {
        'AAPL': base_returns + np.random.normal(0, 0.005, n_periods),
        'MSFT': base_returns + np.random.normal(0, 0.005, n_periods),
        'GOOGL': base_returns + np.random.normal(0, 0.006, n_periods),
        'BTC': np.random.normal(0.002, 0.04, n_periods),  # Less correlated
        'TSLA': base_returns * 1.5 + np.random.normal(0, 0.01, n_periods)  # Higher beta
    }
    
    calculator = RiskCalculator()
    
    # Calculate correlation matrix
    corr_matrix = calculator.calculate_correlation_matrix(returns_dict)
    
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Identify highly correlated pairs
    print("\nHighly Correlated Pairs (correlation > 0.7):")
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            corr = corr_matrix.iloc[i, j]
            if corr > 0.7:
                asset1 = corr_matrix.index[i]
                asset2 = corr_matrix.columns[j]
                print(f"  {asset1} - {asset2}: {corr:.3f}")


def demo_risk_decomposition():
    """Demonstrate risk decomposition analysis."""
    print("\n" + "=" * 80)
    print("Risk Decomposition Demo")
    print("=" * 80)
    
    # Generate sample returns
    np.random.seed(42)
    n_periods = 252
    
    returns_dict = {
        'AAPL': np.random.normal(0.001, 0.02, n_periods),
        'MSFT': np.random.normal(0.0008, 0.018, n_periods),
        'GOOGL': np.random.normal(0.0012, 0.022, n_periods),
        'AMZN': np.random.normal(0.0015, 0.025, n_periods)
    }
    
    # Portfolio weights
    weights = {
        'AAPL': 0.30,
        'MSFT': 0.25,
        'GOOGL': 0.25,
        'AMZN': 0.20
    }
    
    calculator = RiskCalculator()
    
    # Calculate risk decomposition
    risk_decomp = calculator.calculate_risk_decomposition(returns_dict, weights)
    
    print(f"\nPortfolio Volatility: {risk_decomp['portfolio_volatility']:.4f}")
    print(f"                      ({risk_decomp['portfolio_volatility']*100:.2f}%)")
    
    print("\nRisk Contribution by Asset:")
    print(f"{'Asset':<10} {'Weight':<10} {'Volatility':<12} {'Marginal':<12} {'Component':<12} {'% Contrib':<12}")
    print("-" * 80)
    
    for asset, metrics in risk_decomp['assets'].items():
        print(f"{asset:<10} "
              f"{metrics['weight']:<10.2%} "
              f"{metrics['volatility']:<12.4f} "
              f"{metrics['marginal_risk']:<12.4f} "
              f"{metrics['component_risk']:<12.4f} "
              f"{metrics['percent_contribution']:<12.2f}%")
    
    # Verify contributions sum to 100%
    total_contribution = sum(m['percent_contribution'] for m in risk_decomp['assets'].values())
    print(f"\nTotal Contribution: {total_contribution:.2f}%")


def demo_portfolio_metrics():
    """Demonstrate portfolio performance metrics."""
    print("\n" + "=" * 80)
    print("Portfolio Metrics Demo")
    print("=" * 80)
    
    # Generate sample portfolio returns
    np.random.seed(42)
    portfolio_returns = np.random.normal(0.001, 0.02, 252)
    benchmark_returns = np.random.normal(0.0008, 0.015, 252)
    
    # Initialize with 2% risk-free rate
    metrics = PortfolioMetrics(risk_free_rate=0.02)
    
    # Calculate all metrics
    all_metrics = metrics.calculate_all_metrics(portfolio_returns, benchmark_returns)
    
    print("\nRisk Metrics:")
    print(f"  Volatility:          {all_metrics['volatility']:.4f} ({all_metrics['volatility']*100:.2f}%)")
    print(f"  Downside Deviation:  {all_metrics['downside_deviation']:.4f}")
    print(f"  Maximum Drawdown:    {all_metrics['max_drawdown']:.4f} ({all_metrics['max_drawdown']*100:.2f}%)")
    if all_metrics['recovery_time'] is not None:
        print(f"  Recovery Time:       {all_metrics['recovery_time']} days")
    else:
        print(f"  Recovery Time:       Not yet recovered")
    
    print("\nRisk-Adjusted Returns:")
    print(f"  Sharpe Ratio:        {all_metrics['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio:       {all_metrics['sortino_ratio']:.4f}")
    print(f"  Calmar Ratio:        {all_metrics['calmar_ratio']:.4f}")
    
    print("\nBenchmark-Relative Metrics:")
    print(f"  Beta:                {all_metrics['beta']:.4f}")
    print(f"  Tracking Error:      {all_metrics['tracking_error']:.4f}")
    print(f"  Information Ratio:   {all_metrics['information_ratio']:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    if all_metrics['sharpe_ratio'] > 1.0:
        print("  ✓ Good risk-adjusted returns (Sharpe > 1.0)")
    else:
        print("  ✗ Below-average risk-adjusted returns (Sharpe < 1.0)")
    
    if all_metrics['beta'] > 1.0:
        print(f"  Portfolio is {all_metrics['beta']:.2f}x more volatile than benchmark")
    else:
        print(f"  Portfolio is {all_metrics['beta']:.2f}x less volatile than benchmark")


def demo_drawdown_analysis():
    """Demonstrate drawdown analysis."""
    print("\n" + "=" * 80)
    print("Drawdown Analysis Demo")
    print("=" * 80)
    
    # Generate returns with a significant drawdown
    np.random.seed(42)
    returns = np.concatenate([
        np.random.normal(0.002, 0.015, 100),   # Bull market
        np.random.normal(-0.003, 0.025, 50),   # Bear market (drawdown)
        np.random.normal(0.0015, 0.018, 102)   # Recovery
    ])
    
    metrics = PortfolioMetrics()
    
    # Calculate maximum drawdown
    max_dd = metrics.calculate_maximum_drawdown(returns)
    print(f"\nMaximum Drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
    
    # Calculate recovery information
    recovery_info = metrics.calculate_recovery_time(returns)
    
    print(f"\nDrawdown Details:")
    print(f"  Peak:           Day {recovery_info['drawdown_start']}")
    print(f"  Trough:         Day {recovery_info['drawdown_end']}")
    print(f"  Drawdown:       {recovery_info['max_drawdown']:.4f} ({recovery_info['max_drawdown']*100:.2f}%)")
    
    if recovery_info['recovery_end'] is not None:
        print(f"  Recovery:       Day {recovery_info['recovery_end']}")
        print(f"  Recovery Time:  {recovery_info['recovery_time']} days")
        print(f"  Total Duration: {recovery_info['recovery_end'] - recovery_info['drawdown_start']} days")
    else:
        print(f"  Recovery:       Not yet recovered")
    
    # Calculate drawdown series
    dd_series = metrics.calculate_drawdown_series(returns)
    
    print(f"\nDrawdown Statistics:")
    print(f"  Average Drawdown:    {abs(np.mean(dd_series)):.4f}")
    print(f"  Median Drawdown:     {abs(np.median(dd_series)):.4f}")
    print(f"  95th Percentile DD:  {abs(np.percentile(dd_series, 5)):.4f}")


def demo_portfolio_var_cvar():
    """Demonstrate portfolio-level VaR and CVaR."""
    print("\n" + "=" * 80)
    print("Portfolio VaR and CVaR Demo")
    print("=" * 80)
    
    # Generate correlated returns
    np.random.seed(42)
    n_periods = 252
    
    returns_dict = {
        'AAPL': np.random.normal(0.001, 0.02, n_periods),
        'MSFT': np.random.normal(0.0008, 0.018, n_periods),
        'GOOGL': np.random.normal(0.0012, 0.022, n_periods)
    }
    
    weights = {
        'AAPL': 0.40,
        'MSFT': 0.35,
        'GOOGL': 0.25
    }
    
    calculator = RiskCalculator()
    
    # Calculate individual asset VaR
    print("\nIndividual Asset VaR (95%):")
    for asset, returns in returns_dict.items():
        var = calculator.calculate_var(returns, 0.95)
        print(f"  {asset}: {var:.4f} ({var*100:.2f}%)")
    
    # Calculate portfolio VaR
    portfolio_var = calculator.calculate_portfolio_var(returns_dict, weights, 0.95)
    portfolio_cvar = calculator.calculate_portfolio_cvar(returns_dict, weights, 0.95)
    
    print(f"\nPortfolio VaR (95%):  {portfolio_var:.4f} ({portfolio_var*100:.2f}%)")
    print(f"Portfolio CVaR (95%): {portfolio_cvar:.4f} ({portfolio_cvar*100:.2f}%)")
    
    # Calculate weighted average of individual VaRs
    weighted_var = sum(
        weights[asset] * calculator.calculate_var(returns, 0.95)
        for asset, returns in returns_dict.items()
    )
    
    print(f"\nWeighted Avg VaR:     {weighted_var:.4f} ({weighted_var*100:.2f}%)")
    
    diversification_benefit = weighted_var - portfolio_var
    print(f"\nDiversification Benefit: {diversification_benefit:.4f} ({diversification_benefit*100:.2f}%)")
    print(f"Benefit Percentage:      {(diversification_benefit/weighted_var)*100:.2f}%")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ARA AI Risk Management Demo" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    
    demo_var_cvar()
    demo_correlation_analysis()
    demo_risk_decomposition()
    demo_portfolio_metrics()
    demo_drawdown_analysis()
    demo_portfolio_var_cvar()
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nFor more information, see ara/risk/README.md")


if __name__ == "__main__":
    main()
