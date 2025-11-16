"""
Portfolio Optimization Demo

This script demonstrates the portfolio optimization capabilities including:
- Modern Portfolio Theory (MPT) optimization
- Black-Litterman model
- Risk Parity optimization
- Kelly Criterion position sizing
- Mean-CVaR optimization
- Efficient frontier calculation
- Portfolio constraints and rebalancing
- Portfolio analysis and visualization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ara.risk import (
    PortfolioOptimizer,
    PortfolioConstraints,
    TransactionCostModel,
    PortfolioRebalancer,
    RebalanceFrequency,
    PortfolioAnalyzer,
    RiskCalculator
)


def generate_sample_returns(n_assets=5, n_periods=252, seed=42):
    """Generate sample return data for demonstration."""
    np.random.seed(seed)
    
    # Asset names
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'][:n_assets]
    
    # Generate correlated returns
    mean_returns = np.random.uniform(0.0005, 0.0015, n_assets)
    volatilities = np.random.uniform(0.015, 0.030, n_assets)
    
    # Create correlation matrix
    correlation = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    
    # Generate returns
    returns_dict = {}
    for i, asset in enumerate(assets):
        returns = np.random.normal(mean_returns[i], volatilities[i], n_periods)
        returns_dict[asset] = returns
    
    return returns_dict, assets


def demo_mpt_optimization():
    """Demonstrate Modern Portfolio Theory optimization."""
    print("\n" + "="*80)
    print("MODERN PORTFOLIO THEORY (MPT) OPTIMIZATION")
    print("="*80)
    
    # Generate sample data
    returns_dict, assets = generate_sample_returns(n_assets=4)
    
    # Create optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # 1. Maximize Sharpe Ratio
    print("\n1. Maximum Sharpe Ratio Portfolio:")
    result = optimizer.optimize_mpt(returns_dict, objective='max_sharpe')
    print(f"   Expected Return: {result['expected_return']:.2%}")
    print(f"   Volatility: {result['volatility']:.2%}")
    print(f"   Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print("   Weights:")
    for asset, weight in result['weights'].items():
        print(f"     {asset}: {weight:.2%}")
    
    # 2. Minimum Risk Portfolio
    print("\n2. Minimum Risk Portfolio:")
    result = optimizer.optimize_mpt(returns_dict, objective='min_risk')
    print(f"   Expected Return: {result['expected_return']:.2%}")
    print(f"   Volatility: {result['volatility']:.2%}")
    print(f"   Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print("   Weights:")
    for asset, weight in result['weights'].items():
        print(f"     {asset}: {weight:.2%}")
    
    # 3. With constraints
    print("\n3. Constrained Optimization (max 30% per asset):")
    constraints = {'min_weight': 0.05, 'max_weight': 0.30}
    result = optimizer.optimize_mpt(returns_dict, objective='max_sharpe', constraints=constraints)
    print(f"   Expected Return: {result['expected_return']:.2%}")
    print(f"   Volatility: {result['volatility']:.2%}")
    print(f"   Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print("   Weights:")
    for asset, weight in result['weights'].items():
        print(f"     {asset}: {weight:.2%}")


def demo_black_litterman():
    """Demonstrate Black-Litterman optimization."""
    print("\n" + "="*80)
    print("BLACK-LITTERMAN MODEL")
    print("="*80)
    
    # Generate sample data
    returns_dict, assets = generate_sample_returns(n_assets=3)
    
    # Market capitalizations (in billions)
    market_caps = {
        'AAPL': 2500,
        'MSFT': 2300,
        'GOOGL': 1500
    }
    
    # Investor views (expected annual returns)
    views = {
        'AAPL': 0.15,  # Expect 15% return
        'MSFT': 0.12,  # Expect 12% return
        'GOOGL': 0.18  # Expect 18% return
    }
    
    # View confidence (0-1, higher = more confident)
    view_confidence = {
        'AAPL': 0.7,
        'MSFT': 0.6,
        'GOOGL': 0.8
    }
    
    # Create optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    print("\nOptimizing with Black-Litterman model...")
    result = optimizer.optimize_black_litterman(
        returns_dict,
        market_caps,
        views,
        view_confidence
    )
    
    print(f"\nExpected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    
    print("\nOptimal Weights:")
    for asset, weight in result['weights'].items():
        print(f"  {asset}: {weight:.2%}")
    
    print("\nBlack-Litterman Returns (vs Implied Returns):")
    for asset in assets:
        bl_ret = result['bl_returns'][asset]
        impl_ret = result['implied_returns'][asset]
        print(f"  {asset}: BL={bl_ret:.2%}, Implied={impl_ret:.2%}")


def demo_risk_parity():
    """Demonstrate Risk Parity optimization."""
    print("\n" + "="*80)
    print("RISK PARITY OPTIMIZATION")
    print("="*80)
    
    # Generate sample data
    returns_dict, assets = generate_sample_returns(n_assets=4)
    
    # Create optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    print("\nOptimizing with Risk Parity approach...")
    result = optimizer.optimize_risk_parity(returns_dict)
    
    print(f"\nExpected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    
    print("\nOptimal Weights:")
    for asset, weight in result['weights'].items():
        print(f"  {asset}: {weight:.2%}")
    
    print("\nRisk Contributions (should be approximately equal):")
    for asset, contrib in result['risk_contributions'].items():
        print(f"  {asset}: {contrib:.4f}")


def demo_kelly_criterion():
    """Demonstrate Kelly Criterion position sizing."""
    print("\n" + "="*80)
    print("KELLY CRITERION POSITION SIZING")
    print("="*80)
    
    # Generate sample data
    returns_dict, assets = generate_sample_returns(n_assets=3)
    
    # Predictions (expected annual returns)
    predictions = {
        'AAPL': 0.20,  # Bullish
        'MSFT': 0.10,  # Neutral
        'GOOGL': 0.15  # Moderately bullish
    }
    
    # Create optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    print("\nCalculating Kelly weights based on predictions...")
    weights = optimizer.calculate_kelly_weights(
        returns_dict,
        predictions,
        max_position_size=0.40
    )
    
    print("\nKelly Criterion Weights:")
    for asset, weight in weights.items():
        pred = predictions.get(asset, 0)
        print(f"  {asset}: {weight:.2%} (predicted return: {pred:.2%})")


def demo_mean_cvar():
    """Demonstrate Mean-CVaR optimization."""
    print("\n" + "="*80)
    print("MEAN-CVAR OPTIMIZATION")
    print("="*80)
    
    # Generate sample data
    returns_dict, assets = generate_sample_returns(n_assets=4)
    
    # Create optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    print("\nOptimizing with Mean-CVaR approach...")
    result = optimizer.optimize_mean_cvar(returns_dict, confidence_level=0.95)
    
    print(f"\nExpected Return: {result['expected_return']:.2%}")
    print(f"Volatility: {result['volatility']:.2%}")
    print(f"VaR (95%): {result['var_95']:.2%}")
    print(f"CVaR (95%): {result['cvar_95']:.2%}")
    print(f"Return/CVaR Ratio: {result['return_cvar_ratio']:.3f}")
    
    print("\nOptimal Weights:")
    for asset, weight in result['weights'].items():
        print(f"  {asset}: {weight:.2%}")


def demo_efficient_frontier():
    """Demonstrate efficient frontier calculation."""
    print("\n" + "="*80)
    print("EFFICIENT FRONTIER")
    print("="*80)
    
    # Generate sample data
    returns_dict, assets = generate_sample_returns(n_assets=4)
    
    # Create optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    print("\nCalculating efficient frontier (30 points)...")
    frontier = optimizer.calculate_efficient_frontier(returns_dict, n_points=30)
    
    print(f"\nFrontier calculated with {len(frontier['frontier'])} points")
    
    print("\nMax Sharpe Portfolio:")
    max_sharpe = frontier['max_sharpe']
    print(f"  Expected Return: {max_sharpe['expected_return']:.2%}")
    print(f"  Volatility: {max_sharpe['volatility']:.2%}")
    print(f"  Sharpe Ratio: {max_sharpe['sharpe_ratio']:.3f}")
    print("  Weights:")
    for asset, weight in max_sharpe['weights'].items():
        print(f"    {asset}: {weight:.2%}")
    
    print("\nMin Risk Portfolio:")
    min_risk = frontier['min_risk']
    print(f"  Expected Return: {min_risk['expected_return']:.2%}")
    print(f"  Volatility: {min_risk['volatility']:.2%}")
    print(f"  Sharpe Ratio: {min_risk['sharpe_ratio']:.3f}")
    print("  Weights:")
    for asset, weight in min_risk['weights'].items():
        print(f"    {asset}: {weight:.2%}")


def demo_strategy_comparison():
    """Demonstrate strategy comparison."""
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    # Generate sample data
    returns_dict, assets = generate_sample_returns(n_assets=3)
    
    # Create optimizer
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    print("\nComparing optimization strategies...")
    comparison = optimizer.compare_strategies(returns_dict)
    
    print("\nStrategy Comparison:")
    print(comparison.to_string())


def demo_constraints_and_rebalancing():
    """Demonstrate portfolio constraints and rebalancing."""
    print("\n" + "="*80)
    print("PORTFOLIO CONSTRAINTS AND REBALANCING")
    print("="*80)
    
    # Current portfolio
    current_weights = {
        'AAPL': 0.35,
        'MSFT': 0.40,
        'GOOGL': 0.25
    }
    
    # Target portfolio
    target_weights = {
        'AAPL': 0.30,
        'MSFT': 0.35,
        'GOOGL': 0.35
    }
    
    # Current prices
    prices = {
        'AAPL': 150.0,
        'MSFT': 300.0,
        'GOOGL': 120.0
    }
    
    portfolio_value = 100000.0
    
    # 1. Validate constraints
    print("\n1. Validating Portfolio Constraints:")
    constraints = PortfolioConstraints(min_weight=0.10, max_weight=0.40)
    is_valid, violations = constraints.validate_weights(current_weights)
    print(f"   Current portfolio valid: {is_valid}")
    if violations:
        print("   Violations:")
        for violation in violations:
            print(f"     - {violation}")
    
    # 2. Calculate rebalancing trades
    print("\n2. Calculating Rebalancing Trades:")
    cost_model = TransactionCostModel(commission_rate=0.001, spread_rate=0.0005)
    rebalancer = PortfolioRebalancer(cost_model=cost_model, constraints=constraints)
    
    trades = rebalancer.calculate_rebalance_trades(
        current_weights,
        target_weights,
        portfolio_value,
        prices
    )
    
    print(f"   Number of trades: {len(trades)}")
    for trade in trades:
        print(f"\n   {trade.action} {trade.symbol}:")
        print(f"     Quantity: {trade.quantity:.2f} shares")
        print(f"     Value: ${trade.quantity * trade.current_price:,.2f}")
        print(f"     Weight change: {trade.weight_change:+.2%}")
        print(f"     Transaction cost: ${trade.transaction_cost:.2f}")
    
    # 3. Execute rebalancing
    print("\n3. Executing Rebalancing:")
    result = rebalancer.rebalance(
        current_weights,
        target_weights,
        portfolio_value,
        prices,
        datetime.now()
    )
    
    print(f"   Total transaction cost: ${result.total_transaction_cost:.2f}")
    print(f"   Cost as % of portfolio: {result.total_transaction_cost / portfolio_value:.3%}")
    print(f"   Maximum drift: {result.max_drift:.2%}")
    
    # 4. Check if rebalancing is needed
    print("\n4. Checking Rebalancing Triggers:")
    should_rebal = rebalancer.should_rebalance(
        current_weights,
        target_weights,
        datetime.now(),
        frequency=RebalanceFrequency.THRESHOLD,
        drift_threshold=0.05
    )
    print(f"   Should rebalance (5% threshold): {should_rebal}")


def demo_portfolio_analysis():
    """Demonstrate portfolio analysis and visualization."""
    print("\n" + "="*80)
    print("PORTFOLIO ANALYSIS AND REPORTING")
    print("="*80)
    
    # Generate sample data
    returns_dict, assets = generate_sample_returns(n_assets=4)
    
    # Create optimizer and get optimal portfolio
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    result = optimizer.optimize_mpt(returns_dict, objective='max_sharpe')
    
    weights = result['weights']
    
    # Create analyzer
    analyzer = PortfolioAnalyzer()
    
    # 1. Risk decomposition
    print("\n1. Risk Decomposition Analysis:")
    risk_calc = RiskCalculator()
    risk_decomp = risk_calc.calculate_risk_decomposition(returns_dict, weights)
    
    print(f"   Portfolio Volatility: {risk_decomp['portfolio_volatility']:.4f}")
    print("\n   Risk Contributions:")
    for asset, metrics in risk_decomp['assets'].items():
        print(f"     {asset}:")
        print(f"       Weight: {metrics['weight']:.2%}")
        print(f"       Contribution: {metrics['percent_contribution']:.1f}%")
    
    # 2. Scenario analysis
    print("\n2. Scenario Analysis:")
    scenarios = {
        'Market Crash': {'AAPL': -0.30, 'MSFT': -0.25, 'GOOGL': -0.28, 'AMZN': -0.32},
        'Bull Market': {'AAPL': 0.25, 'MSFT': 0.20, 'GOOGL': 0.22, 'AMZN': 0.30},
        'Tech Selloff': {'AAPL': -0.15, 'MSFT': -0.12, 'GOOGL': -0.18, 'AMZN': -0.20}
    }
    
    scenario_results = analyzer.perform_scenario_analysis(returns_dict, weights, scenarios)
    print("\n   Scenario Results:")
    print(scenario_results.to_string(index=False))
    
    # 3. Stress testing
    print("\n3. Stress Testing:")
    stress_results = analyzer.perform_stress_test(returns_dict, weights)
    print("\n   Stress Test Results:")
    print(stress_results.to_string(index=False))
    
    # 4. Portfolio comparison
    print("\n4. Portfolio Comparison:")
    
    # Get different portfolio strategies
    max_sharpe = optimizer.optimize_mpt(returns_dict, objective='max_sharpe')
    min_risk = optimizer.optimize_mpt(returns_dict, objective='min_risk')
    risk_parity = optimizer.optimize_risk_parity(returns_dict)
    
    portfolios = {
        'Max Sharpe': max_sharpe,
        'Min Risk': min_risk,
        'Risk Parity': risk_parity
    }
    
    comparison = analyzer.compare_portfolios(portfolios)
    print("\n   Portfolio Comparison:")
    print(comparison.to_string())


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("PORTFOLIO OPTIMIZATION DEMONSTRATION")
    print("="*80)
    print("\nThis demo showcases the comprehensive portfolio optimization capabilities")
    print("of the ARA AI system, including multiple optimization strategies,")
    print("constraints, rebalancing, and analysis tools.")
    
    try:
        # Run demonstrations
        demo_mpt_optimization()
        demo_black_litterman()
        demo_risk_parity()
        demo_kelly_criterion()
        demo_mean_cvar()
        demo_efficient_frontier()
        demo_strategy_comparison()
        demo_constraints_and_rebalancing()
        demo_portfolio_analysis()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print("\nAll portfolio optimization features demonstrated successfully!")
        print("\nKey Features:")
        print("  ✓ Modern Portfolio Theory (MPT)")
        print("  ✓ Black-Litterman Model")
        print("  ✓ Risk Parity Optimization")
        print("  ✓ Kelly Criterion Position Sizing")
        print("  ✓ Mean-CVaR Optimization")
        print("  ✓ Efficient Frontier Calculation")
        print("  ✓ Portfolio Constraints")
        print("  ✓ Transaction Cost Modeling")
        print("  ✓ Rebalancing Strategies")
        print("  ✓ Portfolio Analysis & Reporting")
        print("  ✓ Scenario Analysis & Stress Testing")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
