"""
Multi-Asset Correlation Analysis Demo

Demonstrates the correlation analysis capabilities including:
- Rolling correlation calculation
- Correlation breakdown detection
- Lead-lag relationship detection
- Pairs trading opportunities
- Cross-asset prediction enhancement
- Arbitrage detection
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, '.')

from ara.correlation import (
    CorrelationAnalyzer,
    PairsTradingAnalyzer,
    CrossAssetPredictor
)


def generate_sample_data(
    n_days: int = 365,
    correlation: float = 0.8,
    lag: int = 0
) -> tuple:
    """Generate correlated sample price data"""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate base series
    returns1 = np.random.normal(0.001, 0.02, n_days)
    prices1 = 100 * np.exp(np.cumsum(returns1))
    
    # Generate correlated series
    noise = np.random.normal(0, 0.01, n_days)
    returns2 = correlation * returns1 + np.sqrt(1 - correlation**2) * noise
    
    # Apply lag if specified
    if lag > 0:
        returns2 = np.roll(returns2, lag)
        returns2[:lag] = 0
    
    prices2 = 100 * np.exp(np.cumsum(returns2))
    
    series1 = pd.Series(prices1, index=dates, name='Asset1')
    series2 = pd.Series(prices2, index=dates, name='Asset2')
    
    return series1, series2


def demo_correlation_analysis():
    """Demonstrate basic correlation analysis"""
    print("=" * 80)
    print("CORRELATION ANALYSIS DEMO")
    print("=" * 80)
    
    # Generate sample data with same date range
    print("\n1. Generating sample data...")
    n_days = 365
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate BTC and ETH
    btc_prices, eth_prices = generate_sample_data(n_days=n_days, correlation=0.85)
    btc_prices.index = dates
    eth_prices.index = dates
    
    # Generate AAPL and MSFT with same dates
    aapl_prices, msft_prices = generate_sample_data(n_days=n_days, correlation=0.75)
    aapl_prices.index = dates
    msft_prices.index = dates
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer(
        min_window=7,
        max_window=365,
        breakdown_threshold=0.3
    )
    
    # Calculate rolling correlation
    print("\n2. Calculating rolling correlations...")
    rolling_corr_30 = analyzer.calculate_rolling_correlation(
        btc_prices, eth_prices, window=30
    )
    rolling_corr_90 = analyzer.calculate_rolling_correlation(
        btc_prices, eth_prices, window=90
    )
    
    print(f"   30-day correlation: {rolling_corr_30.iloc[-1]:.3f}")
    print(f"   90-day correlation: {rolling_corr_90.iloc[-1]:.3f}")
    
    # Calculate correlation matrix
    print("\n3. Calculating correlation matrix...")
    asset_data = {
        'BTC': btc_prices,
        'ETH': eth_prices,
        'AAPL': aapl_prices,
        'MSFT': msft_prices
    }
    
    corr_matrix = analyzer.calculate_correlation_matrix(asset_data)
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Detect correlation breakdowns
    print("\n4. Detecting correlation breakdowns...")
    breakdowns = analyzer.detect_correlation_breakdowns(
        btc_prices,
        eth_prices,
        'BTC',
        'ETH',
        short_window=30,
        long_window=90
    )
    
    if breakdowns:
        print(f"   Found {len(breakdowns)} correlation breakdowns")
        for breakdown in breakdowns[-3:]:  # Show last 3
            print(f"   - {breakdown.detection_date.date()}: "
                  f"{breakdown.previous_correlation:.3f} → "
                  f"{breakdown.current_correlation:.3f} "
                  f"(change: {breakdown.change:+.3f})")
    else:
        print("   No significant correlation breakdowns detected")
    
    # Analyze correlation stability
    print("\n5. Analyzing correlation stability...")
    stability = analyzer.calculate_correlation_stability(
        btc_prices, eth_prices, window=30, lookback_periods=12
    )
    
    print(f"   Mean correlation: {stability['mean_correlation']:.3f}")
    print(f"   Std deviation: {stability['std_correlation']:.3f}")
    print(f"   Stability score: {stability['stability_score']:.3f}")
    print(f"   Range: [{stability['min_correlation']:.3f}, "
          f"{stability['max_correlation']:.3f}]")
    
    # Classify correlation regime
    print("\n6. Classifying correlation regime...")
    regime = analyzer.analyze_correlation_regime(btc_prices, eth_prices, window=30)
    print(f"   Current regime: {regime}")


def demo_lead_lag_detection():
    """Demonstrate lead-lag relationship detection"""
    print("\n" + "=" * 80)
    print("LEAD-LAG RELATIONSHIP DETECTION DEMO")
    print("=" * 80)
    
    # Generate data with lead-lag relationship
    print("\n1. Generating data with 3-day lead-lag...")
    leading_prices, lagging_prices = generate_sample_data(
        n_days=365,
        correlation=0.8,
        lag=3
    )
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer()
    
    # Detect lead-lag relationship
    print("\n2. Detecting lead-lag relationship...")
    lead_lag = analyzer.detect_lead_lag_relationship(
        leading_prices,
        lagging_prices,
        'LeadingAsset',
        'LaggingAsset',
        max_lag=10
    )
    
    if lead_lag:
        print(f"   ✓ Lead-lag relationship detected!")
        print(f"   Leading asset: {lead_lag.leading_asset}")
        print(f"   Lagging asset: {lead_lag.lagging_asset}")
        print(f"   Optimal lag: {lead_lag.optimal_lag_days} days")
        print(f"   Correlation at lag: {lead_lag.correlation_at_lag:.3f}")
        print(f"   Confidence: {lead_lag.confidence:.3f}")
    else:
        print("   No significant lead-lag relationship found")


def demo_pairs_trading():
    """Demonstrate pairs trading analysis"""
    print("\n" + "=" * 80)
    print("PAIRS TRADING DEMO")
    print("=" * 80)
    
    # Generate sample data for pairs
    print("\n1. Generating sample pairs data...")
    asset_data = {}
    
    # Create highly correlated pairs
    base_prices, _ = generate_sample_data(n_days=180, correlation=0.9)
    asset_data['AAPL'] = base_prices
    
    for i, symbol in enumerate(['MSFT', 'GOOGL', 'AMZN']):
        _, corr_prices = generate_sample_data(n_days=180, correlation=0.85 - i*0.1)
        asset_data[symbol] = corr_prices
    
    # Initialize analyzer
    pairs_analyzer = PairsTradingAnalyzer(
        correlation_threshold=0.8,
        entry_z_score=2.0,
        exit_z_score=0.5
    )
    
    # Analyze specific pair
    print("\n2. Analyzing AAPL-MSFT pair...")
    opportunity = pairs_analyzer.analyze_pair(
        asset_data['AAPL'],
        asset_data['MSFT'],
        'AAPL',
        'MSFT'
    )
    
    if opportunity:
        print(f"   ✓ Pairs trading opportunity found!")
        print(f"   Correlation: {opportunity.correlation:.3f}")
        print(f"   Cointegration score: {opportunity.cointegration_score:.3f}")
        print(f"   Current spread: {opportunity.current_spread:.3f}")
        print(f"   Mean spread: {opportunity.mean_spread:.3f}")
        print(f"   Std spread: {opportunity.std_spread:.3f}")
        print(f"   Z-score: {opportunity.z_score:.3f}")
        print(f"   Signal: {opportunity.signal}")
        print(f"   Confidence: {opportunity.confidence:.3f}")
        
        # Calculate position sizes
        capital = 100000
        pos1, pos2 = pairs_analyzer.calculate_position_sizes(
            opportunity, capital, risk_per_trade=0.02
        )
        print(f"\n   Position sizing (${capital:,.0f} capital, 2% risk):")
        print(f"   {opportunity.asset1}: ${pos1:,.2f}")
        print(f"   {opportunity.asset2}: ${pos2:,.2f}")
    else:
        print("   No suitable pairs trading opportunity")
    
    # Find all opportunities
    print("\n3. Scanning for all pairs opportunities...")
    opportunities = pairs_analyzer.identify_pairs_opportunities(
        asset_data,
        min_correlation=0.7
    )
    
    print(f"   Found {len(opportunities)} opportunities")
    
    if opportunities:
        print("\n   Top 3 opportunities:")
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"   {i}. {opp.asset1}-{opp.asset2}")
            print(f"      Correlation: {opp.correlation:.3f}, "
                  f"Signal: {opp.signal}, "
                  f"Confidence: {opp.confidence:.3f}")


def demo_cross_asset_prediction():
    """Demonstrate cross-asset prediction enhancement"""
    print("\n" + "=" * 80)
    print("CROSS-ASSET PREDICTION DEMO")
    print("=" * 80)
    
    # Generate sample crypto data
    print("\n1. Generating sample crypto data...")
    btc_prices, _ = generate_sample_data(n_days=180, correlation=0.9)
    
    asset_data = {
        'BTC': btc_prices
    }
    
    # Generate correlated altcoins
    for symbol, corr in [('ETH', 0.85), ('LTC', 0.75), ('XRP', 0.65)]:
        _, prices = generate_sample_data(n_days=180, correlation=corr)
        asset_data[symbol] = prices
    
    # Initialize predictor
    predictor = CrossAssetPredictor(
        min_correlation=0.5,
        feature_lookback=30
    )
    
    # Identify related assets
    print("\n2. Identifying assets related to ETH...")
    related = predictor.identify_related_assets(
        'ETH',
        asset_data,
        top_n=3
    )
    
    print("   Related assets:")
    for asset, corr in related:
        print(f"   - {asset}: {corr:.3f}")
    
    # Create cross-asset features
    print("\n3. Creating cross-asset features for ETH...")
    features = predictor.create_cross_asset_features(
        'ETH',
        asset_data,
        max_features=6
    )
    
    print(f"   Created {len(features)} features:")
    for feature in features[:3]:
        print(f"   - {feature.source_asset}_{feature.feature_type} "
              f"(lag={feature.lag_days}, importance={feature.importance:.3f})")
    
    # Calculate feature values
    print("\n4. Calculating feature values...")
    feature_values = predictor.calculate_cross_asset_feature_values(
        features,
        asset_data
    )
    
    print("   Feature values:")
    for name, value in list(feature_values.items())[:3]:
        print(f"   - {name}: {value:.4f}")
    
    # Adjust prediction
    print("\n5. Adjusting ETH prediction using correlations...")
    current_eth = asset_data['ETH'].iloc[-1]
    base_prediction = current_eth * 1.05  # 5% increase
    
    # Simulate predictions for related assets
    related_predictions = {
        'BTC': asset_data['BTC'].iloc[-1] * 1.08,  # 8% increase
        'LTC': asset_data['LTC'].iloc[-1] * 1.03   # 3% increase
    }
    
    adjusted_prediction = predictor.adjust_prediction_with_correlations(
        'ETH',
        base_prediction,
        related_predictions,
        asset_data,
        adjustment_strength=0.3
    )
    
    print(f"   Current price: ${current_eth:.2f}")
    print(f"   Base prediction: ${base_prediction:.2f} "
          f"({(base_prediction/current_eth - 1)*100:+.1f}%)")
    print(f"   Adjusted prediction: ${adjusted_prediction:.2f} "
          f"({(adjusted_prediction/current_eth - 1)*100:+.1f}%)")
    print(f"   Adjustment: ${adjusted_prediction - base_prediction:+.2f}")


def demo_arbitrage_detection():
    """Demonstrate arbitrage opportunity detection"""
    print("\n" + "=" * 80)
    print("ARBITRAGE DETECTION DEMO")
    print("=" * 80)
    
    # Generate sample data
    print("\n1. Generating sample data with mispricing...")
    asset_data = {}
    
    for symbol in ['ASSET1', 'ASSET2', 'ASSET3', 'ASSET4']:
        prices, _ = generate_sample_data(n_days=180, correlation=0.8)
        asset_data[symbol] = prices
    
    # Create predictions with intentional mispricing
    current_prices = {k: v.iloc[-1] for k, v in asset_data.items()}
    predictions = {
        'ASSET1': current_prices['ASSET1'] * 1.10,  # 10% up
        'ASSET2': current_prices['ASSET2'] * 1.02,  # 2% up (mispriced)
        'ASSET3': current_prices['ASSET3'] * 1.08,  # 8% up
        'ASSET4': current_prices['ASSET4'] * 1.09   # 9% up
    }
    
    # Initialize predictor
    predictor = CrossAssetPredictor()
    
    # Detect arbitrage opportunities
    print("\n2. Detecting arbitrage opportunities...")
    opportunities = predictor.detect_arbitrage_opportunities(
        asset_data,
        predictions,
        min_mispricing=0.02
    )
    
    if opportunities:
        print(f"   Found {len(opportunities)} arbitrage opportunities")
        print("\n   Top opportunities:")
        
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"\n   {i}. {opp.asset1} vs {opp.asset2}")
            print(f"      Type: {opp.opportunity_type}")
            print(f"      Mispricing: {opp.mispricing_pct*100:.2f}%")
            print(f"      Confidence: {opp.confidence:.3f}")
            print(f"      {opp.asset1}: ${opp.current_price1:.2f} → "
                  f"${opp.expected_price1:.2f}")
            print(f"      {opp.asset2}: ${opp.current_price2:.2f} → "
                  f"${opp.expected_price2:.2f}")
    else:
        print("   No arbitrage opportunities detected")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("MULTI-ASSET CORRELATION ANALYSIS - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nThis demo showcases the correlation analysis capabilities:")
    print("- Rolling correlation calculation")
    print("- Correlation breakdown detection")
    print("- Lead-lag relationship detection")
    print("- Pairs trading opportunities")
    print("- Cross-asset prediction enhancement")
    print("- Arbitrage detection")
    
    try:
        demo_correlation_analysis()
        demo_lead_lag_detection()
        demo_pairs_trading()
        demo_cross_asset_prediction()
        demo_arbitrage_detection()
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nAll correlation analysis features demonstrated successfully!")
        print("See ara/correlation/README.md for more details and usage examples.")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
