# Multi-Asset Correlation Analysis

Comprehensive correlation analysis module for cross-asset trading, pairs trading, and inter-market relationship modeling.

## Features

### 1. Correlation Analysis (`CorrelationAnalyzer`)

- **Rolling Correlation Calculation**: Calculate correlations with windows from 7 to 365 days
- **Correlation Matrix**: Generate correlation matrices for multiple assets
- **Correlation Breakdown Detection**: Identify when correlations change by more than 0.3
- **Lead-Lag Relationship Detection**: Discover which assets lead or lag others
- **Correlation Stability Metrics**: Measure how stable correlations are over time
- **Correlation Regime Classification**: Classify correlation strength (strong positive, weak, strong negative, etc.)

### 2. Pairs Trading (`PairsTradingAnalyzer`)

- **Opportunity Identification**: Find pairs with correlation > 0.8
- **Cointegration Testing**: Test for mean-reverting relationships
- **Spread Analysis**: Analyze spread statistics and z-scores
- **Trading Signals**: Generate entry/exit signals based on z-scores
- **Position Sizing**: Calculate optimal position sizes for pairs trades
- **Half-Life Calculation**: Estimate mean reversion speed

### 3. Cross-Asset Prediction (`CrossAssetPredictor`)

- **Related Asset Identification**: Find assets most correlated with target
- **Cross-Asset Features**: Create features from related assets (e.g., BTC price for altcoins)
- **Prediction Adjustment**: Adjust predictions based on correlated asset movements
- **Arbitrage Detection**: Identify statistical arbitrage opportunities
- **Inter-Market Modeling**: Build models of cross-market relationships
- **Lead-Lag Features**: Incorporate lagged features from leading assets

## Quick Start

### Basic Correlation Analysis

```python
from ara.correlation import CorrelationAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = CorrelationAnalyzer(
    min_window=7,
    max_window=365,
    breakdown_threshold=0.3
)

# Calculate rolling correlation
btc_prices = pd.Series(...)  # Your BTC price data
eth_prices = pd.Series(...)  # Your ETH price data

rolling_corr = analyzer.calculate_rolling_correlation(
    btc_prices,
    eth_prices,
    window=30
)

# Detect correlation breakdowns
breakdowns = analyzer.detect_correlation_breakdowns(
    btc_prices,
    eth_prices,
    'BTC',
    'ETH',
    short_window=30,
    long_window=90
)

for breakdown in breakdowns:
    print(f"Breakdown detected on {breakdown.detection_date}")
    print(f"Correlation changed from {breakdown.previous_correlation:.2f} "
          f"to {breakdown.current_correlation:.2f}")

# Detect lead-lag relationships
lead_lag = analyzer.detect_lead_lag_relationship(
    btc_prices,
    eth_prices,
    'BTC',
    'ETH',
    max_lag=10
)

if lead_lag:
    print(f"{lead_lag.leading_asset} leads {lead_lag.lagging_asset} "
          f"by {lead_lag.optimal_lag_days} days")
```

### Pairs Trading

```python
from ara.correlation import PairsTradingAnalyzer

# Initialize analyzer
pairs_analyzer = PairsTradingAnalyzer(
    correlation_threshold=0.8,
    entry_z_score=2.0,
    exit_z_score=0.5
)

# Analyze a specific pair
opportunity = pairs_analyzer.analyze_pair(
    aapl_prices,
    msft_prices,
    'AAPL',
    'MSFT'
)

if opportunity:
    print(f"Pairs trading opportunity: {opportunity.asset1}-{opportunity.asset2}")
    print(f"Correlation: {opportunity.correlation:.2f}")
    print(f"Current Z-score: {opportunity.z_score:.2f}")
    print(f"Signal: {opportunity.signal}")
    print(f"Confidence: {opportunity.confidence:.2f}")

# Find all opportunities in a portfolio
asset_data = {
    'AAPL': aapl_prices,
    'MSFT': msft_prices,
    'GOOGL': googl_prices,
    'AMZN': amzn_prices
}

opportunities = pairs_analyzer.identify_pairs_opportunities(asset_data)

for opp in opportunities[:5]:  # Top 5 opportunities
    print(f"{opp.asset1}-{opp.asset2}: {opp.signal} (confidence: {opp.confidence:.2f})")
```

### Cross-Asset Prediction Enhancement

```python
from ara.correlation import CrossAssetPredictor, CorrelationAnalyzer

# Initialize predictor
predictor = CrossAssetPredictor(
    correlation_analyzer=CorrelationAnalyzer(),
    min_correlation=0.5
)

# Identify related assets
asset_data = {
    'BTC': btc_prices,
    'ETH': eth_prices,
    'LTC': ltc_prices,
    'XRP': xrp_prices
}

related = predictor.identify_related_assets(
    'ETH',
    asset_data,
    top_n=3
)

print("Assets most correlated with ETH:")
for asset, corr in related:
    print(f"  {asset}: {corr:.2f}")

# Create cross-asset features
features = predictor.create_cross_asset_features(
    'ETH',
    asset_data,
    max_features=10
)

print(f"\nCreated {len(features)} cross-asset features")
for feature in features[:3]:
    print(f"  {feature.source_asset} {feature.feature_type} "
          f"(lag={feature.lag_days}, importance={feature.importance:.2f})")

# Calculate feature values
feature_values = predictor.calculate_cross_asset_feature_values(
    features,
    asset_data
)

print("\nFeature values:")
for name, value in list(feature_values.items())[:5]:
    print(f"  {name}: {value:.4f}")

# Adjust prediction using correlations
base_prediction = 3500.0  # Your base ETH prediction
related_predictions = {
    'BTC': 65000.0,
    'LTC': 180.0
}

adjusted_prediction = predictor.adjust_prediction_with_correlations(
    'ETH',
    base_prediction,
    related_predictions,
    asset_data,
    adjustment_strength=0.3
)

print(f"\nBase prediction: ${base_prediction:.2f}")
print(f"Adjusted prediction: ${adjusted_prediction:.2f}")

# Detect arbitrage opportunities
predictions = {
    'BTC': 65000.0,
    'ETH': 3500.0,
    'LTC': 180.0
}

arbitrage_opps = predictor.detect_arbitrage_opportunities(
    asset_data,
    predictions,
    min_mispricing=0.02
)

print(f"\nFound {len(arbitrage_opps)} arbitrage opportunities")
for opp in arbitrage_opps[:3]:
    print(f"  {opp.asset1}-{opp.asset2}: {opp.mispricing_pct*100:.2f}% mispricing")
```

## Use Cases

### 1. Portfolio Diversification

Use correlation analysis to ensure your portfolio is properly diversified:

```python
# Calculate correlation matrix
corr_matrix = analyzer.calculate_correlation_matrix(asset_data)

# Identify highly correlated assets (> 0.8)
high_corr_pairs = []
for i in range(len(corr_matrix)):
    for j in range(i+1, len(corr_matrix)):
        if corr_matrix.iloc[i, j] > 0.8:
            high_corr_pairs.append((
                corr_matrix.index[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print("Highly correlated pairs (consider reducing exposure):")
for asset1, asset2, corr in high_corr_pairs:
    print(f"  {asset1}-{asset2}: {corr:.2f}")
```

### 2. Hedging Strategies

Find negatively correlated assets for hedging:

```python
# Find assets with negative correlation
for asset in asset_data.keys():
    related = predictor.identify_related_assets(asset, asset_data, top_n=10)
    
    negative_corr = [(a, c) for a, c in related if c < -0.5]
    
    if negative_corr:
        print(f"\nHedging options for {asset}:")
        for hedge_asset, corr in negative_corr:
            print(f"  {hedge_asset}: {corr:.2f}")
```

### 3. Market Regime Detection

Monitor correlation breakdowns as market regime indicators:

```python
# Check for recent correlation breakdowns
for i, asset1 in enumerate(list(asset_data.keys())):
    for asset2 in list(asset_data.keys())[i+1:]:
        breakdowns = analyzer.detect_correlation_breakdowns(
            asset_data[asset1],
            asset_data[asset2],
            asset1,
            asset2
        )
        
        if breakdowns:
            recent = [b for b in breakdowns if 
                     (datetime.now() - b.detection_date).days < 30]
            
            if recent:
                print(f"Recent breakdown: {asset1}-{asset2}")
                print(f"  Correlation changed by {recent[-1].change:.2f}")
```

### 4. Crypto Market Analysis

Analyze Bitcoin's influence on altcoins:

```python
# BTC as leading indicator for altcoins
altcoins = ['ETH', 'LTC', 'XRP', 'ADA', 'DOT']
btc_influence = {}

for altcoin in altcoins:
    if altcoin not in asset_data:
        continue
    
    lead_lag = analyzer.detect_lead_lag_relationship(
        asset_data['BTC'],
        asset_data[altcoin],
        'BTC',
        altcoin
    )
    
    if lead_lag and lead_lag.leading_asset == 'BTC':
        btc_influence[altcoin] = {
            'lag_days': lead_lag.optimal_lag_days,
            'correlation': lead_lag.correlation_at_lag
        }

print("BTC's influence on altcoins:")
for altcoin, info in btc_influence.items():
    print(f"  {altcoin}: leads by {info['lag_days']} days "
          f"(corr={info['correlation']:.2f})")
```

## Requirements

Requirement 9 from the specification:

- **9.1**: Calculate rolling correlations between any two assets with windows from 7 to 365 days ✓
- **9.2**: Identify correlation breakdowns when correlation changes by more than 0.3 in 30 days ✓
- **9.3**: Provide cross-asset predictions that account for inter-market relationships ✓
- **9.4**: Detect lead-lag relationships between correlated assets ✓
- **9.5**: Suggest pairs trading opportunities when correlation exceeds 0.8 ✓

## API Reference

### CorrelationAnalyzer

#### Methods

- `calculate_rolling_correlation(data1, data2, window)`: Calculate rolling correlation
- `calculate_correlation_matrix(data, window)`: Generate correlation matrix
- `detect_correlation_breakdowns(data1, data2, asset1_name, asset2_name, short_window, long_window)`: Find breakdowns
- `detect_lead_lag_relationship(data1, data2, asset1_name, asset2_name, max_lag)`: Detect lead-lag
- `calculate_correlation_stability(data1, data2, window, lookback_periods)`: Measure stability
- `analyze_correlation_regime(data1, data2, window)`: Classify correlation strength

### PairsTradingAnalyzer

#### Methods

- `identify_pairs_opportunities(data, min_correlation)`: Find all pairs opportunities
- `analyze_pair(data1, data2, asset1_name, asset2_name)`: Analyze specific pair
- `calculate_position_sizes(opportunity, capital, risk_per_trade)`: Calculate position sizes

### CrossAssetPredictor

#### Methods

- `identify_related_assets(target_asset, available_assets, top_n)`: Find related assets
- `create_cross_asset_features(target_asset, available_assets, max_features)`: Create features
- `calculate_cross_asset_feature_values(features, asset_data, date)`: Calculate feature values
- `adjust_prediction_with_correlations(target_asset, base_prediction, related_predictions, asset_data, adjustment_strength)`: Adjust predictions
- `detect_arbitrage_opportunities(asset_data, predictions, min_mispricing)`: Find arbitrage
- `build_inter_market_model(asset_relationships, asset_data)`: Build relationship model

## Performance Considerations

- **Caching**: Correlation calculations are cached to avoid redundant computation
- **Vectorization**: Uses NumPy/Pandas vectorized operations for speed
- **Lazy Evaluation**: Only calculates correlations when needed
- **Batch Processing**: Can process multiple asset pairs in parallel

## Best Practices

1. **Window Selection**: Use shorter windows (7-30 days) for volatile markets, longer windows (90-365 days) for stable markets
2. **Correlation Threshold**: Adjust based on asset class (stocks: 0.5+, crypto: 0.7+)
3. **Lead-Lag Analysis**: Most useful for related assets in different time zones or markets
4. **Pairs Trading**: Combine correlation with cointegration testing for best results
5. **Cross-Asset Features**: Use leading indicators (e.g., BTC for altcoins, SPY for stocks)

## Limitations

- Correlation is not causation - high correlation doesn't imply predictive power
- Past correlations may not persist in the future
- Correlation can break down during market stress
- Lead-lag relationships can reverse or disappear
- Pairs trading requires careful risk management

## Future Enhancements

- Dynamic correlation modeling with regime switching
- Multi-asset correlation networks
- Correlation forecasting
- Integration with portfolio optimization
- Real-time correlation monitoring
