"""
Cryptocurrency Data Providers Demo
Demonstrates the new crypto data fetching capabilities
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara.data import (
    BinanceProvider,
    CoinbaseProvider,
    KrakenProvider,
    CryptoDataAggregator,
    OnChainMetricsProvider,
    DeFiDataProvider
)


async def demo_exchange_providers():
    """Demonstrate cryptocurrency exchange providers"""
    print("\n" + "="*60)
    print("CRYPTOCURRENCY EXCHANGE PROVIDERS DEMO")
    print("="*60)
    
    # Initialize providers
    print("\n1. Initializing exchange providers...")
    binance = BinanceProvider()
    coinbase = CoinbaseProvider()
    kraken = KrakenProvider()
    
    print(f"   ✓ Binance: {binance.name}")
    print(f"   ✓ Coinbase: {coinbase.name}")
    print(f"   ✓ Kraken: {kraken.name}")
    
    # Fetch historical data
    print("\n2. Fetching historical data for BTC/USDT...")
    try:
        df = await binance.fetch_historical('BTC/USDT', period='5d', interval='1d')
        print(f"   ✓ Fetched {len(df)} days of data")
        print(f"   Latest close: ${df['Close'].iloc[-1]:,.2f}")
        print(f"   24h change: {((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100:.2f}%")
    except Exception as e:
        print(f"   ⚠ Could not fetch data (network required): {e}")
    
    # Fetch real-time data
    print("\n3. Fetching real-time ticker data...")
    try:
        ticker = await binance.fetch_realtime('BTC/USDT')
        print(f"   ✓ Current price: ${ticker['price']:,.2f}")
        print(f"   ✓ 24h volume: ${ticker['volume_quote']:,.0f}")
        print(f"   ✓ 24h change: {ticker['change_pct_24h']:.2f}%")
    except Exception as e:
        print(f"   ⚠ Could not fetch real-time data: {e}")


async def demo_data_aggregator():
    """Demonstrate data aggregation from multiple exchanges"""
    print("\n" + "="*60)
    print("MULTI-EXCHANGE DATA AGGREGATOR DEMO")
    print("="*60)
    
    # Initialize aggregator
    print("\n1. Initializing data aggregator...")
    aggregator = CryptoDataAggregator()
    print(f"   ✓ Aggregator initialized with {len(aggregator.providers)} providers")
    
    # Show major cryptocurrencies
    print("\n2. Supported major cryptocurrencies (50+):")
    major_cryptos = aggregator.get_major_cryptocurrencies()
    print(f"   ✓ Total: {len(major_cryptos)} cryptocurrencies")
    print(f"   Top 10: {', '.join(major_cryptos[:10])}")
    
    # Fetch with automatic failover
    print("\n3. Fetching data with automatic failover...")
    try:
        df = await aggregator.fetch_historical('BTC/USDT', period='5d')
        print(f"   ✓ Successfully fetched from: {df['source'].iloc[0]}")
        print(f"   ✓ Data quality score: {df['quality_score'].mean():.2f}")
        print(f"   ✓ Rows: {len(df)}")
    except Exception as e:
        print(f"   ⚠ Could not fetch data: {e}")


async def demo_onchain_metrics():
    """Demonstrate on-chain metrics provider"""
    print("\n" + "="*60)
    print("ON-CHAIN METRICS PROVIDER DEMO")
    print("="*60)
    
    async with OnChainMetricsProvider() as provider:
        # Fetch network metrics
        print("\n1. Fetching network metrics for BTC...")
        metrics_df = await provider.fetch_network_metrics('BTC', days=30)
        print(f"   ✓ Fetched {len(metrics_df.columns)} metrics")
        print(f"   ✓ Metrics: {', '.join(metrics_df.columns[:5])}")
        
        # Calculate derived metrics
        print("\n2. Calculating derived metrics...")
        import pandas as pd
        price_df = pd.DataFrame({
            'Close': [50000] * len(metrics_df)
        }, index=metrics_df.index)
        
        derived = provider.calculate_derived_metrics(metrics_df, price_df)
        print(f"   ✓ Calculated {len(derived.columns)} derived metrics")
        if len(derived.columns) > 0:
            print(f"   ✓ Metrics: {', '.join(derived.columns)}")
        
        # Fetch exchange flows
        print("\n3. Fetching exchange flow data...")
        flows_df = await provider.fetch_exchange_flows('BTC', days=30)
        print(f"   ✓ Fetched {len(flows_df)} days of flow data")
        print(f"   ✓ Columns: {', '.join(flows_df.columns)}")
        
        # Track whale wallets
        print("\n4. Tracking whale wallet activity...")
        whales = await provider.track_whale_wallets('BTC', threshold=1000000)
        print(f"   ✓ Found {len(whales)} large transactions")
        if whales:
            print(f"   ✓ Largest: ${whales[0]['value_usd']:,.0f}")
        
        # Calculate network health
        print("\n5. Calculating network health score...")
        health = provider.calculate_network_health_score(metrics_df)
        print(f"   ✓ Average health score: {health.mean():.1f}/100")
        print(f"   ✓ Current health: {health.iloc[-1]:.1f}/100")


async def demo_defi_provider():
    """Demonstrate DeFi data provider"""
    print("\n" + "="*60)
    print("DEFI DATA PROVIDER DEMO")
    print("="*60)
    
    async with DeFiDataProvider() as provider:
        # Fetch TVL data
        print("\n1. Fetching Total Value Locked (TVL) data...")
        tvl_df = await provider.fetch_tvl_data()
        print(f"   ✓ Fetched {len(tvl_df)} days of TVL data")
        if len(tvl_df) > 0:
            print(f"   ✓ Latest TVL: ${tvl_df['tvl'].iloc[-1]:,.0f}")
        
        # Fetch lending rates
        print("\n2. Fetching lending rates...")
        rates = await provider.fetch_lending_rates('aave', 'USDC')
        print(f"   ✓ Protocol: {rates['protocol']}")
        print(f"   ✓ Supply APY: {rates['supply_apy']:.2f}%")
        print(f"   ✓ Borrow APY: {rates['borrow_apy']:.2f}%")
        print(f"   ✓ Utilization: {rates['utilization_rate']*100:.1f}%")
        
        # Fetch liquidation data
        print("\n3. Fetching liquidation data...")
        liq_df = await provider.fetch_liquidation_data(days=30)
        print(f"   ✓ Fetched {len(liq_df)} days of liquidation data")
        print(f"   ✓ Total liquidations: {liq_df['liquidation_count'].sum():.0f}")
        print(f"   ✓ Total volume: ${liq_df['liquidation_volume_usd'].sum():,.0f}")
        
        # Fetch stablecoin supply
        print("\n4. Fetching stablecoin supply...")
        stable_df = await provider.fetch_stablecoin_supply('USDT')
        print(f"   ✓ Fetched {len(stable_df)} days of supply data")
        if 'circulating' in stable_df.columns:
            print(f"   ✓ Latest supply: ${stable_df['circulating'].iloc[-1]:,.0f}")
        
        # Get market overview
        print("\n5. Getting DeFi market overview...")
        overview = await provider.get_defi_market_overview()
        print(f"   ✓ Total TVL: ${overview['total_tvl']:,.0f}")
        print(f"   ✓ 30d change: {overview['tvl_change_30d']*100:.2f}%")
        
        # Calculate DeFi metrics
        print("\n6. Calculating DeFi metrics...")
        import pandas as pd
        price_df = pd.DataFrame({
            'Close': [100] * len(tvl_df)
        }, index=tvl_df.index)
        
        metrics = provider.calculate_defi_metrics(tvl_df, price_df, rates)
        print(f"   ✓ Calculated {len(metrics.columns)} metrics")
        if len(metrics.columns) > 0:
            print(f"   ✓ Metrics: {', '.join(list(metrics.columns)[:5])}")
        
        # Calculate risk score
        print("\n7. Calculating DeFi risk score...")
        risk = provider.calculate_defi_risk_score(liq_df, tvl_df)
        print(f"   ✓ Average risk: {risk.mean():.1f}/100")
        print(f"   ✓ Current risk: {risk.iloc[-1]:.1f}/100")


async def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("ARA AI - CRYPTOCURRENCY DATA PROVIDERS")
    print("Comprehensive Demo of New Features")
    print("="*60)
    
    try:
        # Run demos
        await demo_exchange_providers()
        await demo_data_aggregator()
        await demo_onchain_metrics()
        await demo_defi_provider()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("  ✓ Multi-exchange cryptocurrency data fetching")
        print("  ✓ Automatic failover and data aggregation")
        print("  ✓ Support for 50+ major cryptocurrencies")
        print("  ✓ On-chain metrics (network activity, whale tracking)")
        print("  ✓ DeFi data (TVL, lending rates, liquidations)")
        print("  ✓ Real-time and historical data")
        print("  ✓ Data quality scoring and validation")
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
