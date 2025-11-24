"""
Demo of the enhanced data layer with multi-provider support
Demonstrates caching, validation, and data aggregation
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
    DataAggregator,
    CacheManager,
    ValidationConfig,
    ImputationStrategy,
    OutlierMethod
)
from ara.utils import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)


async def demo_basic_failover():
    """Demonstrate basic failover between providers"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Failover")
    print("="*60)
    
    # Initialize providers
    providers = [
        BinanceProvider(),
        CoinbaseProvider(),
        KrakenProvider()
    ]
    
    # Create aggregator with failover enabled
    aggregator = DataAggregator(
        providers=providers,
        enable_failover=True,
        enable_aggregation=False,
        primary_provider='binance'
    )
    
    # Fetch data (will try Binance first, then failover if needed)
    symbol = 'BTC/USDT'
    print(f"\nFetching {symbol} data with failover...")
    
    try:
        df = await aggregator.fetch_historical(
            symbol=symbol,
            period='1mo',
            interval='1d'
        )
        
        print(f"\n✓ Successfully fetched {len(df)} rows")
        print(f"  Source: {df.attrs.get('source', 'unknown')}")
        print(f"  Quality Score: {df.attrs.get('quality_score', 0):.2f}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        # Show provider statistics
        print("\nProvider Statistics:")
        for stats in aggregator.get_provider_stats():
            print(f"  {stats['provider']}: "
                  f"Success Rate={stats['success_rate']:.1%}, "
                  f"Avg Latency={stats['avg_latency']:.2f}s")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")


async def demo_data_aggregation():
    """Demonstrate multi-source data aggregation"""
    print("\n" + "="*60)
    print("DEMO 2: Multi-Source Aggregation")
    print("="*60)
    
    # Initialize providers
    providers = [
        BinanceProvider(),
        CoinbaseProvider(),
        KrakenProvider()
    ]
    
    # Create aggregator with aggregation enabled
    aggregator = DataAggregator(
        providers=providers,
        enable_failover=True,
        enable_aggregation=True  # Fetch from all sources and combine
    )
    
    # Fetch and aggregate data
    symbol = 'ETH/USDT'
    print(f"\nFetching {symbol} from multiple sources and aggregating...")
    
    try:
        df = await aggregator.fetch_historical(
            symbol=symbol,
            period='7d',
            interval='1d'
        )
        
        print(f"\n✓ Successfully aggregated data from multiple sources")
        print(f"  Total Rows: {len(df)}")
        print(f"  Sources: {df.attrs.get('sources', [])}")
        print(f"  Quality Score: {df.attrs.get('quality_score', 0):.2f}")
        print(f"\nAggregated data:")
        print(df.tail(5))
        
    except Exception as e:
        print(f"\n✗ Error: {e}")


async def demo_caching():
    """Demonstrate caching functionality"""
    print("\n" + "="*60)
    print("DEMO 3: Caching System")
    print("="*60)
    
    # Initialize cache manager
    cache_manager = CacheManager(
        l1_size_mb=50,  # 50MB L1 cache
        l1_ttl=60,      # 60 second TTL
        l2_enabled=True,  # Enable Redis if available
        l2_ttl=3600     # 1 hour TTL
    )
    
    # Initialize aggregator with cache
    providers = [BinanceProvider()]
    aggregator = DataAggregator(
        providers=providers,
        cache_manager=cache_manager
    )
    
    symbol = 'BTC/USDT'
    
    # First fetch (cache miss)
    print(f"\nFirst fetch of {symbol} (cache miss)...")
    import time
    start = time.time()
    df1 = await aggregator.fetch_historical(symbol, period='1mo', use_cache=True)
    time1 = time.time() - start
    print(f"  Time: {time1:.2f}s")
    
    # Second fetch (cache hit)
    print(f"\nSecond fetch of {symbol} (cache hit)...")
    start = time.time()
    df2 = await aggregator.fetch_historical(symbol, period='1mo', use_cache=True)
    time2 = time.time() - start
    print(f"  Time: {time2:.2f}s")
    print(f"  Speedup: {time1/time2:.1f}x faster")
    
    # Show cache statistics
    print("\nCache Statistics:")
    stats = cache_manager.get_stats()
    print(f"  L1 Hit Rate: {stats['l1']['hit_rate']:.1%}")
    print(f"  L1 Size: {stats['l1']['size_mb']:.2f} MB")
    print(f"  L1 Items: {stats['l1']['items']}")
    if stats['l2']:
        print(f"  L2 Hit Rate: {stats['l2']['hit_rate']:.1%}")
        print(f"  L2 Connected: {stats['l2']['connected']}")


async def demo_validation():
    """Demonstrate data validation and cleaning"""
    print("\n" + "="*60)
    print("DEMO 4: Data Validation & Cleaning")
    print("="*60)
    
    # Configure validation
    config = ValidationConfig(
        min_data_points=20,
        max_missing_percentage=15.0,
        outlier_method=OutlierMethod.IQR,
        imputation_strategy=ImputationStrategy.LINEAR_INTERPOLATE,
        check_consistency=True
    )
    
    # Initialize aggregator with validation
    providers = [BinanceProvider()]
    aggregator = DataAggregator(
        providers=providers,
        validation_config=config
    )
    
    # Fetch and clean data
    symbol = 'BTC/USDT'
    print(f"\nFetching and validating {symbol}...")
    
    df = await aggregator.fetch_historical(
        symbol=symbol,
        period='1mo',
        clean_data=True  # Enable cleaning
    )
    
    # Get quality report
    quality_report = df.attrs.get('quality_report')
    
    print(f"\n✓ Data Quality Report:")
    print(f"  Quality Score: {quality_report.quality_score:.2f}/1.0")
    print(f"  Total Rows: {quality_report.total_rows}")
    print(f"  Missing Data: {quality_report.missing_percentage:.1f}%")
    print(f"  Outliers: {quality_report.outliers_percentage:.1f}%")
    print(f"  Validation Passed: {quality_report.passed_validation}")
    
    if quality_report.consistency_issues:
        print(f"\n  Consistency Issues:")
        for issue in quality_report.consistency_issues:
            print(f"    - {issue}")
    
    if quality_report.recommendations:
        print(f"\n  Recommendations:")
        for rec in quality_report.recommendations:
            print(f"    - {rec}")


async def demo_rate_limiting():
    """Demonstrate rate limiting"""
    print("\n" + "="*60)
    print("DEMO 5: Rate Limiting")
    print("="*60)
    
    # Initialize aggregator with rate limiting
    providers = [BinanceProvider()]
    aggregator = DataAggregator(
        providers=providers,
        rate_limit=2.0  # 2 requests per second
    )
    
    # Make multiple requests
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    print(f"\nFetching {len(symbols)} symbols with rate limiting (2 req/s)...")
    import time
    start = time.time()
    
    tasks = []
    for symbol in symbols:
        task = aggregator.fetch_realtime(symbol, use_cache=False)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed = time.time() - start
    print(f"\n✓ Completed {len(symbols)} requests in {elapsed:.2f}s")
    print(f"  Average: {elapsed/len(symbols):.2f}s per request")
    
    # Show rate limiter stats
    print("\nRate Limiter Statistics:")
    for provider, stats in aggregator.get_rate_limiter_stats().items():
        print(f"  {provider}: Rate={stats['rate']}/s, "
              f"Burst={stats['burst_size']}, "
              f"Available={stats['available_tokens']:.1f}")


async def demo_realtime_with_cache():
    """Demonstrate real-time data fetching with caching"""
    print("\n" + "="*60)
    print("DEMO 6: Real-Time Data with Caching")
    print("="*60)
    
    # Initialize aggregator
    providers = [BinanceProvider(), CoinbaseProvider()]
    aggregator = DataAggregator(
        providers=providers,
        enable_failover=True
    )
    
    symbol = 'BTC/USDT'
    
    # Fetch real-time data multiple times
    print(f"\nFetching real-time {symbol} data 3 times...")
    
    for i in range(3):
        data = await aggregator.fetch_realtime(symbol, use_cache=True)
        print(f"\n  Fetch {i+1}:")
        print(f"    Price: ${data['price']:,.2f}")
        print(f"    Volume 24h: {data['volume']:,.2f}")
        print(f"    Change 24h: {data['change_pct_24h']:+.2f}%")
        print(f"    Source: {data['exchange']}")
        
        await asyncio.sleep(1)  # Wait 1 second


async def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("ARA AI - Enhanced Data Layer Demo")
    print("="*60)
    
    try:
        # Run demos
        await demo_basic_failover()
        await demo_data_aggregation()
        await demo_caching()
        await demo_validation()
        await demo_rate_limiting()
        await demo_realtime_with_cache()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
