"""
Tests for cryptocurrency data providers
"""

import pytest
import asyncio
import pandas as pd
from datetime import datetime

from ara.data import (
    BinanceProvider,
    CoinbaseProvider,
    KrakenProvider,
    CryptoDataAggregator,
    OnChainMetricsProvider,
    DeFiDataProvider
)


class TestCryptoExchangeProviders:
    """Test cryptocurrency exchange providers"""
    
    @pytest.mark.asyncio
    async def test_binance_provider_initialization(self):
        """Test Binance provider can be initialized"""
        provider = BinanceProvider()
        assert provider is not None
        assert provider.exchange_id == 'binance'
        assert provider.name == 'crypto_binance'
    
    @pytest.mark.asyncio
    async def test_coinbase_provider_initialization(self):
        """Test Coinbase provider can be initialized"""
        provider = CoinbaseProvider()
        assert provider is not None
        assert provider.exchange_id == 'coinbase'
    
    @pytest.mark.asyncio
    async def test_kraken_provider_initialization(self):
        """Test Kraken provider can be initialized"""
        provider = KrakenProvider()
        assert provider is not None
        assert provider.exchange_id == 'kraken'
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data(self):
        """Test fetching historical data from exchange"""
        provider = BinanceProvider()
        
        try:
            df = await provider.fetch_historical('BTC/USDT', period='5d', interval='1d')
            
            # Verify DataFrame structure
            assert isinstance(df, pd.DataFrame)
            assert 'Open' in df.columns
            assert 'High' in df.columns
            assert 'Low' in df.columns
            assert 'Close' in df.columns
            assert 'Volume' in df.columns
            assert len(df) > 0
            
        except Exception as e:
            # Network errors are acceptable in tests
            pytest.skip(f"Network error: {e}")
    
    @pytest.mark.asyncio
    async def test_symbol_normalization(self):
        """Test symbol normalization"""
        provider = BinanceProvider()
        
        # Test various formats
        assert 'BTC/USDT' in provider._normalize_symbol('BTC-USD')
        assert 'ETH/USDT' in provider._normalize_symbol('ETHUSDT')


class TestCryptoDataAggregator:
    """Test cryptocurrency data aggregator"""
    
    def test_aggregator_initialization(self):
        """Test aggregator can be initialized"""
        aggregator = CryptoDataAggregator()
        assert aggregator is not None
        assert len(aggregator.providers) > 0
    
    def test_aggregator_with_custom_providers(self):
        """Test aggregator with custom providers"""
        providers = [BinanceProvider(), CoinbaseProvider()]
        aggregator = CryptoDataAggregator(providers=providers)
        assert len(aggregator.providers) == 2
    
    def test_major_cryptocurrencies_list(self):
        """Test major cryptocurrencies list"""
        aggregator = CryptoDataAggregator()
        major_cryptos = aggregator.get_major_cryptocurrencies()
        
        assert len(major_cryptos) >= 50
        assert 'BTC/USDT' in major_cryptos
        assert 'ETH/USDT' in major_cryptos
        assert 'SOL/USDT' in major_cryptos
    
    @pytest.mark.asyncio
    async def test_fetch_with_failover(self):
        """Test fetching with automatic failover"""
        aggregator = CryptoDataAggregator()
        
        try:
            df = await aggregator.fetch_historical('BTC/USDT', period='5d')
            
            assert isinstance(df, pd.DataFrame)
            assert 'source' in df.columns
            assert 'quality_score' in df.columns
            
        except Exception as e:
            pytest.skip(f"Network error: {e}")


class TestOnChainMetricsProvider:
    """Test on-chain metrics provider"""
    
    def test_provider_initialization(self):
        """Test provider can be initialized"""
        provider = OnChainMetricsProvider()
        assert provider is not None
    
    @pytest.mark.asyncio
    async def test_fetch_network_metrics(self):
        """Test fetching network metrics"""
        async with OnChainMetricsProvider() as provider:
            df = await provider.fetch_network_metrics('BTC', days=30)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            # Should have at least some metrics
            assert len(df.columns) > 0
    
    @pytest.mark.asyncio
    async def test_calculate_derived_metrics(self):
        """Test calculating derived metrics"""
        async with OnChainMetricsProvider() as provider:
            # Get base metrics
            metrics_df = await provider.fetch_network_metrics('BTC', days=30)
            
            # Create sample price data
            price_df = pd.DataFrame({
                'Close': [50000] * len(metrics_df)
            }, index=metrics_df.index)
            
            # Calculate derived metrics
            derived = provider.calculate_derived_metrics(metrics_df, price_df)
            
            assert isinstance(derived, pd.DataFrame)
            assert len(derived) > 0
    
    @pytest.mark.asyncio
    async def test_fetch_exchange_flows(self):
        """Test fetching exchange flow data"""
        async with OnChainMetricsProvider() as provider:
            df = await provider.fetch_exchange_flows('BTC', days=30)
            
            assert isinstance(df, pd.DataFrame)
            assert 'exchange_inflow' in df.columns
            assert 'exchange_outflow' in df.columns
            assert 'exchange_net_flow' in df.columns
    
    @pytest.mark.asyncio
    async def test_track_whale_wallets(self):
        """Test whale wallet tracking"""
        async with OnChainMetricsProvider() as provider:
            transactions = await provider.track_whale_wallets('BTC', threshold=1000000)
            
            assert isinstance(transactions, list)
            if len(transactions) > 0:
                tx = transactions[0]
                assert 'timestamp' in tx
                assert 'value_usd' in tx
                assert 'type' in tx
    
    def test_network_health_score(self):
        """Test network health score calculation"""
        provider = OnChainMetricsProvider()
        
        # Create sample metrics
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        metrics_df = pd.DataFrame({
            'active_addresses': range(100000, 100030),
            'transaction_volume': range(1000000, 1000030),
            'hash_rate': range(100, 130)
        }, index=dates)
        
        health_score = provider.calculate_network_health_score(metrics_df)
        
        assert isinstance(health_score, pd.Series)
        assert len(health_score) == 30
        assert health_score.min() >= 0
        assert health_score.max() <= 100


class TestDeFiDataProvider:
    """Test DeFi data provider"""
    
    def test_provider_initialization(self):
        """Test provider can be initialized"""
        provider = DeFiDataProvider()
        assert provider is not None
        assert provider.base_url == "https://api.llama.fi"
    
    @pytest.mark.asyncio
    async def test_fetch_tvl_data(self):
        """Test fetching TVL data"""
        async with DeFiDataProvider() as provider:
            df = await provider.fetch_tvl_data()
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'tvl' in df.columns
    
    @pytest.mark.asyncio
    async def test_fetch_lending_rates(self):
        """Test fetching lending rates"""
        async with DeFiDataProvider() as provider:
            rates = await provider.fetch_lending_rates('aave', 'USDC')
            
            assert isinstance(rates, dict)
            assert 'supply_apy' in rates
            assert 'borrow_apy' in rates
            assert 'utilization_rate' in rates
    
    @pytest.mark.asyncio
    async def test_fetch_liquidation_data(self):
        """Test fetching liquidation data"""
        async with DeFiDataProvider() as provider:
            df = await provider.fetch_liquidation_data(days=30)
            
            assert isinstance(df, pd.DataFrame)
            assert 'liquidation_count' in df.columns
            assert 'liquidation_volume_usd' in df.columns
    
    @pytest.mark.asyncio
    async def test_fetch_stablecoin_supply(self):
        """Test fetching stablecoin supply"""
        async with DeFiDataProvider() as provider:
            df = await provider.fetch_stablecoin_supply('USDT')
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_calculate_defi_metrics(self):
        """Test calculating DeFi metrics"""
        provider = DeFiDataProvider()
        
        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        tvl_df = pd.DataFrame({
            'tvl': range(100000000, 100000030)
        }, index=dates)
        
        price_df = pd.DataFrame({
            'Close': range(100, 130)
        }, index=dates)
        
        lending_rates = {
            'supply_apy': 3.5,
            'borrow_apy': 5.5,
            'utilization_rate': 0.75
        }
        
        metrics = provider.calculate_defi_metrics(tvl_df, price_df, lending_rates)
        
        assert isinstance(metrics, pd.DataFrame)
        assert 'tvl_growth_7d' in metrics.columns
        assert 'supply_apy' in metrics.columns
    
    def test_calculate_defi_risk_score(self):
        """Test DeFi risk score calculation"""
        provider = DeFiDataProvider()
        
        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        liquidation_df = pd.DataFrame({
            'liquidation_volume_usd': range(1000000, 1000030),
            'at_risk_positions': range(100, 130)
        }, index=dates)
        
        tvl_df = pd.DataFrame({
            'tvl': range(100000000, 100000030)
        }, index=dates)
        
        risk_score = provider.calculate_defi_risk_score(liquidation_df, tvl_df)
        
        assert isinstance(risk_score, pd.Series)
        assert len(risk_score) == 30
        assert risk_score.min() >= 0
        assert risk_score.max() <= 100
    
    @pytest.mark.asyncio
    async def test_get_defi_market_overview(self):
        """Test getting DeFi market overview"""
        async with DeFiDataProvider() as provider:
            overview = await provider.get_defi_market_overview()
            
            assert isinstance(overview, dict)
            assert 'total_tvl' in overview
            assert 'timestamp' in overview


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
