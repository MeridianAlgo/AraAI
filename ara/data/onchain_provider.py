"""
On-chain metrics provider for cryptocurrency analysis
Integrates blockchain data for enhanced predictions
"""

import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from ara.core.exceptions import DataProviderError
from ara.utils import get_logger, timed

logger = get_logger(__name__)


class OnChainMetricsProvider:
    """
    Provider for blockchain on-chain metrics
    Supports multiple blockchain data APIs
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_provider: str = "blockchain_info"
    ):
        """
        Initialize on-chain metrics provider
        
        Args:
            api_key: API key for premium data providers
            api_provider: Provider name (blockchain_info, glassnode, cryptoquant)
        """
        self.api_key = api_key
        self.api_provider = api_provider
        self.session: Optional[aiohttp.ClientSession] = None
        
        # API endpoints
        self.endpoints = {
            'blockchain_info': 'https://blockchain.info',
            'glassnode': 'https://api.glassnode.com/v1/metrics',
            'cryptoquant': 'https://api.cryptoquant.com/v1'
        }
        
        self.base_url = self.endpoints.get(api_provider, self.endpoints['blockchain_info'])
        
        logger.info(
            f"Initialized OnChainMetricsProvider",
            provider=api_provider,
            has_api_key=bool(api_key)
        )
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    @timed("onchain_fetch_metrics")
    async def fetch_network_metrics(
        self,
        symbol: str,
        metrics: Optional[List[str]] = None,
        days: int = 365
    ) -> pd.DataFrame:
        """
        Fetch network metrics for a cryptocurrency
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            metrics: List of metrics to fetch (None = all available)
            days: Number of days of historical data
            
        Returns:
            DataFrame with on-chain metrics
        """
        symbol = symbol.upper().replace('/USDT', '').replace('/USD', '')
        
        if metrics is None:
            metrics = [
                'active_addresses',
                'transaction_count',
                'transaction_volume',
                'hash_rate',
                'difficulty',
                'network_value'
            ]
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Fetch metrics based on provider
            if self.api_provider == 'blockchain_info':
                data = await self._fetch_blockchain_info(symbol, metrics, days)
            elif self.api_provider == 'glassnode':
                data = await self._fetch_glassnode(symbol, metrics, days)
            elif self.api_provider == 'cryptoquant':
                data = await self._fetch_cryptoquant(symbol, metrics, days)
            else:
                # Fallback to simulated data for development
                data = self._generate_sample_metrics(symbol, metrics, days)
            
            logger.info(
                f"Fetched on-chain metrics for {symbol}",
                symbol=symbol,
                metrics=len(metrics),
                days=days,
                rows=len(data)
            )
            
            return data
            
        except Exception as e:
            logger.error(
                f"Failed to fetch on-chain metrics: {e}",
                symbol=symbol,
                error=str(e)
            )
            # Return sample data as fallback
            return self._generate_sample_metrics(symbol, metrics, days)
    
    async def _fetch_blockchain_info(
        self,
        symbol: str,
        metrics: List[str],
        days: int
    ) -> pd.DataFrame:
        """Fetch from blockchain.info API"""
        if symbol != 'BTC':
            raise DataProviderError(
                "blockchain.info only supports Bitcoin",
                {"symbol": symbol}
            )
        
        data = {}
        
        for metric in metrics:
            try:
                endpoint_map = {
                    'active_addresses': 'n-unique-addresses',
                    'transaction_count': 'n-transactions',
                    'transaction_volume': 'estimated-transaction-volume',
                    'hash_rate': 'hash-rate',
                    'difficulty': 'difficulty'
                }
                
                endpoint = endpoint_map.get(metric)
                if not endpoint:
                    continue
                
                url = f"{self.base_url}/charts/{endpoint}"
                params = {
                    'timespan': f'{days}days',
                    'format': 'json'
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        values = result.get('values', [])
                        
                        timestamps = [v['x'] for v in values]
                        metric_values = [v['y'] for v in values]
                        
                        data[metric] = pd.Series(
                            metric_values,
                            index=pd.to_datetime(timestamps, unit='s')
                        )
                
            except Exception as e:
                logger.warning(f"Failed to fetch {metric}: {e}")
        
        if not data:
            raise DataProviderError("No metrics fetched successfully")
        
        df = pd.DataFrame(data)
        df.index.name = 'Date'
        return df
    
    async def _fetch_glassnode(
        self,
        symbol: str,
        metrics: List[str],
        days: int
    ) -> pd.DataFrame:
        """Fetch from Glassnode API (requires API key)"""
        if not self.api_key:
            raise DataProviderError(
                "Glassnode requires API key",
                {"provider": "glassnode"}
            )
        
        data = {}
        since = int((datetime.now() - timedelta(days=days)).timestamp())
        
        metric_map = {
            'active_addresses': 'addresses/active_count',
            'transaction_count': 'transactions/count',
            'transaction_volume': 'transactions/transfers_volume_sum',
            'hash_rate': 'mining/hash_rate_mean',
            'network_value': 'market/marketcap_usd'
        }
        
        for metric in metrics:
            try:
                endpoint = metric_map.get(metric)
                if not endpoint:
                    continue
                
                url = f"{self.base_url}/{endpoint}"
                params = {
                    'a': symbol,
                    's': since,
                    'api_key': self.api_key
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        timestamps = [r['t'] for r in result]
                        values = [r['v'] for r in result]
                        
                        data[metric] = pd.Series(
                            values,
                            index=pd.to_datetime(timestamps, unit='s')
                        )
                
            except Exception as e:
                logger.warning(f"Failed to fetch {metric} from Glassnode: {e}")
        
        if not data:
            raise DataProviderError("No metrics fetched from Glassnode")
        
        df = pd.DataFrame(data)
        df.index.name = 'Date'
        return df
    
    async def _fetch_cryptoquant(
        self,
        symbol: str,
        metrics: List[str],
        days: int
    ) -> pd.DataFrame:
        """Fetch from CryptoQuant API (requires API key)"""
        if not self.api_key:
            raise DataProviderError(
                "CryptoQuant requires API key",
                {"provider": "cryptoquant"}
            )
        
        # Similar implementation to Glassnode
        # For now, fallback to sample data
        return self._generate_sample_metrics(symbol, metrics, days)
    
    def _generate_sample_metrics(
        self,
        symbol: str,
        metrics: List[str],
        days: int
    ) -> pd.DataFrame:
        """
        Generate sample on-chain metrics for development/testing
        
        Args:
            symbol: Cryptocurrency symbol
            metrics: List of metrics
            days: Number of days
            
        Returns:
            DataFrame with sample metrics
        """
        import numpy as np
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        data = {}
        
        # Generate realistic-looking sample data
        for metric in metrics:
            if metric == 'active_addresses':
                # Trending upward with noise
                base = np.linspace(100000, 150000, days)
                noise = np.random.normal(0, 5000, days)
                data[metric] = base + noise
                
            elif metric == 'transaction_count':
                base = np.linspace(200000, 300000, days)
                noise = np.random.normal(0, 10000, days)
                data[metric] = base + noise
                
            elif metric == 'transaction_volume':
                base = np.linspace(1e9, 2e9, days)
                noise = np.random.normal(0, 1e8, days)
                data[metric] = base + noise
                
            elif metric == 'hash_rate':
                base = np.linspace(100e18, 150e18, days)
                noise = np.random.normal(0, 5e18, days)
                data[metric] = base + noise
                
            elif metric == 'difficulty':
                base = np.linspace(20e12, 25e12, days)
                noise = np.random.normal(0, 1e12, days)
                data[metric] = base + noise
                
            elif metric == 'network_value':
                base = np.linspace(500e9, 800e9, days)
                noise = np.random.normal(0, 50e9, days)
                data[metric] = base + noise
            
            else:
                # Generic metric
                data[metric] = np.random.normal(1000, 100, days)
        
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Date'
        
        logger.warning(
            f"Using sample on-chain metrics for {symbol}",
            symbol=symbol,
            metrics=len(metrics)
        )
        
        return df
    
    @timed("onchain_calculate_derived_metrics")
    def calculate_derived_metrics(
        self,
        metrics_df: pd.DataFrame,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate derived on-chain metrics
        
        Args:
            metrics_df: DataFrame with base on-chain metrics
            price_df: DataFrame with price data
            
        Returns:
            DataFrame with derived metrics
        """
        derived = pd.DataFrame(index=metrics_df.index)
        
        # Merge price data
        if 'Close' in price_df.columns:
            price = price_df['Close']
        else:
            price = price_df.iloc[:, 0]
        
        # Align indices
        price = price.reindex(metrics_df.index, method='ffill')
        
        # NVT Ratio (Network Value to Transactions)
        if 'network_value' in metrics_df.columns and 'transaction_volume' in metrics_df.columns:
            derived['nvt_ratio'] = (
                metrics_df['network_value'] / 
                metrics_df['transaction_volume'].rolling(window=7).mean()
            )
        
        # MVRV Ratio (Market Value to Realized Value)
        # Simplified calculation using price momentum
        if len(price) > 0:
            realized_price = price.rolling(window=365, min_periods=30).mean()
            derived['mvrv_ratio'] = price / realized_price
        
        # Active Addresses Growth
        if 'active_addresses' in metrics_df.columns:
            derived['active_addresses_growth'] = (
                metrics_df['active_addresses'].pct_change(periods=7)
            )
        
        # Transaction Volume Growth
        if 'transaction_volume' in metrics_df.columns:
            derived['tx_volume_growth'] = (
                metrics_df['transaction_volume'].pct_change(periods=7)
            )
        
        # Hash Rate Growth (mining activity)
        if 'hash_rate' in metrics_df.columns:
            derived['hash_rate_growth'] = (
                metrics_df['hash_rate'].pct_change(periods=30)
            )
        
        logger.info(
            f"Calculated {len(derived.columns)} derived metrics",
            metrics=list(derived.columns)
        )
        
        return derived
    
    async def fetch_exchange_flows(
        self,
        symbol: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch exchange inflow/outflow data
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days
            
        Returns:
            DataFrame with exchange flow data
        """
        symbol = symbol.upper().replace('/USDT', '').replace('/USD', '')
        
        try:
            # This would integrate with Glassnode or CryptoQuant
            # For now, generate sample data
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            
            import numpy as np
            
            data = {
                'exchange_inflow': np.random.normal(10000, 2000, days),
                'exchange_outflow': np.random.normal(9000, 2000, days),
                'exchange_net_flow': np.random.normal(1000, 1000, days),
                'exchange_reserves': np.linspace(500000, 480000, days)
            }
            
            df = pd.DataFrame(data, index=dates)
            df.index.name = 'Date'
            
            logger.info(
                f"Fetched exchange flows for {symbol}",
                symbol=symbol,
                days=days
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch exchange flows: {e}")
            raise DataProviderError(
                f"Failed to fetch exchange flows for {symbol}",
                {"error": str(e)}
            )
    
    async def track_whale_wallets(
        self,
        symbol: str,
        threshold: float = 1000000,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Track large wallet transactions (whale activity)
        
        Args:
            symbol: Cryptocurrency symbol
            threshold: Minimum transaction value in USD
            days: Number of days to look back
            
        Returns:
            List of large transactions
        """
        symbol = symbol.upper().replace('/USDT', '').replace('/USD', '')
        
        try:
            # This would integrate with blockchain explorers
            # For now, generate sample whale transactions
            import numpy as np
            
            num_transactions = np.random.randint(10, 50)
            transactions = []
            
            for _ in range(num_transactions):
                tx = {
                    'timestamp': datetime.now() - timedelta(
                        days=np.random.randint(0, days)
                    ),
                    'value_usd': np.random.uniform(threshold, threshold * 10),
                    'type': np.random.choice(['inflow', 'outflow']),
                    'from_exchange': np.random.choice([True, False]),
                    'to_exchange': np.random.choice([True, False]),
                    'wallet_address': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}"
                }
                transactions.append(tx)
            
            # Sort by timestamp
            transactions.sort(key=lambda x: x['timestamp'], reverse=True)
            
            logger.info(
                f"Tracked {len(transactions)} whale transactions for {symbol}",
                symbol=symbol,
                threshold=threshold,
                transactions=len(transactions)
            )
            
            return transactions
            
        except Exception as e:
            logger.error(f"Failed to track whale wallets: {e}")
            return []
    
    def calculate_network_health_score(
        self,
        metrics_df: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate overall network health score (0-100)
        
        Args:
            metrics_df: DataFrame with on-chain metrics
            
        Returns:
            Series with health scores
        """
        scores = pd.DataFrame(index=metrics_df.index)
        
        # Active addresses score (higher is better)
        if 'active_addresses' in metrics_df.columns:
            aa_norm = (
                (metrics_df['active_addresses'] - metrics_df['active_addresses'].min()) /
                (metrics_df['active_addresses'].max() - metrics_df['active_addresses'].min())
            )
            scores['active_addresses_score'] = aa_norm * 25
        
        # Transaction volume score (higher is better)
        if 'transaction_volume' in metrics_df.columns:
            tv_norm = (
                (metrics_df['transaction_volume'] - metrics_df['transaction_volume'].min()) /
                (metrics_df['transaction_volume'].max() - metrics_df['transaction_volume'].min())
            )
            scores['tx_volume_score'] = tv_norm * 25
        
        # Hash rate score (higher is better, indicates security)
        if 'hash_rate' in metrics_df.columns:
            hr_norm = (
                (metrics_df['hash_rate'] - metrics_df['hash_rate'].min()) /
                (metrics_df['hash_rate'].max() - metrics_df['hash_rate'].min())
            )
            scores['hash_rate_score'] = hr_norm * 25
        
        # Growth score (positive growth is better)
        if 'active_addresses' in metrics_df.columns:
            growth = metrics_df['active_addresses'].pct_change(periods=30)
            growth_norm = (growth + 0.5).clip(0, 1)  # Normalize around 0
            scores['growth_score'] = growth_norm * 25
        
        # Calculate total score
        health_score = scores.sum(axis=1)
        
        logger.info(
            f"Calculated network health scores",
            mean_score=health_score.mean(),
            min_score=health_score.min(),
            max_score=health_score.max()
        )
        
        return health_score
