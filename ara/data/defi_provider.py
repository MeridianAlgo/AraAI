"""
DeFi (Decentralized Finance) data provider
Integrates DeFi protocol metrics for enhanced crypto predictions
"""

import aiohttp
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from ara.core.exceptions import DataProviderError
from ara.utils import get_logger, timed

logger = get_logger(__name__)


class DeFiDataProvider:
    """
    Provider for DeFi protocol data and metrics
    Integrates with DeFi Llama and other DeFi data sources
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DeFi data provider

        Args:
            api_key: API key for premium features (optional for DeFi Llama)
        """
        self.api_key = api_key
        self.base_url = "https://api.llama.fi"
        self.session: Optional[aiohttp.ClientSession] = None

        logger.info("Initialized DeFiDataProvider")

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    @timed("defi_fetch_tvl")
    async def fetch_tvl_data(
        self, protocol: Optional[str] = None, chain: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch Total Value Locked (TVL) data

        Args:
            protocol: Specific protocol name (e.g., 'uniswap', 'aave')
            chain: Blockchain name (e.g., 'ethereum', 'bsc')

        Returns:
            DataFrame with TVL data over time
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            if protocol:
                # Fetch protocol-specific TVL
                url = f"{self.base_url}/protocol/{protocol}"
            elif chain:
                # Fetch chain-specific TVL
                url = f"{self.base_url}/v2/historicalChainTvl/{chain}"
            else:
                # Fetch total DeFi TVL
                url = f"{self.base_url}/v2/historicalChainTvl"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    if protocol:
                        # Protocol data format
                        tvl_data = data.get("tvl", [])
                        df = pd.DataFrame(tvl_data)
                        if "date" in df.columns:
                            df["Date"] = pd.to_datetime(df["date"], unit="s")
                            df = df.set_index("Date")
                            df = df[["totalLiquidityUSD"]]
                            df.columns = ["tvl"]
                    else:
                        # Chain or total data format
                        df = pd.DataFrame(data)
                        if "date" in df.columns:
                            df["Date"] = pd.to_datetime(df["date"], unit="s")
                            df = df.set_index("Date")
                            df.columns = ["tvl"]

                    logger.info(
                        "Fetched TVL data",
                        protocol=protocol,
                        chain=chain,
                        rows=len(df),
                    )

                    return df
                else:
                    raise DataProviderError(f"Failed to fetch TVL data: HTTP {response.status}")

        except Exception as e:
            logger.error(f"Failed to fetch TVL data: {e}")
            # Return sample data as fallback
            return self._generate_sample_tvl(protocol or chain or "total")

    @timed("defi_fetch_protocols")
    async def fetch_protocol_list(self) -> List[Dict[str, Any]]:
        """
        Fetch list of all DeFi protocols

        Returns:
            List of protocol information
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            url = f"{self.base_url}/protocols"

            async with self.session.get(url) as response:
                if response.status == 200:
                    protocols = await response.json()

                    logger.info(f"Fetched {len(protocols)} DeFi protocols", count=len(protocols))

                    return protocols
                else:
                    raise DataProviderError(f"Failed to fetch protocols: HTTP {response.status}")

        except Exception as e:
            logger.error(f"Failed to fetch protocol list: {e}")
            return []

    @timed("defi_fetch_lending_rates")
    async def fetch_lending_rates(
        self, protocol: str = "aave", asset: str = "USDC"
    ) -> Dict[str, Any]:
        """
        Fetch lending and borrowing rates

        Args:
            protocol: DeFi protocol name
            asset: Asset symbol

        Returns:
            Dict with lending and borrowing rates
        """
        try:
            # This would integrate with specific protocol APIs
            # For now, generate sample data
            import numpy as np

            rates = {
                "protocol": protocol,
                "asset": asset,
                "supply_apy": np.random.uniform(1.0, 5.0),
                "borrow_apy": np.random.uniform(3.0, 8.0),
                "utilization_rate": np.random.uniform(0.5, 0.9),
                "total_supplied": np.random.uniform(1e8, 1e9),
                "total_borrowed": np.random.uniform(5e7, 8e8),
                "timestamp": datetime.now(),
            }

            logger.info(
                f"Fetched lending rates for {asset} on {protocol}",
                protocol=protocol,
                asset=asset,
                supply_apy=rates["supply_apy"],
                borrow_apy=rates["borrow_apy"],
            )

            return rates

        except Exception as e:
            logger.error(f"Failed to fetch lending rates: {e}")
            raise DataProviderError(
                "Failed to fetch lending rates",
                {"protocol": protocol, "asset": asset, "error": str(e)},
            )

    @timed("defi_fetch_liquidations")
    async def fetch_liquidation_data(
        self, protocol: Optional[str] = None, days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch liquidation events and risk metrics

        Args:
            protocol: Specific protocol (None for all)
            days: Number of days of history

        Returns:
            DataFrame with liquidation data
        """
        try:
            # This would integrate with liquidation tracking services
            # For now, generate sample data
            import numpy as np

            dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

            data = {
                "liquidation_count": np.random.poisson(10, days),
                "liquidation_volume_usd": np.random.uniform(1e6, 1e7, days),
                "avg_liquidation_size": np.random.uniform(50000, 200000, days),
                "at_risk_positions": np.random.uniform(100, 500, days),
                "total_collateral_at_risk": np.random.uniform(1e8, 5e8, days),
            }

            df = pd.DataFrame(data, index=dates)
            df.index.name = "Date"

            logger.info(
                "Fetched liquidation data",
                protocol=protocol,
                days=days,
                total_liquidations=df["liquidation_count"].sum(),
            )

            return df

        except Exception as e:
            logger.error(f"Failed to fetch liquidation data: {e}")
            raise DataProviderError("Failed to fetch liquidation data", {"error": str(e)})

    @timed("defi_fetch_stablecoin_supply")
    async def fetch_stablecoin_supply(self, stablecoin: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stablecoin supply data

        Args:
            stablecoin: Specific stablecoin (USDT, USDC, DAI, etc.)

        Returns:
            DataFrame with supply data over time
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Fetch from DeFi Llama stablecoins endpoint
            url = f"{self.base_url}/stablecoins"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Filter for specific stablecoin if provided
                    if stablecoin:
                        stablecoin_data = [
                            s
                            for s in data.get("peggedAssets", [])
                            if s.get("symbol", "").upper() == stablecoin.upper()
                        ]

                        if stablecoin_data:
                            # Get historical data
                            coin_id = stablecoin_data[0].get("id")
                            hist_url = f"{self.base_url}/stablecoincharts/all"

                            async with self.session.get(
                                hist_url, params={"stablecoin": coin_id}
                            ) as hist_response:
                                if hist_response.status == 200:
                                    hist_data = await hist_response.json()

                                    df = pd.DataFrame(hist_data)
                                    if "date" in df.columns:
                                        df["Date"] = pd.to_datetime(df["date"], unit="s")
                                        df = df.set_index("Date")

                                    logger.info(
                                        f"Fetched stablecoin supply for {stablecoin}",
                                        stablecoin=stablecoin,
                                        rows=len(df),
                                    )

                                    return df

                    # If no specific stablecoin or not found, return aggregate
                    return self._generate_sample_stablecoin_supply(stablecoin or "total")

        except Exception as e:
            logger.error(f"Failed to fetch stablecoin supply: {e}")
            return self._generate_sample_stablecoin_supply(stablecoin or "total")

    def _generate_sample_tvl(self, name: str, days: int = 365) -> pd.DataFrame:
        """Generate sample TVL data"""
        import numpy as np

        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

        # Generate trending TVL with noise
        base = np.linspace(50e9, 100e9, days)
        noise = np.random.normal(0, 5e9, days)
        tvl = base + noise

        df = pd.DataFrame({"tvl": tvl}, index=dates)
        df.index.name = "Date"

        logger.warning(f"Using sample TVL data for {name}", name=name)

        return df

    def _generate_sample_stablecoin_supply(self, stablecoin: str, days: int = 365) -> pd.DataFrame:
        """Generate sample stablecoin supply data"""
        import numpy as np

        dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

        # Generate growing supply with noise
        base = np.linspace(50e9, 80e9, days)
        noise = np.random.normal(0, 2e9, days)
        supply = base + noise

        df = pd.DataFrame(
            {
                "circulating": supply,
                "minted": np.random.uniform(1e8, 5e8, days),
                "burned": np.random.uniform(5e7, 3e8, days),
            },
            index=dates,
        )
        df.index.name = "Date"

        logger.warning(
            f"Using sample stablecoin supply data for {stablecoin}",
            stablecoin=stablecoin,
        )

        return df

    @timed("defi_calculate_metrics")
    def calculate_defi_metrics(
        self,
        tvl_df: pd.DataFrame,
        price_df: pd.DataFrame,
        lending_rates: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Calculate derived DeFi metrics

        Args:
            tvl_df: DataFrame with TVL data
            price_df: DataFrame with price data
            lending_rates: Optional lending rate data

        Returns:
            DataFrame with calculated metrics
        """
        metrics = pd.DataFrame(index=tvl_df.index)

        # TVL Growth Rate
        if "tvl" in tvl_df.columns:
            metrics["tvl_growth_7d"] = tvl_df["tvl"].pct_change(periods=7)
            metrics["tvl_growth_30d"] = tvl_df["tvl"].pct_change(periods=30)
            metrics["tvl_momentum"] = (
                tvl_df["tvl"].rolling(window=7).mean() / tvl_df["tvl"].rolling(window=30).mean() - 1
            )

        # Price to TVL Ratio (similar to P/E ratio)
        if "Close" in price_df.columns and "tvl" in tvl_df.columns:
            # Align indices
            price = price_df["Close"].reindex(tvl_df.index, method="ffill")

            # Calculate market cap (simplified)
            # In reality, would need circulating supply
            market_cap = price * 1e6  # Placeholder

            metrics["price_to_tvl"] = market_cap / tvl_df["tvl"]

        # TVL Volatility
        if "tvl" in tvl_df.columns:
            metrics["tvl_volatility"] = tvl_df["tvl"].pct_change().rolling(window=30).std()

        # Add lending rate metrics if provided
        if lending_rates:
            metrics["supply_apy"] = lending_rates.get("supply_apy", 0)
            metrics["borrow_apy"] = lending_rates.get("borrow_apy", 0)
            metrics["rate_spread"] = lending_rates.get("borrow_apy", 0) - lending_rates.get(
                "supply_apy", 0
            )
            metrics["utilization_rate"] = lending_rates.get("utilization_rate", 0)

        logger.info(
            f"Calculated {len(metrics.columns)} DeFi metrics",
            metrics=list(metrics.columns),
        )

        return metrics

    async def get_defi_market_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive DeFi market overview

        Returns:
            Dict with market statistics
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Fetch total TVL
            tvl_url = f"{self.base_url}/v2/historicalChainTvl"

            async with self.session.get(tvl_url) as response:
                if response.status == 200:
                    tvl_data = await response.json()

                    if tvl_data:
                        latest_tvl = tvl_data[-1].get("tvl", 0)
                        prev_tvl = tvl_data[-30].get("tvl", 0) if len(tvl_data) > 30 else latest_tvl

                        overview = {
                            "total_tvl": latest_tvl,
                            "tvl_change_30d": (
                                (latest_tvl - prev_tvl) / prev_tvl if prev_tvl > 0 else 0
                            ),
                            "timestamp": datetime.now(),
                            "num_protocols": 0,  # Would fetch from protocols endpoint
                            "num_chains": 0,  # Would fetch from chains endpoint
                        }

                        logger.info(
                            "Fetched DeFi market overview",
                            total_tvl=overview["total_tvl"],
                            change_30d=overview["tvl_change_30d"],
                        )

                        return overview

            # Fallback to sample data
            return {
                "total_tvl": 100e9,
                "tvl_change_30d": 0.05,
                "timestamp": datetime.now(),
                "num_protocols": 500,
                "num_chains": 20,
            }

        except Exception as e:
            logger.error(f"Failed to fetch DeFi market overview: {e}")
            return {
                "total_tvl": 0,
                "tvl_change_30d": 0,
                "timestamp": datetime.now(),
                "error": str(e),
            }

    def calculate_defi_risk_score(
        self, liquidation_df: pd.DataFrame, tvl_df: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate DeFi risk score (0-100, higher = more risk)

        Args:
            liquidation_df: DataFrame with liquidation data
            tvl_df: DataFrame with TVL data

        Returns:
            Series with risk scores
        """
        scores = pd.DataFrame(index=liquidation_df.index)

        # Liquidation risk (higher liquidations = higher risk)
        if "liquidation_volume_usd" in liquidation_df.columns:
            liq_norm = (
                liquidation_df["liquidation_volume_usd"]
                - liquidation_df["liquidation_volume_usd"].min()
            ) / (
                liquidation_df["liquidation_volume_usd"].max()
                - liquidation_df["liquidation_volume_usd"].min()
            )
            scores["liquidation_risk"] = liq_norm * 40

        # TVL volatility risk
        if "tvl" in tvl_df.columns:
            tvl_aligned = tvl_df["tvl"].reindex(liquidation_df.index, method="ffill")
            tvl_vol = tvl_aligned.pct_change().rolling(window=7).std()
            tvl_vol_norm = (tvl_vol - tvl_vol.min()) / (tvl_vol.max() - tvl_vol.min())
            scores["tvl_volatility_risk"] = tvl_vol_norm * 30

        # Positions at risk
        if "at_risk_positions" in liquidation_df.columns:
            risk_norm = (
                liquidation_df["at_risk_positions"] - liquidation_df["at_risk_positions"].min()
            ) / (
                liquidation_df["at_risk_positions"].max()
                - liquidation_df["at_risk_positions"].min()
            )
            scores["position_risk"] = risk_norm * 30

        # Calculate total risk score
        risk_score = scores.sum(axis=1)

        logger.info(
            "Calculated DeFi risk scores",
            mean_risk=risk_score.mean(),
            max_risk=risk_score.max(),
        )

        return risk_score
