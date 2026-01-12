"""
Portfolio Constraints and Rebalancing

Implements portfolio constraints, transaction cost modeling, and rebalancing logic.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class RebalanceFrequency(Enum):
    """Rebalancing frequency options."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    THRESHOLD = "threshold"  # Rebalance when drift exceeds threshold


@dataclass
class Trade:
    """Represents a single trade for rebalancing."""

    symbol: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    current_price: float
    current_weight: float
    target_weight: float
    weight_change: float
    transaction_cost: float


@dataclass
class RebalanceResult:
    """Results from a rebalancing operation."""

    trades: List[Trade]
    total_transaction_cost: float
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    drift: Dict[str, float]
    max_drift: float
    rebalance_date: datetime
    portfolio_value: float


class PortfolioConstraints:
    """
    Manage portfolio constraints including position limits and sector exposure.

    This class provides methods for:
    - Position size constraints (min/max weights)
    - Sector exposure limits
    - Asset class limits
    - Concentration limits
    """

    def __init__(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        sector_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        asset_class_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        max_concentration: Optional[float] = None,
    ):
        """
        Initialize PortfolioConstraints.

        Args:
            min_weight: Minimum weight per asset (default: 0.0)
            max_weight: Maximum weight per asset (default: 1.0)
            sector_limits: Dictionary mapping sector names to (min, max) weight tuples
            asset_class_limits: Dictionary mapping asset classes to (min, max) weight tuples
            max_concentration: Maximum weight for top N assets combined
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.sector_limits = sector_limits or {}
        self.asset_class_limits = asset_class_limits or {}
        self.max_concentration = max_concentration

    def validate_weights(
        self,
        weights: Dict[str, float],
        asset_sectors: Optional[Dict[str, str]] = None,
        asset_classes: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate portfolio weights against constraints.

        Args:
            weights: Dictionary mapping asset names to weights
            asset_sectors: Optional dictionary mapping assets to sectors
            asset_classes: Optional dictionary mapping assets to asset classes

        Returns:
            Tuple of (is_valid, list_of_violations)

        Example:
            >>> constraints = PortfolioConstraints(min_weight=0.05, max_weight=0.30)
            >>> weights = {'AAPL': 0.35, 'MSFT': 0.40, 'GOOGL': 0.25}
            >>> is_valid, violations = constraints.validate_weights(weights)
        """
        violations = []

        # Check if weights sum to 1
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-4):
            violations.append(f"Weights sum to {total_weight:.4f}, not 1.0")

        # Check individual asset constraints
        for asset, weight in weights.items():
            if weight < self.min_weight - 1e-6:
                violations.append(
                    f"{asset} weight {weight:.4f} below minimum {self.min_weight:.4f}"
                )
            if weight > self.max_weight + 1e-6:
                violations.append(
                    f"{asset} weight {weight:.4f} above maximum {self.max_weight:.4f}"
                )

        # Check sector constraints
        if asset_sectors and self.sector_limits:
            sector_weights = {}
            for asset, weight in weights.items():
                if asset in asset_sectors:
                    sector = asset_sectors[asset]
                    sector_weights[sector] = sector_weights.get(sector, 0.0) + weight

            for sector, (min_limit, max_limit) in self.sector_limits.items():
                sector_weight = sector_weights.get(sector, 0.0)
                if sector_weight < min_limit - 1e-6:
                    violations.append(
                        f"Sector {sector} weight {sector_weight:.4f} below minimum {min_limit:.4f}"
                    )
                if sector_weight > max_limit + 1e-6:
                    violations.append(
                        f"Sector {sector} weight {sector_weight:.4f} above maximum {max_limit:.4f}"
                    )

        # Check asset class constraints
        if asset_classes and self.asset_class_limits:
            class_weights = {}
            for asset, weight in weights.items():
                if asset in asset_classes:
                    asset_class = asset_classes[asset]
                    class_weights[asset_class] = (
                        class_weights.get(asset_class, 0.0) + weight
                    )

            for asset_class, (min_limit, max_limit) in self.asset_class_limits.items():
                class_weight = class_weights.get(asset_class, 0.0)
                if class_weight < min_limit - 1e-6:
                    violations.append(
                        f"Asset class {asset_class} weight {class_weight:.4f} below minimum {min_limit:.4f}"
                    )
                if class_weight > max_limit + 1e-6:
                    violations.append(
                        f"Asset class {asset_class} weight {class_weight:.4f} above maximum {max_limit:.4f}"
                    )

        # Check concentration limit
        if self.max_concentration is not None:
            sorted_weights = sorted(weights.values(), reverse=True)
            # Check top 3 assets
            top_3_weight = sum(sorted_weights[:3])
            if top_3_weight > self.max_concentration + 1e-6:
                violations.append(
                    f"Top 3 assets concentration {top_3_weight:.4f} exceeds limit {self.max_concentration:.4f}"
                )

        is_valid = len(violations) == 0
        return is_valid, violations

    def apply_constraints(
        self,
        weights: Dict[str, float],
        asset_sectors: Optional[Dict[str, str]] = None,
        asset_classes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Apply constraints to weights, adjusting them to be valid.

        Args:
            weights: Dictionary mapping asset names to weights
            asset_sectors: Optional dictionary mapping assets to sectors
            asset_classes: Optional dictionary mapping assets to asset classes

        Returns:
            Adjusted weights that satisfy constraints

        Example:
            >>> constraints = PortfolioConstraints(min_weight=0.05, max_weight=0.30)
            >>> weights = {'AAPL': 0.35, 'MSFT': 0.40, 'GOOGL': 0.25}
            >>> adjusted = constraints.apply_constraints(weights)
        """
        adjusted_weights = weights.copy()

        # Apply individual asset constraints
        for asset in adjusted_weights:
            adjusted_weights[asset] = np.clip(
                adjusted_weights[asset], self.min_weight, self.max_weight
            )

        # Normalize to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {
                asset: w / total_weight for asset, w in adjusted_weights.items()
            }

        # Check and adjust sector constraints (iterative approach)
        if asset_sectors and self.sector_limits:
            for _ in range(10):  # Max 10 iterations
                sector_weights = {}
                for asset, weight in adjusted_weights.items():
                    if asset in asset_sectors:
                        sector = asset_sectors[asset]
                        sector_weights[sector] = (
                            sector_weights.get(sector, 0.0) + weight
                        )

                # Adjust if sector limits violated
                adjustments_made = False
                for sector, (min_limit, max_limit) in self.sector_limits.items():
                    sector_weight = sector_weights.get(sector, 0.0)

                    if sector_weight > max_limit:
                        # Reduce weights proportionally
                        scale_factor = max_limit / sector_weight
                        for asset, weight in adjusted_weights.items():
                            if asset_sectors.get(asset) == sector:
                                adjusted_weights[asset] *= scale_factor
                        adjustments_made = True

                    elif sector_weight < min_limit and sector_weight > 0:
                        # Increase weights proportionally
                        scale_factor = min_limit / sector_weight
                        for asset, weight in adjusted_weights.items():
                            if asset_sectors.get(asset) == sector:
                                adjusted_weights[asset] *= scale_factor
                        adjustments_made = True

                if not adjustments_made:
                    break

                # Renormalize
                total_weight = sum(adjusted_weights.values())
                if total_weight > 0:
                    adjusted_weights = {
                        asset: w / total_weight for asset, w in adjusted_weights.items()
                    }

        return adjusted_weights

    def to_dict(self) -> Dict:
        """Convert constraints to dictionary for serialization."""
        return {
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "sector_limits": self.sector_limits,
            "asset_class_limits": self.asset_class_limits,
            "max_concentration": self.max_concentration,
        }


class TransactionCostModel:
    """
    Model transaction costs for portfolio rebalancing.

    This class calculates costs including:
    - Commission fees
    - Bid-ask spread
    - Market impact
    - Slippage
    """

    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1% per trade
        spread_rate: float = 0.0005,  # 0.05% bid-ask spread
        market_impact_rate: float = 0.0001,  # 0.01% market impact
        min_commission: float = 1.0,  # Minimum commission per trade
    ):
        """
        Initialize TransactionCostModel.

        Args:
            commission_rate: Commission as fraction of trade value
            spread_rate: Bid-ask spread as fraction of price
            market_impact_rate: Market impact as fraction of trade value
            min_commission: Minimum commission per trade in dollars
        """
        self.commission_rate = commission_rate
        self.spread_rate = spread_rate
        self.market_impact_rate = market_impact_rate
        self.min_commission = min_commission

    def calculate_trade_cost(self, trade_value: float, is_buy: bool = True) -> float:
        """
        Calculate total transaction cost for a single trade.

        Args:
            trade_value: Dollar value of the trade
            is_buy: Whether this is a buy (True) or sell (False) trade

        Returns:
            Total transaction cost in dollars

        Example:
            >>> cost_model = TransactionCostModel()
            >>> cost = cost_model.calculate_trade_cost(10000, is_buy=True)
        """
        if trade_value <= 0:
            return 0.0

        # Commission
        commission = max(trade_value * self.commission_rate, self.min_commission)

        # Bid-ask spread (pay spread on both buy and sell)
        spread_cost = trade_value * self.spread_rate

        # Market impact (larger trades have more impact)
        market_impact = trade_value * self.market_impact_rate

        total_cost = commission + spread_cost + market_impact

        return float(total_cost)

    def calculate_rebalance_cost(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total cost to rebalance from current to target weights.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value in dollars
            prices: Current prices for each asset

        Returns:
            Tuple of (total_cost, cost_per_asset)

        Example:
            >>> cost_model = TransactionCostModel()
            >>> current = {'AAPL': 0.4, 'MSFT': 0.6}
            >>> target = {'AAPL': 0.5, 'MSFT': 0.5}
            >>> prices = {'AAPL': 150.0, 'MSFT': 300.0}
            >>> total_cost, costs = cost_model.calculate_rebalance_cost(
            ...     current, target, 100000, prices
            ... )
        """
        total_cost = 0.0
        cost_per_asset = {}

        # Get all assets
        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        for asset in all_assets:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)

            # Calculate weight change
            weight_change = target_weight - current_weight

            if abs(weight_change) < 1e-6:
                cost_per_asset[asset] = 0.0
                continue

            # Calculate trade value
            trade_value = abs(weight_change) * portfolio_value

            # Calculate cost
            is_buy = weight_change > 0
            cost = self.calculate_trade_cost(trade_value, is_buy)

            cost_per_asset[asset] = cost
            total_cost += cost

        return float(total_cost), cost_per_asset


class PortfolioRebalancer:
    """
    Manage portfolio rebalancing with various strategies and schedules.

    This class provides methods for:
    - Scheduled rebalancing (daily, weekly, monthly, etc.)
    - Threshold-based rebalancing
    - Transaction cost-aware rebalancing
    - Tax-aware rebalancing (optional)
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        constraints: Optional[PortfolioConstraints] = None,
    ):
        """
        Initialize PortfolioRebalancer.

        Args:
            cost_model: Transaction cost model (default: standard model)
            constraints: Portfolio constraints (default: no constraints)
        """
        self.cost_model = cost_model or TransactionCostModel()
        self.constraints = constraints or PortfolioConstraints()
        self.last_rebalance_date = None

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        current_date: datetime,
        frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
        drift_threshold: float = 0.05,
    ) -> bool:
        """
        Determine if portfolio should be rebalanced.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            current_date: Current date
            frequency: Rebalancing frequency
            drift_threshold: Maximum allowed drift for threshold-based rebalancing

        Returns:
            True if rebalancing is recommended

        Example:
            >>> rebalancer = PortfolioRebalancer()
            >>> current = {'AAPL': 0.45, 'MSFT': 0.55}
            >>> target = {'AAPL': 0.5, 'MSFT': 0.5}
            >>> should_rebal = rebalancer.should_rebalance(
            ...     current, target, datetime.now(),
            ...     frequency=RebalanceFrequency.THRESHOLD,
            ...     drift_threshold=0.05
            ... )
        """
        # Check frequency-based rebalancing
        if frequency != RebalanceFrequency.THRESHOLD:
            if self.last_rebalance_date is None:
                return True

            days_since_rebalance = (current_date - self.last_rebalance_date).days

            if frequency == RebalanceFrequency.DAILY and days_since_rebalance >= 1:
                return True
            elif frequency == RebalanceFrequency.WEEKLY and days_since_rebalance >= 7:
                return True
            elif frequency == RebalanceFrequency.MONTHLY and days_since_rebalance >= 30:
                return True
            elif (
                frequency == RebalanceFrequency.QUARTERLY and days_since_rebalance >= 90
            ):
                return True
            elif (
                frequency == RebalanceFrequency.ANNUALLY and days_since_rebalance >= 365
            ):
                return True

        # Check drift-based rebalancing
        max_drift = 0.0
        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            drift = abs(current - target)
            max_drift = max(max_drift, drift)

        return max_drift >= drift_threshold

    def calculate_rebalance_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        min_trade_value: float = 100.0,
    ) -> List[Trade]:
        """
        Calculate trades needed to rebalance portfolio.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value in dollars
            prices: Current prices for each asset
            min_trade_value: Minimum trade value to execute (skip small trades)

        Returns:
            List of Trade objects

        Example:
            >>> rebalancer = PortfolioRebalancer()
            >>> current = {'AAPL': 0.4, 'MSFT': 0.6}
            >>> target = {'AAPL': 0.5, 'MSFT': 0.5}
            >>> prices = {'AAPL': 150.0, 'MSFT': 300.0}
            >>> trades = rebalancer.calculate_rebalance_trades(
            ...     current, target, 100000, prices
            ... )
        """
        trades = []

        # Get all assets
        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        for asset in all_assets:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)

            # Calculate weight change
            weight_change = target_weight - current_weight

            if abs(weight_change) < 1e-6:
                continue

            # Calculate trade value
            trade_value = abs(weight_change) * portfolio_value

            # Skip small trades
            if trade_value < min_trade_value:
                continue

            # Determine action
            action = "BUY" if weight_change > 0 else "SELL"

            # Calculate quantity
            price = prices.get(asset, 0.0)
            if price <= 0:
                continue

            quantity = trade_value / price

            # Calculate transaction cost
            is_buy = action == "BUY"
            transaction_cost = self.cost_model.calculate_trade_cost(trade_value, is_buy)

            # Create trade
            trade = Trade(
                symbol=asset,
                action=action,
                quantity=quantity,
                current_price=price,
                current_weight=current_weight,
                target_weight=target_weight,
                weight_change=weight_change,
                transaction_cost=transaction_cost,
            )

            trades.append(trade)

        return trades

    def rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        current_date: datetime,
        min_trade_value: float = 100.0,
    ) -> RebalanceResult:
        """
        Execute portfolio rebalancing.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value in dollars
            prices: Current prices for each asset
            current_date: Current date
            min_trade_value: Minimum trade value to execute

        Returns:
            RebalanceResult object with trade details

        Example:
            >>> rebalancer = PortfolioRebalancer()
            >>> current = {'AAPL': 0.4, 'MSFT': 0.6}
            >>> target = {'AAPL': 0.5, 'MSFT': 0.5}
            >>> prices = {'AAPL': 150.0, 'MSFT': 300.0}
            >>> result = rebalancer.rebalance(
            ...     current, target, 100000, prices, datetime.now()
            ... )
        """
        # Calculate trades
        trades = self.calculate_rebalance_trades(
            current_weights, target_weights, portfolio_value, prices, min_trade_value
        )

        # Calculate total transaction cost
        total_cost = sum(trade.transaction_cost for trade in trades)

        # Calculate drift
        drift = {}
        max_drift = 0.0
        for asset in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            asset_drift = abs(current - target)
            drift[asset] = asset_drift
            max_drift = max(max_drift, asset_drift)

        # Update last rebalance date
        self.last_rebalance_date = current_date

        # Create result
        result = RebalanceResult(
            trades=trades,
            total_transaction_cost=total_cost,
            current_weights=current_weights,
            target_weights=target_weights,
            drift=drift,
            max_drift=max_drift,
            rebalance_date=current_date,
            portfolio_value=portfolio_value,
        )

        return result

    def optimize_rebalance_with_costs(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        cost_threshold: float = 0.01,
    ) -> Dict[str, float]:
        """
        Optimize rebalancing considering transaction costs.

        This method finds the optimal weights that balance getting closer to
        target weights while minimizing transaction costs.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value in dollars
            prices: Current prices for each asset
            cost_threshold: Maximum acceptable cost as fraction of portfolio value

        Returns:
            Optimized target weights that balance tracking and costs

        Example:
            >>> rebalancer = PortfolioRebalancer()
            >>> current = {'AAPL': 0.4, 'MSFT': 0.6}
            >>> target = {'AAPL': 0.5, 'MSFT': 0.5}
            >>> prices = {'AAPL': 150.0, 'MSFT': 300.0}
            >>> optimized = rebalancer.optimize_rebalance_with_costs(
            ...     current, target, 100000, prices, cost_threshold=0.005
            ... )
        """
        # Calculate cost of full rebalance
        full_cost, _ = self.cost_model.calculate_rebalance_cost(
            current_weights, target_weights, portfolio_value, prices
        )

        # If cost is acceptable, use target weights
        if full_cost / portfolio_value <= cost_threshold:
            return target_weights

        # Otherwise, find partial rebalance that meets cost threshold
        # Use linear interpolation between current and target
        best_weights = current_weights.copy()

        for alpha in np.linspace(0, 1, 21):  # Test 21 points
            # Interpolate weights
            test_weights = {}
            for asset in set(current_weights.keys()) | set(target_weights.keys()):
                current = current_weights.get(asset, 0.0)
                target = target_weights.get(asset, 0.0)
                test_weights[asset] = current + alpha * (target - current)

            # Calculate cost
            test_cost, _ = self.cost_model.calculate_rebalance_cost(
                current_weights, test_weights, portfolio_value, prices
            )

            # Check if within threshold
            if test_cost / portfolio_value <= cost_threshold:
                best_weights = test_weights

        return best_weights

    def tax_aware_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float],
        cost_basis: Dict[str, float],
        tax_rate_short_term: float = 0.37,
        tax_rate_long_term: float = 0.20,
        holding_periods: Optional[Dict[str, int]] = None,
    ) -> Dict[str, float]:
        """
        Optimize rebalancing considering tax implications.

        This method adjusts target weights to minimize tax liability from
        selling appreciated assets.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value in dollars
            prices: Current prices for each asset
            cost_basis: Cost basis (purchase price) for each asset
            tax_rate_short_term: Short-term capital gains tax rate
            tax_rate_long_term: Long-term capital gains tax rate
            holding_periods: Days held for each asset (None = assume long-term)

        Returns:
            Tax-optimized target weights

        Example:
            >>> rebalancer = PortfolioRebalancer()
            >>> current = {'AAPL': 0.4, 'MSFT': 0.6}
            >>> target = {'AAPL': 0.5, 'MSFT': 0.5}
            >>> prices = {'AAPL': 150.0, 'MSFT': 300.0}
            >>> cost_basis = {'AAPL': 100.0, 'MSFT': 250.0}
            >>> optimized = rebalancer.tax_aware_rebalance(
            ...     current, target, 100000, prices, cost_basis
            ... )
        """
        if holding_periods is None:
            # Assume all holdings are long-term
            holding_periods = {asset: 366 for asset in current_weights.keys()}

        # Calculate tax liability for each potential sale
        tax_liabilities = {}

        for asset in current_weights.keys():
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights.get(asset, 0.0)

            # Only consider assets we need to sell
            if target_weight >= current_weight:
                tax_liabilities[asset] = 0.0
                continue

            # Calculate gain/loss
            current_price = prices.get(asset, 0.0)
            basis = cost_basis.get(asset, current_price)

            if current_price <= basis:
                # Loss - no tax, actually beneficial
                tax_liabilities[asset] = 0.0
                continue

            # Calculate capital gain
            gain_per_share = current_price - basis

            # Determine tax rate
            days_held = holding_periods.get(asset, 366)
            tax_rate = tax_rate_long_term if days_held > 365 else tax_rate_short_term

            # Calculate tax on potential sale
            weight_to_sell = current_weight - target_weight
            value_to_sell = weight_to_sell * portfolio_value
            shares_to_sell = value_to_sell / current_price
            total_gain = shares_to_sell * gain_per_share
            tax_liability = total_gain * tax_rate

            tax_liabilities[asset] = tax_liability

        # Adjust target weights to minimize tax
        # Prefer selling assets with lower tax liability
        adjusted_weights = target_weights.copy()

        # Sort assets by tax liability (ascending)
        sorted_assets = sorted(tax_liabilities.items(), key=lambda x: x[1])

        # Adjust weights: reduce selling of high-tax assets
        total_adjustment = 0.0
        for asset, tax_liability in sorted_assets:
            if tax_liability > 0:
                current_weight = current_weights.get(asset, 0.0)
                target_weight = target_weights.get(asset, 0.0)

                if target_weight < current_weight:
                    # Reduce the amount we sell
                    reduction = min(
                        (current_weight - target_weight) * 0.5,  # Reduce by up to 50%
                        0.05,  # Max 5% adjustment
                    )
                    adjusted_weights[asset] = target_weight + reduction
                    total_adjustment += reduction

        # Redistribute the adjustment to other assets
        if total_adjustment > 0:
            # Add to assets we're buying
            buy_assets = [
                asset
                for asset in adjusted_weights.keys()
                if adjusted_weights[asset] > current_weights.get(asset, 0.0)
            ]

            if buy_assets:
                adjustment_per_asset = total_adjustment / len(buy_assets)
                for asset in buy_assets:
                    adjusted_weights[asset] += adjustment_per_asset

        # Normalize to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {
                asset: w / total_weight for asset, w in adjusted_weights.items()
            }

        return adjusted_weights
