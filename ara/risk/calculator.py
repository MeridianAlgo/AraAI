"""
Risk Calculator

Implements comprehensive risk metrics including VaR, CVaR, and correlation analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from scipy import stats


class RiskCalculator:
    """
    Calculate various risk metrics for financial assets and portfolios.

    This class provides methods for:
    - Value at Risk (VaR) calculation
    - Conditional Value at Risk (CVaR) calculation
    - Correlation matrix calculation
    - Risk decomposition analysis
    """

    def __init__(self):
        """Initialize the RiskCalculator."""
        pass

    def calculate_var(
        self,
        returns: Union[np.ndarray, pd.Series],
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Value at Risk (VaR) at specified confidence level.

        VaR represents the maximum expected loss over a given time period
        at a specified confidence level.

        Args:
            returns: Array or Series of historical returns
            confidence_level: Confidence level (0.95 for 95%, 0.99 for 99%)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            VaR value (positive number representing potential loss)

        Example:
            >>> calculator = RiskCalculator()
            >>> returns = np.random.normal(0.001, 0.02, 1000)
            >>> var_95 = calculator.calculate_var(returns, confidence_level=0.95)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")

        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")

        if method == "historical":
            # Historical VaR: Use empirical quantile
            var = -np.percentile(returns, (1 - confidence_level) * 100)

        elif method == "parametric":
            # Parametric VaR: Assume normal distribution
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean + z_score * std)

        elif method == "monte_carlo":
            # Monte Carlo VaR: Simulate returns
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            simulated_returns = np.random.normal(mean, std, 10000)
            var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)

        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'historical', 'parametric', or 'monte_carlo'"
            )

        return float(var)

    def calculate_cvar(
        self,
        returns: Union[np.ndarray, pd.Series],
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.

        CVaR represents the expected loss given that the loss exceeds VaR.
        It provides a more comprehensive risk measure than VaR.

        Args:
            returns: Array or Series of historical returns
            confidence_level: Confidence level (0.95 for 95%, 0.99 for 99%)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            CVaR value (positive number representing expected loss beyond VaR)

        Example:
            >>> calculator = RiskCalculator()
            >>> returns = np.random.normal(0.001, 0.02, 1000)
            >>> cvar_95 = calculator.calculate_cvar(returns, confidence_level=0.95)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")

        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")

        if method == "historical":
            # Historical CVaR: Average of losses beyond VaR
            var = self.calculate_var(returns, confidence_level, method="historical")
            # Get all returns worse than VaR
            tail_losses = returns[returns <= -var]
            if len(tail_losses) == 0:
                # If no losses exceed VaR, return VaR itself
                cvar = var
            else:
                cvar = -np.mean(tail_losses)

        elif method == "parametric":
            # Parametric CVaR: Analytical formula for normal distribution
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            z_score = stats.norm.ppf(1 - confidence_level)
            # CVaR formula for normal distribution
            cvar = -(mean - std * stats.norm.pdf(z_score) / (1 - confidence_level))

        elif method == "monte_carlo":
            # Monte Carlo CVaR: Simulate and calculate
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            simulated_returns = np.random.normal(mean, std, 10000)
            var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
            tail_losses = simulated_returns[simulated_returns <= -var]
            cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else var

        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'historical', 'parametric', or 'monte_carlo'"
            )

        return float(cvar)

    def calculate_correlation_matrix(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets.

        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            DataFrame containing correlation matrix

        Example:
            >>> calculator = RiskCalculator()
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 100),
            ...     'MSFT': np.random.normal(0.001, 0.02, 100),
            ...     'GOOGL': np.random.normal(0.001, 0.02, 100)
            ... }
            >>> corr_matrix = calculator.calculate_correlation_matrix(returns)
        """
        if not returns_dict:
            raise ValueError("Returns dictionary cannot be empty")

        # Convert to DataFrame
        df = pd.DataFrame(returns_dict)

        # Calculate correlation matrix
        if method == "pearson":
            corr_matrix = df.corr(method="pearson")
        elif method == "spearman":
            corr_matrix = df.corr(method="spearman")
        elif method == "kendall":
            corr_matrix = df.corr(method="kendall")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'pearson', 'spearman', or 'kendall'")

        return corr_matrix

    def calculate_risk_decomposition(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform risk decomposition analysis for a portfolio.

        This method calculates how much each asset contributes to the total
        portfolio risk, considering correlations between assets.

        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            weights: Dictionary mapping asset names to portfolio weights
                    If None, assumes equal weights

        Returns:
            Dictionary containing risk decomposition metrics:
            - 'marginal_risk': Marginal contribution to risk
            - 'component_risk': Component contribution to risk
            - 'percent_contribution': Percentage contribution to total risk
            - 'portfolio_volatility': Total portfolio volatility

        Example:
            >>> calculator = RiskCalculator()
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 100),
            ...     'MSFT': np.random.normal(0.001, 0.02, 100)
            ... }
            >>> weights = {'AAPL': 0.6, 'MSFT': 0.4}
            >>> decomp = calculator.calculate_risk_decomposition(returns, weights)
        """
        if not returns_dict:
            raise ValueError("Returns dictionary cannot be empty")

        assets = list(returns_dict.keys())

        # Set equal weights if not provided
        if weights is None:
            weights = {asset: 1.0 / len(assets) for asset in assets}

        # Validate weights
        if set(weights.keys()) != set(assets):
            raise ValueError("Weights must be provided for all assets")

        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")

        # Convert to DataFrame and calculate covariance matrix
        df = pd.DataFrame(returns_dict)
        cov_matrix = df.cov()

        # Convert weights to array (in same order as DataFrame columns)
        weight_array = np.array([weights[asset] for asset in df.columns])

        # Calculate portfolio variance and volatility
        portfolio_variance = np.dot(weight_array, np.dot(cov_matrix, weight_array))
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Calculate marginal risk contribution (derivative of portfolio variance w.r.t. weights)
        marginal_risk = np.dot(cov_matrix, weight_array) / portfolio_volatility

        # Calculate component risk contribution
        component_risk = weight_array * marginal_risk

        # Calculate percentage contribution
        percent_contribution = component_risk / portfolio_volatility * 100

        # Build result dictionary
        result = {"portfolio_volatility": float(portfolio_volatility), "assets": {}}

        for i, asset in enumerate(df.columns):
            result["assets"][asset] = {
                "weight": float(weights[asset]),
                "volatility": float(np.sqrt(cov_matrix.iloc[i, i])),
                "marginal_risk": float(marginal_risk[i]),
                "component_risk": float(component_risk[i]),
                "percent_contribution": float(percent_contribution[i]),
            }

        return result

    def calculate_portfolio_var(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        weights: Dict[str, float],
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate portfolio-level Value at Risk.

        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            weights: Dictionary mapping asset names to portfolio weights
            confidence_level: Confidence level (0.95 for 95%, 0.99 for 99%)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            Portfolio VaR value

        Example:
            >>> calculator = RiskCalculator()
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 100),
            ...     'MSFT': np.random.normal(0.001, 0.02, 100)
            ... }
            >>> weights = {'AAPL': 0.6, 'MSFT': 0.4}
            >>> var = calculator.calculate_portfolio_var(returns, weights, 0.95)
        """
        if not returns_dict:
            raise ValueError("Returns dictionary cannot be empty")

        # Validate weights
        if set(weights.keys()) != set(returns_dict.keys()):
            raise ValueError("Weights must be provided for all assets")

        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")

        # Calculate portfolio returns
        df = pd.DataFrame(returns_dict)
        weight_array = np.array([weights[asset] for asset in df.columns])
        portfolio_returns = np.dot(df.values, weight_array)

        # Calculate VaR on portfolio returns
        return self.calculate_var(portfolio_returns, confidence_level, method)

    def calculate_portfolio_cvar(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        weights: Dict[str, float],
        confidence_level: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Calculate portfolio-level Conditional Value at Risk.

        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            weights: Dictionary mapping asset names to portfolio weights
            confidence_level: Confidence level (0.95 for 95%, 0.99 for 99%)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            Portfolio CVaR value

        Example:
            >>> calculator = RiskCalculator()
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 100),
            ...     'MSFT': np.random.normal(0.001, 0.02, 100)
            ... }
            >>> weights = {'AAPL': 0.6, 'MSFT': 0.4}
            >>> cvar = calculator.calculate_portfolio_cvar(returns, weights, 0.95)
        """
        if not returns_dict:
            raise ValueError("Returns dictionary cannot be empty")

        # Validate weights
        if set(weights.keys()) != set(returns_dict.keys()):
            raise ValueError("Weights must be provided for all assets")

        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")

        # Calculate portfolio returns
        df = pd.DataFrame(returns_dict)
        weight_array = np.array([weights[asset] for asset in df.columns])
        portfolio_returns = np.dot(df.values, weight_array)

        # Calculate CVaR on portfolio returns
        return self.calculate_cvar(portfolio_returns, confidence_level, method)
