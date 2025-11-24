"""
Portfolio Optimizer

Implements multiple portfolio optimization strategies including:
- Modern Portfolio Theory (MPT) with efficient frontier
- Black-Litterman model for incorporating predictions
- Risk Parity optimization
- Kelly Criterion for position sizing
- Mean-CVaR optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy import stats
import warnings


class PortfolioOptimizer:
    """
    Portfolio optimization engine with multiple strategies.
    
    This class provides methods for:
    - Modern Portfolio Theory (MPT) optimization
    - Black-Litterman model
    - Risk Parity optimization
    - Kelly Criterion position sizing
    - Mean-CVaR optimization
    - Efficient frontier calculation
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize PortfolioOptimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2% or 0.02)
        """
        self.risk_free_rate = risk_free_rate
    
    def optimize_mpt(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        objective: str = 'max_sharpe',
        constraints: Optional[Dict] = None
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Optimize portfolio using Modern Portfolio Theory (MPT).
        
        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            target_return: Target portfolio return (for min_risk objective)
            target_risk: Target portfolio risk (for max_return objective)
            objective: Optimization objective:
                - 'max_sharpe': Maximize Sharpe ratio (default)
                - 'min_risk': Minimize risk for target return
                - 'max_return': Maximize return for target risk
            constraints: Optional constraints dictionary with keys:
                - 'min_weight': Minimum weight per asset (default: 0)
                - 'max_weight': Maximum weight per asset (default: 1)
                - 'sector_limits': Dict of sector exposure limits
        
        Returns:
            Dictionary containing:
            - 'weights': Optimal portfolio weights
            - 'expected_return': Expected portfolio return
            - 'volatility': Portfolio volatility
            - 'sharpe_ratio': Portfolio Sharpe ratio
        
        Example:
            >>> optimizer = PortfolioOptimizer(risk_free_rate=0.02)
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 252),
            ...     'MSFT': np.random.normal(0.001, 0.02, 252)
            ... }
            >>> result = optimizer.optimize_mpt(returns, objective='max_sharpe')
        """
        if not returns_dict:
            raise ValueError("Returns dictionary cannot be empty")
        
        # Convert to DataFrame
        df = pd.DataFrame(returns_dict)
        assets = df.columns.tolist()
        n_assets = len(assets)
        
        # Calculate expected returns and covariance matrix
        expected_returns = df.mean().values
        cov_matrix = df.cov().values
        
        # Set up constraints
        min_weight = 0.0
        max_weight = 1.0
        if constraints:
            min_weight = constraints.get('min_weight', 0.0)
            max_weight = constraints.get('max_weight', 1.0)
        
        # Bounds for each asset
        bounds = Bounds(
            lb=np.full(n_assets, min_weight),
            ub=np.full(n_assets, max_weight)
        )
        
        # Constraint: weights sum to 1
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Add target return constraint if specified
        if objective == 'min_risk' and target_return is not None:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - target_return
            })
        
        # Add target risk constraint if specified
        if objective == 'max_return' and target_risk is not None:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.sqrt(np.dot(w, np.dot(cov_matrix, w))) - target_risk
            })
        
        # Define objective function
        if objective == 'max_sharpe':
            # Maximize Sharpe ratio (minimize negative Sharpe)
            def objective_func(w):
                port_return = np.dot(w, expected_returns)
                port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
                if port_vol == 0:
                    return 1e10
                sharpe = (port_return - self.risk_free_rate / 252) / port_vol
                return -sharpe  # Minimize negative Sharpe
        
        elif objective == 'min_risk':
            # Minimize portfolio variance
            def objective_func(w):
                return np.dot(w, np.dot(cov_matrix, w))
        
        elif objective == 'max_return':
            # Maximize return (minimize negative return)
            def objective_func(w):
                return -np.dot(w, expected_returns)
        
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Initial guess: equal weights
        w0 = np.full(n_assets, 1.0 / n_assets)
        
        # Optimize
        result = minimize(
            objective_func,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        port_return = np.dot(optimal_weights, expected_returns)
        port_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
        sharpe = (port_return - self.risk_free_rate / 252) / port_vol if port_vol > 0 else 0.0
        
        # Annualize metrics
        annual_return = (1 + port_return) ** 252 - 1
        annual_vol = port_vol * np.sqrt(252)
        annual_sharpe = sharpe * np.sqrt(252)
        
        return {
            'weights': {asset: float(w) for asset, w in zip(assets, optimal_weights)},
            'expected_return': float(annual_return),
            'volatility': float(annual_vol),
            'sharpe_ratio': float(annual_sharpe)
        }

    
    def optimize_black_litterman(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        market_caps: Dict[str, float],
        views: Dict[str, float],
        view_confidence: Optional[Dict[str, float]] = None,
        tau: float = 0.025,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Optimize portfolio using Black-Litterman model.
        
        The Black-Litterman model combines market equilibrium with investor views
        to generate expected returns that incorporate both market consensus and
        personal predictions.
        
        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            market_caps: Dictionary mapping asset names to market capitalizations
            views: Dictionary mapping asset names to expected returns (investor views)
            view_confidence: Dictionary mapping asset names to confidence levels (0-1)
                           Higher confidence = more weight on views vs market
            tau: Scaling factor for uncertainty in prior (default: 0.025)
            constraints: Optional constraints dictionary
        
        Returns:
            Dictionary containing optimal weights and portfolio metrics
        
        Example:
            >>> optimizer = PortfolioOptimizer(risk_free_rate=0.02)
            >>> returns = {'AAPL': np.random.normal(0.001, 0.02, 252),
            ...            'MSFT': np.random.normal(0.001, 0.02, 252)}
            >>> market_caps = {'AAPL': 2.5e12, 'MSFT': 2.3e12}
            >>> views = {'AAPL': 0.15, 'MSFT': 0.12}  # Expected annual returns
            >>> result = optimizer.optimize_black_litterman(returns, market_caps, views)
        """
        if not returns_dict:
            raise ValueError("Returns dictionary cannot be empty")
        
        # Convert to DataFrame
        df = pd.DataFrame(returns_dict)
        assets = df.columns.tolist()
        n_assets = len(assets)
        
        # Validate inputs
        if set(market_caps.keys()) != set(assets):
            raise ValueError("Market caps must be provided for all assets")
        if set(views.keys()) != set(assets):
            raise ValueError("Views must be provided for all assets")
        
        # Calculate covariance matrix
        cov_matrix = df.cov().values
        
        # Calculate market equilibrium weights (based on market caps)
        total_market_cap = sum(market_caps.values())
        market_weights = np.array([market_caps[asset] / total_market_cap for asset in assets])
        
        # Calculate implied equilibrium returns (reverse optimization)
        # Pi = delta * Sigma * w_mkt
        # delta = (E[R_mkt] - rf) / var_mkt (risk aversion coefficient)
        market_return = np.dot(market_weights, df.mean().values)
        market_variance = np.dot(market_weights, np.dot(cov_matrix, market_weights))
        delta = (market_return - self.risk_free_rate / 252) / market_variance if market_variance > 0 else 2.5
        
        implied_returns = delta * np.dot(cov_matrix, market_weights)
        
        # Set up views
        # P matrix: picks out assets with views (identity matrix for absolute views)
        P = np.eye(n_assets)
        
        # Q vector: view returns (convert annual to daily)
        Q = np.array([views[asset] / 252 for asset in assets])
        
        # Omega matrix: uncertainty in views
        if view_confidence is None:
            # Default: equal confidence for all views
            view_confidence = {asset: 0.5 for asset in assets}
        
        # Omega is diagonal with view variances
        # Higher confidence = lower variance
        omega_diag = []
        for asset in assets:
            confidence = view_confidence.get(asset, 0.5)
            # View variance inversely proportional to confidence
            view_var = tau * cov_matrix[assets.index(asset), assets.index(asset)] / confidence
            omega_diag.append(view_var)
        Omega = np.diag(omega_diag)
        
        # Black-Litterman formula
        # E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
        
        tau_sigma = tau * cov_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)
        
        # Posterior precision
        posterior_precision = tau_sigma_inv + np.dot(P.T, np.dot(omega_inv, P))
        posterior_cov = np.linalg.inv(posterior_precision)
        
        # Posterior expected returns
        bl_returns = np.dot(
            posterior_cov,
            np.dot(tau_sigma_inv, implied_returns) + np.dot(P.T, np.dot(omega_inv, Q))
        )
        
        # Optimize using Black-Litterman returns
        # Use mean-variance optimization with BL returns
        
        # Set up constraints
        min_weight = 0.0
        max_weight = 1.0
        if constraints:
            min_weight = constraints.get('min_weight', 0.0)
            max_weight = constraints.get('max_weight', 1.0)
        
        bounds = Bounds(
            lb=np.full(n_assets, min_weight),
            ub=np.full(n_assets, max_weight)
        )
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Maximize Sharpe ratio with BL returns
        def objective_func(w):
            port_return = np.dot(w, bl_returns)
            port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            if port_vol == 0:
                return 1e10
            sharpe = (port_return - self.risk_free_rate / 252) / port_vol
            return -sharpe
        
        # Initial guess: market weights
        w0 = market_weights
        
        # Optimize
        result = minimize(
            objective_func,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        port_return = np.dot(optimal_weights, bl_returns)
        port_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
        sharpe = (port_return - self.risk_free_rate / 252) / port_vol if port_vol > 0 else 0.0
        
        # Annualize metrics
        annual_return = (1 + port_return) ** 252 - 1
        annual_vol = port_vol * np.sqrt(252)
        annual_sharpe = sharpe * np.sqrt(252)
        
        return {
            'weights': {asset: float(w) for asset, w in zip(assets, optimal_weights)},
            'expected_return': float(annual_return),
            'volatility': float(annual_vol),
            'sharpe_ratio': float(annual_sharpe),
            'bl_returns': {asset: float(r * 252) for asset, r in zip(assets, bl_returns)},
            'implied_returns': {asset: float(r * 252) for asset, r in zip(assets, implied_returns)}
        }
    
    def optimize_risk_parity(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        constraints: Optional[Dict] = None
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Optimize portfolio using Risk Parity approach.
        
        Risk Parity allocates capital so that each asset contributes equally
        to the total portfolio risk. This approach provides better diversification
        than traditional mean-variance optimization.
        
        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            constraints: Optional constraints dictionary
        
        Returns:
            Dictionary containing optimal weights and portfolio metrics
        
        Example:
            >>> optimizer = PortfolioOptimizer()
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 252),
            ...     'MSFT': np.random.normal(0.001, 0.02, 252),
            ...     'GOOGL': np.random.normal(0.001, 0.02, 252)
            ... }
            >>> result = optimizer.optimize_risk_parity(returns)
        """
        if not returns_dict:
            raise ValueError("Returns dictionary cannot be empty")
        
        # Convert to DataFrame
        df = pd.DataFrame(returns_dict)
        assets = df.columns.tolist()
        n_assets = len(assets)
        
        # Calculate covariance matrix
        cov_matrix = df.cov().values
        
        # Set up constraints
        min_weight = 0.0
        max_weight = 1.0
        if constraints:
            min_weight = constraints.get('min_weight', 0.0)
            max_weight = constraints.get('max_weight', 1.0)
        
        bounds = Bounds(
            lb=np.full(n_assets, min_weight),
            ub=np.full(n_assets, max_weight)
        )
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Objective: minimize sum of squared differences in risk contributions
        def objective_func(w):
            # Portfolio volatility
            port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            if port_vol == 0:
                return 1e10
            
            # Marginal risk contribution for each asset
            marginal_risk = np.dot(cov_matrix, w) / port_vol
            
            # Risk contribution for each asset
            risk_contrib = w * marginal_risk
            
            # Target: equal risk contribution (1/n of total risk)
            target_contrib = port_vol / n_assets
            
            # Minimize sum of squared deviations from target
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Initial guess: equal weights
        w0 = np.full(n_assets, 1.0 / n_assets)
        
        # Optimize
        result = minimize(
            objective_func,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        expected_returns = df.mean().values
        port_return = np.dot(optimal_weights, expected_returns)
        port_vol = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
        sharpe = (port_return - self.risk_free_rate / 252) / port_vol if port_vol > 0 else 0.0
        
        # Calculate risk contributions
        marginal_risk = np.dot(cov_matrix, optimal_weights) / port_vol if port_vol > 0 else np.zeros(n_assets)
        risk_contrib = optimal_weights * marginal_risk
        
        # Annualize metrics
        annual_return = (1 + port_return) ** 252 - 1
        annual_vol = port_vol * np.sqrt(252)
        annual_sharpe = sharpe * np.sqrt(252)
        
        return {
            'weights': {asset: float(w) for asset, w in zip(assets, optimal_weights)},
            'expected_return': float(annual_return),
            'volatility': float(annual_vol),
            'sharpe_ratio': float(annual_sharpe),
            'risk_contributions': {asset: float(rc) for asset, rc in zip(assets, risk_contrib)}
        }

    
    def calculate_kelly_criterion(
        self,
        win_rate: float,
        win_loss_ratio: float,
        max_position_size: float = 0.25
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly Criterion determines the optimal fraction of capital to allocate
        to maximize long-term growth rate.
        
        Formula: f* = (p * b - q) / b
        where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = win/loss ratio (average win / average loss)
        
        Args:
            win_rate: Probability of winning trade (0-1)
            win_loss_ratio: Ratio of average win to average loss
            max_position_size: Maximum allowed position size (default: 0.25 or 25%)
        
        Returns:
            Optimal position size as fraction of capital
        
        Example:
            >>> optimizer = PortfolioOptimizer()
            >>> # 60% win rate, wins are 1.5x losses on average
            >>> kelly = optimizer.calculate_kelly_criterion(0.6, 1.5)
            >>> print(f"Optimal position size: {kelly:.2%}")
        """
        if not 0 <= win_rate <= 1:
            raise ValueError("Win rate must be between 0 and 1")
        if win_loss_ratio <= 0:
            raise ValueError("Win/loss ratio must be positive")
        if not 0 < max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
        
        # Calculate Kelly fraction
        p = win_rate
        q = 1 - win_rate
        b = win_loss_ratio
        
        kelly_fraction = (p * b - q) / b
        
        # Apply constraints
        kelly_fraction = max(0.0, kelly_fraction)  # No negative positions
        kelly_fraction = min(kelly_fraction, max_position_size)  # Cap at max
        
        return float(kelly_fraction)
    
    def calculate_kelly_weights(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        predictions: Dict[str, float],
        max_position_size: float = 0.25
    ) -> Dict[str, float]:
        """
        Calculate portfolio weights using Kelly Criterion for multiple assets.
        
        Args:
            returns_dict: Dictionary mapping asset names to historical return arrays
            predictions: Dictionary mapping asset names to predicted returns
            max_position_size: Maximum position size per asset (default: 0.25)
        
        Returns:
            Dictionary mapping asset names to optimal weights
        
        Example:
            >>> optimizer = PortfolioOptimizer()
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 252),
            ...     'MSFT': np.random.normal(0.001, 0.02, 252)
            ... }
            >>> predictions = {'AAPL': 0.15, 'MSFT': 0.12}
            >>> weights = optimizer.calculate_kelly_weights(returns, predictions)
        """
        if not returns_dict:
            raise ValueError("Returns dictionary cannot be empty")
        if not predictions:
            raise ValueError("Predictions dictionary cannot be empty")
        
        # Convert to DataFrame
        df = pd.DataFrame(returns_dict)
        assets = df.columns.tolist()
        
        # Calculate statistics for each asset
        kelly_weights = {}
        
        for asset in assets:
            if asset not in predictions:
                kelly_weights[asset] = 0.0
                continue
            
            returns = df[asset].values
            predicted_return = predictions[asset]
            
            # Calculate win rate and win/loss ratio
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            if len(returns) == 0:
                kelly_weights[asset] = 0.0
                continue
            
            win_rate = len(positive_returns) / len(returns)
            
            if len(positive_returns) > 0 and len(negative_returns) > 0:
                avg_win = np.mean(positive_returns)
                avg_loss = abs(np.mean(negative_returns))
                win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
            else:
                win_loss_ratio = 1.0
            
            # Adjust win rate based on prediction
            # If prediction is positive, increase win rate; if negative, decrease
            if predicted_return > 0:
                adjusted_win_rate = min(win_rate * (1 + predicted_return), 0.95)
            else:
                adjusted_win_rate = max(win_rate * (1 + predicted_return), 0.05)
            
            # Calculate Kelly fraction
            kelly_fraction = self.calculate_kelly_criterion(
                adjusted_win_rate,
                win_loss_ratio,
                max_position_size
            )
            
            kelly_weights[asset] = kelly_fraction
        
        # Normalize weights to sum to 1
        total_weight = sum(kelly_weights.values())
        if total_weight > 0:
            kelly_weights = {asset: w / total_weight for asset, w in kelly_weights.items()}
        
        return kelly_weights
    
    def optimize_mean_cvar(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        confidence_level: float = 0.95,
        target_return: Optional[float] = None,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Optimize portfolio using Mean-CVaR optimization.
        
        Mean-CVaR optimization minimizes Conditional Value at Risk (expected loss
        beyond VaR) while targeting a specific return level. This approach is
        more conservative than mean-variance optimization.
        
        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            confidence_level: Confidence level for CVaR (default: 0.95)
            target_return: Target portfolio return (if None, maximizes return/CVaR ratio)
            constraints: Optional constraints dictionary
        
        Returns:
            Dictionary containing optimal weights and portfolio metrics
        
        Example:
            >>> optimizer = PortfolioOptimizer()
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 252),
            ...     'MSFT': np.random.normal(0.001, 0.02, 252)
            ... }
            >>> result = optimizer.optimize_mean_cvar(returns, confidence_level=0.95)
        """
        if not returns_dict:
            raise ValueError("Returns dictionary cannot be empty")
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        # Convert to DataFrame
        df = pd.DataFrame(returns_dict)
        assets = df.columns.tolist()
        n_assets = len(assets)
        
        # Get return matrix (T x N)
        returns_matrix = df.values
        T = len(returns_matrix)
        
        # Calculate expected returns
        expected_returns = df.mean().values
        
        # Set up constraints
        min_weight = 0.0
        max_weight = 1.0
        if constraints:
            min_weight = constraints.get('min_weight', 0.0)
            max_weight = constraints.get('max_weight', 1.0)
        
        bounds = Bounds(
            lb=np.full(n_assets, min_weight),
            ub=np.full(n_assets, max_weight)
        )
        
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        
        # Add target return constraint if specified
        if target_return is not None:
            # Convert annual to daily
            daily_target = target_return / 252
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - daily_target
            })
        
        # Objective: minimize CVaR
        def calculate_cvar(w):
            # Calculate portfolio returns for each time period
            portfolio_returns = np.dot(returns_matrix, w)
            
            # Calculate VaR
            var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            
            # Calculate CVaR (average of losses beyond VaR)
            tail_losses = portfolio_returns[portfolio_returns <= -var]
            if len(tail_losses) == 0:
                cvar = var
            else:
                cvar = -np.mean(tail_losses)
            
            return cvar
        
        if target_return is None:
            # Maximize return/CVaR ratio (minimize negative ratio)
            def objective_func(w):
                port_return = np.dot(w, expected_returns)
                cvar = calculate_cvar(w)
                if cvar == 0:
                    return 1e10
                return -port_return / cvar
        else:
            # Minimize CVaR for target return
            def objective_func(w):
                return calculate_cvar(w)
        
        # Initial guess: equal weights
        w0 = np.full(n_assets, 1.0 / n_assets)
        
        # Optimize
        result = minimize(
            objective_func,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        # Extract optimal weights
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        port_return = np.dot(optimal_weights, expected_returns)
        portfolio_returns = np.dot(returns_matrix, optimal_weights)
        
        # Calculate VaR and CVaR
        var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        tail_losses = portfolio_returns[portfolio_returns <= -var]
        cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        # Calculate volatility
        port_vol = np.std(portfolio_returns, ddof=1)
        
        # Annualize metrics
        annual_return = (1 + port_return) ** 252 - 1
        annual_vol = port_vol * np.sqrt(252)
        annual_var = var * np.sqrt(252)
        annual_cvar = cvar * np.sqrt(252)
        
        return {
            'weights': {asset: float(w) for asset, w in zip(assets, optimal_weights)},
            'expected_return': float(annual_return),
            'volatility': float(annual_vol),
            'var_95': float(annual_var),
            'cvar_95': float(annual_cvar),
            'return_cvar_ratio': float(annual_return / annual_cvar) if annual_cvar > 0 else 0.0
        }

    
    def calculate_efficient_frontier(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        n_points: int = 50,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Union[List, pd.DataFrame]]:
        """
        Calculate the efficient frontier for a set of assets.
        
        The efficient frontier represents the set of optimal portfolios that offer
        the highest expected return for a given level of risk.
        
        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            n_points: Number of points to calculate on the frontier
            constraints: Optional constraints dictionary
        
        Returns:
            Dictionary containing:
            - 'frontier': DataFrame with returns, volatility, and Sharpe ratios
            - 'weights': List of weight dictionaries for each frontier point
            - 'max_sharpe': Portfolio with maximum Sharpe ratio
            - 'min_risk': Portfolio with minimum risk
        
        Example:
            >>> optimizer = PortfolioOptimizer(risk_free_rate=0.02)
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 252),
            ...     'MSFT': np.random.normal(0.001, 0.02, 252),
            ...     'GOOGL': np.random.normal(0.001, 0.02, 252)
            ... }
            >>> frontier = optimizer.calculate_efficient_frontier(returns, n_points=30)
        """
        if not returns_dict:
            raise ValueError("Returns dictionary cannot be empty")
        if n_points < 2:
            raise ValueError("Number of points must be at least 2")
        
        # Convert to DataFrame
        df = pd.DataFrame(returns_dict)
        assets = df.columns.tolist()
        
        # Calculate expected returns
        expected_returns = df.mean().values
        
        # Find min and max possible returns
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        # Calculate optimal portfolio for each target return
        frontier_results = []
        frontier_weights = []
        
        for target_return in target_returns:
            try:
                result = self.optimize_mpt(
                    returns_dict,
                    target_return=target_return,
                    objective='min_risk',
                    constraints=constraints
                )
                
                frontier_results.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio']
                })
                frontier_weights.append(result['weights'])
                
            except Exception as e:
                # Skip points that fail to optimize
                warnings.warn(f"Failed to optimize for target return {target_return}: {e}")
                continue
        
        if not frontier_results:
            raise ValueError("Failed to calculate any points on the efficient frontier")
        
        # Create DataFrame
        frontier_df = pd.DataFrame(frontier_results)
        
        # Find special portfolios
        max_sharpe_idx = frontier_df['sharpe_ratio'].idxmax()
        min_risk_idx = frontier_df['volatility'].idxmin()
        
        max_sharpe_portfolio = {
            'weights': frontier_weights[max_sharpe_idx],
            'expected_return': frontier_df.loc[max_sharpe_idx, 'return'],
            'volatility': frontier_df.loc[max_sharpe_idx, 'volatility'],
            'sharpe_ratio': frontier_df.loc[max_sharpe_idx, 'sharpe_ratio']
        }
        
        min_risk_portfolio = {
            'weights': frontier_weights[min_risk_idx],
            'expected_return': frontier_df.loc[min_risk_idx, 'return'],
            'volatility': frontier_df.loc[min_risk_idx, 'volatility'],
            'sharpe_ratio': frontier_df.loc[min_risk_idx, 'sharpe_ratio']
        }
        
        return {
            'frontier': frontier_df,
            'weights': frontier_weights,
            'max_sharpe': max_sharpe_portfolio,
            'min_risk': min_risk_portfolio
        }
    
    def compare_strategies(
        self,
        returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
        predictions: Optional[Dict[str, float]] = None,
        market_caps: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Compare different optimization strategies side-by-side.
        
        Args:
            returns_dict: Dictionary mapping asset names to return arrays
            predictions: Optional predictions for Black-Litterman
            market_caps: Optional market caps for Black-Litterman
            constraints: Optional constraints dictionary
        
        Returns:
            DataFrame comparing all strategies
        
        Example:
            >>> optimizer = PortfolioOptimizer(risk_free_rate=0.02)
            >>> returns = {
            ...     'AAPL': np.random.normal(0.001, 0.02, 252),
            ...     'MSFT': np.random.normal(0.001, 0.02, 252)
            ... }
            >>> comparison = optimizer.compare_strategies(returns)
        """
        if not returns_dict:
            raise ValueError("Returns dictionary cannot be empty")
        
        results = []
        
        # 1. Max Sharpe (MPT)
        try:
            mpt_result = self.optimize_mpt(
                returns_dict,
                objective='max_sharpe',
                constraints=constraints
            )
            results.append({
                'strategy': 'Max Sharpe (MPT)',
                'expected_return': mpt_result['expected_return'],
                'volatility': mpt_result['volatility'],
                'sharpe_ratio': mpt_result['sharpe_ratio'],
                'weights': str(mpt_result['weights'])
            })
        except Exception as e:
            warnings.warn(f"Max Sharpe optimization failed: {e}")
        
        # 2. Min Risk (MPT)
        try:
            min_risk_result = self.optimize_mpt(
                returns_dict,
                objective='min_risk',
                constraints=constraints
            )
            results.append({
                'strategy': 'Min Risk (MPT)',
                'expected_return': min_risk_result['expected_return'],
                'volatility': min_risk_result['volatility'],
                'sharpe_ratio': min_risk_result['sharpe_ratio'],
                'weights': str(min_risk_result['weights'])
            })
        except Exception as e:
            warnings.warn(f"Min Risk optimization failed: {e}")
        
        # 3. Risk Parity
        try:
            rp_result = self.optimize_risk_parity(
                returns_dict,
                constraints=constraints
            )
            results.append({
                'strategy': 'Risk Parity',
                'expected_return': rp_result['expected_return'],
                'volatility': rp_result['volatility'],
                'sharpe_ratio': rp_result['sharpe_ratio'],
                'weights': str(rp_result['weights'])
            })
        except Exception as e:
            warnings.warn(f"Risk Parity optimization failed: {e}")
        
        # 4. Mean-CVaR
        try:
            cvar_result = self.optimize_mean_cvar(
                returns_dict,
                constraints=constraints
            )
            results.append({
                'strategy': 'Mean-CVaR',
                'expected_return': cvar_result['expected_return'],
                'volatility': cvar_result['volatility'],
                'sharpe_ratio': cvar_result.get('sharpe_ratio', 0.0),
                'weights': str(cvar_result['weights'])
            })
        except Exception as e:
            warnings.warn(f"Mean-CVaR optimization failed: {e}")
        
        # 5. Black-Litterman (if predictions and market caps provided)
        if predictions and market_caps:
            try:
                bl_result = self.optimize_black_litterman(
                    returns_dict,
                    market_caps,
                    predictions,
                    constraints=constraints
                )
                results.append({
                    'strategy': 'Black-Litterman',
                    'expected_return': bl_result['expected_return'],
                    'volatility': bl_result['volatility'],
                    'sharpe_ratio': bl_result['sharpe_ratio'],
                    'weights': str(bl_result['weights'])
                })
            except Exception as e:
                warnings.warn(f"Black-Litterman optimization failed: {e}")
        
        # 6. Equal Weight (baseline)
        try:
            assets = list(returns_dict.keys())
            equal_weights = {asset: 1.0 / len(assets) for asset in assets}
            df = pd.DataFrame(returns_dict)
            expected_returns = df.mean().values
            cov_matrix = df.cov().values
            
            weight_array = np.array([equal_weights[asset] for asset in df.columns])
            port_return = np.dot(weight_array, expected_returns)
            port_vol = np.sqrt(np.dot(weight_array, np.dot(cov_matrix, weight_array)))
            sharpe = (port_return - self.risk_free_rate / 252) / port_vol if port_vol > 0 else 0.0
            
            annual_return = (1 + port_return) ** 252 - 1
            annual_vol = port_vol * np.sqrt(252)
            annual_sharpe = sharpe * np.sqrt(252)
            
            results.append({
                'strategy': 'Equal Weight',
                'expected_return': annual_return,
                'volatility': annual_vol,
                'sharpe_ratio': annual_sharpe,
                'weights': str(equal_weights)
            })
        except Exception as e:
            warnings.warn(f"Equal Weight calculation failed: {e}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Sort by Sharpe ratio (descending)
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
        
        return comparison_df
