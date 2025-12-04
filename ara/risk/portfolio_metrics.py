"""
Portfolio Risk Metrics

Implements comprehensive portfolio-level risk metrics including Sharpe ratio,
Sortino ratio, maximum drawdown, and other performance measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime


class PortfolioMetrics:
    """
    Calculate portfolio-level risk and performance metrics.
    
    This class provides methods for:
    - Portfolio volatility and beta calculation
    - Sharpe, Sortino, and Calmar ratios
    - Maximum drawdown and recovery time
    - Tracking error and information ratio
    - Downside deviation
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize PortfolioMetrics.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2% or 0.02)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_portfolio_volatility(
        self,
        returns: Union[np.ndarray, pd.Series],
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate portfolio volatility (standard deviation of returns).
        
        Args:
            returns: Array or Series of portfolio returns
            annualize: Whether to annualize the volatility
            periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
        
        Returns:
            Portfolio volatility
        
        Example:
            >>> metrics = PortfolioMetrics()
            >>> returns = np.random.normal(0.001, 0.02, 252)
            >>> vol = metrics.calculate_portfolio_volatility(returns)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")
        
        volatility = np.std(returns, ddof=1)
        
        if annualize:
            volatility *= np.sqrt(periods_per_year)
        
        return float(volatility)
    
    def calculate_beta(
        self,
        portfolio_returns: Union[np.ndarray, pd.Series],
        benchmark_returns: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate portfolio beta relative to a benchmark.
        
        Beta measures the portfolio's sensitivity to market movements.
        Beta = 1: Moves with market
        Beta > 1: More volatile than market
        Beta < 1: Less volatile than market
        
        Args:
            portfolio_returns: Array or Series of portfolio returns
            benchmark_returns: Array or Series of benchmark returns
        
        Returns:
            Portfolio beta
        
        Example:
            >>> metrics = PortfolioMetrics()
            >>> portfolio_returns = np.random.normal(0.001, 0.02, 252)
            >>> benchmark_returns = np.random.normal(0.0008, 0.015, 252)
            >>> beta = metrics.calculate_beta(portfolio_returns, benchmark_returns)
        """
        if isinstance(portfolio_returns, pd.Series):
            portfolio_returns = portfolio_returns.values
        if isinstance(benchmark_returns, pd.Series):
            benchmark_returns = benchmark_returns.values
        
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio and benchmark returns must have same length")
        
        if len(portfolio_returns) == 0:
            raise ValueError("Returns arrays cannot be empty")
        
        # Calculate covariance and variance
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
        
        if benchmark_variance == 0:
            raise ValueError("Benchmark variance is zero")
        
        beta = covariance / benchmark_variance
        
        return float(beta)
    
    def calculate_sharpe_ratio(
        self,
        returns: Union[np.ndarray, pd.Series],
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).
        
        Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
        
        Args:
            returns: Array or Series of portfolio returns
            annualize: Whether to annualize the ratio
            periods_per_year: Number of periods per year
        
        Returns:
            Sharpe ratio
        
        Example:
            >>> metrics = PortfolioMetrics(risk_free_rate=0.02)
            >>> returns = np.random.normal(0.001, 0.02, 252)
            >>> sharpe = metrics.calculate_sharpe_ratio(returns)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")
        
        mean_return = np.mean(returns)
        volatility = np.std(returns, ddof=1)
        
        if volatility == 0:
            return 0.0
        
        # Calculate daily risk-free rate
        daily_rf_rate = self.risk_free_rate / periods_per_year
        
        # Calculate Sharpe ratio
        sharpe = (mean_return - daily_rf_rate) / volatility
        
        if annualize:
            sharpe *= np.sqrt(periods_per_year)
        
        return float(sharpe)
    
    def calculate_sortino_ratio(
        self,
        returns: Union[np.ndarray, pd.Series],
        annualize: bool = True,
        periods_per_year: int = 252,
        target_return: Optional[float] = None
    ) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return).
        
        Similar to Sharpe ratio but only considers downside volatility.
        
        Args:
            returns: Array or Series of portfolio returns
            annualize: Whether to annualize the ratio
            periods_per_year: Number of periods per year
            target_return: Target return threshold (default: risk-free rate)
        
        Returns:
            Sortino ratio
        
        Example:
            >>> metrics = PortfolioMetrics(risk_free_rate=0.02)
            >>> returns = np.random.normal(0.001, 0.02, 252)
            >>> sortino = metrics.calculate_sortino_ratio(returns)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")
        
        mean_return = np.mean(returns)
        
        # Use risk-free rate as target if not specified
        if target_return is None:
            target_return = self.risk_free_rate / periods_per_year
        
        # Calculate downside deviation
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            # No downside returns, return a high value
            return float('inf') if mean_return > target_return else 0.0
        
        downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        if downside_deviation == 0:
            return 0.0
        
        # Calculate Sortino ratio
        sortino = (mean_return - target_return) / downside_deviation
        
        if annualize:
            sortino *= np.sqrt(periods_per_year)
        
        return float(sortino)
    
    def calculate_calmar_ratio(
        self,
        returns: Union[np.ndarray, pd.Series],
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (return over maximum drawdown).
        
        Calmar Ratio = Annualized Return / Maximum Drawdown
        
        Args:
            returns: Array or Series of portfolio returns
            annualize: Whether to annualize the return
            periods_per_year: Number of periods per year
        
        Returns:
            Calmar ratio
        
        Example:
            >>> metrics = PortfolioMetrics()
            >>> returns = np.random.normal(0.001, 0.02, 252)
            >>> calmar = metrics.calculate_calmar_ratio(returns)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")
        
        mean_return = np.mean(returns)
        
        if annualize:
            annualized_return = (1 + mean_return) ** periods_per_year - 1
        else:
            annualized_return = mean_return
        
        # Calculate maximum drawdown
        max_dd = self.calculate_maximum_drawdown(returns)
        
        if max_dd == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        calmar = annualized_return / max_dd
        
        return float(calmar)
    
    def calculate_maximum_drawdown(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Calculate maximum drawdown (largest peak-to-trough decline).
        
        Args:
            returns: Array or Series of portfolio returns
        
        Returns:
            Maximum drawdown as a positive percentage (e.g., 0.20 for 20% drawdown)
        
        Example:
            >>> metrics = PortfolioMetrics()
            >>> returns = np.random.normal(0.001, 0.02, 252)
            >>> max_dd = metrics.calculate_maximum_drawdown(returns)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdown at each point
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown is the minimum (most negative) value
        max_drawdown = abs(np.min(drawdown))
        
        return float(max_drawdown)
    
    def calculate_drawdown_series(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Calculate drawdown series over time.
        
        Args:
            returns: Array or Series of portfolio returns
        
        Returns:
            Array of drawdown values at each time point
        
        Example:
            >>> metrics = PortfolioMetrics()
            >>> returns = np.random.normal(0.001, 0.02, 252)
            >>> dd_series = metrics.calculate_drawdown_series(returns)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdown at each point
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown
    
    def calculate_recovery_time(
        self,
        returns: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Union[int, float]]:
        """
        Calculate recovery time from maximum drawdown.
        
        Args:
            returns: Array or Series of portfolio returns
        
        Returns:
            Dictionary containing:
            - 'max_drawdown': Maximum drawdown value
            - 'drawdown_start': Index where max drawdown started
            - 'drawdown_end': Index where max drawdown bottomed
            - 'recovery_end': Index where portfolio recovered (or None if not recovered)
            - 'recovery_time': Number of periods to recover (or None if not recovered)
        
        Example:
            >>> metrics = PortfolioMetrics()
            >>> returns = np.random.normal(0.001, 0.02, 252)
            >>> recovery = metrics.calculate_recovery_time(returns)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")
        
        # Calculate cumulative returns and drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        # Find maximum drawdown point
        max_dd_idx = np.argmin(drawdown)
        max_dd_value = abs(drawdown[max_dd_idx])
        
        # Find start of drawdown (last peak before max drawdown)
        dd_start_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] == 0:
                dd_start_idx = i
                break
        
        # Find recovery point (first time drawdown returns to 0 after max drawdown)
        recovery_idx = None
        for i in range(max_dd_idx + 1, len(drawdown)):
            if drawdown[i] >= 0:
                recovery_idx = i
                break
        
        # Calculate recovery time
        recovery_time = None
        if recovery_idx is not None:
            recovery_time = recovery_idx - max_dd_idx
        
        return {
            'max_drawdown': float(max_dd_value),
            'drawdown_start': int(dd_start_idx),
            'drawdown_end': int(max_dd_idx),
            'recovery_end': int(recovery_idx) if recovery_idx is not None else None,
            'recovery_time': int(recovery_time) if recovery_time is not None else None
        }
    
    def calculate_tracking_error(
        self,
        portfolio_returns: Union[np.ndarray, pd.Series],
        benchmark_returns: Union[np.ndarray, pd.Series],
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate tracking error (volatility of excess returns).
        
        Tracking error measures how closely a portfolio follows its benchmark.
        
        Args:
            portfolio_returns: Array or Series of portfolio returns
            benchmark_returns: Array or Series of benchmark returns
            annualize: Whether to annualize the tracking error
            periods_per_year: Number of periods per year
        
        Returns:
            Tracking error
        
        Example:
            >>> metrics = PortfolioMetrics()
            >>> portfolio_returns = np.random.normal(0.001, 0.02, 252)
            >>> benchmark_returns = np.random.normal(0.0008, 0.015, 252)
            >>> te = metrics.calculate_tracking_error(portfolio_returns, benchmark_returns)
        """
        if isinstance(portfolio_returns, pd.Series):
            portfolio_returns = portfolio_returns.values
        if isinstance(benchmark_returns, pd.Series):
            benchmark_returns = benchmark_returns.values
        
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio and benchmark returns must have same length")
        
        if len(portfolio_returns) == 0:
            raise ValueError("Returns arrays cannot be empty")
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns
        
        # Calculate standard deviation of excess returns
        tracking_error = np.std(excess_returns, ddof=1)
        
        if annualize:
            tracking_error *= np.sqrt(periods_per_year)
        
        return float(tracking_error)
    
    def calculate_information_ratio(
        self,
        portfolio_returns: Union[np.ndarray, pd.Series],
        benchmark_returns: Union[np.ndarray, pd.Series],
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate information ratio (excess return per unit of tracking error).
        
        Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error
        
        Args:
            portfolio_returns: Array or Series of portfolio returns
            benchmark_returns: Array or Series of benchmark returns
            annualize: Whether to annualize the ratio
            periods_per_year: Number of periods per year
        
        Returns:
            Information ratio
        
        Example:
            >>> metrics = PortfolioMetrics()
            >>> portfolio_returns = np.random.normal(0.001, 0.02, 252)
            >>> benchmark_returns = np.random.normal(0.0008, 0.015, 252)
            >>> ir = metrics.calculate_information_ratio(portfolio_returns, benchmark_returns)
        """
        if isinstance(portfolio_returns, pd.Series):
            portfolio_returns = portfolio_returns.values
        if isinstance(benchmark_returns, pd.Series):
            benchmark_returns = benchmark_returns.values
        
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio and benchmark returns must have same length")
        
        if len(portfolio_returns) == 0:
            raise ValueError("Returns arrays cannot be empty")
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns
        mean_excess_return = np.mean(excess_returns)
        
        # Calculate tracking error
        tracking_error = np.std(excess_returns, ddof=1)
        
        if tracking_error == 0:
            return 0.0
        
        # Calculate information ratio
        info_ratio = mean_excess_return / tracking_error
        
        if annualize:
            info_ratio *= np.sqrt(periods_per_year)
        
        return float(info_ratio)
    
    def calculate_downside_deviation(
        self,
        returns: Union[np.ndarray, pd.Series],
        target_return: Optional[float] = None,
        annualize: bool = True,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate downside deviation (volatility of negative returns).
        
        Downside deviation only considers returns below a target threshold,
        providing a better measure of downside risk than standard deviation.
        
        Args:
            returns: Array or Series of portfolio returns
            target_return: Target return threshold (default: 0)
            annualize: Whether to annualize the deviation
            periods_per_year: Number of periods per year
        
        Returns:
            Downside deviation
        
        Example:
            >>> metrics = PortfolioMetrics()
            >>> returns = np.random.normal(0.001, 0.02, 252)
            >>> dd = metrics.calculate_downside_deviation(returns)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")
        
        # Use 0 as target if not specified
        if target_return is None:
            target_return = 0.0
        
        # Calculate downside returns
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
        
        # Calculate downside deviation
        downside_dev = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        if annualize:
            downside_dev *= np.sqrt(periods_per_year)
        
        return float(downside_dev)
    
    def calculate_all_metrics(
        self,
        returns: Union[np.ndarray, pd.Series],
        benchmark_returns: Optional[Union[np.ndarray, pd.Series]] = None,
        periods_per_year: int = 252
    ) -> Dict[str, float]:
        """
        Calculate all portfolio risk metrics at once.
        
        Args:
            returns: Array or Series of portfolio returns
            benchmark_returns: Optional benchmark returns for beta, tracking error, etc.
            periods_per_year: Number of periods per year
        
        Returns:
            Dictionary containing all calculated metrics
        
        Example:
            >>> metrics = PortfolioMetrics(risk_free_rate=0.02)
            >>> returns = np.random.normal(0.001, 0.02, 252)
            >>> all_metrics = metrics.calculate_all_metrics(returns)
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")
        
        result = {
            'volatility': self.calculate_portfolio_volatility(returns, periods_per_year=periods_per_year),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns, periods_per_year=periods_per_year),
            'sortino_ratio': self.calculate_sortino_ratio(returns, periods_per_year=periods_per_year),
            'calmar_ratio': self.calculate_calmar_ratio(returns, periods_per_year=periods_per_year),
            'max_drawdown': self.calculate_maximum_drawdown(returns),
            'downside_deviation': self.calculate_downside_deviation(returns, periods_per_year=periods_per_year)
        }
        
        # Add recovery time info
        recovery_info = self.calculate_recovery_time(returns)
        result['recovery_time'] = recovery_info['recovery_time']
        
        # Add benchmark-relative metrics if benchmark provided
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, pd.Series):
                benchmark_returns = benchmark_returns.values
            
            if len(benchmark_returns) == len(returns):
                result['beta'] = self.calculate_beta(returns, benchmark_returns)
                result['tracking_error'] = self.calculate_tracking_error(
                    returns, benchmark_returns, periods_per_year=periods_per_year
                )
                result['information_ratio'] = self.calculate_information_ratio(
                    returns, benchmark_returns, periods_per_year=periods_per_year
                )
        
        return result
