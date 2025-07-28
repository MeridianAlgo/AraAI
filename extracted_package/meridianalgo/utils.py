"""
Utility Functions Module for MeridianAlgo
Provides helper functions for trading operations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple


class TradeUtils:
    """
    Utility functions for trading operations
    """
    
    @staticmethod
    def calculate_position_size(capital: float, risk_percent: float, 
                              entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            capital: Available capital
            risk_percent: Percentage of capital to risk (0-100)
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            
        Returns:
            float: Position size in units
        """
        risk_amount = capital * (risk_percent / 100)
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        return position_size
    
    @staticmethod
    def calculate_risk_reward_ratio(entry_price: float, target_price: float, 
                                  stop_loss: float) -> float:
        """
        Calculate risk-to-reward ratio
        
        Args:
            entry_price: Entry price
            target_price: Target profit price
            stop_loss: Stop loss price
            
        Returns:
            float: Risk-to-reward ratio
        """
        reward = abs(target_price - entry_price)
        risk = abs(entry_price - stop_loss)
        
        if risk == 0:
            return 0
        
        return reward / risk
    
    @staticmethod
    def calculate_pnl(entry_price: float, exit_price: float, 
                     quantity: float, side: str = "long") -> float:
        """
        Calculate profit/loss for a trade
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            side: "long" or "short"
            
        Returns:
            float: Profit/loss amount
        """
        if side.lower() == "long":
            return (exit_price - entry_price) * quantity
        elif side.lower() == "short":
            return (entry_price - exit_price) * quantity
        else:
            raise ValueError("Side must be 'long' or 'short'")
    
    @staticmethod
    def calculate_pnl_percent(entry_price: float, exit_price: float, 
                            side: str = "long") -> float:
        """
        Calculate profit/loss percentage
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            side: "long" or "short"
            
        Returns:
            float: Profit/loss percentage
        """
        if side.lower() == "long":
            return ((exit_price - entry_price) / entry_price) * 100
        elif side.lower() == "short":
            return ((entry_price - exit_price) / entry_price) * 100
        else:
            raise ValueError("Side must be 'long' or 'short'")
    
    @staticmethod
    def calculate_compound_return(returns: List[float]) -> float:
        """
        Calculate compound return from a list of returns
        
        Args:
            returns: List of return percentages
            
        Returns:
            float: Compound return percentage
        """
        if not returns:
            return 0.0
        
        compound_factor = 1.0
        for ret in returns:
            compound_factor *= (1 + ret / 100)
        
        return (compound_factor - 1) * 100
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate (default: 0.0)
            
        Returns:
            float: Sharpe ratio
        """
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        if len(excess_returns) < 2:
            return 0.0
        
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            equity_curve: List of equity values over time
            
        Returns:
            float: Maximum drawdown percentage
        """
        if not equity_curve:
            return 0.0
        
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        
        return np.min(drawdown) * 100
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> float:
        """
        Calculate win rate from trade history
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            float: Win rate percentage
        """
        if not trades:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return (winning_trades / len(trades)) * 100
    
    @staticmethod
    def calculate_average_win_loss(trades: List[Dict]) -> Tuple[float, float]:
        """
        Calculate average win and average loss
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            Tuple[float, float]: (average_win, average_loss)
        """
        if not trades:
            return 0.0, 0.0
        
        wins = [trade['pnl'] for trade in trades if trade.get('pnl', 0) > 0]
        losses = [trade['pnl'] for trade in trades if trade.get('pnl', 0) < 0]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        
        return avg_win, avg_loss
    
    @staticmethod
    def calculate_profit_factor(trades: List[Dict]) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        
        Args:
            trades: List of trade dictionaries with 'pnl' key
            
        Returns:
            float: Profit factor
        """
        if not trades:
            return 0.0
        
        gross_profit = sum(trade['pnl'] for trade in trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in trades if trade.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def format_currency(amount: float, currency: str = "USD") -> str:
        """
        Format currency amount
        
        Args:
            amount: Amount to format
            currency: Currency code
            
        Returns:
            str: Formatted currency string
        """
        if currency == "USD":
            return f"${amount:,.2f}"
        elif currency == "EUR":
            return f"€{amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """
        Format percentage value
        
        Args:
            value: Percentage value
            decimals: Number of decimal places
            
        Returns:
            str: Formatted percentage string
        """
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def validate_trade_params(symbol: str, quantity: float, price: float) -> bool:
        """
        Validate trade parameters
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            
        Returns:
            bool: True if parameters are valid
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        if quantity <= 0 or not isinstance(quantity, (int, float)):
            return False
        
        if price <= 0 or not isinstance(price, (int, float)):
            return False
        
        return True
    
    @staticmethod
    def round_to_tick_size(price: float, tick_size: float) -> float:
        """
        Round price to nearest tick size
        
        Args:
            price: Price to round
            tick_size: Tick size
            
        Returns:
            float: Rounded price
        """
        if tick_size <= 0:
            return price
        
        return round(price / tick_size) * tick_size 