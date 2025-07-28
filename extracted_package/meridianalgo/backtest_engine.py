"""
Backtest Engine Module for MeridianAlgo
Handles backtesting of trading strategies on historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple


class BacktestEngine:
    """
    Backtesting engine for testing trading strategies on historical data
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize the backtest engine
        
        Args:
            initial_capital: Starting capital for the backtest
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
    def load_data(self, data: pd.DataFrame) -> bool:
        """
        Load historical data for backtesting
        
        Args:
            data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            bool: True if data loaded successfully
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        self.data = data.sort_values('timestamp').reset_index(drop=True)
        return True
    
    def run_backtest(self, strategy: Callable, **strategy_params) -> Dict:
        """
        Run backtest with a given strategy
        
        Args:
            strategy: Function that implements the trading strategy
            **strategy_params: Parameters to pass to the strategy
            
        Returns:
            Dict: Backtest results including performance metrics
        """
        if not hasattr(self, 'data'):
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Initialize equity curve
        self.equity_curve.append({
            'timestamp': self.data.iloc[0]['timestamp'],
            'equity': self.initial_capital,
            'positions': {}
        })
        
        # Run strategy on each data point
        for i, row in self.data.iterrows():
            signal = strategy(row, self.positions, self.current_capital, **strategy_params)
            
            if signal:
                self._execute_signal(signal, row)
            
            # Update equity curve
            current_equity = self._calculate_current_equity(row['close'])
            self.equity_curve.append({
                'timestamp': row['timestamp'],
                'equity': current_equity,
                'positions': self.positions.copy()
            })
        
        return self._calculate_performance_metrics()
    
    def _execute_signal(self, signal: Dict, current_data: pd.Series):
        """
        Execute a trading signal
        
        Args:
            signal: Dictionary containing signal details
            current_data: Current market data
        """
        symbol = signal.get('symbol', 'default')
        action = signal.get('action')  # 'buy' or 'sell'
        quantity = signal.get('quantity')
        price = current_data['close']
        
        if action == 'buy':
            cost = quantity * price
            if cost <= self.current_capital:
                self.current_capital -= cost
                if symbol not in self.positions:
                    self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
                
                # Update average price
                total_quantity = self.positions[symbol]['quantity'] + quantity
                total_cost = (self.positions[symbol]['quantity'] * self.positions[symbol]['avg_price']) + cost
                self.positions[symbol]['avg_price'] = total_cost / total_quantity
                self.positions[symbol]['quantity'] = total_quantity
                
                self.trades.append({
                    'timestamp': current_data['timestamp'],
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': price,
                    'cost': cost
                })
        
        elif action == 'sell':
            if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                revenue = quantity * price
                self.current_capital += revenue
                self.positions[symbol]['quantity'] -= quantity
                
                if self.positions[symbol]['quantity'] <= 0:
                    del self.positions[symbol]
                
                self.trades.append({
                    'timestamp': current_data['timestamp'],
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': quantity,
                    'price': price,
                    'revenue': revenue
                })
    
    def _calculate_current_equity(self, current_price: float) -> float:
        """
        Calculate current equity including unrealized P&L
        
        Args:
            current_price: Current market price
            
        Returns:
            float: Current equity value
        """
        equity = self.current_capital
        
        for symbol, position in self.positions.items():
            unrealized_pnl = (current_price - position['avg_price']) * position['quantity']
            equity += unrealized_pnl
        
        return equity
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics from backtest results
        
        Returns:
            Dict: Performance metrics
        """
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Calculate metrics
        total_return = (equity_df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        annualized_return = total_return * (252 / len(equity_df))  # Assuming daily data
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        returns = equity_df['returns'].dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Maximum drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'final_equity': equity_df['equity'].iloc[-1],
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get the equity curve as a DataFrame
        
        Returns:
            pd.DataFrame: Equity curve data
        """
        return pd.DataFrame(self.equity_curve)
    
    def get_trades(self) -> pd.DataFrame:
        """
        Get all trades as a DataFrame
        
        Returns:
            pd.DataFrame: Trade data
        """
        return pd.DataFrame(self.trades) 