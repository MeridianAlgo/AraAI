"""
Market data management and technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketDataManager:
    """Enhanced market data manager with caching and error handling"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get_stock_data(self, symbol, period="2y", interval="1d"):
        """
        Get stock data with intelligent caching
        
        Args:
            symbol (str): Stock symbol
            period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            pd.DataFrame: Stock data with OHLCV
        """
        try:
            cache_key = f"{symbol}_{period}_{interval}"
            
            # Check cache
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).seconds < self.cache_timeout:
                    return cached_data
            
            # Fetch new data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print(f"No data available for {symbol}")
                return None
            
            # Cache the data
            self.cache[cache_key] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Get current stock price"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                return data['Close'].iloc[-1]
            else:
                # Fallback to daily data
                data = ticker.history(period="1d")
                return data['Close'].iloc[-1] if not data.empty else None
                
        except Exception as e:
            print(f"Error getting current price for {symbol}: {e}")
            return None
    
    def get_stock_info(self, symbol):
        """Get stock information and metadata"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
        except Exception as e:
            print(f"Error getting stock info for {symbol}: {e}")
            return {'symbol': symbol, 'name': symbol}
    
    def get_stock_analysis(self, symbol):
        """Get comprehensive stock analysis"""
        try:
            data = self.get_stock_data(symbol)
            info = self.get_stock_info(symbol)
            
            if data is None:
                return None
            
            # Calculate basic metrics
            current_price = data['Close'].iloc[-1]
            price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
            
            # Volume analysis
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Price ranges
            week_high = data['High'].rolling(5).max().iloc[-1]
            week_low = data['Low'].rolling(5).min().iloc[-1]
            month_high = data['High'].rolling(20).max().iloc[-1]
            month_low = data['Low'].rolling(20).min().iloc[-1]
            
            return {
                'symbol': symbol,
                'info': info,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'volume_ratio': volume_ratio,
                'week_high': week_high,
                'week_low': week_low,
                'month_high': month_high,
                'month_low': month_low,
                'data_points': len(data),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None

class TechnicalIndicators:
    """Enhanced technical indicators calculator"""
    
    def __init__(self):
        pass
    
    def calculate_all_indicators(self, data):
        """Calculate all technical indicators"""
        try:
            df = data.copy()
            
            # Moving Averages
            df = self.add_moving_averages(df)
            
            # Momentum Indicators
            df = self.add_momentum_indicators(df)
            
            # Volatility Indicators
            df = self.add_volatility_indicators(df)
            
            # Volume Indicators
            df = self.add_volume_indicators(df)
            
            # Price-based features
            df = self.add_price_features(df)
            
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return data
    
    def add_moving_averages(self, df):
        """Add moving average indicators"""
        try:
            # Simple Moving Averages
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            df['EMA_50'] = df['Close'].ewm(span=50).mean()
            
            return df
            
        except Exception as e:
            print(f"Error calculating moving averages: {e}")
            return df
    
    def add_momentum_indicators(self, df):
        """Add momentum-based indicators"""
        try:
            # RSI (Relative Strength Index)
            df['RSI'] = self.calculate_rsi(df['Close'])
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Stochastic Oscillator
            df['Stoch_K'], df['Stoch_D'] = self.calculate_stochastic(df)
            
            # Williams %R
            df['Williams_R'] = self.calculate_williams_r(df)
            
            # Commodity Channel Index (CCI)
            df['CCI'] = self.calculate_cci(df)
            
            return df
            
        except Exception as e:
            print(f"Error calculating momentum indicators: {e}")
            return df
    
    def add_volatility_indicators(self, df):
        """Add volatility-based indicators"""
        try:
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Average True Range (ATR)
            df['ATR'] = self.calculate_atr(df)
            
            return df
            
        except Exception as e:
            print(f"Error calculating volatility indicators: {e}")
            return df
    
    def add_volume_indicators(self, df):
        """Add volume-based indicators"""
        try:
            # On-Balance Volume (OBV)
            df['OBV'] = self.calculate_obv(df)
            
            # Volume Moving Average
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Money Flow Index (MFI)
            df['MFI'] = self.calculate_mfi(df)
            
            return df
            
        except Exception as e:
            print(f"Error calculating volume indicators: {e}")
            return df
    
    def add_price_features(self, df):
        """Add price-based features"""
        try:
            # Price changes
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_5'] = df['Close'].pct_change(5)
            df['Price_Change_10'] = df['Close'].pct_change(10)
            
            # Volume changes
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # High-Low spread
            df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
            
            # Open-Close spread
            df['OC_Spread'] = (df['Close'] - df['Open']) / df['Open']
            
            return df
            
        except Exception as e:
            print(f"Error calculating price features: {e}")
            return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        try:
            low_min = df['Low'].rolling(window=k_period).min()
            high_max = df['High'].rolling(window=k_period).max()
            k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            d_percent = k_percent.rolling(window=d_period).mean()
            return k_percent, d_percent
        except:
            return pd.Series([50] * len(df), index=df.index), pd.Series([50] * len(df), index=df.index)
    
    def calculate_williams_r(self, df, period=14):
        """Calculate Williams %R"""
        try:
            high_max = df['High'].rolling(window=period).max()
            low_min = df['Low'].rolling(window=period).min()
            williams_r = -100 * ((high_max - df['Close']) / (high_max - low_min))
            return williams_r
        except:
            return pd.Series([-50] * len(df), index=df.index)
    
    def calculate_cci(self, df, period=20):
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma) / (0.015 * mad)
            return cci
        except:
            return pd.Series([0] * len(df), index=df.index)
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        try:
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr
        except:
            return pd.Series([1] * len(df), index=df.index)
    
    def calculate_obv(self, df):
        """Calculate On-Balance Volume"""
        try:
            obv = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.append(obv[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.append(obv[-1] - df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            return pd.Series(obv, index=df.index)
        except:
            return pd.Series([0] * len(df), index=df.index)
    
    def calculate_mfi(self, df, period=14):
        """Calculate Money Flow Index"""
        try:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = []
            negative_flow = []
            
            for i in range(1, len(typical_price)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.append(money_flow.iloc[i])
                    negative_flow.append(0)
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    positive_flow.append(0)
                    negative_flow.append(money_flow.iloc[i])
                else:
                    positive_flow.append(0)
                    negative_flow.append(0)
            
            positive_flow = [0] + positive_flow
            negative_flow = [0] + negative_flow
            
            positive_mf = pd.Series(positive_flow, index=df.index).rolling(window=period).sum()
            negative_mf = pd.Series(negative_flow, index=df.index).rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
            return mfi.fillna(50)
        except:
            return pd.Series([50] * len(df), index=df.index)