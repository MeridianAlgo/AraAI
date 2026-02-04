"""
Market data management and technical indicators calculation
"""

import numpy as np
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


class MarketDataManager:
    """
    Enhanced market data management with intelligent caching and error handling
    """

    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        self.data_cache = {}

    def get_stock_data(self, symbol, period="2y", interval="1d"):
        """
        Get stock data with intelligent caching and error handling
        """
        try:
            # Check cache first
            if self.cache_manager:
                cached_data = self.cache_manager.load_market_data(symbol)
                if cached_data is not None:
                    return cached_data

            # Fetch fresh data
            ticker = yf.Ticker(symbol)

            # Try different periods if the requested one fails
            periods_to_try = [period, "1y", "6mo", "3mo", "1mo"]

            for p in periods_to_try:
                try:
                    data = ticker.history(period=p, interval=interval)

                    if not data.empty and len(data) >= 30:  # Minimum 30 data points
                        # Clean and validate data
                        data = self._clean_data(data)

                        # Cache the data
                        if self.cache_manager:
                            self.cache_manager.save_market_data(symbol, data)

                        return data

                except Exception as e:
                    if p == periods_to_try[-1]:  # Last attempt
                        raise e
                    continue

            raise ValueError(f"No valid data found for {symbol}")

        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")

    def _clean_data(self, data):
        """Clean and validate market data"""
        try:
            # Remove any rows with NaN values in essential columns
            essential_columns = ["Open", "High", "Low", "Close", "Volume"]
            data = data.dropna(subset=essential_columns)

            # Remove rows where High < Low (data errors)
            data = data[data["High"] >= data["Low"]]

            # Remove rows where Close is 0 or negative
            data = data[data["Close"] > 0]

            # Remove extreme outliers (price changes > 50% in one day)
            price_change = data["Close"].pct_change().abs()
            data = data[price_change < 0.5]

            # Forward fill any remaining NaN values
            data = data.fillna(method="ffill")

            # Ensure we have at least some data
            if len(data) < 10:
                raise ValueError("Insufficient clean data after processing")

            return data

        except Exception as e:
            raise ValueError(f"Data cleaning failed: {str(e)}")

    def get_stock_analysis(self, symbol):
        """Get comprehensive stock analysis"""
        try:
            data = self.get_stock_data(symbol)

            if data.empty:
                return {"error": f"No data available for {symbol}"}

            # Get company info
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Calculate basic metrics
            current_price = data["Close"].iloc[-1]
            price_change_1d = (
                (current_price - data["Close"].iloc[-2]) / data["Close"].iloc[-2] * 100
            )
            price_change_1w = (
                (current_price - data["Close"].iloc[-6]) / data["Close"].iloc[-6] * 100
                if len(data) >= 6
                else 0
            )
            price_change_1m = (
                (current_price - data["Close"].iloc[-21]) / data["Close"].iloc[-21] * 100
                if len(data) >= 21
                else 0
            )

            # Volume analysis
            avg_volume = data["Volume"].mean()
            recent_volume = data["Volume"].iloc[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

            # Volatility
            returns = data["Close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility

            analysis = {
                "symbol": symbol,
                "company_name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "current_price": current_price,
                "price_changes": {
                    "1d": price_change_1d,
                    "1w": price_change_1w,
                    "1m": price_change_1m,
                },
                "volume_analysis": {
                    "current": recent_volume,
                    "average": avg_volume,
                    "ratio": volume_ratio,
                },
                "volatility": volatility,
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "data_points": len(data),
                "data_range": {
                    "start": data.index[0].strftime("%Y-%m-%d"),
                    "end": data.index[-1].strftime("%Y-%m-%d"),
                },
            }

            return analysis

        except Exception as e:
            return {"error": f"Analysis failed for {symbol}: {str(e)}"}


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculation
    """

    def __init__(self):
        self.indicators = {}

    def calculate_all_indicators(self, data):
        """Calculate all technical indicators"""
        try:
            # Make a copy to avoid modifying original data
            enhanced_data = data.copy()

            # Price-based indicators
            enhanced_data = self._calculate_moving_averages(enhanced_data)
            enhanced_data = self._calculate_bollinger_bands(enhanced_data)
            enhanced_data = self._calculate_rsi(enhanced_data)
            enhanced_data = self._calculate_macd(enhanced_data)
            enhanced_data = self._calculate_stochastic(enhanced_data)
            enhanced_data = self._calculate_williams_r(enhanced_data)
            enhanced_data = self._calculate_cci(enhanced_data)
            enhanced_data = self._calculate_atr(enhanced_data)

            # Volume-based indicators
            enhanced_data = self._calculate_obv(enhanced_data)
            enhanced_data = self._calculate_volume_sma(enhanced_data)

            # Price change indicators
            enhanced_data = self._calculate_price_changes(enhanced_data)

            # Remove any NaN values that might have been introduced
            enhanced_data = enhanced_data.fillna(method="ffill").fillna(method="bfill")

            return enhanced_data

        except Exception as e:
            print(f"Warning: Technical indicators calculation failed: {e}")
            return data

    def _calculate_moving_averages(self, data):
        """Calculate various moving averages"""
        try:
            # Simple Moving Averages
            data["SMA_5"] = data["Close"].rolling(window=5).mean()
            data["SMA_10"] = data["Close"].rolling(window=10).mean()
            data["SMA_20"] = data["Close"].rolling(window=20).mean()
            data["SMA_50"] = data["Close"].rolling(window=50).mean()
            data["SMA_100"] = data["Close"].rolling(window=100).mean()
            data["SMA_200"] = data["Close"].rolling(window=200).mean()

            # Exponential Moving Averages
            data["EMA_12"] = data["Close"].ewm(span=12).mean()
            data["EMA_26"] = data["Close"].ewm(span=26).mean()
            data["EMA_50"] = data["Close"].ewm(span=50).mean()

            return data
        except Exception as e:
            print(f"Warning: Moving averages calculation failed: {e}")
            return data

    def _calculate_bollinger_bands(self, data):
        """Calculate Bollinger Bands"""
        try:
            # 20-period Bollinger Bands
            sma_20 = data["Close"].rolling(window=20).mean()
            std_20 = data["Close"].rolling(window=20).std()

            data["BB_Upper"] = sma_20 + (std_20 * 2)
            data["BB_Lower"] = sma_20 - (std_20 * 2)
            data["BB_Middle"] = sma_20
            data["BB_Width"] = data["BB_Upper"] - data["BB_Lower"]
            data["BB_Position"] = (data["Close"] - data["BB_Lower"]) / (
                data["BB_Upper"] - data["BB_Lower"]
            )

            return data
        except Exception as e:
            print(f"Warning: Bollinger Bands calculation failed: {e}")
            return data

    def _calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index"""
        try:
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            data["RSI"] = 100 - (100 / (1 + rs))

            return data
        except Exception as e:
            print(f"Warning: RSI calculation failed: {e}")
            return data

    def _calculate_macd(self, data):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            ema_12 = data["Close"].ewm(span=12).mean()
            ema_26 = data["Close"].ewm(span=26).mean()

            data["MACD"] = ema_12 - ema_26
            data["MACD_Signal"] = data["MACD"].ewm(span=9).mean()
            data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]

            return data
        except Exception as e:
            print(f"Warning: MACD calculation failed: {e}")
            return data

    def _calculate_stochastic(self, data, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        try:
            low_min = data["Low"].rolling(window=k_period).min()
            high_max = data["High"].rolling(window=k_period).max()

            data["Stoch_K"] = 100 * ((data["Close"] - low_min) / (high_max - low_min))
            data["Stoch_D"] = data["Stoch_K"].rolling(window=d_period).mean()

            return data
        except Exception as e:
            print(f"Warning: Stochastic calculation failed: {e}")
            return data

    def _calculate_williams_r(self, data, period=14):
        """Calculate Williams %R"""
        try:
            high_max = data["High"].rolling(window=period).max()
            low_min = data["Low"].rolling(window=period).min()

            data["Williams_R"] = -100 * ((high_max - data["Close"]) / (high_max - low_min))

            return data
        except Exception as e:
            print(f"Warning: Williams %R calculation failed: {e}")
            return data

    def _calculate_cci(self, data, period=20):
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )

            data["CCI"] = (typical_price - sma_tp) / (0.015 * mad)

            return data
        except Exception as e:
            print(f"Warning: CCI calculation failed: {e}")
            return data

    def _calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        try:
            high_low = data["High"] - data["Low"]
            high_close = np.abs(data["High"] - data["Close"].shift())
            low_close = np.abs(data["Low"] - data["Close"].shift())

            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            data["ATR"] = true_range.rolling(window=period).mean()

            return data
        except Exception as e:
            print(f"Warning: ATR calculation failed: {e}")
            return data

    def _calculate_obv(self, data):
        """Calculate On-Balance Volume"""
        try:
            obv = []
            obv_value = 0

            for i in range(len(data)):
                if i == 0:
                    obv.append(data["Volume"].iloc[i])
                else:
                    if data["Close"].iloc[i] > data["Close"].iloc[i - 1]:
                        obv_value += data["Volume"].iloc[i]
                    elif data["Close"].iloc[i] < data["Close"].iloc[i - 1]:
                        obv_value -= data["Volume"].iloc[i]
                    # If close is same, OBV doesn't change
                    obv.append(obv_value)

            data["OBV"] = obv

            return data
        except Exception as e:
            print(f"Warning: OBV calculation failed: {e}")
            return data

    def _calculate_volume_sma(self, data):
        """Calculate Volume Simple Moving Averages"""
        try:
            data["Volume_SMA_10"] = data["Volume"].rolling(window=10).mean()
            data["Volume_SMA_20"] = data["Volume"].rolling(window=20).mean()
            data["Volume_Ratio"] = data["Volume"] / data["Volume_SMA_20"]

            return data
        except Exception as e:
            print(f"Warning: Volume SMA calculation failed: {e}")
            return data

    def _calculate_price_changes(self, data):
        """Calculate price change indicators"""
        try:
            # Price changes
            data["Price_Change"] = data["Close"].pct_change()
            data["Price_Change_5d"] = data["Close"].pct_change(periods=5)
            data["Price_Change_10d"] = data["Close"].pct_change(periods=10)

            # Volume changes
            data["Volume_Change"] = data["Volume"].pct_change()

            # High-Low spread
            data["HL_Spread"] = (data["High"] - data["Low"]) / data["Close"]

            # Gap analysis
            data["Gap"] = (data["Open"] - data["Close"].shift()) / data["Close"].shift()

            return data
        except Exception as e:
            print(f"Warning: Price changes calculation failed: {e}")
            return data

    def get_signal_summary(self, data):
        """Get a summary of technical signals"""
        try:
            if data.empty or len(data) < 20:
                return {"error": "Insufficient data for signal analysis"}

            latest = data.iloc[-1]
            signals = {
                "trend_signals": {},
                "momentum_signals": {},
                "volume_signals": {},
                "overall_signal": "NEUTRAL",
            }

            # Trend signals
            if "SMA_20" in data.columns and "SMA_50" in data.columns:
                if latest["Close"] > latest["SMA_20"] > latest["SMA_50"]:
                    signals["trend_signals"]["sma_trend"] = "BULLISH"
                elif latest["Close"] < latest["SMA_20"] < latest["SMA_50"]:
                    signals["trend_signals"]["sma_trend"] = "BEARISH"
                else:
                    signals["trend_signals"]["sma_trend"] = "NEUTRAL"

            # MACD signal
            if "MACD" in data.columns and "MACD_Signal" in data.columns:
                if latest["MACD"] > latest["MACD_Signal"]:
                    signals["momentum_signals"]["macd"] = "BULLISH"
                else:
                    signals["momentum_signals"]["macd"] = "BEARISH"

            # RSI signal
            if "RSI" in data.columns:
                rsi = latest["RSI"]
                if rsi > 70:
                    signals["momentum_signals"]["rsi"] = "OVERBOUGHT"
                elif rsi < 30:
                    signals["momentum_signals"]["rsi"] = "OVERSOLD"
                else:
                    signals["momentum_signals"]["rsi"] = "NEUTRAL"

            # Volume signal
            if "Volume_Ratio" in data.columns:
                vol_ratio = latest["Volume_Ratio"]
                if vol_ratio > 1.5:
                    signals["volume_signals"]["volume"] = "HIGH"
                elif vol_ratio < 0.5:
                    signals["volume_signals"]["volume"] = "LOW"
                else:
                    signals["volume_signals"]["volume"] = "NORMAL"

            # Calculate overall signal
            bullish_count = sum(
                1
                for signal_group in signals.values()
                if isinstance(signal_group, dict)
                for signal in signal_group.values()
                if "BULLISH" in str(signal)
            )

            bearish_count = sum(
                1
                for signal_group in signals.values()
                if isinstance(signal_group, dict)
                for signal in signal_group.values()
                if "BEARISH" in str(signal)
            )

            if bullish_count > bearish_count:
                signals["overall_signal"] = "BULLISH"
            elif bearish_count > bullish_count:
                signals["overall_signal"] = "BEARISH"
            else:
                signals["overall_signal"] = "NEUTRAL"

            return signals

        except Exception as e:
            return {"error": f"Signal analysis failed: {str(e)}"}


class MarketDataValidator:
    """
    Validate and ensure data quality
    """

    @staticmethod
    def validate_data_quality(data):
        """Validate data quality and return quality score"""
        try:
            if data.empty:
                return {"score": 0, "issues": ["No data available"]}

            issues = []
            score = 100

            # Check for missing values
            missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
            if missing_pct > 5:
                issues.append(f"High missing data: {missing_pct:.1f}%")
                score -= 20
            elif missing_pct > 1:
                score -= 10

            # Check for data consistency
            if "High" in data.columns and "Low" in data.columns:
                invalid_hl = (data["High"] < data["Low"]).sum()
                if invalid_hl > 0:
                    issues.append(f"Invalid High/Low data: {invalid_hl} rows")
                    score -= 15

            # Check for extreme outliers
            if "Close" in data.columns:
                price_changes = data["Close"].pct_change().abs()
                extreme_changes = (price_changes > 0.5).sum()
                if extreme_changes > len(data) * 0.01:  # More than 1% extreme changes
                    issues.append(f"Extreme price changes: {extreme_changes} occurrences")
                    score -= 10

            # Check data recency
            if len(data) > 0:
                last_date = data.index[-1]
                days_old = (datetime.now() - last_date).days
                if days_old > 7:
                    issues.append(f"Data is {days_old} days old")
                    score -= min(20, days_old)

            # Check data completeness
            if len(data) < 30:
                issues.append(f"Limited data: only {len(data)} data points")
                score -= 20

            return {
                "score": max(0, score),
                "issues": issues,
                "data_points": len(data),
                "missing_percentage": missing_pct,
                "quality_grade": (
                    "A"
                    if score >= 90
                    else (
                        "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"
                    )
                ),
            }

        except Exception as e:
            return {"score": 0, "issues": [f"Validation error: {str(e)}"]}

    @staticmethod
    def clean_and_validate(data):
        """Clean data and return validation results"""
        try:
            original_length = len(data)

            # Remove invalid data
            if "High" in data.columns and "Low" in data.columns:
                data = data[data["High"] >= data["Low"]]

            if "Close" in data.columns:
                data = data[data["Close"] > 0]

            # Remove extreme outliers
            if "Close" in data.columns and len(data) > 10:
                price_changes = data["Close"].pct_change().abs()
                data = data[price_changes < 0.5]  # Remove changes > 50%

            # Forward fill missing values
            data = data.fillna(method="ffill")

            # Remove any remaining NaN rows
            data = data.dropna()

            cleaned_length = len(data)
            removed_rows = original_length - cleaned_length

            validation_result = MarketDataValidator.validate_data_quality(data)
            validation_result["rows_removed"] = removed_rows
            validation_result["removal_percentage"] = (
                (removed_rows / original_length * 100) if original_length > 0 else 0
            )

            return data, validation_result

        except Exception as e:
            return data, {"score": 0, "issues": [f"Cleaning error: {str(e)}"]}
