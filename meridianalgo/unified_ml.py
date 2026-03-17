"""
Unified ML System using single PyTorch .pt models
One model for stocks, one for forex
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

from .large_torch_model import AdvancedMLSystem  # noqa: E402


class UnifiedStockML:
    """
    Stock prediction using single PyTorch .pt model
    """

    def __init__(self, model_path="models/stock_model.pt", model_type="stock"):
        self.model_path = Path(model_path)
        self.ml_system = AdvancedMLSystem(self.model_path, model_type=model_type)
        self.feature_count = 44

    def get_model_status(self):
        """Get model status"""
        return {
            "is_trained": self.ml_system.is_trained(),
            "model_path": str(self.model_path),
            "feature_count": self.feature_count,
            "metadata": self.ml_system.get_metadata(),
        }

    def _add_indicators(self, df):
        """Add 44 real technical indicators — no zero padding"""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"].replace(0, 1)  # Avoid div-by-zero for forex

        # === 1-4: Price-based features ===
        df["returns"] = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))
        df["volatility"] = df["returns"].rolling(20).std()

        # True Range and ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # === 5-14: Moving averages (SMA + EMA for 5 periods) ===
        for period in [5, 10, 20, 50, 200]:
            df[f"sma_{period}"] = close.rolling(period).mean()
            df[f"ema_{period}"] = close.ewm(span=period).mean()

        # === 15-17: RSI variants ===
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss_val
        df["rsi"] = 100 - (100 / (1 + rs))
        # Fast RSI (7-period)
        gain_fast = delta.where(delta > 0, 0).rolling(7).mean()
        loss_fast = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs_fast = gain_fast / loss_fast
        df["rsi_fast"] = 100 - (100 / (1 + rs_fast))
        # Stochastic RSI
        rsi_series = df["rsi"]
        rsi_min = rsi_series.rolling(14).min()
        rsi_max = rsi_series.rolling(14).max()
        df["stoch_rsi"] = (rsi_series - rsi_min) / (rsi_max - rsi_min + 1e-8)

        # === 18-20: MACD ===
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # === 21-24: Bollinger Bands ===
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        df["bb_upper"] = sma_20 + (std_20 * 2)
        df["bb_lower"] = sma_20 - (std_20 * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma_20 + 1e-8)
        df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)

        # === 25-27: Volume indicators ===
        df["volume_sma"] = volume.rolling(20).mean()
        df["volume_ratio"] = volume / (df["volume_sma"] + 1e-8)
        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df["obv_norm"] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-8)

        # === 28-30: Momentum ===
        df["momentum"] = close - close.shift(10)
        df["roc"] = ((close - close.shift(10)) / (close.shift(10) + 1e-8)) * 100
        # Williams %R
        highest_14 = high.rolling(14).max()
        lowest_14 = low.rolling(14).min()
        df["williams_r"] = ((highest_14 - close) / (highest_14 - lowest_14 + 1e-8)) * -100

        # === 31-33: Stochastic Oscillator ===
        df["stoch_k"] = ((close - lowest_14) / (highest_14 - lowest_14 + 1e-8)) * 100
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()
        # CCI (Commodity Channel Index)
        typical_price = (high + low + close) / 3
        tp_sma = typical_price.rolling(20).mean()
        tp_mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df["cci"] = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-8)

        # === 34-36: Keltner Channels ===
        kc_mid = close.ewm(span=20).mean()
        kc_atr = df["atr"]
        df["kc_upper"] = kc_mid + (2 * kc_atr)
        df["kc_lower"] = kc_mid - (2 * kc_atr)
        df["kc_pct"] = (close - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"] + 1e-8)

        # === 37-39: ADX (Average Directional Index) ===
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr_14 = df["atr"]
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr_14 + 1e-8))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr_14 + 1e-8))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8))
        df["adx"] = dx.rolling(14).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        # === 40-42: Price position and trend strength ===
        df["price_vs_sma50"] = (close - df["sma_50"]) / (df["sma_50"] + 1e-8)
        df["price_vs_sma200"] = (close - df["sma_200"]) / (df["sma_200"] + 1e-8)
        # Avg True Range as % of price (normalized volatility)
        df["atr_pct"] = df["atr"] / (close + 1e-8)

        # === 43-44: Mean reversion signals ===
        df["zscore_20"] = (close - sma_20) / (std_20 + 1e-8)
        # Distance from 52-week high/low (normalized)
        high_252 = close.rolling(252, min_periods=20).max()
        low_252 = close.rolling(252, min_periods=20).min()
        df["dist_from_high"] = (close - high_252) / (high_252 + 1e-8)

        # Fill NaN and Inf values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)

        return df

    # Ordered list of all 44 feature columns (must match _add_indicators output)
    FEATURE_COLS = [
        # 1-4: Price-based
        "returns", "log_returns", "volatility", "atr",
        # 5-14: Moving averages
        "sma_5", "ema_5", "sma_10", "ema_10", "sma_20", "ema_20",
        "sma_50", "ema_50", "sma_200", "ema_200",
        # 15-17: RSI variants
        "rsi", "rsi_fast", "stoch_rsi",
        # 18-20: MACD
        "macd", "macd_signal", "macd_hist",
        # 21-24: Bollinger Bands
        "bb_upper", "bb_lower", "bb_width", "bb_pct",
        # 25-27: Volume
        "volume_sma", "volume_ratio", "obv_norm",
        # 28-30: Momentum
        "momentum", "roc", "williams_r",
        # 31-33: Stochastic + CCI
        "stoch_k", "stoch_d", "cci",
        # 34-36: Keltner Channels
        "kc_upper", "kc_lower", "kc_pct",
        # 37-39: ADX
        "adx", "plus_di", "minus_di",
        # 40-42: Trend position
        "price_vs_sma50", "price_vs_sma200", "atr_pct",
        # 43-44: Mean reversion
        "zscore_20", "dist_from_high",
    ]

    def _extract_features(self, df):
        """Extract all 44 features from indicator columns"""
        try:
            latest = df.iloc[-1]
            features = [float(latest.get(col, 0)) for col in self.FEATURE_COLS]
            return np.array(features[:44])
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(44)

    def train_from_dataset(
        self,
        dataset_path,
        symbol_name,
        epochs=60,
        batch_size=64,
        lr=0.0005,
        validation_split=0.2,
        metadata=None,
        **kwargs,
    ):
        """
        Train from CSV dataset with metadata

        Args:
            dataset_path: Path to CSV file
            symbol_name: Symbol name
            epochs: Number of training epochs/steps
            batch_size: Batch size
            lr: Learning rate
            validation_split: Validation split ratio
            metadata: Additional metadata (symbol, timeframe, asset_type, etc.)
        """
        try:
            print(f"Loading dataset from {dataset_path}...")

            # Load dataset
            df = pd.read_csv(dataset_path)

            # Validate columns
            required_cols = ["Date", "Open", "High", "Low", "Close"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"Error: Missing columns: {missing}")
                return {"success": False, "error": f"Missing columns: {missing}"}

            # Add Volume if missing
            if "Volume" not in df.columns:
                df["Volume"] = 1000000

            # Convert Date and set index
            df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
            df = df.dropna(subset=["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)

            print(f"Dataset loaded: {len(df)} rows")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")

            # Add indicators
            df = self._add_indicators(df)

            lookback = int(kwargs.get("lookback", 30))

            # Prepare training data
            X = []
            y = []

            for i in range(lookback, len(df) - 1):
                window_features = []
                for j in range(i - lookback + 1, i + 1):
                    window_features.append(self._extract_features(df.iloc[: j + 1]))

                target = (df["Close"].iloc[i + 1] - df["Close"].iloc[i]) / df["Close"].iloc[i]

                X.append(np.array(window_features))
                y.append(target)

            X = np.array(X)
            y = np.array(y)

            # Remove NaN and Inf to prevent training explosion
            mask = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(y)
            X = X[mask]
            y = y[mask]

            if len(X) == 0:
                print("Error: No valid training samples")
                return {"success": False, "error": "No valid training samples"}

            print(f"Training samples: {len(X)}")

            # Add metadata to ML system
            if metadata:
                for key, value in metadata.items():
                    self.ml_system.metadata[key] = value

            self.ml_system.metadata["lookback"] = lookback

            # Train large PyTorch model with validation
            result = self.ml_system.train(
                X,
                y,
                symbol_name,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                validation_split=validation_split,
                cpu_limit=kwargs.get("cpu_limit", 80),
                comet_experiment=kwargs.get("comet_experiment"),
            )

            return result

        except Exception as e:
            print(f"Training failed: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def train_ultimate_models(self, target_symbol, period="2y", **kwargs):
        """Train from online data or custom dataframe"""
        try:
            custom_data = kwargs.get("custom_data")

            if custom_data is not None:
                print(f"Using provided custom data for {target_symbol}...")
                data = custom_data
            else:
                print(f"Downloading data for {target_symbol} ({period})...")
                ticker = yf.Ticker(target_symbol)
                data = ticker.history(period=period)

            if len(data) < 100:
                print(f"Insufficient data: {len(data)} days")
                return {
                    "success": False,
                    "error": f"Insufficient data: {len(data)} days",
                }

            # Save to temp CSV and train
            temp_csv = Path("datasets/temp_train.csv")
            temp_csv.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(temp_csv)

            result = self.train_from_dataset(str(temp_csv), target_symbol, **kwargs)

            # Clean up temp file
            if temp_csv.exists():
                temp_csv.unlink()

            return result

        except Exception as e:
            print(f"Training failed: {e}")
            return {"success": False, "error": str(e)}

    def predict_ultimate(self, symbol, days=5):
        """Make predictions"""
        try:
            if not self.ml_system.is_trained():
                print("Model not trained!")
                return None

            # Get data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")

            if len(data) < 30:
                return {"error": "Insufficient data"}

            current_price = data["Close"].iloc[-1]
            data = self._add_indicators(data)

            # Get historical volatility
            hist_volatility = data["Close"].pct_change().std()

            lookback = int(self.ml_system.get_metadata().get("lookback", 30))
            if len(data) < lookback:
                return {"error": "Insufficient data"}

            # Make predictions
            predictions = []
            window_features = []
            start = len(data) - lookback
            for i in range(start, len(data)):
                window_features.append(self._extract_features(data.iloc[: i + 1]))
            current_features = np.array(window_features)
            pred_price = current_price

            for day in range(1, days + 1):
                # Predict
                pred_return, _ = self.ml_system.predict(current_features.reshape(1, lookback, -1))
                pred_return = (
                    float(pred_return) if np.isscalar(pred_return) else float(pred_return[0])
                )

                # Apply volatility bounds
                max_daily_move = hist_volatility * 2.5
                pred_return = np.clip(pred_return, -max_daily_move, max_daily_move)

                pred_price = float(pred_price * (1 + pred_return))

                # Confidence
                confidence = max(0.5, 1.0 - 0.08 * (day - 1))

                predictions.append(
                    {
                        "day": day,
                        "date": (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d"),
                        "predicted_price": pred_price,
                        "predicted_return": pred_return,
                        "confidence": confidence,
                    }
                )

                # Update features
                current_features = current_features * (1 + pred_return * 0.5)

            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "predictions": predictions,
                "model_accuracy": self.ml_system.get_metadata().get("direction_accuracy", "N/A"),
                "timestamp": datetime.now().isoformat(),
                "trained_on": self.ml_system.get_metadata().get("symbol", "Unknown"),
            }

        except Exception as e:
            print(f"Prediction failed: {e}")
            return {"error": str(e)}
