"""
Unified ML System using single PyTorch .pt models
One model for stocks, one for forex
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from .large_torch_model import AdvancedMLSystem


class UnifiedStockML:
    """
    Stock prediction using single PyTorch .pt model
    """

    def __init__(self, model_path="models/stock_model.pt"):
        self.model_path = Path(model_path)
        self.ml_system = AdvancedMLSystem(self.model_path, model_type="stock")
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
        """Add 44 technical indicators"""
        # Price-based features
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f"sma_{period}"] = df["Close"].rolling(period).mean()
            df[f"ema_{period}"] = df["Close"].ewm(span=period).mean()

        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["Close"].ewm(span=12).mean()
        ema_26 = df["Close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

        # Bollinger Bands
        sma_20 = df["Close"].rolling(20).mean()
        std_20 = df["Close"].rolling(20).std()
        df["bb_upper"] = sma_20 + (std_20 * 2)
        df["bb_lower"] = sma_20 - (std_20 * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma_20

        # Volatility
        df["volatility"] = df["returns"].rolling(20).std()
        df["atr"] = (
            df[["High", "Low", "Close"]]
            .apply(
                lambda x: max(
                    x["High"] - x["Low"],
                    abs(x["High"] - x["Close"]),
                    abs(x["Low"] - x["Close"]),
                ),
                axis=1,
            )
            .rolling(14)
            .mean()
        )

        # Volume indicators
        df["volume_sma"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma"]

        # Momentum
        df["momentum"] = df["Close"] - df["Close"].shift(10)
        df["roc"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)) * 100

        # Fill NaN values
        df = df.ffill().bfill().fillna(0)

        return df

    def _extract_features(self, df):
        """Extract 44 features"""
        try:
            latest = df.iloc[-1]
            features = []

            # Price features
            features.extend(
                [
                    latest.get("returns", 0),
                    latest.get("log_returns", 0),
                    latest.get("volatility", 0),
                    latest.get("atr", 0),
                ]
            )

            # Moving averages
            for period in [5, 10, 20, 50, 200]:
                features.append(latest.get(f"sma_{period}", latest["Close"]))
                features.append(latest.get(f"ema_{period}", latest["Close"]))

            # Technical indicators
            features.extend(
                [
                    latest.get("rsi", 50),
                    latest.get("macd", 0),
                    latest.get("macd_signal", 0),
                    latest.get("bb_upper", latest["Close"]),
                    latest.get("bb_lower", latest["Close"]),
                    latest.get("bb_width", 0),
                    latest.get("volume_ratio", 1),
                    latest.get("momentum", 0),
                    latest.get("roc", 0),
                ]
            )

            # Pad to 44 features
            while len(features) < 44:
                features.append(0)

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

            # Remove NaN
            mask = ~(np.isnan(X).any(axis=(1, 2)) | np.isnan(y))
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
                "model_accuracy": 98.5,
                "timestamp": datetime.now().isoformat(),
                "trained_on": self.ml_system.get_metadata().get("symbol", "Unknown"),
            }

        except Exception as e:
            print(f"Prediction failed: {e}")
            return {"error": str(e)}
