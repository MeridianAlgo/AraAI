"""
Utility classes for GPU management, caching, and accuracy tracking
"""

import json
import pickle
import torch
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class GPUManager:
    """
    Enhanced GPU detection and management for multiple vendors
    """

    def __init__(self):
        self.gpu_info = self.detect_gpu_vendor()
        self.device = self.get_device()

    def detect_gpu_vendor(self):
        """Detect available GPU acceleration"""
        gpu_info = {
            "vendor": "CPU",
            "device_name": "CPU",
            "available": False,
            "details": [],
        }

        try:
            # NVIDIA CUDA
            if torch.cuda.is_available():
                gpu_info["vendor"] = "NVIDIA"
                gpu_info["device_name"] = torch.cuda.get_device_name(0)
                gpu_info["available"] = True
                gpu_info["details"].append(f"NVIDIA CUDA: {gpu_info['device_name']}")
                gpu_info["details"].append(f"CUDA Version: {torch.version.cuda}")
                gpu_info["details"].append(
                    f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )
                return gpu_info
        except Exception:
            pass

        try:
            # Apple MPS (Metal Performance Shaders)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu_info["vendor"] = "Apple"
                gpu_info["device_name"] = "Apple MPS"
                gpu_info["available"] = True
                gpu_info["details"].append("Apple MPS (Metal Performance Shaders)")
                return gpu_info
        except Exception:
            pass

        try:
            # Intel XPU (experimental)
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                gpu_info["vendor"] = "Intel"
                gpu_info["device_name"] = "Intel XPU"
                gpu_info["available"] = True
                gpu_info["details"].append("Intel XPU acceleration")
                return gpu_info
        except Exception:
            pass

        # CPU fallback
        gpu_info["details"].append("CPU-only processing (no GPU acceleration)")
        return gpu_info

    def get_device(self):
        """Get the best available device"""
        if self.gpu_info["available"]:
            if self.gpu_info["vendor"] == "NVIDIA":
                return torch.device("cuda")
            elif self.gpu_info["vendor"] == "Apple":
                return torch.device("mps")
            elif self.gpu_info["vendor"] == "Intel":
                return torch.device("xpu")

        return torch.device("cpu")

    def get_device_info(self):
        """Get detailed device information"""
        return self.gpu_info


class CacheManager:
    """
    Intelligent caching system for predictions and market data
    """

    def __init__(self, cache_dir=".ara_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.predictions_file = self.cache_dir / "predictions.json"
        self.market_data_dir = self.cache_dir / "market_data"
        self.market_data_dir.mkdir(exist_ok=True)

        # Cache settings
        self.prediction_cache_hours = 6
        self.market_data_cache_minutes = 15

    def save_predictions(self, symbol, predictions):
        """Save predictions to cache"""
        try:
            # Load existing cache
            cache_data = self._load_predictions_cache()

            # Add new predictions
            cache_data[symbol] = {
                "predictions": predictions,
                "timestamp": datetime.now().isoformat(),
                "cache_type": "predictions",
            }

            # Save updated cache
            with open(self.predictions_file, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)

        except Exception as e:
            print(f"Warning: Failed to save predictions to cache: {e}")

    def check_cached_predictions(self, symbol, days):
        """Check for cached predictions"""
        try:
            cache_data = self._load_predictions_cache()

            if symbol not in cache_data:
                return None

            cached_entry = cache_data[symbol]
            cache_time = datetime.fromisoformat(cached_entry["timestamp"])

            # Check if cache is still valid
            if datetime.now() - cache_time < timedelta(hours=self.prediction_cache_hours):
                cached_predictions = cached_entry["predictions"]

                # Check if we have enough predictions
                if (
                    isinstance(cached_predictions, dict)
                    and "predictions" in cached_predictions
                    and len(cached_predictions["predictions"]) >= days
                ):
                    return cached_entry

            return None

        except Exception as e:
            print(f"Warning: Failed to check cached predictions: {e}")
            return None

    def save_market_data(self, symbol, data):
        """Save market data to cache"""
        try:
            cache_file = self.market_data_dir / f"{symbol}.pkl"

            cache_entry = {
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
            }

            with open(cache_file, "wb") as f:
                pickle.dump(cache_entry, f)

        except Exception as e:
            print(f"Warning: Failed to save market data to cache: {e}")

    def load_market_data(self, symbol):
        """Load market data from cache"""
        try:
            cache_file = self.market_data_dir / f"{symbol}.pkl"

            if not cache_file.exists():
                return None

            with open(cache_file, "rb") as f:
                cache_entry = pickle.load(f)

            cache_time = datetime.fromisoformat(cache_entry["timestamp"])

            # Check if cache is still valid
            if datetime.now() - cache_time < timedelta(minutes=self.market_data_cache_minutes):
                return cache_entry["data"]

            return None

        except Exception as e:
            print(f"Warning: Failed to load market data from cache: {e}")
            return None

    def _load_predictions_cache(self):
        """Load predictions cache"""
        try:
            if self.predictions_file.exists():
                with open(self.predictions_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}

    def ask_user_choice(self, symbol, cached_result):
        """Ask user whether to use cached predictions"""
        try:
            cache_age = self._calculate_cache_age(cached_result.get("timestamp"))
            print(f"\nFound cached predictions for {symbol} (Age: {cache_age})")

            # For now, automatically use cache if available and recent
            cache_time = datetime.fromisoformat(cached_result["timestamp"])
            if datetime.now() - cache_time < timedelta(hours=2):
                return "use_cached"
            else:
                return "generate_new"

        except Exception:
            return "generate_new"

    def _calculate_cache_age(self, timestamp_str):
        """Calculate cache age"""
        try:
            if not timestamp_str:
                return "Unknown"

            timestamp = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - timestamp

            if age.days > 0:
                return f"{age.days}d {age.seconds // 3600}h"
            else:
                return f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m"

        except Exception:
            return "Unknown"

    def cleanup_old_cache(self):
        """Clean up old cache files"""
        try:
            # Clean old predictions
            cache_data = self._load_predictions_cache()
            cleaned_data = {}

            for symbol, entry in cache_data.items():
                try:
                    cache_time = datetime.fromisoformat(entry["timestamp"])
                    if datetime.now() - cache_time < timedelta(days=7):
                        cleaned_data[symbol] = entry
                except Exception:
                    continue

            with open(self.predictions_file, "w") as f:
                json.dump(cleaned_data, f, indent=2, default=str)

            # Clean old market data
            for cache_file in self.market_data_dir.glob("*.pkl"):
                try:
                    with open(cache_file, "rb") as f:
                        cache_entry = pickle.load(f)

                    cache_time = datetime.fromisoformat(cache_entry["timestamp"])
                    if datetime.now() - cache_time > timedelta(days=1):
                        cache_file.unlink()

                except Exception:
                    continue

        except Exception as e:
            print(f"Warning: Cache cleanup failed: {e}")

    def get_cache_stats(self):
        """Get cache statistics"""
        try:
            stats = {
                "total_predictions": 0,
                "symbols": 0,
                "file_size": 0,
                "market_data_files": 0,
            }

            # Predictions stats
            if self.predictions_file.exists():
                cache_data = self._load_predictions_cache()
                stats["total_predictions"] = len(cache_data)
                stats["symbols"] = len(cache_data)
                stats["file_size"] = self.predictions_file.stat().st_size

            # Market data stats
            stats["market_data_files"] = len(list(self.market_data_dir.glob("*.pkl")))

            return stats

        except Exception:
            return {
                "total_predictions": 0,
                "symbols": 0,
                "file_size": 0,
                "market_data_files": 0,
            }


class AccuracyTracker:
    """
    Track and analyze prediction accuracy over time
    """

    def __init__(self, data_dir=".ara_cache"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.accuracy_file = self.data_dir / "accuracy_tracking.json"
        self.validation_file = self.data_dir / "validation_results.json"

    def record_prediction(self, symbol, prediction_data):
        """Record a prediction for future validation"""
        try:
            # Load existing records
            records = self._load_accuracy_data()

            # Create record entry
            record_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            records[record_id] = {
                "symbol": symbol,
                "prediction_date": datetime.now().isoformat(),
                "predictions": prediction_data.get("predictions", []),
                "current_price": prediction_data.get("current_price", 0),
                "validated": False,
                "validation_date": None,
                "accuracy_metrics": {},
            }

            # Save updated records
            with open(self.accuracy_file, "w") as f:
                json.dump(records, f, indent=2, default=str)

        except Exception as e:
            print(f"Warning: Failed to record prediction: {e}")

    def validate_predictions(self):
        """Validate previous predictions against actual prices"""
        try:
            records = self._load_accuracy_data()
            validation_results = self._load_validation_data()

            validated_count = 0

            for record_id, record in records.items():
                if record["validated"]:
                    continue

                # Check if enough time has passed for validation
                prediction_date = datetime.fromisoformat(record["prediction_date"])
                days_passed = (datetime.now() - prediction_date).days

                if days_passed >= 1:  # At least 1 day has passed
                    # Validate this prediction
                    validation_result = self._validate_single_prediction(record)

                    if validation_result:
                        # Update record
                        records[record_id]["validated"] = True
                        records[record_id]["validation_date"] = datetime.now().isoformat()
                        records[record_id]["accuracy_metrics"] = validation_result

                        # Store validation result
                        validation_results[record_id] = validation_result
                        validated_count += 1

            # Save updated data
            with open(self.accuracy_file, "w") as f:
                json.dump(records, f, indent=2, default=str)

            with open(self.validation_file, "w") as f:
                json.dump(validation_results, f, indent=2, default=str)

            # Calculate summary statistics
            summary = self._calculate_validation_summary(validation_results)
            summary["validated"] = validated_count

            return summary

        except Exception as e:
            print(f"Warning: Validation failed: {e}")
            return {"validated": 0, "accuracy_rate": 0, "avg_error": 0}

    def _validate_single_prediction(self, record):
        """Validate a single prediction record"""
        try:
            import yfinance as yf

            symbol = record["symbol"]
            prediction_date = datetime.fromisoformat(record["prediction_date"])
            predictions = record["predictions"]

            if not predictions:
                return None

            # Get actual prices for validation
            ticker = yf.Ticker(symbol)

            # Get data from prediction date onwards
            start_date = prediction_date.date()
            end_date = (prediction_date + timedelta(days=len(predictions) + 5)).date()

            actual_data = ticker.history(start=start_date, end=end_date)

            if actual_data.empty:
                return None

            # Calculate accuracy for each prediction
            validation_results = {
                "symbol": symbol,
                "prediction_date": record["prediction_date"],
                "validation_date": datetime.now().isoformat(),
                "daily_accuracies": [],
                "overall_accuracy": 0,
                "avg_error": 0,
            }

            total_error = 0
            valid_predictions = 0

            for i, pred in enumerate(predictions):
                target_date = prediction_date + timedelta(days=i + 1)

                # Find actual price for this date
                actual_price = None
                for date_idx in range(len(actual_data)):
                    data_date = actual_data.index[date_idx].date()
                    if data_date >= target_date.date():
                        actual_price = actual_data["Close"].iloc[date_idx]
                        break

                if actual_price is not None:
                    predicted_price = pred.get("predicted_price", 0)
                    error = abs(predicted_price - actual_price) / actual_price * 100

                    daily_accuracy = {
                        "day": i + 1,
                        "predicted_price": predicted_price,
                        "actual_price": actual_price,
                        "error_percent": error,
                        "accurate": error < 3.0,  # Within 3% is considered accurate
                    }

                    validation_results["daily_accuracies"].append(daily_accuracy)
                    total_error += error
                    valid_predictions += 1

            if valid_predictions > 0:
                validation_results["avg_error"] = total_error / valid_predictions
                accurate_count = sum(
                    1 for acc in validation_results["daily_accuracies"] if acc["accurate"]
                )
                validation_results["overall_accuracy"] = (accurate_count / valid_predictions) * 100

                return validation_results

            return None

        except Exception as e:
            print(f"Warning: Single prediction validation failed: {e}")
            return None

    def _calculate_validation_summary(self, validation_results):
        """Calculate summary statistics from validation results"""
        try:
            if not validation_results:
                return {
                    "accuracy_rate": 0,
                    "excellent_rate": 0,
                    "good_rate": 0,
                    "avg_error": 0,
                }

            total_predictions = 0
            accurate_predictions = 0
            excellent_predictions = 0  # < 1% error
            good_predictions = 0  # < 2% error
            total_error = 0

            for result in validation_results.values():
                for daily_acc in result.get("daily_accuracies", []):
                    total_predictions += 1
                    error = daily_acc["error_percent"]
                    total_error += error

                    if error < 3.0:
                        accurate_predictions += 1
                    if error < 1.0:
                        excellent_predictions += 1
                    if error < 2.0:
                        good_predictions += 1

            if total_predictions > 0:
                return {
                    "accuracy_rate": (accurate_predictions / total_predictions) * 100,
                    "excellent_rate": (excellent_predictions / total_predictions) * 100,
                    "good_rate": (good_predictions / total_predictions) * 100,
                    "avg_error": total_error / total_predictions,
                    "total_predictions": total_predictions,
                }

            return {
                "accuracy_rate": 0,
                "excellent_rate": 0,
                "good_rate": 0,
                "avg_error": 0,
                "total_predictions": 0,
            }

        except Exception as e:
            print(f"Warning: Summary calculation failed: {e}")
            return {
                "accuracy_rate": 0,
                "excellent_rate": 0,
                "good_rate": 0,
                "avg_error": 0,
                "total_predictions": 0,
            }

    def analyze_accuracy(self, symbol=None):
        """Analyze accuracy for a specific symbol or all symbols"""
        try:
            validation_results = self._load_validation_data()

            if symbol:
                # Filter for specific symbol
                filtered_results = {
                    k: v
                    for k, v in validation_results.items()
                    if v.get("symbol", "").upper() == symbol.upper()
                }
            else:
                filtered_results = validation_results

            summary = self._calculate_validation_summary(filtered_results)
            summary["symbol"] = symbol if symbol else "All"

            # Add recent performance (last 30 days)
            recent_results = {}
            cutoff_date = datetime.now() - timedelta(days=30)

            for k, v in filtered_results.items():
                validation_date = datetime.fromisoformat(v["validation_date"])
                if validation_date >= cutoff_date:
                    recent_results[k] = v

            if recent_results:
                recent_summary = self._calculate_validation_summary(recent_results)
                summary["recent_stats"] = recent_summary

            return summary

        except Exception as e:
            print(f"Warning: Accuracy analysis failed: {e}")
            return {"symbol": symbol or "All", "accuracy_rate": 0, "avg_error": 0}

    def _load_accuracy_data(self):
        """Load accuracy tracking data"""
        try:
            if self.accuracy_file.exists():
                with open(self.accuracy_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}

    def _load_validation_data(self):
        """Load validation results data"""
        try:
            if self.validation_file.exists():
                with open(self.validation_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}

    def get_accuracy_stats(self):
        """Get overall accuracy statistics"""
        try:
            validation_results = self._load_validation_data()
            return self._calculate_validation_summary(validation_results)
        except Exception:
            return {"accuracy_rate": 0, "avg_error": 0, "total_predictions": 0}

    def cleanup_old_data(self):
        """Clean up old tracking data"""
        try:
            # Clean accuracy data older than 90 days
            records = self._load_accuracy_data()
            cleaned_records = {}

            cutoff_date = datetime.now() - timedelta(days=90)

            for record_id, record in records.items():
                try:
                    prediction_date = datetime.fromisoformat(record["prediction_date"])
                    if prediction_date >= cutoff_date:
                        cleaned_records[record_id] = record
                except Exception:
                    continue

            with open(self.accuracy_file, "w") as f:
                json.dump(cleaned_records, f, indent=2, default=str)

            # Clean validation data
            validation_results = self._load_validation_data()
            cleaned_validation = {}

            for result_id, result in validation_results.items():
                try:
                    validation_date = datetime.fromisoformat(result["validation_date"])
                    if validation_date >= cutoff_date:
                        cleaned_validation[result_id] = result
                except Exception:
                    continue

            with open(self.validation_file, "w") as f:
                json.dump(cleaned_validation, f, indent=2, default=str)

        except Exception as e:
            print(f"Warning: Data cleanup failed: {e}")
