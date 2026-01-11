"""
Core Ara AI functionality - main prediction engine with ensemble ML system
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

from .advanced_ml import AdvancedEnsembleSystem, ChartPatternRecognizer
from .company_analysis import CompanyAnalyzer
from .ai_analysis import LightweightAIAnalyzer
from .fast_ml import FastMLPredictor, FastPatternRecognizer, FastAnalyzer
from .data import MarketDataManager, TechnicalIndicators
from .utils import GPUManager, CacheManager, AccuracyTracker
from .console import ConsoleManager


class AraAI:
    """
    Main Ara AI class with enhanced ensemble ML system and intelligent caching
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.console = ConsoleManager(verbose=verbose)
        self.gpu_manager = GPUManager()
        self.cache_manager = CacheManager()
        self.accuracy_tracker = AccuracyTracker()
        self.data_manager = MarketDataManager(cache_manager=self.cache_manager)
        self.indicators = TechnicalIndicators()

        # Fast ML system (no training required, works immediately)
        self.fast_ml = FastMLPredictor()
        self.fast_patterns = FastPatternRecognizer()
        self.fast_analyzer = FastAnalyzer()

        # Advanced ML system (fallback for complex analysis)
        self.ml_system = AdvancedEnsembleSystem(device=self.gpu_manager.get_device())
        self.pattern_recognizer = ChartPatternRecognizer()
        self.company_analyzer = CompanyAnalyzer()

        # AI analyzer (for detailed analysis when requested)
        gpu_available = self.gpu_manager.gpu_info.get("available", False)
        self.ai_analyzer = LightweightAIAnalyzer(use_gpu=gpu_available)

        # Initialize system
        self._initialize_system()

    def _initialize_system(self):
        """Initialize the Ara AI system"""
        if self.verbose:
            self.console.print_system_info()
            self.console.print_gpu_info(self.gpu_manager.detect_gpu_vendor())

    def predict(self, symbol, days=5, use_cache=True, include_analysis=True):
        """
        Enhanced prediction function with advanced ML, pattern recognition, and company analysis

        Args:
            symbol (str): Stock symbol
            days (int): Number of days to predict
            use_cache (bool): Whether to use cached predictions
            include_analysis (bool): Whether to include comprehensive company analysis

        Returns:
            dict: Enhanced prediction results with patterns and company analysis
        """
        try:
            # Input validation
            if not symbol or not isinstance(symbol, str):
                raise ValueError("Symbol must be a non-empty string")

            if not isinstance(days, int) or days < 1 or days > 30:
                raise ValueError("Days must be an integer between 1 and 30")

            symbol = symbol.upper().strip()

            # Check for existing predictions
            if use_cache:
                try:
                    cached_result = self.cache_manager.check_cached_predictions(
                        symbol, days
                    )
                    if cached_result:
                        choice = self.cache_manager.ask_user_choice(
                            symbol, cached_result
                        )
                        if choice == "use_cached":
                            return self._format_cached_result(cached_result)
                except Exception as cache_error:
                    if self.verbose:
                        self.console.print_warning(f"Cache check failed: {cache_error}")

            # Generate new predictions with advanced analysis
            return self._generate_enhanced_predictions(symbol, days, include_analysis)

        except ValueError as e:
            self.console.print_error(f"Invalid input: {e}")
            return None
        except Exception as e:
            self.console.print_error(f"Prediction failed for {symbol}: {e}")
            if self.verbose:
                import traceback

                self.console.print_error(f"Detailed error: {traceback.format_exc()}")
            return None

    def _generate_enhanced_predictions(self, symbol, days, include_analysis=True):
        """Generate enhanced predictions with advanced ML, patterns, and company analysis"""
        try:
            # Get market data with retry logic
            data = None
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    data = self.data_manager.get_stock_data(symbol)
                    if data is not None and not data.empty:
                        break
                except Exception as data_error:
                    if attempt == max_retries - 1:
                        raise ValueError(
                            f"Failed to fetch data for {symbol} after {max_retries} attempts: {data_error}"
                        )
                    if self.verbose:
                        self.console.print_warning(
                            f"Data fetch attempt {attempt + 1} failed, retrying..."
                        )

            if data is None or data.empty:
                raise ValueError(f"No market data available for {symbol}")

            if len(data) < 50:
                raise ValueError(
                    f"Insufficient data for {symbol} (need at least 50 data points, got {len(data)})"
                )

            # Calculate technical indicators with error handling
            try:
                enhanced_data = self.indicators.calculate_all_indicators(data)
            except Exception as indicator_error:
                if self.verbose:
                    self.console.print_warning(
                        f"Technical indicator calculation failed: {indicator_error}"
                    )
                # Use basic data if indicators fail
                enhanced_data = data

            # Prepare features for ML models
            features = self._prepare_features(enhanced_data)
            if features is None:
                raise ValueError("Failed to prepare features for ML models")

            # Generate fast predictions (primary method)
            try:
                current_price = enhanced_data["Close"].iloc[-1]

                # Use fast ML system (no training required)
                fast_result = self.fast_ml.predict_fast(
                    enhanced_data, current_price, days=days
                )

                if fast_result and "predictions" in fast_result:
                    predictions = fast_result["predictions"]
                    confidence_scores = fast_result.get(
                        "confidence_scores", [0.75] * days
                    )

                    # Fast pattern recognition
                    patterns = self.fast_patterns.detect_patterns_fast(
                        enhanced_data["Close"]
                    )

                    if self.verbose:
                        self.console.print_success(
                            f"Fast prediction completed ({fast_result.get('processing_time', 'fast')})"
                        )
                else:
                    raise ValueError("Fast ML system failed")

            except Exception as fast_error:
                if self.verbose:
                    self.console.print_warning(
                        f"Fast ML failed, using fallback: {fast_error}"
                    )

                # Fallback to simple trend prediction
                try:
                    current_price = enhanced_data["Close"].iloc[-1]
                    predictions = []

                    # Simple trend calculation
                    if len(enhanced_data) >= 10:
                        recent_trend = (
                            enhanced_data["Close"].tail(10).pct_change().mean()
                        )
                    else:
                        recent_trend = 0.001  # Small positive trend

                    for i in range(days):
                        pred_price = current_price * (1 + recent_trend * (i + 1))
                        # Keep predictions reasonable
                        pred_price = max(
                            current_price * 0.9, min(current_price * 1.1, pred_price)
                        )
                        predictions.append(pred_price)

                    confidence_scores = [0.6 - (i * 0.05) for i in range(days)]
                    patterns = []

                except Exception as fallback_error:
                    if self.verbose:
                        self.console.print_error(
                            f"All prediction methods failed: {fallback_error}"
                        )
                    # Ultimate fallback
                    current_price = enhanced_data["Close"].iloc[-1]
                    predictions = [
                        current_price * (1 + 0.01 * i) for i in range(1, days + 1)
                    ]
                    confidence_scores = [0.5] * days
                    patterns = []

            if not predictions or len(predictions) != days:
                raise ValueError(
                    f"Invalid prediction result: expected {days} predictions, got {len(predictions) if predictions else 0}"
                )

            # Perform fast company analysis if requested
            company_analysis = {}
            if include_analysis:
                try:
                    # Get company info for analysis
                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    # Use fast analyzer (instant results)
                    company_analysis = self.fast_analyzer.analyze_fast(
                        symbol, info, data
                    )

                    if self.verbose:
                        self.console.print_success(
                            f"Fast analysis completed ({company_analysis.get('processing_time', 'instant')})"
                        )

                except Exception as analysis_error:
                    if self.verbose:
                        self.console.print_warning(
                            f"Fast analysis failed: {analysis_error}"
                        )
                    company_analysis = {
                        "overall_score": 50,
                        "recommendation": "HOLD",
                        "error": str(analysis_error),
                    }

            # Format enhanced results
            result = self._format_enhanced_result(
                symbol, predictions, data, patterns, confidence_scores, company_analysis
            )
            if not result:
                raise ValueError("Failed to format prediction results")

            # Save predictions to cache (non-critical)
            try:
                self.cache_manager.save_predictions(symbol, result)
            except Exception as cache_error:
                if self.verbose:
                    self.console.print_warning(
                        f"Failed to save to cache: {cache_error}"
                    )

            # Update online learning (non-critical)
            try:
                self._update_online_learning(symbol, predictions, data)
            except Exception as learning_error:
                if self.verbose:
                    self.console.print_warning(
                        f"Online learning update failed: {learning_error}"
                    )

            return result

        except ValueError as e:
            self.console.print_error(str(e))
            return None
        except Exception as e:
            self.console.print_error(
                f"Unexpected error generating predictions for {symbol}: {e}"
            )
            if self.verbose:
                import traceback

                self.console.print_error(f"Detailed error: {traceback.format_exc()}")
            return None

    def _prepare_features(self, data):
        """Prepare features for ML models with enhanced error handling"""
        try:
            # Select relevant features for prediction
            feature_columns = [
                "Close",
                "Volume",
                "High",
                "Low",
                "Open",
                "SMA_20",
                "SMA_50",
                "EMA_12",
                "EMA_26",
                "RSI",
                "MACD",
                "MACD_Signal",
                "BB_Upper",
                "BB_Lower",
                "Stoch_K",
                "Stoch_D",
                "Williams_R",
                "CCI",
                "ATR",
                "OBV",
                "Price_Change",
                "Volume_Change",
            ]

            # Filter available columns
            available_columns = [col for col in feature_columns if col in data.columns]

            # Ensure we have at least basic OHLCV data
            required_columns = ["Close", "Volume", "High", "Low", "Open"]
            missing_required = [
                col for col in required_columns if col not in available_columns
            ]

            if missing_required:
                raise ValueError(f"Missing required columns: {missing_required}")

            if len(available_columns) < 5:
                if self.verbose:
                    self.console.print_warning(
                        f"Limited features available: {len(available_columns)}"
                    )

            # Prepare features with robust handling
            features_df = data[available_columns].copy()

            # Handle missing values more robustly
            features_df = (
                features_df.fillna(method="ffill").fillna(method="bfill").fillna(0)
            )

            # Check for infinite values
            features_df = features_df.replace([float("inf"), float("-inf")], 0)

            # Validate feature matrix
            features_array = features_df.values

            if features_array.shape[0] < 10:
                raise ValueError(
                    f"Insufficient data points: {features_array.shape[0]} (need at least 10)"
                )

            if features_array.shape[1] < 5:
                raise ValueError(
                    f"Insufficient features: {features_array.shape[1]} (need at least 5)"
                )

            # Check for all-zero columns
            zero_columns = (features_array == 0).all(axis=0)
            if zero_columns.any():
                if self.verbose:
                    zero_col_names = [
                        available_columns[i]
                        for i, is_zero in enumerate(zero_columns)
                        if is_zero
                    ]
                    self.console.print_warning(
                        f"Zero-value columns detected: {zero_col_names}"
                    )

            return features_array

        except ValueError as e:
            self.console.print_error(f"Feature preparation error: {e}")
            return None
        except Exception as e:
            self.console.print_error(f"Unexpected error preparing features: {e}")
            return None

    def _fallback_prediction(self, data, days):
        """Fallback prediction method when ML models fail"""
        try:
            if self.verbose:
                self.console.print_warning("Using fallback prediction method")

            # Simple trend-based prediction
            prices = data["Close"].values

            if len(prices) < 5:
                # Not enough data for trend analysis
                current_price = prices[-1]
                return [current_price] * days

            # Calculate recent trend (last 10 days or available data)
            trend_period = min(10, len(prices))
            recent_prices = prices[-trend_period:]

            # Linear trend calculation
            x = range(trend_period)
            coeffs = np.polyfit(x, recent_prices, 1)
            trend_slope = coeffs[0]

            # Generate predictions
            current_price = prices[-1]
            predictions = []

            for i in range(days):
                # Apply trend with some dampening to avoid extreme predictions
                dampening_factor = 0.8**i  # Reduce trend impact over time
                predicted_price = current_price + (
                    trend_slope * (i + 1) * dampening_factor
                )

                # Ensure prediction is reasonable (within 20% of current price)
                max_change = current_price * 0.2
                predicted_price = max(
                    current_price - max_change,
                    min(current_price + max_change, predicted_price),
                )

                predictions.append(predicted_price)

            return predictions

        except Exception as e:
            if self.verbose:
                self.console.print_error(f"Fallback prediction failed: {e}")

            # Ultimate fallback - return current price
            try:
                current_price = data["Close"].iloc[-1]
                return [current_price] * days
            except:
                return [100.0] * days  # Last resort

    def _format_enhanced_result(
        self, symbol, predictions, data, patterns, confidence_scores, company_analysis
    ):
        """Format enhanced prediction results with patterns and analysis"""
        try:
            current_price = data["Close"].iloc[-1]
            result = {
                "symbol": symbol,
                "current_price": current_price,
                "predictions": [],
                "patterns": patterns,
                "company_analysis": company_analysis,
                "timestamp": datetime.now().isoformat(),
                "model_info": (
                    self.ml_system.get_model_info()
                    if hasattr(self.ml_system, "get_model_info")
                    else {}
                ),
                "learning_insights": (
                    self.ml_system.get_learning_insights()
                    if hasattr(self.ml_system, "get_learning_insights")
                    else {}
                ),
            }

            for i, pred_price in enumerate(predictions):
                pred_date = datetime.now() + timedelta(days=i + 1)
                change = pred_price - current_price
                change_pct = (change / current_price) * 100
                confidence = (
                    confidence_scores[i] if i < len(confidence_scores) else 0.75
                )

                result["predictions"].append(
                    {
                        "day": i + 1,
                        "date": pred_date.isoformat(),
                        "predicted_price": float(pred_price),
                        "change": float(change),
                        "change_pct": float(change_pct),
                        "confidence": float(confidence),
                    }
                )

            # Add pattern summary
            if patterns:
                result["pattern_summary"] = {
                    "primary_pattern": patterns[0]["type"] if patterns else None,
                    "signal_direction": (
                        patterns[0]["breakout_direction"] if patterns else "neutral"
                    ),
                    "pattern_confidence": patterns[0]["confidence"] if patterns else 0,
                    "total_patterns_detected": len(patterns),
                }
            else:
                result["pattern_summary"] = {
                    "primary_pattern": None,
                    "signal_direction": "neutral",
                    "pattern_confidence": 0,
                    "total_patterns_detected": 0,
                }

            # Add analysis summary
            if company_analysis and "error" not in company_analysis:
                result["analysis_summary"] = {
                    "overall_score": company_analysis.get("overall_score", 50),
                    "recommendation": company_analysis.get("recommendation", "HOLD"),
                    "financial_grade": company_analysis.get("financial_health", {}).get(
                        "health_grade", "C"
                    ),
                    "risk_grade": company_analysis.get("risk_assessment", {}).get(
                        "risk_grade", "C"
                    ),
                    "valuation_summary": company_analysis.get(
                        "valuation_metrics", {}
                    ).get("summary", "Fair Value"),
                    "market_sentiment": company_analysis.get(
                        "market_intelligence", {}
                    ).get("market_sentiment", "Neutral"),
                }

            return result

        except Exception as e:
            self.console.print_error(f"Error formatting enhanced results: {e}")
            return None

    def _format_prediction_result(self, symbol, predictions, data):
        """Format basic prediction results (backward compatibility)"""
        try:
            current_price = data["Close"].iloc[-1]
            result = {
                "symbol": symbol,
                "current_price": current_price,
                "predictions": [],
                "timestamp": datetime.now().isoformat(),
                "model_info": (
                    self.ml_system.get_model_info()
                    if hasattr(self.ml_system, "get_model_info")
                    else {}
                ),
            }

            for i, pred_price in enumerate(predictions):
                pred_date = datetime.now() + timedelta(days=i + 1)
                change = pred_price - current_price
                change_pct = (change / current_price) * 100

                result["predictions"].append(
                    {
                        "day": i + 1,
                        "date": pred_date.isoformat(),
                        "predicted_price": float(pred_price),
                        "change": float(change),
                        "change_pct": float(change_pct),
                    }
                )

            return result

        except Exception as e:
            self.console.print_error(f"Error formatting results: {e}")
            return None

    def _format_cached_result(self, cached_data):
        """Format cached prediction results"""
        try:
            # Convert cached data to standard format
            return {
                "symbol": cached_data["symbol"],
                "current_price": cached_data.get("current_price", 0),
                "predictions": cached_data.get("predictions", []),
                "timestamp": cached_data.get("timestamp"),
                "cached": True,
                "cache_age": self._calculate_cache_age(cached_data.get("timestamp")),
            }
        except Exception as e:
            self.console.print_error(f"Error formatting cached result: {e}")
            return None

    def _calculate_cache_age(self, timestamp_str):
        """Calculate age of cached data"""
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

    def _update_online_learning(self, symbol, predictions, data):
        """Update online learning system"""
        try:
            current_price = data["Close"].iloc[-1]
            validation_summary = self.accuracy_tracker.validate_predictions()

            learning_data = {
                "symbol": symbol,
                "prediction": predictions[0] if predictions else current_price,
                "actual_price": current_price,
                "timestamp": datetime.now().isoformat(),
                "validation_summary": validation_summary,
            }

            # Update ML system with learning data
            self.ml_system.update_online_learning(learning_data)

        except Exception as e:
            if self.verbose:
                self.console.print_error(f"Online learning update failed: {e}")

    def analyze_accuracy(self, symbol=None):
        """Analyze prediction accuracy"""
        return self.accuracy_tracker.analyze_accuracy(symbol)

    def validate_predictions(self):
        """Validate and cleanup old predictions"""
        return self.accuracy_tracker.validate_predictions()

    def get_system_info(self):
        """Get system information"""
        return {
            "gpu_info": self.gpu_manager.detect_gpu_vendor(),
            "device": str(self.gpu_manager.get_device()),
            "model_info": self.ml_system.get_model_info(),
            "cache_stats": self.cache_manager.get_cache_stats(),
            "accuracy_stats": self.accuracy_tracker.get_accuracy_stats(),
        }


class StockPredictor:
    """Simplified interface for stock prediction (backward compatibility)"""

    def __init__(self, verbose=False):
        self.ara = AraAI(verbose=verbose)

    def predict(self, symbol, days=5):
        """Predict stock prices"""
        return self.ara.predict(symbol, days=days)

    def analyze(self, symbol):
        """Analyze stock with technical indicators"""
        return self.ara.data_manager.get_stock_analysis(symbol)


# Convenience functions for backward compatibility
def predict_stock(symbol, days=5, verbose=False):
    """
    Predict stock prices using Ara AI ensemble system

    Args:
        symbol (str): Stock symbol
        days (int): Number of days to predict
        verbose (bool): Enable verbose output

    Returns:
        dict: Prediction results
    """
    ara = AraAI(verbose=verbose)
    return ara.predict(symbol, days=days)


def analyze_stock(symbol, verbose=False):
    """
    Analyze stock with technical indicators

    Args:
        symbol (str): Stock symbol
        verbose (bool): Enable verbose output

    Returns:
        dict: Analysis results
    """
    ara = AraAI(verbose=verbose)
    return ara.data_manager.get_stock_analysis(symbol)

    def predict_with_ai(self, symbol, days=5, use_cache=True):
        """
        Enhanced prediction with full AI analysis (slower but more comprehensive)

        Args:
            symbol (str): Stock symbol
            days (int): Number of days to predict
            use_cache (bool): Whether to use cached predictions

        Returns:
            dict: Enhanced prediction results with AI analysis
        """
        try:
            if self.verbose:
                self.console.print_info(
                    f"Running comprehensive AI analysis for {symbol}..."
                )

            # Use the full AI system for comprehensive analysis
            return self.predict(
                symbol, days=days, use_cache=use_cache, include_analysis=True
            )

        except Exception as e:
            self.console.print_error(f"AI prediction failed for {symbol}: {e}")
            return None

    def analyze_with_ai(self, symbol):
        """
        Comprehensive AI-powered company analysis (slower but detailed)

        Args:
            symbol (str): Stock symbol

        Returns:
            dict: Comprehensive AI analysis results
        """
        try:
            if self.verbose:
                self.console.print_info(
                    f"Running comprehensive AI analysis for {symbol}..."
                )

            # Use AI analyzer for detailed analysis
            return self.ai_analyzer.analyze_company_with_ai(symbol)

        except Exception as e:
            self.console.print_error(f"AI analysis failed for {symbol}: {e}")
            return {"error": str(e)}
