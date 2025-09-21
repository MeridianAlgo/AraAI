"""
Fast Core System - Ultra-fast predictions without AI model loading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from .fast_ml import FastMLPredictor, FastPatternRecognizer, FastAnalyzer
from .utils import CacheManager
from .console import ConsoleManager

class FastAraAI:
    """
    Ultra-fast AraAI without AI model loading
    Perfect for quick predictions and analysis
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.console = ConsoleManager(verbose=verbose)
        
        # Fast components only (no AI model loading)
        self.cache_manager = CacheManager()
        self.fast_ml = FastMLPredictor()
        self.fast_patterns = FastPatternRecognizer()
        self.fast_analyzer = FastAnalyzer()
        
        if self.verbose:
            self.console.print_success("Fast AraAI initialized (no AI model loading)")
    
    def predict_fast(self, symbol, days=5, use_cache=True):
        """
        Ultra-fast stock prediction
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to predict
            use_cache (bool): Whether to use cached predictions
            
        Returns:
            dict: Fast prediction results
        """
        try:
            symbol = symbol.upper().strip()
            
            # Check cache first
            if use_cache:
                cached_result = self.cache_manager.check_cached_predictions(symbol, days)
                if cached_result:
                    return self._format_cached_result(cached_result)
            
            # Get market data quickly
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="3mo")  # Less data for speed
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            current_price = data['Close'].iloc[-1]
            
            # Fast prediction
            ml_result = self.fast_ml.predict_fast(data, current_price, days=days)
            predictions = ml_result['predictions']
            confidence_scores = ml_result['confidence_scores']
            
            # Fast pattern recognition
            patterns = self.fast_patterns.detect_patterns_fast(data['Close'])
            
            # Fast company analysis
            info = ticker.info
            company_analysis = self.fast_analyzer.analyze_fast(symbol, info, data)
            
            # Format results
            result = self._format_fast_result(symbol, predictions, current_price, patterns, confidence_scores, company_analysis)
            
            # Cache results
            if use_cache:
                self.cache_manager.save_predictions(symbol, result)
            
            return result
            
        except Exception as e:
            self.console.print_error(f"Fast prediction failed for {symbol}: {e}")
            return None
    
    def _format_fast_result(self, symbol, predictions, current_price, patterns, confidence_scores, company_analysis):
        """Format fast prediction results"""
        try:
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'predictions': [],
                'patterns': patterns,
                'timestamp': datetime.now().isoformat(),
                'processing_mode': 'fast'
            }
            
            # Format predictions
            for i, pred_price in enumerate(predictions):
                pred_date = datetime.now() + timedelta(days=i+1)
                change = pred_price - current_price
                change_pct = (change / current_price) * 100
                confidence = confidence_scores[i] if i < len(confidence_scores) else 0.75
                
                result['predictions'].append({
                    'day': i + 1,
                    'date': pred_date.isoformat(),
                    'predicted_price': float(pred_price),
                    'change': float(change),
                    'change_pct': float(change_pct),
                    'confidence': float(confidence)
                })
            
            # Add pattern summary
            if patterns:
                result['pattern_summary'] = {
                    'primary_pattern': patterns[0]['type'],
                    'signal_direction': patterns[0]['breakout_direction'],
                    'pattern_confidence': patterns[0]['confidence'],
                    'total_patterns_detected': len(patterns)
                }
            
            # Add analysis summary
            if company_analysis:
                result['analysis_summary'] = {
                    'overall_score': company_analysis.get('overall_score', 50),
                    'recommendation': company_analysis.get('recommendation', 'HOLD'),
                    'financial_grade': company_analysis.get('financial_grade', 'C'),
                    'risk_grade': company_analysis.get('risk_grade', 'Medium'),
                    'valuation_summary': company_analysis.get('valuation_summary', 'Fair Value'),
                    'market_sentiment': company_analysis.get('market_sentiment', 'Neutral')
                }
            
            return result
            
        except Exception as e:
            self.console.print_error(f"Error formatting fast results: {e}")
            return None
    
    def _format_cached_result(self, cached_data):
        """Format cached results"""
        try:
            return {
                'symbol': cached_data.get('symbol', 'Unknown'),
                'current_price': cached_data.get('current_price', 0),
                'predictions': cached_data.get('predictions', []),
                'timestamp': cached_data.get('timestamp'),
                'cached': True,
                'processing_mode': 'cached'
            }
        except Exception:
            return None