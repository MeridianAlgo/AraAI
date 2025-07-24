"""
Historical Volatility Analysis for Realistic Predictions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging
from models import StockData
from data_manager import stock_data_manager

logger = logging.getLogger(__name__)

class VolatilityAnalyzer:
    """Analyzes historical volatility patterns to constrain predictions"""
    
    def __init__(self):
        self.volatility_cache = {}
    
    def analyze_historical_volatility(self, symbol: str, days: int = 252) -> Dict:
        """Analyze historical daily price changes for a stock"""
        
        # Check cache first
        cache_key = f"{symbol}_{days}"
        if cache_key in self.volatility_cache:
            return self.volatility_cache[cache_key]
        
        # Get historical data
        historical_data = stock_data_manager.get_historical_data(symbol, days)
        
        if len(historical_data) < 10:
            logger.warning(f"Insufficient historical data for {symbol}")
            return self._default_volatility_stats()
        
        # Calculate daily percentage changes
        daily_changes = []
        for i in range(1, len(historical_data)):
            prev_price = historical_data[i-1].close_price
            curr_price = historical_data[i].close_price
            pct_change = (curr_price - prev_price) / prev_price
            daily_changes.append(pct_change)
        
        daily_changes = np.array(daily_changes)
        
        # Calculate comprehensive volatility statistics
        volatility_stats = {
            'mean_change': np.mean(daily_changes),
            'std_change': np.std(daily_changes),
            'min_change': np.min(daily_changes),
            'max_change': np.max(daily_changes),
            'percentiles': {
                'p1': np.percentile(daily_changes, 1),
                'p5': np.percentile(daily_changes, 5),
                'p10': np.percentile(daily_changes, 10),
                'p25': np.percentile(daily_changes, 25),
                'p50': np.percentile(daily_changes, 50),
                'p75': np.percentile(daily_changes, 75),
                'p90': np.percentile(daily_changes, 90),
                'p95': np.percentile(daily_changes, 95),
                'p99': np.percentile(daily_changes, 99)
            },
            'extreme_moves': {
                'up_5pct_days': np.sum(daily_changes > 0.05),
                'down_5pct_days': np.sum(daily_changes < -0.05),
                'up_10pct_days': np.sum(daily_changes > 0.10),
                'down_10pct_days': np.sum(daily_changes < -0.10),
                'total_days': len(daily_changes)
            },
            'volatility_regime': self._classify_volatility_regime(daily_changes),
            'typical_range': {
                'normal_up': np.percentile(daily_changes[daily_changes > 0], 75) if np.any(daily_changes > 0) else 0.02,
                'normal_down': np.percentile(daily_changes[daily_changes < 0], 25) if np.any(daily_changes < 0) else -0.02,
                'extreme_up': np.percentile(daily_changes, 95),
                'extreme_down': np.percentile(daily_changes, 5)
            }
        }
        
        # Cache the results
        self.volatility_cache[cache_key] = volatility_stats
        
        logger.info(f"Volatility analysis for {symbol}: "
                   f"Mean: {volatility_stats['mean_change']:.3f}, "
                   f"Std: {volatility_stats['std_change']:.3f}, "
                   f"95th percentile: {volatility_stats['percentiles']['p95']:.3f}")
        
        return volatility_stats
    
    def _classify_volatility_regime(self, daily_changes: np.ndarray) -> str:
        """Classify the current volatility regime"""
        std_change = np.std(daily_changes)
        
        if std_change < 0.015:  # Less than 1.5% daily volatility
            return 'LOW'
        elif std_change < 0.025:  # Less than 2.5% daily volatility
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _default_volatility_stats(self) -> Dict:
        """Default volatility stats for when insufficient data is available"""
        return {
            'mean_change': 0.001,
            'std_change': 0.02,
            'min_change': -0.10,
            'max_change': 0.10,
            'percentiles': {
                'p1': -0.06, 'p5': -0.04, 'p10': -0.03, 'p25': -0.015,
                'p50': 0.001, 'p75': 0.015, 'p90': 0.03, 'p95': 0.04, 'p99': 0.06
            },
            'extreme_moves': {
                'up_5pct_days': 5, 'down_5pct_days': 5,
                'up_10pct_days': 1, 'down_10pct_days': 1,
                'total_days': 100
            },
            'volatility_regime': 'MEDIUM',
            'typical_range': {
                'normal_up': 0.02, 'normal_down': -0.02,
                'extreme_up': 0.04, 'extreme_down': -0.04
            }
        }
    
    def constrain_prediction(self, raw_prediction: float, symbol: str, confidence: float) -> Tuple[float, str]:
        """Constrain prediction to realistic ranges based on historical volatility"""
        
        # Get volatility stats
        vol_stats = self.analyze_historical_volatility(symbol)
        
        # Determine constraint level based on confidence
        if confidence >= 0.8:
            # High confidence: allow up to 90th percentile moves
            max_up = vol_stats['percentiles']['p90']
            max_down = vol_stats['percentiles']['p10']
            constraint_level = "90th percentile"
        elif confidence >= 0.6:
            # Medium confidence: allow up to 75th percentile moves
            max_up = vol_stats['percentiles']['p75']
            max_down = vol_stats['percentiles']['p25']
            constraint_level = "75th percentile"
        else:
            # Low confidence: constrain to typical daily range
            max_up = vol_stats['typical_range']['normal_up']
            max_down = vol_stats['typical_range']['normal_down']
            constraint_level = "typical range"
        
        # Apply constraints
        constrained_prediction = np.clip(raw_prediction, max_down, max_up)
        
        # Log if prediction was constrained
        if abs(constrained_prediction - raw_prediction) > 0.001:
            logger.info(f"Prediction constrained for {symbol}: "
                       f"{raw_prediction:.3f} -> {constrained_prediction:.3f} "
                       f"(using {constraint_level})")
        
        return constrained_prediction, constraint_level
    
    def get_prediction_context(self, symbol: str, predicted_change: float) -> Dict:
        """Get context about how unusual this prediction is historically"""
        
        vol_stats = self.analyze_historical_volatility(symbol)
        
        # Calculate percentile of this prediction
        percentile = self._calculate_percentile(predicted_change, vol_stats)
        
        # Determine how unusual this move would be
        if abs(predicted_change) > abs(vol_stats['percentiles']['p95']):
            rarity = "EXTREMELY RARE (>95th percentile)"
            frequency = f"Occurs ~{252 * 0.05:.0f} times per year"
        elif abs(predicted_change) > abs(vol_stats['percentiles']['p90']):
            rarity = "RARE (>90th percentile)"
            frequency = f"Occurs ~{252 * 0.10:.0f} times per year"
        elif abs(predicted_change) > abs(vol_stats['percentiles']['p75']):
            rarity = "UNCOMMON (>75th percentile)"
            frequency = f"Occurs ~{252 * 0.25:.0f} times per year"
        else:
            rarity = "TYPICAL"
            frequency = f"Occurs ~{252 * 0.50:.0f} times per year"
        
        return {
            'percentile': percentile,
            'rarity': rarity,
            'frequency': frequency,
            'historical_context': {
                'similar_moves_up': vol_stats['extreme_moves']['up_5pct_days'] if predicted_change > 0.05 else "N/A",
                'similar_moves_down': vol_stats['extreme_moves']['down_5pct_days'] if predicted_change < -0.05 else "N/A",
                'total_days_analyzed': vol_stats['extreme_moves']['total_days']
            },
            'volatility_regime': vol_stats['volatility_regime']
        }
    
    def _calculate_percentile(self, value: float, vol_stats: Dict) -> float:
        """Calculate what percentile this value represents"""
        percentiles = vol_stats['percentiles']
        
        if value >= percentiles['p99']:
            return 99
        elif value >= percentiles['p95']:
            return 95
        elif value >= percentiles['p90']:
            return 90
        elif value >= percentiles['p75']:
            return 75
        elif value >= percentiles['p50']:
            return 50
        elif value >= percentiles['p25']:
            return 25
        elif value >= percentiles['p10']:
            return 10
        elif value >= percentiles['p5']:
            return 5
        else:
            return 1
    
    def generate_volatility_report(self, symbol: str) -> str:
        """Generate a human-readable volatility report"""
        
        vol_stats = self.analyze_historical_volatility(symbol)
        
        report = f"""
📊 HISTORICAL VOLATILITY ANALYSIS - {symbol}
{'='*50}

📈 DAILY PRICE CHANGES:
• Average daily change: {vol_stats['mean_change']*100:+.2f}%
• Standard deviation: {vol_stats['std_change']*100:.2f}%
• Largest gain: {vol_stats['max_change']*100:+.2f}%
• Largest loss: {vol_stats['min_change']*100:+.2f}%

📊 TYPICAL RANGES:
• 50% of days: {vol_stats['percentiles']['p25']*100:+.2f}% to {vol_stats['percentiles']['p75']*100:+.2f}%
• 80% of days: {vol_stats['percentiles']['p10']*100:+.2f}% to {vol_stats['percentiles']['p90']*100:+.2f}%
• 90% of days: {vol_stats['percentiles']['p5']*100:+.2f}% to {vol_stats['percentiles']['p95']*100:+.2f}%

🚨 EXTREME MOVES:
• Days with >5% gains: {vol_stats['extreme_moves']['up_5pct_days']} ({vol_stats['extreme_moves']['up_5pct_days']/vol_stats['extreme_moves']['total_days']*100:.1f}%)
• Days with >5% losses: {vol_stats['extreme_moves']['down_5pct_days']} ({vol_stats['extreme_moves']['down_5pct_days']/vol_stats['extreme_moves']['total_days']*100:.1f}%)
• Days with >10% moves: {vol_stats['extreme_moves']['up_10pct_days'] + vol_stats['extreme_moves']['down_10pct_days']} total

⚡ VOLATILITY REGIME: {vol_stats['volatility_regime']}
📅 Analysis period: {vol_stats['extreme_moves']['total_days']} trading days
"""
        
        return report

# Global instance
volatility_analyzer = VolatilityAnalyzer()

if __name__ == "__main__":
    # Test the volatility analyzer
    analyzer = VolatilityAnalyzer()
    report = analyzer.generate_volatility_report("AAPL")
    print(report)