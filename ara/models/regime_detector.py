"""
Market regime detection for ARA AI
"""
import pandas as pd
import numpy as np
from typing import Dict, Any

class RegimeDetector:
    """Detects market regimes (Bullish, Bearish, Neutral, Volatile)"""
    
    def detect_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect current market regime based on historical data
        """
        if len(data) < 20:
            return {"regime": "unknown", "confidence": 0.5}
            
        # Very simple momentum and volatility based detection
        returns = data["close"].pct_change()
        volatility = returns.std()
        sma_20 = data["close"].rolling(20).mean().iloc[-1]
        sma_50 = data["close"].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
        
        current_price = data["close"].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            regime = "bullish"
            confidence = 0.8
        elif current_price < sma_20 < sma_50:
            regime = "bearish"
            confidence = 0.8
        elif volatility > returns.std() * 1.5:
            regime = "volatile"
            confidence = 0.7
        else:
            regime = "neutral"
            confidence = 0.6
            
        return {
            "regime": regime,
            "confidence": confidence,
            "volatility": float(volatility),
            "transition_probabilities": {
                "bullish": 0.7 if regime == "bullish" else 0.1,
                "bearish": 0.7 if regime == "bearish" else 0.1,
                "neutral": 0.5 if regime == "neutral" else 0.2
            },
            "duration": 5 # Placeholder
        }
