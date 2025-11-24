"""
Market Regime Detection Demo
Demonstrates the regime detection system and adaptive predictions
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara.models.regime_detector import RegimeDetector, RegimeType
from ara.models.regime_adjustments import RegimeAdaptivePredictions


def fetch_sample_data(symbol: str = "AAPL", period: str = "2y") -> pd.DataFrame:
    """Fetch sample data for demonstration"""
    print(f"Fetching {period} of data for {symbol}...")
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period)
    
    # Rename columns to lowercase
    data.columns = [col.lower() for col in data.columns]
    
    print(f"Fetched {len(data)} days of data")
    return data


def demonstrate_regime_detection():
    """Demonstrate basic regime detection"""
    print("\n" + "="*80)
    print("REGIME DETECTION DEMONSTRATION")
    print("="*80)
    
    # Fetch data
    data = fetch_sample_data("AAPL", "2y")
    
    # Initialize and fit regime detector
    print("\n1. Initializing and fitting regime detector...")
    detector = RegimeDetector(lookback_period=252)
    detector.fit(data)
    print("✓ Regime detector fitted successfully")
    
    # Detect current regime
    print("\n2. Detecting current market regime...")
    regime_info = detector.detect(data.tail(100))
    
    print(f"\nCurrent Regime: {regime_info['current_regime'].upper()}")
    print(f"Confidence: {regime_info['confidence']:.2%}")
    print(f"Stability Score: {regime_info['stability_score']:.2%}")
    print(f"Duration in Regime: {regime_info['duration_in_regime']} days")
    print(f"Expected Duration: {regime_info['expected_duration']} days")
    
    print("\nRegime Probabilities:")
    for regime, prob in regime_info['regime_probabilities'].items():
        print(f"  {regime:20s}: {prob:6.2%}")
    
    print("\nTransition Probabilities:")
    for regime, prob in regime_info['transition_probabilities'].items():
        print(f"  → {regime:20s}: {prob:6.2%}")
    
    # Get regime statistics
    print("\n3. Regime Statistics:")
    stats = detector.get_regime_statistics()
    if stats:
        print(f"Total Observations: {stats['total_observations']}")
        print("\nRegime Distribution:")
        for regime, pct in stats['regime_distribution'].items():
            print(f"  {regime:20s}: {pct:6.2%}")
    
    return detector, data


def demonstrate_regime_adjustments():
    """Demonstrate regime-adaptive prediction adjustments"""
    print("\n" + "="*80)
    print("REGIME-ADAPTIVE PREDICTIONS DEMONSTRATION")
    print("="*80)
    
    # Fetch data
    data = fetch_sample_data("AAPL", "2y")
    
    # Initialize regime detector and fit
    print("\n1. Setting up regime-adaptive predictions...")
    detector = RegimeDetector(lookback_period=252)
    detector.fit(data)
    
    adaptive_predictions = RegimeAdaptivePredictions(regime_detector=detector)
    print("✓ Regime-adaptive predictions initialized")
    
    # Simulate predictions for different regimes
    print("\n2. Demonstrating adjustments for different regimes...")
    
    # Generate sample predictions
    requested_days = 30
    base_predictions = np.linspace(150, 160, requested_days)
    base_confidence = 0.75
    base_std = 2.0
    
    # Test each regime
    for regime in RegimeType:
        print(f"\n--- {regime.value.upper()} REGIME ---")
        
        # Adjust horizon
        adjusted_days = adaptive_predictions.adjust_prediction_horizon(requested_days, regime)
        print(f"Prediction Horizon: {requested_days} → {adjusted_days} days")
        
        # Adjust confidence intervals
        test_predictions = base_predictions[:adjusted_days]
        lower, upper = adaptive_predictions.adjust_confidence_intervals(
            test_predictions, base_std, regime
        )
        
        avg_interval = np.mean(upper - lower)
        print(f"Average Confidence Interval Width: {avg_interval:.2f}")
        
        # Get feature importance
        feature_importance = adaptive_predictions.get_regime_feature_importance(regime)
        print("Feature Importance Weights:")
        for feature, weight in feature_importance.items():
            print(f"  {feature:15s}: {weight:.2f}")
        
        # Get preferred models
        preferred_models = adaptive_predictions.get_preferred_models(regime)
        print(f"Preferred Models: {', '.join(preferred_models[:3])}")
    
    return adaptive_predictions, data


def demonstrate_full_workflow():
    """Demonstrate complete workflow with regime detection and adjustments"""
    print("\n" + "="*80)
    print("COMPLETE REGIME-ADAPTIVE WORKFLOW")
    print("="*80)
    
    # Fetch data
    data = fetch_sample_data("AAPL", "2y")
    
    # Initialize system
    print("\n1. Initializing regime-adaptive system...")
    detector = RegimeDetector(lookback_period=252)
    detector.fit(data)
    
    adaptive_predictions = RegimeAdaptivePredictions(regime_detector=detector)
    print("✓ System initialized")
    
    # Generate sample predictions
    print("\n2. Generating sample predictions...")
    requested_days = 30
    base_predictions = np.random.randn(requested_days).cumsum() + 150
    base_confidence = 0.75
    base_std = 2.0
    
    # Apply regime adjustments
    print("\n3. Applying regime-based adjustments...")
    result = adaptive_predictions.apply_regime_adjustments(
        predictions=base_predictions,
        requested_days=requested_days,
        base_confidence=base_confidence,
        base_std=base_std,
        data=data.tail(100)
    )
    
    # Display results
    print("\n4. Adjustment Results:")
    print(f"\nRegime: {result['regime_info']['current_regime'].upper()}")
    print(f"Regime Confidence: {result['regime_info']['confidence']:.2%}")
    
    print(f"\nAdjustment Summary:")
    for key, value in result['adjustment_summary'].items():
        print(f"  {key:20s}: {value}")
    
    print(f"\nPrediction Details:")
    print(f"  Original Days: {result['original_days']}")
    print(f"  Adjusted Days: {result['adjusted_days']}")
    print(f"  Base Confidence: {result['base_confidence']:.2%}")
    print(f"  Adjusted Confidence: {result['adjusted_confidence']:.2%}")
    
    print(f"\nPreferred Models:")
    for model in result['preferred_models'][:5]:
        print(f"  • {model}")
    
    # Display alerts if any
    if result['alerts']:
        print(f"\n⚠ ALERTS ({len(result['alerts'])}):")
        for alert in result['alerts']:
            print(f"\n  Type: {alert['type']}")
            print(f"  Severity: {alert['severity']}")
            print(f"  Message: {alert['message']}")
            if 'recommendations' in alert:
                print(f"  Recommendations:")
                for rec in alert['recommendations'][:3]:
                    print(f"    - {rec}")
    else:
        print("\n✓ No alerts")
    
    # Show sample predictions with confidence intervals
    print("\n5. Sample Predictions (first 5 days):")
    print(f"{'Day':>5} {'Prediction':>12} {'Lower Bound':>12} {'Upper Bound':>12} {'Interval':>10}")
    print("-" * 60)
    
    for i in range(min(5, result['adjusted_days'])):
        pred = result['adjusted_predictions'][i]
        lower = result['lower_bounds'][i]
        upper = result['upper_bounds'][i]
        interval = upper - lower
        print(f"{i+1:5d} {pred:12.2f} {lower:12.2f} {upper:12.2f} {interval:10.2f}")
    
    return result


def demonstrate_regime_change_detection():
    """Demonstrate regime change detection and alerts"""
    print("\n" + "="*80)
    print("REGIME CHANGE DETECTION")
    print("="*80)
    
    # Fetch data for a volatile period
    print("\n1. Fetching data with potential regime changes...")
    data = fetch_sample_data("AAPL", "1y")
    
    # Initialize system
    detector = RegimeDetector(lookback_period=252)
    detector.fit(data)
    
    adaptive_predictions = RegimeAdaptivePredictions(regime_detector=detector)
    
    # Simulate regime detection over time
    print("\n2. Detecting regimes over time...")
    window_size = 100
    step_size = 20
    
    regime_timeline = []
    
    for i in range(0, len(data) - window_size, step_size):
        window_data = data.iloc[i:i+window_size]
        regime_info = detector.detect(window_data)
        
        regime_timeline.append({
            'date': window_data.index[-1],
            'regime': regime_info['current_regime'],
            'confidence': regime_info['confidence']
        })
    
    # Display regime timeline
    print("\nRegime Timeline:")
    print(f"{'Date':12s} {'Regime':20s} {'Confidence':>12s}")
    print("-" * 50)
    
    prev_regime = None
    for entry in regime_timeline[-10:]:  # Show last 10
        regime = entry['regime']
        marker = " ← CHANGE" if prev_regime and regime != prev_regime else ""
        print(f"{str(entry['date'].date()):12s} {regime:20s} {entry['confidence']:11.2%}{marker}")
        prev_regime = regime
    
    # Check for regime changes
    print("\n3. Checking for regime change alerts...")
    
    # Simulate checking with different regimes
    test_regimes = [RegimeType.BULL, RegimeType.HIGH_VOLATILITY, RegimeType.BEAR]
    
    for i, regime in enumerate(test_regimes[:-1]):
        next_regime = test_regimes[i + 1]
        
        # Create mock regime info
        regime_info = {
            'current_regime': next_regime.value,
            'confidence': 0.85,
            'stability_score': 0.7
        }
        
        alert = adaptive_predictions.check_regime_change(regime, regime_info)
        
        if alert:
            print(f"\n⚠ REGIME CHANGE ALERT:")
            print(f"  {alert['previous_regime']} → {alert['new_regime']}")
            print(f"  Confidence: {alert['confidence']:.2%}")
            print(f"  Recommendations:")
            for rec in alert['recommendations'][:3]:
                print(f"    - {rec}")
    
    # Display all alerts
    recent_alerts = adaptive_predictions.get_recent_alerts()
    print(f"\n4. Total Alerts Generated: {len(recent_alerts)}")


def demonstrate_save_load():
    """Demonstrate saving and loading regime detector state"""
    print("\n" + "="*80)
    print("SAVE/LOAD DEMONSTRATION")
    print("="*80)
    
    # Create and fit detector
    print("\n1. Creating and fitting regime detector...")
    data = fetch_sample_data("AAPL", "1y")
    
    detector = RegimeDetector(lookback_period=252)
    detector.fit(data)
    
    regime_info = detector.detect(data.tail(100))
    print(f"Original Regime: {regime_info['current_regime']}")
    print(f"Original Confidence: {regime_info['confidence']:.2%}")
    
    # Save state
    print("\n2. Saving detector state...")
    save_path = Path(".ara_cache/regime_detector_demo.json")
    detector.save(save_path)
    print(f"✓ Saved to {save_path}")
    
    # Load state
    print("\n3. Loading detector state...")
    new_detector = RegimeDetector()
    new_detector.load(save_path)
    print("✓ State loaded successfully")
    
    # Verify
    new_regime_info = new_detector.detect(data.tail(100))
    print(f"\nLoaded Regime: {new_regime_info['current_regime']}")
    print(f"Loaded Confidence: {new_regime_info['confidence']:.2%}")
    
    # Verify they match
    if (regime_info['current_regime'] == new_regime_info['current_regime'] and
        abs(regime_info['confidence'] - new_regime_info['confidence']) < 0.01):
        print("\n✓ Save/Load verification successful!")
    else:
        print("\n✗ Save/Load verification failed!")


def main():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print("MARKET REGIME DETECTION SYSTEM - COMPREHENSIVE DEMO")
    print("="*80)
    
    try:
        # Run demonstrations
        demonstrate_regime_detection()
        demonstrate_regime_adjustments()
        demonstrate_full_workflow()
        demonstrate_regime_change_detection()
        demonstrate_save_load()
        
        print("\n" + "="*80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
