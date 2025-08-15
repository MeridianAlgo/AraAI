#!/usr/bin/env python3
"""
Basic Ara AI Usage Example
"""

from meridianalgo import quick_predict, analyze_accuracy

def main():
    """Basic usage examples"""
    
    print("🚀 Ara AI Basic Usage Examples")
    print("=" * 40)
    
    # Quick prediction
    print("\n1. Quick Prediction:")
    result = quick_predict('AAPL', days=5)
    if result:
        print(f"AAPL predictions: {len(result['predictions'])} days")
        for pred in result['predictions']:
            print(f"  Day {pred['day']}: ${pred['predicted_price']:.2f}")
    
    # Accuracy analysis
    print("\n2. Accuracy Analysis:")
    accuracy = analyze_accuracy('AAPL')
    if accuracy:
        print(f"AAPL accuracy: {accuracy['accuracy_rate']:.1f}%")

if __name__ == "__main__":
    main()
