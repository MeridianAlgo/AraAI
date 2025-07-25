#!/usr/bin/env python3
"""
Smart Trader CLI Entry Point
Ultra-Accurate AI Stock Analysis with Universal GPU Support
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def main():
    """Main CLI entry point for Smart Trader"""
    try:
        # Import and run the main smart trader application
        from smart_trader_fixed import main as smart_trader_main
        smart_trader_main()
    except ImportError:
        # Fallback to the original smart_trader.py
        try:
            from smart_trader import main as smart_trader_main
            smart_trader_main()
        except ImportError:
            print("Error: Smart Trader modules not found. Please ensure the package is properly installed.")
            sys.exit(1)
    except Exception as e:
        print(f"Error running Smart Trader: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()