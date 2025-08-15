#!/usr/bin/env python3
import subprocess
import sys
import os

def main():
    try:
        # Change to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Try to run the main launcher
        if os.path.exists("run_ara.py"):
            subprocess.run([sys.executable, "run_ara.py"])
        elif os.path.exists("ara.py"):
            print("Ara AI Stock Analysis Platform")
            print("=" * 40)
            symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
            if symbol:
                subprocess.run([sys.executable, "ara.py", symbol])
            else:
                print("No symbol provided. Example: python ara.py AAPL")
        else:
            print("Error: Neither run_ara.py nor ara.py found")
            print("Please ensure you're in the correct directory")
        
        input("\nPress Enter to exit...")
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
