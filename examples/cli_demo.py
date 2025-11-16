"""
ARA AI Enhanced CLI Demo
Demonstrates the new CLI commands and features
"""

import subprocess
import sys


def run_command(cmd):
    """Run a CLI command and display output"""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Command timed out")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run CLI demo commands"""
    print("ARA AI Enhanced CLI Demo")
    print("=" * 60)
    
    # Demo commands (these will show help/usage)
    demo_commands = [
        # Show version and help
        "python -m ara.cli --version",
        "python -m ara.cli --help",
        
        # Show command group help
        "python -m ara.cli predict --help",
        "python -m ara.cli crypto --help",
        "python -m ara.cli backtest --help",
        "python -m ara.cli portfolio --help",
        "python -m ara.cli models --help",
        "python -m ara.cli market --help",
        "python -m ara.cli config --help",
    ]
    
    print("\nNote: This demo shows CLI help messages.")
    print("To run actual predictions, use commands like:")
    print("  python -m ara.cli predict AAPL --days 7")
    print("  python -m ara.cli crypto predict BTC --days 7")
    print("  python -m ara.cli backtest AAPL --period 1y")
    print()
    
    success_count = 0
    for cmd in demo_commands:
        if run_command(cmd):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Demo completed: {success_count}/{len(demo_commands)} commands successful")
    print('='*60)


if __name__ == '__main__':
    main()
