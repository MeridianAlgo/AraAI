#!/usr/bin/env python3
"""
API Key Test Script for Ara AI Stock Analysis
Tests if your API keys are properly configured
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def test_yahoo_finance():
    """Test Yahoo Finance connection"""
    try:
        import yfinance as yf
        
        # Test with a simple stock fetch
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if info and 'symbol' in info:
            return True, "Yahoo Finance connection successful"
        else:
            return False, "Yahoo Finance returned empty data"
            
    except ImportError:
        return False, "yfinance package not installed"
    except Exception as e:
        return False, f"Yahoo Finance error: {str(e)}"

def main():
    console.print(Panel.fit(
        "[bold cyan]Ara AI Stock Analysis - API Key Test[/bold cyan]",
        border_style="cyan"
    ))
    
    # Load environment variables
    if os.path.exists('.env'):
        load_dotenv()
        console.print("✅ Found .env file")
    else:
        console.print("❌ No .env file found")
        console.print("Please run the installation script first or create a .env file")
        return
    
    console.print("\n[bold yellow]Testing System Components...[/bold yellow]\n")
    
    # Test Yahoo Finance
    console.print("🔍 Testing Yahoo Finance connection...")
    success, message = test_yahoo_finance()
    
    if success:
        console.print(f"✅ Yahoo Finance: {message}")
        console.print("\n[bold green]🎉 System ready! You can use Ara AI immediately![/bold green]")
        console.print("\n[bold cyan]Try running:[/bold cyan]")
        console.print("  python ara.py AAPL --verbose")
        console.print("  python run_ara.py")
    else:
        console.print(f"❌ Yahoo Finance: {message}")
        console.print("\n[bold red]⚠️  System setup issue[/bold red]")
        console.print("\n[bold yellow]Troubleshooting:[/bold yellow]")
        console.print("1. Check your internet connection")
        console.print("2. Ensure yfinance is installed: pip install yfinance")
        console.print("3. Try running the installation script again")
    
    # Check optional APIs
    console.print("\n[bold blue]Optional Features Status:[/bold blue]")
    
    alpaca_key = os.getenv('ALPACA_API_KEY')
    if alpaca_key and alpaca_key != 'your_alpaca_api_key_here':
        console.print("✅ Alpaca API key configured (live trading enabled)")
    else:
        console.print("⚪ Alpaca API key not configured (paper trading only)")
    
    news_key = os.getenv('NEWS_API_KEY')
    if news_key and news_key != 'your_news_api_key_here':
        console.print("✅ News API key configured (enhanced sentiment analysis)")
    else:
        console.print("⚪ News API key not configured (basic sentiment analysis)")
    
    console.print("\n[bold cyan]💡 No API keys required for basic stock analysis![/bold cyan]")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Test cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error running test: {str(e)}[/red]")