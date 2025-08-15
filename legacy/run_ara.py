#!/usr/bin/env python3
"""
Ara AI Stock Analysis - Interactive Launcher
Easy-to-use launcher with API key validation and symbol input
"""

import os
import sys
import subprocess
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

console = Console()

def check_system_ready():
    """Check if system is ready to run"""
    try:
        import yfinance as yf
        # Test basic functionality
        ticker = yf.Ticker("AAPL")
        return True, "System ready - Yahoo Finance working"
    except ImportError:
        return False, "yfinance package not installed"
    except Exception as e:
        return False, f"System check failed: {str(e)}"

def main():
    console.clear()
    
    # Header
    header = Text()
    header.append("🚀 ARA AI STOCK ANALYSIS 🚀\n", style="bold cyan")
    header.append("Perfect Prediction System", style="bold yellow")
    
    console.print(Panel.fit(header, border_style="cyan"))
    
    # Check system readiness
    system_ok, system_message = check_system_ready()
    
    if not system_ok:
        console.print(f"\n❌ {system_message}")
        console.print("\n[bold red]System Setup Required![/bold red]")
        console.print("\n[yellow]Please run:[/yellow]")
        console.print("  python test_api.py")
        console.print("\nOr install missing dependencies:")
        console.print("  pip install yfinance")
        return
    
    console.print(f"\n✅ {system_message}")
    
    # Get stock symbol
    console.print("\n[bold green]Ready to analyze stocks![/bold green]")
    console.print("\n[yellow]Popular symbols:[/yellow] AAPL, NVDA, TSLA, MSFT, GOOGL, AMZN, META")
    
    symbol = Prompt.ask("\n[bold cyan]Enter stock symbol[/bold cyan]", default="AAPL")
    
    if not symbol:
        console.print("[red]No symbol entered. Exiting.[/red]")
        return
    
    # Ask for verbose mode
    verbose = Prompt.ask("\n[yellow]Detailed analysis?[/yellow]", choices=["y", "n"], default="y")
    
    # Build command
    cmd = ["python", "ara.py", symbol.upper()]
    if verbose.lower() == 'y':
        cmd.append("--verbose")
    
    console.print(f"\n[bold green]🔍 Analyzing {symbol.upper()}...[/bold green]")
    console.print("[dim]This may take a moment for the first run...[/dim]\n")
    
    try:
        # Run the analysis
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Error running analysis: {e}[/red]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
    
    console.print("\n[dim]Press Enter to exit...[/dim]")
    input()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Launcher cancelled[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")