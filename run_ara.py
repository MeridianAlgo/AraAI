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

def check_api_setup():
    """Check if API keys are properly configured"""
    if not os.path.exists('.env'):
        return False, "No .env file found"
    
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key or api_key == 'your_gemini_api_key_here':
        return False, "Gemini API key not configured"
    
    return True, "API keys configured"

def main():
    console.clear()
    
    # Header
    header = Text()
    header.append("🚀 ARA AI STOCK ANALYSIS 🚀\n", style="bold cyan")
    header.append("Perfect Prediction System", style="bold yellow")
    
    console.print(Panel.fit(header, border_style="cyan"))
    
    # Check API setup
    api_ok, api_message = check_api_setup()
    
    if not api_ok:
        console.print(f"\n❌ {api_message}")
        console.print("\n[bold red]API Setup Required![/bold red]")
        console.print("\n[yellow]Please run:[/yellow]")
        console.print("  python test_api.py")
        console.print("\nOr set up your API keys manually:")
        console.print("1. Get Gemini API key: https://makersuite.google.com/app/apikey")
        console.print("2. Edit .env file and add your key")
        console.print("3. Run this launcher again")
        return
    
    console.print(f"\n✅ {api_message}")
    
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