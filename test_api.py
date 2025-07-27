#!/usr/bin/env python3
"""
API Key Test Script for Ara AI Stock Analysis
Tests if your API keys are properly configured
"""

import os
import sys
from dotenv import load_dotenv
import requests
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def test_gemini_api():
    """Test Gemini API key"""
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key or api_key == 'your_gemini_api_key_here':
        return False, "API key not set or still using placeholder"
    
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": "Hello, this is a test. Please respond with 'API test successful'."}]
            }]
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            return True, "API key is valid and working"
        elif response.status_code == 400:
            return False, f"Invalid API key or request format (Status: {response.status_code})"
        elif response.status_code == 403:
            return False, "API key is invalid or access denied"
        else:
            return False, f"API request failed with status: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Network error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

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
    
    console.print("\n[bold yellow]Testing API Keys...[/bold yellow]\n")
    
    # Test Gemini API
    console.print("🔍 Testing Gemini API key...")
    success, message = test_gemini_api()
    
    if success:
        console.print(f"✅ Gemini API: {message}")
        console.print("\n[bold green]🎉 All API tests passed! You're ready to use Ara AI![/bold green]")
        console.print("\n[bold cyan]Try running:[/bold cyan]")
        console.print("  python ara.py AAPL --verbose")
    else:
        console.print(f"❌ Gemini API: {message}")
        console.print("\n[bold red]⚠️  API setup required before using Ara AI[/bold red]")
        console.print("\n[bold yellow]Setup Instructions:[/bold yellow]")
        console.print("1. Visit: https://makersuite.google.com/app/apikey")
        console.print("2. Sign in with your Google account")
        console.print("3. Click 'Create API Key'")
        console.print("4. Copy the key")
        console.print("5. Edit your .env file and replace 'your_gemini_api_key_here'")
        console.print("6. Save the file and run this test again")
    
    # Check other optional APIs
    console.print("\n[bold blue]Optional API Status:[/bold blue]")
    
    alpaca_key = os.getenv('ALPACA_API_KEY')
    if alpaca_key and alpaca_key != 'your_alpaca_api_key_here':
        console.print("✅ Alpaca API key configured")
    else:
        console.print("⚪ Alpaca API key not configured (optional for live trading)")
    
    news_key = os.getenv('NEWS_API_KEY')
    if news_key and news_key != 'your_news_api_key_here':
        console.print("✅ News API key configured")
    else:
        console.print("⚪ News API key not configured (optional for sentiment analysis)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Test cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error running test: {str(e)}[/red]")