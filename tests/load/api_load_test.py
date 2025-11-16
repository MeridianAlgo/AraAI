"""
API load testing using Locust.

Tests API performance under various load conditions.
"""

try:
    from locust import HttpUser, task, between, events
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    print("Locust not installed. Install with: pip install locust")

import json
import random


if LOCUST_AVAILABLE:
    class PredictionUser(HttpUser):
        """Simulated user making prediction requests."""
        
        wait_time = between(1, 3)  # Wait 1-3 seconds between requests
        
        def on_start(self):
            """Called when user starts."""
            self.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            
        @task(3)
        def predict_stock(self):
            """Make a prediction request (most common)."""
            symbol = random.choice(self.symbols)
            
            response = self.client.post(
                "/api/v1/predict",
                json={
                    "symbol": symbol,
                    "days": random.choice([5, 10, 30]),
                    "include_analysis": False
                },
                headers={"Content-Type": "application/json"}
            )
            
        @task(1)
        def get_market_regime(self):
            """Get market regime."""
            symbol = random.choice(self.symbols)
            
            response = self.client.get(
                f"/api/v1/market/regime?symbol={symbol}"
            )
            
        @task(1)
        def list_models(self):
            """List available models."""
            response = self.client.get("/api/v1/models/status")
            
        @task(2)
        def health_check(self):
            """Health check endpoint."""
            response = self.client.get("/health")


    class BacktestUser(HttpUser):
        """Simulated user running backtests."""
        
        wait_time = between(5, 10)  # Backtests take longer
        
        def on_start(self):
            """Called when user starts."""
            self.symbols = ["AAPL", "MSFT", "GOOGL"]
            
        @task
        def run_backtest(self):
            """Run a backtest."""
            symbol = random.choice(self.symbols)
            
            response = self.client.post(
                "/api/v1/backtest",
                json={
                    "symbol": symbol,
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31"
                },
                headers={"Content-Type": "application/json"}
            )


    class PortfolioUser(HttpUser):
        """Simulated user optimizing portfolios."""
        
        wait_time = between(3, 7)
        
        @task
        def optimize_portfolio(self):
            """Optimize a portfolio."""
            assets = random.sample(
                ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BTC-USD"],
                k=3
            )
            
            response = self.client.post(
                "/api/v1/portfolio/optimize",
                json={
                    "assets": assets,
                    "risk_tolerance": random.choice(["conservative", "moderate", "aggressive"])
                },
                headers={"Content-Type": "application/json"}
            )


def run_load_test():
    """Run load test programmatically."""
    import subprocess
    import sys
    
    if not LOCUST_AVAILABLE:
        print("Locust not available. Skipping load test.")
        return
        
    print("Starting load test...")
    print("This will test the API with simulated users.")
    print("Press Ctrl+C to stop.")
    
    try:
        # Run locust in headless mode
        subprocess.run([
            sys.executable, "-m", "locust",
            "-f", __file__,
            "--headless",
            "--users", "10",
            "--spawn-rate", "2",
            "--run-time", "60s",
            "--host", "http://localhost:8000"
        ])
    except KeyboardInterrupt:
        print("\nLoad test stopped.")
    except Exception as e:
        print(f"Load test error: {e}")


if __name__ == "__main__":
    if LOCUST_AVAILABLE:
        # Run with: locust -f api_load_test.py --host=http://localhost:8000
        print("Run with: locust -f api_load_test.py --host=http://localhost:8000")
        print("Or call run_load_test() to run programmatically")
    else:
        print("Install locust to run load tests: pip install locust")
