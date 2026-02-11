#!/usr/bin/env python3
"""
Backtest script for ARA AI models across multiple historical periods.
Uses walk-forward validation to ensure robustness and uses the precise
feature extraction logic from the unified training system.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yfinance as yf
from rich.console import Console
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ara.backtesting.engine import BacktestConfig, BacktestEngine
from ara.models.ensemble import EnhancedEnsemble
from meridianalgo.unified_ml import UnifiedStockML

console = Console()


def get_data_with_features(symbol, start_date, end_date):
    """Fetch data and pre-calculate all 44 features using model logic"""
    console.print(f"Fetching data for {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)

    if df.empty or len(df) < 200:
        raise ValueError(f"Insufficient data for {symbol} (Need at least 200 rows, got {len(df)})")

    # Standardize columns
    df.columns = [c.lower() for c in df.columns]

    # Calculate target (next day return)
    df["target"] = df["close"].pct_change().shift(-1)

    # Use UnifiedStockML logic for indicators and features
    uml = UnifiedStockML()

    # 1. Add indicators
    df_with_indicators = uml._add_indicators(
        df.copy().rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
    )

    # 2. Extract exactly 44 features for each row
    console.print(f"Extracting 44 model features for each of {len(df)} rows...")
    feature_matrix = []
    for i in range(len(df_with_indicators)):
        # _extract_features takes a df and uses the last row
        # So we pass the slice up to i
        feat = uml._extract_features(df_with_indicators.iloc[: i + 1])
        feature_matrix.append(feat)

    feature_matrix = np.array(feature_matrix)

    # 3. Add features to a fresh dataframe to avoid name conflicts
    result_df = df.copy()
    result_df["target"] = df["target"]  # Ensure target is there

    feature_cols = []
    for i in range(44):
        col_name = f"feat_{i}"
        result_df[col_name] = feature_matrix[:, i]
        feature_cols.append(col_name)

    # Remove last row as it has no target
    result_df = result_df.dropna(subset=["target"])

    return result_df, feature_cols


def run_multi_period_backtest(symbol, asset_type, periods=5, window_size=252, model_path=None):
    """Run backtest across multiple sliding windows"""

    # Initialize model with correct path
    if model_path is None:
        if asset_type == "stock":
            model_path = "models/unified_stock_model.pt"
        else:
            model_path = "models/unified_forex_model.pt"

    model = EnhancedEnsemble(symbol=symbol, model_type=asset_type, model_path=model_path)

    def predict_fn(X):
        # X shape is (n_samples, 44)
        # model.predict handles internal batching and device placement
        pred, _ = model.predict(X)
        return pred.flatten()

    # Configuration for multiple periods (Walk-forward)
    config = BacktestConfig(
        train_window_days=window_size,
        test_window_days=30,  # 1 month test periods
        step_size_days=30,  # Step forward 1 month
        holdout_ratio=0.1,  # 10% final holdout
    )

    engine = BacktestEngine(config=config)

    # Fetch 3 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)

    data, feature_cols = get_data_with_features(symbol, start_date, end_date)

    console.print(f"Running walk-forward backtest for {symbol}...")
    result = engine.run_backtest(
        symbol=symbol,
        data=data,
        model_predict_fn=predict_fn,
        feature_columns=feature_cols,
        target_column="target",
        price_column="close",
    )

    return result


def display_results(result):
    """Display the results of the multi-period backtest"""
    console.print(f"\n[bold cyan]Multi-Period Backtest Summary: {result.symbol}[/bold cyan]")
    console.print("Backtest Framework: Walk-Forward Validation")
    console.print(
        f"Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}"
    )

    # Summary Table
    table = Table(title="Overall Aggregated Metrics")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", style="green")

    m = result.metrics
    table.add_row("Directional Accuracy", f"{m.directional_accuracy:.2%}")
    table.add_row("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")
    table.add_row("Max Drawdown", f"{m.max_drawdown:.2%}")
    table.add_row("Win Rate", f"{m.win_rate:.2%}")
    table.add_row("Profit Factor", f"{m.profit_factor:.2f}")
    table.add_row("Total Return (Strategy)", f"{m.total_return:.2%}")
    table.add_row("Annualized Return", f"{m.annualized_return:.2%}")

    console.print(table)

    # Periodic Results (Walk-forward windows)
    console.print("\n[bold]Historical Testing Periods (Walk-Forward):[/bold]")
    period_table = Table()
    period_table.add_column("Window", style="cyan")
    period_table.add_column("Date Range", style="yellow")
    period_table.add_column("Accuracy", style="green")
    period_table.add_column("Period Return", style="magenta")
    period_table.add_column("Max Drawdown", style="red")

    for i, res in enumerate(result.walk_forward_results):
        win_m = res["metrics"]["classification_metrics"]
        summ_m = res["metrics"]["summary"]
        fin_m = res["metrics"]["financial_metrics"]

        period_table.add_row(
            f"#{i+1}",
            f"{res['dates'][0].strftime('%Y-%m')} to {res['dates'][-1].strftime('%Y-%m')}",
            f"{win_m['directional_accuracy']:.2%}",
            f"{summ_m['total_return']:.2%}",
            f"{fin_m['max_drawdown']:.2%}",
        )

    console.print(period_table)

    # Monte Carlo results if available
    if result.monte_carlo_results:
        mc = result.monte_carlo_results
        mc_table = Table(title="Monte Carlo Robustness Simulation (1000 runs)")
        mc_table.add_column("Metric", style="cyan")
        mc_table.add_column("Value", style="green")
        mc_table.add_row("Probability of Positive Return", f"{mc['probability_positive']:.2%}")
        mc_table.add_row("Mean Strategy Return", f"{mc['mean_return']:.2%}")
        mc_table.add_row("95% Confidence Interval (Lower)", f"{mc['lower_bound']:.2%}")
        mc_table.add_row("95% Confidence Interval (Upper)", f"{mc['upper_bound']:.2%}")
        console.print(mc_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-period Backtester with Walk-Forward Validation"
    )
    parser.add_argument("--symbol", default="AAPL", help="Symbol to backtest")
    parser.add_argument("--type", default="stock", choices=["stock", "forex"], help="Asset type")
    parser.add_argument(
        "--window", type=int, default=252, help="Training window size in days (default: 1 year)"
    )
    parser.add_argument("--model-path", help="Optional path to specific .pt model")

    args = parser.parse_args()

    try:
        res = run_multi_period_backtest(
            args.symbol, args.type, window_size=args.window, model_path=args.model_path
        )
        display_results(res)
    except Exception as e:
        console.print(f"[bold red]Backtest failed: {e}[/bold red]")
        import traceback

        traceback.print_exc()
