"""
Backtesting commands
"""

from datetime import datetime, timedelta

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ara.cli.utils import handle_error

console = Console()


@click.command()
@click.argument("symbol")
@click.option("--start", "-s", type=str, help="Start date (YYYY-MM-DD)")
@click.option("--end", "-e", type=str, help="End date (YYYY-MM-DD)")
@click.option(
    "--period",
    "-p",
    type=str,
    default="1y",
    help="Period (1y, 2y, 5y) - alternative to start/end",
)
@click.option(
    "--strategy",
    type=click.Choice(["buy_hold", "momentum", "mean_reversion"]),
    default="momentum",
    help="Trading strategy",
)
@click.option("--initial-capital", type=float, default=10000.0, help="Initial capital")
@click.option("--report", "-r", type=click.Path(), help="Save detailed report to file")
@click.option("--plot", is_flag=True, help="Generate equity curve plot")
def backtest(symbol, start, end, period, strategy, initial_capital, report, plot):
    """
    Backtest prediction accuracy on historical data

    Examples:

        ara backtest AAPL --period 2y

        ara backtest MSFT --start 2020-01-01 --end 2023-12-31

        ara backtest BTC --strategy momentum --plot
    """
    try:
        # Parse dates
        if start and end:
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d")
        else:
            end_date = datetime.now()
            if period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "2y":
                start_date = end_date - timedelta(days=730)
            elif period == "5y":
                start_date = end_date - timedelta(days=1825)
            else:
                start_date = end_date - timedelta(days=365)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running backtest for {symbol.upper()}...", total=100)

            from ara.backtesting.engine import BacktestEngine

            engine = BacktestEngine()

            # Run backtest
            result = engine.run_backtest(
                symbol=symbol.upper(),
                start_date=start_date,
                end_date=end_date,
                strategy=strategy,
                initial_capital=initial_capital,
                progress_callback=lambda p: progress.update(task, completed=p),
            )

            progress.update(task, completed=100)

        # Display results
        _display_backtest_results(result, symbol, start_date, end_date)

        # Generate plot if requested
        if plot:
            _generate_equity_plot(result, symbol)
            console.print("[green]Equity curve plot saved[/green]")

        # Save report if requested
        if report:
            _save_backtest_report(result, report)
            console.print(f"[green]Detailed report saved to {report}[/green]")

    except Exception as e:
        handle_error(e, "Backtest failed")


def _display_backtest_results(result, symbol, start_date, end_date):
    """Display backtest results"""
    console.print(f"\n[bold cyan]Backtest Results for {symbol.upper()}[/bold cyan]")
    console.print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    console.print(f"Total Predictions: {result['total_predictions']}")

    # Performance metrics table
    table = Table(title="\nPerformance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Accuracy metrics
    table.add_row("Accuracy", f"{result['accuracy']:.2%}")
    table.add_row("Precision", f"{result['precision']:.2%}")
    table.add_row("Recall", f"{result['recall']:.2%}")
    table.add_row("F1 Score", f"{result['f1_score']:.2%}")
    table.add_row("Directional Accuracy", f"{result['directional_accuracy']:.2%}")

    # Error metrics
    table.add_row("", "")  # Separator
    table.add_row("MAE", f"{result['mae']:.4f}")
    table.add_row("RMSE", f"{result['rmse']:.4f}")
    table.add_row("MAPE", f"{result['mape']:.2%}")

    # Financial metrics
    table.add_row("", "")  # Separator
    table.add_row("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
    table.add_row("Sortino Ratio", f"{result['sortino_ratio']:.2f}")
    table.add_row("Calmar Ratio", f"{result['calmar_ratio']:.2f}")
    table.add_row("Max Drawdown", f"{result['max_drawdown']:.2%}")
    table.add_row("Win Rate", f"{result['win_rate']:.2%}")
    table.add_row("Profit Factor", f"{result['profit_factor']:.2f}")

    console.print(table)

    # Regime-specific performance
    if "regime_performance" in result:
        console.print("\n[bold]Performance by Market Regime:[/bold]")
        regime_table = Table()
        regime_table.add_column("Regime", style="cyan")
        regime_table.add_column("Accuracy", style="green")
        regime_table.add_column("Sharpe", style="yellow")

        for regime, metrics in result["regime_performance"].items():
            regime_table.add_row(
                regime.title(),
                f"{metrics['accuracy']:.2%}",
                f"{metrics['sharpe_ratio']:.2f}",
            )

        console.print(regime_table)


def _generate_equity_plot(result, symbol):
    """Generate equity curve plot"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        equity_curve = result["equity_curve"]
        drawdown_curve = result["drawdown_curve"]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Equity Curve", "Drawdown"),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve["equity"],
                name="Equity",
                line=dict(color="green", width=2),
            ),
            row=1,
            col=1,
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown_curve.index,
                y=drawdown_curve["drawdown"],
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="red", width=1),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(title=f"Backtest Results - {symbol.upper()}", height=800, showlegend=True)

        fig.write_html(f"backtest_{symbol.lower()}_equity_curve.html")

    except ImportError:
        console.print("[yellow]Plotly not installed. Skipping plot generation.[/yellow]")


def _save_backtest_report(result, filepath):
    """Save detailed backtest report"""
    import json
    from pathlib import Path

    path = Path(filepath)

    if path.suffix == ".json":
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)
    elif path.suffix == ".html":
        _generate_html_report(result, filepath)
    else:
        raise ValueError(f"Unsupported report format: {path.suffix}")


def _generate_html_report(result, filepath):
    """Generate HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Backtest Report</h1>
        <h2>Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Accuracy</td><td>{result["accuracy"]:.2%}</td></tr>
            <tr><td>Sharpe Ratio</td><td>{result["sharpe_ratio"]:.2f}</td></tr>
            <tr><td>Max Drawdown</td><td>{result["max_drawdown"]:.2%}</td></tr>
        </table>
    </body>
    </html>
    """

    with open(filepath, "w") as f:
        f.write(html_content)
