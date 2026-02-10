"""
Cryptocurrency-specific prediction commands
"""

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ara.cli.utils import format_percentage, format_price, handle_error

console = Console()


@click.group()
def crypto():
    """Cryptocurrency prediction and analysis commands"""
    pass


@crypto.command("predict")
@click.argument("symbol")
@click.option("--days", "-d", default=7, type=int, help="Number of days to predict")
@click.option(
    "--exchange",
    "-ex",
    multiple=True,
    help="Specific exchanges to use (binance, coinbase, kraken)",
)
@click.option("--onchain", is_flag=True, help="Include on-chain metrics analysis")
@click.option("--defi", is_flag=True, help="Include DeFi metrics analysis")
@click.option("--export", "-e", type=click.Path(), help="Export results to file")
def crypto_predict(symbol, days, exchange, onchain, defi, export):
    """
    Predict cryptocurrency prices

    Examples:

        ara crypto predict BTC --days 7

        ara crypto predict ETH --onchain --defi

        ara crypto predict BTC --exchange binance --exchange coinbase
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Analyzing {symbol.upper()}...", total=None)

            from ara.api.prediction_engine import PredictionEngine

            engine = PredictionEngine()

            # Make crypto prediction with specific options
            result = engine.predict_crypto(
                symbol=symbol.upper(),
                days=days,
                exchanges=list(exchange) if exchange else None,
                include_onchain=onchain,
                include_defi=defi,
            )

            progress.update(task, completed=True)

        # Display results
        _display_crypto_results(result, onchain, defi)

        # Export if requested
        if export:
            _export_crypto_results(result, export)
            console.print(f"[green]Results exported to {export}[/green]")

    except Exception as e:
        handle_error(e, "Crypto prediction failed")


@crypto.command("list")
@click.option("--top", "-t", default=50, type=int, help="Number of top cryptocurrencies to list")
def crypto_list(top):
    """List supported cryptocurrencies"""
    try:
        from ara.data.crypto_provider import CryptoExchangeProvider

        provider = CryptoExchangeProvider()
        symbols = provider.get_supported_symbols()[:top]

        console.print(f"\n[bold cyan]Top {top} Supported Cryptocurrencies[/bold cyan]\n")

        # Display in columns
        cols = 4
        for i in range(0, len(symbols), cols):
            row = symbols[i : i + cols]
            console.print("  ".join(f"{sym:15}" for sym in row))

    except Exception as e:
        handle_error(e, "Failed to list cryptocurrencies")


@crypto.command("compare")
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--metric",
    "-m",
    type=click.Choice(["price", "volume", "volatility", "correlation"]),
    default="price",
    help="Comparison metric",
)
def crypto_compare(symbols, metric):
    """
    Compare multiple cryptocurrencies

    Example:

        ara crypto compare BTC ETH BNB --metric volatility
    """
    try:
        console.print(f"\n[bold cyan]Comparing {', '.join(symbols)} by {metric}[/bold cyan]\n")

        from ara.correlation.analyzer import CorrelationAnalyzer

        analyzer = CorrelationAnalyzer()

        if metric == "correlation":
            # Show correlation matrix
            corr_matrix = analyzer.calculate_correlation_matrix(list(symbols))
            _display_correlation_matrix(corr_matrix, symbols)
        else:
            # Show comparison table
            _display_crypto_comparison(symbols, metric)

    except Exception as e:
        handle_error(e, "Comparison failed")


def _display_crypto_results(result, show_onchain, show_defi):
    """Display cryptocurrency prediction results"""
    console.print(f"\n[bold cyan]Crypto Prediction Results for {result['symbol']}[/bold cyan]")
    console.print(f"Current Price: [green]{format_price(result['current_price'])}[/green]")
    console.print(f"24h Volume: {result.get('volume_24h', 'N/A')}")
    console.print(f"Market Cap: {result.get('market_cap', 'N/A')}")

    # Predictions table
    table = Table(title="\nPrice Predictions")
    table.add_column("Day", style="cyan")
    table.add_column("Date", style="white")
    table.add_column("Predicted Price", style="green")
    table.add_column("Change", style="yellow")
    table.add_column("Confidence", style="magenta")

    for pred in result.get("predictions", []):
        change = pred["predicted_return"]
        change_color = "green" if change > 0 else "red"

        table.add_row(
            str(pred["day"]),
            pred["date"],
            format_price(pred["predicted_price"]),
            f"[{change_color}]{format_percentage(change)}[/{change_color}]",
            f"{pred['confidence']:.1%}",
        )

    console.print(table)

    # On-chain metrics
    if show_onchain and "onchain_metrics" in result:
        console.print("\n[bold]On-Chain Metrics:[/bold]")
        metrics = result["onchain_metrics"]
        console.print(f"  Active Addresses: {metrics.get('active_addresses', 'N/A')}")
        console.print(f"  Transaction Volume: {metrics.get('tx_volume', 'N/A')}")
        console.print(f"  Network Hash Rate: {metrics.get('hash_rate', 'N/A')}")
        console.print(f"  Exchange Inflow: {metrics.get('exchange_inflow', 'N/A')}")

    # DeFi metrics
    if show_defi and "defi_metrics" in result:
        console.print("\n[bold]DeFi Metrics:[/bold]")
        metrics = result["defi_metrics"]
        console.print(f"  Total Value Locked: {metrics.get('tvl', 'N/A')}")
        console.print(f"  Lending Rate: {metrics.get('lending_rate', 'N/A')}")
        console.print(f"  Borrowing Rate: {metrics.get('borrowing_rate', 'N/A')}")


def _display_correlation_matrix(matrix, symbols):
    """Display correlation matrix"""
    table = Table(title="Correlation Matrix")
    table.add_column("", style="cyan")
    for sym in symbols:
        table.add_column(sym, style="white")

    for i, sym1 in enumerate(symbols):
        row = [sym1]
        for j, sym2 in enumerate(symbols):
            corr = matrix[i][j]
            color = "green" if corr > 0.7 else "yellow" if corr > 0.3 else "white"
            row.append(f"[{color}]{corr:.2f}[/{color}]")
        table.add_row(*row)

    console.print(table)


def _display_crypto_comparison(symbols, metric):
    """Display cryptocurrency comparison"""
    table = Table(title=f"Comparison by {metric.title()}")
    table.add_column("Symbol", style="cyan")
    table.add_column("Current Price", style="green")
    table.add_column(metric.title(), style="yellow")
    table.add_column("24h Change", style="magenta")

    # Fetch data for each symbol
    from ara.data.crypto_provider import CryptoExchangeProvider

    provider = CryptoExchangeProvider()

    for symbol in symbols:
        data = provider.fetch_realtime(symbol)
        table.add_row(
            symbol,
            format_price(data.get("price", 0)),
            str(data.get(metric, "N/A")),
            format_percentage(data.get("change_24h", 0)),
        )

    console.print(table)


def _export_crypto_results(result, filepath):
    """Export crypto results to file"""
    import json
    from pathlib import Path

    path = Path(filepath)

    if path.suffix == ".json":
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)
    elif path.suffix == ".csv":
        import pandas as pd

        df = pd.DataFrame(result["predictions"])
        df.to_csv(filepath, index=False)
