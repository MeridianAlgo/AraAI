"""
Portfolio management commands
"""

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from ara.cli.utils import format_price, format_percentage, handle_error

console = Console()


@click.group()
def portfolio():
    """Portfolio optimization and analysis commands"""
    pass


@portfolio.command("optimize")
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--risk-tolerance",
    "-r",
    type=click.Choice(["conservative", "moderate", "aggressive"]),
    default="moderate",
    help="Risk tolerance level",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["mpt", "black_litterman", "risk_parity", "kelly"]),
    default="mpt",
    help="Optimization method",
)
@click.option("--constraints", "-c", type=str, help="Constraints file (JSON)")
@click.option("--export", "-e", type=click.Path(), help="Export results to file")
def optimize(symbols, risk_tolerance, method, constraints, export):
    """
    Optimize portfolio allocation

    Examples:

        ara portfolio optimize AAPL MSFT GOOGL --risk-tolerance moderate

        ara portfolio optimize BTC ETH BNB --method risk_parity

        ara portfolio optimize AAPL MSFT --constraints limits.json
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Optimizing portfolio...", total=None)

            from ara.risk.optimizer import PortfolioOptimizer

            optimizer = PortfolioOptimizer()

            # Load constraints if provided
            constraint_dict = None
            if constraints:
                import json

                with open(constraints, "r") as f:
                    constraint_dict = json.load(f)

            # Optimize portfolio
            result = optimizer.optimize(
                assets=list(symbols),
                risk_tolerance=risk_tolerance,
                method=method,
                constraints=constraint_dict,
            )

            progress.update(task, completed=True)

        # Display results
        _display_optimization_results(result)

        # Export if requested
        if export:
            _export_portfolio_results(result, export)
            console.print(f"[green]Results exported to {export}[/green]")

    except Exception as e:
        handle_error(e, "Portfolio optimization failed")


@portfolio.command("analyze")
@click.argument("symbols", nargs=-1, required=True)
@click.option("--weights", "-w", type=str, help="Current weights (comma-separated)")
@click.option("--show-correlations", is_flag=True, help="Show correlation matrix")
@click.option("--risk-metrics", is_flag=True, help="Show detailed risk metrics")
def analyze(symbols, weights, show_correlations, risk_metrics):
    """
    Analyze portfolio composition and risk

    Examples:

        ara portfolio analyze AAPL MSFT GOOGL --show-correlations

        ara portfolio analyze BTC ETH --weights 0.6,0.4 --risk-metrics
    """
    try:
        from ara.risk.portfolio_analysis import PortfolioAnalyzer

        analyzer = PortfolioAnalyzer()

        # Parse weights if provided
        weight_dict = None
        if weights:
            weight_list = [float(w) for w in weights.split(",")]
            if len(weight_list) != len(symbols):
                raise ValueError("Number of weights must match number of symbols")
            weight_dict = dict(zip(symbols, weight_list))

        # Analyze portfolio
        result = analyzer.analyze(assets=list(symbols), weights=weight_dict)

        # Display results
        _display_analysis_results(result, show_correlations, risk_metrics)

    except Exception as e:
        handle_error(e, "Portfolio analysis failed")


@portfolio.command("rebalance")
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "--current-weights",
    "-c",
    type=str,
    required=True,
    help="Current weights (comma-separated)",
)
@click.option(
    "--target-weights",
    "-t",
    type=str,
    required=True,
    help="Target weights (comma-separated)",
)
@click.option("--capital", type=float, default=10000.0, help="Total portfolio value")
def rebalance(symbols, current_weights, target_weights, capital):
    """
    Calculate rebalancing trades
    
    Example:
    
        ara portfolio rebalance AAPL MSFT GOOGL \\
            --current-weights 0.4,0.3,0.3 \\
            --target-weights 0.33,0.33,0.34 \\
            --capital 50000
    """
    try:
        from ara.risk.portfolio_analysis import PortfolioAnalyzer

        # Parse weights
        current = [float(w) for w in current_weights.split(",")]
        target = [float(w) for w in target_weights.split(",")]

        if len(current) != len(symbols) or len(target) != len(symbols):
            raise ValueError("Number of weights must match number of symbols")

        analyzer = PortfolioAnalyzer()

        # Calculate rebalancing trades
        trades = analyzer.calculate_rebalancing_trades(
            assets=list(symbols),
            current_weights=dict(zip(symbols, current)),
            target_weights=dict(zip(symbols, target)),
            total_capital=capital,
        )

        # Display trades
        _display_rebalancing_trades(trades, capital)

    except Exception as e:
        handle_error(e, "Rebalancing calculation failed")


def _display_optimization_results(result):
    """Display portfolio optimization results"""
    console.print("\n[bold cyan]Portfolio Optimization Results[/bold cyan]")
    console.print(f"Expected Return: [green]{format_percentage(result['expected_return'])}[/green]")
    console.print(
        f"Expected Volatility: [yellow]{format_percentage(result['expected_volatility'])}[/yellow]"
    )
    console.print(f"Sharpe Ratio: [magenta]{result['sharpe_ratio']:.2f}[/magenta]")

    # Optimal weights table
    table = Table(title="\nOptimal Asset Allocation")
    table.add_column("Asset", style="cyan")
    table.add_column("Weight", style="green")
    table.add_column("Expected Return", style="yellow")

    for asset, weight in result["optimal_weights"].items():
        table.add_row(
            asset,
            f"{weight:.2%}",
            format_percentage(result.get("asset_returns", {}).get(asset, 0)),
        )

    console.print(table)

    # Risk metrics
    console.print("\n[bold]Risk Metrics:[/bold]")
    console.print(f"  VaR (95%): {format_percentage(result['var_95'])}")
    console.print(f"  CVaR (95%): {format_percentage(result['cvar_95'])}")


def _display_analysis_results(result, show_correlations, show_risk):
    """Display portfolio analysis results"""
    console.print("\n[bold cyan]Portfolio Analysis[/bold cyan]")

    # Composition table
    table = Table(title="Portfolio Composition")
    table.add_column("Asset", style="cyan")
    table.add_column("Weight", style="green")
    table.add_column("Value", style="yellow")
    table.add_column("Return", style="magenta")

    for asset, data in result["composition"].items():
        table.add_row(
            asset,
            f"{data['weight']:.2%}",
            format_price(data["value"]),
            format_percentage(data["return"]),
        )

    console.print(table)

    # Risk metrics
    if show_risk:
        console.print("\n[bold]Risk Metrics:[/bold]")
        metrics = result["risk_metrics"]
        console.print(f"  Portfolio Volatility: {format_percentage(metrics['volatility'])}")
        console.print(f"  Portfolio Beta: {metrics['beta']:.2f}")
        console.print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        console.print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        console.print(f"  Max Drawdown: {format_percentage(metrics['max_drawdown'])}")

    # Correlation matrix
    if show_correlations:
        console.print("\n[bold]Correlation Matrix:[/bold]")
        _display_correlation_matrix(result["correlation_matrix"])


def _display_correlation_matrix(matrix):
    """Display correlation matrix"""
    table = Table()
    symbols = list(matrix.keys())

    table.add_column("", style="cyan")
    for sym in symbols:
        table.add_column(sym, style="white")

    for sym1 in symbols:
        row = [sym1]
        for sym2 in symbols:
            corr = matrix[sym1][sym2]
            color = "green" if corr > 0.7 else "yellow" if corr > 0.3 else "white"
            row.append(f"[{color}]{corr:.2f}[/{color}]")
        table.add_row(*row)

    console.print(table)


def _display_rebalancing_trades(trades, capital):
    """Display rebalancing trades"""
    console.print(
        f"\n[bold cyan]Rebalancing Trades (Total Capital: {format_price(capital)})[/bold cyan]"
    )

    table = Table()
    table.add_column("Asset", style="cyan")
    table.add_column("Action", style="yellow")
    table.add_column("Quantity", style="white")
    table.add_column("Value", style="green")
    table.add_column("Current Weight", style="magenta")
    table.add_column("Target Weight", style="magenta")

    for trade in trades:
        action_color = "green" if trade["action"] == "BUY" else "red"
        table.add_row(
            trade["symbol"],
            f"[{action_color}]{trade['action']}[/{action_color}]",
            f"{trade['quantity']:.4f}",
            format_price(trade["value"]),
            f"{trade['current_weight']:.2%}",
            f"{trade['target_weight']:.2%}",
        )

    console.print(table)

    # Transaction costs
    total_cost = sum(trade.get("transaction_cost", 0) for trade in trades)
    console.print(f"\n[yellow]Estimated Transaction Costs: {format_price(total_cost)}[/yellow]")


def _export_portfolio_results(result, filepath):
    """Export portfolio results to file"""
    import json
    from pathlib import Path

    path = Path(filepath)

    if path.suffix == ".json":
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported export format: {path.suffix}")
