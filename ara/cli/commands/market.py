"""
Market analysis commands
"""

import click
from rich.console import Console
from rich.table import Table
from ara.cli.utils import handle_error

console = Console()


@click.group()
def market():
    """Market analysis and monitoring commands"""
    pass


@market.command("regime")
@click.argument("symbol")
@click.option("--history", "-h", is_flag=True, help="Show regime history")
def regime(symbol, history):
    """
    Detect current market regime

    Examples:

        ara market regime AAPL

        ara market regime BTC --history
    """
    try:
        from ara.models.regime_detector import RegimeDetector

        detector = RegimeDetector()
        result = detector.detect(symbol.upper())

        # Display current regime
        console.print(f"\n[bold cyan]Market Regime for {symbol.upper()}[/bold cyan]")

        regime_colors = {
            "bull": "green",
            "bear": "red",
            "sideways": "yellow",
            "high_volatility": "magenta",
        }

        current_regime = result["current_regime"]
        color = regime_colors.get(current_regime, "white")

        console.print(f"Current Regime: [{color}]{current_regime.upper()}[/{color}]")
        console.print(f"Confidence: {result['confidence']:.1%}")
        console.print(f"Duration: {result['duration_in_regime']} days")

        # Transition probabilities
        console.print("\n[bold]Transition Probabilities:[/bold]")
        for regime_type, prob in result["transition_probabilities"].items():
            console.print(f"  {regime_type.title()}: {prob:.1%}")

        # Regime features
        console.print("\n[bold]Regime Characteristics:[/bold]")
        for feature, value in result["regime_features"].items():
            console.print(f"  {feature}: {value:.4f}")

        # History
        if history:
            _display_regime_history(result.get("history", []))

    except Exception as e:
        handle_error(e, "Regime detection failed")


@market.command("sentiment")
@click.argument("symbol")
@click.option(
    "--sources",
    "-s",
    multiple=True,
    type=click.Choice(["twitter", "reddit", "news", "all"]),
    default=["all"],
    help="Sentiment sources",
)
@click.option(
    "--timeframe", "-t", type=str, default="24h", help="Timeframe (1h, 24h, 7d)"
)
def sentiment(symbol, sources, timeframe):
    """
    Analyze market sentiment

    Examples:

        ara market sentiment AAPL

        ara market sentiment BTC --sources twitter --sources reddit

        ara market sentiment TSLA --timeframe 7d
    """
    try:
        from ara.sentiment.aggregator import SentimentAggregator

        aggregator = SentimentAggregator()

        # Get sentiment
        if "all" in sources:
            sources = ["twitter", "reddit", "news"]

        result = aggregator.analyze(
            symbol=symbol.upper(), sources=list(sources), timeframe=timeframe
        )

        # Display sentiment
        _display_sentiment_results(result, symbol)

    except Exception as e:
        handle_error(e, "Sentiment analysis failed")


@market.command("correlations")
@click.argument("symbols", nargs=-1, required=True)
@click.option("--window", "-w", type=int, default=30, help="Rolling window in days")
@click.option("--heatmap", is_flag=True, help="Generate correlation heatmap")
def correlations(symbols, window, heatmap):
    """
    Analyze asset correlations

    Examples:

        ara market correlations AAPL MSFT GOOGL

        ara market correlations BTC ETH BNB --window 90 --heatmap
    """
    try:
        from ara.correlation.analyzer import CorrelationAnalyzer

        analyzer = CorrelationAnalyzer()

        # Calculate correlations
        result = analyzer.calculate_correlation_matrix(
            assets=list(symbols), window=window
        )

        # Display correlation matrix
        _display_correlation_matrix(result, symbols)

        # Identify pairs
        pairs = analyzer.identify_pairs_trading_opportunities(list(symbols))
        if pairs:
            console.print("\n[bold]Pairs Trading Opportunities:[/bold]")
            for pair in pairs:
                console.print(
                    f"  {pair['asset1']} - {pair['asset2']}: "
                    f"Correlation = {pair['correlation']:.2f}"
                )

        # Generate heatmap if requested
        if heatmap:
            _generate_correlation_heatmap(result, symbols)
            console.print("[green]Correlation heatmap saved[/green]")

    except Exception as e:
        handle_error(e, "Correlation analysis failed")


@market.command("indicators")
@click.argument("symbol")
@click.option(
    "--category",
    "-c",
    type=click.Choice(["trend", "momentum", "volatility", "volume", "all"]),
    default="all",
    help="Indicator category",
)
@click.option("--timeframe", "-t", type=str, default="1d", help="Timeframe")
def indicators(symbol, category, timeframe):
    """
    Calculate technical indicators

    Examples:

        ara market indicators AAPL --category trend

        ara market indicators BTC --timeframe 4h
    """
    try:
        from ara.features.calculator import IndicatorCalculator

        calculator = IndicatorCalculator()

        # Calculate indicators
        result = calculator.calculate_indicators(
            symbol=symbol.upper(), category=category, timeframe=timeframe
        )

        # Display indicators
        _display_indicators(result, category)

    except Exception as e:
        handle_error(e, "Indicator calculation failed")


def _display_regime_history(history):
    """Display regime history"""
    console.print("\n[bold]Regime History:[/bold]")

    table = Table()
    table.add_column("Date", style="cyan")
    table.add_column("Regime", style="yellow")
    table.add_column("Duration", style="white")
    table.add_column("Confidence", style="green")

    for entry in history[-10:]:  # Last 10 entries
        table.add_row(
            entry["date"],
            entry["regime"].title(),
            f"{entry['duration']} days",
            f"{entry['confidence']:.1%}",
        )

    console.print(table)


def _display_sentiment_results(result, symbol):
    """Display sentiment analysis results"""
    console.print(f"\n[bold cyan]Sentiment Analysis for {symbol.upper()}[/bold cyan]")

    # Overall sentiment
    overall = result["overall_sentiment"]
    sentiment_color = (
        "green" if overall > 0.2 else "red" if overall < -0.2 else "yellow"
    )
    console.print(
        f"Overall Sentiment: [{sentiment_color}]{overall:+.2f}[/{sentiment_color}]"
    )

    # By source
    table = Table(title="\nSentiment by Source")
    table.add_column("Source", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Volume", style="yellow")
    table.add_column("Momentum", style="magenta")

    for source, data in result["by_source"].items():
        score_color = "green" if data["score"] > 0 else "red"
        table.add_row(
            source.title(),
            f"[{score_color}]{data['score']:+.2f}[/{score_color}]",
            str(data["volume"]),
            f"{data['momentum']:+.2f}",
        )

    console.print(table)

    # Trending topics
    if "trending_topics" in result:
        console.print("\n[bold]Trending Topics:[/bold]")
        for topic in result["trending_topics"][:5]:
            console.print(f"  • {topic}")


def _display_correlation_matrix(matrix, symbols):
    """Display correlation matrix"""
    console.print("\n[bold cyan]Correlation Matrix[/bold cyan]")

    table = Table()
    table.add_column("", style="cyan")
    for sym in symbols:
        table.add_column(sym, style="white")

    for i, sym1 in enumerate(symbols):
        row = [sym1]
        for j, sym2 in enumerate(symbols):
            corr = matrix[i][j]
            if i == j:
                color = "white"
            elif corr > 0.7:
                color = "green"
            elif corr > 0.3:
                color = "yellow"
            elif corr < -0.3:
                color = "red"
            else:
                color = "white"
            row.append(f"[{color}]{corr:.2f}[/{color}]")
        table.add_row(*row)

    console.print(table)


def _display_indicators(result, category):
    """Display technical indicators"""
    console.print(f"\n[bold cyan]Technical Indicators ({category.title()})[/bold cyan]")

    table = Table()
    table.add_column("Indicator", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Signal", style="yellow")

    for indicator, data in result.items():
        signal_color = (
            "green"
            if data["signal"] == "BUY"
            else "red" if data["signal"] == "SELL" else "yellow"
        )
        table.add_row(
            indicator,
            f"{data['value']:.4f}",
            f"[{signal_color}]{data['signal']}[/{signal_color}]",
        )

    console.print(table)


def _generate_correlation_heatmap(matrix, symbols):
    """Generate correlation heatmap"""
    try:
        import plotly.graph_objects as go
        import numpy as np

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=symbols,
                y=symbols,
                colorscale="RdYlGn",
                zmid=0,
                text=np.round(matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )

        fig.update_layout(
            title="Asset Correlation Heatmap",
            xaxis_title="Assets",
            yaxis_title="Assets",
            width=800,
            height=800,
        )

        fig.write_html("correlation_heatmap.html")

    except ImportError:
        console.print(
            "[yellow]Plotly not installed. Skipping heatmap generation.[/yellow]"
        )
