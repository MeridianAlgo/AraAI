"""
Prediction commands for stocks and forex
"""

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from ara.cli.utils import format_price, format_percentage, handle_error

console = Console()


@click.command()
@click.argument('symbol')
@click.option('--days', '-d', default=5, type=int, help='Number of days to predict')
@click.option('--analysis', '-a', type=click.Choice(['basic', 'full', 'minimal']), 
              default='basic', help='Analysis level')
@click.option('--no-cache', is_flag=True, help='Skip cache and generate fresh predictions')
@click.option('--export', '-e', type=click.Path(), help='Export results to file (JSON/CSV)')
@click.option('--confidence-threshold', '-c', type=float, default=0.0, 
              help='Minimum confidence threshold (0-1)')
@click.pass_context
def predict(ctx, symbol, days, analysis, no_cache, export, confidence_threshold):
    """
    Predict stock or forex prices
    
    Examples:
    
        ara predict AAPL --days 7
        
        ara predict EURUSD --analysis full
        
        ara predict MSFT --export results.json
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Analyzing {symbol.upper()}...", total=None)
            
            # Import here to avoid slow startup
            from ara.api.prediction_engine import PredictionEngine
            
            engine = PredictionEngine()
            
            # Make prediction
            result = engine.predict(
                symbol=symbol.upper(),
                days=days,
                include_analysis=(analysis != 'minimal'),
                use_cache=(not no_cache)
            )
            
            progress.update(task, completed=True)
        
        # Filter by confidence if threshold is set
        if confidence_threshold > 0:
            if result.get('confidence', {}).get('overall', 0) < confidence_threshold:
                console.print(f"[yellow]Warning: Prediction confidence ({result['confidence']['overall']:.2%}) "
                            f"is below threshold ({confidence_threshold:.2%})[/yellow]")
                return
        
        # Display results
        _display_prediction_results(result, analysis)
        
        # Export if requested
        if export:
            _export_results(result, export)
            console.print(f"[green]Results exported to {export}[/green]")
            
    except Exception as e:
        handle_error(e, "Prediction failed")


def _display_prediction_results(result, analysis_level):
    """Display prediction results in a formatted table"""
    console.print(f"\n[bold cyan]Prediction Results for {result['symbol']}[/bold cyan]")
    console.print(f"Current Price: [green]{format_price(result['current_price'])}[/green]")
    
    # Confidence score
    confidence = result.get('confidence', {})
    conf_color = "green" if confidence.get('overall', 0) > 0.75 else "yellow" if confidence.get('overall', 0) > 0.5 else "red"
    console.print(f"Confidence: [{conf_color}]{confidence.get('overall', 0):.1%}[/{conf_color}]")
    
    # Predictions table
    table = Table(title="\nPrice Predictions")
    table.add_column("Day", style="cyan")
    table.add_column("Date", style="white")
    table.add_column("Predicted Price", style="green")
    table.add_column("Change", style="yellow")
    table.add_column("Confidence", style="magenta")
    
    for pred in result.get('predictions', []):
        change = pred['predicted_return']
        change_color = "green" if change > 0 else "red"
        
        table.add_row(
            str(pred['day']),
            pred['date'],
            format_price(pred['predicted_price']),
            f"[{change_color}]{format_percentage(change)}[/{change_color}]",
            f"{pred['confidence']:.1%}"
        )
    
    console.print(table)
    
    # Market regime
    if 'regime' in result:
        regime = result['regime']
        console.print(f"\n[bold]Market Regime:[/bold] {regime['current_regime'].upper()}")
        console.print(f"Regime Confidence: {regime['confidence']:.1%}")
    
    # Top factors (if full analysis)
    if analysis_level == 'full' and 'explanations' in result:
        explanations = result['explanations']
        console.print("\n[bold]Top Contributing Factors:[/bold]")
        for i, factor in enumerate(explanations.get('top_factors', [])[:5], 1):
            contrib_color = "green" if factor['contribution'] > 0 else "red"
            console.print(f"  {i}. {factor['name']}: [{contrib_color}]{format_percentage(factor['contribution'])}[/{contrib_color}]")
            console.print(f"     {factor['description']}")


def _export_results(result, filepath):
    """Export results to file"""
    import json
    from pathlib import Path
    
    path = Path(filepath)
    
    if path.suffix == '.json':
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    elif path.suffix == '.csv':
        import pandas as pd
        df = pd.DataFrame(result['predictions'])
        df.to_csv(filepath, index=False)
    else:
        raise ValueError(f"Unsupported export format: {path.suffix}")
