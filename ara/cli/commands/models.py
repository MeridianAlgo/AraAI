"""
Model management commands
"""

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from ara.cli.utils import handle_error

console = Console()


@click.group()
def models():
    """Model management and training commands"""
    pass


@models.command('list')
@click.option('--symbol', '-s', type=str, help='Filter by symbol')
@click.option('--show-metrics', is_flag=True, help='Show performance metrics')
def list_models(symbol, show_metrics):
    """
    List available models
    
    Example:
    
        ara models list --show-metrics
        
        ara models list --symbol AAPL
    """
    try:
        from ara.models.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        models_list = registry.list_models(symbol=symbol)
        
        if not models_list:
            console.print("[yellow]No models found[/yellow]")
            return
        
        # Display models table
        table = Table(title="Available Models")
        table.add_column("Model ID", style="cyan")
        table.add_column("Symbol", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Version", style="magenta")
        table.add_column("Trained", style="white")
        
        if show_metrics:
            table.add_column("Accuracy", style="green")
            table.add_column("Sharpe", style="yellow")
        
        for model in models_list:
            row = [
                model['id'],
                model['symbol'],
                model['type'],
                model['version'],
                model['trained_date']
            ]
            
            if show_metrics:
                row.extend([
                    f"{model.get('accuracy', 0):.2%}",
                    f"{model.get('sharpe_ratio', 0):.2f}"
                ])
            
            table.add_row(*row)
        
        console.print(table)
        
    except Exception as e:
        handle_error(e, "Failed to list models")


@models.command('train')
@click.argument('symbol')
@click.option('--model-type', '-m', type=click.Choice(['transformer', 'cnn_lstm', 'ensemble', 'all']),
              default='ensemble', help='Model type to train')
@click.option('--period', '-p', type=str, default='5y', help='Training data period')
@click.option('--epochs', '-e', type=int, default=100, help='Number of training epochs')
@click.option('--validate', is_flag=True, help='Run validation after training')
def train(symbol, model_type, period, epochs, validate):
    """
    Train a new model
    
    Examples:
    
        ara models train AAPL --model-type transformer
        
        ara models train BTC --period 3y --epochs 200 --validate
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Training {model_type} model for {symbol.upper()}...", total=100)
            
            from ara.models.model_registry import ModelRegistry
            
            registry = ModelRegistry()
            
            # Train model
            result = registry.train_model(
                symbol=symbol.upper(),
                model_type=model_type,
                period=period,
                epochs=epochs,
                progress_callback=lambda p: progress.update(task, completed=p)
            )
            
            progress.update(task, completed=100)
        
        # Display training results
        console.print(f"\n[green]Model trained successfully![/green]")
        console.print(f"Model ID: {result['model_id']}")
        console.print(f"Training Time: {result['training_time']:.2f}s")
        console.print(f"Final Loss: {result['final_loss']:.4f}")
        
        # Run validation if requested
        if validate:
            console.print("\n[cyan]Running validation...[/cyan]")
            validation_result = registry.validate_model(result['model_id'])
            _display_validation_results(validation_result)
        
    except Exception as e:
        handle_error(e, "Model training failed")


@models.command('compare')
@click.argument('symbol')
@click.option('--models', '-m', type=str, help='Model IDs to compare (comma-separated)')
@click.option('--metric', type=click.Choice(['accuracy', 'sharpe', 'speed']),
              default='accuracy', help='Comparison metric')
def compare(symbol, models, metric):
    """
    Compare model performance
    
    Example:
    
        ara models compare AAPL --metric accuracy
        
        ara models compare BTC --models model1,model2,model3
    """
    try:
        from ara.models.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        
        # Get models to compare
        if models:
            model_ids = models.split(',')
        else:
            # Compare all models for the symbol
            model_ids = None
        
        # Compare models
        comparison = registry.compare_models(
            symbol=symbol.upper(),
            model_ids=model_ids,
            metric=metric
        )
        
        # Display comparison
        _display_model_comparison(comparison, metric)
        
    except Exception as e:
        handle_error(e, "Model comparison failed")


@models.command('deploy')
@click.argument('model_id')
@click.option('--force', is_flag=True, help='Force deployment without validation')
def deploy(model_id, force):
    """
    Deploy a model to production
    
    Example:
    
        ara models deploy model_abc123
    """
    try:
        from ara.models.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        
        if not force:
            # Validate before deployment
            console.print("[cyan]Validating model...[/cyan]")
            validation = registry.validate_model(model_id)
            
            if validation['accuracy'] < 0.70:
                console.print(f"[red]Model accuracy ({validation['accuracy']:.2%}) is below threshold (70%)[/red]")
                if not click.confirm("Deploy anyway?"):
                    return
        
        # Deploy model
        result = registry.deploy_model(model_id)
        
        console.print(f"[green]Model {model_id} deployed successfully![/green]")
        console.print(f"Deployment ID: {result['deployment_id']}")
        console.print(f"Status: {result['status']}")
        
    except Exception as e:
        handle_error(e, "Model deployment failed")


@models.command('delete')
@click.argument('model_id')
@click.option('--force', is_flag=True, help='Skip confirmation')
def delete(model_id, force):
    """
    Delete a model
    
    Example:
    
        ara models delete model_abc123
    """
    try:
        from ara.models.model_registry import ModelRegistry
        
        if not force:
            if not click.confirm(f"Are you sure you want to delete model {model_id}?"):
                return
        
        registry = ModelRegistry()
        registry.delete_model(model_id)
        
        console.print(f"[green]Model {model_id} deleted successfully[/green]")
        
    except Exception as e:
        handle_error(e, "Model deletion failed")


def _display_validation_results(result):
    """Display model validation results"""
    table = Table(title="Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Accuracy", f"{result['accuracy']:.2%}")
    table.add_row("Precision", f"{result['precision']:.2%}")
    table.add_row("Recall", f"{result['recall']:.2%}")
    table.add_row("F1 Score", f"{result['f1_score']:.2%}")
    table.add_row("MAE", f"{result['mae']:.4f}")
    table.add_row("RMSE", f"{result['rmse']:.4f}")
    
    console.print(table)


def _display_model_comparison(comparison, metric):
    """Display model comparison results"""
    console.print(f"\n[bold cyan]Model Comparison (by {metric.title()})[/bold cyan]")
    
    table = Table()
    table.add_column("Rank", style="cyan")
    table.add_column("Model ID", style="white")
    table.add_column("Type", style="yellow")
    table.add_column(metric.title(), style="green")
    table.add_column("Inference Time", style="magenta")
    
    for i, model in enumerate(comparison['models'], 1):
        table.add_row(
            str(i),
            model['id'],
            model['type'],
            f"{model[metric]:.2%}" if metric == 'accuracy' else f"{model[metric]:.2f}",
            f"{model['inference_time']:.3f}s"
        )
    
    console.print(table)
    
    # Recommendation
    best_model = comparison['models'][0]
    console.print(f"\n[green]Recommended: {best_model['id']} ({best_model['type']})[/green]")
