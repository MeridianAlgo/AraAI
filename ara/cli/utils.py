"""
CLI utility functions
"""

from typing import Any

from rich.console import Console

console = Console()


def format_price(price: float) -> str:
    """Format price with currency symbol"""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.2f}"
    else:
        return f"${price:.6f}"


def format_percentage(value: float) -> str:
    """Format percentage value"""
    return f"{value:+.2%}"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousands separator"""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def handle_error(error: Exception, context: str = "Operation failed"):
    """Handle and display errors"""
    from ara.core.exceptions import (
        APIError,
        AraAIException,
        CacheError,
        DataProviderError,
        ModelError,
        ValidationError,
    )

    error_type = type(error).__name__

    # Custom error messages for known exceptions
    if isinstance(error, DataProviderError):
        console.print(f"[red]Data Provider Error:[/red] {str(error)}")
        console.print("[yellow]Tip: Check your internet connection and API keys[/yellow]")
    elif isinstance(error, ModelError):
        console.print(f"[red]Model Error:[/red] {str(error)}")
        console.print(
            "[yellow]Tip: Try retraining the model or using a different model type[/yellow]"
        )
    elif isinstance(error, ValidationError):
        console.print(f"[red]Validation Error:[/red] {str(error)}")
        console.print("[yellow]Tip: Check your input parameters[/yellow]")
    elif isinstance(error, CacheError):
        console.print(f"[red]Cache Error:[/red] {str(error)}")
        console.print("[yellow]Tip: Try clearing the cache with --no-cache flag[/yellow]")
    elif isinstance(error, APIError):
        console.print(f"[red]API Error:[/red] {str(error)}")
        console.print("[yellow]Tip: Check API rate limits and authentication[/yellow]")
    elif isinstance(error, AraAIException):
        console.print(f"[red]{context}:[/red] {str(error)}")
    else:
        # Generic error
        console.print(f"[red]{context}:[/red] {str(error)}")
        console.print(f"[dim]Error type: {error_type}[/dim]")

    # Show traceback in verbose mode
    import os

    if os.getenv("ARA_DEBUG", "").lower() == "true":
        console.print_exception()


def confirm_action(message: str, default: bool = False) -> bool:
    """Prompt user for confirmation"""
    from rich.prompt import Confirm

    return Confirm.ask(message, default=default)


def prompt_choice(message: str, choices: list, default: Any = None) -> str:
    """Prompt user to choose from options"""
    from rich.prompt import Prompt

    return Prompt.ask(message, choices=choices, default=default)


def display_progress(message: str, total: int = None):
    """Create a progress bar"""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn() if total else TextColumn(""),
        console=console,
    )

    return progress


def create_table(title: str = None, **kwargs):
    """Create a formatted table"""
    from rich.table import Table

    return Table(title=title, **kwargs)


def print_success(message: str):
    """Print success message"""
    console.print(f"[green]✓[/green] {message}")


def print_warning(message: str):
    """Print warning message"""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_error(message: str):
    """Print error message"""
    console.print(f"[red]✗[/red] {message}")


def print_info(message: str):
    """Print info message"""
    console.print(f"[cyan]ℹ[/cyan] {message}")
