"""
Configuration management commands
"""

import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from ara.cli.utils import handle_error

console = Console()


@click.group()
def config():
    """Configuration management commands"""
    pass


@config.command("init")
@click.option("--interactive", "-i", is_flag=True, help="Interactive configuration wizard")
def init(interactive):
    """
    Initialize ARA AI configuration

    Example:

        ara config init --interactive
    """
    try:
        from ara.config.config import Config
        from pathlib import Path

        config_path = Path.home() / ".ara" / "config.yaml"

        if config_path.exists():
            if not Confirm.ask(f"Configuration already exists at {config_path}. Overwrite?"):
                return

        if interactive:
            _interactive_config_wizard(config_path)
        else:
            # Create default config
            Config.create_default_config(config_path)
            console.print(f"[green]Default configuration created at {config_path}[/green]")
            console.print("[yellow]Run 'ara config init --interactive' to customize[/yellow]")

    except Exception as e:
        handle_error(e, "Configuration initialization failed")


@config.command("show")
@click.option("--section", "-s", type=str, help="Show specific section")
def show(section):
    """
    Show current configuration

    Examples:

        ara config show

        ara config show --section data_providers
    """
    try:
        from ara.config.config import Config

        config = Config()

        if section:
            _display_config_section(config, section)
        else:
            _display_full_config(config)

    except Exception as e:
        handle_error(e, "Failed to show configuration")


@config.command("set")
@click.argument("key")
@click.argument("value")
def set_value(key, value):
    """
    Set configuration value

    Examples:

        ara config set cache.enabled true

        ara config set data_providers.primary alpha_vantage
    """
    try:
        from ara.config.config import Config

        config = Config()
        config.set(key, value)
        config.save()

        console.print(f"[green]Configuration updated: {key} = {value}[/green]")

    except Exception as e:
        handle_error(e, "Failed to set configuration")


@config.command("get")
@click.argument("key")
def get_value(key):
    """
    Get configuration value

    Example:

        ara config get cache.enabled
    """
    try:
        from ara.config.config import Config

        config = Config()
        value = config.get(key)

        console.print(f"{key}: [green]{value}[/green]")

    except Exception as e:
        handle_error(e, "Failed to get configuration")


@config.command("validate")
def validate():
    """Validate configuration"""
    try:
        from ara.config.config import Config

        config = Config()
        errors = config.validate()

        if not errors:
            console.print("[green]Configuration is valid[/green]")
        else:
            console.print("[red]Configuration errors found:[/red]")
            for error in errors:
                console.print(f"  • {error}")

    except Exception as e:
        handle_error(e, "Configuration validation failed")


@config.command("reset")
@click.option("--force", is_flag=True, help="Skip confirmation")
def reset(force):
    """Reset configuration to defaults"""
    try:
        if not force:
            if not Confirm.ask("Are you sure you want to reset configuration to defaults?"):
                return

        from ara.config.config import Config
        from pathlib import Path

        config_path = Path.home() / ".ara" / "config.yaml"
        Config.create_default_config(config_path)

        console.print("[green]Configuration reset to defaults[/green]")

    except Exception as e:
        handle_error(e, "Configuration reset failed")


def _interactive_config_wizard(config_path):
    """Interactive configuration wizard"""
    console.print("\n[bold cyan]ARA AI Configuration Wizard[/bold cyan]\n")

    config_data = {}

    # Data providers
    console.print("[bold]Data Providers[/bold]")
    config_data["data_providers"] = {
        "primary": Prompt.ask(
            "Primary data provider",
            choices=["yfinance", "alpha_vantage", "polygon"],
            default="yfinance",
        ),
        "fallback": Prompt.ask(
            "Fallback data provider",
            choices=["yfinance", "alpha_vantage", "polygon"],
            default="alpha_vantage",
        ),
    }

    # API keys
    console.print("\n[bold]API Keys (optional)[/bold]")
    alpha_vantage_key = Prompt.ask("Alpha Vantage API key", default="")
    polygon_key = Prompt.ask("Polygon API key", default="")

    if alpha_vantage_key or polygon_key:
        config_data["api_keys"] = {}
        if alpha_vantage_key:
            config_data["api_keys"]["alpha_vantage"] = alpha_vantage_key
        if polygon_key:
            config_data["api_keys"]["polygon"] = polygon_key

    # Cache settings
    console.print("\n[bold]Cache Settings[/bold]")
    config_data["cache"] = {
        "enabled": Confirm.ask("Enable caching", default=True),
        "ttl": int(Prompt.ask("Cache TTL (seconds)", default="3600")),
        "max_size": int(Prompt.ask("Max cache size (MB)", default="1000")),
    }

    # Model settings
    console.print("\n[bold]Model Settings[/bold]")
    config_data["models"] = {
        "default_ensemble": Confirm.ask("Use ensemble by default", default=True),
        "gpu_enabled": Confirm.ask("Enable GPU acceleration", default=True),
        "auto_retrain": Confirm.ask("Enable automatic retraining", default=False),
    }

    # Save configuration
    from ara.config.config import Config

    Config.save_config(config_data, config_path)

    console.print(f"\n[green]Configuration saved to {config_path}[/green]")
    console.print("[yellow]You can modify it later with 'ara config set' commands[/yellow]")


def _display_full_config(config):
    """Display full configuration"""
    console.print("\n[bold cyan]ARA AI Configuration[/bold cyan]\n")

    config_dict = config.to_dict()

    for section, values in config_dict.items():
        console.print(f"[bold]{section}:[/bold]")
        if isinstance(values, dict):
            for key, value in values.items():
                console.print(f"  {key}: [green]{value}[/green]")
        else:
            console.print(f"  [green]{values}[/green]")
        console.print()


def _display_config_section(config, section):
    """Display specific configuration section"""
    console.print(f"\n[bold cyan]{section}[/bold cyan]\n")

    values = config.get_section(section)

    if isinstance(values, dict):
        table = Table()
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")

        for key, value in values.items():
            table.add_row(key, str(value))

        console.print(table)
    else:
        console.print(f"[green]{values}[/green]")
