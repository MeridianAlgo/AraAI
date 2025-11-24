"""
Main CLI entry point with Click framework
"""

import click
from rich.console import Console
from ara.cli import commands

console = Console()


@click.group()
@click.version_option(version="4.0.0", prog_name="ARA AI")
@click.pass_context
def cli(ctx):
    """
    ARA AI - World-Class Financial Prediction System
    
    Supports stocks, cryptocurrencies, and forex with advanced ML models.
    """
    ctx.ensure_object(dict)
    ctx.obj['console'] = console


# Register command groups
cli.add_command(commands.predict)
cli.add_command(commands.crypto)
cli.add_command(commands.backtest)
cli.add_command(commands.portfolio)
cli.add_command(commands.models)
cli.add_command(commands.market)
cli.add_command(commands.config)
cli.add_command(commands.monitoring)


def main():
    """Main entry point for the CLI"""
    cli(obj={})


if __name__ == '__main__':
    main()
