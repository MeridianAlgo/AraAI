"""
CLI command modules
"""

from ara.cli.commands.predict import predict
from ara.cli.commands.crypto import crypto
from ara.cli.commands.backtest import backtest
from ara.cli.commands.portfolio import portfolio
from ara.cli.commands.models import models
from ara.cli.commands.market import market
from ara.cli.commands.config import config
from ara.cli.commands.monitoring import monitoring

__all__ = [
    "predict",
    "crypto",
    "backtest",
    "portfolio",
    "models",
    "market",
    "config",
    "monitoring",
]
