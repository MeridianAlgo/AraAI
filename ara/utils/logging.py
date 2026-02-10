"""
Structured logging system with context
Supports JSON and text formats
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ara.config import get_config


class JSONFormatter(logging.Formatter):
    """JSON log formatter"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_data)


class ContextLogger:
    """
    Logger with context support
    Allows adding context that persists across log calls
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}

    def add_context(self, **kwargs) -> None:
        """Add context fields"""
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context"""
        self.context.clear()

    def _log(self, level: int, message: str, **kwargs) -> None:
        """Internal log method with context"""
        extra = {**self.context, **kwargs}
        self.logger.log(level, message, extra={"extra": extra})

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message"""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message"""
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback"""
        self.logger.exception(message, extra={"extra": {**self.context, **kwargs}})


def setup_logging(config: Optional[Any] = None) -> None:
    """
    Setup logging configuration

    Args:
        config: Configuration object (uses global config if None)
    """
    if config is None:
        config = get_config()

    log_config = config.logging

    # Get log level
    level = getattr(logging, log_config.level.upper())

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if log_config.format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    if log_config.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_config.file:
        log_path = Path(log_config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> ContextLogger:
    """
    Get a context logger

    Args:
        name: Logger name

    Returns:
        ContextLogger instance
    """
    return ContextLogger(name)


# Initialize logging on import
try:
    setup_logging()
except Exception:
    # Fallback to basic logging if config fails
    logging.basicConfig(level=logging.INFO)
