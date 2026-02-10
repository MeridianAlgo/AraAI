"""
WebSocket support for real-time updates
"""

from ara.api.websocket.connection_manager import ConnectionManager
from ara.api.websocket.handlers import (
    handle_alerts_ws,
    handle_market_data_ws,
    handle_predictions_ws,
)

__all__ = [
    "ConnectionManager",
    "handle_predictions_ws",
    "handle_market_data_ws",
    "handle_alerts_ws",
]
