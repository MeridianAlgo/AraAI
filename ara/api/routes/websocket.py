"""
WebSocket routes for real-time updates
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from typing import Optional
import logging

from ara.api.websocket.handlers import (
    handle_predictions_ws,
    handle_market_data_ws,
    handle_alerts_ws
)
from ara.api.auth.dependencies import get_current_user_ws

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/predictions/{symbol}")
async def websocket_predictions(
    websocket: WebSocket,
    symbol: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time prediction updates
    
    Connect to receive real-time prediction updates for a specific symbol.
    
    **Connection URL**: `ws://host:port/ws/predictions/{symbol}?token=YOUR_TOKEN`
    
    **Message Types (Client -> Server)**:
    - `{"type": "pong"}` - Heartbeat response
    - `{"type": "subscribe", "symbol": "AAPL"}` - Change symbol subscription
    - `{"type": "request_prediction"}` - Request immediate prediction
    
    **Message Types (Server -> Client)**:
    - `{"type": "connected", ...}` - Connection confirmation
    - `{"type": "ping", ...}` - Heartbeat ping
    - `{"type": "prediction_update", ...}` - New prediction available
    - `{"type": "error", ...}` - Error message
    
    Args:
        symbol: Asset symbol to watch (e.g., AAPL, BTC-USD)
        token: Optional authentication token
    """
    # Authenticate user if token provided
    user_id = None
    if token:
        try:
            user_id = await get_current_user_ws(token)
        except Exception as e:
            logger.warning(f"WebSocket authentication failed: {e}")
            # Continue without authentication (public access)
    
    await handle_predictions_ws(websocket, symbol, user_id)


@router.websocket("/market-data/{symbol}")
async def websocket_market_data(
    websocket: WebSocket,
    symbol: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for streaming real-time market data
    
    Connect to receive real-time price and volume updates for a specific symbol.
    
    **Connection URL**: `ws://host:port/ws/market-data/{symbol}?token=YOUR_TOKEN`
    
    **Message Types (Client -> Server)**:
    - `{"type": "pong"}` - Heartbeat response
    - `{"type": "subscribe", "symbol": "AAPL"}` - Change symbol subscription
    
    **Message Types (Server -> Client)**:
    - `{"type": "connected", ...}` - Connection confirmation
    - `{"type": "ping", ...}` - Heartbeat ping
    - `{"type": "market_data_update", ...}` - New market data
    - `{"type": "error", ...}` - Error message
    
    Args:
        symbol: Asset symbol to watch (e.g., AAPL, BTC-USD)
        token: Optional authentication token
    """
    # Authenticate user if token provided
    user_id = None
    if token:
        try:
            user_id = await get_current_user_ws(token)
        except Exception as e:
            logger.warning(f"WebSocket authentication failed: {e}")
            # Continue without authentication (public access)
    
    await handle_market_data_ws(websocket, symbol, user_id)


@router.websocket("/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for streaming alerts and notifications
    
    Connect to receive real-time alerts and notifications.
    
    **Connection URL**: `ws://host:port/ws/alerts?token=YOUR_TOKEN`
    
    **Message Types (Client -> Server)**:
    - `{"type": "pong"}` - Heartbeat response
    - `{"type": "configure_alerts", "config": {...}}` - Configure alert preferences
    
    **Message Types (Server -> Client)**:
    - `{"type": "connected", ...}` - Connection confirmation
    - `{"type": "ping", ...}` - Heartbeat ping
    - `{"type": "alert", ...}` - New alert notification
    - `{"type": "error", ...}` - Error message
    
    Args:
        token: Optional authentication token
    """
    # Authenticate user if token provided
    user_id = None
    if token:
        try:
            user_id = await get_current_user_ws(token)
        except Exception as e:
            logger.warning(f"WebSocket authentication failed: {e}")
            # Continue without authentication (public access)
    
    await handle_alerts_ws(websocket, user_id)
