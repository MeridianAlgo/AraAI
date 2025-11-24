"""
WebSocket endpoint handlers for real-time updates
"""

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from typing import Optional
import asyncio
import json
import logging
from datetime import datetime

from ara.api.websocket.connection_manager import connection_manager
from ara.api.auth.dependencies import get_current_user_ws
from ara.core.exceptions import AraAIException

logger = logging.getLogger(__name__)


async def handle_predictions_ws(
    websocket: WebSocket,
    symbol: str,
    user_id: Optional[str] = None
):
    """
    WebSocket handler for real-time prediction updates
    
    Endpoint: /ws/predictions/{symbol}
    
    Features:
    - Real-time prediction updates when new data arrives
    - Confidence score updates
    - Model performance metrics
    
    Args:
        websocket: WebSocket connection
        symbol: Asset symbol to watch
        user_id: Authenticated user ID (optional)
    """
    connection_id = await connection_manager.connect(
        websocket,
        channel="predictions",
        user_id=user_id
    )
    
    # Set the symbol this connection is watching
    connection_manager.set_connection_symbol(connection_id, symbol.upper())
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "channel": "predictions",
            "symbol": symbol.upper(),
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat(),
            "message": f"Connected to prediction updates for {symbol.upper()}"
        })
        
        # Listen for messages from client
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                msg_type = message.get("type")
                
                if msg_type == "pong":
                    # Update heartbeat
                    connection_manager.update_heartbeat(connection_id)
                
                elif msg_type == "subscribe":
                    # Change symbol subscription
                    new_symbol = message.get("symbol", "").upper()
                    if new_symbol:
                        connection_manager.set_connection_symbol(connection_id, new_symbol)
                        await websocket.send_json({
                            "type": "subscribed",
                            "symbol": new_symbol,
                            "timestamp": datetime.now().isoformat()
                        })
                
                elif msg_type == "request_prediction":
                    # Client requesting immediate prediction
                    await websocket.send_json({
                        "type": "prediction_requested",
                        "symbol": symbol.upper(),
                        "status": "processing",
                        "timestamp": datetime.now().isoformat(),
                        "message": "Prediction request received, processing..."
                    })
                    # Note: Actual prediction would be triggered here
                    # and sent via broadcast when complete
                
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
            
            except WebSocketDisconnect:
                break
            
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Error processing message",
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from predictions/{symbol}")
    
    except Exception as e:
        logger.error(f"Error in predictions WebSocket: {e}")
    
    finally:
        connection_manager.disconnect(connection_id)


async def handle_market_data_ws(
    websocket: WebSocket,
    symbol: str,
    user_id: Optional[str] = None
):
    """
    WebSocket handler for streaming real-time market data
    
    Endpoint: /ws/market-data/{symbol}
    
    Features:
    - Real-time price updates
    - Volume and trade data
    - Order book updates (if available)
    
    Args:
        websocket: WebSocket connection
        symbol: Asset symbol to watch
        user_id: Authenticated user ID (optional)
    """
    connection_id = await connection_manager.connect(
        websocket,
        channel="market-data",
        user_id=user_id
    )
    
    # Set the symbol this connection is watching
    connection_manager.set_connection_symbol(connection_id, symbol.upper())
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "channel": "market-data",
            "symbol": symbol.upper(),
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat(),
            "message": f"Connected to market data stream for {symbol.upper()}"
        })
        
        # Listen for messages from client
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                msg_type = message.get("type")
                
                if msg_type == "pong":
                    # Update heartbeat
                    connection_manager.update_heartbeat(connection_id)
                
                elif msg_type == "subscribe":
                    # Change symbol subscription
                    new_symbol = message.get("symbol", "").upper()
                    if new_symbol:
                        connection_manager.set_connection_symbol(connection_id, new_symbol)
                        await websocket.send_json({
                            "type": "subscribed",
                            "symbol": new_symbol,
                            "timestamp": datetime.now().isoformat()
                        })
                
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
            
            except WebSocketDisconnect:
                break
            
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Error processing message",
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from market-data/{symbol}")
    
    except Exception as e:
        logger.error(f"Error in market-data WebSocket: {e}")
    
    finally:
        connection_manager.disconnect(connection_id)


async def handle_alerts_ws(
    websocket: WebSocket,
    user_id: Optional[str] = None
):
    """
    WebSocket handler for streaming alerts and notifications
    
    Endpoint: /ws/alerts
    
    Features:
    - Real-time alert notifications
    - Price alerts
    - Prediction confidence alerts
    - System notifications
    
    Args:
        websocket: WebSocket connection
        user_id: Authenticated user ID (optional)
    """
    connection_id = await connection_manager.connect(
        websocket,
        channel="alerts",
        user_id=user_id
    )
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "channel": "alerts",
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to alerts stream"
        })
        
        # Listen for messages from client
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                msg_type = message.get("type")
                
                if msg_type == "pong":
                    # Update heartbeat
                    connection_manager.update_heartbeat(connection_id)
                
                elif msg_type == "configure_alerts":
                    # Client configuring alert preferences
                    alert_config = message.get("config", {})
                    await websocket.send_json({
                        "type": "alerts_configured",
                        "config": alert_config,
                        "timestamp": datetime.now().isoformat(),
                        "message": "Alert configuration updated"
                    })
                
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
            
            except WebSocketDisconnect:
                break
            
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Error processing message",
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from alerts")
    
    except Exception as e:
        logger.error(f"Error in alerts WebSocket: {e}")
    
    finally:
        connection_manager.disconnect(connection_id)


# Helper function to broadcast prediction updates
async def broadcast_prediction_update(
    symbol: str,
    prediction_data: dict
):
    """
    Broadcast prediction update to all connected clients watching this symbol
    
    Args:
        symbol: Asset symbol
        prediction_data: Prediction data to broadcast
    """
    message = {
        "type": "prediction_update",
        "symbol": symbol.upper(),
        "data": prediction_data,
        "timestamp": datetime.now().isoformat()
    }
    
    await connection_manager.broadcast(
        message,
        channel="predictions",
        symbol=symbol.upper()
    )


# Helper function to broadcast market data updates
async def broadcast_market_data(
    symbol: str,
    market_data: dict
):
    """
    Broadcast market data update to all connected clients watching this symbol
    
    Args:
        symbol: Asset symbol
        market_data: Market data to broadcast
    """
    message = {
        "type": "market_data_update",
        "symbol": symbol.upper(),
        "data": market_data,
        "timestamp": datetime.now().isoformat()
    }
    
    await connection_manager.broadcast(
        message,
        channel="market-data",
        symbol=symbol.upper()
    )


# Helper function to send alerts
async def send_alert(
    alert_data: dict,
    user_id: Optional[str] = None
):
    """
    Send alert to connected clients
    
    Args:
        alert_data: Alert data to send
        user_id: Optional user ID to send to specific user
    """
    message = {
        "type": "alert",
        "data": alert_data,
        "timestamp": datetime.now().isoformat()
    }
    
    await connection_manager.broadcast(
        message,
        channel="alerts"
    )
