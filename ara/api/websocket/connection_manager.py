"""
WebSocket connection manager for handling multiple client connections
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates

    Features:
    - Connection tracking by channel and user
    - Heartbeat mechanism for connection health
    - Broadcast and targeted messaging
    - Automatic cleanup of dead connections
    """

    def __init__(self, heartbeat_interval: int = 30):
        """
        Initialize connection manager

        Args:
            heartbeat_interval: Seconds between heartbeat pings
        """
        # Active connections: {channel: {connection_id: websocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {}

        # Connection metadata: {connection_id: {user_id, channel, connected_at}}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}

        # Heartbeat tracking: {connection_id: last_pong_time}
        self.heartbeats: Dict[str, datetime] = {}

        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def connect(
        self, websocket: WebSocket, channel: str, user_id: Optional[str] = None
    ) -> str:
        """
        Accept and register a new WebSocket connection

        Args:
            websocket: WebSocket connection
            channel: Channel name (e.g., 'predictions', 'market-data', 'alerts')
            user_id: Optional user identifier for authentication

        Returns:
            Connection ID
        """
        await websocket.accept()

        # Generate unique connection ID
        connection_id = str(uuid.uuid4())

        # Initialize channel if needed
        if channel not in self.active_connections:
            self.active_connections[channel] = {}

        # Register connection
        self.active_connections[channel][connection_id] = websocket

        # Store metadata
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "channel": channel,
            "connected_at": datetime.now(),
            "symbol": None,  # Will be set for symbol-specific channels
        }

        # Initialize heartbeat
        self.heartbeats[connection_id] = datetime.now()

        # Start heartbeat task if not running
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(
            f"WebSocket connected: {connection_id} on channel '{channel}' "
            f"(user: {user_id or 'anonymous'})"
        )

        return connection_id

    def disconnect(self, connection_id: str):
        """
        Remove a connection

        Args:
            connection_id: Connection identifier
        """
        if connection_id not in self.connection_metadata:
            return

        metadata = self.connection_metadata[connection_id]
        channel = metadata["channel"]

        # Remove from active connections
        if channel in self.active_connections:
            self.active_connections[channel].pop(connection_id, None)

            # Clean up empty channels
            if not self.active_connections[channel]:
                del self.active_connections[channel]

        # Remove metadata and heartbeat
        self.connection_metadata.pop(connection_id, None)
        self.heartbeats.pop(connection_id, None)

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_personal_message(self, message: Dict[str, Any], connection_id: str):
        """
        Send message to a specific connection

        Args:
            message: Message data
            connection_id: Target connection ID
        """
        if connection_id not in self.connection_metadata:
            return

        metadata = self.connection_metadata[connection_id]
        channel = metadata["channel"]

        if channel not in self.active_connections:
            return

        websocket = self.active_connections[channel].get(connection_id)
        if websocket:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)

    async def broadcast(self, message: Dict[str, Any], channel: str, symbol: Optional[str] = None):
        """
        Broadcast message to all connections on a channel

        Args:
            message: Message data
            channel: Channel name
            symbol: Optional symbol filter (only send to connections watching this symbol)
        """
        if channel not in self.active_connections:
            return

        # Get connections to send to
        connections = []
        for conn_id, websocket in self.active_connections[channel].items():
            # Filter by symbol if specified
            if symbol:
                metadata = self.connection_metadata.get(conn_id, {})
                if metadata.get("symbol") != symbol:
                    continue

            connections.append((conn_id, websocket))

        # Send to all matching connections
        for conn_id, websocket in connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {conn_id}: {e}")
                self.disconnect(conn_id)

    def set_connection_symbol(self, connection_id: str, symbol: str):
        """
        Set the symbol a connection is watching

        Args:
            connection_id: Connection identifier
            symbol: Symbol to watch
        """
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["symbol"] = symbol

    def get_connection_count(self, channel: Optional[str] = None) -> int:
        """
        Get number of active connections

        Args:
            channel: Optional channel filter

        Returns:
            Connection count
        """
        if channel:
            return len(self.active_connections.get(channel, {}))
        return sum(len(conns) for conns in self.active_connections.values())

    def get_channels(self) -> Set[str]:
        """
        Get all active channels

        Returns:
            Set of channel names
        """
        return set(self.active_connections.keys())

    async def _heartbeat_loop(self):
        """
        Background task to send heartbeat pings and detect dead connections
        """
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Check all connections
                dead_connections = []
                current_time = datetime.now()

                for conn_id, last_pong in list(self.heartbeats.items()):
                    # Check if connection is stale (no pong in 2x heartbeat interval)
                    if (current_time - last_pong).total_seconds() > self.heartbeat_interval * 2:
                        dead_connections.append(conn_id)
                        continue

                    # Send ping
                    try:
                        await self.send_personal_message(
                            {"type": "ping", "timestamp": current_time.isoformat()},
                            conn_id,
                        )
                    except Exception as e:
                        logger.error(f"Error sending ping to {conn_id}: {e}")
                        dead_connections.append(conn_id)

                # Clean up dead connections
                for conn_id in dead_connections:
                    logger.warning(f"Removing dead connection: {conn_id}")
                    self.disconnect(conn_id)

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    def update_heartbeat(self, connection_id: str):
        """
        Update heartbeat timestamp for a connection (called on pong)

        Args:
            connection_id: Connection identifier
        """
        if connection_id in self.heartbeats:
            self.heartbeats[connection_id] = datetime.now()


# Global connection manager instance
connection_manager = ConnectionManager()
