"""
WebSocket connection manager for real-time updates.
Manages client connections and broadcasts alerts.
"""

import logging
import json
from typing import List, Dict, Any
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
import asyncio

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting"""

    def __init__(self):
        """Initialize connection manager"""
        self.active_connections: List[WebSocket] = []
        self.client_info: Dict[WebSocket, Dict[str, Any]] = {}
        logger.info("WebSocket ConnectionManager initialized")

    async def connect(self, websocket: WebSocket, client_id: str = None):
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            client_id: Optional client identifier
        """
        await websocket.accept()
        self.active_connections.append(websocket)

        # Store client info
        self.client_info[websocket] = {
            'client_id': client_id or f"client_{len(self.active_connections)}",
            'connected_at': datetime.now().isoformat(),
            'messages_sent': 0
        }

        logger.info(f"WebSocket client connected: {self.client_info[websocket]['client_id']} "
                   f"(Total clients: {len(self.active_connections)})")

    def disconnect(self, websocket: WebSocket):
        """
        Remove WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self.active_connections:
            client_id = self.client_info.get(websocket, {}).get('client_id', 'unknown')
            self.active_connections.remove(websocket)

            if websocket in self.client_info:
                del self.client_info[websocket]

            logger.info(f"WebSocket client disconnected: {client_id} "
                       f"(Remaining clients: {len(self.active_connections)})")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send message to specific client.

        Args:
            message: Message dictionary
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_json(message)

            if websocket in self.client_info:
                self.client_info[websocket]['messages_sent'] += 1

        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict, exclude: WebSocket = None):
        """
        Broadcast message to all connected clients.

        Args:
            message: Message dictionary to broadcast
            exclude: Optional WebSocket to exclude from broadcast
        """
        if not self.active_connections:
            logger.debug("No active connections for broadcast")
            return

        # Add timestamp to message
        message['timestamp'] = datetime.now().isoformat()

        disconnected = []

        for connection in self.active_connections:
            if connection == exclude:
                continue

            try:
                await connection.send_json(message)

                if connection in self.client_info:
                    self.client_info[connection]['messages_sent'] += 1

            except WebSocketDisconnect:
                disconnected.append(connection)
                logger.warning(f"Client disconnected during broadcast")
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

        logger.debug(f"Broadcast message to {len(self.active_connections) - len(disconnected)} clients")

    async def broadcast_alert(self, alert_data: dict):
        """
        Broadcast alert to all connected clients.

        Args:
            alert_data: Alert information dictionary
        """
        message = {
            'type': 'alert',
            'data': alert_data
        }
        await self.broadcast(message)
        logger.info(f"Alert broadcast: {alert_data.get('event_type')} (ID: {alert_data.get('id')})")

    async def broadcast_recognition(self, recognition_data: dict):
        """
        Broadcast recognition event to all connected clients.

        Args:
            recognition_data: Recognition information dictionary
        """
        message = {
            'type': 'recognition',
            'data': recognition_data
        }
        await self.broadcast(message)

    async def broadcast_status(self, status_data: dict):
        """
        Broadcast system status update to all connected clients.

        Args:
            status_data: System status dictionary
        """
        message = {
            'type': 'status',
            'data': status_data
        }
        await self.broadcast(message)

    async def send_welcome_message(self, websocket: WebSocket):
        """
        Send welcome message to newly connected client.

        Args:
            websocket: WebSocket connection
        """
        welcome = {
            'type': 'welcome',
            'data': {
                'message': 'Connected to Face Recognition Alert System',
                'client_id': self.client_info.get(websocket, {}).get('client_id'),
                'connected_at': self.client_info.get(websocket, {}).get('connected_at'),
                'active_clients': len(self.active_connections)
            }
        }
        await self.send_personal_message(welcome, websocket)

    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.

        Returns:
            Dictionary with connection stats
        """
        total_messages = sum(
            info.get('messages_sent', 0)
            for info in self.client_info.values()
        )

        return {
            'active_connections': len(self.active_connections),
            'total_messages_sent': total_messages,
            'clients': [
                {
                    'client_id': info['client_id'],
                    'connected_at': info['connected_at'],
                    'messages_sent': info['messages_sent']
                }
                for info in self.client_info.values()
            ]
        }

    async def ping_clients(self):
        """Send ping to all clients to keep connections alive"""
        ping_message = {
            'type': 'ping',
            'data': {'timestamp': datetime.now().isoformat()}
        }
        await self.broadcast(ping_message)


# Global connection manager instance
manager = ConnectionManager()
