"""
WebSocket API routes for real-time updates.
"""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional

from app.core.websocket_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time alert updates.

    Args:
        websocket: WebSocket connection
        client_id: Optional client identifier

    Usage:
        ws://localhost:8000/ws/alerts
        ws://localhost:8000/ws/alerts?client_id=dashboard_1
    """
    await manager.connect(websocket, client_id)

    try:
        # Send welcome message
        await manager.send_welcome_message(websocket)

        # Keep connection alive and handle incoming messages
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()

            # Echo back for testing (optional)
            logger.debug(f"Received from client {client_id}: {data}")

            # Handle client messages (ping, subscribe, etc.)
            if data == "ping":
                await manager.send_personal_message(
                    {'type': 'pong', 'data': {}},
                    websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected normally")

    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(websocket)


@router.get("/ws/stats")
async def get_websocket_stats():
    """
    Get WebSocket connection statistics.

    Returns:
        Connection statistics including active clients
    """
    stats = manager.get_connection_stats()

    return {
        'success': True,
        **stats
    }
