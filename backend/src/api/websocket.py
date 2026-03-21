"""WebSocket endpoint for streaming query responses."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.websocket("/ws/query")
async def query_stream(websocket: WebSocket):
    """Stream query responses via WebSocket.

    Full implementation in Sprint 6. Currently accepts connection and echoes back.
    """
    await websocket.accept()
    try:
        while True:
            await websocket.receive_json()
            await websocket.send_json({
                "type": "error",
                "content": "Streaming query not yet implemented. See Sprint 6.",
            })
    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
