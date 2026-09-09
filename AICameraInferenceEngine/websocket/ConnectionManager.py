from time import sleep
from typing import Dict, List
from websocket.Message import WebSocketMessage
from fastapi import WebSocket
from loguru import logger
from starlette.websockets import WebSocketDisconnect


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    def get_connection_count(self) -> int:
        return len(self.active_connections)

    def get_connected_clients(self) -> List[str]:
        return list(self.active_connections.keys())

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        await self.send_personal_message(WebSocketMessage('connected', client_id), websocket)
        
    def disconnect(self, client_id: str):
        if self.active_connections.get(client_id):
            self.active_connections.pop(client_id)

    async def emit(self, client_id: str, message: WebSocketMessage):
        if client_id in self.active_connections:
            try:
                await self.send_personal_message(message, self.active_connections[client_id])
            except (WebSocketDisconnect, ConnectionResetError, Exception) as e:
                logger.warning(f"Failed to emit message to client {client_id}: {str(e)}")
                self.disconnect(client_id)
        
    async def send_personal_message(self, message: WebSocketMessage, websocket: WebSocket):
        try:
            await websocket.send_json(message.dump())
        except WebSocketDisconnect:
            logger.warning("WebSocket disconnected while sending message")
            raise
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {str(e)}")
            raise

    async def broadcast(self, message: WebSocketMessage):
        disconnected_clients = []

        for client_id in list(self.active_connections.keys()):
            try:
                websocket = self.active_connections[client_id]
                await self.send_personal_message(message, websocket)
            except (WebSocketDisconnect, ConnectionResetError, Exception) as e:
                logger.warning(f"Failed to send message to client {client_id}: {str(e)}")
                disconnected_clients.append(client_id)

        # 清理無效連接
        for client_id in disconnected_clients:
            self.disconnect(client_id)
            logger.info(f"Cleaned up disconnected client: {client_id}")

connection_manager = ConnectionManager()