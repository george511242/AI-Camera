import os
from typing import Awaitable, Callable
from websocket.ConnectionManager import connection_manager
from websocket.Message import WebSocketMessage
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from settings import settings
from loguru import logger

def register_startup_event(
    app: FastAPI,
) -> Callable[[], Awaitable[None]]:
    """
    Actions to run on application startup.

    This function uses fastAPI app to store data
    in the state, such as db_engine.

    :param app: the fastAPI application.
    :return: function that actually performs actions.
    """
    app.state.websocket = connection_manager
    @app.on_event("startup")
    async def _startup() -> None:
        logger.info('AUO AI Camera service starting ...')
        app.middleware_stack = None
        app.middleware_stack = app.build_middleware_stack()
        pass

    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        logger.info(f"Client {client_id} attempting to connect")

        try:
            await connection_manager.connect(client_id, websocket)
            logger.info(f"Client {client_id} connected successfully")

            while True:
                try:
                    data = await websocket.receive_text()
                    logger.debug(f"Received data from client {client_id}: {data}")

                except WebSocketDisconnect:
                    logger.info(f"Client {client_id} disconnected normally")
                    break
                except Exception as e:
                    logger.error(f"Error receiving data from client {client_id}: {str(e)}")
                    break

        except Exception as e:
            logger.error(f"Error in WebSocket connection for client {client_id}: {str(e)}")

        finally:
            connection_manager.disconnect(client_id)
            logger.info(f"Client {client_id} disconnected and cleaned up")
            try:
                if len(connection_manager.active_connections) > 0:
                    await connection_manager.broadcast(
                        WebSocketMessage('client', data=f"Client #{client_id} left the chat")
                    )
            except Exception as e:
                logger.error(f"Error broadcasting disconnect message for client {client_id}: {str(e)}")

    return _startup


def register_shutdown_event(
    app: FastAPI,
) -> Callable[[], Awaitable[None]]:
    """
    Actions to run on application's shutdown.

    :param app: fastAPI application.
    :return: function that actually performs actions.
    """

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.warning('AUO AI Camera service stopping ...')
        pass 

    return _shutdown
