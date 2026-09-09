from fastapi import FastAPI
from redis import ConnectionPool, Redis
from configs.Environment import get_environment_variables
from loguru import logger
from settings import settings
env = get_environment_variables()

def init_redis(app: FastAPI) -> None:  # pragma: no cover
    """
    Creates connection pool for redis.

    :param app: current fastapi application.
    """
    app.state.redis_pool = ConnectionPool(
    # host='aicamera_redis', port=6379, decode_responses=True)
    host=settings.redis_host, port=settings.redis_port, decode_responses=True)

def shutdown_redis(app: FastAPI) -> None:  # pragma: no cover
    """
    Closes redis connection pool.

    :param app: current FastAPI app.
    """
    app.state.redis_pool.disconnect()
