from fastapi import FastAPI, Request, BackgroundTasks
from routers.AppRouter import AppRouter
from routers.DeviceRouter import DeviceRouter
from pathlib import Path
from server.lifetime import register_shutdown_event, register_startup_event
from server.logging import configure_logging
from loguru import logger
from main import lifespan, main
# Application Environment Configuration
APP_ROOT = Path(__file__).parent.parent
def get_app() -> FastAPI:
    """
    Get FastAPI application.

    This is the main constructor of an application.

    :return: application.
    """
    configure_logging()
    logger.info('Starting AUO AI Camera Application ...')
    app = FastAPI(
        title='Person Tracking',
        version='0.1.0',
        # openapi_tags=Tags,
        lifespan=lifespan
    )
    logger.info('Register startup and shutdown events ...')
    register_startup_event(app)
    register_shutdown_event(app)
    app.include_router(DeviceRouter)
    app.include_router(AppRouter)
    return app