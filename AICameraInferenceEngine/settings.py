import enum
import os
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

from pydantic_settings import BaseSettings

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):  # noqa: WPS600
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 5555
    workers_count: int = 1  # quantity of workers for uvicorn
    workers_timeout: int = 300  # timeout for workers
    reload: bool = False  # Enable uvicorn reloading

    # Inference
    max_fps: float = 1 / 10
    max_debug_images: int = 2000
    debug_image_folder: str = '/storage/debug_image'
    max_fallback_queue: int = 50000  # Sycn records to S4M pending Queue

    # Current environment
    environment: str = "dev"

    # Logger
    log_level: LogLevel = LogLevel.DEBUG

    # Variables for Redis
    redis_host: str = "aicamera_redis"
    redis_port: int = 6379
    # redis_user: Optional[str] = None
    # redis_pass: Optional[str] = None
    # redis_base: Optional[int] = None

    remote_log_token: str = 'c3BhY2U0bTpxJHNHaFgkaXZKcD5Celtt'

    # backend
    backend_host: str = 'aicamera_web'
    backend_port: str = '5000'

    @property
    def backend_url(self) -> str:
        return f'http://{self.backend_host}:{self.backend_port}'

    # database
    temp_queue_sqlite_path: str = '/app/storage/temp_queue.sqlite'

    # NPU setting
    useNPU: bool = True
    rknpu_platform: str = "rk3566"
    age_model_version: str = "v2"


settings = Settings()
