import asyncio
import logging
import sys
from typing import Union, Any
import requests, redis
import json
from loguru import logger

from settings import settings
from upload_to_wms import init_camera
from utils.readConfig import config, get_config
from settings import settings
POOL = redis.ConnectionPool(host=settings.redis_host,
                            port=settings.redis_port, decode_responses=True)
REDIS = redis.Redis(connection_pool=POOL)

class TruncatingFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, max_length=100):
        super().__init__(fmt, datefmt)
        self.max_length = max_length
    
    def format(self, record):
        formatted = super().format(record)
        
        if len(formatted) > self.max_length:
            return formatted[:self.max_length] + "..."
        return formatted

class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentation.

    This handler intercepts all log requests and
    passes them to loguru.

    For more info see:
    https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        """
        Propagates logs to loguru.

        :param record: record to log.
        """
        try:
            level: Union[str, int] = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )

def configure_logging() -> None:  # pragma: no cover
    """Configures logging."""
    intercept_handler = InterceptHandler()
    logging.basicConfig(handlers=[intercept_handler], level=logging.NOTSET)

    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("uvicorn."):
            logging.getLogger(logger_name).handlers = []
        if logger_name.startswith("taskiq."):
            logging.getLogger(logger_name).root.handlers = [intercept_handler]

    # change handler for default uvicorn logger
    logging.getLogger("uvicorn").handlers = [intercept_handler]
    logging.getLogger("uvicorn.access").handlers = [intercept_handler]

    # set logs output, level and format
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level.value,
    )
    
    logger_server = get_config('API', 'wms_device_log')
    device_id = get_config('SYSTEM', 'device_id', 'Default')
    if not logger_server:
        init_camera()
        logger_server = get_config('API', 'wms_device_log')
    
    # def remote_record(record: str):
    #     cid = REDIS.get('cid')
    #     if not cid or cid == '':
    #         return
    #     if not logger_server:
    #         return
    #     splited_record = record.split(':::')
    #     if 'urllib3.connectionpool' in splited_record[2]:
    #         return
    #     try:
    #         payload = {
    #             "time": splited_record[0],
    #             "level": splited_record[1],
    #             "source": splited_record[2],
    #             "message": f"[{device_id}]: " + splited_record[3]
    #         }
    #         header = {
    #             'authorization': f'Basic {settings.remote_log_token}',
    #             'cid': cid,
    #             'logtype': 'camera'
    #         }
    #         response = requests.post(logger_server, json.dumps(payload), headers=header, timeout=2)
    #     except Exception as e:
    #         # logger.debug(f'Failed to submit log to remote logger server: CID: {cid}')
    #         pass
        
    # logger.add(remote_record, format="{time}:::{level}:::{module}:{function}:::{message}", level='INFO', enqueue=True)
    logger.add("/logs/persion_tracking_log.txt", rotation="00:00", encoding="utf-8", enqueue=True, retention="5 days", compression="zip")
