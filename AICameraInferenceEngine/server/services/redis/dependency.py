from typing import Union

from redis import Redis
from starlette.requests import Request
from taskiq import TaskiqDepends


def get_redis_pool(
    request: Request = TaskiqDepends(),
) -> Union[Redis, None]:  # pragma: no cover
    return request.app.state.redis_pool
