from typing import Sequence
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from ..model import Model, Result
from ..logger import Log, LoggerModel
import time
import cv2


class Workflow:
    class TS_MODE(Enum):
        CURRENT = 1
        IDX = 2

    def __init__(self, models: list[Model | list[Model]] = [], loggers: list[LoggerModel] = [], ts_mode=TS_MODE.CURRENT):
        self.ts_mode = ts_mode
        self.ts = 0
        self.models: Sequence[Model | list[Model]] = models
        self.loggers: Sequence[LoggerModel] = loggers
        self.ts_func = {
            self.TS_MODE.CURRENT: self.get_current_ts,
            self.TS_MODE.IDX: self.get_idx_ts
        }[ts_mode]
        self._executor = ThreadPoolExecutor(max_workers=2)

    def run(self, img: cv2.typing.MatLike, results: list[Result] = []):
        for model in self.models:
            if isinstance(model, list):
                self._run_parallel(model, img, results)
            else:
                model(img, results)
        return results

    def _run_parallel(self, models: list[Model], img: cv2.typing.MatLike, results: list[Result]):
        if not models:
            return
        futures = [self._executor.submit(model, img, results) for model in models]
        for f in futures:
            f.result()

    def get_logs(self, img: cv2.typing.MatLike, results: list[Result]):
        ts = self.ts_func()

        logs: 'list[Log]' = []
        for logger in self.loggers:
            logs.extend(logger(img, results, ts))
        return logs

    def get_current_ts(self):
        return round(time.time())

    def get_idx_ts(self):
        ts = self.ts
        self.ts += 1
        return ts
