from ..model import Result, Log
from .model import LoggerModel
import datetime
import cv2


class IdCountLogger(LoggerModel):
    def __init__(self, send_every_n_sec: int | float = 3600, sync_midnight=False):
        super().__init__()
        self.send_every_n_sec = send_every_n_sec
        self.sync_midnight = sync_midnight
        self.start_ts = None
        self.next_send_ts: int | float = float("inf")
        self.id_set = set()
        self.prev_id_set = set()

    def init_ts(self, ts: int):
        if self.start_ts is None:
            self.start_ts = ts
            if self.sync_midnight:
                now = datetime.datetime.fromtimestamp(ts)
                next_send_ts = datetime.datetime(now.year, now.month, now.day)
                while next_send_ts < now:
                    next_send_ts += datetime.timedelta(seconds=self.send_every_n_sec)
                self.next_send_ts = next_send_ts.timestamp()
            else:
                self.next_send_ts = ts + self.send_every_n_sec

    def need_to_send(self, ts: int):
        return self.start_ts is not None and ts > self.next_send_ts

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]', ts: int):
        logs: list[Log] = []

        self.init_ts(ts)
        self.id_set |= {result.id for result in results if result.id is not None} - self.prev_id_set

        if self.need_to_send(ts):
            logs.append(Log().set(
                source=self.__class__.__name__,
                start_ts=self.start_ts,
                end_ts=ts,
                id_count=len(self.id_set),
            ))
            self.start_ts = ts
            self.next_send_ts += self.send_every_n_sec
            self.prev_id_set = self.id_set.copy()
            self.id_set.clear()
        return logs
