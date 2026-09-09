from ..model import Result, Log
from .model import LoggerModel
import cv2


class RegionLogger(LoggerModel):
    def __init__(self, name: str, region: list[float]):
        super().__init__()
        self.name = name
        self.region = region
        self.enter_info: dict[str, int] = {}

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]', ts: int):
        logs: list[Log] = []
        for result in results:
            assert result.id is not None
            assert result.center is not None
            id = result.id
            in_region_before = result.id in self.enter_info
            in_region_now = True
            in_region_now &= self.region[0] <= result.center[0] <= self.region[2]
            in_region_now &= self.region[1] <= result.center[1] <= self.region[3]
            if in_region_before and (result.is_lost or not in_region_now):
                enter_ts = self.enter_info.pop(id)
                if not result.is_small_hits:
                    logs.append(Log().set(
                        source=self.__class__.__name__,
                        id=id,
                        region=self.name,
                        enter_date=enter_ts,
                        leave_date=ts,
                        age=self.get_mean(result.get_attr_from_history("age")),
                        gender=self.get_mean(result.get_attr_from_history("gender"), 0.5),
                    ))
            elif not in_region_before and in_region_now:
                self.enter_info[id] = ts
        return logs
