import cv2


class Log:
    def __init__(self) -> None:
        self.source: str | None = None
        self.id: str | None = None
        self.region: str | None = None
        self.age: float | None = None
        self.gender: float | None = None
        self.pitch: float | None = None
        self.roll: float | None = None
        self.yaw: float | None = None
        self.enter_date: int | None = None
        self.leave_date: int | None = None
        self.start_date: int | None = None
        self.end_date: int | None = None
        self.start_ts: int | None = None
        self.end_ts: int | None = None
        self.id_count: int | None = None
        self.group_code: str | None = None
        self.group_dest: str | None = None
        self.snapshot: cv2.typing.MatLike | None = None
        self.snapshot_mosaic: cv2.typing.MatLike | None = None
        self.is_legacy: bool = False

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        return self

    def to_dict(self, ignore_img: bool = False):
        result = dict()
        for attr, val in self.__dict__.items():
            if val is None:
                continue
            if ignore_img and attr in ["snapshot", "snapshot_mosaic"]:
                continue
            result[attr] = val
        return result
