from typing import Any
from time import time
import cv2
from collections import deque


class Result:
    def __init__(self) -> None:
        self.ts = time()
        self.creator: str | None = None
        self.id: str | None = None
        self.score: float | None = None
        self.coco_class_idx: int | None = None
        self.imagenet_class_idx: int | None = None
        self.imagenet_class_score: float | None = None
        self.imagenet_class: str | None = None
        self.imagenet_class_idxes: list[int] | None = None
        self.imagenet_class_scores: list[float] | None = None
        self.imagenet_classes: list[str] | None = None
        self.contour: list[list[float]] | None = None
        self.raw_bbox: list[float] | None = None
        self.bbox: list[float] | None = None
        self.center: list[float] | None = None
        self.eye_left: list[float] | None = None
        self.eye_right: list[float] | None = None
        self.nose: list[float] | None = None
        self.mouth_right: list[float] | None = None
        self.mouth_left: list[float] | None = None
        self.ear_left: list[float] | None = None
        self.ear_right: list[float] | None = None
        self.shoulder_left: list[float] | None = None
        self.shoulder_right: list[float] | None = None
        self.elbow_left: list[float] | None = None
        self.elbow_right: list[float] | None = None
        self.wrist_left: list[float] | None = None
        self.wrist_right: list[float] | None = None
        self.hip_left: list[float] | None = None
        self.hip_right: list[float] | None = None
        self.knee_left: list[float] | None = None
        self.knee_right: list[float] | None = None
        self.ankle_left: list[float] | None = None
        self.ankle_right: list[float] | None = None
        self.is_coco_kps: bool = False
        self.crop_info: list[float] | None = None
        self.crop_infos: list[list[float]] | None = None
        self.age: float | None = None
        self.gender: float | None = None
        self.history: deque[Result] = deque([])
        self.pitch: float | None = None
        self.roll: float | None = None
        self.yaw: float | None = None
        self.first_hit_ts: float | None = None
        self.hits: int = 0
        self.disappear_count: int = 0
        self.is_small_hits: bool = False
        self.is_lost: bool = False
        self.focus_count: int | None = None
        self.canny: cv2.typing.MatLike | None = None
        self.edges_count: int | None = None
        self.total_pixels: int | None = None
        self.debug1 = None
        self.debug2 = None
        self.debug3 = None

    def set(self, **kwargs: Any):
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        return self

    def get_attr_from_history(self, attr: str) -> list[Any]:
        return [getattr(result, attr) for result in self.history if getattr(result, attr) is not None]

    def to_dict(self, ignore_img: bool = False):
        result = dict()
        for attr, val in self.__dict__.items():
            if attr == "history":
                continue
            if ignore_img and attr in ["canny"]:
                continue
            else:
                result[attr] = val
        return result

    def clone(self, ignore_deque: bool = True):
        new = Result()
        for attr, val in self.__dict__.items():
            if isinstance(val, deque):
                if not ignore_deque:
                    setattr(new, attr, val.copy())
            else:
                setattr(new, attr, val)
        return new
