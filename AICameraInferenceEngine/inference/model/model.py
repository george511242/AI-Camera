from pathlib import Path
from .result import Result
from enum import Enum
from collections import deque
import cv2


def get_lines_from_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return [l.rstrip() for l in f]


class Model:
    MAX_BATCH_SIZE = 10
    MAX_SEQ_SIZE = 60 * 10  # seconds * ideal FPS
    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    imagenet_classes = get_lines_from_file(Path(__file__).parent / "imagenet_1000.txt")
    coco_obj_classes = get_lines_from_file(Path(__file__).parent / "coco_80.txt")
    coco_kp_labels = [
        "nose", "eye_left", "eye_right", "ear_left", "ear_right",
        "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
        "wrist_left", "wrist_right", "hip_left", "hip_right",
        "knee_left", "knee_right", "ankle_left", "ankle_right"
    ]

    class SSRNET_TARGET(Enum):
        AGE = 1
        GENDER = 2

    class ID_MODE(Enum):
        UUID = 1
        HEX = 2

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]') -> 'list[Result]':
        """
        Fit the model to the image and results.
        :param img: The input image.
        :param results: The list of results to be processed.
        :return: The processed results.
        """
        raise NotImplementedError('__call__ is not implemented in the base class Model')

    def is_attributes_fulfilled(self, result: Result) -> bool:
        return True

    def get_mean(self, seq: list[float] | deque[float], binary_threshold: float | None = None):
        if not len(seq):
            return None
        mean = sum(seq) / len(seq)
        if binary_threshold is not None:
            return int(mean >= binary_threshold)
        return mean
