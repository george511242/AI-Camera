import cv2
import numpy as np

from ..model import Result
from ..utils import img_utils
from .model import RKNNModel


class MobileNetV3Age(RKNNModel):
    """Age v4 runtime adapter for the 112x112 RGB float32 contract."""

    def __init__(self, weight_path: str, margin: float = 0) -> None:
        super().__init__(weight_path)
        self.margin = margin
        self.input_size = (112, 112)

    @staticmethod
    def prepare_input(face: cv2.typing.MatLike) -> np.ndarray:
        face = img_utils.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        return img_utils.to_batch([face])

    def __call__(self, img: cv2.typing.MatLike, results: list[Result]) -> list[Result]:
        for result in results:
            face = img_utils.crop(img, result.bbox, self.margin)
            face = self.prepare_input(face)
            prediction = np.reshape(self.session.inference([face]), -1)[0]
            result.set(age=prediction)
        return results
