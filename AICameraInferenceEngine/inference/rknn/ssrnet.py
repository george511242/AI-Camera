from ..model import Model, Result
from ..utils import img_utils
from .model import RKNNModel
import cv2
import numpy as np


class SSRNet(RKNNModel):
    def __init__(self, weight_path: str, tag: Model.SSRNET_TARGET, margin: float = 0) -> None:
        super().__init__(weight_path)
        self.tag = tag
        self.margin = margin
        self.input_size = (64, 64)

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]') -> 'list[Result]':
        for result in results:
            # preprocess
            face = img_utils.crop(img, result.bbox, self.margin)
            face = img_utils.resize(face, self.input_size)
            face = img_utils.to_batch([face])

            # inference
            predict = np.reshape(self.session.inference([face]), -1)[0]

            # postprocess
            result.set(
                age=predict if self.tag == Model.SSRNET_TARGET.AGE else None,
                gender=predict if self.tag == Model.SSRNET_TARGET.GENDER else None,
            )

        return results
