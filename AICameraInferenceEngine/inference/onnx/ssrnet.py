from ..model import Model, Result
from ..utils import img_utils
from .model import ONNXModel
import cv2
import numpy as np


class SSRNet(ONNXModel):
    def __init__(self, weight_path: str, tag: Model.SSRNET_TARGET, margin: float = 0) -> None:
        super().__init__(weight_path)
        self.tag = tag
        self.margin = margin
        self.input_size = (64, 64)

    def predict_single(self, img: cv2.typing.MatLike):
        face = img_utils.resize(img, self.input_size)
        faces = img_utils.to_batch([face])
        return np.array(self.session.run(None, {'input': faces})).reshape(-1)[0]

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]') -> 'list[Result]':
        if not len(results):
            return results

        # preprocess
        faces = list()
        for result in results:
            face = img_utils.crop(img, result.bbox, self.margin)
            face = img_utils.resize(face, self.input_size)
            faces.append(face)
        faces = img_utils.to_batch(faces)

        # inference
        predict = np.array(self.session.run(None, {'input': faces})).reshape(-1).tolist()

        # postprocess
        for result, output in zip(results, predict):
            result.set(
                age=output if self.tag == Model.SSRNET_TARGET.AGE else None,
                gender=output if self.tag == Model.SSRNET_TARGET.GENDER else None,
            )

        return results
