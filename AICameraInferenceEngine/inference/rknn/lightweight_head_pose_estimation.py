from ..model import Result
from ..utils import img_utils
from .model import RKNNModel
import cv2
import numpy as np


class LightweightHeadPoseEstimation(RKNNModel):
    def __init__(self, weight_path, margin=.6) -> None:
        super().__init__(weight_path)
        self.margin = margin
        self.input_size = (224, 224)

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]') -> 'list[Result]':
        for result in results:
            # preprocess
            if result.crop_info is not None:
                face = img_utils.crop(img, result.crop_info, self.margin)
            else:
                face = img_utils.crop(img, result.bbox, self.margin)
            face = img_utils.pad_zero(face, 1)
            face = img_utils.resize(face, self.input_size)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # face = img_utils.z_norm(face, [0., 0., 0.], [255., 255., 255.])
            # face = img_utils.z_norm(face, self.imagenet_mean, self.imagenet_std)
            face = img_utils.to_batch([face]).astype(np.float32)

            # inference
            roll, yaw, pitch = self.session.inference([face], data_type='float32')

            # postprocess
            result.set(
                yaw=yaw[0],
                roll=roll[0],
                pitch=pitch[0],
            )

        return results
