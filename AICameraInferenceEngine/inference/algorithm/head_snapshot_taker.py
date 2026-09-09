import cv2
from ..model import Model, Result
from ..utils import img_utils


class HeadSnapshotTaker(Model):
    def __init__(self) -> None:
        pass

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]'):
        img_h, img_w, _ = img.shape
        margin = 0
        if img_h > img_w:
            margin = [img_h / img_w - 1, 0, img_h / img_w - 1, 0]
        if img_w > img_h:
            margin = [0, img_w / img_h - 1, 0, img_w / img_h - 1]

        for result in results:
            if result.bbox is not None and result.nose is not None:
                width = (result.bbox[2] - result.bbox[0]) / 3 / 2
                result.set(
                    crop_info=img_utils.crop_info([
                        result.nose[0] - width,
                        result.nose[1] - width,
                        result.nose[0] + width,
                        result.nose[1] + width,
                    ], margin)
                )
        return results
