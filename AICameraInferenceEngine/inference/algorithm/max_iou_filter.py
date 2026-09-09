import numpy as np
import cv2
from ..model import Model, Result
from ..utils import img_utils


class MaxIouFilter(Model):
    def __init__(self, region: list[float]) -> None:
        '''
        region: [x1, y1, x2, y2]\n
        '''
        self.region = region

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]'):
        if not len(results):
            return results

        results_with_iou = sorted([
            (result, img_utils.iou_of(np.array([self.region]), np.array([result.bbox]))[0])
            for result in results
        ], key=lambda x: x[1], reverse=True)
        if results_with_iou[0][1] > 0:
            return [results_with_iou[0][0]]
        return []
