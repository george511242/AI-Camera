from ..model import Model, Result
from ..utils import img_utils
import cv2
import numpy as np


class Canny(Model):
    def __init__(self, kernel_size: int = 5, low_threshold: int = 100, high_threshold: int = 200):
        self.kernel_size = kernel_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]') -> 'list[Result]':
        canny = img_utils.to_gray(img)
        canny = img_utils.gaussian_blur(canny, self.kernel_size)
        canny = img_utils.canny_edge(canny, self.low_threshold, self.high_threshold)
        edges_count = np.count_nonzero(canny)
        total_pixels = canny.shape[0] * canny.shape[1]
        canny = img_utils.gray_to_rgb(canny)

        results.append(Result().set(
            canny=canny,
            edges_count=edges_count,
            total_pixels=total_pixels,
        ))
        return results
