import cv2
from ..model import Model, Result
from ..utils import img_utils


class HandsSnapshotTaker(Model):
    def __init__(self) -> None:
        pass

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]'):
        for result in results:
            crop_infos = list()
            if result.wrist_left is not None and result.shoulder_left is not None:
                dist = self._get_dist(result.wrist_left, result.shoulder_left)
                if dist:
                    crop_infos.append(self._get_crop_from_kps(result.wrist_left, dist))
            if result.wrist_right is not None and result.shoulder_right is not None:
                dist = self._get_dist(result.wrist_right, result.shoulder_right)
                if dist:
                    crop_infos.append(self._get_crop_from_kps(result.wrist_right, dist))
            result.set(
                crop_infos=crop_infos,
            )
        return results

    def _get_crop_from_kps(self, kp, dist, margin=-.25):
        to_crop = [
            kp[0] - dist,
            kp[1] - dist,
            kp[0] + dist,
            kp[1] + dist,
        ]
        return img_utils.crop_info(to_crop, margin)

    def _get_dist(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
