from ..model import Result
from ..utils import img_utils
from .model import ONNXModel
import cv2
import numpy as np


class YuNet(ONNXModel):
    def __init__(self, weight_path: str, w: int = 640, h: int = 640, score_threshold: float = .75) -> None:
        super().__init__(weight_path)
        self.w = w
        self.h = h
        self.score_threshold = score_threshold
        self.nms_threshold = .3
        self.strides = [8, 16, 32]
        self.strides_anchor_centers = [np.stack(np.mgrid[:(h // stride), :(w // stride)][::-1], axis=-1).reshape(-1, 2).astype(np.float32) * stride for stride in self.strides]
        self.NK = 5

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]') -> 'list[Result]':
        scores, bboxes, kpss = list(), list(), list()

        img_h, img_w = img.shape[:2]
        _, _, pad_left, _, pad_top, _ = img_utils.pad_zero_info(img_w, img_h, 1)

        side = float(max(img_w, img_h))
        scale_x = side / self.w
        scale_y = side / self.h

        # preprocess
        input_img = img_utils.pad_zero(img, 1)
        input_img = img_utils.resize(input_img, (self.w, self.h))
        input_img = img_utils.switch_br(input_img)
        input_img = img_utils.channel_first(input_img)
        input_img = img_utils.to_batch([input_img])

        # inference
        output = self.session.run(None, {'input': input_img})

        # postprocess
        for (i, stride), anchor_centers in zip(enumerate(self.strides), self.strides_anchor_centers):
            cls_pred = np.clip(output[i], 0, 1)[0]
            obj_pred = np.clip(output[i + len(self.strides)], 0, 1)[0]
            reg_pred = output[i + len(self.strides) * 2][0]
            kps_pred = output[i + len(self.strides) * 3][0]

            bbox_cxy = reg_pred[:, :2] * stride + anchor_centers[:]
            bbox_wh = np.exp(reg_pred[:, 2:]) * stride
            tl_x = (bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.)
            tl_y = (bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.)
            br_x = (bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.)
            br_y = (bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.)

            bboxes.append(np.stack([tl_x, tl_y, br_x, br_y], -1))

            per_kps = np.concatenate([((kps_pred[:, [2 * i, 2 * i + 1]] * stride) + anchor_centers) for i in range(self.NK)], axis=-1)
            kpss.append(per_kps)
            scores.append(np.sqrt(cls_pred * obj_pred))

        scores = np.concatenate(scores, axis=0).reshape(-1)
        bboxes = np.concatenate(bboxes, axis=0)
        kpss = np.concatenate(kpss, axis=0)

        score_mask = (scores >= self.score_threshold)
        scores = scores[score_mask]
        bboxes = bboxes[score_mask]
        kpss = kpss[score_mask]

        keep = img_utils.nms(bboxes, scores, 0, self.nms_threshold)[:self.MAX_BATCH_SIZE]
        scores = scores[keep].tolist()
        kpss = kpss[keep, :].tolist()
        bboxes = bboxes[keep, :].tolist()

        for score, raw_bbox, kp in zip(scores, bboxes, kpss):
            bbox = np.clip([
                (raw_bbox[0] * scale_x - pad_left) / img_w,
                (raw_bbox[1] * scale_y - pad_top) / img_h,
                (raw_bbox[2] * scale_x - pad_left) / img_w,
                (raw_bbox[3] * scale_y - pad_top) / img_h,
            ], 0, 1).tolist()
            if (bbox[0] >= bbox[2]) or (bbox[1] >= bbox[3]):
                continue
            center = (bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.
            eye_right = (
                (kp[0] * scale_x - pad_left) / img_w,
                (kp[1] * scale_y - pad_top) / img_h,
            )
            eye_left = (
                (kp[2] * scale_x - pad_left) / img_w,
                (kp[3] * scale_y - pad_top) / img_h,
            )
            nose = (
                (kp[4] * scale_x - pad_left) / img_w,
                (kp[5] * scale_y - pad_top) / img_h,
            )
            mouth_right = (
                (kp[6] * scale_x - pad_left) / img_w,
                (kp[7] * scale_y - pad_top) / img_h,
            )
            mouth_left = (
                (kp[8] * scale_x - pad_left) / img_w,
                (kp[9] * scale_y - pad_top) / img_h,
            )
            results.append(Result().set(
                creator=self.__class__.__name__,
                raw_bbox=raw_bbox,
                bbox=bbox,
                score=score,
                center=center,
                eye_right=eye_right,
                eye_left=eye_left,
                nose=nose,
                mouth_right=mouth_right,
                mouth_left=mouth_left
            ))
        return results
