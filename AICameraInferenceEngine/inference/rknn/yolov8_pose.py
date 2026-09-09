from ..model import Result
from ..utils import img_utils
from .model import RKNNModel
import cv2
import numpy as np


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint


class YoloV8Pose(RKNNModel):
    def __init__(
        self,
        weight_path: str,
        w: int = 640,
        h: int = 640,
        score_threshold: float = .5,
        kp_threshold: float = .5,
        nms_thresh: float = .4
    ) -> None:
        super().__init__(weight_path)
        self.w = w
        self.h = h
        self.nms_thresh = nms_thresh
        self.score_threshold = score_threshold
        self.kp_threshold = kp_threshold
        self.not_ignore_kp_labels = ["nose", "eye_left", "eye_right", "ear_left", "ear_right",]

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]') -> 'list[Result]':
        img_h, img_w = img.shape[:2]
        _, _, pad_left, _, pad_top, _ = img_utils.pad_zero_info(img_w, img_h, 1)

        side = float(max(img_w, img_h))
        scale_x = side / self.w
        scale_y = side / self.h

        # preprocess
        input_img = img_utils.pad_zero(img, 1, 56)
        input_img = img_utils.resize(input_img, (self.w, self.h))
        input_img = img_utils.switch_br(input_img)
        input_img = img_utils.to_batch([input_img])

        # inference
        outputs = self.session.inference([input_img])

        # postprocess
        processed = []
        keypoints = outputs[3]
        for x in outputs[:3]:
            index, stride = 0, 0
            if x.shape[2] == 20:
                stride = 32
                index = 20 * 4 * 20 * 4 + 20 * 2 * 20 * 2
            if x.shape[2] == 40:
                stride = 16
                index = 20 * 4 * 20 * 4
            if x.shape[2] == 80:
                stride = 8
                index = 0
            feature = x.reshape(1, 65, -1)
            output = self.process(feature, keypoints, index, x.shape[3], x.shape[2], stride)
            processed += output
        predbox = self.NMS(processed)[:self.MAX_BATCH_SIZE]

        for i in range(len(predbox)):
            classId = predbox[i].classId
            score = predbox[i].score
            bbox = np.clip([
                (predbox[i].xmin * scale_x - pad_left) / img_w,
                (predbox[i].ymin * scale_y - pad_top) / img_h,
                (predbox[i].xmax * scale_x - pad_left) / img_w,
                (predbox[i].ymax * scale_y - pad_top) / img_h,
            ], 0, 1).tolist()
            if (bbox[0] >= bbox[2]) or (bbox[1] >= bbox[3]):
                continue
            center = (bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.
            keypoints = predbox[i].keypoint.reshape(-1, 3)  # keypoint [x, y, conf]
            keypoints[..., 0] = (keypoints[..., 0] * scale_x - pad_left) / img_w
            keypoints[..., 1] = (keypoints[..., 1] * scale_y - pad_top) / img_h

            kp_dict = dict()
            for kp_label, kp in zip(self.coco_kp_labels, keypoints):
                if kp[2] >= self.kp_threshold or kp_label in self.not_ignore_kp_labels:
                    kp_dict[kp_label] = kp

            results.append(Result().set(
                creator=self.__class__.__name__,
                bbox=bbox,
                center=center,
                score=score,
                is_coco_kps=True,
                **kp_dict,
            ))
        return results

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def IOU(self, xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
        xmin = max(xmin1, xmin2)
        ymin = max(ymin1, ymin2)
        xmax = min(xmax1, xmax2)
        ymax = min(ymax1, ymax2)

        innerWidth = xmax - xmin
        innerHeight = ymax - ymin

        innerWidth = innerWidth if innerWidth > 0 else 0
        innerHeight = innerHeight if innerHeight > 0 else 0

        innerArea = innerWidth * innerHeight

        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

        total = area1 + area2 - innerArea

        return innerArea / total

    def process(self, out, keypoints, index, model_w, model_h, stride, scale_w=1, scale_h=1):
        """
        Vectorized processing:
        - out: shape (1, 65, N)  where N == model_w*model_h
        - keypoints: original keypoint array, index offset applied later
        """
        # remove leading batch dim
        out = out[0]  # shape (65, N)

        # 1) split xywh (first 64) and conf (64:)
        N = out.shape[1]
        # xywh raw: (64, N) -> reshape to (4, 16, N)
        xywh_raw = out[:64, :].reshape(4, 16, N)  # (4, 16, N)
        conf_all = self.sigmoid(out[64:, :])      # shape (1, N) or (C, N) depending on model
        # Ensure conf_all is (C, N)
        if conf_all.ndim == 1:
            conf_all = conf_all[np.newaxis, :]
        # number of classes
        C = conf_all.shape[0]

        # 2) find candidate positions where any class > threshold
        mask = conf_all > self.score_threshold     # (C, N)
        if not np.any(mask):
            return []

        cls_idxs, pos_idxs = np.nonzero(mask)      # arrays of length K
        K = pos_idxs.size

        # 3) compute grid coordinates (w_idx, h_idx)
        w_idx = pos_idxs % model_w                # (K,)
        h_idx = pos_idxs // model_w               # (K,)

        # 4) compute expected xywh via softmax-with-index in vectorized manner
        # xywh_raw[:, :, pos_idxs] -> shape (4, 16, K)
        xy_selected = xywh_raw[:, :, pos_idxs]    # (4, 16, K)
        # apply softmax along the 16-dim axis (axis=1)
        probs = self.softmax(xy_selected, axis=1)  # (4, 16, K)
        # weighted sum with [0..15]
        idx_vector = np.arange(16).reshape(16, 1)  # (16,1)
        # broadcast multiply and sum over axis=1 -> result (4, 1, K) -> squeeze -> (4, K)
        xy_sum = np.sum(probs * idx_vector[np.newaxis, :, :], axis=1)  # (4, K)

        # 5) reconstruct original box coordinates (vectorized)
        # follow the original math:
        # xy_temp[0] = (w + 0.5) - xy[0]
        # xy_temp[1] = (h + 0.5) - xy[1]
        # xy_temp[2] = (w + 0.5) + xy[2]
        # xy_temp[3] = (h + 0.5) + xy[3]
        wf = w_idx.astype(np.float32) + 0.5  # (K,)
        hf = h_idx.astype(np.float32) + 0.5

        xy_temp0 = wf - xy_sum[0, :]
        xy_temp1 = hf - xy_sum[1, :]
        xy_temp2 = wf + xy_sum[2, :]
        xy_temp3 = hf + xy_sum[3, :]

        center_x = (xy_temp0 + xy_temp2) * 0.5
        center_y = (xy_temp1 + xy_temp3) * 0.5
        width = (xy_temp2 - xy_temp0)
        height = (xy_temp3 - xy_temp1)

        # multiply by stride
        center_x *= stride
        center_y *= stride
        width *= stride
        height *= stride

        # convert to xmin,ymin,xmax,ymax scaled
        xmin = (center_x - (width * 0.5)) * scale_w
        ymin = (center_y - (height * 0.5)) * scale_h
        xmax = (center_x + (width * 0.5)) * scale_w
        ymax = (center_y + (height * 0.5)) * scale_h

        # 6) gather confidences for each detection
        scores = conf_all[cls_idxs, pos_idxs]   # (K,)

        # 7) gather keypoints for each detection (vectorized)
        # original code: keypoint = keypoints[..., (h * model_w) + w + index]
        kp_pos = pos_idxs + index  # (K,)
        # keypoints shape in the original code is probably (num_kp, N_total) or (..., total)
        # assume keypoints is an array where last axis matches original grid count
        kp_selected = keypoints[..., kp_pos]  # shape: (..., K)
        # ensure floor as original code did: keypoint[..., 0:2] = keypoint[..., 0:2] // 1
        kp_selected[..., :2, :] = np.floor(kp_selected[..., :2, :])

        # 8) build DetectBox list (still K Python objects, but K << H*W)
        out_list = []
        for i in range(K):
            c = int(cls_idxs[i])
            s = float(scores[i])
            box = DetectBox(
                c,
                s,
                float(xmin[i]),
                float(ymin[i]),
                float(xmax[i]),
                float(ymax[i]),
                kp_selected[..., i].copy()
            )
            out_list.append(box)
        return out_list

    def NMS(self, detectResult):
        """
        Faster NMS using numpy vectorized operations.
        Input: list of DetectBox
        Output: filtered list
        """
        if not detectResult:
            return []

        # convert to arrays in a single pass instead of six list comprehensions
        data = np.array(
            [(b.xmin, b.ymin, b.xmax, b.ymax, b.score, b.classId) for b in detectResult],
            dtype=np.float32,
        )
        xs1 = data[:, 0]
        ys1 = data[:, 1]
        xs2 = data[:, 2]
        ys2 = data[:, 3]
        scores = data[:, 4]
        classes = data[:, 5].astype(np.int32)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            # only compare with same-class boxes
            same_class_mask = classes[order] == classes[i]
            idxs = order[1:][same_class_mask[1:]]  # exclude first entry which is i itself

            if idxs.size == 0:
                order = order[1:]
                continue

            # compute IoU of box i with these idxs
            xx1 = np.maximum(xs1[i], xs1[idxs])
            yy1 = np.maximum(ys1[i], ys1[idxs])
            xx2 = np.minimum(xs2[i], xs2[idxs])
            yy2 = np.minimum(ys2[i], ys2[idxs])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (xs2[i] - xs1[i]) * (ys2[i] - ys1[i])
            areas = (xs2[idxs] - xs1[idxs]) * (ys2[idxs] - ys1[idxs])
            union = area_i + areas - inter
            ious = inter / (union + 1e-6)

            # keep those with IoU <= thresh
            keep_mask = ious <= self.nms_thresh
            # map back to order positions; build new order
            # order = [order[1:][positions where same_class_mask==False] + order[1:][positions where same_class_mask==True][keep_mask]]
            # simpler: remove idxs where ious > thresh
            suppressed = idxs[~keep_mask]
            # create a set for quick removal
            if suppressed.size > 0:
                suppressed_set = set(suppressed.tolist())
                order = np.array([o for o in order if o not in suppressed_set], dtype=np.int32)
                # remove the picked i as well
                order = order[order != i]
            else:
                # just remove i
                order = order[order != i]

        # return DetectBox list in original order of kept indices sorted by score desc
        return [detectResult[i] for i in keep]
