from ..model import Result
from ..utils import img_utils
from .model import ONNXModel
import cv2
import numpy as np


class Yolo11(ONNXModel):
    def __init__(
        self,
        weight_path: str,
        w: int = 640,
        h: int = 640,
        score_threshold: float = .25,
        cls_score_threshold: dict[int, float] = {67: 0.001},
        nms_thresh: float = .45
    ) -> None:
        super().__init__(weight_path)
        self.w = w
        self.h = h
        self.score_threshold = score_threshold
        self.cls_score_threshold = cls_score_threshold
        self.nms_thresh = nms_thresh

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]') -> 'list[Result]':
        img_h, img_w = img.shape[:2]
        _, _, pad_left, _, pad_top, _ = img_utils.pad_zero_info(img_w, img_h, 1)

        side = float(max(img_w, img_h))
        scale_x = side / self.w
        scale_y = side / self.h

        # preprocess
        input_img = img_utils.pad_zero(img, 1, 0)
        input_img = img_utils.resize(input_img, (self.w, self.h))
        input_img = img_utils.z_norm(input_img, [0., 0., 0.], [255., 255., 255.])
        input_img = img_utils.channel_first(input_img)
        input_img = img_utils.to_batch([input_img])

        # inference
        outputs = self.session.run(None, {"images": input_img})

        # postprocess
        boxes, classes, scores = self.post_process(outputs)
        for [x1, y1, x2, y2], cls, score in zip(boxes[:self.MAX_BATCH_SIZE], classes[:self.MAX_BATCH_SIZE], scores[:self.MAX_BATCH_SIZE]):
            bbox = np.clip([
                (x1 * scale_x - pad_left) / img_w,
                (y1 * scale_y - pad_top) / img_h,
                (x2 * scale_x - pad_left) / img_w,
                (y2 * scale_y - pad_top) / img_h,
            ], 0, 1).tolist()
            if (bbox[0] >= bbox[2]) or (bbox[1] >= bbox[3]):
                continue
            center = (bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.
            results.append(Result().set(
                creator=self.__class__.__name__,
                bbox=bbox,
                center=center,
                score=score,
                coco_class_idx=cls,
            ))
        return results

    def dfl(self, position):
        # Distribution Focal Loss (DFL)
        x = np.asarray(position, dtype=np.float32)
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        # reshape: (n, 4, mc, h, w)
        y = x.reshape(n, p_num, mc, h, w)
        # softmax along axis=2
        y_exp = np.exp(y - np.max(y, axis=2, keepdims=True))
        y = y_exp / np.sum(y_exp, axis=2, keepdims=True)
        # acc_matrix: (1, 1, mc, 1, 1)
        acc_matrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
        # 加權平均
        y = np.sum(y * acc_matrix, axis=2)

        return y

    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.h // grid_h, self.w // grid_w]).reshape(1, 2, 1, 1)

        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy

    def post_process(self, input_data):
        boxes, scores, classes_conf = [], [], []
        default_branch = 3
        pair_per_branch = len(input_data) // default_branch
        # Python 忽略 score_sum 输出
        for i in range(default_branch):
            boxes.append(self.box_process(input_data[pair_per_branch * i]))
            classes_conf.append(input_data[pair_per_branch * i + 1])
            scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return [], [], []

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses).tolist()
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        scores = class_max_score * box_confidences

        N = scores.shape[0]
        thresholds = np.full(N, self.score_threshold, dtype=float)
        for cls, val in self.cls_score_threshold.items():
            thresholds[classes == cls] = float(val)

        keep_mask = scores >= thresholds
        if not np.any(keep_mask):
            return np.zeros((0, boxes.shape[1])), np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)

        return boxes[keep_mask], classes[keep_mask], scores[keep_mask]

    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep
