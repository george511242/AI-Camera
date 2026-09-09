from ..model import Model, Result
from ..utils import img_utils
from time import time
from collections import deque
import uuid
import cv2
import numpy as np


class Match:
    def __init__(self, pred_idx: int, det_idx: int, iou: float, dist: float):
        self.pred_idx = pred_idx
        self.det_idx = det_idx
        self.iou = iou
        self.dist = dist


class KalmanTracker:
    _state_dim = 8
    _meas_dim = 4

    A = np.eye(_state_dim, dtype=np.float32)
    for i in range(4):
        A[i, i + 4] = 1.0
    H = np.zeros((_meas_dim, _state_dim), dtype=np.float32)
    H[np.arange(_meas_dim), np.arange(_meas_dim)] = 1
    Q = np.eye(_state_dim, dtype=np.float32) * 1e-2
    R = np.eye(_meas_dim, dtype=np.float32) * 1e-1
    P0 = np.eye(_state_dim, dtype=np.float32)

    def __init__(self, id: str, result: Result, max_seq_size: int | None = None):
        self.id = id
        self.result = result.set(first_hit_ts=time())
        self.history: deque[Result] = deque([], maxlen=max_seq_size)

        self.kf = cv2.KalmanFilter(self._state_dim, self._meas_dim)
        # cv2.KalmanFilter setters perform an internal cv::Mat copy, so the
        # Python-level .copy() was redundant. Assign the shared class matrices
        # directly to avoid five per-tracker allocations.
        self.kf.transitionMatrix = self.A
        self.kf.measurementMatrix = self.H
        self.kf.processNoiseCov = self.Q
        self.kf.measurementNoiseCov = self.R
        self.kf.errorCovPost = self.P0

        cx, cy = result.center
        x1, y1, x2, y2 = result.bbox
        w = x2 - x1
        h = y2 - y1
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        self.kf.predict()
        self.update(result)

    def predict(self):
        pred = self.kf.predict()
        cx, cy, w, h = pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0]
        return [cx, cy, w, h]

    def update(self, result: Result):
        cx, cy = result.center
        x1, y1, x2, y2 = result.bbox
        w = x2 - x1
        h = y2 - y1
        meas = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(meas)

        self.history.append(result)
        old = self.result.clone()
        self.result = result.set(
            id=self.id,
            first_hit_ts=old.first_hit_ts,
            hits=old.hits,
            disappear_count=0,
            history=self.history,
        )


class Sort(Model):
    def __init__(
        self,
        max_disappear: int = 5,
        max_distance: float = 0.15,
        iou_threshold: float = 0.001,
        min_hits: int = 1,
        max_ts: int | float = float('inf'),
        creator_filter: None | str = None,
        id_mode: Model.ID_MODE = Model.ID_MODE.UUID,
    ):
        self.max_age = max_disappear
        self.gate_dist = max_distance
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_ts = max_ts
        self.creator_filter = creator_filter
        self.trackers: list[KalmanTracker] = []
        self.id_mode = id_mode
        self.obj_count = 0
        self.id_func = {
            Model.ID_MODE.UUID: self.get_new_uuid,
            Model.ID_MODE.HEX: self.get_new_hex_id,
        }[id_mode]

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]') -> 'list[Result]':
        results_to_track, results_to_ignore = list(), list()
        if self.creator_filter:
            for r in results:
                if r.creator == self.creator_filter:
                    results_to_track.append(r)
                else:
                    results_to_ignore.append(r)
        else:
            results_to_track = results
        preds = [trk.predict() for trk in self.trackers]

        # match candidates
        matches_all: list[Match] = []
        for i, pred in enumerate(preds):
            [x, y, w, h] = pred
            x1 = x - w / 2.0
            y1 = y - h / 2.0
            x2 = x + w / 2.0
            y2 = y + h / 2.0
            for j, result in enumerate(results_to_track):
                dist = img_utils.get_distance(result.center, [x, y])
                if dist > self.gate_dist:
                    continue
                iou = img_utils.iou_of(np.array([[x1, y1, x2, y2]]), np.array([result.bbox]))[0]
                matches_all.append(Match(i, j, iou, dist))

        # sort by iou desc, dist asc
        matches_all.sort(key=lambda m: (-m.iou, m.dist))
        used_pred = set()
        used_det = set()
        matches: dict[int, int] = {}
        for m in matches_all:
            if m.pred_idx not in used_pred and m.det_idx not in used_det:
                used_pred.add(m.pred_idx)
                used_det.add(m.det_idx)
                matches[m.pred_idx] = m.det_idx

        # update matched
        matched_ids = set()
        for p_idx, d_idx in matches.items():
            self.trackers[p_idx].update(results_to_track[d_idx])
            matched_ids.add(self.trackers[p_idx].id)

        # create new for unmatched dets
        for j in range(len(results_to_track)):
            if j not in used_det:
                kt = KalmanTracker(self.id_func(), results_to_track[j], self.MAX_SEQ_SIZE)
                self.trackers.append(kt)
                matched_ids.add(kt.id)

        # prepare output and prune
        alive: list[KalmanTracker] = []
        output: list[Result] = []
        for trk in self.trackers:
            trk.result.set(
                hits=trk.result.hits + 1,
                disappear_count=trk.result.disappear_count + (trk.id not in matched_ids),
            ).set(
                is_small_hits=trk.result.hits < self.min_hits,
                is_lost=trk.result.disappear_count > self.max_age,
            )
            output.append(trk.result)
            if not trk.result.is_lost and not time() - trk.result.first_hit_ts > self.max_ts:
                alive.append(trk)
        self.trackers = alive
        return output + results_to_ignore

    def get_new_uuid(self):
        return uuid.uuid4().hex

    def get_new_hex_id(self):
        hex_id = f'{self.obj_count:#06x}'
        self.obj_count += 1
        return hex_id
