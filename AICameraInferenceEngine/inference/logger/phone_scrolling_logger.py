from ..model import Result, Log
from .model import LoggerModel
from typing import Literal
import math
import cv2
import numpy as np
from collections import deque
import subprocess
np.seterr(all='ignore')


class PhoneScrollingLogger(LoggerModel):
    def __init__(
        self,
        scrolling_queue_size: int = 10,
        min_scrolling_count: int = 5,
        take_snapshot_scrolling_count: int = 1,
        check_eye_lower_than_ear: bool = True,
        head_down_ratio: float = 0.4,
        wrist_to_shoulder_upper: float = -0.1,
        wrist_to_shoulder_lower: float = 0.4,
        elbow_angle: float = 80,
        check_is_front_side: bool = False,
        bbox_move_threshold: float = 0.4,
        bbox_size_change_threshold: float = 0.015,
        snapshot_margin: float = 0.2,
        ignore_count_after_send: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.scrolling_queue_size = scrolling_queue_size
        self.min_scrolling_count = min_scrolling_count
        self.take_snapshot_scrolling_count = take_snapshot_scrolling_count
        self.check_eye_lower_than_ear = check_eye_lower_than_ear
        self.head_down_ratio = head_down_ratio
        self.wrist_to_shoulder_upper = wrist_to_shoulder_upper
        self.wrist_to_shoulder_lower = wrist_to_shoulder_lower
        self.elbow_angle = elbow_angle
        self.check_is_front_side = check_is_front_side
        self.bbox_move_threshold = bbox_move_threshold
        self.bbox_size_change_threshold = bbox_size_change_threshold
        self.count_after_send: dict[str, int] = {}
        self.snapshot_dict: dict[str, dict[Literal["raw", "mosaic"], cv2.typing.MatLike]] = {}
        self.scrolling_info: dict[str, deque] = {}
        self.scrolling_queue_default = deque(maxlen=scrolling_queue_size)
        self.scrolling_queue_default.extend([False] * scrolling_queue_size)
        self.snapshot_margin = snapshot_margin
        self.ignore_count_after_send = ignore_count_after_send

    def __call__(self, img: cv2.typing.MatLike | None, results: 'list[Result]', ts: int):
        logs: list[Log] = []

        for id in list(self.count_after_send.keys()):
            self.count_after_send[id] += 1
            if self.count_after_send[id] >= self.ignore_count_after_send:
                self.count_after_send.pop(id)

        for result in results:
            if result.is_lost:
                self.scrolling_info.pop(result.id)
                self.count_after_send.pop(result.id, None)
                self.snapshot_dict.pop(result.id, None)
                continue
            is_scrolling = self.is_scrolling(result) and not result.disappear_count
            scrolling_queue = self.scrolling_info.get(result.id, self.scrolling_queue_default.copy())
            scrolling_queue.append(ts if is_scrolling else False)
            is_scrolling_count = self.scrolling_queue_size - scrolling_queue.count(False)
            if img is not None and is_scrolling_count == self.take_snapshot_scrolling_count and result.id not in self.snapshot_dict:
                from ..utils import img_utils
                snapshot = img.copy()
                snapshot_mosaic = img.copy()
                if result.nose is not None:
                    # 以鼻子為中心加入馬賽克
                    width = (result.bbox[2] - result.bbox[0]) / 3 / 2
                    nose_x, nose_y, _ = result.nose
                    snapshot_mosaic = img_utils.fill_mosaic(snapshot_mosaic, [nose_x - width, nose_y - width, nose_x + width, nose_y + width])
                snapshot = img_utils.crop(snapshot, result.bbox, self.snapshot_margin)
                snapshot_mosaic = img_utils.crop(snapshot_mosaic, result.bbox, self.snapshot_margin)
                self.snapshot_dict[result.id] = {
                    "raw": snapshot,
                    "mosaic": snapshot_mosaic,
                }
            if is_scrolling_count >= self.min_scrolling_count and result.id not in self.count_after_send:
                scrolling_start_ts = next((x for x in scrolling_queue if x), None)
                logs.append(Log().set(
                    source=self.__class__.__name__,
                    id=result.id,
                    start_date=scrolling_start_ts,
                    end_date=ts,
                    snapshot=self.snapshot_dict[result.id].get("raw", None),
                    snapshot_mosaic=self.snapshot_dict[result.id].get("mosaic", None),
                ))
                scrolling_queue.extend([False] * self.scrolling_queue_size)
                self.count_after_send[result.id] = 0
            self.scrolling_info[result.id] = scrolling_queue
        if len(logs):
            try:
                # Check if audio card 2 exists before playing
                result = subprocess.run(["aplay", "-l"], capture_output=True, text=True)
                if "card 2:" in result.stdout:
                    subprocess.Popen(["aplay", "-D", "plughw:2,0", "/app/audio/no-phone.wav"])
                else:
                    print("Audio card 2 not found, skipping sound playback")
            except Exception as e:
                print("Failed to play sound:", e)
        return logs

    def is_scrolling(self, result: 'Result') -> bool:
        if result.bbox is None:
            return False

        # check head down
        is_head_down = self._check_is_head_down(result)

        # check if wrist in bbox
        is_wrist_valid = self._check_wrist_in_bbox(result)
        if self.check_is_front_side:
            is_wrist_valid &= self._check_wrist_x_valid(result)

        # check elbow curve
        left_curve = self._check_elbow_curve(result.elbow_left, result.shoulder_left, result.hip_left, result.wrist_left, self.elbow_angle)
        right_curve = self._check_elbow_curve(result.elbow_right, result.shoulder_right, result.hip_right, result.wrist_right, self.elbow_angle)
        is_elbow_curve = left_curve or right_curve

        # check is walking
        # is_walking = self._check_is_walking(result)
        is_walking = self._check_is_walking_by_body(result)

        return is_head_down and is_wrist_valid and is_elbow_curve and is_walking

    def _check_is_head_down(self, result: 'Result') -> bool:
        left_head_down_score = -float('inf')
        right_head_down_score = -float('inf')
        left_eye_lower_than_ear = self.check_eye_lower_than_ear and result.eye_left is not None and result.ear_left is not None and result.eye_left[1] > result.ear_left[1]
        right_eye_lower_than_ear = self.check_eye_lower_than_ear and result.eye_right is not None and result.ear_right is not None and result.eye_right[1] > result.ear_right[1]
        if result.eye_left is not None and result.ear_left is not None and result.shoulder_left is not None:
            head_hight = result.shoulder_left[1] - result.bbox[1]
            left_head_down_score = (result.eye_left[1] - result.bbox[1]) / head_hight
        if result.eye_right is not None and result.ear_right is not None and result.shoulder_right is not None:
            head_hight = result.shoulder_right[1] - result.bbox[1]
            right_head_down_score = (result.eye_right[1] - result.bbox[1]) / head_hight
        return left_head_down_score > self.head_down_ratio or right_head_down_score > self.head_down_ratio or left_eye_lower_than_ear or right_eye_lower_than_ear

    def _check_wrist_in_bbox(self, result: 'Result') -> bool:
        is_wrist_valid = True
        if result.wrist_left is not None:
            is_wrist_valid &= (result.bbox[1] < result.wrist_left[1] < result.bbox[3] and result.bbox[0] < result.wrist_left[0] < result.bbox[2])
        if result.wrist_right is not None:
            is_wrist_valid &= (result.bbox[1] < result.wrist_right[1] < result.bbox[3] and result.bbox[0] < result.wrist_right[0] < result.bbox[2])
        return is_wrist_valid

    def _check_wrist_x_valid(self, result: 'Result') -> bool:
        is_wrist_valid = True
        if result.wrist_left is not None and result.wrist_right is not None:
            is_wrist_valid = result.wrist_left[0] >= result.wrist_right[0]
        return is_wrist_valid

    def _check_is_walking(self, result: 'Result') -> bool:
        if len(result.history) < 2:
            return False

        bbox_list = list(result.get_attr_from_history("bbox"))[-self.scrolling_queue_size:]
        move_dist_list = list()
        size_change_list = list()
        for i in range(1, len(bbox_list)):
            prev_bbox = bbox_list[i - 1]
            curr_bbox = bbox_list[i]

            prev_x1, prev_y1, prev_x2, prev_y2 = prev_bbox
            curr_x1, curr_y1, curr_x2, curr_y2 = curr_bbox

            prev_w = prev_x2 - prev_x1
            prev_h = prev_y2 - prev_y1
            curr_w = curr_x2 - curr_x1
            curr_h = curr_y2 - curr_y1

            # 計算邊界框中心點的移動距離
            center_prev_x = (prev_x1 + prev_x2) / 2
            center_prev_y = (prev_y1 + prev_y2) / 2
            center_curr_x = (curr_x1 + curr_x2) / 2
            center_curr_y = (curr_y1 + curr_y2) / 2

            move_dist = ((center_curr_x - center_prev_x) ** 2 + (center_curr_y - center_prev_y) ** 2) ** 0.5 / prev_w
            size_change = abs((curr_w * curr_h) - (prev_w * prev_h)) / (prev_w * prev_h)
            move_dist_list.append(move_dist)
            size_change_list.append(size_change)

        # 判斷移動距離和邊界框大小變化是否超過閾值
        return np.mean(move_dist_list) > self.bbox_move_threshold or np.mean(size_change_list) > self.bbox_size_change_threshold

    def _check_is_walking_by_body(self, result: 'Result') -> bool:
        body_kps_list = [
            [
                h.shoulder_left,
                h.shoulder_right,
                h.hip_right,
                h.hip_left,
            ] for h in result.history
        ][-self.scrolling_queue_size:]
        move_dist_list = list()
        size_change_list = list()
        for (prev_shoulder_l, prev_shoulder_r, prev_hip_r, prev_hip_l), (curr_shoulder_l, curr_shoulder_r, curr_hip_r, curr_hip_l) in zip(
            body_kps_list,
            body_kps_list[1:]
        ):
            if any([k is None for k in [prev_shoulder_l, prev_shoulder_r, prev_hip_r, prev_hip_l, curr_shoulder_l, curr_shoulder_r, curr_hip_r, curr_hip_l]]):
                move_dist_list.append(-float("inf"))
                size_change_list.append(-float("inf"))
                continue

            prev_x_list = [prev_shoulder_l[0], prev_shoulder_r[0], prev_hip_l[0], prev_hip_r[0]]
            prev_y_list = [prev_shoulder_l[1], prev_shoulder_r[1], prev_hip_l[1], prev_hip_r[1]]
            curr_x_list = [curr_shoulder_l[0], curr_shoulder_r[0], curr_hip_l[0], curr_hip_r[0]]
            curr_y_list = [curr_shoulder_l[1], curr_shoulder_r[1], curr_hip_l[1], curr_hip_r[1]]
            prev_center_x = (max(prev_x_list) + min(prev_x_list)) / 2
            prev_center_y = (max(prev_y_list) + min(prev_y_list)) / 2
            curr_center_x = (max(curr_x_list) + min(curr_x_list)) / 2
            curr_center_y = (max(curr_y_list) + min(curr_y_list)) / 2

            prev_w = max(prev_x_list) - min(prev_x_list)
            prev_area = self._shoelace_formula([prev_shoulder_l, prev_shoulder_r, prev_hip_r, prev_hip_l])
            curr_area = self._shoelace_formula([curr_shoulder_l, curr_shoulder_r, curr_hip_r, curr_hip_l])

            move_dist = ((curr_center_x - prev_center_x) ** 2 + (curr_center_y - prev_center_y) ** 2) ** 0.5 / prev_w
            size_change = abs(curr_area - prev_area) / (prev_area)
            move_dist_list.append(move_dist)
            size_change_list.append(size_change)

        # 判斷移動距離和邊界框大小變化是否超過閾值
        if not move_dist_list or not size_change_list:
            return False
        return np.mean(move_dist_list) > self.bbox_move_threshold or np.mean(size_change_list) > self.bbox_size_change_threshold

    def _get_angle_new(
        self,
        vertex: list[float],
        pt1: list[float],
        pt2: list[float],
    ) -> float:
        """計算兩個向量之間的角度"""
        a = np.array(pt1)
        b = np.array(vertex)
        c = np.array(pt2)

        ba = np.linalg.norm(a - b)
        bc = np.linalg.norm(c - b)
        if ba == 0 or bc == 0:
            return float("inf")

        cosine_angle = np.dot(ba, bc) / (ba * bc)
        # 處理數值誤差，確保 cosine_angle 在有效範圍內
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cosine_angle))

    def _check_elbow_curve(
        self,
        elbow: list[float] | None,
        shoulder: list[float] | None,
        hip: list[float] | None,
        wrist: list[float] | None,
        angle: int | float | None = None,
    ):
        # 檢查必要的關鍵點是否存在
        if shoulder is None or hip is None or wrist is None:
            return False

        body_height = hip[1] - shoulder[1]

        # 以身長為分母，計算手腕到肩膀的距離比例
        wrist_to_shoulder = (wrist[1] - shoulder[1]) / body_height
        if not self.wrist_to_shoulder_upper <= wrist_to_shoulder <= self.wrist_to_shoulder_lower:
            return False

        # 如果有手肘關鍵點，檢查手腕是否高於手肘
        if elbow is not None and wrist[1] >= elbow[1]:
            return False

        # 如果提供了角度參數，計算並檢查角度
        if angle is not None:
            current_angle = self._get_angle(
                edge1=shoulder,
                edge2=wrist,
                center=hip
            )
            if current_angle > angle:
                # print('角度例外')
                return False

        return True

    def _get_angle(
        self,
        edge1: list[float],
        edge2: list[float],
        center: list[float]
    ) -> float:
        """計算兩個向量之間的角度"""
        try:
            a = np.array(edge1)
            b = np.array(center)
            c = np.array(edge2)

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            # 處理數值誤差，確保 cosine_angle 在有效範圍內
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            return int(np.degrees(np.arccos(cosine_angle)))
        except Exception as e:
            return float("inf")

    def _shoelace_formula(self, points: list[list[float]]) -> float:
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        area = abs(area) / 2.0
        return area
