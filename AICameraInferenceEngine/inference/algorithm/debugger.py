import cv2
import numpy as np
from ..model import Model, Result


class Debugger(Model):
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]'):
        return results
    
        counter = dict()
        for result in results:
            counter[result.creator] = counter.get(result.creator, 0) + 1
        # print(f"Debugger: {counter}")
        # return results
        return [result for result in results if result.creator == "YoloV8Pose"]
        for result in results:
            debug1_1 = self._get_head_debug(result.bbox, result.shoulder_left, result.eye_left)
            debug1_2 = self._get_head_debug(result.bbox, result.shoulder_right, result.eye_right)
            debug2_1 = self._get_wrist_debug(result.shoulder_left, result.hip_left, result.wrist_left)
            debug2_2 = self._get_wrist_debug(result.shoulder_right, result.hip_right, result.wrist_right)
            debug3_1 = self._get_angle(result.shoulder_left, result.wrist_left, result.hip_left)
            debug3_2 = self._get_angle(result.shoulder_right, result.wrist_right, result.hip_right)
            debug1 = f"{debug1_1:.3f}, {debug1_2:.3f}"
            debug2 = f"{debug2_1:.3f}, {debug2_2:.3f}"
            debug3 = f"{debug3_1:.1f}, {debug3_2:.1f}"
            result.set(
                # debug1=debug1,
                # debug2=debug2,
                # debug3=debug3,
            )

        return results

    def _check_is_walking_by_body(self, result: 'Result') -> bool:
        shoulder_left_list = list(result.get_attr_from_history("shoulder_left"))[-self.scrolling_queue_size:]
        shoulder_right_list = list(result.get_attr_from_history("shoulder_right"))[-self.scrolling_queue_size:]
        hip_left_list = list(result.get_attr_from_history("hip_left"))[-self.scrolling_queue_size:]
        hip_right_list = list(result.get_attr_from_history("hip_right"))[-self.scrolling_queue_size:]
        move_dist_list = list()
        size_change_list = list()
        for (prev_shoulder_l, curr_shoulder_l), (prev_shoulder_r, curr_shoulder_r), (prev_hip_l, curr_hip_l), (prev_hip_r, curr_hip_r) in zip(
            zip(shoulder_left_list, shoulder_left_list[1:]),
            zip(shoulder_right_list, shoulder_right_list[1:]),
            zip(hip_left_list, hip_left_list[1:]),
            zip(hip_right_list, hip_right_list[1:])
        ):
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
            size_change = abs((curr_area) - (prev_area)) / (prev_area)
            move_dist_list.append(move_dist)
            size_change_list.append(size_change)

        # 判斷移動距離和邊界框大小變化是否超過閾值
        if not len(move_dist_list) or not len(size_change_list):
            return None, None
        return np.mean(move_dist_list),  np.mean(size_change_list)

    def _shoelace_formula(self, points: list[list[float]]) -> float:
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        area = abs(area) / 2.0
        return area

    def _get_angle(
        self,
        edge1: list[float] | None,
        edge2: list[float] | None,
        center: list[float] | None,
    ) -> float:
        if edge1 is None or edge2 is None or center is None:
            return float("inf")
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

    def _get_head_debug(
        self,
        bbox: list[float],
        shoulder: list[float] | None,
        eye: list[float] | None,
    ):
        if eye is None or shoulder is None:
            return -float('inf')
        return (eye[1] - bbox[1]) / (shoulder[1] - bbox[1])

    def _get_wrist_debug(
        self,
        shoulder: list[float] | None,
        hip: list[float] | None,
        wrist: list[float] | None
    ):
        if shoulder is None or hip is None or wrist is None:
            return float('inf')
        return (wrist[1] - shoulder[1]) / (hip[1] - shoulder[1])
