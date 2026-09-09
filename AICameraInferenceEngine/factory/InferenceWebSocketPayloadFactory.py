
from classes import InferenceConfig
from inference.model.result import Result
import numpy as np
from math import cos, sin
from loguru import logger

coco_kp_labels = [
    "nose", "eye_left", "eye_right", "ear_left", "ear_right",
    "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
    "wrist_left", "wrist_right", "hip_left", "hip_right",
    "knee_left", "knee_right", "ankle_left", "ankle_right"
]

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


def create_leave_zone_payload(frame, results: Result, config: InferenceConfig) -> dict:
    """
    Create a WebSocket payload for leave zone events.

    Args:
        result (Result): The inference result.
        config (InferenceConfig): The inference configuration.

    Returns:
        dict: The WebSocket payload.
    """
    data = []
    for result in results:
        data.append({
            "id": result.id,
            "bbox": result.bbox,
            "score": result.score,
    })
    return data

def create_age_gender_payload(frame ,results: list[Result], config: InferenceConfig) -> dict:
    """
    Create a WebSocket payload for age and gender events.

    Args:
        results (Result): The inference results.
        config (InferenceConfig): The inference configuration.

    Returns:
        dict: The WebSocket payload.
    """
    data = []
    for result in results:
        data.append({
            "id": result.id,
            "bbox": result.bbox,
            "age": result.age,
            "gender": result.gender,
        })
    return data

def create_head_tracking_payload(frame, results: list[Result], config: InferenceConfig):
    data = []
    
    for result in results:
        if result.bbox is not None and all([result.pitch is not None, result.roll is not None, result.yaw is not None]):
            x1, y1, x2, y2 = result.bbox
            length = x2 - x1
            center_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            r = result.roll * np.pi / 180
            y = -(result.yaw * np.pi / 180)
            p = result.pitch * np.pi / 180

            # 計算向量
            pitch_x = cos(y) * cos(r)
            pitch_y = cos(p) * sin(r) + cos(r) * sin(p) * sin(y)
            yaw_x = -cos(y) * sin(r)
            yaw_y = cos(p) * cos(r) - sin(p) * sin(y) * sin(r)
            roll_x = sin(y)
            roll_y = -cos(y) * sin(p)

            head_data = {
                'lines': [
                    {
                        'start': center_pos,
                        'end': (center_pos[0] + pitch_x * length, center_pos[1] + pitch_y * length),
                    },
                    {
                        'start': center_pos,
                        'end': (center_pos[0] + yaw_x * length, center_pos[1] + yaw_y * length),
                    },
                    {
                        'start': center_pos,
                        'end': (center_pos[0] + roll_x * length, center_pos[1] + roll_y * length),
                    }
                ]
            }
            
            data.append({
                "id": result.id,
                "bbox": result.bbox,
                "head_data": head_data
            })
    
    return data

def create_all_in_one_payload(frame, results: list[Result], config: InferenceConfig):
    data = []
    head_tracking = create_head_tracking_payload(frame, results, config)
    for index, result in enumerate(results):
        data.append({
            "id": result.id,
            "bbox": result.bbox,
            "score": result.score,
            "age": result.age + config.config["custom"].get("age_offset", 0),
            "gender": result.gender,
            "head_pose": head_tracking[index]['head_data'] if index < len(head_tracking) else None
        })

    return data

def create_post_payload(frame, results: list[Result], config: InferenceConfig):
    data = []

    for result in results:
        if not (result.is_coco_kps and result.bbox is not None):
            continue
            
        result_dict = result.to_dict()
        
        # 收集關鍵點資料
        keypoints = []
        for i, label in enumerate(coco_kp_labels):
            if result_dict.get(label, None) is not None:
                x, y, conf = result_dict[label]
                keypoints.append({
                    'label': label,
                    'position': { "x": x, "y": y },
                    'confidence': conf,
                })
        
        # 收集骨架連線資料
        skeleton_lines = []
        for i, (start, end) in enumerate(skeleton):
            if result_dict.get(coco_kp_labels[start - 1], None) is not None and result_dict.get(coco_kp_labels[end - 1], None) is not None:
                x1, y1, _ = result_dict[coco_kp_labels[start - 1]]
                x2, y2, _ = result_dict[coco_kp_labels[end - 1]]
                skeleton_lines.append({
                    'start': { "x": x1, "y": y1 },
                    'end': { "x": x2, "y": y2 },
                })

        data.append({
            "id": result.id,
            "bbox": result.bbox,
            "score": result.score,
            "post_data": {
                'keypoints': keypoints,
                'skeleton': skeleton_lines,
            }
        })
    return data

def create_inference_websocket_payload(frame, results: list[Result], config: InferenceConfig) -> dict:
    """
    Create a WebSocket payload for inference requests.

    Args:
        data (dict): The data to include in the payload.

    Returns:
        dict: The WebSocket payload.
    """
    if config.get('mode') == 'leave-zone':
        return create_leave_zone_payload(frame, results, config)
    elif config.get('mode') == 'leave-zone-with-age-gender':
        return create_age_gender_payload(frame, results, config)
    elif config.get('mode') == 'leave-zone-with-eye-tracking':
        return create_head_tracking_payload(frame, results, config)
    elif config.get('mode') == 'leave-zone-with-facial-features':
        return create_all_in_one_payload(frame, results, config)
    elif config.get('mode') == 'phone-pose-detection':
        return create_post_payload(frame, results, config)

    return None