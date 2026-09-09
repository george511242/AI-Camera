from ..model import Model, Result
from math import cos, sin
import numpy as np
import cv2

pose_palette = np.array([
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
    [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
    [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
], dtype=np.uint8)
kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
            [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]


def draw_result(
    ori_img: cv2.typing.MatLike,
    result: Result,
    text_offset_y=12,
    show_score=True,
    show_crop_info=True,
    show_imagenet_class=True,
    show_coco_class=True,
    show_id=True,
    show_tracking_info=True,
    show_age_gender=True,
    show_coco_kps=True,
    show_pose=True,
    show_pose_info=True,
    show_landmarks=True,
    show_focus_count=True,
    show_debug=True,
    ignore_disappear=True,
    text_color=(255, 255, 255),
    **kwargs
):
    img = ori_img.copy()
    if result.bbox is not None:
        if ignore_disappear and result.disappear_count:
            return img
        img_h, img_w = img.shape[0], img.shape[1]

        # bbox
        y_offset = text_offset_y
        bbox = result.bbox
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if show_crop_info:
            if result.crop_info is not None:
                cx1, cy1, cx2, cy2 = result.crop_info
                cx1, cy1, cx2, cy2 = int(cx1 * img_w), int(cy1 * img_h), int(cx2 * img_w), int(cy2 * img_h)
                img = cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (240, 240, 240), 1)
            if result.crop_infos is not None:
                for [cx1, cy1, cx2, cy2] in result.crop_infos:
                    cx1, cy1, cx2, cy2 = int(cx1 * img_w), int(cy1 * img_h), int(cx2 * img_w), int(cy2 * img_h)
                    img = cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (240, 240, 240), 1)

        # score
        if show_score and result.score is not None:
            y_offset -= 20
            img = cv2.putText(img, f'Score: {result.score * 100:.0f}%', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # coco class
        if show_coco_class and result.coco_class_idx is not None:
            y_offset -= 20
            img = cv2.putText(img, Model.coco_obj_classes[int(result.coco_class_idx)], (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # imagenet class
        if show_imagenet_class and result.imagenet_class_idx is not None:
            y_offset -= 20
            img = cv2.putText(img, f"ImageNet: {result.imagenet_class_idx:d}", (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # landmarks
        if show_landmarks:
            for pos in [result.eye_right, result.eye_left, result.nose, result.mouth_right, result.mouth_left]:
                if pos is None:
                    continue
                x, y = pos
                img = cv2.circle(img, [int(x * img_w), int(y * img_h)], 1, (0, 155, 255), 1)

        # tracking
        if show_id and result.id is not None:
            y_offset -= 20
            img = cv2.putText(img, f'ID: {result.id[-6:]}', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        if show_tracking_info:
            if result.hits is not None:
                y_offset -= 20
                img = cv2.putText(img, f'Hits: {result.hits}', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # age, gender
        if show_age_gender:
            if result.age is not None:
                y_offset -= 20
                img = cv2.putText(img, f'Age: {result.age:.0f}', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            if result.gender is not None:
                y_offset -= 20
                img = cv2.putText(img, f'Gender: {result.gender:.0f}', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # head
        if show_pose and all([result.pitch is not None, result.roll is not None, result.yaw is not None]):
            center_pos = result.nose if result.nose is not None else result.center
            center_pos = center_pos[0] * img_w, center_pos[1] * img_h
            length = result.bbox[2] - result.bbox[0]
            length /= 3 if result.creator in ["YoloV8Pose"] else 1
            length *= img_w

            r = result.roll * np.pi / 180
            y = -(result.yaw * np.pi / 180)
            p = result.pitch * np.pi / 180

            # X-Axis pointing to right. drawn in red
            pitch_x = cos(y) * cos(r)
            pitch_y = cos(p) * sin(r) + cos(r) * sin(p) * sin(y)

            # Y-Axis | drawn in green
            #        v
            yaw_x = -cos(y) * sin(r)
            yaw_y = cos(p) * cos(r) - sin(p) * sin(y) * sin(r)

            # Z-Axis (out of the screen) drawn in blue
            roll_x = sin(y)
            roll_y = -cos(y) * sin(p)

            dx, dy, dz = (pitch_x, pitch_y), (yaw_x, yaw_y), (roll_x, roll_y)
            img = cv2.line(img, tuple(np.round(center_pos).astype(int)), tuple(np.round(center_pos + np.array(dx) * length).astype(int)), (255, 0, 0), 2)
            img = cv2.line(img, tuple(np.round(center_pos).astype(int)), tuple(np.round(center_pos + np.array(dy) * length).astype(int)), (0, 255, 0), 2)
            img = cv2.line(img, tuple(np.round(center_pos).astype(int)), tuple(np.round(center_pos + np.array(dz) * length).astype(int)), (0, 0, 255), 2)

        # coco keypoints
        if show_coco_kps and result.is_coco_kps:
            result_dict = result.to_dict()
            for i, label in enumerate(Model.coco_kp_labels):
                if result_dict.get(label, None) is not None:
                    x, y, conf = result_dict[label]
                    img = cv2.circle(img, (int(x * img_w), int(y * img_h)), 2, kpt_color[i].tolist(), 1)
            for i, (start, end) in enumerate(skeleton):
                if result_dict.get(Model.coco_kp_labels[start - 1], None) is not None and result_dict.get(Model.coco_kp_labels[end - 1], None) is not None:
                    x_from, y_from, _ = result_dict[Model.coco_kp_labels[start - 1]]
                    x_to, y_to, _ = result_dict[Model.coco_kp_labels[end - 1]]
                    img = cv2.line(img, (int(x_from * img_w), int(y_from * img_h)), (int(x_to * img_w), int(y_to * img_h)), limb_color[i].tolist(), 1)

        if show_pose_info:
            if result.roll is not None:
                y_offset -= 20
                img = cv2.putText(img, f'Roll: {result.roll:.0f}', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            if result.yaw is not None:
                y_offset -= 20
                img = cv2.putText(img, f'Yaw: {result.yaw:.0f}', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            if result.pitch is not None:
                y_offset -= 20
                img = cv2.putText(img, f'Pitch: {result.roll:.0f}', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        if show_focus_count and result.focus_count is not None:
            y_offset -= 20
            img = cv2.putText(img, f'Focus: {result.focus_count}', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # debugs text
        if show_debug:
            if result.debug3 is not None:
                y_offset -= 20
                img = cv2.putText(img, f'3: {result.debug3}', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            if result.debug2 is not None:
                y_offset -= 20
                img = cv2.putText(img, f'2: {result.debug2}', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            if result.debug1 is not None:
                y_offset -= 20
                img = cv2.putText(img, f'1: {result.debug1}', (x1, y1 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return img


def draw_region(img: cv2.typing.MatLike, name: str, region: list[float], color: tuple[int, int, int] = (255, 0, 0), alpha: float = 0.4):
    h, w, _ = img.shape
    x1, y1, x2, y2 = region
    x1 = int(x1 * w)
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)

    (tw, th), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
    pad = 6  # rectangle thickness (5) rounded up
    roi_x1 = max(0, min(x1, x1 + 20) - pad)
    roi_y1 = max(0, min(y1, y1 + 50 - th) - pad)
    roi_x2 = min(w, max(x2, x1 + 20 + tw) + pad)
    roi_y2 = min(h, max(y2, y1 + 50 + baseline) + pad)

    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return img

    roi_backup = img[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
    cv2.putText(img, name, (x1 + 20, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA)
    roi_view = img[roi_y1:roi_y2, roi_x1:roi_x2]
    cv2.addWeighted(roi_view, alpha, roi_backup, 1 - alpha, 0, dst=roi_view)
    return img


def draw_grid(img: cv2.typing.MatLike, grid_shape: tuple[int, int] = (10, 10), color: tuple[int, int, int] = (100, 100, 100), thickness: int = 1, alpha: float = 0.4):
    h, w, _ = img.shape
    ori_img = img.copy()
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    img = cv2.addWeighted(img, alpha, ori_img, 1 - alpha, 0)  # type: ignore
    return img
