import cv2
import numpy as np
from numpy.typing import NDArray
import base64


def resize(img: cv2.typing.MatLike, target_size: list[int] | tuple[int, int]):
    """
    Resize the image to the target size.
    :param img: The input image.
    :param target_size: The target size in the format [width, height].
    :return: The resized image.
    """
    h, w = img.shape[:2]
    if w == target_size[0] and h == target_size[1]:
        return img
    return cv2.resize(img, target_size)


def switch_br(img: cv2.typing.MatLike):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def crop_info(bbox: list[float], margin: float | list[float] = 0):
    x1, y1, x2, y2 = bbox
    if isinstance(margin, list):
        margin_x1, margin_y1, margin_x2, margin_y2 = margin
    else:
        margin_x1, margin_y1, margin_x2, margin_y2 = margin, margin, margin, margin
    w = x2 - x1
    h = y2 - y1
    return (
        max(x1 - margin_x1 * w, 0),
        max(y1 - margin_y1 * h, 0),
        min(x2 + margin_x2 * w, 1),
        min(y2 + margin_y2 * h, 1)
    )


def crop(img: cv2.typing.MatLike, bbox: list[float], margin: float | list[float] = 0):
    """
    Crop the image with the given bounding box and margin.
    :param img: The input image.
    :param bbox: The bounding box in the format [x1, y1, x2, y2], where the value is between 0 and 1.
    :param margin: The margin to be added to the bounding box.
    :return: The cropped image.
    """
    img_h, img_w = img.shape[:2]
    mx1, my1, mx2, my2 = crop_info(bbox, margin)

    l = int(mx1 * img_w)
    t = int(my1 * img_h)
    r = int(mx2 * img_w)
    b = int(my2 * img_h)

    return img[t: b, l: r, :]


def fill_mosaic(img: cv2.typing.MatLike, bbox: list[float], margin: float | list[float] = 0, level=10):
    img_h, img_w = img.shape[:2]
    mx1, my1, mx2, my2 = crop_info(bbox, margin)
    l = int(mx1 * img_w)
    t = int(my1 * img_h)
    r = int(mx2 * img_w)
    b = int(my2 * img_h)
    h, w = max(int((r - l) / level), 1), max(int((b - t) / level), 1)
    mosaic = img[t: b, l: r, :]
    mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(mosaic, (r - l, b - t), interpolation=cv2.INTER_NEAREST)
    img[t: b, l: r, :] = mosaic
    return img


def fill_constant(img: cv2.typing.MatLike, bbox: list[float], constant: int = 0):
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    l = int(x1 * img_w)
    t = int(y1 * img_h)
    r = int(x2 * img_w)
    b = int(y2 * img_h)
    img[t: b, l: r, :] = constant
    return img


def pad_zero_info(w: int, h: int, target_aspect_ratio: float) -> list[int]:
    new_w, new_h = w, h
    pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    current_ar = w / h
    if current_ar == target_aspect_ratio:
        return [new_w, new_h, pad_left, pad_right, pad_top, pad_bottom]
    if current_ar > target_aspect_ratio:
        new_h = int(w / target_aspect_ratio)
    else:
        new_w = int(h * target_aspect_ratio)
    pad_x = new_w - w
    pad_y = new_h - h
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    return [new_w, new_h, pad_left, pad_right, pad_top, pad_bottom]


def pad_zero(img: cv2.typing.MatLike, target_aspect_ratio: float, fill_value: int = 0):
    img_h, img_w = img.shape[:2]
    _, _, pad_left, pad_right, pad_top, pad_bottom = pad_zero_info(img_w, img_h, target_aspect_ratio)
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[fill_value, fill_value, fill_value])


def to_gray(img: cv2.typing.MatLike):
    """
    Convert the image to grayscale.
    :param img: The input image.
    :return: The grayscale image.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gray_to_rgb(img: cv2.typing.MatLike):
    """
    Convert the grayscale image to RGB.
    :param img: The input grayscale image.
    :return: The RGB image.
    """
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def z_norm(img: cv2.typing.MatLike, mean: list[float], std: list[float]):
    return (img.astype(np.float32) - mean) / std


def gaussian_blur(img: cv2.typing.MatLike, kernel_size: int):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny_edge(img: cv2.typing.MatLike, low_threshold: int = 100, high_threshold: int = 200):
    """
    Apply Canny edge detection to the image.
    :param img: The input image.
    :param low_threshold: The lower threshold for the hysteresis procedure.
    :param high_threshold: The upper threshold for the hysteresis procedure.
    :return: The image with edges detected.
    """
    return cv2.Canny(img, low_threshold, high_threshold)


def channel_first(img: cv2.typing.MatLike):
    return np.transpose(img, (2, 0, 1))


def nms(boxes: NDArray[np.float32], scores: NDArray[np.float32], score_threshold: float, nms_threshold: float) -> list[int]:
    """
    Non-maximum suppression for bounding boxes.
    :param boxes: Bounding boxes in the format [x1, y1, x2, y2].
    :param scores: Confidence scores for each bounding box.
    :param score_threshold: Minimum score threshold for a bounding box to be considered.
    :param nms_threshold: IoU threshold for non-maximum suppression.
    :return: Indices of the bounding boxes to keep after non-maximum suppression.
    """
    bboxes = boxes.copy()
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    keep = cv2.dnn.NMSBoxes(
        bboxes=boxes.astype(np.int16).tolist(),
        scores=scores.tolist(),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        eta=1,
        top_k=5000
    )
    if len(keep):
        return keep.flatten()  # type: ignore
    else:
        return []


def expand_dim(img: cv2.typing.MatLike):
    return np.expand_dims(img, 0)


def concat(imgs: list[cv2.typing.MatLike], axis=0):
    return np.concatenate(imgs, axis=axis)


def to_batch(imgs: list[cv2.typing.MatLike]):
    # Preserve the source dtype (typically uint8). RKNN INT8-quantized models
    # accept uint8 input natively and quantize internally; forcing float32 here
    # quadruples memory bandwidth and makes RKNN re-quantize on set_inputs.
    return np.ascontiguousarray(imgs)


def get_distance(pos1, pos2): return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5


def softmax(x, axis=None):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def get_frame_from_video(path: str, max_frame_count=float('inf')):
    frame_list = list()
    cap = cv2.VideoCapture(path)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while cap.isOpened() and count < max_frame_count:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            break
        frame_list.append(frame)
        count += 1
    cap.release()
    return frame_list, (frame_height, frame_width), fps


def get_video_writer(path: str, fps: float, size: tuple[int, int]):
    """
        videoWriter = getVideoWriter(path, fps, size)
        videoWriter.write(img)
        ...
        videoWriter.release()
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    video = cv2.VideoWriter(path, fourcc, fps, size)
    return video

# deprecated below


def cropFaceAffine(img, detection):
    img_h, img_w = img.shape[0], img.shape[1]

    # Adapted from imutils package
    left_eye_coord = (0.70, 0.35)
    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    right_eye_x = 1.0 - left_eye_coord[0]

    le = detection['left_eye']
    re = detection['right_eye']
    lePixel = [le[0] * img_w, le[1] * img_h]
    rePixel = [re[0] * img_w, re[1] * img_h]
    originPixcel = ((lePixel[0] + rePixel[0]) / 2, (lePixel[1] + rePixel[1]) / 2)

    # compute the angle between the eye centroids
    dY = rePixel[1] - lePixel[1]
    dX = rePixel[0] - lePixel[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    scale = ((right_eye_x - left_eye_coord[0]) * 112) / dist

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(originPixcel, angle, scale)

    # update the translation component of the matrix
    tX = 112 * 0.5
    tY = 112 * left_eye_coord[1]
    M[0, 2] += (tX - originPixcel[0])
    M[1, 2] += (tY - originPixcel[1])

    # apply the affine transformation
    return cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_CUBIC)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    # _, indexes = scores.sort(descending=True)
    indexes = np.argsort(scores)
    # indexes = indexes[:candidate_size]
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        # current = indexes[0]
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        # indexes = indexes[1:]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def encode_image(img: cv2.typing.MatLike) -> str:
    """
    將 OpenCV 圖像轉換為 base64 字串

    Args:
        img: OpenCV 圖像

    Returns:
        base64 編碼的圖像字串
    """
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')
