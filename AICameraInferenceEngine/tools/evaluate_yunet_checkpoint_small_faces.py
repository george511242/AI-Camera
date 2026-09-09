#!/usr/bin/env python3
"""Read-only YuNet checkpoint evaluation on the fixed WIDER small-face subset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass
class Detection:
    bbox: list[float]
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--score-threshold", type=float, default=0.75)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--input-color", choices=("bgr", "rgb"), default="bgr")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_wider_boxes(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    index = 0
    while index < len(lines):
        name = lines[index].strip()
        index += 1
        count = int(lines[index])
        index += 1
        boxes = []
        for _ in range(count):
            x, y, w, h = [float(value) for value in lines[index].split()[:4]]
            index += 1
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
        records.append((name, boxes))
    return records


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


class CheckpointYuNet:
    def __init__(self, args: argparse.Namespace):
        sys.path.insert(0, str(args.trainer_root))
        from yunet_train.tasks.face import build_yunet

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        self.model = build_yunet(config.get("variant", "yunet_n"))
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        self.model.load_state_dict(state_dict, strict=True)
        self.device = torch.device(args.device)
        self.model.to(self.device).eval()
        self.score_threshold = args.score_threshold
        self.input_color = args.input_color
        self.strides = (8, 16, 32)

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> list[Detection]:
        image_h, image_w = image.shape[:2]
        side = max(image_w, image_h)
        pad_left = (side - image_w) // 2
        pad_right = side - image_w - pad_left
        pad_top = (side - image_h) // 2
        pad_bottom = side - image_h - pad_top
        padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        resized = cv2.resize(padded, (640, 640), interpolation=cv2.INTER_LINEAR)
        if self.input_color == "rgb":
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            np.ascontiguousarray(resized.transpose(2, 0, 1))
        ).float().unsqueeze(0).to(self.device)
        cls_preds, bbox_preds, obj_preds, _ = self.model(tensor)

        all_boxes, all_scores = [], []
        for stride, cls_pred, bbox_pred, obj_pred in zip(
            self.strides, cls_preds, bbox_preds, obj_preds
        ):
            cls = cls_pred.sigmoid().permute(0, 2, 3, 1).reshape(-1)
            obj = obj_pred.sigmoid().permute(0, 2, 3, 1).reshape(-1)
            reg = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            height, width = cls_pred.shape[2:]
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=self.device),
                torch.arange(width, device=self.device),
                indexing="ij",
            )
            centers = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2) * stride
            center_xy = reg[:, :2] * stride + centers
            size_wh = reg[:, 2:].exp() * stride
            boxes = torch.cat((center_xy - size_wh / 2, center_xy + size_wh / 2), dim=1)
            scores = torch.sqrt(cls * obj)
            keep = scores >= self.score_threshold
            all_boxes.append(boxes[keep])
            all_scores.append(scores[keep])

        if not all_scores or sum(scores.numel() for scores in all_scores) == 0:
            return []
        boxes = torch.cat(all_boxes).cpu().numpy()
        scores = torch.cat(all_scores).cpu().numpy()
        # Preserve the deployed VAC helper's historical OpenCV call exactly:
        # it passes integer xyxy values to NMSBoxes despite preparing an xywh copy.
        keep = cv2.dnn.NMSBoxes(
            boxes.astype(np.int16).tolist(), scores.tolist(), 0.0, 0.3,
            eta=1, top_k=5000,
        )
        if len(keep) == 0:
            return []

        scale = side / 640.0
        detections = []
        for index in np.asarray(keep).reshape(-1):
            box = boxes[index]
            normalized = np.clip([
                (box[0] * scale - pad_left) / image_w,
                (box[1] * scale - pad_top) / image_h,
                (box[2] * scale - pad_left) / image_w,
                (box[3] * scale - pad_top) / image_h,
            ], 0.0, 1.0)
            if normalized[0] < normalized[2] and normalized[1] < normalized[3]:
                detections.append(Detection(normalized.tolist(), float(scores[index])))
        return detections


def evaluate_image(detections, gt_boxes, target_indices, width, height):
    predicted = np.asarray([item.bbox for item in detections], dtype=np.float32)
    if len(predicted):
        predicted *= [width, height, width, height]
    matched = 0
    matched_scores = []
    for target_index in target_indices:
        if len(predicted):
            overlaps = np.asarray([
                overlap(box, gt_boxes[target_index]) for box in predicted
            ])
            best = int(np.argmax(overlaps))
            if overlaps[best] >= 0.5:
                matched += 1
                matched_scores.append(detections[best].score)
    true_detections = sum(
        any(overlap(box, gt) >= 0.5 for gt in gt_boxes) for box in predicted
    )
    return len(target_indices), matched, len(detections), true_detections, matched_scores


def main() -> None:
    args = parse_args()
    detector = CheckpointYuNet(args)
    selected = []
    for name, boxes in load_wider_boxes(args.annotations):
        image = cv2.imread(str(args.images / name))
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = 640 / max(width, height)
        targets = [
            index for index, box in enumerate(boxes)
            if 20 <= (box[3] - box[1]) * scale <= 50
        ]
        if targets and len(boxes) <= 10:
            selected.append((name, boxes, targets, image))
        if len(selected) >= args.max_images:
            break

    totals = dict(targets=0, matched=0, detections=0, true_detections=0)
    bin_totals = {
        "20-30px": {"targets": 0, "matched": 0},
        "30-40px": {"targets": 0, "matched": 0},
        "40-50px": {"targets": 0, "matched": 0},
    }
    scores = []
    started = time.perf_counter()
    for number, (_, boxes, targets, image) in enumerate(selected, 1):
        detections = detector(image)
        result = evaluate_image(
            detections, boxes, targets, image.shape[1], image.shape[0]
        )
        for key, value in zip(totals, result[:4]):
            totals[key] += value
        scores.extend(result[4])
        effective_scale = 640 / max(image.shape[1], image.shape[0])
        for label, lower, upper in (
            ("20-30px", 20, 30),
            ("30-40px", 30, 40),
            ("40-50px", 40, 50.0000001),
        ):
            bin_indices = [
                index for index in targets
                if lower <= (boxes[index][3] - boxes[index][1]) * effective_scale < upper
            ]
            bin_result = evaluate_image(
                detections, boxes, bin_indices, image.shape[1], image.shape[0]
            )
            bin_totals[label]["targets"] += bin_result[0]
            bin_totals[label]["matched"] += bin_result[1]
        if number % 25 == 0:
            print(f"{number}/{len(selected)}", flush=True)
    elapsed = time.perf_counter() - started
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "protocol": {
            "selection": "same deterministic first-200 WIDER val small-face subset",
            "effective_face_height_px": [20, 50],
            "max_gt_faces_per_image": 10,
            "score_threshold": args.score_threshold,
            "score_formula": "sqrt(sigmoid(cls) * sigmoid(objectness))",
            "nms_iou": 0.3,
            "match_iou": args.iou_threshold,
            "input_color": args.input_color,
            "device": args.device,
        },
        "images": len(selected),
        "target_faces": totals["targets"],
        "matched_faces": totals["matched"],
        "recall_iou_0_5": totals["matched"] / totals["targets"],
        "detections": totals["detections"],
        "true_detections": totals["true_detections"],
        "detection_precision_proxy": (
            totals["true_detections"] / totals["detections"]
            if totals["detections"] else 0.0
        ),
        "matched_score_median": float(np.median(scores)) if scores else None,
        "face_height_bins": {
            label: {
                **values,
                "recall_iou_0_5": values["matched"] / values["targets"]
                if values["targets"] else 0.0,
            }
            for label, values in bin_totals.items()
        },
        "elapsed_seconds": elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
