#!/usr/bin/env python3
"""Compare baseline and one fixed tiling strategy on WIDER validation."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from inference.onnx.yunet import YuNet


def load_wider_boxes(path):
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
            values = [float(value) for value in lines[index].split()[:4]]
            index += 1
            x, y, w, h = values
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
        records.append((name, boxes))
    return records


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def normalized_nms(detections, threshold=0.3):
    order = sorted(range(len(detections)), key=lambda i: detections[i].score, reverse=True)
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [
            index for index in order
            if overlap(detections[current].bbox, detections[index].bbox) <= threshold
        ]
    return [detections[index] for index in keep]


def tiled_detect(detector, image):
    height, width = image.shape[:2]
    tile_w, tile_h = round(width * 0.6), round(height * 0.6)
    origins = [(0, 0), (width - tile_w, 0), (0, height - tile_h),
               (width - tile_w, height - tile_h)]
    detections = detector(image.astype(np.float32), [])
    for x0, y0 in origins:
        tile = image[y0:y0 + tile_h, x0:x0 + tile_w].astype(np.float32)
        for result in detector(tile, []):
            x1, y1, x2, y2 = result.bbox
            result.bbox = [
                (x0 + x1 * tile_w) / width,
                (y0 + y1 * tile_h) / height,
                (x0 + x2 * tile_w) / width,
                (y0 + y2 * tile_h) / height,
            ]
            detections.append(result)
    if not detections:
        return []
    return normalized_nms(detections)


def evaluate(detections, gt_boxes, target_indices, width, height):
    norm = np.asarray([item.bbox for item in detections])
    if len(norm):
        norm = norm * [width, height, width, height]
    matched = 0
    matched_scores = []
    for index in target_indices:
        if len(norm):
            overlaps = np.asarray([overlap(box, gt_boxes[index]) for box in norm])
            best = int(np.argmax(overlaps))
            if overlaps[best] >= 0.5:
                matched += 1
                matched_scores.append(float(detections[best].score))
    true_detections = sum(
        any(overlap(box, gt) >= 0.5 for gt in gt_boxes)
        for box in norm
    )
    return {
        "targets": len(target_indices),
        "matched": matched,
        "detections": len(detections),
        "true_detections": true_detections,
        "scores": matched_scores,
    }


def summarize(records, elapsed):
    targets = sum(item["targets"] for item in records)
    matched = sum(item["matched"] for item in records)
    detections = sum(item["detections"] for item in records)
    true_detections = sum(item["true_detections"] for item in records)
    scores = [score for item in records for score in item["scores"]]
    return {
        "images": len(records),
        "target_faces": targets,
        "matched_faces": matched,
        "recall_iou_0_5": matched / targets if targets else 0,
        "detections": detections,
        "detection_precision_proxy": true_detections / detections if detections else 0,
        "matched_score_median": float(np.median(scores)) if scores else None,
        "elapsed_seconds": elapsed,
        "mean_seconds_per_image": elapsed / len(records) if records else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--score-threshold", type=float, default=0.75)
    args = parser.parse_args()

    candidates = []
    for name, boxes in load_wider_boxes(args.annotations):
        image = cv2.imread(str(args.images / name))
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = 640 / max(width, height)
        target_indices = [
            index for index, box in enumerate(boxes)
            if 20 <= (box[3] - box[1]) * scale <= 50
        ]
        if target_indices and len(boxes) <= 10:
            candidates.append((name, boxes, target_indices, image))
        if len(candidates) >= args.max_images:
            break

    detector = YuNet(str(args.onnx), score_threshold=args.score_threshold)
    baseline, tiled = [], []
    baseline_time = tiled_time = 0.0
    details = []
    for number, (name, boxes, targets, image) in enumerate(candidates, 1):
        start = time.perf_counter()
        base_detections = detector(image.astype(np.float32), [])
        baseline_time += time.perf_counter() - start
        start = time.perf_counter()
        tile_detections = tiled_detect(detector, image)
        tiled_time += time.perf_counter() - start
        height, width = image.shape[:2]
        base_metrics = evaluate(base_detections, boxes, targets, width, height)
        tile_metrics = evaluate(tile_detections, boxes, targets, width, height)
        baseline.append(base_metrics)
        tiled.append(tile_metrics)
        details.append({"file": name, "baseline": base_metrics, "tiled": tile_metrics})
        if number % 25 == 0:
            print(f"{number}/{len(candidates)}", flush=True)

    report = {
        "selection": "first deterministic WIDER val images with 1-10 GT faces and at least one face whose effective 640-input height is 20-50px",
        "target_definition": "GT face height * 640 / max(image width, image height)",
        "score_threshold": args.score_threshold,
        "baseline": summarize(baseline, baseline_time),
        "baseline_plus_tiled_2x2": summarize(tiled, tiled_time),
        "tiling": {"full_frame_passes": 1, "tiles": 4, "tile_width_fraction": 0.6, "tile_height_fraction": 0.6, "nms_iou": 0.3},
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
