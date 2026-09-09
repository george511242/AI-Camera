#!/usr/bin/env python3
"""Analyze frozen YuNet small-face failures without changing benchmark data."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from inference.onnx.yunet import YuNet


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def classify(row):
    if row.get("failure_reason") == "yunet_missed":
        return "complete_miss"
    if row.get("failure_reason") == "yunet_score_below_threshold":
        return "low_confidence"
    if row.get("failure_reason") == "wrong_face_matched" or row.get("iou", 0) < 0.5:
        return "poor_iou"
    return "success"


def add_low_threshold_diagnostics(rows, model_path, limit_per_group):
    detector = YuNet(str(model_path), score_threshold=0.01)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["target_face_height_px"], classify(row))].append(row)

    selected = []
    for key, items in sorted(grouped.items()):
        dataset, height, category = key
        if height not in (27, 40):
            continue
        limit = len(items) if dataset == "utkface" and height == 27 and category != "complete_miss" else limit_per_group
        selected.extend(items[:limit])

    for index, row in enumerate(selected, 1):
        image = cv2.imread(row["file"])
        if image is None:
            row["diagnostic_error"] = "image_read_failed"
            continue
        detections = detector(image.astype(np.float32), [])
        gt = np.asarray(row["gt_bbox"], dtype=np.float64) / [640, 480, 640, 480]
        if detections:
            best_score = max(detections, key=lambda item: float(item.score))
            best_iou = max(detections, key=lambda item: iou(item.bbox, gt))
            row["low_threshold_max_score"] = round(float(best_score.score), 6)
            row["low_threshold_best_iou"] = round(iou(best_iou.bbox, gt), 6)
            row["low_threshold_best_iou_score"] = round(float(best_iou.score), 6)
            row["low_threshold_best_bbox"] = [round(float(v), 6) for v in best_iou.bbox]
        else:
            row["low_threshold_max_score"] = 0.0
            row["low_threshold_best_iou"] = 0.0
            row["low_threshold_best_iou_score"] = 0.0
            row["low_threshold_best_bbox"] = None
        if index % 25 == 0:
            print(f"diagnostic inference {index}/{len(selected)}", flush=True)
    return selected


def draw_sample(row, tile_size=(240, 180)):
    image = cv2.imread(row["file"])
    if image is None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
    gt = np.asarray(row["gt_bbox"], dtype=int)
    cv2.rectangle(image, tuple(gt[:2]), tuple(gt[2:]), (0, 255, 0), 2)
    bbox = row.get("detected_bbox") or row.get("low_threshold_best_bbox")
    if bbox:
        box = (np.asarray(bbox) * [640, 480, 640, 480]).astype(int)
        cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 2)
    score = row.get("yunet_score")
    if score is None:
        score = row.get("low_threshold_best_iou_score", 0)
    label = f"{row['dataset']} h{row['target_face_height_px']} {classify(row)} s={score:.3f} iou={row.get('iou', 0):.2f}"
    cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
    cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
    return cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA)


def write_contact_sheets(rows, output_dir):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["target_face_height_px"], classify(row))].append(row)
    for (dataset, height, category), items in sorted(grouped.items()):
        if height not in (27, 40):
            continue
        items = items[:30]
        if not items:
            continue
        tiles = [draw_sample(row) for row in items]
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
        cv2.imwrite(str(output_dir / f"{dataset}_h{height}_{category}.jpg"), sheet)


def content_statistics(rows):
    result = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        values = []
        for row in rows:
            if row["dataset"] != dataset or row["target_face_height_px"] != 27:
                continue
            image = cv2.imread(row["file"])
            if image is None:
                continue
            mask = np.max(image, axis=2) > 8
            ys, xs = np.where(mask)
            if len(xs):
                values.append(((xs.max() - xs.min() + 1), (ys.max() - ys.min() + 1), mask.mean()))
        arr = np.asarray(values, dtype=float)
        result[dataset] = {
            "samples": len(values),
            "median_nonblack_width": round(float(np.median(arr[:, 0])), 2),
            "median_nonblack_height": round(float(np.median(arr[:, 1])), 2),
            "median_nonblack_fraction": round(float(np.median(arr[:, 2])), 6),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-per-group", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.results.open()]
    taxonomy = Counter((row["dataset"], row["target_face_height_px"], classify(row)) for row in rows)
    selected = add_low_threshold_diagnostics(rows, args.onnx, args.limit_per_group)
    write_contact_sheets(selected, args.output_dir)
    report = {
        "source_results": str(args.results),
        "onnx": str(args.onnx),
        "taxonomy": [
            {"dataset": key[0], "height": key[1], "category": key[2], "count": count}
            for key, count in sorted(taxonomy.items())
        ],
        "canvas_content_statistics_27px": content_statistics(rows),
        "diagnostic_samples": selected,
    }
    (args.output_dir / "failure_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "diagnostic_samples"}, indent=2))


if __name__ == "__main__":
    main()
