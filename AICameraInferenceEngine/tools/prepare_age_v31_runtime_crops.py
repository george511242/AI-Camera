#!/usr/bin/env python3
"""Build Age v3.1 crops using the deployed YuNet/crop/resize semantics."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.onnx.yunet import YuNet
from inference.utils import img_utils


CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def place_utk_on_canvas(image, target_height):
    scale = target_height / image.shape[0]
    width = max(1, round(image.shape[1] * scale))
    height = max(1, round(image.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    scaled = cv2.resize(image, (width, height), interpolation=interpolation)
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), np.uint8)
    offset_x = CANVAS_WIDTH / 2 - width / 2
    offset_y = CANVAS_HEIGHT / 2 - height / 2
    left = round(offset_x)
    top = round(offset_y)
    canvas[top : top + height, left : left + width] = scaled
    gt = np.array(
        [offset_x / CANVAS_WIDTH, offset_y / CANVAS_HEIGHT,
         (offset_x + width) / CANVAS_WIDTH, (offset_y + height) / CANVAS_HEIGHT],
        dtype=np.float64,
    )
    # Match frozen benchmark's JPEG quality 95 before detector inference.
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("failed to encode synthetic canvas")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR), gt


def best_detection(detector, image, gt_bbox):
    # The ONNX graph expects float32 0..255; RKNN accepts the uint8 equivalent.
    candidates = detector(image.astype(np.float32), [])
    if not candidates:
        return None, 0.0
    best = max(candidates, key=lambda result: iou(result.bbox, gt_bbox))
    return best, iou(best.bbox, gt_bbox)


def process_variant(detector, image, gt_bbox, row, split, variant, output_root):
    detection, overlap = best_detection(detector, image, gt_bbox)
    record = {
        "source_id": row["source_id"],
        "source_file": row["file"],
        "age": row["age"],
        "gender": row["gender"],
        "age_bin": row["age_bin"],
        "split": split,
        "variant": variant,
        "detected": False,
        "bbox": None,
        "score": None,
        "iou": overlap,
        "file": None,
    }
    if detection is None or overlap < 0.5 or float(detection.score) < 0.75:
        return record
    face = img_utils.crop(image, detection.bbox, margin=0.45)
    if not face.size:
        return record
    face = img_utils.resize(face, (64, 64))
    relative = Path(split) / variant / f"{row['source_id']}.png"
    path = output_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), face):
        raise RuntimeError(f"failed to write {path}")
    record.update(
        {
            "detected": True,
            "bbox": [round(float(value), 8) for value in detection.bbox],
            "score": float(detection.score),
            "iou": float(overlap),
            "file": str(relative),
            "crop_sha256": sha256(path),
        }
    )
    return record


def write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--yunet", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector = YuNet(str(args.yunet))
    metadata = {
        "yunet": str(args.yunet.resolve()),
        "yunet_sha256": sha256(args.yunet),
        "cache_root": str(args.cache_root.resolve()),
        "margin": 0.45,
        "input_size": [64, 64],
        "color": "BGR",
        "value_range": [0, 255],
        "normalization": None,
        "variants": ["160px", "80px", "40px"],
        "splits": {},
    }
    for split in args.splits:
        source_manifest = args.split_dir / f"{split}.jsonl"
        rows = [json.loads(line) for line in source_manifest.open()]
        audit = []
        for index, row in enumerate(rows, 1):
            image = cv2.imread(str(args.image_root / row["file"]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot decode {row['file']}")
            for height in (160, 80, 40):
                canvas, gt = place_utk_on_canvas(image, height)
                audit.append(process_variant(
                    detector, canvas, gt, row, split, f"{height}px", args.cache_root
                ))
            if index % 250 == 0:
                print(f"[{split}] {index}/{len(rows)}", flush=True)
        successes = [row for row in audit if row["detected"]]
        write_jsonl(args.output_dir / f"{split}_runtime_all.jsonl", audit)
        write_jsonl(args.output_dir / f"{split}_runtime_success.jsonl", successes)
        metadata["splits"][split] = {
            "sources": len(rows),
            "rows": len(audit),
            "successes": len(successes),
            "success_by_variant": dict(Counter(
                row["variant"] for row in successes
            )),
            "audit_manifest_sha256": sha256(args.output_dir / f"{split}_runtime_all.jsonl"),
            "success_manifest_sha256": sha256(args.output_dir / f"{split}_runtime_success.jsonl"),
        }
    (args.output_dir / "runtime_crop_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
