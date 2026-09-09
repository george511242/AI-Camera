#!/usr/bin/env python3
"""Evaluate an Age ONNX model on frozen UTKFace GT crops."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.utils import img_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--margin", type=float, default=0.45)
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 3, 64, 64]:
        raise RuntimeError(f"unexpected input: {session.get_inputs()[0].shape}")
    if len(session.get_outputs()) != 1 or session.get_outputs()[0].shape != [1, 1]:
        raise RuntimeError("unexpected Age ONNX output")
    with args.manifest.open() as handle:
        rows = [json.loads(line) for line in handle if '"dataset": "utkface"' in line]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with args.out.open("w") as output:
        for index, row in enumerate(rows, 1):
            record = dict(row)
            record.update({
                "version": args.version, "crop_source": "gt",
                "backend": "onnxruntime", "evaluated_tasks": ["age"],
                "predicted_age": None, "predicted_gender": None,
                "predicted_pitch": None, "predicted_yaw": None,
                "predicted_roll": None, "failure_reason": None,
            })
            try:
                path = args.data_root / f"h{row['target_face_height_px']}" / "utkface" / Path(row["file"]).name
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"unable to read image: {path}")
                bbox = np.asarray(row["gt_bbox"], dtype=np.float64) / [640, 480, 640, 480]
                face = img_utils.resize(img_utils.crop(image, bbox.tolist(), args.margin), (64, 64))
                tensor = np.transpose(np.asarray([face], dtype=np.float32), (0, 3, 1, 2))
                record["predicted_age"] = round(float(session.run(None, {"input": tensor})[0].reshape(-1)[0]), 7)
            except Exception as error:
                record["failure_reason"] = "runtime_error"
                record["error"] = str(error)[:300]
            output.write(json.dumps(record) + "\n")
            if index % 250 == 0:
                print(f"{index}/{len(rows)}", flush=True)
    print(f"DONE {len(rows)} rows in {time.time() - started:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
