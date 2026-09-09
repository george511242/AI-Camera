#!/usr/bin/env python3
"""Evaluate a Gender ONNX model on frozen UTKFace GT crops."""

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


CANVAS_W = 640
CANVAS_H = 480


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--margin", type=float, default=0.45)
    args = parser.parse_args()

    session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or inputs[0].name != "input":
        raise RuntimeError(f"unexpected ONNX inputs: {[(i.name, i.shape) for i in inputs]}")
    if inputs[0].shape != [1, 3, 64, 64]:
        raise RuntimeError(f"unexpected ONNX input shape: {inputs[0].shape}")
    if len(outputs) != 1 or outputs[0].shape != [1, 1]:
        raise RuntimeError(f"unexpected ONNX outputs: {[(o.name, o.shape) for o in outputs]}")

    with args.manifest.open() as handle:
        rows = [json.loads(line) for line in handle]
    rows = [row for row in rows if row["dataset"] == "utkface"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with args.out.open("w") as output_handle:
        for index, row in enumerate(rows, 1):
            record = dict(row)
            record.update({
                "version": args.version,
                "crop_source": "gt",
                "backend": "onnxruntime",
                "evaluated_tasks": ["gender"],
                "predicted_age": None,
                "predicted_gender": None,
                "predicted_pitch": None,
                "predicted_yaw": None,
                "predicted_roll": None,
                "failure_reason": None,
            })
            try:
                source_name = Path(row["file"]).name
                image_path = (args.data_root / f"h{row['target_face_height_px']}"
                              / "utkface" / source_name)
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f"unable to read image: {image_path}")
                bbox = np.asarray(row["gt_bbox"], dtype=np.float64)
                bbox /= [CANVAS_W, CANVAS_H, CANVAS_W, CANVAS_H]
                face = img_utils.crop(image, bbox.tolist(), args.margin)
                face = img_utils.resize(face, (64, 64))
                model_input = np.transpose(
                    img_utils.to_batch([face]).astype(np.float32), (0, 3, 1, 2)
                )
                prediction = session.run(None, {"input": model_input})[0]
                record["predicted_gender"] = round(
                    float(np.asarray(prediction).reshape(-1)[0]), 7
                )
            except Exception as error:
                record["failure_reason"] = "runtime_error"
                record["error"] = str(error)[:300]
            output_handle.write(json.dumps(record) + "\n")
            if index % 250 == 0:
                print(f"{index}/{len(rows)}", flush=True)
    print(f"DONE {len(rows)} rows in {time.time() - started:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
