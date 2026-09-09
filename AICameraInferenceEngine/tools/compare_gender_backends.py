#!/usr/bin/env python3
"""Compare Gender ONNX and RKNN simulator outputs on one VAC-style crop."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.utils import img_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--rk3566", type=Path, required=True)
    parser.add_argument("--rk3588", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    return parser.parse_args()


def infer_rknn_simulator(onnx_path: Path, target: str, face: np.ndarray) -> float:
    rknn = RKNN(verbose=False)
    try:
        result = rknn.config(
            target_platform=target,
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
        )
        if result != 0:
            raise RuntimeError(f"config failed for {target}: {result}")
        result = rknn.load_onnx(model=str(onnx_path))
        if result != 0:
            raise RuntimeError(f"load_onnx failed for {target}: {result}")
        result = rknn.build(do_quantization=False)
        if result != 0:
            raise RuntimeError(f"build failed for {target}: {result}")
        result = rknn.init_runtime()
        if result != 0:
            raise RuntimeError(f"init_runtime failed for {target}: {result}")
        output = rknn.inference(inputs=[face], data_format=["nhwc"])
        return float(np.asarray(output).reshape(-1)[0])
    finally:
        rknn.release()


def main():
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Unable to read image: {args.image}")

    face = img_utils.crop(image, args.bbox, margin=0.45)
    face = img_utils.resize(face, (64, 64))
    face = np.ascontiguousarray([face], dtype=np.uint8)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_input = np.transpose(face.astype(np.float32), (0, 3, 1, 2))
    onnx_output = float(session.run(None, {"input": onnx_input})[0].reshape(-1)[0])
    for path in (args.rk3566, args.rk3588):
        if not path.is_file():
            raise FileNotFoundError(f"Exported RKNN model not found: {path}")
    rk3566_output = infer_rknn_simulator(args.onnx, "rk3566", face)
    rk3588_output = infer_rknn_simulator(args.onnx, "rk3588", face)

    results = {
        "image": str(args.image),
        "bbox": args.bbox,
        "margin": 0.45,
        "crop_shape": list(face.shape),
        "crop_dtype": str(face.dtype),
        "onnx": onnx_output,
        "rk3566_simulator": rk3566_output,
        "rk3588_simulator": rk3588_output,
        "rk3566_abs_error": abs(rk3566_output - onnx_output),
        "rk3588_abs_error": abs(rk3588_output - onnx_output),
        "simulator_abs_difference": abs(rk3566_output - rk3588_output),
    }

    args.artifacts.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.artifacts / "test_face_gender_crop.png"), face[0])
    (args.artifacts / "gender_backend_comparison.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="ascii",
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
