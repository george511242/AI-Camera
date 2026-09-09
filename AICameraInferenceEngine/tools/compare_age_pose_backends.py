#!/usr/bin/env python3
"""Compare Age and Head Pose ONNX and RKNN simulator outputs."""

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


POSE_MEAN = [123.675, 116.28, 103.53]
POSE_STD = [58.395, 57.12, 57.375]


def infer_simulator(onnx_path, target, image, mean, std):
    rknn = RKNN(verbose=False)
    try:
        steps = (
            ("config", rknn.config(target_platform=target, mean_values=[mean], std_values=[std])),
            ("load_onnx", rknn.load_onnx(model=str(onnx_path))),
            ("build", rknn.build(do_quantization=False)),
            ("init_runtime", rknn.init_runtime()),
        )
        for name, result in steps:
            if result != 0:
                raise RuntimeError(f"{name} failed for {target}: {result}")
        return [
            float(np.asarray(output).reshape(-1)[0])
            for output in rknn.inference(inputs=[image], data_format=["nhwc"])
        ]
    finally:
        rknn.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True)
    parser.add_argument("--age-onnx", type=Path, required=True)
    parser.add_argument("--pose-onnx", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Unable to read image: {args.image}")

    age = img_utils.resize(img_utils.crop(image, args.bbox, 0.45), (64, 64))
    age = np.ascontiguousarray([age], dtype=np.uint8)
    age_onnx_input = np.transpose(age.astype(np.float32), (0, 3, 1, 2))
    age_onnx = float(
        ort.InferenceSession(str(args.age_onnx), providers=["CPUExecutionProvider"])
        .run(None, {"input": age_onnx_input})[0]
        .reshape(-1)[0]
    )

    pose = img_utils.crop(image, args.bbox, 0.6)
    pose = img_utils.resize(img_utils.pad_zero(pose, 1), (224, 224))
    pose = cv2.cvtColor(pose, cv2.COLOR_BGR2RGB)
    pose = np.ascontiguousarray([pose], dtype=np.float32)
    pose_onnx_input = np.transpose(pose, (0, 3, 1, 2))
    pose_onnx_input = (
        pose_onnx_input - np.asarray(POSE_MEAN, dtype=np.float32)[None, :, None, None]
    ) / np.asarray(POSE_STD, dtype=np.float32)[None, :, None, None]
    pose_onnx = [
        float(np.asarray(output).reshape(-1)[0])
        for output in ort.InferenceSession(
            str(args.pose_onnx), providers=["CPUExecutionProvider"]
        ).run(None, {"input": pose_onnx_input})
    ]

    age_rk3566 = infer_simulator(args.age_onnx, "rk3566", age, [0, 0, 0], [1, 1, 1])
    age_rk3588 = infer_simulator(args.age_onnx, "rk3588", age, [0, 0, 0], [1, 1, 1])
    pose_rk3566 = infer_simulator(args.pose_onnx, "rk3566", pose, POSE_MEAN, POSE_STD)
    pose_rk3588 = infer_simulator(args.pose_onnx, "rk3588", pose, POSE_MEAN, POSE_STD)

    results = {
        "image": str(args.image),
        "bbox": args.bbox,
        "age": {
            "margin": 0.45,
            "onnx": age_onnx,
            "rk3566_simulator": age_rk3566[0],
            "rk3588_simulator": age_rk3588[0],
        },
        "pose": {
            "margin": 0.6,
            "output_order": ["roll", "yaw", "pitch"],
            "onnx": pose_onnx,
            "rk3566_simulator": pose_rk3566,
            "rk3588_simulator": pose_rk3588,
        },
    }

    args.artifacts.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.artifacts / "test_face_age_crop.png"), age[0])
    cv2.imwrite(
        str(args.artifacts / "test_face_pose_crop_rgb.png"),
        cv2.cvtColor(pose[0].astype(np.uint8), cv2.COLOR_RGB2BGR),
    )
    (args.artifacts / "age_pose_backend_comparison.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
