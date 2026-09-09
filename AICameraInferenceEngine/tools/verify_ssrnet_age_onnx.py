#!/usr/bin/env python3
"""Verify an exported SSR-Net Age ONNX contract and numeric agreement."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from export_ssrnet_gender_onnx import load_official_module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-bin", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-4)
    return parser.parse_args()


def prepare_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode image: {path}")
    image = cv2.resize(image, (34, 34), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, 15, 15, 15, 15, cv2.BORDER_CONSTANT)
    return image.astype(np.float32)


def shape(value_info):
    return [dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim]


def main():
    args = parse_args()
    graph = onnx.load(args.onnx)
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise RuntimeError("expected exactly one ONNX input and one output")
    input_shape = shape(graph.graph.input[0])
    output_shape = shape(graph.graph.output[0])
    if input_shape != [1, 3, 64, 64] or output_shape != [1, 1]:
        raise RuntimeError(f"unexpected ONNX shapes: {input_shape} -> {output_shape}")

    by_bin = defaultdict(list)
    with args.manifest.open() as handle:
        for line in handle:
            row = json.loads(line)
            by_bin[row["age_bin"]].append(row)
    selected = []
    for age_bin in sorted(by_bin):
        selected.extend(sorted(by_bin[age_bin], key=lambda row: row["source_id"])[
            : args.samples_per_bin
        ])

    import tensorflow as tf

    model = load_official_module(args.source).SSR_net(64, [3, 3, 3], 1, 1)()
    model.load_weights(args.weights)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    differences = []
    samples = []
    for row in selected:
        image = prepare_image(args.image_root / row["file"])
        tf_value = float(np.ravel(model(image[None], training=False).numpy())[0])
        nchw = np.transpose(image[None], (0, 3, 1, 2))
        onnx_value = float(np.ravel(session.run(None, {input_name: nchw})[0])[0])
        difference = abs(tf_value - onnx_value)
        differences.append(difference)
        samples.append(
            {
                "source_id": row["source_id"],
                "age": row["age"],
                "tensorflow": tf_value,
                "onnx": onnx_value,
                "absolute_difference": difference,
            }
        )

    result = {
        "onnx_checker": "PASS",
        "input": {"name": graph.graph.input[0].name, "shape": input_shape},
        "output": {"name": graph.graph.output[0].name, "shape": output_shape},
        "sample_count": len(samples),
        "max_absolute_difference": max(differences),
        "mean_absolute_difference": float(np.mean(differences)),
        "tolerance": args.atol,
        "numeric_comparison": "PASS" if max(differences) <= args.atol else "FAIL",
        "samples": samples,
        "tensorflow_version": tf.__version__,
        "onnxruntime_version": ort.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    if result["numeric_comparison"] != "PASS":
        raise RuntimeError("TensorFlow/ONNX numeric comparison exceeded tolerance")


if __name__ == "__main__":
    main()
