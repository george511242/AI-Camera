#!/usr/bin/env python3
"""Export a selected Age v3.2 ordinal checkpoint with scalar output."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ssrnet_age_v32_ordinal import build_model, initial_cutpoints
from train_ssrnet_age_v31 import read_runtime_manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--selected-weights", type=Path, required=True)
    parser.add_argument("--base-split-dir", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=73)
    return parser.parse_args()


def shape(value_info):
    return [dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim]


def main():
    args = parse_args()
    train_rows = [
        json.loads(line)
        for line in (args.base_split_dir / "train.jsonl").read_text().splitlines()
    ]
    min_age = min(row["age"] for row in train_rows)
    max_age = max(row["age"] for row in train_rows)
    args.min_age = min_age
    args.num_thresholds = max_age - min_age
    ordinal_model, _ = build_model(
        args, initial_cutpoints(train_rows, min_age, max_age)
    )
    ordinal_model.load_weights(args.selected_weights)

    head = ordinal_model.get_layer("ordered_ordinal_head")
    score_kernel = head.score_kernel.numpy()
    cutpoints = head.cutpoints().numpy()
    embedding = ordinal_model.get_layer("ordinal_embedding").output
    logits = keras.layers.Dense(
        args.num_thresholds, name="ordinal_logits", trainable=False
    )(embedding)
    probabilities = keras.layers.Activation("sigmoid", name="ordinal_probabilities")(
        logits
    )
    scalar = keras.layers.Lambda(
        lambda value: tf.cast(min_age, value.dtype)
        + tf.reduce_sum(value, axis=-1, keepdims=True),
        name="expected_age",
    )(probabilities)
    deployment_model = keras.Model(ordinal_model.input, scalar)
    deployment_model.get_layer("ordinal_logits").set_weights([
        np.repeat(score_kernel, args.num_thresholds, axis=1), -cutpoints
    ])

    inputs = keras.Input(shape=(3, 64, 64), batch_size=1, name="input")
    outputs = tf.identity(
        deployment_model(tf.transpose(inputs, (0, 2, 3, 1)), training=False),
        name="Identity",
    )
    export_model = keras.Model(inputs, outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(
        export_model,
        input_signature=(tf.TensorSpec((1, 3, 64, 64), tf.float32, name="input"),),
        opset=13,
        output_path=str(args.output),
    )

    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    input_shape = shape(graph.graph.input[0])
    output_shape = shape(graph.graph.output[0])
    if input_shape != [1, 3, 64, 64] or output_shape != [1, 1]:
        raise RuntimeError(f"unexpected ONNX contract: {input_shape} -> {output_shape}")

    rows = read_runtime_manifest(
        args.runtime_manifest, args.cache_root, {"80px", "40px"}
    )
    indexes = np.linspace(0, len(rows) - 1, args.sample_count, dtype=int)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    differences = []
    samples = []
    for index in indexes:
        row = rows[index]
        image = cv2.imread(row["file"], cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
        framework_probabilities = ordinal_model(image[None], training=False).numpy()
        framework_age = float(min_age + framework_probabilities.sum())
        onnx_age = float(session.run(
            None, {input_name: np.transpose(image[None], (0, 3, 1, 2))}
        )[0].ravel()[0])
        difference = abs(framework_age - onnx_age)
        differences.append(difference)
        samples.append({
            "source_id": row["source_id"], "variant": row["variant"],
            "framework": framework_age, "onnx": onnx_age,
            "absolute_difference": difference,
        })
    result = {
        "onnx_checker": "PASS",
        "input_shape": input_shape,
        "output_shape": output_shape,
        "sample_count": len(samples),
        "max_absolute_difference": max(differences),
        "mean_absolute_difference": float(np.mean(differences)),
        "numeric_comparison": "PASS" if max(differences) <= 1e-4 else "FAIL",
        "samples": samples,
    }
    args.verification_output.parent.mkdir(parents=True, exist_ok=True)
    args.verification_output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    print(f"output={args.output} size_bytes={args.output.stat().st_size}")
    if result["numeric_comparison"] != "PASS":
        raise RuntimeError("framework/ONNX comparison failed")


if __name__ == "__main__":
    main()
