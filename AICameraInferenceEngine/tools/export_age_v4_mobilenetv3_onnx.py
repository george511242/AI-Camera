#!/usr/bin/env python3
"""Export the selected Age v4 checkpoint with scalar deployment output."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from tensorflow import keras

from train_age_v4_mobilenetv3 import build_model
from train_ssrnet_age_v31 import read_runtime_manifest
from train_ssrnet_age_v32_ordinal import initial_cutpoints


def tensor_shape(value_info):
    return [dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-split-dir", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--selected-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=73)
    args = parser.parse_args()

    train_rows = [json.loads(line) for line in
                  (args.base_split_dir / "train.jsonl").read_text().splitlines()]
    ordinal_model, _ = build_model(
        initial_cutpoints(train_rows, 1, 99), str(args.pretrained_weights)
    )
    ordinal_model.load_weights(args.selected_weights)
    head = ordinal_model.get_layer("ordered_ordinal_head")
    embedding = ordinal_model.get_layer("ordinal_embedding").output
    logits = keras.layers.Dense(98, name="ordinal_logits", trainable=False)(embedding)
    probabilities = keras.layers.Activation("sigmoid", name="ordinal_probabilities")(
        logits
    )
    scalar = keras.layers.Lambda(
        lambda value: 1.0 + tf.reduce_sum(value, axis=-1, keepdims=True),
        name="expected_age",
    )(probabilities)
    deployment = keras.Model(ordinal_model.input, scalar)
    deployment.get_layer("ordinal_logits").set_weights([
        np.repeat(head.score_kernel.numpy(), 98, axis=1), -head.cutpoints().numpy()
    ])

    nchw = keras.Input((3, 112, 112), batch_size=1, name="input")
    output = tf.identity(
        deployment(tf.transpose(nchw, (0, 2, 3, 1)), training=False), name="output"
    )
    export_model = keras.Model(nchw, output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(
        export_model,
        input_signature=(tf.TensorSpec((1, 3, 112, 112), tf.float32, name="input"),),
        opset=13,
        output_path=str(args.output),
    )
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    input_shape = tensor_shape(graph.graph.input[0])
    output_shape = tensor_shape(graph.graph.output[0])
    if input_shape != [1, 3, 112, 112] or output_shape != [1, 1]:
        raise RuntimeError(f"unexpected ONNX contract {input_shape} -> {output_shape}")

    rows = read_runtime_manifest(args.runtime_manifest, args.cache_root, {"80px", "40px"})
    indexes = np.linspace(0, len(rows) - 1, args.sample_count, dtype=int)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    differences = []
    samples = []
    for index in indexes:
        row = rows[index]
        bgr = cv2.imread(row["file"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        framework_age = float(1.0 + ordinal_model(rgb[None], training=False).numpy().sum())
        onnx_age = float(session.run(
            None, {input_name: np.transpose(rgb[None], (0, 3, 1, 2))}
        )[0].ravel()[0])
        difference = abs(framework_age - onnx_age)
        differences.append(difference)
        samples.append({
            "source_id": row["source_id"], "variant": row["variant"],
            "framework": framework_age, "onnx": onnx_age,
            "absolute_difference": difference,
        })
    result = {
        "onnx_checker": "PASS", "opset": 13,
        "input_name": graph.graph.input[0].name, "input_shape": input_shape,
        "output_name": graph.graph.output[0].name, "output_shape": output_shape,
        "sample_count": len(samples),
        "max_absolute_difference": max(differences),
        "mean_absolute_difference": float(np.mean(differences)),
        "numeric_comparison": "PASS" if max(differences) <= 1e-3 else "FAIL",
        "samples": samples,
    }
    args.verification_output.parent.mkdir(parents=True, exist_ok=True)
    args.verification_output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    print(f"output={args.output} size_bytes={args.output.stat().st_size}")
    if result["numeric_comparison"] != "PASS":
        raise RuntimeError("framework/ONNX numerical comparison failed")


if __name__ == "__main__":
    main()
