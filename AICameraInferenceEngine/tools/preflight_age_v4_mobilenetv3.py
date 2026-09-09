#!/usr/bin/env python3
"""Export the untrained Age v4 deployment graph for RKNN preflight."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from tensorflow import keras


MIN_AGE = 1
MAX_AGE = 99
NUM_THRESHOLDS = MAX_AGE - MIN_AGE


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shape(value_info):
    return [dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim]


def build_deployment_model():
    image = keras.Input((112, 112, 3), batch_size=1, name="rgb_image_0_255")
    backbone = keras.applications.MobileNetV3Large(
        input_shape=(112, 112, 3),
        alpha=1.0,
        minimalistic=False,
        include_top=False,
        weights=None,
        include_preprocessing=True,
        pooling="avg",
    )
    features = backbone(image)
    embedding = keras.layers.Dense(128, activation="relu", name="ordinal_embedding")(
        features
    )

    # The trained rank-consistent head is exported as its exact standard Dense
    # equivalent: one shared score copied over all thresholds minus cutpoints.
    logits = keras.layers.Dense(NUM_THRESHOLDS, name="ordinal_logits")(embedding)
    probabilities = keras.layers.Activation("sigmoid", name="ordinal_probabilities")(
        logits
    )
    age = keras.layers.Lambda(
        lambda value: tf.cast(MIN_AGE, value.dtype)
        + tf.reduce_sum(value, axis=-1, keepdims=True),
        name="expected_age",
    )(probabilities)
    model = keras.Model(image, age, name="age_v4_mobilenetv3_large")

    rng = np.random.default_rng(20260905)
    shared_score = rng.normal(0.0, 0.05, (128, 1)).astype(np.float32)
    cutpoints = np.linspace(-4.0, 4.0, NUM_THRESHOLDS, dtype=np.float32)
    model.get_layer("ordinal_logits").set_weights([
        np.repeat(shared_score, NUM_THRESHOLDS, axis=1), -cutpoints
    ])
    return model, backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(20260905)
    model, backbone = build_deployment_model()
    nchw = keras.Input((3, 112, 112), batch_size=1, name="input")
    output = tf.identity(
        model(tf.transpose(nchw, (0, 2, 3, 1)), training=False), name="output"
    )
    export_model = keras.Model(nchw, output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(
        export_model,
        input_signature=(
            tf.TensorSpec((1, 3, 112, 112), tf.float32, name="input"),
        ),
        opset=args.opset,
        output_path=str(args.output),
    )
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    input_shape = tensor_shape(graph.graph.input[0])
    output_shape = tensor_shape(graph.graph.output[0])
    if input_shape != [1, 3, 112, 112] or output_shape != [1, 1]:
        raise RuntimeError(f"unexpected contract: {input_shape} -> {output_shape}")

    sample = np.random.default_rng(20260905).uniform(
        0.0, 255.0, (1, 3, 112, 112)
    ).astype(np.float32)
    framework = export_model(sample, training=False).numpy()
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    runtime = session.run(None, {session.get_inputs()[0].name: sample})[0]
    difference = np.abs(framework - runtime)
    metadata = {
        "status": "PASS",
        "purpose": "untrained dual-platform RKNN compatibility preflight",
        "framework": f"TensorFlow {tf.__version__}",
        "implementation": "tf.keras.applications.MobileNetV3Large alpha=1.0",
        "pretrained_weights": None,
        "input_name": graph.graph.input[0].name,
        "input_shape": input_shape,
        "input_semantics": "NCHW RGB float32 0..255; built-in preprocessing to -1..1",
        "output_name": graph.graph.output[0].name,
        "output_shape": output_shape,
        "output_semantics": "1 + sum(sigmoid(98 ordered ordinal logits))",
        "opset": args.opset,
        "parameter_count": export_model.count_params(),
        "backbone_parameter_count": backbone.count_params(),
        "onnx_size_bytes": args.output.stat().st_size,
        "onnx_sha256": sha256(args.output),
        "onnx_checker": "PASS",
        "framework_onnx_max_absolute_difference": float(difference.max()),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
