#!/usr/bin/env python3
"""Export the official SSR-Net gender checkpoint to ONNX."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import tensorflow as tf
import tf2onnx
import keras as standalone_keras
from tensorflow import keras


def load_official_module(source_path: Path):
    """Load the legacy Keras source with compatibility aliases for Keras 2.15."""
    convolutional = types.ModuleType("keras.layers.convolutional")
    convolutional.Conv2D = keras.layers.Conv2D
    convolutional.AveragePooling2D = keras.layers.AveragePooling2D
    convolutional.MaxPooling2D = keras.layers.MaxPooling2D
    sys.modules[convolutional.__name__] = convolutional

    normalization = types.ModuleType("keras.layers.normalization")
    normalization.BatchNormalization = keras.layers.BatchNormalization
    sys.modules[normalization.__name__] = normalization

    topology = types.ModuleType("keras.engine.topology")
    topology.Layer = keras.layers.Layer
    sys.modules["keras.engine"] = types.ModuleType("keras.engine")
    sys.modules[topology.__name__] = topology

    standalone_keras.backend.image_dim_ordering = lambda: "tf"

    spec = importlib.util.spec_from_file_location("official_ssrnet_model", source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    module = load_official_module(args.source)
    model = module.SSR_net_general(64, [3, 3, 3], 1, 1)()
    model.load_weights(args.weights)

    inputs = keras.Input(shape=(3, 64, 64), batch_size=1, name="input")
    nhwc_inputs = tf.transpose(inputs, (0, 2, 3, 1))
    outputs = tf.identity(model(nhwc_inputs, training=False), name="Identity")
    export_model = keras.Model(inputs=inputs, outputs=outputs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    signature = (tf.TensorSpec((1, 3, 64, 64), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(
        export_model,
        input_signature=signature,
        opset=13,
        output_path=str(args.output),
    )
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
