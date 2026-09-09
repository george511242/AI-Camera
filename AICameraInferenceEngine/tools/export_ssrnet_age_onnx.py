#!/usr/bin/env python3
"""Export the official SSR-Net Wiki age checkpoint to ONNX."""

import argparse
from pathlib import Path

import tensorflow as tf
import tf2onnx
from tensorflow import keras

from export_ssrnet_gender_onnx import load_official_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    module = load_official_module(args.source)
    model = module.SSR_net(64, [3, 3, 3], 1, 1)()
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
