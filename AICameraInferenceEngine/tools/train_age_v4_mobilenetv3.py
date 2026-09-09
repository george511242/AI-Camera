#!/usr/bin/env python3
"""Train the single Age v4 MobileNetV3-Large ordinal candidate."""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from train_ssrnet_age_v3 import file_sha256, json_ready
from train_ssrnet_age_v31 import VARIANT_WEIGHTS, read_runtime_manifest
from train_ssrnet_age_v32_ordinal import (
    AGE_BUCKETS,
    ExpectedMAE,
    ExpectedWithinFive,
    OrderedOrdinalHead,
    expected_age,
    initial_cutpoints,
    metrics,
    ordinal_targets,
)


def decode_cached(path, age, training):
    # cv2-written cache files decode as RGB. MobileNetV3 receives RGB 0..255;
    # its built-in Rescaling layer performs the ImageNet -1..1 transform.
    image = tf.cast(tf.io.decode_png(tf.io.read_file(path), channels=3), tf.float32)
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=18.0)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.clip_by_value(image, 0.0, 255.0)
    image = tf.image.resize(image, (112, 112), method="bilinear")
    image.set_shape((112, 112, 3))
    target = tf.reshape(ordinal_targets(age, 1, 98), (98,))
    return image, target


def dataset(rows, args, training, repeat=False):
    value = tf.data.Dataset.from_tensor_slices(
        ([row["file"] for row in rows], [row["age"] for row in rows])
    )
    if training:
        value = value.shuffle(len(rows), seed=args.seed, reshuffle_each_iteration=True)
    if repeat:
        value = value.repeat()
    options = tf.data.Options()
    options.deterministic = True
    value = value.with_options(options).map(
        lambda path, age: decode_cached(path, age, training),
        num_parallel_calls=args.workers,
        deterministic=True,
    )
    return value.batch(args.batch_size, drop_remainder=training).prefetch(args.prefetch)


def weighted_training_dataset(rows, args):
    values, weights = [], []
    for variant in sorted(VARIANT_WEIGHTS):
        group = [row for row in rows if row["variant"] == variant]
        values.append(dataset(group, args, True, True).unbatch())
        weights.append(VARIANT_WEIGHTS[variant])
    return tf.data.Dataset.sample_from_datasets(
        values, weights=weights, seed=args.seed
    ).batch(args.batch_size, drop_remainder=True).prefetch(args.prefetch)


def build_model(cutpoints, weights):
    image = keras.Input((112, 112, 3), name="rgb_image_0_255")
    backbone = keras.applications.MobileNetV3Large(
        input_shape=(112, 112, 3), alpha=1.0, minimalistic=False,
        include_top=False, weights=weights, include_preprocessing=True,
        pooling="avg",
    )
    features = backbone(image)
    embedding = keras.layers.Dense(
        128, activation="relu", bias_initializer="zeros", name="ordinal_embedding"
    )(features)
    probabilities = OrderedOrdinalHead(
        cutpoints, name="ordered_ordinal_head"
    )(embedding)
    return keras.Model(image, probabilities, name="age_v4_mobilenetv3_large"), backbone


def ordinal_bce(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred)


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=ordinal_bce,
        metrics=[ExpectedWithinFive(1), ExpectedMAE(1)],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-split-dir", type=Path, required=True)
    parser.add_argument("--runtime-split-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-weights", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    base_train = [json.loads(line) for line in
                  (args.base_split_dir / "train.jsonl").read_text().splitlines()]
    if (min(row["age"] for row in base_train), max(row["age"] for row in base_train)) != (1, 99):
        raise RuntimeError("Age v4 requires the frozen 1..99 training contract")
    train = read_runtime_manifest(
        args.runtime_split_dir / "train_runtime_success.jsonl", args.cache_root,
        {"160px", "80px", "40px"},
    )
    validation = read_runtime_manifest(
        args.runtime_split_dir / "validation_runtime_success.jsonl", args.cache_root,
        {"80px", "40px"},
    )
    if not args.pretrained_weights.is_file():
        raise FileNotFoundError(args.pretrained_weights)
    model, backbone = build_model(
        initial_cutpoints(base_train, 1, 99), str(args.pretrained_weights)
    )
    validation_ds = dataset(validation, args, False)
    smoke_images, smoke_targets = next(iter(validation_ds))
    with tf.GradientTape() as tape:
        smoke_predictions = model(smoke_images, training=True)
        smoke_loss = tf.reduce_mean(ordinal_bce(smoke_targets, smoke_predictions))
    gradients = tape.gradient(smoke_loss, model.trainable_weights)
    gradient_norms = {
        weight.name: float(tf.linalg.global_norm([gradient]))
        for weight, gradient in zip(model.trainable_weights, gradients)
        if gradient is not None
    }
    head_gradient = gradient_norms.get("ordered_ordinal_head/score_kernel:0", 0.0)
    backbone_gradient = max(
        (value for name, value in gradient_norms.items()
         if "Conv/Kernel" in name or "expanded_conv" in name), default=0.0
    )
    if smoke_predictions.shape != (len(smoke_images), 98):
        raise RuntimeError(f"unexpected output shape {smoke_predictions.shape}")
    if head_gradient <= 0 or backbone_gradient <= 0:
        raise RuntimeError(
            f"gradient smoke failed: head={head_gradient}, backbone={backbone_gradient}"
        )
    print(
        f"smoke PASS input={smoke_images.shape} output={smoke_predictions.shape} "
        f"loss={float(smoke_loss):.6f} head_gradient={head_gradient:.6g} "
        f"backbone_gradient={backbone_gradient:.6g}"
    )
    if args.preflight_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = args.output_dir / "age_v4_best.weights.h5"
    latest = args.output_dir / "age_v4_latest.weights.h5"
    training = weighted_training_dataset(train, args)
    backbone.trainable = False
    compile_model(model, args.warmup_learning_rate)
    warmup = model.fit(
        training, validation_data=validation_ds,
        steps_per_epoch=max(1, len(train) // args.batch_size),
        epochs=args.warmup_epochs, verbose=2,
    )
    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, args.learning_rate)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best, monitor="val_expected_within_5_years", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(latest, save_weights_only=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_expected_age_mae", factor=0.5, patience=2,
            min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_expected_within_5_years", mode="max", patience=5,
            restore_best_weights=False, verbose=1,
        ),
    ]
    started = time.time()
    history = model.fit(
        training, validation_data=validation_ds,
        steps_per_epoch=max(1, len(train) // args.batch_size),
        epochs=args.epochs, callbacks=callbacks, verbose=2,
    )
    elapsed = time.time() - started
    model.load_weights(best)
    probabilities = model.predict(validation_ds, verbose=0)
    run = {
        "architecture": "MobileNetV3-Large alpha=1.0, 112x112",
        "initialization": "Keras ImageNet no-top weights",
        "pretrained_weights": str(args.pretrained_weights),
        "pretrained_weights_sha256": file_sha256(args.pretrained_weights),
        "objective": "rank-consistent ordinal BCE",
        "input": "RGB float32 0..255; built-in MobileNetV3 -1..1 preprocessing",
        "sampling": "natural age; 40/35/25 distance variants",
        "variant_weights": VARIANT_WEIGHTS,
        "seed": args.seed,
        "train_runtime_count": len(train),
        "validation_runtime_count": len(validation),
        "train_variants": dict(Counter(row["variant"] for row in train)),
        "selected_validation": metrics(validation, probabilities, 1),
        "epochs_completed": len(history.epoch),
        "elapsed_seconds": elapsed,
        "best_weights_sha256": file_sha256(best),
        "latest_weights_sha256": file_sha256(latest),
        "parameter_count": model.count_params(),
        "tensorflow": tf.__version__,
        "visible_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
    }
    for name, payload in (
        ("config.json", vars(args)), ("warmup_history.json", warmup.history),
        ("history.json", history.history), ("training_run.json", run),
    ):
        (args.output_dir / name).write_text(
            json.dumps(json_ready(payload), indent=2, default=str, sort_keys=True) + "\n"
        )
    print(json.dumps(json_ready(run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
