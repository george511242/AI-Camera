#!/usr/bin/env python3
"""Train the single Age v3.2 rank-consistent ordinal candidate."""

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_ssrnet_gender_onnx import load_official_module
from train_ssrnet_age_v3 import file_sha256, json_ready
from train_ssrnet_age_v31 import (
    VARIANT_WEIGHTS,
    cached_dataset,
    read_runtime_manifest,
)


AGE_BUCKETS = (
    (0, 9, "0-9"), (10, 19, "10-19"), (20, 29, "20-29"),
    (30, 39, "30-39"), (40, 49, "40-49"), (50, 59, "50-59"),
    (60, 69, "60-69"), (70, 79, "70-79"), (80, 200, "80+"),
)


class OrderedOrdinalHead(keras.layers.Layer):
    """Shared rank score with strictly ordered learned cutpoints."""

    def __init__(self, initial_cutpoints, **kwargs):
        super().__init__(**kwargs)
        self.initial_cutpoints = np.asarray(initial_cutpoints, np.float32)
        self.num_thresholds = len(self.initial_cutpoints)

    def build(self, input_shape):
        self.score_kernel = self.add_weight(
            "score_kernel",
            shape=(int(input_shape[-1]), 1),
            initializer=keras.initializers.RandomNormal(stddev=0.05),
        )
        self.cutpoint_base = self.add_weight(
            "cutpoint_base",
            shape=(1,),
            initializer=keras.initializers.Constant(self.initial_cutpoints[0]),
        )
        gaps = np.maximum(np.diff(self.initial_cutpoints), 1e-5)
        raw_gaps = np.log(np.expm1(gaps))
        self.cutpoint_gaps = self.add_weight(
            "cutpoint_gaps",
            shape=(self.num_thresholds - 1,),
            initializer=keras.initializers.Constant(raw_gaps),
        )

    def cutpoints(self):
        gaps = tf.nn.softplus(self.cutpoint_gaps)
        return self.cutpoint_base + tf.concat(
            [tf.zeros((1,), self.dtype), tf.cumsum(gaps)], axis=0
        )

    def call(self, inputs):
        score = tf.matmul(inputs, self.score_kernel)
        return tf.sigmoid(score - self.cutpoints()[None, :])


def ordinal_targets(ages, min_age, num_thresholds):
    ages = tf.cast(tf.reshape(ages, (-1, 1)), tf.int32)
    thresholds = tf.range(min_age, min_age + num_thresholds)[None, :]
    return tf.cast(ages > thresholds, tf.float32)


def ordinal_dataset(rows, args, training, repeat=False):
    base = cached_dataset(rows, args, training, repeat)
    return base.map(
        lambda image, age: (
            image,
            ordinal_targets(age, args.min_age, args.num_thresholds),
        ),
        num_parallel_calls=args.workers,
        deterministic=True,
    ).prefetch(args.prefetch)


def distance_weighted_ordinal_dataset(rows, args):
    datasets = []
    weights = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        datasets.append(ordinal_dataset(group, args, True, True).unbatch())
        weights.append(VARIANT_WEIGHTS[variant])
    dataset = tf.data.Dataset.sample_from_datasets(
        datasets, weights=weights, seed=args.seed
    )
    return dataset.batch(args.batch_size, drop_remainder=True).prefetch(args.prefetch)


def expected_age(probabilities, min_age):
    return tf.cast(min_age, probabilities.dtype) + tf.reduce_sum(
        probabilities, axis=-1
    )


class ExpectedMAE(keras.metrics.Metric):
    def __init__(self, min_age, name="expected_age_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_age = min_age
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight
        true_age = self.min_age + tf.reduce_sum(y_true, axis=-1)
        error = tf.abs(expected_age(y_pred, self.min_age) - true_age)
        self.total.assign_add(tf.reduce_sum(error))
        self.count.assign_add(tf.cast(tf.size(error), self.dtype))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class ExpectedWithinFive(ExpectedMAE):
    def __init__(self, min_age, name="expected_within_5_years", **kwargs):
        super().__init__(min_age, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight
        true_age = self.min_age + tf.reduce_sum(y_true, axis=-1)
        hits = tf.cast(
            tf.abs(expected_age(y_pred, self.min_age) - true_age) <= 5.0,
            self.dtype,
        )
        self.total.assign_add(tf.reduce_sum(hits))
        self.count.assign_add(tf.cast(tf.size(hits), self.dtype))


def metrics(rows, probabilities, min_age):
    labels = np.asarray([row["age"] for row in rows], np.float32)
    predictions = min_age + probabilities.sum(axis=1)

    def summarize(mask):
        error = predictions[mask] - labels[mask]
        return {
            "count": int(mask.sum()),
            "within_5_years": float(np.mean(np.abs(error) <= 5)),
            "mae": float(np.mean(np.abs(error))),
            "mean_prediction": float(np.mean(predictions[mask])),
            "mean_signed_error": float(np.mean(error)),
        }

    result = summarize(np.ones(len(rows), dtype=bool))
    result["age_buckets"] = {
        name: summarize((labels >= low) & (labels <= high))
        for low, high, name in AGE_BUCKETS
        if np.any((labels >= low) & (labels <= high))
    }
    result["by_variant"] = {}
    for variant in ("80px", "40px"):
        mask = np.asarray([row["variant"] == variant for row in rows])
        result["by_variant"][variant] = summarize(mask)
    violations = probabilities[:, 1:] > probabilities[:, :-1] + 1e-7
    result["monotonicity"] = {
        "sample_violation_rate": float(np.mean(np.any(violations, axis=1))),
        "pair_violation_rate": float(np.mean(violations)),
        "max_violation": float(np.max(
            np.maximum(probabilities[:, 1:] - probabilities[:, :-1], 0.0)
        )),
    }
    return result


def initial_cutpoints(rows, min_age, max_age):
    ages = np.asarray([row["age"] for row in rows])
    probabilities = np.asarray([
        np.clip(np.mean(ages > threshold), 1e-4, 1 - 1e-4)
        for threshold in range(min_age, max_age)
    ])
    return -np.log(probabilities / (1.0 - probabilities))


def build_model(args, cutpoints):
    module = load_official_module(args.source)
    scalar_model = module.SSR_net(64, [3, 3, 3], 1, 1)()
    scalar_model.load_weights(args.initial_weights)
    feature_layer_names = (
        "flatten", "flatten_1", "flatten_2", "flatten_3", "flatten_4", "flatten_5"
    )
    features = keras.layers.Concatenate(name="ordinal_stage_features")(
        [scalar_model.get_layer(name).output for name in feature_layer_names]
    )
    embedding = keras.layers.Dense(
        32,
        activation="relu",
        bias_initializer=keras.initializers.Constant(0.1),
        name="ordinal_embedding",
    )(features)
    probabilities = OrderedOrdinalHead(cutpoints, name="ordered_ordinal_head")(
        embedding
    )
    model = keras.Model(scalar_model.input, probabilities, name="age_v32_ordinal")
    backbone_layers = [layer for layer in scalar_model.layers if layer.name != "pred_a"]
    return model, backbone_layers


def ordinal_loss(min_age, mae_lambda):
    binary_crossentropy = keras.losses.BinaryCrossentropy()

    def loss(y_true, y_pred):
        main = binary_crossentropy(y_true, y_pred)
        true_age = tf.cast(min_age, y_true.dtype) + tf.reduce_sum(y_true, axis=-1)
        auxiliary = tf.reduce_mean(
            tf.abs(expected_age(y_pred, min_age) - true_age)
        )
        return main + mae_lambda * auxiliary

    loss.__name__ = "ordinal_bce_with_expected_age_mae"
    return loss


def compile_model(model, args):
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss=ordinal_loss(args.min_age, args.mae_lambda),
        metrics=[ExpectedWithinFive(args.min_age), ExpectedMAE(args.min_age)],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--base-split-dir", type=Path, required=True)
    parser.add_argument("--runtime-split-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment", choices=("O1", "O2"), default="O1")
    parser.add_argument("--mae-lambda", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.experiment == "O1" and args.mae_lambda != 0.0:
        raise ValueError("O1 requires --mae-lambda 0")
    if args.experiment == "O2" and args.mae_lambda <= 0.0:
        raise ValueError("O2 requires a positive --mae-lambda")
    tf.config.set_visible_devices([], "GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    base_rows = [json.loads(line) for line in
                 (args.base_split_dir / "train.jsonl").read_text().splitlines()]
    validation_base = [json.loads(line) for line in
                       (args.base_split_dir / "validation.jsonl").read_text().splitlines()]
    ages = [row["age"] for row in base_rows]
    args.min_age, max_age = min(ages), max(ages)
    args.num_thresholds = max_age - args.min_age
    train = read_runtime_manifest(
        args.runtime_split_dir / "train_runtime_success.jsonl",
        args.cache_root, {"160px", "80px", "40px"},
    )
    validation = read_runtime_manifest(
        args.runtime_split_dir / "validation_runtime_success.jsonl",
        args.cache_root, {"80px", "40px"},
    )
    model, backbone = build_model(args, initial_cutpoints(base_rows, args.min_age, max_age))
    batch = next(iter(ordinal_dataset(validation[:args.batch_size], args, False)))
    output = model(batch[0], training=False)
    if output.shape != (len(batch[0]), args.num_thresholds):
        raise RuntimeError(f"unexpected ordinal output shape: {output.shape}")
    if tf.reduce_any(output[:, 1:] > output[:, :-1] + 1e-7):
        raise RuntimeError("rank-consistency preflight failed")
    with tf.GradientTape() as tape:
        checked = model(batch[0], training=True)
        checked_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(batch[1], checked)
        )
    gradients = tape.gradient(checked_loss, model.trainable_weights)
    gradient_norms = {
        weight.name: float(tf.linalg.global_norm([gradient]))
        for weight, gradient in zip(model.trainable_weights, gradients)
        if gradient is not None
    }
    head_gradient = gradient_norms.get(
        "ordered_ordinal_head/score_kernel:0", 0.0
    )
    backbone_gradient = max(
        (value for name, value in gradient_norms.items() if name.startswith("conv2d")),
        default=0.0,
    )
    if head_gradient <= 0.0 or backbone_gradient <= 0.0:
        raise RuntimeError(
            f"gradient preflight failed: head={head_gradient} backbone={backbone_gradient}"
        )
    print(
        f"preflight: input={batch[0].shape} output={output.shape} monotonic=PASS "
        f"head_gradient={head_gradient:.6g} backbone_gradient={backbone_gradient:.6g}"
    )
    if args.preflight_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"age_v32_{args.experiment.lower()}"
    best = args.output_dir / f"{prefix}_best.weights.h5"
    latest = args.output_dir / f"{prefix}_latest.weights.h5"
    training = distance_weighted_ordinal_dataset(train, args)
    validation_ds = ordinal_dataset(validation, args, False)
    for layer in backbone:
        layer.trainable = False
    compile_model(model, args)
    warmup = model.fit(
        training, validation_data=validation_ds,
        steps_per_epoch=max(1, len(train) // args.batch_size),
        epochs=args.warmup_epochs, verbose=2,
    )
    for layer in backbone:
        layer.trainable = True
    compile_model(model, args)
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
    age_counts = lambda rows: dict(sorted(Counter(row["age"] for row in rows).items()))
    decade_counts = lambda rows: dict(sorted(
        Counter((row["age"] // 10) * 10 for row in rows).items()
    ))
    run = {
        "experiment": args.experiment,
        "formulation": "rank-consistent ordinal BCE",
        "expected_age_mae_lambda": args.mae_lambda,
        "age_range": [args.min_age, max_age],
        "num_thresholds": args.num_thresholds,
        "thresholds": [args.min_age, max_age - 1],
        "feature_layers": [
            "flatten", "flatten_1", "flatten_2", "flatten_3", "flatten_4", "flatten_5"
        ],
        "sampling": "natural age; 40/35/25 distance variants",
        "variant_weights": VARIANT_WEIGHTS,
        "train_source_age_counts": age_counts(base_rows),
        "validation_source_age_counts": age_counts(validation_base),
        "train_source_decade_counts": decade_counts(base_rows),
        "validation_source_decade_counts": decade_counts(validation_base),
        "train_runtime_count": len(train),
        "validation_runtime_count": len(validation),
        "warmup_epochs": args.warmup_epochs,
        "epochs_completed": len(history.epoch),
        "elapsed_seconds": elapsed,
        "selected_validation": metrics(validation, probabilities, args.min_age),
        "best_weights_sha256": file_sha256(best),
        "latest_weights_sha256": file_sha256(latest),
        "tensorflow": tf.__version__,
        "device": "cpu",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(json_ready(vars(args)), indent=2, default=str, sort_keys=True) + "\n"
    )
    (args.output_dir / "warmup_history.json").write_text(
        json.dumps(json_ready(warmup.history), indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(json_ready(history.history), indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "training_run.json").write_text(
        json.dumps(json_ready(run), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(run, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
