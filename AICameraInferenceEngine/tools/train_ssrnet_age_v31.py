#!/usr/bin/env python3
"""Controlled Age v3.1 runtime-crop ablation training."""

import argparse
import json
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
from train_ssrnet_age_v3 import (
    WithinYears,
    dataset_for_rows as v2_dataset_for_rows,
    file_sha256,
    json_ready,
    make_training_dataset as v2_training_dataset,
    metrics_from_predictions,
    random_crop_jitter,
    read_manifest,
)


VARIANT_WEIGHTS = {"160px": 0.40, "80px": 0.35, "40px": 0.25}


def read_runtime_manifest(path, cache_root, variants):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    selected = []
    for row in rows:
        if row["variant"] not in variants:
            continue
        row["file"] = str((cache_root / row["file"]).resolve())
        selected.append(row)
    return selected


def decode_cached(path, age, training, jitter):
    image = tf.io.decode_png(tf.io.read_file(path), channels=3)
    image = tf.reverse(image, axis=[-1])
    image = tf.cast(image, tf.float32)
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=18.0)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.clip_by_value(image, 0.0, 255.0)
        if jitter:
            image = random_crop_jitter(image)
    image = tf.image.resize(image, (64, 64))
    image.set_shape((64, 64, 3))
    return image, tf.reshape(tf.cast(age, tf.float32), (1,))


def cached_dataset(rows, args, training, repeat=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        ([row["file"] for row in rows], [row["age"] for row in rows])
    )
    if training:
        dataset = dataset.shuffle(len(rows), seed=args.seed, reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat()
    options = tf.data.Options()
    options.deterministic = True
    dataset = dataset.with_options(options).map(
        lambda path, age: decode_cached(
            path, age, training, args.experiment == "B2"
        ),
        num_parallel_calls=args.workers,
        deterministic=True,
    )
    return dataset.batch(args.batch_size).prefetch(args.prefetch)


def balanced_cached_dataset(rows, args, distance_weighted=False):
    datasets, weights = [], []
    bins = sorted({row["age_bin"] for row in rows})
    variants = sorted({row["variant"] for row in rows})
    for age_bin in bins:
        for variant in variants:
            group = [
                row for row in rows
                if row["age_bin"] == age_bin and row["variant"] == variant
            ]
            if not group:
                continue
            datasets.append(cached_dataset(group, args, True, True).unbatch())
            variant_weight = VARIANT_WEIGHTS[variant] if distance_weighted else 1.0
            weights.append(variant_weight / len(bins))
    weights = np.asarray(weights, np.float64)
    weights /= weights.sum()
    dataset = tf.data.Dataset.sample_from_datasets(
        datasets, weights=weights.tolist(), seed=args.seed
    )
    return dataset.batch(args.batch_size, drop_remainder=True).prefetch(args.prefetch)


def distance_weighted_cached_dataset(rows, args):
    datasets, weights = [], []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        datasets.append(cached_dataset(group, args, True, True).unbatch())
        weights.append(VARIANT_WEIGHTS[variant])
    dataset = tf.data.Dataset.sample_from_datasets(
        datasets, weights=weights, seed=args.seed
    )
    return dataset.batch(args.batch_size, drop_remainder=True).prefetch(args.prefetch)


def predict_metrics(model, rows, args):
    predictions = np.ravel(model.predict(cached_dataset(rows, args, False), verbose=0))
    result = metrics_from_predictions(rows, predictions)
    result["by_variant"] = {}
    for variant in sorted({row["variant"] for row in rows}):
        indexes = [index for index, row in enumerate(rows) if row["variant"] == variant]
        subset = [rows[index] for index in indexes]
        result["by_variant"][variant] = metrics_from_predictions(
            subset, predictions[indexes]
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", choices=("B0", "B1", "B2", "B3", "B4", "B5"), required=True
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--base-split-dir", type=Path, required=True)
    parser.add_argument("--runtime-split-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    tf.config.set_visible_devices([], "GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    validation = read_runtime_manifest(
        args.runtime_split_dir / "validation_runtime_success.jsonl",
        args.cache_root,
        {"80px", "40px"},
    )
    test = read_runtime_manifest(
        args.runtime_split_dir / "test_runtime_success.jsonl",
        args.cache_root,
        {"80px", "40px"},
    )
    if args.experiment == "B0":
        train = read_manifest(args.base_split_dir / "train.jsonl", args.image_root)
        v2_args = argparse.Namespace(
            batch_size=args.batch_size,
            distance_augmentation="v2",
            camera_augmentation="v2",
            bbox_jitter="none",
            workers=args.workers,
            prefetch=args.prefetch,
            seed=args.seed,
            age_balance="uniform-decade",
        )
        training, _ = v2_training_dataset(train, v2_args)
        training_input = "v2 resize 24..42 then pad 64"
        sampling = "uniform decade"
    else:
        variants = (
            {"160px", "80px", "40px"}
            if args.experiment in {"B3", "B5"}
            else {"160px"}
        )
        train = read_runtime_manifest(
            args.runtime_split_dir / "train_runtime_success.jsonl",
            args.cache_root,
            variants,
        )
        if args.experiment == "B4":
            training = cached_dataset(train, args, True, True)
            sampling = "natural"
        elif args.experiment == "B5":
            training = distance_weighted_cached_dataset(train, args)
            sampling = "natural age; 40/35/25 distance variants"
        else:
            training = balanced_cached_dataset(
                train, args, distance_weighted=args.experiment == "B3"
            )
            sampling = "uniform decade"
        training_input = "+".join(sorted(variants)) + " runtime crops"

    model = load_official_module(args.source).SSR_net(64, [3, 3, 3], 1, 1)()
    model.load_weights(args.initial_weights)
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="mae",
        metrics=[WithinYears()],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = args.output_dir / "age_v31_best.weights.h5"
    latest = args.output_dir / "age_v31_latest.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best, monitor="val_within_5_years", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(latest, save_weights_only=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1,
        ),
    ]
    initial_validation = predict_metrics(model, validation, args)
    steps = max(1, len(train) // args.batch_size)
    started = time.time()
    history = model.fit(
        training,
        validation_data=cached_dataset(validation, args, False),
        steps_per_epoch=steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - started
    model.load_weights(best)
    run = {
        "experiment": args.experiment,
        "seed": args.seed,
        "training_input": training_input,
        "sampling": sampling,
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "validation_variants": dict(Counter(row["variant"] for row in validation)),
        "initial_validation": initial_validation,
        "selected_validation": predict_metrics(model, validation, args),
        "selected_test": predict_metrics(model, test, args),
        "epochs_completed": len(history.epoch),
        "steps_per_epoch": steps,
        "elapsed_seconds": elapsed,
        "best_weights_sha256": file_sha256(best),
        "latest_weights_sha256": file_sha256(latest),
        "tensorflow": tf.__version__,
        "device": "cpu",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(json_ready(vars(args)), indent=2, default=str, sort_keys=True) + "\n"
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(json_ready(history.history), indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "training_run.json").write_text(
        json.dumps(json_ready(run), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(json_ready(run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
