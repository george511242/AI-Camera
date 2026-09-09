#!/usr/bin/env python3
"""Configurable Age v3 fine-tuning for the deployable SSR-Net model."""

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from export_ssrnet_gender_onnx import load_official_module


AGE_BUCKETS = (
    (0, 9, "0-9"),
    (10, 19, "10-19"),
    (20, 29, "20-29"),
    (30, 39, "30-39"),
    (40, 49, "40-49"),
    (50, 59, "50-59"),
    (60, 69, "60-69"),
    (70, 116, "70+"),
)


class WithinYears(keras.metrics.Metric):
    def __init__(self, years=5.0, name="within_5_years", **kwargs):
        super().__init__(name=name, **kwargs)
        self.years = years
        self.hits = self.add_weight(name="hits", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight
        errors = tf.abs(tf.reshape(y_true, (-1,)) - tf.reshape(y_pred, (-1,)))
        self.hits.assign_add(
            tf.reduce_sum(tf.cast(errors <= self.years, self.dtype))
        )
        self.count.assign_add(tf.cast(tf.size(errors), self.dtype))

    def result(self):
        return tf.math.divide_no_nan(self.hits, self.count)

    def reset_state(self):
        self.hits.assign(0)
        self.count.assign(0)


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path, image_root: Path):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        row["file"] = str((image_root / row["file"]).resolve())
    return rows


def random_crop_jitter(image):
    scale_y = tf.random.uniform((), 0.90, 1.0)
    scale_x = tf.random.uniform((), 0.90, 1.0)
    shift_y = tf.random.uniform((), -0.05, 0.05)
    shift_x = tf.random.uniform((), -0.05, 0.05)
    center_y = tf.clip_by_value(0.5 + shift_y, scale_y / 2, 1 - scale_y / 2)
    center_x = tf.clip_by_value(0.5 + shift_x, scale_x / 2, 1 - scale_x / 2)
    box = [[
        center_y - scale_y / 2,
        center_x - scale_x / 2,
        center_y + scale_y / 2,
        center_x + scale_x / 2,
    ]]
    return tf.image.crop_and_resize(image[None], box, [0], (64, 64))[0]


def apply_distance_augmentation(image, mode, training):
    if not training:
        size = tf.constant(34, tf.int32)
        image = tf.image.resize(image, (size, size), antialias=True)
        return tf.image.resize_with_crop_or_pad(image, 64, 64)
    if mode == "v2":
        size = tf.random.uniform((), 24, 43, dtype=tf.int32)
        image = tf.image.resize(image, (size, size), antialias=True)
        return tf.image.resize_with_crop_or_pad(image, 64, 64)
    if mode == "mixed-80-40":
        draw = tf.random.uniform(())
        # Clean 40%, moderate 80px-like degradation 35%, 40px-like 25%.
        size = tf.where(draw < 0.40, 64, tf.where(draw < 0.75, 48, 24))
        image = tf.image.resize(image, (size, size), antialias=True)
        return tf.image.resize(image, (64, 64), method="bilinear", antialias=True)
    return tf.image.resize(image, (64, 64), antialias=True)


def apply_camera_augmentation(image):
    image = tf.image.random_brightness(image, max_delta=12.0)
    image = tf.image.random_contrast(image, lower=0.90, upper=1.10)

    def blur():
        kernel = tf.constant([1.0, 2.0, 1.0], tf.float32)
        kernel = tf.tensordot(kernel, kernel, axes=0)
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[:, :, None, None], [1, 1, 3, 1])
        return tf.nn.depthwise_conv2d(
            image[None], kernel, strides=[1, 1, 1, 1], padding="SAME"
        )[0]

    image = tf.cond(tf.random.uniform(()) < 0.20, blur, lambda: image)
    noise_std = tf.random.uniform((), 0.0, 3.0)
    image = image + tf.random.normal(tf.shape(image), stddev=noise_std)

    def jpeg_roundtrip():
        encoded = tf.io.encode_jpeg(
            tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8), quality=80
        )
        return tf.cast(tf.io.decode_jpeg(encoded, channels=3), tf.float32)

    image = tf.cond(tf.random.uniform(()) < 0.20, jpeg_roundtrip, lambda: image)
    return tf.clip_by_value(image, 0.0, 255.0)


def decode_image(path, age, training, distance_mode, camera_mode, bbox_jitter):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.reverse(image, axis=[-1])
    image = tf.cast(image, tf.float32)
    if training:
        image = tf.image.random_flip_left_right(image)
        if bbox_jitter == "moderate":
            image = random_crop_jitter(image)
    image = apply_distance_augmentation(image, distance_mode, training)
    if training and camera_mode == "v2":
        image = tf.image.random_brightness(image, max_delta=18.0)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.clip_by_value(image, 0.0, 255.0)
    elif training and camera_mode == "mild":
        image = apply_camera_augmentation(image)
    image.set_shape((64, 64, 3))
    return image, tf.reshape(tf.cast(age, tf.float32), (1,))


def dataset_for_rows(rows, args, training, seed=0, repeat=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        ([row["file"] for row in rows], [row["age"] for row in rows])
    )
    if training:
        dataset = dataset.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat()
    options = tf.data.Options()
    options.deterministic = True
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        lambda path, age: decode_image(
            path,
            age,
            training,
            args.distance_augmentation,
            args.camera_augmentation,
            args.bbox_jitter,
        ),
        num_parallel_calls=args.workers,
        deterministic=True,
    )
    return dataset.batch(args.batch_size).prefetch(args.prefetch)


def make_training_dataset(rows, args):
    bins = sorted({row["age_bin"] for row in rows})
    datasets = []
    counts = []
    for age_bin in bins:
        bin_rows = [row for row in rows if row["age_bin"] == age_bin]
        counts.append(len(bin_rows))
        datasets.append(
            dataset_for_rows(bin_rows, args, True, args.seed + age_bin, True).unbatch()
        )
    if args.age_balance == "uniform-decade":
        weights = np.ones(len(bins), dtype=np.float64)
    else:
        # Per-sample weight is 1/sqrt(bin count), so bin probability is sqrt(count).
        weights = np.sqrt(np.asarray(counts, dtype=np.float64))
    weights /= weights.sum()
    dataset = tf.data.Dataset.sample_from_datasets(
        datasets, weights=weights.tolist(), seed=args.seed
    )
    dataset = dataset.batch(args.batch_size, drop_remainder=True)
    return dataset.prefetch(args.prefetch), dict(zip(map(str, bins), weights.tolist()))


def metrics_from_predictions(rows, predictions):
    labels = np.asarray([row["age"] for row in rows], dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    signed = predictions - labels

    def summarize(mask):
        errors = signed[mask]
        if not len(errors):
            return {"count": 0}
        return {
            "count": int(mask.sum()),
            "mae": float(np.abs(errors).mean()),
            "within_3_years": float((np.abs(errors) <= 3).mean()),
            "within_5_years": float((np.abs(errors) <= 5).mean()),
            "within_10_years": float((np.abs(errors) <= 10).mean()),
            "mean_prediction": float(predictions[mask].mean()),
            "mean_signed_error": float(errors.mean()),
        }

    result = summarize(np.ones(len(rows), dtype=bool))
    result["age_buckets"] = {}
    for low, high, label in AGE_BUCKETS:
        mask = (labels >= low) & (labels <= high)
        result["age_buckets"][label] = summarize(mask)
    return result


def evaluate(model, rows, args):
    dataset = dataset_for_rows(rows, args, False)
    predictions = np.ravel(model.predict(dataset, verbose=0))
    return metrics_from_predictions(rows, predictions)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--age-balance",
        choices=("uniform-decade", "sqrt-inverse-10y"),
        default="uniform-decade",
    )
    parser.add_argument("--loss", choices=("mae", "smooth-l1"), default="mae")
    parser.add_argument(
        "--distance-augmentation",
        choices=("none", "v2", "mixed-80-40"),
        default="v2",
    )
    parser.add_argument(
        "--camera-augmentation", choices=("none", "v2", "mild"), default="v2"
    )
    parser.add_argument(
        "--bbox-jitter", choices=("none", "moderate"), default="none"
    )
    parser.add_argument("--huber-delta", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "gpu" and not gpus:
        raise RuntimeError("--device gpu requested, but TensorFlow found no GPU")
    selected_device = "gpu" if gpus else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    manifests = {
        name: args.split_dir / f"{name}.jsonl"
        for name in ("train", "validation", "test")
    }
    train = read_manifest(manifests["train"], args.image_root)
    validation = read_manifest(manifests["validation"], args.image_root)
    test = read_manifest(manifests["test"], args.image_root)
    model = load_official_module(args.source).SSR_net(64, [3, 3, 3], 1, 1)()
    model.load_weights(args.initial_weights)
    loss = "mae" if args.loss == "mae" else keras.losses.Huber(args.huber_delta)
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss=loss,
        metrics=[WithinYears()],
    )

    training_dataset, sampling_weights = make_training_dataset(train, args)
    initial_validation = evaluate(model, validation, args)
    steps = max(1, len(train) // args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = args.output_dir / "age_v3_best.weights.h5"
    latest = args.output_dir / "age_v3_latest.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best,
            monitor="val_within_5_years",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(latest, save_weights_only=True, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_within_5_years",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    started = time.time()
    history = model.fit(
        training_dataset,
        validation_data=dataset_for_rows(validation, args, False),
        steps_per_epoch=steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - started
    model.load_weights(best)

    config = vars(args).copy()
    config.update(
        {
            "source": str(args.source.resolve()),
            "initial_weights": str(args.initial_weights.resolve()),
            "image_root": str(args.image_root.resolve()),
            "split_dir": str(args.split_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
        }
    )
    run = {
        "config": config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "tensorflow_built_with_cuda": tf.test.is_built_with_cuda(),
            "physical_gpus": [gpu.name for gpu in gpus],
            "selected_device": selected_device,
        },
        "dataset": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "manifest_sha256": {
                name: file_sha256(path) for name, path in manifests.items()
            },
            "train_age_bins": dict(sorted(Counter(row["age_bin"] for row in train).items())),
            "sampling_bin_probabilities": sampling_weights,
        },
        "initial_validation": initial_validation,
        "selected_validation": evaluate(model, validation, args),
        "selected_test": evaluate(model, test, args),
        "epochs_completed": len(history.epoch),
        "steps_per_epoch": steps,
        "elapsed_seconds": elapsed,
        "seconds_per_epoch": elapsed / len(history.epoch),
        "best_weights_sha256": file_sha256(best),
        "latest_weights_sha256": file_sha256(latest),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(json_ready(config), indent=2, sort_keys=True) + "\n"
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
