#!/usr/bin/env python3
"""Age-balanced small-face fine-tuning for the official SSR-Net age model."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from export_ssrnet_gender_onnx import load_official_module


class WithinFiveYears(keras.metrics.Metric):
    def __init__(self, name="within_5_years", **kwargs):
        super().__init__(name=name, **kwargs)
        self.hits = self.add_weight(name="hits", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight
        errors = tf.abs(tf.reshape(y_true, (-1,)) - tf.reshape(y_pred, (-1,)))
        self.hits.assign_add(tf.reduce_sum(tf.cast(errors <= 5.0, self.dtype)))
        self.count.assign_add(tf.cast(tf.size(errors), self.dtype))

    def result(self):
        return tf.math.divide_no_nan(self.hits, self.count)

    def reset_state(self):
        self.hits.assign(0)
        self.count.assign(0)


def read_manifest(path: Path, image_root: Path):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        row["file"] = str((image_root / row["file"]).resolve())
    return rows


def decode_image(path, age, training):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.reverse(image, axis=[-1])
    image = tf.cast(image, tf.float32)
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=18.0)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.clip_by_value(image, 0.0, 255.0)
        face_size = tf.random.uniform((), 24, 43, dtype=tf.int32)
    else:
        face_size = tf.constant(34, dtype=tf.int32)
    image = tf.image.resize(image, (face_size, face_size), antialias=True)
    image = tf.image.resize_with_crop_or_pad(image, 64, 64)
    image.set_shape((64, 64, 3))
    return image, tf.reshape(tf.cast(age, tf.float32), (1,))


def dataset_for_rows(rows, batch_size, training, seed=0, repeat=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        ([row["file"] for row in rows], [row["age"] for row in rows])
    )
    if training:
        dataset = dataset.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat()
    return dataset.map(
        lambda path, age: decode_image(path, age, training),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_balanced_training_dataset(rows, batch_size, seed):
    bins = sorted({row["age_bin"] for row in rows})
    datasets = []
    for age_bin in bins:
        bin_rows = [row for row in rows if row["age_bin"] == age_bin]
        datasets.append(dataset_for_rows(bin_rows, 1, True, seed + age_bin, True).unbatch())
    return tf.data.Dataset.sample_from_datasets(
        datasets, weights=[1.0 / len(bins)] * len(bins), seed=seed
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def evaluate(model, rows, batch_size):
    dataset = dataset_for_rows(rows, batch_size, False)
    predictions = np.ravel(model.predict(dataset, verbose=0))
    labels = np.asarray([row["age"] for row in rows], dtype=np.float32)
    errors = np.abs(predictions - labels)
    return {
        "count": len(rows),
        "mae": float(errors.mean()),
        "within_5_years": float((errors <= 5.0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    train = read_manifest(args.train_manifest, args.image_root)
    validation = read_manifest(args.validation_manifest, args.image_root)
    test = read_manifest(args.test_manifest, args.image_root)
    model = load_official_module(args.source).SSR_net(64, [3, 3, 3], 1, 1)()
    model.load_weights(args.initial_weights)
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="mae",
        metrics=[WithinFiveYears()],
    )

    initial_validation = evaluate(model, validation, args.batch_size)
    steps = max(1, len(train) // args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = args.output_dir / "age_ssrnet_v2_best.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best, monitor="val_loss", mode="min", save_best_only=True,
            save_weights_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1,
        ),
    ]
    history = model.fit(
        make_balanced_training_dataset(train, args.batch_size, args.seed),
        validation_data=dataset_for_rows(validation, args.batch_size, False),
        steps_per_epoch=steps, epochs=args.epochs, callbacks=callbacks, verbose=2,
    )
    model.load_weights(best)
    selected = args.output_dir / "age_ssrnet_v2_selected.weights.h5"
    model.save_weights(selected)
    run = {
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.epoch),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "steps_per_epoch": steps,
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
        "initial_validation": initial_validation,
        "selected_validation": evaluate(model, validation, args.batch_size),
        "selected_test": evaluate(model, test, args.batch_size),
    }
    (args.output_dir / "history.json").write_text(
        json.dumps(history.history, indent=2, default=float) + "\n"
    )
    (args.output_dir / "training_run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run, indent=2))
    print(f"selected weights: {selected}")


if __name__ == "__main__":
    main()
