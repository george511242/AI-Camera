#!/usr/bin/env python3
"""Class-balanced fine-tuning for the official SSR-Net gender model."""

import argparse
import importlib.util
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras as standalone_keras
from tensorflow import keras


class BalancedBinaryAccuracy(keras.metrics.Metric):
    def __init__(self, name="balanced_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positive = self.add_weight(name="tp", initializer="zeros")
        self.true_negative = self.add_weight(name="tn", initializer="zeros")
        self.positive = self.add_weight(name="p", initializer="zeros")
        self.negative = self.add_weight(name="n", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        del sample_weight
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.bool)
        y_pred = tf.reshape(y_pred, (-1,)) >= 0.5
        self.true_positive.assign_add(tf.reduce_sum(tf.cast(y_true & y_pred, self.dtype)))
        self.true_negative.assign_add(tf.reduce_sum(tf.cast(~y_true & ~y_pred, self.dtype)))
        self.positive.assign_add(tf.reduce_sum(tf.cast(y_true, self.dtype)))
        self.negative.assign_add(tf.reduce_sum(tf.cast(~y_true, self.dtype)))

    def result(self):
        male_recall = tf.math.divide_no_nan(self.true_positive, self.positive)
        female_recall = tf.math.divide_no_nan(self.true_negative, self.negative)
        return (male_recall + female_recall) / 2.0

    def reset_state(self):
        for variable in self.variables:
            variable.assign(0)


def load_official_module(source_path: Path):
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


def read_manifest(path: Path, image_root: Path):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        row["file"] = str((image_root / row["file"]).resolve())
    return rows


def decode_image(path, label, training):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.reverse(image, axis=[-1])  # VAC inference supplies OpenCV BGR.
    image = tf.cast(image, tf.float32)

    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=18.0)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        image = tf.clip_by_value(image, 0.0, 255.0)
        face_size = tf.random.uniform((), minval=24, maxval=43, dtype=tf.int32)
    else:
        face_size = tf.constant(34, dtype=tf.int32)

    # A 0.45 VAC margin makes the face occupy about 1/1.9 of the crop.
    image = tf.image.resize(image, (face_size, face_size), antialias=True)
    image = tf.image.resize_with_crop_or_pad(image, 64, 64)
    image.set_shape((64, 64, 3))
    return image, tf.reshape(tf.cast(label, tf.float32), (1,))


def make_training_dataset(rows, batch_size, seed):
    by_gender = {
        gender: [row for row in rows if row["gender"] == gender]
        for gender in (0, 1)
    }
    half = batch_size // 2
    datasets = []
    for gender in (0, 1):
        files = [row["file"] for row in by_gender[gender]]
        labels = [1 - gender for gender in [row["gender"] for row in by_gender[gender]]]
        dataset = tf.data.Dataset.from_tensor_slices((files, labels))
        dataset = dataset.shuffle(len(files), seed=seed + gender,
                                  reshuffle_each_iteration=True).repeat()
        dataset = dataset.map(
            lambda path, label: decode_image(path, label, True),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        ).batch(half, drop_remainder=True)
        datasets.append(dataset)

    def merge(male_batch, female_batch):
        images = tf.concat((male_batch[0], female_batch[0]), axis=0)
        labels = tf.concat((male_batch[1], female_batch[1]), axis=0)
        order = tf.random.shuffle(tf.range(batch_size), seed=seed)
        return tf.gather(images, order), tf.gather(labels, order)

    return tf.data.Dataset.zip(tuple(datasets)).map(
        merge, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    ).prefetch(tf.data.AUTOTUNE)


def make_evaluation_dataset(rows, batch_size):
    files = [row["file"] for row in rows]
    labels = [1 - row["gender"] for row in rows]  # output >=0.5 means male
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    return dataset.map(
        lambda path, label: decode_image(path, label, False),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def evaluate_model(model, rows, batch_size):
    dataset = make_evaluation_dataset(rows, batch_size)
    values = model.evaluate(dataset, verbose=0, return_dict=True)
    predictions = np.ravel(model.predict(dataset, verbose=0)) >= 0.5
    labels = np.array([1 - row["gender"] for row in rows], dtype=bool)
    male_count = int(labels.sum())
    female_count = int((~labels).sum())
    values.update({
        "male_recall": float((predictions & labels).sum() / male_count),
        "female_recall": float(((~predictions) & (~labels)).sum() / female_count),
        "count": len(rows),
    })
    return {key: float(value) if isinstance(value, (np.floating, float)) else value
            for key, value in values.items()}


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
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("batch-size must be an even number >= 2")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    train_rows = read_manifest(args.train_manifest, args.image_root)
    validation_rows = read_manifest(args.validation_manifest, args.image_root)
    test_rows = read_manifest(args.test_manifest, args.image_root)
    module = load_official_module(args.source)
    model = module.SSR_net_general(64, [3, 3, 3], 1, 1)()
    model.load_weights(args.initial_weights)
    model.compile(
        optimizer=keras.optimizers.Adam(args.learning_rate),
        loss="mae",
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
            BalancedBinaryAccuracy(),
        ],
    )

    train_dataset = make_training_dataset(train_rows, args.batch_size, args.seed)
    validation_dataset = make_evaluation_dataset(validation_rows, args.batch_size)
    initial_validation = evaluate_model(model, validation_rows, args.batch_size)
    steps_per_epoch = max(1, (2 * max(
        sum(row["gender"] == 0 for row in train_rows),
        sum(row["gender"] == 1 for row in train_rows),
    )) // args.batch_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "gender_ssrnet_v2_best.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_balanced_accuracy", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_balanced_accuracy", mode="max", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1,
        ),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    model.load_weights(best_path)
    selected_path = args.output_dir / "gender_ssrnet_v2_selected.weights.h5"
    model.save_weights(selected_path)
    selected_validation = evaluate_model(model, validation_rows, args.batch_size)
    selected_test = evaluate_model(model, test_rows, args.batch_size)
    history_path = args.output_dir / "history.json"
    history_path.write_text(json.dumps(history.history, indent=2, default=float) + "\n")
    run_path = args.output_dir / "training_run.json"
    run_path.write_text(json.dumps({
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.epoch),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "steps_per_epoch": steps_per_epoch,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "test_count": len(test_rows),
        "initial_validation": initial_validation,
        "selected_validation": selected_validation,
        "selected_test": selected_test,
    }, indent=2, sort_keys=True) + "\n")
    print(f"best weights: {best_path}")
    print(f"selected weights: {selected_path}")
    print(f"history: {history_path}")
    print(f"training run: {run_path}")


if __name__ == "__main__":
    main()
