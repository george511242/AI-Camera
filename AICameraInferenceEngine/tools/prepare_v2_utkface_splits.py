#!/usr/bin/env python3
"""Create deterministic UTKFace splits while excluding the frozen benchmark."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_label(path: Path):
    parts = path.name.split("_")
    if len(parts) < 4:
        return None
    try:
        age, gender = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not 0 <= age <= 116 or gender not in (0, 1):
        return None
    return age, gender


def read_benchmark_ids(path: Path):
    ids = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") == "utkface":
                ids.add(row["source_id"])
    return ids


def allocate_bucket(rows, rng, train_ratio, val_ratio):
    rng.shuffle(rows)
    count = len(rows)
    train_end = round(count * train_ratio)
    val_end = train_end + round(count * val_ratio)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def write_jsonl(path: Path, rows):
    with path.open("w") as handle:
        for row in sorted(rows, key=lambda item: item["source_id"]):
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows):
    return {
        "count": len(rows),
        "gender": dict(sorted(Counter(row["gender"] for row in rows).items())),
        "age_bin": dict(sorted(Counter(row["age_bin"] for row in rows).items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utk-dir", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        parser.error("train and validation ratios must be positive")
    if args.train_ratio + args.val_ratio >= 1:
        parser.error("train_ratio + val_ratio must be less than 1")

    benchmark_ids = read_benchmark_ids(args.benchmark_manifest)
    buckets = defaultdict(list)
    external = []
    for path in sorted(args.utk_dir.glob("*.jpg")):
        label = parse_label(path)
        if label is None:
            continue
        age, gender = label
        row = {
            "source_id": path.name,
            "file": path.name,
            "age": age,
            "gender": gender,
            "age_bin": min(age // 10, 10),
        }
        if path.name in benchmark_ids:
            external.append(row)
        else:
            buckets[(gender, row["age_bin"])].append(row)

    rng = random.Random(args.seed)
    train, val, test = [], [], []
    for key in sorted(buckets):
        parts = allocate_bucket(
            buckets[key], rng, args.train_ratio, args.val_ratio
        )
        train.extend(parts[0])
        val.extend(parts[1])
        test.extend(parts[2])

    split_ids = [set(row["source_id"] for row in rows)
                 for rows in (train, val, test, external)]
    for index, left in enumerate(split_ids):
        for right in split_ids[index + 1:]:
            if left & right:
                raise RuntimeError("split overlap detected")
    if set(row["source_id"] for row in external) != benchmark_ids:
        raise RuntimeError("benchmark manifest and UTKFace directory do not match")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    split_rows = {
        "train": train,
        "validation": val,
        "test": test,
        "external_benchmark": external,
    }
    for name, rows in split_rows.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[name] = path

    metadata = {
        "seed": args.seed,
        "strategy": "stratified by gender and age decade",
        "ratios_for_non_benchmark": {
            "train": args.train_ratio,
            "validation": args.val_ratio,
            "test": 1 - args.train_ratio - args.val_ratio,
        },
        "benchmark_manifest": str(args.benchmark_manifest.resolve()),
        "benchmark_source_count": len(benchmark_ids),
        "identity_isolation": False,
        "identity_isolation_note": (
            "UTKFace filenames do not provide a reliable person identity. "
            "Derived samples are grouped by source_id, but identity-level isolation "
            "cannot be guaranteed."
        ),
        "splits": {
            name: {**summarize(rows), "sha256": sha256(paths[name])}
            for name, rows in split_rows.items()
        },
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
