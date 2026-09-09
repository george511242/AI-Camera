#!/usr/bin/env python3
"""Create deterministic, content-isolated UTKFace splits for Age v3."""

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


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_benchmark_ids(path: Path):
    ids = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") == "utkface":
                ids.add(row["source_id"])
    return ids


def make_row(path: Path, content_sha256: str):
    age, gender = parse_label(path)
    return {
        "source_id": path.name,
        "file": path.name,
        "age": age,
        "gender": gender,
        "age_bin": min(age // 10, 10),
        "content_sha256": content_sha256,
    }


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


def manifest_sha256(path: Path):
    return file_sha256(path)


def summarize(rows):
    return {
        "count": len(rows),
        "unique_content_sha256": len({row["content_sha256"] for row in rows}),
        "gender": dict(sorted(Counter(row["gender"] for row in rows).items())),
        "age_bin": dict(sorted(Counter(row["age_bin"] for row in rows).items())),
    }


def overlap_counts(split_rows):
    result = {}
    names = list(split_rows)
    for index, left_name in enumerate(names):
        left = split_rows[left_name]
        left_ids = {row["source_id"] for row in left}
        left_hashes = {row["content_sha256"] for row in left}
        for right_name in names[index + 1 :]:
            right = split_rows[right_name]
            key = f"{left_name}_intersection_{right_name}"
            result[key] = {
                "source_id": len(left_ids & {row["source_id"] for row in right}),
                "content_sha256": len(
                    left_hashes & {row["content_sha256"] for row in right}
                ),
            }
    return result


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
    paths = []
    invalid_files = []
    for path in sorted(args.utk_dir.glob("*.jpg")):
        if parse_label(path) is None:
            invalid_files.append(path.name)
        else:
            paths.append(path)

    missing_benchmark = sorted(benchmark_ids - {path.name for path in paths})
    if missing_benchmark:
        raise RuntimeError(
            f"{len(missing_benchmark)} benchmark sources are missing or invalid"
        )

    by_hash = defaultdict(list)
    for path in paths:
        by_hash[file_sha256(path)].append(path)

    benchmark_hashes = {
        digest
        for digest, group in by_hash.items()
        if any(path.name in benchmark_ids for path in group)
    }
    external = []
    benchmark_content_aliases = []
    duplicate_aliases = []
    conflicting_groups = []
    candidates = []

    for digest, group in sorted(by_hash.items()):
        benchmark_members = [path for path in group if path.name in benchmark_ids]
        non_benchmark = [path for path in group if path.name not in benchmark_ids]
        external.extend(make_row(path, digest) for path in benchmark_members)

        if digest in benchmark_hashes:
            benchmark_content_aliases.extend(
                {
                    "source_id": path.name,
                    "content_sha256": digest,
                    "reason": "content matches frozen benchmark",
                }
                for path in non_benchmark
            )
            continue
        if not non_benchmark:
            continue

        labels = {parse_label(path) for path in non_benchmark}
        if len(labels) != 1:
            conflicting_groups.append(
                {
                    "content_sha256": digest,
                    "files": [path.name for path in non_benchmark],
                    "labels": [list(label) for label in sorted(labels)],
                    "reason": "identical content has conflicting age/gender labels",
                }
            )
            continue

        canonical = min(non_benchmark, key=lambda path: path.name)
        candidates.append(make_row(canonical, digest))
        duplicate_aliases.extend(
            {
                "source_id": path.name,
                "canonical_source_id": canonical.name,
                "content_sha256": digest,
                "reason": "duplicate content with matching label",
            }
            for path in non_benchmark
            if path != canonical
        )

    buckets = defaultdict(list)
    for row in candidates:
        buckets[(row["gender"], row["age_bin"])].append(row)

    rng = random.Random(args.seed)
    train, validation, test = [], [], []
    for key in sorted(buckets):
        parts = allocate_bucket(
            buckets[key], rng, args.train_ratio, args.val_ratio
        )
        train.extend(parts[0])
        validation.extend(parts[1])
        test.extend(parts[2])

    split_rows = {
        "train": train,
        "validation": validation,
        "test": test,
        "external_benchmark": external,
    }
    overlaps = overlap_counts(split_rows)
    if any(value != 0 for pair in overlaps.values() for value in pair.values()):
        raise RuntimeError(f"split overlap detected: {overlaps}")
    if {row["source_id"] for row in external} != benchmark_ids:
        raise RuntimeError("frozen benchmark sources were not preserved exactly")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = {}
    for name, rows in split_rows.items():
        path = args.output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        manifest_paths[name] = path

    quarantine = {
        "invalid_files": invalid_files,
        "benchmark_content_aliases": benchmark_content_aliases,
        "duplicate_aliases": duplicate_aliases,
        "conflicting_content_groups": conflicting_groups,
    }
    quarantine_path = args.output_dir / "quarantine.json"
    quarantine_path.write_text(json.dumps(quarantine, indent=2, sort_keys=True) + "\n")

    metadata = {
        "version": "age-v3",
        "seed": args.seed,
        "strategy": (
            "benchmark-exclusive SHA256 grouping, exact-content deduplication, "
            "then stratification by gender and age decade"
        ),
        "ratios_for_non_benchmark": {
            "train": args.train_ratio,
            "validation": args.val_ratio,
            "test": 1 - args.train_ratio - args.val_ratio,
        },
        "utk_dir": str(args.utk_dir.resolve()),
        "benchmark_manifest": str(args.benchmark_manifest.resolve()),
        "benchmark_manifest_sha256": manifest_sha256(args.benchmark_manifest),
        "benchmark_source_count": len(benchmark_ids),
        "raw_valid_file_count": len(paths),
        "quarantine": {
            "invalid_file_count": len(invalid_files),
            "benchmark_content_alias_count": len(benchmark_content_aliases),
            "duplicate_alias_count": len(duplicate_aliases),
            "conflicting_group_count": len(conflicting_groups),
            "conflicting_file_count": sum(
                len(group["files"]) for group in conflicting_groups
            ),
            "manifest": quarantine_path.name,
        },
        "overlap_validation": overlaps,
        "splits": {
            name: {
                **summarize(rows),
                "manifest": path.name,
                "manifest_sha256": manifest_sha256(path),
            }
            for (name, rows), path in zip(split_rows.items(), manifest_paths.values())
        },
        "identity_isolation": False,
        "identity_isolation_note": (
            "UTKFace filenames do not expose stable person identities. Source-path "
            "and exact-content isolation are enforced; person-level isolation cannot "
            "be proven from this dataset metadata."
        ),
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
