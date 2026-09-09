#!/usr/bin/env python3
"""Create leakage-resistant fixed manifests for Head Pose and YuNet v2."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import scipy.io as sio


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_group(path):
    family = path.parent.name.removesuffix("_Flip")
    base = re.sub(r"_\d+$", "", path.stem)
    return f"{family}/{base}"


def build_pose(root, aflw_root, output, seed, validation_ratio):
    rows = []
    for image in sorted(root.glob("*/*.jpg")):
        if image.parent.name == "landmarks":
            continue
        annotation = image.with_suffix(".mat")
        if not annotation.is_file():
            raise FileNotFoundError(f"missing pose annotation: {annotation}")
        pose = sio.loadmat(annotation)["Pose_Para"].reshape(-1)[:3]
        rows.append({
            "file": str(image.relative_to(root)),
            "annotation": str(annotation.relative_to(root)),
            "source_group": pose_group(image),
            "pitch_rad": float(pose[0]),
            "yaw_rad": float(pose[1]),
            "roll_rad": float(pose[2]),
        })

    groups = sorted({row["source_group"] for row in rows})
    random.Random(seed).shuffle(groups)
    validation_groups = set(groups[:round(len(groups) * validation_ratio)])
    train = [row for row in rows if row["source_group"] not in validation_groups]
    validation = [row for row in rows if row["source_group"] in validation_groups]

    external = []
    for image in sorted(aflw_root.glob("*.jpg")):
        annotation = image.with_suffix(".mat")
        if annotation.is_file():
            external.append({"file": image.name, "annotation": annotation.name})

    return {
        "train": (train, write_jsonl(output / "pose_train.jsonl", train)),
        "validation": (validation, write_jsonl(output / "pose_validation.jsonl", validation)),
        "external_test": (external, write_jsonl(output / "pose_external_aflw2000.jsonl", external)),
        "group_count": len(groups),
    }


def labelv2_filenames(path, include_empty):
    names = []
    current = None
    has_annotation = False

    def flush():
        if current is not None and (include_empty or has_annotation):
            names.append(current)

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("# "):
                flush()
                current = line[2:].split()[0]
                has_annotation = False
            elif line.strip():
                has_annotation = True
    flush()
    return names


def build_yunet(root, scface_root, output):
    split_rows = {}
    for split, image_dir_name in (("train", "WIDER_train"), ("validation", "WIDER_val")):
        ann_split = "train" if split == "train" else "val"
        ann = root / "labelv2" / ann_split / "labelv2.txt"
        image_dir = root / image_dir_name / "images"
        rows = [{"file": name, "annotation_file": str(ann.relative_to(root))}
                for name in labelv2_filenames(ann, include_empty=split != "train")]
        missing = [row["file"] for row in rows if not (image_dir / row["file"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{split} has {len(missing)} missing images")
        split_rows[split] = (rows, write_jsonl(output / f"yunet_{split}.jsonl", rows))

    landmark_table = scface_root / "mugshot_frontal_cropped.txt"
    with landmark_table.open() as handle:
        names = [line.split()[0] for line in handle if len(line.split()) == 9]
    external = [{"source_id": name, "file": f"mugshot_frontal_cropped_all/{name}.JPG"}
                for name in names]
    missing = [row["file"] for row in external if not (scface_root / row["file"]).is_file()]
    if missing:
        raise FileNotFoundError(f"SCFace has {len(missing)} missing images")
    split_rows["external_test"] = (
        external, write_jsonl(output / "yunet_external_scface.jsonl", external)
    )
    return split_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--aflw-root", type=Path, required=True)
    parser.add_argument("--wider-root", type=Path, required=True)
    parser.add_argument("--scface-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--pose-validation-ratio", type=float, default=0.15)
    args = parser.parse_args()

    pose = build_pose(
        args.pose_root, args.aflw_root, args.output_dir,
        args.seed, args.pose_validation_ratio,
    )
    yunet = build_yunet(args.wider_root, args.scface_root, args.output_dir)
    metadata = {
        "seed": args.seed,
        "pose": {
            "policy": "300W-LP grouped by original source before deterministic 85/15 split; AFLW2000 frozen external test",
            "source_group_count": pose["group_count"],
            "splits": {name: {"count": len(value[0]), "sha256": value[1]}
                       for name, value in pose.items() if name != "group_count"},
        },
        "yunet": {
            "policy": "official WIDER FACE train/validation partitions; SCFace frozen external test",
            "splits": {name: {"count": len(value[0]), "sha256": value[1]}
                       for name, value in yunet.items()},
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
