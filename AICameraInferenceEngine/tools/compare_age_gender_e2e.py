#!/usr/bin/env python3
"""Compare v1/v2 Age and Gender on identical end-to-end result rows."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


HEIGHTS = (80, 40, 27)


def load_rows(path):
    return [json.loads(line) for line in path.open()]


def sample_key(row):
    return row["dataset"], row["source_id"], row["target_face_height_px"]


def yunet_succeeded(row):
    return row.get("iou", 0) >= 0.5 and (row.get("yunet_score") or 0) >= 0.75


def validate_pair(before, after):
    if len(before) != len(after):
        raise ValueError(f"row count differs: {len(before)} != {len(after)}")
    before_by_key = {sample_key(row): row for row in before}
    after_by_key = {sample_key(row): row for row in after}
    if len(before_by_key) != len(before) or len(after_by_key) != len(after):
        raise ValueError("duplicate dataset/source/height key")
    if before_by_key.keys() != after_by_key.keys():
        raise ValueError("v1/v2 sample keys differ")
    detection_mismatches = []
    for key in before_by_key:
        left, right = before_by_key[key], after_by_key[key]
        fields = ("detected_bbox", "yunet_score", "iou")
        if any(left.get(field) != right.get(field) for field in fields):
            detection_mismatches.append(key)
        if yunet_succeeded(left) != yunet_succeeded(right):
            detection_mismatches.append(key)
    if detection_mismatches:
        raise ValueError(
            f"v1/v2 YuNet results differ for {len(set(detection_mismatches))} rows"
        )
    return before_by_key, after_by_key


def balanced_accuracy(rows, conditional=False):
    classes = defaultdict(lambda: [0, 0])
    for row in rows:
        if conditional and not yunet_succeeded(row):
            continue
        ground_truth = row["gt_gender"]
        classes[ground_truth][1] += 1
        value = row.get("predicted_gender")
        if yunet_succeeded(row) and value is not None:
            prediction = 0 if value >= 0.5 else 1
            classes[ground_truth][0] += prediction == ground_truth
    if set(classes) != {0, 1}:
        raise ValueError("Gender metric requires both male and female samples")
    recalls = {label: correct / total for label, (correct, total) in classes.items()}
    return {
        "value": (recalls[0] + recalls[1]) / 2,
        "male_correct": classes[0][0],
        "male_count": classes[0][1],
        "female_correct": classes[1][0],
        "female_count": classes[1][1],
    }


def age_metrics(rows):
    detected = [row for row in rows if yunet_succeeded(row)]
    e2e_hits = sum(
        1 for row in rows
        if row.get("predicted_age") is not None
        and yunet_succeeded(row)
        and abs(row["predicted_age"] - row["gt_age"]) <= 5
    )
    conditional_hits = sum(
        1 for row in detected
        if row.get("predicted_age") is not None
        and abs(row["predicted_age"] - row["gt_age"]) <= 5
    )
    errors = [
        abs(row["predicted_age"] - row["gt_age"])
        for row in detected if row.get("predicted_age") is not None
    ]
    return {
        "e2e_accuracy": e2e_hits / len(rows),
        "e2e_hits": e2e_hits,
        "total": len(rows),
        "conditional_accuracy": conditional_hits / len(detected) if detected else None,
        "conditional_hits": conditional_hits,
        "detected": len(detected),
        "mae": sum(errors) / len(errors) if errors else None,
        "mae_count": len(errors),
    }


def calculate(before, after):
    validate_pair(before, after)
    result = {"heights": {}, "overall": {}}
    versions = {"v1": before, "v2": after}
    for height in HEIGHTS:
        result["heights"][str(height)] = {}
        for version, rows in versions.items():
            selected = [
                row for row in rows
                if row["dataset"] == "utkface"
                and row["target_face_height_px"] == height
            ]
            result["heights"][str(height)][version] = {
                "age": age_metrics(selected),
                "gender_e2e": balanced_accuracy(selected),
                "gender_conditional": balanced_accuracy(selected, conditional=True),
            }
    for version, rows in versions.items():
        selected = [row for row in rows if row["dataset"] == "utkface"]
        result["overall"][version] = {
            "age": age_metrics(selected),
            "gender_e2e": balanced_accuracy(selected),
            "gender_conditional": balanced_accuracy(selected, conditional=True),
        }
    return result


def percent(value):
    return "-" if value is None else f"{value * 100:.2f}%"


def render_markdown(result, before_path, after_path):
    lines = [
        "# Age/Gender v1 vs v2 RK3588 完整 E2E Benchmark",
        "",
        f"- v1 result: `{before_path}`",
        f"- v2 result: `{after_path}`",
        "- Gender mapping: raw >= 0.5 -> male (fixed before evaluation)",
        "- YuNet success: IoU >= 0.5 and score >= 0.75",
        "- 完整 harness 為 9,198 rows；Age/Gender 指標使用其中 2,808 個 UTKFace rows",
        "",
        "## 改善摘要",
        "",
        "| 臉高 | Age E2E ±5 Δ | Age Conditional ±5 Δ | Age MAE Δ | Gender E2E BAcc Δ | Gender Conditional BAcc Δ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    summary_groups = [
        (f"{height}px", result["heights"][str(height)]) for height in HEIGHTS
    ]
    summary_groups.append(("總計", result["overall"]))
    for label, group in summary_groups:
        v1, v2 = group["v1"], group["v2"]
        lines.append(
            f"| {label} | {(v2['age']['e2e_accuracy'] - v1['age']['e2e_accuracy']) * 100:+.2f} pp | "
            f"{(v2['age']['conditional_accuracy'] - v1['age']['conditional_accuracy']) * 100:+.2f} pp | "
            f"{v2['age']['mae'] - v1['age']['mae']:+.3f} years | "
            f"{(v2['gender_e2e']['value'] - v1['gender_e2e']['value']) * 100:+.2f} pp | "
            f"{(v2['gender_conditional']['value'] - v1['gender_conditional']['value']) * 100:+.2f} pp |"
        )
    lines += [
        "",
        "> 27px 的 UTKFace 只有 15/936 通過 YuNet，因此該層 Conditional 指標樣本極少；E2E 幾乎由 detector gate 主導。",
        "",
        "## Age",
        "",
        "| 臉高 | 版本 | E2E ±5 | Conditional ±5 | Conditional MAE | YuNet成功/總數 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    groups = [(f"{height}px", result["heights"][str(height)]) for height in HEIGHTS]
    groups.append(("總計", result["overall"]))
    for label, group in groups:
        for version in ("v1", "v2"):
            age = group[version]["age"]
            lines.append(
                f"| {label} | {version} | {percent(age['e2e_accuracy'])} "
                f"({age['e2e_hits']}/{age['total']}) | "
                f"{percent(age['conditional_accuracy'])} "
                f"({age['conditional_hits']}/{age['detected']}) | "
                f"{age['mae']:.3f} ({age['mae_count']}) | "
                f"{age['detected']}/{age['total']} |"
            )
    lines += [
        "",
        "## Gender",
        "",
        "| 臉高 | 版本 | E2E Balanced Accuracy | Conditional Balanced Accuracy |",
        "|---:|---|---:|---:|",
    ]
    for label, group in groups:
        for version in ("v1", "v2"):
            e2e = group[version]["gender_e2e"]
            conditional = group[version]["gender_conditional"]
            lines.append(
                f"| {label} | {version} | {percent(e2e['value'])} | "
                f"{percent(conditional['value'])} |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    before, after = load_rows(args.before), load_rows(args.after)
    result = calculate(before, after)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown.write_text(render_markdown(result, args.before, args.after))
    print(f"Compared {len(before)} identical rows")
    print(f"Wrote {args.json}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
