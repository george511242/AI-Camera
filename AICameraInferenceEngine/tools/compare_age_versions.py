#!/usr/bin/env python3
"""Compare Age E2E and conditional metrics across matched benchmark results."""

import argparse
import json
from pathlib import Path


BUCKETS = ((0, 9, "0-9"), (10, 19, "10-19"), (20, 29, "20-29"),
           (30, 39, "30-39"), (40, 49, "40-49"), (50, 59, "50-59"),
           (60, 69, "60-69"), (70, 79, "70-79"), (80, 200, "80+"))


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def sample_key(row):
    return row["dataset"], row["source_id"], row["target_face_height_px"]


def validate_matched(reference, candidate, candidate_name):
    if [sample_key(row) for row in reference] != [sample_key(row) for row in candidate]:
        raise RuntimeError(f"sample keys/order differ for {candidate_name}")
    fields = ("detected_bbox", "yunet_score", "iou", "failure_reason")
    mismatches = sum(
        any(left.get(field) != right.get(field) for field in fields)
        for left, right in zip(reference, candidate)
    )
    if mismatches:
        raise RuntimeError(f"{candidate_name} has {mismatches} YuNet/failure mismatches")


def metrics(rows):
    detected = [
        row for row in rows
        if row.get("failure_reason") is None and row.get("predicted_age") is not None
    ]
    errors = [row["predicted_age"] - row["gt_age"] for row in detected]
    hits = sum(abs(error) <= 5 for error in errors)
    return {
        "total": len(rows),
        "detected": len(detected),
        "hits_within_5": hits,
        "e2e_within_5": hits / len(rows) if rows else None,
        "conditional_within_5": hits / len(detected) if detected else None,
        "conditional_mae": (
            sum(abs(error) for error in errors) / len(errors) if errors else None
        ),
        "mean_prediction": (
            sum(row["predicted_age"] for row in detected) / len(detected)
            if detected else None
        ),
        "mean_signed_error": sum(errors) / len(errors) if errors else None,
    }


def analyze(rows):
    utk = [row for row in rows if row["dataset"] == "utkface"]
    result = {
        str(height): metrics(
            [row for row in utk if row["target_face_height_px"] == height]
        ) for height in (80, 40, 27)
    }
    result["80+40"] = metrics(
        [row for row in utk if row["target_face_height_px"] in (80, 40)]
    )
    result["overall"] = metrics(utk)
    result["age_buckets_80+40"] = {}
    primary = [row for row in utk if row["target_face_height_px"] in (80, 40)]
    for low, high, label in BUCKETS:
        result["age_buckets_80+40"][label] = metrics(
            [row for row in primary if low <= row["gt_age"] <= high]
        )
    return result


def pct(value):
    return "-" if value is None else f"{value * 100:.2f}%"


def number(value):
    return "-" if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--v31", type=Path)
    parser.add_argument("--v32", type=Path)
    parser.add_argument("--v4", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    paths = {"v1": args.v1, "v2": args.v2, "v3": args.v3}
    if args.v31:
        paths["v3.1"] = args.v31
    if args.v32:
        paths["v3.2"] = args.v32
    if args.v4:
        paths["v4"] = args.v4
    raw = {name: read_rows(path) for name, path in paths.items()}
    for name, rows in raw.items():
        if name != "v1":
            validate_matched(raw["v1"], rows, name)
    if any(row.get("failure_reason") == "runtime_error" for rows in raw.values() for row in rows):
        raise RuntimeError("runtime_error found in benchmark results")
    result = {
        "validation": {
            "matched_rows": len(raw["v1"]),
            "runtime_errors": 0,
            "yunet_fields_match": True,
        },
        "source_files": {name: str(path) for name, path in paths.items()},
        "versions": {name: analyze(rows) for name, rows in raw.items()},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# Age {' / '.join(paths)} RK3588 E2E Benchmark",
        "",
        f"{len(paths)} 版使用相同的 9,198-row frozen manifest、YuNet、crop margin 與計分門檻。",
        "樣本鍵及 YuNet bbox/score/IoU 完全一致，runtime error 均為 0。",
        "",
        "## 分距離結果",
        "",
        "| 距離 | 版本 | YuNet 成功 | E2E +/-5 | Conditional +/-5 | Conditional MAE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {"80": "80px", "40": "40px", "27": "27px", "80+40": "80+40px", "overall": "Overall"}
    for level in ("80", "40", "27", "80+40", "overall"):
        for version in paths:
            value = result["versions"][version][level]
            lines.append(
                f"| {labels[level]} | {version} | {value['detected']}/{value['total']} | "
                f"{pct(value['e2e_within_5'])} | {pct(value['conditional_within_5'])} | "
                f"{number(value['conditional_mae'])} |"
            )

    lines += [
        "",
        f"## {list(paths)[-1]} 相對差異",
        "",
        "| 範圍 | Conditional +/-5 vs v1 | vs v2 | vs v3.1 | vs v3.2 | MAE vs v1 | vs v2 | vs v3.1 | vs v3.2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in ("80", "40", "27", "80+40", "overall"):
        versions = result["versions"]
        candidate = list(paths)[-1]
        v1, v2, selected = versions["v1"][level], versions["v2"][level], versions[candidate][level]
        v31 = versions.get("v3.1", {}) .get(level)
        v32 = versions.get("v3.2", {}) .get(level)
        conditional_vs_v31 = (
            f"{(selected['conditional_within_5']-v31['conditional_within_5'])*100:+.2f} pp"
            if v31 else "-"
        )
        mae_vs_v31 = (
            f"{selected['conditional_mae']-v31['conditional_mae']:+.3f}"
            if v31 else "-"
        )
        conditional_vs_v32 = (
            f"{(selected['conditional_within_5']-v32['conditional_within_5'])*100:+.2f} pp"
            if v32 else "-"
        )
        mae_vs_v32 = (
            f"{selected['conditional_mae']-v32['conditional_mae']:+.3f}"
            if v32 else "-"
        )
        lines.append(
            f"| {labels[level]} | {(selected['conditional_within_5']-v1['conditional_within_5'])*100:+.2f} pp | "
            f"{(selected['conditional_within_5']-v2['conditional_within_5'])*100:+.2f} pp | "
            f"{conditional_vs_v31} | "
            f"{conditional_vs_v32} | "
            f"{selected['conditional_mae']-v1['conditional_mae']:+.3f} | "
            f"{selected['conditional_mae']-v2['conditional_mae']:+.3f} | "
            f"{mae_vs_v31} | {mae_vs_v32} |"
        )

    lines += [
        "",
        "## 80px + 40px 年齡桶",
        "",
        "| 年齡 | 版本 | Conditional N | +/-5 | MAE | Mean prediction | Signed error |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for bucket in (label for _, _, label in BUCKETS):
        for version in paths:
            value = result["versions"][version]["age_buckets_80+40"][bucket]
            lines.append(
                f"| {bucket} | {version} | {value['detected']} | "
                f"{pct(value['conditional_within_5'])} | {number(value['conditional_mae'])} | "
                f"{number(value['mean_prediction'])} | {number(value['mean_signed_error'])} |"
            )

    lines += ["", "## 判讀", "", "部署決策需同時依據 80px + 40px、各年齡桶與 frozen E2E 結果。",
              "27px 的 Conditional 樣本極少且受 YuNet detector gate 主導，不用於 Age 模型選擇。"]
    args.markdown.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.json}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
