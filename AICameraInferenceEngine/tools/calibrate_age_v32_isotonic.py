#!/usr/bin/env python3
"""Cross-fit one monotonic isotonic calibration for Age v3.2."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


BUCKETS = (
    (0, 9, "0-9"), (10, 19, "10-19"), (20, 29, "20-29"),
    (30, 39, "30-39"), (40, 49, "40-49"), (50, 59, "50-59"),
    (60, 69, "60-69"), (70, 79, "70-79"), (80, 200, "80+"),
)


def fit_isotonic(x, y):
    order = np.argsort(x, kind="stable")
    sorted_x, sorted_y = x[order], y[order]
    unique_x, starts, counts = np.unique(
        sorted_x, return_index=True, return_counts=True
    )
    means = np.add.reduceat(sorted_y, starts) / counts
    blocks = []
    for index, (mean, count) in enumerate(zip(means, counts)):
        blocks.append([index, index, float(count), float(mean * count)])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[3] / left[2] <= right[3] / right[2]:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    fitted = np.empty(len(unique_x), np.float64)
    for start, end, weight, total in blocks:
        fitted[start:end + 1] = total / weight
    return unique_x.astype(np.float64), fitted


def apply_isotonic(values, knots_x, knots_y):
    return np.interp(values, knots_x, knots_y, left=knots_y[0], right=knots_y[-1])


def metrics(rows, predictions):
    labels = np.asarray([row["age"] for row in rows], np.float64)

    def summarize(mask):
        errors = predictions[mask] - labels[mask]
        return {
            "count": int(mask.sum()),
            "within_5_years": float(np.mean(np.abs(errors) <= 5)),
            "mae": float(np.mean(np.abs(errors))),
            "mean_prediction": float(np.mean(predictions[mask])),
            "mean_signed_error": float(np.mean(errors)),
        }

    result = summarize(np.ones(len(rows), bool))
    result["by_variant"] = {}
    for variant in ("80px", "40px"):
        mask = np.asarray([row["variant"] == variant for row in rows])
        result["by_variant"][variant] = summarize(mask)
    result["age_buckets"] = {}
    for low, high, name in BUCKETS:
        mask = (labels >= low) & (labels <= high)
        result["age_buckets"][name] = summarize(mask)
    return result


def fold_for(source_id, folds):
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    rows = []
    for line in args.manifest.read_text().splitlines():
        row = json.loads(line)
        if row["variant"] not in ("80px", "40px"):
            continue
        row["file"] = str((args.cache_root / row["file"]).resolve())
        rows.append(row)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    raw = []
    for row in rows:
        image = cv2.imread(row["file"], cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
        raw.append(float(session.run(
            None, {input_name: np.transpose(image[None], (0, 3, 1, 2))}
        )[0].ravel()[0]))
    raw = np.asarray(raw, np.float64)
    labels = np.asarray([row["age"] for row in rows], np.float64)
    folds = np.asarray([fold_for(row["source_id"], args.folds) for row in rows])
    calibrated = np.empty_like(raw)
    fold_audit = []
    for fold in range(args.folds):
        fit = folds != fold
        evaluate = folds == fold
        fit_sources = {rows[index]["source_id"] for index in np.flatnonzero(fit)}
        eval_sources = {rows[index]["source_id"] for index in np.flatnonzero(evaluate)}
        if fit_sources & eval_sources:
            raise RuntimeError(f"source leakage in fold {fold}")
        knots_x, knots_y = fit_isotonic(raw[fit], labels[fit])
        calibrated[evaluate] = apply_isotonic(raw[evaluate], knots_x, knots_y)
        fold_audit.append({
            "fold": fold,
            "fit_rows": int(fit.sum()), "evaluation_rows": int(evaluate.sum()),
            "fit_sources": len(fit_sources), "evaluation_sources": len(eval_sources),
            "source_overlap": 0, "knots": len(knots_x),
        })

    raw_metrics = metrics(rows, raw)
    calibrated_metrics = metrics(rows, calibrated)
    overall_gain_pp = 100 * (
        calibrated_metrics["within_5_years"] - raw_metrics["within_5_years"]
    )
    overall_mae_gain = raw_metrics["mae"] - calibrated_metrics["mae"]
    high_age_pass = all(
        (
            100 * (calibrated_metrics["age_buckets"][bucket]["within_5_years"]
                   - raw_metrics["age_buckets"][bucket]["within_5_years"]) >= 2.0
            or raw_metrics["age_buckets"][bucket]["mae"]
            - calibrated_metrics["age_buckets"][bucket]["mae"] >= 0.5
        ) for bucket in ("70-79", "80+")
    )
    young_guardrail = all(
        calibrated_metrics["age_buckets"][bucket]["within_5_years"]
        >= raw_metrics["age_buckets"][bucket]["within_5_years"] - 0.02
        for bucket in ("0-9", "10-19", "20-29")
    )
    passed = (
        (overall_gain_pp >= 2.0 or overall_mae_gain >= 0.5)
        and high_age_pass and young_guardrail
    )
    final_x, final_y = fit_isotonic(raw, labels)
    result = {
        "method": "source-grouped 5-fold cross-fitted isotonic regression (PAVA)",
        "selection_data": "v3 leakage-free validation runtime crops only",
        "rows": len(rows), "sources": len({row["source_id"] for row in rows}),
        "fold_audit": fold_audit,
        "raw": raw_metrics, "cross_fitted_calibrated": calibrated_metrics,
        "decision": {
            "overall_within_5_gain_pp": overall_gain_pp,
            "overall_mae_gain_years": overall_mae_gain,
            "high_age_pass": high_age_pass,
            "young_guardrail_pass": young_guardrail,
            "passes_hard_stop": passed,
        },
        "residual_risk": "Neural O1/O2 selection already used this validation split; cross-fitting prevents calibrator fit/evaluation overlap but does not create a new neural-model holdout.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    parameters = {
        "enabled": False,
        "method": "linear-interpolated isotonic PAVA",
        "input": "raw scalar age", "output": "calibrated scalar age",
        "fit_rows": len(rows), "fit_sources": result["sources"],
        "knots_x": final_x.tolist(), "knots_y": final_y.tolist(),
        "frozen_for_benchmark": passed,
    }
    args.parameters.parent.mkdir(parents=True, exist_ok=True)
    args.parameters.write_text(json.dumps(parameters, indent=2) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
