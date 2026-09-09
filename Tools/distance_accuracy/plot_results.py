#!/usr/bin/env python3
"""4-panel Before/After chart per test plan §11.1 (grey=Before, blue=After)."""
import argparse
import json
import math
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LEVELS = [80, 40, 27]
X = [1, 2, 3]  # meters


def keyed(path):
    d = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        d[(r['dataset'], r['target_face_height_px'])].append(r)
    return d


def circ(p, t):
    return abs(((p - t + 180) % 360) - 180)


def yunet_rate(rows):
    n = len(rows)
    k = sum(1 for r in rows if r['failure_reason'] is None
            and r['iou'] >= 0.5 and (r['yunet_score'] or 0) >= 0.75)
    return k, n


def age_rate(rows):
    n = len(rows)
    k = sum(1 for r in rows if r['failure_reason'] is None
            and r['predicted_age'] is not None
            and abs(r['predicted_age'] - r['gt_age']) <= 5)
    return k, n


def gender_bacc(rows, gender_positive):
    per_class = defaultdict(lambda: [0, 0])
    for r in rows:
        gt = r['gt_gender']
        per_class[gt][1] += 1
        if r['failure_reason'] is None and r['predicted_gender'] is not None:
            positive_label = 0 if gender_positive == 'male' else 1
            pred = positive_label if r['predicted_gender'] >= 0.5 else 1 - positive_label
            if pred == gt:
                per_class[gt][0] += 1
    recalls = [c / t for c, t in per_class.values() if t]
    return (sum(recalls) / len(recalls), len(rows))


def pose_rate(rows):
    n = len(rows)
    k = sum(1 for r in rows if r['failure_reason'] is None
            and r['predicted_pitch'] is not None
            and all(e <= 15 for e in (circ(r['predicted_pitch'], r['gt_pitch']),
                                      circ(r['predicted_yaw'], r['gt_yaw']),
                                      circ(r['predicted_roll'], r['gt_roll']))))
    return k, n


def wilson(k, n, z=1.96):
    if not n:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


panels = [
    ('YuNet detection success', 'scface', yunet_rate, 'rate'),
    ('Age ±5yr hit rate (E2E)', 'utkface', age_rate, 'rate'),
    ('Gender balanced accuracy (E2E)', 'utkface', gender_bacc, 'bacc'),
    ('HeadPose ±15° 3-axis hit (E2E)', 'aflw2000', pose_rate, 'rate'),
]

ap = argparse.ArgumentParser()
ap.add_argument('--before', default='eval_before.jsonl')
ap.add_argument('--after', default='eval_after.jsonl')
ap.add_argument('--out', default='distance_accuracy_before_after.png')
ap.add_argument('--gender-positive', choices=['male', 'female'], default='male')
args = ap.parse_args()

before, after = keyed(args.before), keyed(args.after)
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=False)
for ax, (title, ds, fn, kind) in zip(axes, panels):
    for data, color, label in [(before, 'grey', 'Before (1b439be)'),
                               (after, 'tab:blue', 'After (current)')]:
        ys, cis, ns = [], [], []
        for lv in LEVELS:
            rows = data.get((ds, lv), [])
            if not rows:
                ys.append(float('nan'))
                cis.append((0.0, 0.0))
                ns.append(0)
                continue
            if fn is gender_bacc:
                result = fn(rows, args.gender_positive)
            else:
                result = fn(rows)
            if kind == 'rate':
                k, n = result
                lo, hi = wilson(k, n)
                ys.append(100 * k / n)
                cis.append((100 * (k / n - lo), 100 * (hi - k / n)))
            else:
                acc, n = result
                ys.append(100 * acc)
                lo, hi = wilson(int(round(acc * n)), n)
                cis.append((100 * (acc - lo), 100 * (hi - acc)))
            ns.append(n)
        err = list(zip(*cis))
        ax.errorbar(X, ys, yerr=err, color=color, marker='o', capsize=3, label=label)
        for x, y, n in zip(X, ys, ns):
            ax.annotate(f'{y:.0f}%\nN={n}', (x, y), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=7, color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(X, [f'1m\n(80px)', f'2m\n(40px)', f'3m\n(27px)'])
    ax.set_ylim(0, 105)
    ax.set_ylabel('%')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc='upper right')
fig.suptitle('AI Camera Before vs After — distance accuracy (offline pixel degradation, 640×480, HFOV 83°)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(args.out, dpi=150)
print(f'saved {args.out}')
