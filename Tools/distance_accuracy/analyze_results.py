#!/usr/bin/env python3
"""Compute E2E or GT-crop metrics from eval JSONL results.

Metrics per (dataset, face-height level), end-to-end (YuNet miss = downstream fail):
  - YuNet (scface):  detection success = IoU>=0.5 & score>=0.75
  - Age (utkface):   |pred-gt| <= age-tol hit rate (default 5, override with --age-tol)
  - Gender (utkface): balanced accuracy (mean per-class recall)
  - HeadPose (aflw2000):
      --pose-mode 3axis (default): all of |pitch|,|yaw|,|roll| circular error <= 15 deg
      --pose-mode 2axis: Yaw + Pitch only (Roll dropped).
        hit = circ_err(yaw) <= --pose-yaw-tol
              AND (pred_pitch - gt_pitch) in [--pose-pitch-low, --pose-pitch-high]
        Defaults for 2axis (per agreed spec): yaw tol 25°, pitch signed range [-40, +25]
        (== strict hit [yaw<=15, pitch in [-30,+15]] OR near-hit within 10° of that
        boundary, which algebraically collapses to the single extended window above).
"""
import argparse
import json
import math
from collections import defaultdict

LEVELS = [80, 40, 27]
DIST = {80: '1m', 40: '2m', 27: '3m'}


def parse_heights(spec: str):
    """Parse '--heights' spec like '115:1m,57:2m,38:3m' -> (levels, dist_labels)."""
    levels, dist = [], {}
    for part in spec.split(','):
        h, label = part.split(':')
        h = int(h)
        levels.append(h)
        dist[h] = label
    return levels, dist


def load(path):
    rows = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        rows[(r['dataset'], r['target_face_height_px'])].append(r)
    return rows


def circ_err(pred, gt):
    return abs(((pred - gt + 180) % 360) - 180)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def pct(k, n):
    return 100.0 * k / n if n else 0.0


def metrics(rows, gender_positive, age_tol=5, pose_mode='3axis',
            pose_yaw_tol=25.0, pose_pitch_low=-40.0, pose_pitch_high=25.0):
    out = {}
    # --- YuNet on scface ---
    sc = rows.get(('scface', None), [])
    if sc:
        n = len(sc)
        k = sum(1 for r in sc if r['failure_reason'] is None
                and r['iou'] >= 0.5 and (r['yunet_score'] or 0) >= 0.75)
        out['yunet'] = (k, n)
    # --- Age/Gender on utkface ---
    utk = rows.get(('utkface', None), [])
    if utk:
        n = len(utk)
        evaluated_tasks = set().union(*(
            set(r.get('evaluated_tasks', ('age', 'gender', 'pose'))) for r in utk
        ))
        if 'age' in evaluated_tasks:
            age_k = sum(1 for r in utk if r['failure_reason'] is None
                        and r['predicted_age'] is not None
                        and abs(r['predicted_age'] - r['gt_age']) <= age_tol)
            out['age'] = (age_k, n)
        # UTKFace labels: 0=male, 1=female. The model output mapping is explicit.
        if 'gender' in evaluated_tasks:
            per_class = defaultdict(lambda: [0, 0])
            for r in utk:
                gt = r['gt_gender']
                per_class[gt][1] += 1
                if r['failure_reason'] is None and r['predicted_gender'] is not None:
                    positive_label = 0 if gender_positive == 'male' else 1
                    pred = positive_label if r['predicted_gender'] >= 0.5 else 1 - positive_label
                    if pred == gt:
                        per_class[gt][0] += 1
            recalls = [c / t for c, t in per_class.values() if t]
            out['gender'] = (sum(recalls) / len(recalls) if recalls else 0.0, n)
    # --- HeadPose on aflw2000 ---
    af = rows.get(('aflw2000', None), [])
    if af:
        n = len(af)
        k = 0
        for r in af:
            if r['failure_reason'] is not None or r['predicted_pitch'] is None:
                continue
            if pose_mode == '2axis':
                yaw_err = circ_err(r['predicted_yaw'], r['gt_yaw'])
                # signed pitch error, wrapped to (-180, 180]
                pitch_signed_err = ((r['predicted_pitch'] - r['gt_pitch'] + 180) % 360) - 180
                if yaw_err <= pose_yaw_tol and pose_pitch_low <= pitch_signed_err <= pose_pitch_high:
                    k += 1
            else:
                errs = [circ_err(r['predicted_pitch'], r['gt_pitch']),
                        circ_err(r['predicted_yaw'], r['gt_yaw']),
                        circ_err(r['predicted_roll'], r['gt_roll'])]
                if all(e <= 15 for e in errs):
                    k += 1
        out['pose'] = (k, n)
    return out


def age_gender_diagnostics(rows, gender_positive):
    age_errors = [abs(r['predicted_age'] - r['gt_age']) for r in rows
                  if r['failure_reason'] is None and r['predicted_age'] is not None]
    positive_label = 0 if gender_positive == 'male' else 1
    recalls = {}
    for label, name in ((0, 'male'), (1, 'female')):
        class_rows = [r for r in rows if r['gt_gender'] == label]
        correct = 0
        for r in class_rows:
            if r['failure_reason'] is not None or r['predicted_gender'] is None:
                continue
            pred = positive_label if r['predicted_gender'] >= 0.5 else 1 - positive_label
            correct += pred == label
        recalls[name] = (correct, len(class_rows))
    return (sum(age_errors) / len(age_errors) if age_errors else None,
            len(age_errors), recalls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', help='Analyze one model/platform result file.')
    ap.add_argument('--before', help='Before result file for a comparison.')
    ap.add_argument('--after', help='After result file for a comparison.')
    ap.add_argument(
        '--heights', default='80:1m,40:2m,27:3m',
        help="'height_px:label' pairs, comma-separated. Default is HFOV 83°. "
             "Use '115:1m,57:2m,38:3m' for HFOV 63°.")
    ap.add_argument('--age-tol', type=float, default=5,
                     help='Age hit tolerance in years (default 5).')
    ap.add_argument('--gender-positive', choices=['male', 'female'], default='male',
                    help='Class represented by raw model output >=0.5 (default: male).')
    ap.add_argument('--pose-mode', choices=['3axis', '2axis'], default='3axis',
                     help='3axis = Pitch/Yaw/Roll all <=15deg (legacy). '
                          '2axis = Yaw/Pitch only, see --pose-yaw-tol/--pose-pitch-low/-high.')
    ap.add_argument('--pose-yaw-tol', type=float, default=25.0,
                     help='2axis mode: Yaw circular error hit threshold (default 25).')
    ap.add_argument('--pose-pitch-low', type=float, default=-40.0,
                     help='2axis mode: min signed pitch error (pred-gt) for hit (default -40).')
    ap.add_argument('--pose-pitch-high', type=float, default=25.0,
                     help='2axis mode: max signed pitch error (pred-gt) for hit (default +25).')
    args = ap.parse_args()
    if not args.result and not (args.before and args.after):
        ap.error('provide --result, or provide both --before and --after')
    if args.result and (args.before or args.after):
        ap.error('--result cannot be combined with --before/--after')
    global LEVELS, DIST
    LEVELS, DIST = parse_heights(args.heights)

    # flatten rows keyed by (dataset, level)
    def keyed(path):
        d = defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            d[(r['dataset'], r['target_face_height_px'])].append(r)
        return d

    result = keyed(args.result) if args.result else None
    before = keyed(args.before) if args.before else None
    after = keyed(args.after) if args.after else None

    active_sets = [data for data in (result, before, after) if data is not None]
    crop_sources = {
        row.get('crop_source', 'yunet')
        for data in active_sets for rows in data.values() for row in rows
    }
    if len(crop_sources) != 1:
        ap.error(f'result files use incompatible crop sources: {sorted(crop_sources)}')
    crop_source = next(iter(crop_sources), 'yunet')
    evaluation_label = 'GT crop' if crop_source == 'gt' else 'E2E'

    print(f'# gender mapping: raw>=0.5 -> {args.gender_positive} (explicit)')

    title = {
        'yunet': 'YuNet det rate (IoU>=0.5,score>=0.75)',
        'age': f'Age ±{args.age_tol:g}yr hit rate ({evaluation_label})',
        'gender': f'Gender balanced acc ({evaluation_label})',
        'pose': (f'HeadPose ±15° 3-axis hit ({evaluation_label})'
                 if args.pose_mode == '3axis'
                 else f'HeadPose 2-axis hit ({evaluation_label}) [yaw<={args.pose_yaw_tol:g}°, '
                      f'pitch∈[{args.pose_pitch_low:g}°,{args.pose_pitch_high:g}°]]'),
    }
    ds_of = {'yunet': 'scface', 'age': 'utkface', 'gender': 'utkface', 'pose': 'aflw2000'}

    if result is not None:
        print('\n| 模型 | 距離(臉高) | N | 結果 | 95% CI |')
        print('|---|---|---:|---:|---|')
    else:
        print('\n| 模型 | 距離(臉高) | N | Before | After | Δ (pp) | 95% CI(B) | 95% CI(A) |')
        print('|---|---|---:|---:|---:|---:|---|---|')
    for key in ['yunet', 'age', 'gender', 'pose']:
        ds = ds_of[key]
        for lv in LEVELS:
            if result is not None:
                measured = metrics({(ds, None): result.get((ds, lv), [])},
                                   args.gender_positive, age_tol=args.age_tol,
                                   pose_mode=args.pose_mode,
                                   pose_yaw_tol=args.pose_yaw_tol,
                                   pose_pitch_low=args.pose_pitch_low,
                                   pose_pitch_high=args.pose_pitch_high)
                if key not in measured:
                    continue
                value, n = measured[key]
                if key == 'gender':
                    print(f'| {title[key]} | {DIST[lv]}({lv}px) | {n} | {value*100:.1f}% | - |')
                else:
                    lo, hi = wilson(value, n)
                    print(f'| {title[key]} | {DIST[lv]}({lv}px) | {n} | {pct(value,n):.1f}% | [{lo*100:.1f},{hi*100:.1f}] |')
                continue
            mb = metrics({(ds, None): before.get((ds, lv), [])}, args.gender_positive,
                         age_tol=args.age_tol, pose_mode=args.pose_mode,
                         pose_yaw_tol=args.pose_yaw_tol,
                         pose_pitch_low=args.pose_pitch_low,
                         pose_pitch_high=args.pose_pitch_high)
            ma = metrics({(ds, None): after.get((ds, lv), [])}, args.gender_positive,
                         age_tol=args.age_tol, pose_mode=args.pose_mode,
                         pose_yaw_tol=args.pose_yaw_tol,
                         pose_pitch_low=args.pose_pitch_low,
                         pose_pitch_high=args.pose_pitch_high)
            if key not in mb or key not in ma:
                continue
            if key == 'gender':
                b_acc, n = mb[key]
                a_acc, _ = ma[key]
                print(f'| {title[key]} | {DIST[lv]}({lv}px) | {n} | {b_acc*100:.1f}% | {a_acc*100:.1f}% | {(a_acc-b_acc)*100:+.1f} | - | - |')
            else:
                kb, n = mb[key]
                ka, _ = ma[key]
                lb, ub = wilson(kb, n)
                la, ua = wilson(ka, n)
                print(f'| {title[key]} | {DIST[lv]}({lv}px) | {n} | {pct(kb,n):.1f}% | {pct(ka,n):.1f}% | {pct(ka,n)-pct(kb,n):+.1f} | [{lb*100:.1f},{ub*100:.1f}] | [{la*100:.1f},{ua*100:.1f}] |')

    if result is not None:
        print('\n# Age/Gender diagnostics:')
        print('| 距離(臉高) | Age MAE | Age valid | Male recall | Female recall |')
        print('|---|---:|---:|---:|---:|')
        for lv in LEVELS:
            rows = result.get(('utkface', lv), [])
            if not rows:
                continue
            age_mae, age_n, recalls = age_gender_diagnostics(rows, args.gender_positive)
            male_k, male_n = recalls['male']
            female_k, female_n = recalls['female']
            age_mae_text = f'{age_mae:.2f}' if age_mae is not None else '-'
            print(f'| {DIST[lv]}({lv}px) | {age_mae_text} | {age_n}/{len(rows)} | '
                  f'{pct(male_k,male_n):.1f}% ({male_k}/{male_n}) | '
                  f'{pct(female_k,female_n):.1f}% ({female_k}/{female_n}) |')

    # extra: detection breakdown for downstream datasets (context)
    if crop_source == 'gt':
        return
    print('\n# YuNet detection rate on downstream datasets (context, IoU>=0.5 & score>=0.75):')
    for ds in ['utkface', 'aflw2000']:
        for lv in LEVELS:
            datasets = [('result', result)] if result is not None else [('before', before), ('after', after)]
            for tag, data in datasets:
                rows = data.get((ds, lv), [])
                if not rows:
                    continue
                k = sum(1 for r in rows if r['iou'] >= 0.5
                        and (r['yunet_score'] or 0) >= 0.75)
                print(f'  {ds:9s} {DIST[lv]:>3s} {tag}: {pct(k,len(rows)):.1f}% ({k}/{len(rows)})')


if __name__ == '__main__':
    main()
