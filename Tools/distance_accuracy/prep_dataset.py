#!/usr/bin/env python3
"""Build offline pixel-degradation test set per AI_camera_distance_accuracy_test_plan §6.1.

For each source sample, scales the image so the ground-truth face height equals
the target pixel height (default 80px=1m, 40px=2m, 27px=3m @ 640x480, HFOV 83°;
override with --heights for other HFOV, e.g. 115px=1m/57px=2m/38px=3m @ HFOV 63°),
then places it centered on a fixed black 640x480 canvas (INTER_AREA downscale,
no sharpening/SR). GT bbox is recorded in canvas coordinates.

Datasets:
  - SCFace mugshot_frontal_cropped_all -> YuNet (GT bbox from eye/nose/mouth landmarks)
  - UTKFace crop_part1 -> Age/Gender (stratified sample)
  - AFLW2000 -> Head Pose (GT bbox from 68 landmarks, pose from Pose_Para)
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

CANVAS_W, CANVAS_H = 640, 480
TARGET_HEIGHTS = [80, 40, 27]  # 1m, 2m, 3m
DISTANCE_OF = {80: 1.0, 40: 2.0, 27: 3.0}

SCFACE_DIR = Path('/Users/yung/kaggle/SCface_database')
UTK_DIR = Path('/Users/yung/kaggle/utkface-new/crop_part1')
AFLW_DIR = Path('/Users/yung/kaggle/AFLW2000')
OUT_DIR = Path('/Users/yung/kaggle/_distance_test_degraded')

UTK_SAMPLE = 1000
SEED = 42


def load_scface_landmarks() -> dict:
    """mugshot_frontal_cropped.txt: name, LEx LEy, REx REy, Nx Ny, Mx My (1600x1200 crops)."""
    table = {}
    with open(SCFACE_DIR / 'mugshot_frontal_cropped.txt') as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 9:
                table[parts[0]] = [float(v) for v in parts[1:]]
    return table


def scface_gt_bbox(pts):
    """Documented rule: face box from landmarks.
    cx = eye midpoint; w = 2.2 * inter-eye; top = eye_y - 1.2*d; bottom = mouth_y + 0.5*d."""
    lex, ley, rex, rey, _nx, _ny, mx, my = pts
    d = abs(rex - lex)
    cx = (lex + rex) / 2.0
    eye_y = (ley + rey) / 2.0
    x1, x2 = cx - 1.1 * d, cx + 1.1 * d
    y1, y2 = eye_y - 1.2 * d, my + 0.5 * d
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def aflw_gt_bbox(mat) -> np.ndarray:
    """GT bbox from 68 landmarks, expanded x1.3 (w) / x1.35 (h) to cover forehead/chin."""
    pt = mat['pt3d_68']
    x1, y1, x2, y2 = pt[0].min(), pt[1].min(), pt[0].max(), pt[1].max()
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w *= 1.3
    h *= 1.35
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


def place_on_canvas(img: np.ndarray, gt_bbox: np.ndarray, target_h: int):
    """Scale so GT face height == target_h; paste face center at canvas center; clip."""
    face_h = gt_bbox[3] - gt_bbox[1]
    if face_h < 5:
        return None, None
    s = target_h / face_h
    new_w = max(1, int(round(img.shape[1] * s)))
    new_h = max(1, int(round(img.shape[0] * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    scaled = cv2.resize(img, (new_w, new_h), interpolation=interp)
    gt = gt_bbox * s
    cx, cy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    ox, oy = CANVAS_W / 2 - cx, CANVAS_H / 2 - cy  # paste offset
    # source rect that lands on canvas
    sx1 = max(0, -ox)
    sy1 = max(0, -oy)
    sx2 = min(new_w, CANVAS_W - ox)
    sy2 = min(new_h, CANVAS_H - oy)
    if sx2 <= sx1 or sy2 <= sy1:
        return None, None
    dx1, dy1 = int(round(ox + sx1)), int(round(oy + sy1))
    sx1i, sy1i, sx2i, sy2i = int(round(sx1)), int(round(sy1)), int(round(sx2)), int(round(sy2))
    canvas[dy1:dy1 + (sy2i - sy1i), dx1:dx1 + (sx2i - sx1i)] = scaled[sy1i:sy2i, sx1i:sx2i]
    gt_canvas = gt + np.array([ox, oy, ox, oy])
    return canvas, gt_canvas


def write_sample(canvas, gt_canvas, out_path, record):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    record['file'] = str(out_path)
    record['gt_bbox'] = [round(float(v), 2) for v in gt_canvas]
    return record


def build_scface(targets):
    lm = load_scface_landmarks()
    src_dir = SCFACE_DIR / 'mugshot_frontal_cropped_all'
    rows = []
    for name, pts in sorted(lm.items()):
        img_path = src_dir / f'{name}.JPG'
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt = scface_gt_bbox(pts)
        for th in targets:
            canvas, gtc = place_on_canvas(img, gt, th)
            if canvas is None:
                continue
            rows.append(write_sample(
                canvas, gtc, OUT_DIR / f'h{th}' / 'scface' / f'{name}.jpg',
                {'dataset': 'scface', 'source_id': name,
                 'distance_m': DISTANCE_OF[th], 'target_face_height_px': th}))
    return rows


def parse_utk_label(name):
    p = name.split('_')
    if len(p) < 4:
        return None
    try:
        age, gender = int(p[0]), int(p[1])
    except ValueError:
        return None
    if not (0 <= age <= 116 and gender in (0, 1)):
        return None
    return age, gender


def build_utk(targets, sample_n):
    rows_all = []
    for f in sorted(UTK_DIR.glob('*.jpg')):
        label = parse_utk_label(f.name)
        if label:
            rows_all.append((f, label[0], label[1]))
    rng = random.Random(SEED)
    buckets = {}
    for f, age, gender in rows_all:
        buckets.setdefault((min(age // 10, 10), gender), []).append((f, age, gender))
    sampled = []
    per_bucket = max(1, sample_n // len(buckets))
    for key in sorted(buckets):
        items = buckets[key]
        rng.shuffle(items)
        sampled.extend(items[:per_bucket])
    rows = []
    for f, age, gender in sampled:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = np.array([0, 0, w, h], dtype=np.float64)  # chip is a tight face crop
        for th in targets:
            canvas, gtc = place_on_canvas(img, gt, th)
            if canvas is None:
                continue
            rows.append(write_sample(
                canvas, gtc, OUT_DIR / f'h{th}' / 'utkface' / f.name,
                {'dataset': 'utkface', 'source_id': f.name,
                 'distance_m': DISTANCE_OF[th], 'target_face_height_px': th,
                 'gt_age': age, 'gt_gender': gender}))
    return rows


def build_aflw(targets):
    rows = []
    for img_path in sorted(AFLW_DIR.glob('*.jpg')):
        mat_path = img_path.with_suffix('.mat')
        if not mat_path.exists():
            continue
        mat = sio.loadmat(str(mat_path))
        pose = np.degrees(mat['Pose_Para'].ravel()[:3])  # pitch, yaw, roll
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt = aflw_gt_bbox(mat)
        if (gt[3] - gt[1]) < 40:  # degenerate landmark box
            continue
        for th in targets:
            canvas, gtc = place_on_canvas(img, gt, th)
            if canvas is None:
                continue
            rows.append(write_sample(
                canvas, gtc, OUT_DIR / f'h{th}' / 'aflw2000' / img_path.name,
                {'dataset': 'aflw2000', 'source_id': img_path.name,
                 'distance_m': DISTANCE_OF[th], 'target_face_height_px': th,
                 'gt_pitch': round(float(pose[0]), 3),
                 'gt_yaw': round(float(pose[1]), 3),
                 'gt_roll': round(float(pose[2]), 3)}))
    return rows


def parse_heights(spec: str):
    """Parse '--heights' spec like '115:1.0,57:2.0,38:3.0' -> (targets, distance_of)."""
    targets, distance_of = [], {}
    for part in spec.split(','):
        h, d = part.split(':')
        h = int(h)
        targets.append(h)
        distance_of[h] = float(d)
    return targets, distance_of


def main():
    global SCFACE_DIR, UTK_DIR, AFLW_DIR, OUT_DIR, DISTANCE_OF
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(OUT_DIR))
    ap.add_argument('--scface-dir', default=str(SCFACE_DIR),
                    help='SCFace root containing mugshot_frontal_cropped.txt.')
    ap.add_argument('--utk-dir', default=str(UTK_DIR),
                    help='UTKFace cropped image directory, usually crop_part1.')
    ap.add_argument('--aflw-dir', default=str(AFLW_DIR),
                    help='AFLW2000 directory containing matching .jpg and .mat files.')
    ap.add_argument('--utk-sample', type=int, default=UTK_SAMPLE)
    ap.add_argument(
        '--heights', default='80:1.0,40:2.0,27:3.0',
        help="'height_px:distance_m' pairs, comma-separated. "
             "Default is HFOV 83° theoretical values (1/2/3m). "
             "Use '115:1.0,57:2.0,38:3.0' for HFOV 63°.")
    args = ap.parse_args()
    SCFACE_DIR = Path(args.scface_dir)
    UTK_DIR = Path(args.utk_dir)
    AFLW_DIR = Path(args.aflw_dir)
    OUT_DIR = Path(args.out)
    targets, DISTANCE_OF = parse_heights(args.heights)
    all_rows = build_scface(targets) + build_utk(targets, args.utk_sample) + build_aflw(targets)
    manifest = OUT_DIR / 'manifest.jsonl'
    with open(manifest, 'w') as fh:
        for row in all_rows:
            fh.write(json.dumps(row) + '\n')
    print(f'wrote {len(all_rows)} samples -> {manifest}')


if __name__ == '__main__':
    main()
