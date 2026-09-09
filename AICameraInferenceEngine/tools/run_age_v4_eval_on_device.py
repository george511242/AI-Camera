#!/usr/bin/env python3
"""Age/Gender/Head Pose accuracy harness, runs on the Orange Pi RKNN NPU.

Supports end-to-end YuNet crops and model-only ground-truth crops. In GT mode,
SCFace rows are skipped because they have no downstream task.

Works against both Before (1b439be) and After (current) trees: pass --margins
matching each version's factory defaults.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from inference.model import Model, Result
from inference.rknn.lightweight_head_pose_estimation import LightweightHeadPoseEstimation
from inference.rknn.model import RKNNModel
from inference.rknn.ssrnet import SSRNet
from inference.rknn.yunet import YuNet
from inference.utils import img_utils

CANVAS_W, CANVAS_H = 640, 480
DEFAULT_MODEL_ROOT = 'model/versions/v1_baseline'
DEFAULT_MODEL_NAMES = {
    'yunet': 'yunet_n_640_640_fp16.rknn',
    'age': 'age_ssrnet_wiki_64x64_fp16.rknn',
    'gender': 'gender_ssrnet_wiki_64x64_fp16.rknn',
    'pose': 'head_pose_lightweight_b66_224x224_fp16.rknn',
}


class AgeV4(RKNNModel):
    """Benchmark-only adapter for the 112x112 RGB scalar Age v4 contract."""

    def __init__(self, weight_path, margin):
        super().__init__(weight_path)
        self.margin = margin

    def __call__(self, image, results):
        for result in results:
            face = img_utils.crop(image, result.bbox, self.margin)
            face = img_utils.resize(face, (112, 112))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
            face = img_utils.to_batch([face])
            prediction = np.reshape(self.session.inference([face]), -1)[0]
            result.set(age=prediction)
        return results


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data-root',
                    help='Override the generated dataset root on this device.')
    ap.add_argument('--crop-source', choices=['yunet', 'gt'], default='yunet',
                    help='Use YuNet detections (E2E) or manifest GT bbox (model-only).')
    ap.add_argument('--version', required=True,
                    help='Label stored in each result row, for example v1-rk3588.')
    ap.add_argument('--model-root', default=DEFAULT_MODEL_ROOT,
                    help='Model version directory containing rknn/<platform>.')
    ap.add_argument('--platform', required=True, choices=['rk3566', 'rk3588'])
    ap.add_argument('--yunet-model', default=DEFAULT_MODEL_NAMES['yunet'])
    ap.add_argument('--age-model', default=DEFAULT_MODEL_NAMES['age'])
    ap.add_argument('--gender-model', default=DEFAULT_MODEL_NAMES['gender'])
    ap.add_argument('--pose-model', default=DEFAULT_MODEL_NAMES['pose'])
    ap.add_argument('--age-margin', type=float, required=True)
    ap.add_argument('--gender-margin', type=float, required=True)
    ap.add_argument('--pose-margin', type=float, required=True)
    args = ap.parse_args()

    model_dir = Path(args.model_root).resolve() / 'rknn' / args.platform
    required_models = set(DEFAULT_MODEL_NAMES)
    if args.crop_source == 'gt':
        required_models.remove('yunet')
    model_paths = {
        name: model_dir / getattr(args, f'{name}_model') for name in DEFAULT_MODEL_NAMES
    }
    missing = [str(model_paths[name]) for name in required_models
               if not model_paths[name].is_file()]
    if missing:
        ap.error('model file(s) not found: ' + ', '.join(missing))
    print(f'[{args.version}] platform={args.platform} model_dir={model_dir}', flush=True)
    print(f'[{args.version}] crop_source={args.crop_source}', flush=True)
    for name in sorted(required_models):
        path = model_paths[name]
        print(f'[{args.version}] {name}_model={path}', flush=True)

    yunet = YuNet(str(model_paths['yunet'])) if args.crop_source == 'yunet' else None
    age_m = AgeV4(str(model_paths['age']), margin=args.age_margin)
    gender_m = SSRNet(str(model_paths['gender']), Model.SSRNET_TARGET.GENDER,
                      margin=args.gender_margin)
    pose_m = LightweightHeadPoseEstimation(str(model_paths['pose']),
                                           margin=args.pose_margin)

    rows = [json.loads(line) for line in open(args.manifest)]
    if args.crop_source == 'gt':
        rows = [row for row in rows if row['dataset'] in ('utkface', 'aflw2000')]
    out_fh = open(args.out, 'w')
    t_all = time.time()
    for i, row in enumerate(rows, 1):
        rec = dict(row)
        rec['version'] = args.version
        rec['crop_source'] = args.crop_source
        rec['crop_bbox'] = None
        rec['detected_bbox'] = None
        rec['yunet_score'] = None
        rec['iou'] = 0.0
        rec['predicted_age'] = None
        rec['predicted_gender'] = None
        rec['predicted_pitch'] = None
        rec['predicted_yaw'] = None
        rec['predicted_roll'] = None
        rec['failure_reason'] = None
        try:
            image_path = Path(rec['file'])
            if args.data_root:
                image_path = (Path(args.data_root) / f"h{rec['target_face_height_px']}"
                              / rec['dataset'] / image_path.name)
            img = cv2.imread(str(image_path))
            if img is None:
                rec['failure_reason'] = 'runtime_error'
                out_fh.write(json.dumps(rec) + '\n')
                continue
            gt = np.array(rec['gt_bbox'], dtype=np.float64)
            gt_norm = gt / [CANVAS_W, CANVAS_H, CANVAS_W, CANVAS_H]
            if args.crop_source == 'gt':
                best = Result().set(bbox=gt_norm.tolist())
                rec['crop_bbox'] = [round(v, 5) for v in best.bbox]
            else:
                results = yunet(img, [])
                best, best_iou = None, 0.0
                for r in results:
                    ov = iou(np.array(r.bbox), gt_norm)
                    if ov > best_iou:
                        best, best_iou = r, ov
                if best is None:
                    rec['failure_reason'] = 'yunet_missed'
                    out_fh.write(json.dumps(rec) + '\n')
                    continue
                rec['detected_bbox'] = [round(v, 5) for v in best.bbox]
                rec['crop_bbox'] = rec['detected_bbox']
                rec['yunet_score'] = round(float(best.score), 5)
                rec['iou'] = round(best_iou, 5)
                if best_iou < 0.5 or float(best.score) < 0.75:
                    rec['failure_reason'] = ('wrong_face_matched' if best_iou < 0.5
                                             else 'yunet_score_below_threshold')
                    out_fh.write(json.dumps(rec) + '\n')
                    continue
            picked = [best]
            ds = rec['dataset']
            if ds == 'utkface':
                age_m(img, picked)
                gender_m(img, picked)
                rec['predicted_age'] = round(float(best.age), 3)
                rec['predicted_gender'] = round(float(best.gender), 5)
            elif ds == 'aflw2000':
                pose_m(img, picked)
                rec['predicted_pitch'] = round(float(np.ravel(best.pitch)[0]), 3)
                rec['predicted_yaw'] = round(float(np.ravel(best.yaw)[0]), 3)
                rec['predicted_roll'] = round(float(np.ravel(best.roll)[0]), 3)
            out_fh.write(json.dumps(rec) + '\n')
        except Exception as exc:  # keep sample in denominator
            rec['failure_reason'] = 'runtime_error'
            rec['error'] = str(exc)[:200]
            out_fh.write(json.dumps(rec) + '\n')
        if i % 250 == 0:
            el = time.time() - t_all
            print(f'[{args.version}] {i}/{len(rows)} elapsed {el:.0f}s', flush=True)
    out_fh.close()
    print(f'[{args.version}] DONE {len(rows)} rows in {time.time()-t_all:.0f}s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
