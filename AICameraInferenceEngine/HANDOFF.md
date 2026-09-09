# VAC AI Camera Handoff

Last updated: 2026-09-06 (Asia/Taipei)

## Current scope

Current work is YuNet small-face detection. Do not modify or retrain Age,
Gender, or Head Pose. Do not rerun frozen benchmarks or tune score thresholds.

- Age v2 remains production default.
- Age v4 is an opt-in candidate via `AGE_MODEL_VERSION=v4`.
- Production YuNet remains unchanged.
- YuNet `run1` was stopped after plateau/regression; do not resume it.
- No second YuNet training run has been launched.

Repository:
`/home/g2004/projects/VAC-AI-Camera-Source/vac-camera/vac-camera/AICameraInferenceEngine`

Branch: `master` tracking `origin/master`. The worktree contains many intentional
uncommitted model, tooling, runtime, test, and documentation changes. Do not
reset or delete them.

## YuNet diagnosis

Frozen detector success rates using the existing production YuNet:

| Dataset | 80px | 40px | 27px |
|---|---:|---:|---:|
| UTKFace | 914/936 (97.6%) | 614/936 (65.6%) | 15/936 (1.6%) |
| AFLW2000 | 1788/2000 (89.4%) | 1284/2000 (64.2%) | 547/2000 (27.4%) |
| SCFace | 130/130 (100%, where evaluated) | 130/130 (100%) | 107/130 (82.3%) |

Low-threshold inspection showed that most misses have usable localization but
insufficient confidence. UTKFace 27px also contains a strong synthetic-domain
artifact: a tight source chip becomes an isolated roughly 29x30 patch on a
black canvas, while SCFace/AFLW retain more head/background context. Conclusion:
domain artifact and genuine small-face confidence/scale sensitivity both
contribute. Do not train only on black-canvas UTKFace.

Fixed WIDER research subset:

- First deterministic 200 WIDER val images with 1-10 GT faces and at least one
  face whose projected 640-input height is 20-50px.
- 728 target faces.
- Score threshold 0.75, IoU >= 0.5, NMS IoU 0.3.
- Production baseline recall 37.64%, precision proxy 97.48%.
- Threshold 0.50 gave 44.09% recall / 90.99% precision proxy.
- Full frame plus four fixed tiles gave 60.71% / 96.98%, at about 4x cost.
- Do not rerun or tune these threshold/tiling experiments.

Full diagnosis:
`model/versions/yunet_v2_small_face/analysis/small_face_stop_point.zh-TW.md`

## Upstream and environment

Official source checkout:
`model/versions/yunet_v2_small_face/training/vendor/libfacedetection.train`

- Repository: `ShiqiYu/libfacedetection.train`
- Pinned commit: `dca340aa082c71081a68d17db8e58b33a58a914b`
- Upstream checkpoint: `weights/yunet_n.pth`
- Checkpoint SHA256:
  `9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06`

Dedicated conda environment:

- Name: `vac-yunet-gpu`
- Python: 3.11.16
- PyTorch: 2.11.0+cu128
- TorchVision: 0.26.0+cu128
- NumPy: 2.4.4
- OpenCV: 4.13.0
- SciPy: 1.17.1
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU, capability 12.0

Host WSL GPU works. Codex execution sandbox does not expose `/dev/dxg`; never
diagnose sandbox CUDA absence as a host WSL/NVIDIA failure. GPU commands must
run from the normal VS Code WSL terminal. Do not reinstall WSL, NVIDIA drivers,
CUDA, or PyTorch, and do not modify `pytorch-test`, `vac-age-v3`,
`rknn-toolkit2`, or conda base.

Smoke passed at batch 16/workers 0: peak reserved VRAM 2,558 MiB, process max
RSS about 2.25 GiB, one epoch plus validation wall time 6:08, no CUDA/WSL
instability.

## Pretrained initialization audit

`--init-weights` uses strict model loading. Independent verification:

- Checkpoint/model state keys: 208/208
- Missing keys: `[]`
- Unexpected keys: `[]`
- Loaded trainable parameters: 75,856/75,856
- Loaded state values including buffers: 77,890/77,890
- Every loaded tensor equals the upstream checkpoint

Run1 was correctly initialized from upstream and was not trained from scratch.

## Run1 configuration and state

Run directory:
`model/versions/yunet_v2_small_face/training/run1/`

Exact configuration:

```bash
/home/g2004/miniforge3/envs/vac-yunet-gpu/bin/python \
  -m yunet_train.cli.train \
  --variant yunet_n \
  --ann-file /home/g2004/datasets/vac-distance/source/widerface/labelv2/train/labelv2.txt \
  --img-prefix /home/g2004/datasets/vac-distance/source/widerface/WIDER_train/WIDER_train/images \
  --val-ann-file /home/g2004/datasets/vac-distance/source/widerface/labelv2/val/labelv2.txt \
  --val-img-prefix /home/g2004/datasets/vac-distance/source/widerface/WIDER_val/WIDER_val/images \
  --init-weights model/versions/yunet_v2_small_face/training/vendor/libfacedetection.train/weights/yunet_n.pth \
  --small-face-policy --seed 20260905 \
  --image-size 640 --min-face-size 10 \
  --epochs 40 --batch-size 16 --workers 0 --device cuda \
  --lr 0.001 --lr-steps 24 34 --warmup-iters 100 \
  --checkpoint-interval 5 --eval-interval 1 --no-tensorboard \
  --log-interval 50 \
  --work-dir model/versions/yunet_v2_small_face/training/run1
```

Small-face policy:

- Crop choices: 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 2.5
- Resolution: 60% clean, 25% 0.75x, 15% 0.5x
- Mild blur 15%, JPEG quality 75-95 at 20%, brightness/contrast 25%,
  Gaussian noise sigma 1-3 at 15%
- Training uses WIDER train 12,876 images; validation uses 3,226 images

Important checkpoints:

- `run1/best_loss.pth`: epoch 2, eval loss 3.096332
- `run1/epoch_5.pth`
- `run1/epoch_10.pth`
- `run1/epoch_15.pth`
- `run1/latest.pth`: completed epoch 16 training state

Run1 stopped during epoch 16 validation. There is no completed epoch-16 eval
checkpoint. Do not resume it.

Latest relevant metrics:

| Epoch | Train loss | Eval loss | LR |
|---|---:|---:|---:|
| 2 | 2.666233 | 3.096332 | 0.001 |
| 15 | 2.663561 | 3.300912 | 0.001 |
| 16 | 2.665918 | incomplete | 0.001 |

LR staying at 0.001 is expected: milestones are epochs 24 and 34 with gamma
0.1. The 100-iteration warmup finishes inside epoch 1; epoch logs show the
end-of-epoch LR.

Validation kps loss is always zero because validation labelv2 has bbox only:
39,697 validation faces and zero visible keypoints. Train has 159,393 faces,
76,006 with visible keypoints. This is expected, not a missing kps forward pass.

## Required checkpoint comparison

This requested comparison is already complete. Do not rerun it unless artifacts
are missing. Exact fixed threshold 0.75 / IoU 0.5 results:

| Model | 20-50px recall | Precision proxy | 20-30px | 30-40px | 40-50px |
|---|---:|---:|---:|---:|---:|
| Upstream `yunet_n.pth` | 37.64% | 98.16% | 22.13% | 42.81% | 48.45% |
| `run1/best_loss.pth` | 38.46% | 97.52% | 24.68% | 42.81% | 48.45% |
| `run1/epoch_15.pth` | 32.14% | 99.19% | 18.72% | 35.45% | 43.30% |

Best-loss gained only 0.82 pp overall, not a meaningful improvement. Epoch 15
regressed in every size bin. This justified stopping run1.

Reports/artifacts:

- `model/versions/yunet_v2_small_face/validation/checkpoint_screening/epoch_5_10.zh-TW.md`
- `model/versions/yunet_v2_small_face/validation/checkpoint_screening/epoch_15_stop_decision.zh-TW.md`
- Corresponding JSON files in the same directory

## Modified YuNet files

Project files added/updated for this stage:

- `tools/analyze_yunet_small_faces.py`
- `tools/evaluate_yunet_wider_tiling.py`
- `tools/evaluate_yunet_checkpoint_small_faces.py`
- `tools/preflight_yunet_gpu.py`
- `tools/run_yunet_v2_gpu_smoke.sh`
- `tools/run_yunet_v2_full_gpu.sh`
- `tools/run_yunet_v2_lr1e4_gpu.sh`
- `model/versions/yunet_v2_small_face/README.md`
- `model/versions/yunet_v2_small_face/analysis/`
- `model/versions/yunet_v2_small_face/training/PREPARATION.zh-TW.md`
- `model/versions/yunet_v2_small_face/training/UPSTREAM.md`
- `model/versions/yunet_v2_small_face/validation/checkpoint_screening/`
- `HANDOFF.md`

Pinned vendor checkout modifications:

- `pyproject.toml`: editable-install license metadata compatibility
- `yunet_train/cli/train.py`: init weights, seed, small-face policy, logging
- `yunet_train/tasks/face/transforms.py`: conservative small-face policy
- `yunet_train/tasks/face/__init__.py`: transform export

Production `inference/rknn/yunet.py`, model paths, thresholds, and models were
not modified by YuNet training work.

## Known issues and next decision

1. Existing VAC `inference/utils/img_utils.py::nms()` creates an xywh copy but
   passes integer xyxy to OpenCV `NMSBoxes`. A temporary research calculation
   with standard xywh suggested a large recall gain, but this has not yet been
   promoted or fully validated. Do not modify production before a controlled
   read-only A/B and regression tests.
2. Training uses BGR (`to_rgb=False`) while the current VAC ONNX wrapper swaps
   BGR input to RGB. Fixed checkpoint screening preserves VAC RGB semantics;
   this train/deploy mismatch must be resolved before another serious run.
3. Validation loss is not WIDER AP/recall and cannot select the detector.
4. Run1's policy/LR did not materially improve recall and later overfit.
5. RKNN conversion/deployment has not been attempted for this failed candidate.

One controlled follow-up is prepared but not launched:

- Script: `tools/run_yunet_v2_lr1e4_gpu.sh`
- Output: `model/versions/yunet_v2_small_face/training/run2_lr1e4/`
- Same data, augmentation, architecture, initialization, seed, batch 16,
  workers 0; only LR changes from 1e-3 to 1e-4, capped at 15 epochs.

Next agent should first review the completed upstream/best-loss/epoch-15
comparison with the user. Do not rerun it. Then choose exactly one next action:

- Preferred high-ROI step: controlled read-only legacy-xyxy versus correct-xywh
  NMS A/B using the original model, without touching production; or
- Launch the prepared LR=1e-4 run2 only after explicit user approval and after
  deciding how to keep training/runtime color semantics consistent.

Do not launch another experiment automatically.
