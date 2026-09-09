#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/g2004/projects/VAC-AI-Camera-Source/vac-camera/vac-camera/AICameraInferenceEngine"
PY="/home/g2004/miniforge3/envs/vac-yunet-gpu/bin/python"
TRAINER="$ROOT/model/versions/yunet_v2_small_face/training/vendor/libfacedetection.train"
DATA="/home/g2004/datasets/vac-distance/source/widerface"
OUT="$ROOT/model/versions/yunet_v2_small_face/training/smoke_gpu"

cd "$ROOT"
mkdir -p "$OUT"

echo "=== host resources before smoke ==="
date --iso-8601=seconds
free -h
nvidia-smi
"$PY" -c 'import torch; print("torch:", torch.__version__); print("cuda:", torch.version.cuda); print("available:", torch.cuda.is_available()); print("gpu:", torch.cuda.get_device_name(0)); print("capability:", torch.cuda.get_device_capability(0))'

echo "=== one-batch forward/backward ==="
"$PY" tools/preflight_yunet_gpu.py \
  --trainer-root "$TRAINER" \
  --ann-file "$DATA/labelv2/train/labelv2.txt" \
  --img-prefix "$DATA/WIDER_train/WIDER_train/images" \
  --weights "$TRAINER/weights/yunet_n.pth" \
  --batch-size 16 \
  --workers 0 \
  --seed 20260905 \
  --output "$OUT/one_batch.json"

echo "=== one full training epoch plus validation loss ==="
/usr/bin/time -v "$PY" -m yunet_train.cli.train \
  --variant yunet_n \
  --ann-file "$DATA/labelv2/train/labelv2.txt" \
  --img-prefix "$DATA/WIDER_train/WIDER_train/images" \
  --val-ann-file "$DATA/labelv2/val/labelv2.txt" \
  --val-img-prefix "$DATA/WIDER_val/WIDER_val/images" \
  --init-weights "$TRAINER/weights/yunet_n.pth" \
  --small-face-policy \
  --seed 20260905 \
  --image-size 640 \
  --min-face-size 10 \
  --epochs 1 \
  --batch-size 16 \
  --workers 0 \
  --device cuda \
  --lr 0.001 \
  --lr-steps 24 34 \
  --warmup-iters 100 \
  --checkpoint-interval 1 \
  --eval-interval 1 \
  --no-tensorboard \
  --log-interval 50 \
  --work-dir "$OUT" \
  2>&1 | tee "$OUT/terminal.log"

echo "=== host resources after smoke ==="
free -h
nvidia-smi
echo "Smoke artifacts: $OUT"

