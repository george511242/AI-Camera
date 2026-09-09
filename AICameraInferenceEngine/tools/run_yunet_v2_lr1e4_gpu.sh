#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/g2004/projects/VAC-AI-Camera-Source/vac-camera/vac-camera/AICameraInferenceEngine"
PY="/home/g2004/miniforge3/envs/vac-yunet-gpu/bin/python"
TRAINER="$ROOT/model/versions/yunet_v2_small_face/training/vendor/libfacedetection.train"
DATA="/home/g2004/datasets/vac-distance/source/widerface"
OUT="$ROOT/model/versions/yunet_v2_small_face/training/run2_lr1e4"
WEIGHTS="$TRAINER/weights/yunet_n.pth"

cd "$ROOT"

if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite controlled experiment directory: $OUT" >&2
  exit 2
fi

for path in \
  "$PY" \
  "$WEIGHTS" \
  "$DATA/labelv2/train/labelv2.txt" \
  "$DATA/WIDER_train/WIDER_train/images" \
  "$DATA/labelv2/val/labelv2.txt" \
  "$DATA/WIDER_val/WIDER_val/images"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 3
  fi
done

if ! "$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'; then
  echo "CUDA is unavailable; experiment was not started." >&2
  exit 4
fi

mkdir -p "$OUT"
{
  echo "experiment=YuNet v2 controlled LR 1e-4"
  echo "launch_requires_explicit_approval=true"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "upstream_commit=$(git -C "$TRAINER" rev-parse HEAD)"
  echo "pretrained_sha256=$(sha256sum "$WEIGHTS" | awk '{print $1}')"
  "$PY" -c 'import platform, torch; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"gpu={torch.cuda.get_device_name(0)}")'
} | tee "$OUT/environment.log"

/usr/bin/time -v "$PY" -m yunet_train.cli.train \
  --variant yunet_n \
  --ann-file "$DATA/labelv2/train/labelv2.txt" \
  --img-prefix "$DATA/WIDER_train/WIDER_train/images" \
  --val-ann-file "$DATA/labelv2/val/labelv2.txt" \
  --val-img-prefix "$DATA/WIDER_val/WIDER_val/images" \
  --init-weights "$WEIGHTS" \
  --small-face-policy \
  --seed 20260905 \
  --image-size 640 \
  --min-face-size 10 \
  --epochs 15 \
  --batch-size 16 \
  --workers 0 \
  --device cuda \
  --lr 0.0001 \
  --lr-steps 24 34 \
  --warmup-iters 100 \
  --checkpoint-interval 5 \
  --eval-interval 1 \
  --no-tensorboard \
  --log-interval 50 \
  --work-dir "$OUT" \
  2>&1 | tee "$OUT/terminal.log"

echo "Controlled experiment artifacts: $OUT" | tee -a "$OUT/terminal.log"
