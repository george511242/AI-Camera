# YuNet v2 GPU training preparation

## 已完成

- Conda env：`vac-yunet-gpu`，Python 3.11.16。
- PyTorch：2.11.0+cu128；TorchVision 0.26.0+cu128。
- 最小 training dependencies：NumPy 2.4.4、OpenCV 4.13.0.92、tqdm 4.67.3、
  PyYAML 6.0.3、SciPy 1.17.1。
- Upstream commit：`dca340aa082c71081a68d17db8e58b33a58a914b`。
- 官方 `yunet_n.pth` SHA256：
  `9c7c9e14a4d60e57491d77e4d8b0d451bd834f300e05c0dea7fd1f6533817b06`。
- Checkpoint strict load、12-output shape 與 75,856 parameters：PASS。
- WIDER train/val loading：12,876 / 3,226 images，PASS。
- CPU-side batch-2 forward/backward：loss 1.333844、max backbone/head gradient
  norm 5.9010/6.5334，PASS。
- `--init-weights`、`--seed`、`--small-face-policy` 與 CUDA peak-memory log
  已加入 pinned vendor checkout。

沒有安裝 TensorBoard、ONNX、ONNX Runtime，因為本階段 smoke 使用
`--no-tensorboard` 且不進行 export。Editable package 的 `pip check` 會列出這三個
未滿足項目，這是刻意的最小依賴狀態；export 前再安裝官方鎖定版本。

## 唯一 small-face policy

- Crop choices：`0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 2.5`。
- Resolution：60% clean、25% 0.75x、15% 0.5x，先 `INTER_AREA` downscale
  再 `INTER_LINEAR` upscale。
- Independent mild effects：Gaussian blur 15%；JPEG quality 75-95 為 20%；
  brightness/contrast 25%；Gaussian noise sigma 1-3 為 15%。
- 未使用 UTKFace black-canvas 作 training data，validation transform 不變。

## Host WSL GPU smoke：PASS

2026-09-05 已在一般 VS Code WSL terminal 完成：

- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU，compute capability 12.0。
- PyTorch/CUDA：2.11.0+cu128 / CUDA 12.8。
- One-batch：`[16,3,640,640]`，loss 2.471263；backbone/head 最大 gradient
  norm 2.1963/1.8436，均非零。
- One-batch peak VRAM：allocated 2,242.5 MiB、reserved 2,502 MiB。
- 完整 epoch：805 steps、305.605 秒、42.15 samples/s、training loss
  2.698211。
- Validation：3,226 images / 202 steps、約 51.9 秒、loss 3.200083。
- 整段 wall time：6:08.27；CUDA peak allocated/reserved：2,475.7/2,558 MiB。
- Process maximum RSS：2,361,160 KiB（約 2.25 GiB）；command 本身 swaps 0。
- Exit status 0，沒有 CUDA error、OOM 或 WSL instability。
- `latest.pth`、`epoch_1.pth`、`eval_epoch_1.pth`、`best_loss.pth` 均已落盤且
  可由 PyTorch 載入。

完整輸出位於：

`model/versions/yunet_v2_small_face/training/smoke_gpu/`

Validation 目前回報的是 loss，不是 WIDER AP/recall；正式 candidate selection 仍須
使用固定 WIDER detection evaluator。Smoke 證明 GPU training path 與資料流程穩定。

## 預備 full-training command

使用者已於 2026-09-05 批准啟動單一正式 candidate。請在一般 VS Code WSL
terminal、repository root 執行：

```bash
bash tools/run_yunet_v2_full_gpu.sh
```

正式產物固定存放於：

`model/versions/yunet_v2_small_face/training/run1/`

腳本會拒絕覆寫已存在的 `run1/`，並記錄 environment、upstream commit、pretrained
SHA256、完整 terminal log、每 epoch metrics 與 checkpoints。其等效 training command
如下：

```bash
ROOT=/home/g2004/projects/VAC-AI-Camera-Source/vac-camera/vac-camera/AICameraInferenceEngine
PY=/home/g2004/miniforge3/envs/vac-yunet-gpu/bin/python
TRAINER=$ROOT/model/versions/yunet_v2_small_face/training/vendor/libfacedetection.train
DATA=/home/g2004/datasets/vac-distance/source/widerface

$PY -m yunet_train.cli.train \
  --variant yunet_n \
  --ann-file "$DATA/labelv2/train/labelv2.txt" \
  --img-prefix "$DATA/WIDER_train/WIDER_train/images" \
  --val-ann-file "$DATA/labelv2/val/labelv2.txt" \
  --val-img-prefix "$DATA/WIDER_val/WIDER_val/images" \
  --init-weights "$TRAINER/weights/yunet_n.pth" \
  --small-face-policy --seed 20260905 \
  --image-size 640 --min-face-size 10 \
  --epochs 40 --batch-size 16 --workers 0 --device cuda \
  --lr 0.001 --lr-steps 24 34 --warmup-iters 100 \
  --checkpoint-interval 5 --eval-interval 1 --no-tensorboard \
  --log-interval 50 \
  --work-dir "$ROOT/model/versions/yunet_v2_small_face/training/run1"
```

建議維持 batch 16 / workers 0。雖然約 2.56 GiB peak VRAM 尚有餘裕，但目前速度
已足夠且穩定性優先。依 smoke 的 305.6 秒 train + 51.9 秒 validation 推估，40
epochs 約 3 小時 58 分，再加 checkpoint/啟停開銷，應預留約 4-4.5 小時。
