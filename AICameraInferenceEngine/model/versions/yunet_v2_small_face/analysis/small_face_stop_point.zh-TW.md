# YuNet 小臉診斷與訓練前 Stop Point

本階段未修改 frozen benchmark、正式 YuNet runtime 或 production model，也未
啟動訓練。Frozen results 僅用於診斷；inference strategy 使用 WIDER FACE
validation research subset 評估。

## 1. Failure analysis

| Dataset | 40px success | miss / poor IoU | 27px success | miss / poor IoU |
|---|---:|---:|---:|---:|
| SCFace | 130/130（100%） | 0 / 0 | 107/130（82.3%） | 23 / 0 |
| UTKFace | 614/936（65.6%） | 322 / 0 | 15/936（1.6%） | 921 / 0 |
| AFLW2000 | 1284/2000（64.2%） | 700 / 16 | 547/2000（27.4%） | 1444 / 9 |

正式 class 在 score 0.75 前即移除候選，因此另外以相同 ONNX、score 0.01
重跑代表樣本：

- UTKFace 40px misses（N=30）：GT-matched median score 0.543、median IoU
  0.806，30/30 的 IoU >= 0.5。
- UTKFace 27px misses（N=30）：median score 0.044、median IoU 0.609，21/30
  仍有 IoU >= 0.5 的低分候選。
- AFLW2000 40/27px misses（各 N=30）：median score 0.349/0.279；23/30、
  20/30 候選 IoU >= 0.5。
- SCFace 27px misses（N=23）：median score 0.522、median IoU 0.769；21/23
  候選 IoU >= 0.5。

主要 failure mode 是 classification/objectness 信心不足，而非 bbox localization。
完整 JSON 與 contact sheets 位於 `analysis/frozen_failures/`；綠框為 GT、紅框為
原始或低門檻候選。

## 2. Benchmark / domain 差異

三資料集共用相同 generator：640x480 黑 canvas、依 GT face height 縮放、
`INTER_AREA` downscale、中心放置、JPEG quality 95、相同 bbox transformation。
但來源語意不同：UTKFace 把整張 tight chip 當 GT；SCFace/AFLW2000 的 GT
由 landmarks 建立，因此縮放時保留較多頭部、姿態與背景 context。

| 27px dataset | median non-black 寬/高 | canvas non-black fraction |
|---|---:|---:|
| UTKFace | 29 / 30px | 0.249% |
| SCFace | 36 / 48.5px | 0.563% |
| AFLW2000 | 46 / 47px | 0.632% |

判讀為 **C：domain artifact 與真正的小臉 confidence/capacity 問題並存**。
UTKFace 27px 不應單獨主導訓練，但 AFLW2000/WIDER 自然場景也證明問題真實。

## 3. 低成本 inference checks

固定研究集為 WIDER val 依檔名順序首 200 張符合條件影像：每張 1-10 個 GT
faces，至少一張臉投影至 640 input 後高度 20-50px，共 728 targets。

| 方法 | IoU>=0.5 recall | Precision proxy | CPU time/image |
|---|---:|---:|---:|
| score 0.75 full frame | 37.64% | 97.48% | 12.18 ms |
| score 0.50 full frame | 44.09% | 90.99% | 12.21 ms |
| score 0.75 full frame + 4 fixed tiles | 60.71% | 96.98% | 48.66 ms |

只測一個 threshold（0.50）與一個 tiling 設計（四張 60% width/height tiles 加
full-frame/global NMS），沒有 grid。Lower threshold 僅 +6.45 pp 且 precision
下降；tiling +23.08 pp 但約 4 倍成本，兩者都未達直接取代訓練的條件。

## 4. Fine-tune 決策

Fine-tune **有正當性但目前不可啟動**。WIDER natural validation baseline 仍低，
tiling 的提升證明尺度敏感，單純降 threshold 又不能兼顧 precision。建議只做一個
YuNet_n small-face candidate，80px 為 guardrail、40px 優先、27px 次要。

## 5-7. Framework / GPU

- 官方 framework：`ShiqiYu/libfacedetection.train`，直接 PyTorch pipeline，入口
  `python -m yunet_train.cli.train`，支援 `yunet_n`、WIDER 與 12-output ONNX。
- 官方 repository 已 checkout 並鎖定 commit
  `dca340aa082c71081a68d17db8e58b33a58a914b`。
- 已建立隔離 conda env `vac-yunet-gpu`：Python 3.11.16、PyTorch
  2.11.0+cu128、TorchVision 0.26.0+cu128；未修改 `pytorch-test`、
  `rknn-toolkit2`、`vac-age-v3` 或 base。
- 使用者已在一般 WSL terminal 驗證 `/dev/dxg`、`nvidia-smi` 與 CUDA 正常，GPU
  為 NVIDIA GeForce RTX 5070 Ti Laptop GPU。
- Codex execution sandbox 未 expose `/dev/dxg`，所以其中的
  `torch.cuda.is_available()==False` 只代表 sandbox 限制，不是 host WSL、driver
  或 NVIDIA 故障。
- Host GPU smoke 的固定命令與環境詳情記錄於
  `training/PREPARATION.zh-TW.md`。

## 8-10. Data、augmentation、檔案

- Train：WIDER train 12,876 images + SCRFD labelv2。
- Validation：固定 WIDER val 3,226 images；SCFace/UTKFace/AFLW2000 保留為
  candidate 選定後唯一一次 external frozen test。
- 使用相符 pretrained `yunet_n` 初始化；保留 normal/large faces，適度增加自然
  context 的 20-50px faces。
- 單一 conservative policy：`INTER_AREA` degradation、mild blur/JPEG、
  brightness/contrast/noise；不可只訓練 black-canvas UTKFace。
- 已在 pinned vendor checkout 實作 opt-in `--small-face-policy`，並新增
  `tools/preflight_yunet_gpu.py` 與 `tools/run_yunet_v2_gpu_smoke.sh`；正式
  `inference/rknn/yunet.py` 仍未修改。

## 11-15. Command、資源與輸出

Host 端 GPU smoke 的唯一入口為：

```bash
bash tools/run_yunet_v2_gpu_smoke.sh
```

此腳本依序執行真實 WIDER batch 16 / workers 0 的 forward、backward、non-zero
gradient 與完整 1 epoch train + validation。VRAM 暫以 <6 GiB 為目標。建議的唯一
40-epoch full-training command 已記錄於 `training/PREPARATION.zh-TW.md`，但在 smoke
結果回報與使用者批准前不得啟動。

所有產物放在 `model/versions/yunet_v2_small_face/`。目前只剩 host terminal 的
GPU preflight 與 1-epoch smoke 尚未執行；完成後須先回報 VRAM/RAM、epoch time、
training/validation loss 與任何 CUDA/WSL 異常，再決定是否批准 full training。
