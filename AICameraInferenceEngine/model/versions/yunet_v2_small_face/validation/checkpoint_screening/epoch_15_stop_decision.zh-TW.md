# YuNet v2 Epoch 15 Early-stop Decision

使用固定 WIDER small-face research subset：200 images、728 張 20-50px GT
faces、score threshold 0.75、IoU >= 0.5、現行 VAC RGB preprocessing 與歷史
integer-xyxy NMS 行為。沒有調整 threshold。

## Read-only checkpoint 結果

| 模型 | 20-50px recall | Precision proxy | 20-30px | 30-40px | 40-50px |
|---|---:|---:|---:|---:|---:|
| Production ONNX reference | 37.64% | 97.48% | - | - | - |
| Official pretrained control | 37.64% | 98.16% | 22.13% | 42.81% | 48.45% |
| `best_loss.pth`（epoch 2） | 38.46% | 97.52% | 24.68% | 42.81% | 48.45% |
| `epoch_15.pth` | 32.14% | 99.19% | 18.72% | 35.45% | 43.30% |

`best_loss.pth` 只增加 0.82 pp overall recall，且改善完全來自 20-30px；這不算
material gain。Epoch 15 相對 pretrained overall 下降 5.49 pp，20-30、30-40、
40-50px 全部下降。Precision 上升不能抵銷 recall 的明顯退步，因本階段主要目標
是改善 small-face recall 且 baseline precision 已很高。

## Loss 與 LR

- Epoch 2：train 2.666233、eval 3.096332，是最低 eval loss。
- Epoch 15：train 2.663561、eval 3.300912。
- Training loss 幾乎沒有改善，eval loss 惡化，與 detector recall regression 一致。
- Scheduler milestones 明確設定為 epoch 24、34，gamma 0.1。因此 epoch 1-23 的
  base LR 按設計維持 0.001；前 100 optimizer iterations 的 linear warmup 已在
  epoch 1 內完成，而 epoch log 記錄的是整輪結束 LR，所以從 epoch 1 顯示 0.001。

## 決策

符合預先指定的停止條件：epoch 15 未 materially improve original/best-loss
checkpoint。正式 run 應以 `Ctrl+C` 正常停止，不繼續等待 epoch 40。保留所有既有
checkpoints 與 logs，不啟動另一個 training run。

本結果指出目前的單一 augmentation/fine-tuning policy 沒有改善目標，而不是 YuNet
架構已被證明無法改善。後續若另行批准新實驗，應先處理 training BGR 與 VAC runtime
RGB semantics，並重新檢查 augmentation/crop policy；不得直接以本 run 繼續調參。

## Pretrained initialization 稽核

`--init-weights` 呼叫 `load_model_weights_only(..., strict=True)`。另以逐 tensor
稽核確認：

- Checkpoint/model state keys：208 / 208
- Missing keys：`[]`
- Unexpected keys：`[]`
- Loaded trainable parameters：75,856
- Total trainable parameters：75,856
- Loaded state values（含 buffers）：77,890 / 77,890
- 所有載入後 tensor 與 checkpoint：逐值相等

因此 run1 正確從 upstream `weights/yunet_n.pth` 初始化，沒有漏載或錯配參數。

## 尚未啟動的 controlled run2

因 run1 沒有 meaningful gain，已準備唯一的 `LR=1e-4` 實驗入口
`tools/run_yunet_v2_lr1e4_gpu.sh`。它保持 dataset、augmentation、architecture、
pretrained initialization、seed、batch 16、workers 0 與其他設定不變，最多 15
epochs，產物預定放在 `training/run2_lr1e4/`。目前未獲准啟動，該目錄尚未建立。
