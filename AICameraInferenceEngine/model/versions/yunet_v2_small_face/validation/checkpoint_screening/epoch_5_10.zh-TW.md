# YuNet v2 Epoch 5 / 10 Read-only Screening

本評估沒有停止或修改正式 training。只讀取已完整寫入的 `epoch_5.pth` 與
`epoch_10.pth`，使用 CPU 單執行緒、低排程優先權執行。

固定 subset 與原 small-face diagnosis 相同：WIDER validation 中依檔名順序前
200 張符合條件影像，共 728 張投影至 640 input 後高度為 20-50px 的 GT faces。
固定 score threshold 0.75、IoU >= 0.5、NMS IoU 0.3，沒有調 threshold。

## 結果

| 模型 | Input color | Recall | Precision proxy | Matched / GT |
|---|---|---:|---:|---:|
| production ONNX baseline | VAC runtime RGB | 37.64% | 97.48% | 274 / 728 |
| official pretrained control | RGB | 37.64% | 98.16% | 274 / 728 |
| epoch 5 | RGB | 35.16% | 97.37% | 256 / 728 |
| epoch 10 | RGB | 37.91% | 97.67% | 276 / 728 |
| official pretrained control | BGR | 53.71% | 98.14% | 391 / 728 |
| epoch 5 | BGR | 51.92% | 97.73% | 378 / 728 |
| epoch 10 | BGR | 53.30% | 97.58% | 388 / 728 |

Official pretrained checkpoint 的 RGB recall 與 production ONNX baseline 完全
一致，證明 read-only evaluator 已對齊現行 VAC protocol。Precision proxy 有
0.68 pp 差異，可能來自 framework/ONNX 數值或候選排序細節，因此 checkpoint
選擇以同一 PyTorch evaluator 的 pretrained control 為直接對照。

截至 epoch 10，fine-tuned checkpoint 在 RGB/BGR 都從 epoch 5 回升。RGB recall
只比 control 高 0.27 pp，precision proxy低 0.49 pp；BGR 尚未超越 control。這不是
明顯改善，但也不構成立即停止條件。Training 可繼續，不應只依 validation loss
選 checkpoint；正式完成後仍須用固定 detection evaluator 比較所有保存點。

Evaluator 忠實保留現行 `img_utils.nms()` 的歷史行為：OpenCV `NMSBoxes` 收到
integer xyxy，而不是標準 xywh。早期用標準 xywh 的暫時結果已撤銷，不得引用。

## Validation kps loss 為零

完整 annotation 統計：

| Split | Images | Faces | Faces with visible keypoints | Visible points |
|---|---:|---:|---:|---:|
| Train | 12,876 | 159,393 | 76,006 | 380,030 |
| Validation | 3,226 | 39,697 | 0 | 0 |

Training `labelv2` 含 bbox、5 landmarks 與 visibility；validation `labelv2` 只有 bbox。
Parser 對缺少 landmarks 的 validation face 建立全零 keypoint weights，criterion
因此讓 kps loss 為 0。這符合現有 annotation/evaluation contract，不是 kps head
未執行或 loss pipeline 漏算。它也表示 validation kps loss 無法用於 checkpoint
selection；landmark 品質必須使用另有 landmark GT 的資料驗證。

## 產物

- `pretrained_rgb.json` / `pretrained_bgr.json`
- `epoch_5_rgb.json` / `epoch_5_bgr.json`
- `epoch_10_rgb.json` / `epoch_10_bgr.json`

所有 JSON 都記錄 checkpoint SHA256、protocol、matched/detection counts 與完整精度。
