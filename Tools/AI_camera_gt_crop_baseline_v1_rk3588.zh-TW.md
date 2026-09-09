# AI Camera v1 RK3588 GT-crop Baseline

## 1. 目的

建立不受 YuNet 偵測率與偵測框偏移影響的 v1 model-only baseline，供後續 Age/Gender v2 fine-tune 比較。

本測試直接使用衍生資料 manifest 的 ground-truth bbox，再套用 VAC 現有模型 margin：

- Age margin：0.45
- Gender margin：0.45
- Head Pose margin：0.6

## 2. 環境

| 項目 | 值 |
|---|---|
| 日期 | 2026-09-05 |
| 裝置 | Orange Pi 5 Ultra / RK3588 |
| RKNN Toolkit Lite2 | 2.3.2 |
| RKNN Runtime | 2.3.2 |
| RKNPU driver | 0.9.6 |
| 模型 | `v1_baseline/rknn/rk3588` FP16 |
| Crop source | manifest ground-truth bbox |
| 有效樣本 | 8,808 |
| 執行時間 | 69 秒 |
| Runtime error | 0 |

SCFace 只用於 YuNet，因此 GT-crop 模式不包含其 390 筆資料。

## 3. 主要指標

| 模型 | 距離（臉高） | N | 結果 | 95% CI |
|---|---|---:|---:|---|
| Age，絕對誤差 <= 5 歲 | 1m（80px） | 936 | 22.5% | [20.0, 25.3] |
| Age，絕對誤差 <= 5 歲 | 2m（40px） | 936 | 20.8% | [18.4, 23.6] |
| Age，絕對誤差 <= 5 歲 | 3m（27px） | 936 | 17.0% | [14.7, 19.5] |
| Gender，Balanced Accuracy | 1m（80px） | 936 | 73.1% | 未計算 |
| Gender，Balanced Accuracy | 2m（40px） | 936 | 69.6% | 未計算 |
| Gender，Balanced Accuracy | 3m（27px） | 936 | 65.7% | 未計算 |
| Head Pose，三軸誤差都 <= 15 度 | 1m（80px） | 2,000 | 55.0% | [52.8, 57.1] |
| Head Pose，三軸誤差都 <= 15 度 | 2m（40px） | 2,000 | 54.0% | [51.8, 56.2] |
| Head Pose，三軸誤差都 <= 15 度 | 3m（27px） | 2,000 | 48.8% | [46.6, 51.0] |

## 4. Age/Gender 診斷

| 距離（臉高） | Age MAE | Age valid | Male recall | Female recall |
|---|---:|---:|---:|---:|
| 1m（80px） | 14.28 | 936/936 | 93.3%（430/461） | 52.8%（251/475） |
| 2m（40px） | 16.37 | 936/936 | 94.8%（437/461） | 44.4%（211/475） |
| 3m（27px） | 19.47 | 936/936 | 95.7%（441/461） | 35.8%（170/475） |

Gender 使用固定映射：UTKFace `0=male, 1=female`，模型 `raw >= 0.5 -> male`。

## 5. 與 E2E 結果對照

| 指標 | 80px E2E -> GT | 40px E2E -> GT | 27px E2E -> GT |
|---|---:|---:|---:|
| Age ±5 歲 | 19.7% -> 22.5% | 13.6% -> 20.8% | 0.4% -> 17.0% |
| Gender Balanced Accuracy | 70.8% -> 73.1% | 47.0% -> 69.6% | 1.0% -> 65.7% |
| Head Pose 三軸 ±15 度 | 60.9% -> 55.0% | 47.0% -> 54.0% | 21.6% -> 48.8% |

27px Age/Gender E2E 幾乎歸零的主要原因是 YuNet 在該 UTKFace 測試層級的通過率只有 1.6%，而不是 Age/Gender 完全失去辨識能力。

Head Pose 在 80px 使用 YuNet crop 反而高於目前的 GT bbox crop，表示 AFLW2000 由 landmark 擴張產生的 GT bbox 與模型偏好的 face crop 可能不同。此現象需要在調整 Head Pose 前另外研究，不應拿來調整 Age/Gender fine-tune。

## 6. v2 驗收基準

v2 應使用同一份 manifest、相同 crop margin、固定 Gender mapping，並至少報告：

- Age MAE
- Age ±5 歲命中率
- Gender Balanced Accuracy
- Male recall
- Female recall
- 80px、40px、27px 分層結果
- ONNX、RK3566 RKNN、RK3588 RKNN 的輸出差異

Gender 的主要改善目標是提升 female recall，同時避免 male recall 大幅下降。只提高總 accuracy 或讓模型更偏向 male 不算改善。

## 7. 結果檔

```text
/home/g2004/datasets/vac-distance/eval_v1_rk3588_gt_crop.jsonl
```

- 大小：5,436,201 bytes
- SHA-256：`df5f601f93287fb827cd741eacaeaa9d2df2df0cde8caed71e721fbe46a1eda4`
- Orange Pi：`/home/orangepi/vac-v1-distance-test/eval_v1_rk3588_gt_crop.jsonl`

## 8. 警告

RKNN Lite 對 static-shape 模型查詢 dynamic range 時發出警告。三個模型均成功初始化，8,808 筆全部完成，無 runtime error。
