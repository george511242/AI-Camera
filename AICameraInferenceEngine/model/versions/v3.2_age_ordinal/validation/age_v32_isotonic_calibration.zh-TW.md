# Age v3.2 Isotonic Calibration

本實驗只執行一次固定的 monotonic isotonic calibration，未重新訓練神經網路，
也未使用 frozen RK3588 benchmark label 擬合或選擇校正器。

## 方法與資料安全

- 資料：v3 leakage-free validation 的 80px + 40px runtime crops
- 共有 2,144 rows、1,279 個 source images
- 以 `source_id` 的 SHA256 固定分派至 5 folds
- 每 fold 使用其他 4 folds 擬合 PAVA isotonic mapping，僅評估 held-out fold
- 同一 source 的 80px/40px 永遠位於同一 fold
- 5 folds 的 fit/evaluation source overlap 均為 0
- RAW 與 CALIBRATED 指標均來自相同 2,144 rows

殘餘限制：O1/O2 neural model selection 已使用這份 validation split；cross-fitting
排除了校正器本身的 fit/evaluation overlap，但不能把此資料變成新的 neural-model
holdout。

## Overall 與距離

| 範圍 | 版本 | N | +/-5 | MAE | Signed error |
|---|---|---:|---:|---:|---:|
| Overall | RAW | 2,144 | 55.41% | 6.311 | -0.796 |
| Overall | CALIBRATED | 2,144 | 54.48% | 6.378 | +0.006 |
| 80px | RAW | 1,279 | 57.31% | 6.116 | -0.830 |
| 80px | CALIBRATED | 1,279 | 55.90% | 6.203 | -0.020 |
| 40px | RAW | 865 | 52.60% | 6.600 | -0.747 |
| 40px | CALIBRATED | 865 | 52.37% | 6.637 | +0.043 |

校正後 overall +/-5 下降 0.93 pp，MAE 惡化 0.067 年。整體 signed
error 接近零，但這不代表個別樣本誤差變小。

## 年齡桶

| 年齡 | N | RAW +/-5 | CAL +/-5 | RAW MAE | CAL MAE | RAW bias | CAL bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-9 | 646 | 84.37% | 82.66% | 2.841 | 2.982 | +1.975 | +2.257 |
| 10-19 | 281 | 47.33% | 48.04% | 6.347 | 6.431 | +2.956 | +3.233 |
| 20-29 | 382 | 55.24% | 54.19% | 5.546 | 5.728 | +1.021 | +1.268 |
| 30-39 | 248 | 44.76% | 39.92% | 6.902 | 7.641 | -0.269 | +0.518 |
| 40-49 | 151 | 41.72% | 33.77% | 8.246 | 8.886 | -2.907 | -1.321 |
| 50-59 | 197 | 34.01% | 28.43% | 9.229 | 9.618 | -4.441 | -2.453 |
| 60-69 | 128 | 32.81% | 43.75% | 10.655 | 9.563 | -9.128 | -7.066 |
| 70-79 | 61 | 26.23% | 42.62% | 12.003 | 9.568 | -12.003 | -9.288 |
| 80+ | 50 | 0.00% | 8.00% | 18.459 | 16.284 | -18.459 | -16.276 |

## Hard Stop 決策

預先固定的通過條件為：overall +/-5 至少 +2 pp 或 MAE 至少改善 0.5 年，
同時 70-79 與 80+ 均須實質改善，且 0-9、10-19、20-29 任一桶不得下降
超過 2 pp。

- Overall benefit：FAIL
- High-age guardrail：PASS
- Young-age guardrail：PASS
- 最終決策：**FAIL / STOP**

因此不執行 calibrated frozen RK3588 benchmark、不啟用校正參數，也不再嘗試
其他 calibration grid。`calibration/isotonic.json` 僅保留實驗稽核用途，
其中 `enabled=false`、`frozen_for_benchmark=false`。

Age v3.2 raw 不升級為 production default。SSR-Net 路線至此停止微調；下一個可能
帶來較大增益的方向是另外評估更強且 RKNN-compatible 的 age-estimation model
family。
