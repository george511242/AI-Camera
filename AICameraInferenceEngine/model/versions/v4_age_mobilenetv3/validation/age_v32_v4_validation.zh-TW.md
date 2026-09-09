# Age v3.2 / v4 Validation Comparison

本比較只使用 leakage-safe validation 的相同 2,144 個 80px/40px runtime
crops，不使用 frozen RK3588 benchmark label。

## Overall 與距離

| 範圍 | 版本 | N | +/-5 | MAE | Signed error |
|---|---|---:|---:|---:|---:|
| 80+40px | v3.2 | 2,144 | 55.41% | 6.311 | -0.796 |
| 80+40px | v4 | 2,144 | 59.61% | 6.048 | -1.869 |
| 80px | v3.2 | 1,279 | 57.31% | 6.116 | -0.830 |
| 80px | v4 | 1,279 | 60.59% | 5.933 | -1.775 |
| 40px | v3.2 | 865 | 52.60% | 6.600 | -0.747 |
| 40px | v4 | 865 | 58.15% | 6.219 | -2.009 |

V4 overall 相對 v3.2 為 `+4.20 pp`、MAE 改善 `0.263` 年。主要 +/-5
success gate 通過；偏好的 0.7 年 MAE 改善未達成。

## 年齡桶

| 年齡 | N | v3.2 +/-5 | v4 +/-5 | v3.2 MAE | v4 MAE | v4 mean pred | v4 bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0-9 | 646 | 84.37% | 90.09% | 2.841 | 1.984 | 4.759 | +1.146 |
| 10-19 | 281 | 47.33% | 61.21% | 6.347 | 4.922 | 16.487 | +1.555 |
| 20-29 | 382 | 55.24% | 60.99% | 5.546 | 4.796 | 24.547 | -0.029 |
| 30-39 | 248 | 44.76% | 43.15% | 6.902 | 7.325 | 33.581 | -0.653 |
| 40-49 | 151 | 41.72% | 37.75% | 8.246 | 9.079 | 38.714 | -5.895 |
| 50-59 | 197 | 34.01% | 34.01% | 9.229 | 11.072 | 47.159 | -7.100 |
| 60-69 | 128 | 32.81% | 25.00% | 10.655 | 13.132 | 53.051 | -11.011 |
| 70-79 | 61 | 26.23% | 31.15% | 12.003 | 10.810 | 64.987 | -9.095 |
| 80+ | 50 | 0.00% | 18.00% | 18.459 | 15.223 | 68.590 | -15.190 |

## 判讀

- Primary validation gate：PASS（+4.20 pp）
- MAE preferred target：未達成（僅改善 0.263 年）
- 0-29 guardrail：PASS，三個桶均提升
- Older-age result：70-79 與 80+ 改善，但 60-69 明顯退步
- Monotonic violations：0

此結果足以依預定流程進入 trained ONNX/RKNN 與 RK3588 sanity，但不是自動
promotion。中年及 60-69 regression 必須保留在最終部署決策中。

## Training

- ImageNet-pretrained MobileNetV3-Large, 112x112
- O1 rank-consistent 98-threshold ordinal BCE
- Natural age sampling；160/80/40px = 40/35/25
- Seed 20260905；batch 64；workers 4；CPU
- 1 epoch head warm-up；14/20 unfrozen epochs，early stopping
- Fine-tuning elapsed：1,267.15 seconds
- Selected epoch：9（依 validation +/-5）
- Best SHA256：`9f3e9850cb4f07d20cbb0ee90ddc87574c4ee00c74a9eb4b3a8c0d5af2033272`

GPU batch forward/backward 曾通過，但完整 training-flow 隨後 exit 139。
TensorFlow 2.15 不含 RTX 5070 Ti compute capability 12.0 的編譯 kernel，因此
正式訓練固定使用 CPU，不應在本實驗重試 GPU。
