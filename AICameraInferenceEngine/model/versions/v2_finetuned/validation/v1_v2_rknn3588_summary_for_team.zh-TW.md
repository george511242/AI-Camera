# VAC Age/Gender v1 → v2 效果摘要

## 測試結論

v2 已在 Orange Pi RK3588 NPU 完成相同條件的完整 E2E benchmark。相較 v1，
Age 與 Gender 都有明顯改善，尤其是 YuNet 成功偵測到臉後的模型準確度。

本次比較固定使用：

- 相同 9,198-row frozen manifest
- 相同 YuNet、Head Pose、bbox、crop、threshold 與 Gender mapping
- Age/Gender 使用其中 2,808 筆 UTKFace 測試資料
- RKNN FP16、RKNN Runtime 2.3.2、RK3588 NPU


## 整體提升

| 指標 | v1 | v2 | 改善 |
|---|---:|---:|---:|
| Age E2E ±5 | 11.22% | 15.42% | **+4.20 pp** |
| Age Conditional ±5 | 20.41% | 28.06% | **+7.65 pp** |
| Age Conditional MAE | 15.858 年 | 11.967 年 | **減少 3.891 年** |
| Gender E2E Balanced Accuracy | 39.57% | 44.53% | **+4.96 pp** |
| Gender Conditional Balanced Accuracy | 72.89% | 81.44% | **+8.56 pp** |

`Conditional` 只計算 YuNet 成功偵測的樣本；`E2E` 會把 detector miss 一併算入，
所以更接近整體系統實際表現。

## 主要距離結果

| 臉高 | Age Conditional ±5 | Age MAE | Gender Conditional BAcc |
|---|---:|---:|---:|
| 80px | 20.13% → 28.67%（**+8.53 pp**） | 15.443 → 11.612（**-3.831 年**） | 72.72% → 80.46%（**+7.74 pp**） |
| 40px | 20.68% → 27.04%（**+6.36 pp**） | 16.483 → 12.428（**-4.055 年**） | 73.16% → 82.86%（**+9.70 pp**） |
| 27px | 26.67% → 33.33%（+6.66 pp） | 15.540 → 14.690（-0.850 年） | 62.50% → 70.83%（+8.33 pp） |

27px 的 UTKFace 只有 15/936 筆通過 YuNet，因此該列樣本太少，不能作為
Age/Gender 模型的主要判斷依據。這個距離的 E2E 瓶頸主要是 YuNet detector。

## 模型版本差異

- v1：原始 Wiki pretrained SSR-Net Age/Gender baseline。
- v2：使用 leakage-safe UTKFace split 進行 fine-tune。
- 輸入輸出介面保持不變：`[1,3,64,64] → [1,1]`。
- v2 不需要修改 YuNet、Head Pose 或既有 Age/Gender runtime postprocessing。
- RK3566 與 RK3588 使用同一份 ONNX，但分別產生不同 RKNN artifact。

## RK3566 待驗證

RK3566 FP16 模型已使用 RKNN-Toolkit2 2.3.2 成功轉換，且 RK3566 simulator
輸出與 ONNX/RK3588 simulator 接近；目前缺少 RK3566 實體板驗證。

在 RK3566 上確認：

1. Age/Gender v2 RKNN 可正常 `load_rknn` 與 `init_runtime`。
4. 執行相同 benchmark harness，確認 accuracy 與 RK3588 趨勢一致。

RK3566 v2 模型：

- `model/versions/v2_finetuned/rknn/rk3566/age_ssrnet_v2_64x64_fp16.rknn`
- `model/versions/v2_finetuned/rknn/rk3566/gender_ssrnet_v2_64x64_fp16.rknn`

RK3588 v2 模型：

- `model/versions/v2_finetuned/rknn/rk3588/age_ssrnet_v2_64x64_fp16.rknn`
- `model/versions/v2_finetuned/rknn/rk3588/gender_ssrnet_v2_64x64_fp16.rknn`



在相同 RK3588 E2E 測試下，v2 將 Age ±5 條件準確率由 20.41% 提升至
28.06%，MAE 減少 3.891 年；Gender 條件 Balanced Accuracy 由 72.89%
提升至 81.44%。RK3588 已完成實機驗證，RK3566 已完成轉換與 simulator 驗證，
仍需實體板測試。
