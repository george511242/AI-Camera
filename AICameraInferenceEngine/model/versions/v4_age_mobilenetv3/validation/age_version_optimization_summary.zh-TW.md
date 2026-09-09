# Age v1-v4 優化與外部結果比較

本文件整理 Age 模型各版本的技術變更與實際效果。內部版本比較全部使用同一份
9,198-row frozen RK3588 manifest；Age 主評估範圍為 80px + 40px。27px 僅
15/936 張通過 YuNet，因此不作為 Age 模型選擇依據。

## 各版本做了什麼

| 版本 | 主要變更 | 結論 |
|---|---|---|
| v1 | 原始 SSR-Net 64x64 scalar regression | 原始 baseline |
| v2 | UTKFace fine-tune、decade-balanced sampling、小臉訓練流程 | 明顯優於 v1，目前仍為 production default |
| v3 | 建立 SHA256 leakage-safe split；測試 sampling、Huber、distance/camera augmentation、bbox jitter；最後只採 sqrt inverse-frequency sampling | internal validation 改善，但 frozen E2E 低於 v2，不部署 |
| v3.1 | 保留 SSR-Net；訓練/驗證改成與 VAC 完全一致的 YuNet bbox、margin 0.45 crop、64x64 resize；natural sampling；40/35/25 的 160/80/40px 距離分布 | 修正 train/runtime domain gap，帶來最大單次 E2E 提升 |
| v3.2 | 保留 v3.1 pipeline；scalar regression 改成 98 thresholds 的 rank-consistent ordinal BCE，輸出仍為 scalar age | MAE 改善，但 +/-5 幾乎與 v3.1 持平；不升級 production |
| v4 | ImageNet-pretrained MobileNetV3-Large、112x112 RGB、相同 ordinal head 與 runtime crop semantics | 40px 與 MAE 顯著改善；作為 opt-in candidate，v2 仍是預設 |

## Frozen RK3588 主結果

| 版本 | 80+40 E2E +/-5 | 80+40 Conditional +/-5 | Conditional MAE | Conditional vs v1 | MAE vs v1 | 與上一版 |
|---|---:|---:|---:|---:|---:|---|
| v1 | 16.61% | 20.35% | 15.861 | baseline | baseline | baseline |
| v2 | 22.86% | 28.01% | 11.940 | +7.66 pp | -3.921 年 | +7.66 pp、MAE -3.921 年 |
| v3 | 21.31% | 26.11% | 12.854 | +5.76 pp | -3.007 年 | -1.90 pp、MAE +0.914 年 |
| v3.1 | 30.82% | 37.76% | 10.172 | +17.41 pp | -5.689 年 | +11.65 pp、MAE -2.682 年 |
| v3.2 | 30.77% | 37.70% | 9.790 | +17.35 pp | -6.071 年 | -0.06 pp、MAE -0.382 年 |
| v4 | 33.39% | 40.90% | 8.842 | +20.55 pp | -7.019 年 | +3.21 pp、MAE -0.947 年 |

從 baseline v1 到 v4，Conditional +/-5 約為原來的 2.01 倍，MAE 降低
44.3%；若把 detector failure 也算入，E2E +/-5 從 16.61% 提升至 33.39%，
增加 16.78 pp。

最大的技術收益不是單純加大模型，而是 v3.1 的 runtime-faithful crop：它相對
v3 一次增加 11.65 pp。v4 的主要額外價值集中在 40px，Conditional +/-5
由 v3.2 的 35.67% 提升至 41.69%（+6.03 pp），MAE 降低 1.837 年。

## 年齡桶的取捨

- v4 相對 v3.2 改善 0-9、10-19、20-29、50-59、70-79 與 80+ 的 +/-5。
- 30-39、40-49、60-69 的 +/-5 下降；其中 60-69 從 41.61% 降至
  34.90%，是部署時最重要的 regression。
- v4 的 80+ 雖由 v3.2 的 3.17% 回升至 10.32%，仍低於 v2 的 24.60%；
  高齡低估尚未解決。
- 因此 v4 是整體較強、尤其適合遠距小臉的 candidate，但不是每個年齡桶都勝過
  v2，也不應僅憑 overall 自動取代 production。

## 與同事舊結果比較

同事簡報提供兩組資料：

- HFOV 63°：115/57/38px 的 Age +/-5 為 11.2% / 10.9% / 10.4%。
- HFOV 120°優化後：80/40/27px 的 Age +/-5 為 10.6% / 11.2% / 7.6%。

HFOV 120°的像素設定可與目前 80/40/27px 對齊，但兩份測試的 detector、crop、
資料生成與計分實作未被證明完全相同，所以以下只能視為參考差距：

| 距離 | 同事 HFOV 120°優化後 | v4 E2E +/-5 | 表面差距 | v4 Conditional +/-5 | YuNet 成功率 |
|---|---:|---:|---:|---:|---:|
| 80px | 10.6% | 39.42% | +28.82 pp | 40.37% | 97.6%（914/936） |
| 40px | 11.2% | 27.35% | +16.15 pp | 41.69% | 65.6%（614/936） |
| 27px | 7.6% | 0.53% | -7.07 pp | 33.33% | 1.6%（15/936） |

80px 與 40px 顯示目前 v4 明顯高於同事簡報數字；40px 的 E2E 被 YuNet 成功率
限制，但仍高 16.15 pp。27px 則相反：通過 YuNet 後 Age 並不差，但 98.4%
樣本根本沒有進入 Age model，因此 E2E 落後 7.07 pp，問題主要是 detector gate，
不能解讀成 v4 Age backbone 較差。

HFOV 63°的臉高是 115/57/38px，和目前 frozen benchmark 不同，不能逐列直接
相減。僅就數值範圍而言，同事為 10.4-11.2%，目前 v4 在較接近的 80/40px
E2E 為 39.42%/27.35%，但這不是嚴格 apples-to-apples 結論。

## 部署判讀

- 穩健 production 選擇：v2，已有最保守的版本定位，且高齡部分桶仍有優勢。
- 整體準確度與遠距候選：v4；80+40 Conditional +/-5 比 v2 高 12.89 pp，
  MAE 少 3.098 年，Age-only RK3588 平均 latency 約 4.668 ms。
- v4 已提供 `AGE_MODEL_VERSION=v4` opt-in；尚未自動 promotion。
- RK3588 已完成實機測試。RK3566 已完成 ONNX -> RKNN 轉換，但尚無實體板驗證。
- 若要正式宣稱超越同事版本，應在同一 camera HFOV、同一原始影像、同一 YuNet
  與同一 E2E harness 下重跑兩個模型；目前比較適合作為方向性證據。
