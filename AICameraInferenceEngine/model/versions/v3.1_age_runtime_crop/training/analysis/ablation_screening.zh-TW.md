# Age v3.1 Controlled Ablation

所有 screening 實驗使用相同 SSR-Net、初始權重、MAE loss、Adam 1e-4、seed
20260905、8 epochs，以及相同的 runtime-faithful 80px + 40px validation。

| 實驗 | 唯一主要變更 | Validation +/-5 | MAE | Internal test +/-5 | MAE |
|---|---|---:|---:|---:|---:|
| B0 | v2 training control | 25.14% | 12.327 | 25.21% | 12.679 |
| B1 | 160px runtime crop | 27.75% | 11.662 | 26.08% | 12.048 |
| B2 | B1 + mild jitter | 25.05% | 12.276 | 24.89% | 12.585 |
| B3 | B1 + 80/40px degradation | 28.78% | 10.589 | 26.68% | 10.924 |
| B4 | B1 + natural age sampling | 28.36% | 10.034 | 25.34% | 10.354 |

B2 未通過，未納入 B5。B3 的 +/-5 最佳，B4 的 MAE 最佳，因此 B5 只組合
runtime-faithful 160/80/40px crops、40/35/25 距離權重，以及 natural age
sampling；沒有加入 Huber、blur、JPEG 或 jitter。

## B5 Full Schedule

B5 訓練 20 epochs，最佳 checkpoint 在 epoch 20。共同 validation 為 52.66%
+/-5、7.225 MAE；internal test 為 54.36%、7.011 MAE。

| 年齡桶 | Validation N | +/-5 | MAE | Signed error |
|---|---:|---:|---:|---:|
| 0-9 | 646 | 84.06% | 4.319 | +3.959 |
| 10-19 | 281 | 43.06% | 6.958 | +2.178 |
| 20-29 | 382 | 51.05% | 6.254 | +0.454 |
| 30-39 | 248 | 34.68% | 7.997 | -0.291 |
| 40-49 | 151 | 42.38% | 8.192 | -2.809 |
| 50-59 | 197 | 36.55% | 9.679 | -5.283 |
| 60-69 | 128 | 25.78% | 11.759 | -8.376 |
| 70+ | 111 | 13.51% | 15.523 | -12.731 |

結果顯示 runtime crop hypothesis 成立，但高齡低估仍然明顯。最終是否部署必須
依 frozen RK3588 E2E benchmark 判斷，不能只依 internal validation。
