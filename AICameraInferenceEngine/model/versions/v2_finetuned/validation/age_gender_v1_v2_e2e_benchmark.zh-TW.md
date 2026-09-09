# Age/Gender v1 vs v2 RK3588 完整 E2E Benchmark

- v1 result: `/home/g2004/datasets/vac-distance/eval_v1_rk3588.jsonl`
- v2 result: `/home/g2004/datasets/vac-distance/eval_v2_age_gender_rk3588.jsonl`
- Gender mapping: raw >= 0.5 -> male (fixed before evaluation)
- YuNet success: IoU >= 0.5 and score >= 0.75
- 完整 harness 為 9,198 rows；Age/Gender 指標使用其中 2,808 個 UTKFace rows

## 改善摘要

| 臉高 | Age E2E ±5 Δ | Age Conditional ±5 Δ | Age MAE Δ | Gender E2E BAcc Δ | Gender Conditional BAcc Δ |
|---:|---:|---:|---:|---:|---:|
| 80px | +8.33 pp | +8.53 pp | -3.830 years | +7.65 pp | +7.74 pp |
| 40px | +4.17 pp | +6.35 pp | -4.055 years | +7.01 pp | +9.70 pp |
| 27px | +0.11 pp | +6.67 pp | -0.850 years | +0.21 pp | +8.33 pp |
| 總計 | +4.20 pp | +7.65 pp | -3.891 years | +4.96 pp | +8.56 pp |

> 27px 的 UTKFace 只有 15/936 通過 YuNet，因此該層 Conditional 指標樣本極少；E2E 幾乎由 detector gate 主導。

## Age

| 臉高 | 版本 | E2E ±5 | Conditional ±5 | Conditional MAE | YuNet成功/總數 |
|---:|---|---:|---:|---:|---:|
| 80px | v1 | 19.66% (184/936) | 20.13% (184/914) | 15.443 (914) | 914/936 |
| 80px | v2 | 27.99% (262/936) | 28.67% (262/914) | 11.612 (914) | 914/936 |
| 40px | v1 | 13.57% (127/936) | 20.68% (127/614) | 16.483 (614) | 614/936 |
| 40px | v2 | 17.74% (166/936) | 27.04% (166/614) | 12.428 (614) | 614/936 |
| 27px | v1 | 0.43% (4/936) | 26.67% (4/15) | 15.540 (15) | 15/936 |
| 27px | v2 | 0.53% (5/936) | 33.33% (5/15) | 14.690 (15) | 15/936 |
| 總計 | v1 | 11.22% (315/2808) | 20.41% (315/1543) | 15.858 (1543) | 1543/2808 |
| 總計 | v2 | 15.42% (433/2808) | 28.06% (433/1543) | 11.967 (1543) | 1543/2808 |

## Gender

| 臉高 | 版本 | E2E Balanced Accuracy | Conditional Balanced Accuracy |
|---:|---|---:|---:|
| 80px | v1 | 70.81% | 72.72% |
| 80px | v2 | 78.46% | 80.46% |
| 40px | v1 | 46.96% | 73.16% |
| 40px | v2 | 53.97% | 82.86% |
| 27px | v1 | 0.95% | 62.50% |
| 27px | v2 | 1.16% | 70.83% |
| 總計 | v1 | 39.57% | 72.89% |
| 總計 | v2 | 44.53% | 81.44% |
