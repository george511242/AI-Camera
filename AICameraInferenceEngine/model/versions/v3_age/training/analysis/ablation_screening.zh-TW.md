# Age v3 A0-A5 Screening

固定條件：clean v3 split、seed `20260905`、Wiki Age pretrained SSR-Net、
batch 64、最多 8 epochs、CPU、以 validation +/-5 選 checkpoint。

| 實驗 | 唯一變更 | Best epoch | Validation +/-5 | Validation MAE | Test +/-5 | Test MAE |
|---|---|---:|---:|---:|---:|---:|
| A0 | v2-equivalent baseline | 8 | 26.08% | 12.280 | 27.61% | 11.933 |
| A1 | sqrt inverse-frequency sampling | 7 | **26.85%** | **10.820** | **28.07%** | **10.581** |
| A2 | SmoothL1, delta 5 | 8 | 25.62% | 12.156 | 27.61% | 11.809 |
| A3 | mixed clean/80px/40px degradation | 1 | 22.15% | 14.750 | 22.16% | 14.810 |
| A4 | mild camera-quality augmentation | 6 | 25.15% | 12.714 | 26.07% | 12.567 |
| A5 | moderate crop jitter | 6 | 25.54% | 12.802 | 26.15% | 12.487 |

只有 A1 同時改善 validation +/-5 與 MAE，並在 internal test 重現改善。
A2-A5 不納入 A6。A3 的 degradation 定義與目前 validation/runtime crop
表徵不相符，結果明顯退步；在重新建立更精確的 deployed-crop 模擬前不應使用。

A6 使用 A1 設定執行完整 20 epochs。這個選擇不涉及模型架構、輸入輸出
contract 或部署 preprocessing 變更。

## A6 完整訓練

| 項目 | Validation | Internal test |
|---|---:|---:|
| +/-5 | 29.92% | 30.06% |
| MAE | 9.104 | 8.903 |
| +/-3 | 18.31% | 17.79% |
| +/-10 | 58.08% | 58.97% |

- Best epoch：20/20
- 訓練時間：124.39 秒，平均 6.22 秒/epoch
- Best/latest SHA256：`31c020ac465e3d3522139f33cd54dea52de8a087a66ea06107399f4619951f3a`
- 裝置：CPU，TensorFlow 2.15.1，Python 3.10.21

A6 在乾淨 internal split 達到 MAE 小於 10 年的初始目標，但 +/-5 尚未達到
38-40%。是否能改善實際 80px/40px Conditional 指標仍需 ONNX export 後使用
完全相同的 frozen benchmark 驗證，不能由 internal test 直接推論。
