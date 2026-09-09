# AI Camera v1 RK3588 距離準確度測試結果

## 1. 測試摘要

| 項目 | 值 |
|---|---|
| 測試日期 | 2026-09-04 |
| 裝置 | Orange Pi 5 Ultra / RK3588 |
| RKNN Toolkit Lite2 | 2.3.2 |
| RKNN Runtime | 2.3.2 |
| RKNPU driver | 0.9.6 |
| 模型版本 | `v1_baseline/rknn/rk3588` FP16 |
| Git 基準 commit | `1b439be34caed5c9b2a6a81686e4d3225e725795` |
| 程式狀態 | working tree modified，不能視為純 commit 結果 |
| 測試樣本 | 9,198 張 |
| 執行時間 | 386 秒 |
| Harness 平均吞吐量 | 約 23.8 samples/s |
| Runtime error | 0 |

Harness 吞吐量包含圖片讀取、YuNet、bbox 配對和下游模型推論，不等同單一模型 FPS 或正式攝影機 pipeline FPS。

## 2. 測試資料

| 資料集 | 用途 | 每個距離樣本數 | 總數 |
|---|---|---:|---:|
| SCFace | YuNet | 130 | 390 |
| UTKFace | Age / Gender | 936 | 2,808 |
| AFLW2000-3D | Head Pose | 2,000 | 6,000 |
| 合計 |  | 3,066 | 9,198 |

所有來源圖片被縮放並置於 `640x480` 黑色畫布，使用三個目標臉高：

- 80 px：標記為 1m
- 40 px：標記為 2m
- 27 px：標記為 3m

這是離線像素退化測試。距離標籤來自臉高換算，不是使用 Orange Pi 攝影機在真實距離拍攝。

Manifest SHA-256：

```text
264af6af1f237ba772ffea2cb0e2cbed3e88c77f1580ddd6d157012f0dd2d7f2
```

## 3. 模型識別

| 模型 | 檔案大小 | SHA-256 |
|---|---:|---|
| Age | 510,961 bytes | `5760f2d6eddbe1c48f531be6b695bb4d4e1c80f7fa72e70b7888f065a6e7fd5c` |
| Gender | 508,913 bytes | `9b98e672a221488884bb193c04d80befd41c5a8949fc4096968903af7f6e5ec2` |
| Head Pose | 1,157,796 bytes | `7ae857cc948e36097237c0052c8d6939222b0dc0b247cfa881fe3e8cd7b8a37d` |
| YuNet | 1,177,699 bytes | `80405947de174074fd16e72bf627d59faf5af0fcb51afec279a826523d0921be` |

## 4. 主要結果

| 模型 | 距離（臉高） | N | 結果 | 95% CI |
|---|---|---:|---:|---|
| YuNet，IoU >= 0.5 且 score >= 0.75 | 1m（80px） | 130 | 100.0% | [97.1, 100.0] |
| YuNet，IoU >= 0.5 且 score >= 0.75 | 2m（40px） | 130 | 100.0% | [97.1, 100.0] |
| YuNet，IoU >= 0.5 且 score >= 0.75 | 3m（27px） | 130 | 82.3% | [74.8, 87.9] |
| Age，絕對誤差 <= 5 歲 E2E | 1m（80px） | 936 | 19.7% | [17.2, 22.3] |
| Age，絕對誤差 <= 5 歲 E2E | 2m（40px） | 936 | 13.6% | [11.5, 15.9] |
| Age，絕對誤差 <= 5 歲 E2E | 3m（27px） | 936 | 0.4% | [0.2, 1.1] |
| Gender，Balanced Accuracy E2E | 1m（80px） | 936 | 70.8% | 未計算 |
| Gender，Balanced Accuracy E2E | 2m（40px） | 936 | 47.0% | 未計算 |
| Gender，Balanced Accuracy E2E | 3m（27px） | 936 | 1.0% | 未計算 |
| Head Pose，三軸環狀誤差都 <= 15 度 E2E | 1m（80px） | 2,000 | 60.9% | [58.7, 63.0] |
| Head Pose，三軸環狀誤差都 <= 15 度 E2E | 2m（40px） | 2,000 | 47.0% | [44.8, 49.2] |
| Head Pose，三軸環狀誤差都 <= 15 度 E2E | 3m（27px） | 2,000 | 21.6% | [19.8, 23.4] |

Gender 使用預先固定的類別定義：UTKFace `0=male, 1=female`，模型輸出 `raw >= 0.5 -> male`。沒有使用測試答案自動選擇映射。

## 5. YuNet 對下游資料的偵測率

| 資料集 | 1m（80px） | 2m（40px） | 3m（27px） |
|---|---:|---:|---:|
| UTKFace | 97.6%（914/936） | 65.6%（614/936） | 1.6%（15/936） |
| AFLW2000-3D | 89.4%（1788/2000） | 64.2%（1284/2000） | 27.4%（547/2000） |

Age 與 Gender 的 3m E2E 成績主要受 YuNet 在 27px UTKFace 上的低偵測率限制，不能直接解讀成下游模型單獨準確率。

## 6. 失敗統計

| 狀態 | 數量 |
|---|---:|
| 通過 YuNet E2E 條件 | 5,529 |
| YuNet 漏偵 | 3,606 |
| 最佳偵測框 IoU < 0.5 | 63 |
| YuNet score < 0.75 | 0 |
| Runtime error | 0 |

YuNet 失敗的樣本會保留在 Age、Gender、Head Pose 的分母，而且不執行下游模型。

## 7. 輸出與重現

完整結果檔：

```text
/home/g2004/datasets/vac-distance/eval_v1_rk3588.jsonl
```

結果檔 SHA-256：

```text
b35f54a66c328d5477288455fda7ab0fa55ba6a46502f660e9045f5553cf36fa
```

Orange Pi 隔離測試目錄：

```text
/home/orangepi/vac-v1-distance-test
```

重複執行方式請見 `AI_camera_distance_accuracy_test_SOP.zh-TW.md`。

## 8. 警告與限制

- RKNN Lite 對 static-shape 模型查詢 dynamic range 時回報警告；四個模型仍成功初始化並完成全部推論。
- 本測試只能證明 RK3588 實機執行結果，不能代替 RK3566 實機驗證。
- 離線臉高退化不能完整模擬真實距離的鏡頭模糊、曝光、雜訊、壓縮、姿態和遮擋。
- 若要比較 fine-tuned 模型，必須沿用相同 manifest、門檻、crop margin 與 Gender mapping。
