# Age v3 Phase 1-3 報告

## 資料隔離結果

v2 split 在檔名層級沒有重疊，但 exact-content SHA256 audit 發現 train、
validation、internal test 與 frozen benchmark 之間存在重複內容。Age v3
因此改用 SHA256 group 建立全新的固定 split，沒有修改 benchmark manifest。

| Split | 筆數 | Unique content SHA256 |
|---|---:|---:|
| Train | 6,075 | 6,075 |
| Validation | 1,300 | 1,300 |
| Internal test | 1,304 | 1,304 |
| External benchmark | 936 | 932 |

所有 split pair 的 `source_id` 與 content SHA256 overlap 都是 0。External
benchmark 內原有的 4 筆重複內容保留，因為 frozen benchmark 不得修改。

清理統計：

- 排除與 benchmark 內容相同但檔名不同的檔案：13
- 移除標註一致的 duplicate aliases：23
- 隔離標註互相衝突的 identical-content groups：61 組、127 個檔案
- 無效 UTKFace 檔名：2

完整檔案清單位於 `../splits/quarantine.json`，manifest hashes、seed 與交集
驗證位於 `../splits/metadata.json`。

## Age v2 的 80px + 40px Conditional 誤差

| 實際年齡 | 偵測成功 N | MAE | +/-5 | Mean signed error |
|---|---:|---:|---:|---:|
| 0-9 | 160 | 18.430 | 0.00% | +18.430 |
| 10-19 | 162 | 9.371 | 30.25% | +9.371 |
| 20-29 | 174 | 4.796 | 70.11% | +0.693 |
| 30-39 | 171 | 8.804 | 26.90% | -1.512 |
| 40-49 | 159 | 11.819 | 24.53% | -3.038 |
| 50-59 | 154 | 13.694 | 22.08% | -4.202 |
| 60-69 | 149 | 13.079 | 20.13% | -2.632 |
| 70+ | 399 | 13.786 | 27.07% | -6.202 |
| Overall | 1,528 | 11.940 | 28.01% | +0.217 |

Age v2 明顯高估兒童、低估中高齡，呈現向中間年齡回歸。整體 signed error
接近零是兩側偏差互相抵銷，不能視為無偏差。27px 只有 15 個 YuNet 成功
樣本，不用於 v3 model selection。

## 停點

A0-A5 尚未開始。下一步是在使用者核准後建立 Age 專用訓練環境與可配置
training script，先跑單一 epoch smoke test，再執行固定 8 epochs screening。
