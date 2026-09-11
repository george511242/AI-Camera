# 模型選版與使用說明

更新日期：2026-09-11。適用人臉偵測、年齡、性別與頭部姿態；人物偵測／骨架模型不在本次選版範圍。

**後續工程整合與實機資格測試請使用：原始 YuNet＋corrected BGR/xywh、Age v4、Gender v2、Pose v2 E1，並明確指定 `RKNPU_ARTIFACT_SET=qualified_20260911`。** 這組是目前候選，尚未核准取代 production。維護現行正式環境時，保留原有版本與設定。

本文依 [HANDOFF 最新紀錄](../HANDOFF.md)與[三平台資格報告](../reports/05_integration/three_platform_candidate_qualification_20260911.zh-TW.md)整理；後續狀態更新以 HANDOFF 最上方紀錄為準。

## 1. 該選哪個版本？

| 元件 | 目前工程候選 | 程式選版設定 | 現行 production 預設 |
|---|---|---|---|
| YuNet 人臉偵測 | 原始 v1 權重＋BGR／正確 xywh NMS | `YUNET_PIPELINE=bgr_xywh` | 原始權重＋`legacy` |
| Age 年齡 | v4 MobileNetV3-Large，112×112 | `AGE_MODEL_VERSION=v4` | v2 |
| Gender 性別 | v2 SSR-Net，64×64 | `GENDER_MODEL_VERSION=v2` | v1 |
| Head Pose 頭部姿態 | frozen v2 E1（固定 epoch 1） | `HEAD_POSE_MODEL_VERSION=v2_e1` | v1 |

YuNet 的改善是前後處理修正，沒有換成小臉微調權重。Pose 必須選 `pose_v2_runtime` 的 E1，不能拿舊 v2 實驗 checkpoint 代替。選版原因與歷史成效見[版本對照](../reports/00_overview/project_versions_and_status.zh-TW.md)。

**不要因版本號較新就替換：**

- Age v3／v3.1／v3.2 為歷史研究；目前候選仍是 v4。
- Age v5：實驗 FAILED／CLOSED，不採用。
- Age v6 MIVIA：LICENSE BLOCKED／ARCHIVED，未下載或產生可用模型。
- Age v6 公司資料自訓：僅策略提案，資料權利待審查，尚無權重；報告中的預定路徑不代表檔案存在。
- YuNet run1／run2：研究已停止，不採用。
- `original/` 是原始 RK3566 檔案封存區；不可拿其中的 RKNN 直接跑其他 SoC，也不要覆寫或刪除。

## 2. 模型檔案在哪裡？

以下路徑皆相對於 **`AICameraInferenceEngine/model/`**（本 README 所在目錄）。應用程式使用絕對路徑 `/model/versions/...`，因此部署時須將此完整 `model/` 目錄掛載到 `/model`，保留版本及平台子目錄。

### RKNN：裝置 NPU 使用

所有列出的候選都是 FP16，各平台使用自己的 binary。表內 `<soc>` 必須替換為實際晶片的 `rk3566`、`rk3576` 或 `rk3588`。

| 元件 | 目錄 | RK3566／RK3588 檔名 | RK3576 檔名 |
|---|---|---|---|
| YuNet | `versions/v1_baseline/rknn/<soc>/` | `yunet_n_640_640_fp16_qualified_20260911.rknn` | `yunet_n_640_640_fp16.rknn` |
| Age v4 | `versions/v4_age_mobilenetv3/rknn/<soc>/` | `age_v4_mobilenetv3_large_112_fp16_qualified_20260911.rknn` | `age_v4_mobilenetv3_large_112_fp16.rknn` |
| Gender v2 | `versions/v2_finetuned/rknn/<soc>/` | `gender_ssrnet_v2_64x64_fp16.rknn` | 同左 |
| Pose v2 E1 | `versions/pose_v2_runtime/rknn/<soc>/` | `head_pose_v2_e1_224x224_fp16.rknn` | 同左 |

例如 RK3588 的 Age 完整應用程式路徑為：

```text
/model/versions/v4_age_mobilenetv3/rknn/rk3588/age_v4_mobilenetv3_large_112_fp16_qualified_20260911.rknn
```

RK3566／RK3588 的 YuNet 與 Age 舊檔仍保留，但缺完整來源雜湊追溯證據，9/11 另建上述帶日期的新檔。**未設定 artifact set 時，程式仍會選歷史檔案。** RK3576 的新檔沒有日期後綴；Gender／Pose 則沿用已查核檔案，不要自行改名。

### ONNX：離線參考與數值比對使用

| 元件 | Canonical ONNX 路徑 |
|---|---|
| YuNet | [versions/v1_baseline/onnx/yunet_n_640_640.onnx](versions/v1_baseline/onnx/yunet_n_640_640.onnx) |
| Age v4 | [versions/v4_age_mobilenetv3/onnx/age_v4_mobilenetv3_large_112.onnx](versions/v4_age_mobilenetv3/onnx/age_v4_mobilenetv3_large_112.onnx) |
| Gender v2 | [versions/v2_finetuned/onnx/gender_ssrnet_v2_64x64.onnx](versions/v2_finetuned/onnx/gender_ssrnet_v2_64x64.onnx) |
| Pose v2 E1 | [versions/pose_v2_runtime/onnx/head_pose_v2_e1_224x224.onnx](versions/pose_v2_runtime/onnx/head_pose_v2_e1_224x224.onnx) |

每個模型的三平台 RKNN 都對應同一個 canonical ONNX SHA256。來源 checkpoint、各檔 SHA256、Toolkit2 2.3.2 轉換紀錄與警告見 [qualification/result.json](qualification/three_platform_20260911/result.json)。交付時依此核對雜湊；不要使用 `preflight`／未訓練模型或僅憑檔名相似選檔。

## 3. 如何明確啟用候選？

以下是 **Linux 工程整合範例，尚未完成完整 app／相機驗證**。在已安裝專案依賴、可使用 RKNN runtime、已掛載 `/model` 的環境，於 `AICameraInferenceEngine/` 執行。Redis、設定後端與影像來源也須可用。

```bash
(
  export RKNPU_PLATFORM=rk3588
  export RKNPU_ARTIFACT_SET=qualified_20260911
  export YUNET_PIPELINE=bgr_xywh
  export AGE_MODEL_VERSION=v4
  export GENDER_MODEL_VERSION=v2
  export HEAD_POSE_MODEL_VERSION=v2_e1
  python -m server
)
```

- 依實際硬體將 `RKNPU_PLATFORM` 改為 `rk3566` 或 `rk3576`；不可跨晶片共用 RKNN。
- 容器啟動時，以上六個變數須傳入容器環境，僅在 host shell export 不會自動傳入容器。
- 既有 [candidate-rk3588.env](../configs/candidate-rk3588.env) **未包含** `RKNPU_ARTIFACT_SET`；使用該檔時仍須補上 `qualified_20260911`。
- 使用 `leave-zone-with-facial-features` 模式才會包含這四個模型；其他模式僅載入其既有子集合。
- 啟動入口是 `python -m server`；`python main.py` 不會啟動伺服器。
- 未支援的版本／平台會直接報錯，不會自動換模型。RK3576 沒有歷史 Age v2／Gender v1／Pose v1 的路徑對應，不能只改平台而保留預設版本。

若要在 RK3588 還原歷史模型組合，可明確指定以下設定；RK3566 同理改平台。這是回復歷史設定，不代表新增部署認證。

```bash
RKNPU_PLATFORM=rk3588 RKNPU_ARTIFACT_SET=historical \
YUNET_PIPELINE=legacy AGE_MODEL_VERSION=v2 \
GENDER_MODEL_VERSION=v1 HEAD_POSE_MODEL_VERSION=v1 python -m server
```

## 4. 前後處理不能混用

| 模型 | 輸入與裁切 | 輸出／注意事項 |
|---|---|---|
| YuNet | BGR、0–255、補方形至 640×640 | 沿用 12 個輸出與既有 decode；NMS 前將 xyxy 轉 xywh，score 0.75／NMS 0.3 |
| Age v4 | RGB、0–255、112×112、margin 0.45 | 內建 −1～1 正規化；輸出已是單一年齡，不可再做 `1+sum` 解碼 |
| Gender v2 | BGR、0–255、64×64、margin 0.45 | 單一值 ≥0.5 判男，否則女；不要誤用 Age 的 RGB 前處理 |
| Pose v2 E1 | RGB、224×224、margin 0.6、既有方形補邊 | 輸出順序 **roll、yaw、pitch**，單位為度；不是人體骨架 pose |

ONNX 輸入為 float32 NCHW。既有 RKNN host 輸入採 NHWC raw pixels，Age／Pose 使用 float32；**FP16 是模型計算格式，不能直接推定 host buffer 也應傳 float16**。實際 native tensor dtype／layout／stride 仍須在目標裝置查詢確認。

Pose 的 ONNX 端需要 ImageNet 正規化；RKNN 端由轉換設定的 mean/std 處理，host 傳 RGB 0–255，不能重複正規化。裁切、補邊與 resize 請沿用既有程式；自行新增 alignment 或改 margin，將無法直接沿用既有評估結論。

## 5. 已驗證到哪裡？有哪些限制？

| 項目 | 截至 2026-09-11 的狀態 |
|---|---|
| 四模型 × RK3566／RK3576／RK3588 FP16 轉換 | 全部 PASS |
| 本次資格查核的三平台實機數值比對 | 全部 PENDING |
| 歷史 RK3588 離線實機評估 | 有紀錄；新 YuNet／Age binary 不繼承舊檔的 runtime PASS |
| 完整 app、實體相機、Android／JNI 整合 | 尚未完成驗證 |
| Phase1 HFOV63° 驗收 | ONNX CPU 離線測試仍有未達標項目；不能宣稱通過 |
| 商用／產品出貨權利 | 本次技術轉換未新增授權認定，需依既有權利審查另行確認 |

Phase1 的 115／57／38px 是 HFOV63° 下 1／2／3m 的離線模擬條件，不是實體相機距離驗證。年齡、女性性別準確率等仍有差距，完整數字見 [Phase1 驗收差距](../reports/01_benchmark/phase1_hfov63_acceptance_gap_20260910.zh-TW.md)。

資料去重或模型轉換成功不代表訓練資料／權重已獲商用及再散布許可；現有資料與 v6 的限制見 [Age v6 權利與候選報告](../reports/02_age_gender/age_v6_license_safe_candidate_20260911.md)。不要將本文的工程候選推薦當成產品出貨許可。

交接後優先使用已完成的 binary 與 [qualification 固定輸入及紀錄](qualification/three_platform_20260911/)進行各平台實機 parity、tensor query，再驗證 app／Android／相機整合。Age ONNX→實機 RKNN 誤差門檻為 ≤0.1 歲，其餘依資格報告與既有模型規範。無設備時保留 PENDING，不需要重訓或重跑已完成的轉換／benchmark。

新增模型版本時建立新目錄，保留原 artifact；各版 README 應記錄來源、checkpoint 雜湊、ONNX 介面、Toolkit 版本、轉換設定、平台、警告及驗證結果。其他歷史分析由[報告中心](../reports/README.md)查閱。
