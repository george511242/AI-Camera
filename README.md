# VAC AI Camera 模型下載與選版

本頁只列目前建議使用的臉部模型：**YuNet** 人臉偵測、**MobileAgeNet** 年齡首選、**ConvNeXt-Tiny** 年齡準確度備選、**V118 DINOv2-Small** 最新發布 Age 研究版、**EdgeFace-XXS 112** 性別首選、**EdgeFace-XXS 224** 原性別備選、**Gender224 預測框適配版** 最新發布，以及 **Head Pose v2 E1**。

> 更新：2026-11-18（新增最新版 Gender224；原 Gender224、112 與 Age 版本及下載均保留）。推薦代表工程候選，不代表已切換 production，也不代表商用或再散布權利已核准。

## 模型總覽

| 任務 | 推薦模型 | Params | ONNX / RK3588 大小 | 驗證表現 | RK3588 mean / p95 | ONNX | RK3566 | RK3576 | RK3588 |
|---|---|---:|---:|---|---:|:---:|:---:|:---:|:---:|
| Face | **YuNet v1 + corrected BGR/xywh** | — | 0.32 / 1.18 MB | 115 / 57 / 38 px 偵測成功率：97.59 / 95.14 / 88.91% | 35.17 / 43.61 ms | ✅ | ✅ | ✅ | ✅ 實機 |
| Age | **MobileAgeNet（低延遲首選）** | 3.22M | 12.86 / 7.85 MB | V31 DEV：MAE 6.367、CS@7 65.80% | **12.67 / 16.68 ms** | ✅ | ✅ | ✅ | ✅ 實機 parity |
| Age | **ConvNeXt-Tiny（準確度備選）** | 27.90M | 111.66 / 57.68 MB | V31 DEV：**MAE 5.692、CS@7 69.84%** | 55.82 / 60.56 ms | ✅ | ✅ | ✅ | ✅ 實機 parity |
| Age | **V118 DINOv2-Small（最新發布）** | 21.67M | 86.83 / 47.29 MB | Clean4953 ONNX：**MAE 5.533、CS@7 73.57%**；完整門檻未達 | 74.76 / 81.55 ms | ✅ parity | ✅ conversion | ✅ conversion | 12筆工程試跑／完整精度待驗 |
| Gender | **V3 EdgeFace-XXS** | 1.16M | 4.82 / 4.18 MB | 115 / 57 / 38 / 29 / 20 px<br>RK3588 **E2E BA：81.73 / 81.87 / 80.55 / 53.93 / 0.53%** | 7.26 / 9.14 ms | ✅ | ✅ | ✅ | ✅ 實機 parity |
| Gender | EdgeFace-XXS 224（備選） | 1.16M | 4.82 / 4.40 MB | 115 / 57 / 38 / 29 / 20 px<br>RK3588 **E2E BA：65.24 / 60.67 / 67.06 / 43.77 / 0.21%** | 21.11 / 32.73 ms | ✅ | ✅ | ✅ | 五尺度實測／parity 有 FAIL |
| Gender | **EdgeFace-XXS 224 預測框適配（最新發布）** | 1.16M | 4.82 / 4.40 MB | 115 / 57 / 38 / 29 / 20 px<br>RK3588 **E2E BA：87.51 / 84.60 / 83.98 / 55.85 / 0.53%** | **21.07 / 33.09 ms** | ✅ parity | ✅ conversion | ✅ conversion | 五尺度實測／parity 有 FAIL |
| Head Pose | **v2 E1** | — | 0.85 / 1.16 MB | 6,000 張 frozen test：MAE 5.87°、±10° 75.19% | 5.43 / 7.29 ms | ✅ | ✅ | ✅ | ✅ 實機 |

**Gender三列統一比較115／57／38／29／20px的RK3588端到端E2E BA，包含YuNet偵測，漏檢計失敗；不是80／40／27px的GT-crop單模型BA。** BA=(男性Recall+女性Recall)/2；原224在115px女性Recall只有32.00%，最新224為79.58%。

大小採十進位 MB；延遲皆為 Orange Pi 5 Ultra / RK3588、RKNN Runtime 2.3.2、driver 0.9.6 的 model-only 實測，不能直接視為完整 app FPS。RK3566／RK3576 目前僅 conversion PASS，physical runtime 仍為 PENDING。

## 下載與 repo 路徑

大型 `.onnx`／`.rknn` 由 Git LFS 管理；clone 後請執行 `git lfs pull`。RKNN binary 只適用標示的 SoC，不可跨平台共用。

### YuNet — 人臉偵測

Repo：`AICameraInferenceEngine/model/versions/v1_baseline/`

[ONNX](AICameraInferenceEngine/model/versions/v1_baseline/onnx/yunet_n_640_640.onnx) · [RK3566](AICameraInferenceEngine/model/versions/v1_baseline/rknn/rk3566/yunet_n_640_640_fp16_qualified_20260911.rknn) · [RK3576](AICameraInferenceEngine/model/versions/v1_baseline/rknn/rk3576/yunet_n_640_640_fp16.rknn) · [RK3588](AICameraInferenceEngine/model/versions/v1_baseline/rknn/rk3588/yunet_n_640_640_fp16_qualified_20260911.rknn) · [說明](AICameraInferenceEngine/model/versions/v1_baseline/README.md)

- 來源：[ShiqiYu/libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)
- BGR 0–255，補方形後 resize 640×640；沿用 12-output decode，NMS 前使用正確 `xywh`，score 0.75、NMS 0.3。

### MobileAgeNet — 年齡低延遲首選

Repo：`AICameraInferenceEngine/model/versions/age_deployment_20261002/mobileagenet_age/`

[ONNX](AICameraInferenceEngine/model/versions/age_deployment_20261002/mobileagenet_age/model.onnx) · [RK3566](AICameraInferenceEngine/model/versions/age_deployment_20261002/mobileagenet_age/rk3566_fp16.rknn) · [RK3576](AICameraInferenceEngine/model/versions/age_deployment_20261002/mobileagenet_age/rk3576_fp16.rknn) · [RK3588](AICameraInferenceEngine/model/versions/age_deployment_20261002/mobileagenet_age/rk3588_fp16.rknn) · [SHA256](AICameraInferenceEngine/model/versions/age_deployment_20261002/mobileagenet_age/SHA256SUMS.txt)

- 來源：本專案 MobileAgeNet-style scalar-age 訓練，selected epoch 12。
- 224×224 RGB face crop、float32 0–255；正規化已嵌入，輸出為 0–80 歲 scalar age。

### ConvNeXt-Tiny — 年齡準確度備選

Repo：`AICameraInferenceEngine/model/versions/age_deployment_20261002/convnext_tiny_age/`

[ONNX](AICameraInferenceEngine/model/versions/age_deployment_20261002/convnext_tiny_age/model.onnx) · [RK3566](AICameraInferenceEngine/model/versions/age_deployment_20261002/convnext_tiny_age/rk3566_fp16.rknn) · [RK3576](AICameraInferenceEngine/model/versions/age_deployment_20261002/convnext_tiny_age/rk3576_fp16.rknn) · [RK3588](AICameraInferenceEngine/model/versions/age_deployment_20261002/convnext_tiny_age/rk3588_fp16.rknn) · [SHA256](AICameraInferenceEngine/model/versions/age_deployment_20261002/convnext_tiny_age/SHA256SUMS.txt)

- 來源：ImageNet-pretrained ConvNeXt-Tiny，本專案 direct age fine-tune，selected epoch 5。
- 224×224 RGB face crop、float32 0–255；輸出 100 ordinal logits，`age = sum(sigmoid(logits))`。

[年齡模型完整前處理、轉換與 parity 說明](AICameraInferenceEngine/model/versions/age_deployment_20261002/README.md)

### Age V118 DINOv2-Small — 最新發布的 Age 研究版本（新增）

Repo：`AICameraInferenceEngine/model/versions/age_v118_dinov2_small_20261117/`

[ONNX](AICameraInferenceEngine/model/versions/age_v118_dinov2_small_20261117/model.onnx) · [RK3566](AICameraInferenceEngine/model/versions/age_v118_dinov2_small_20261117/rk3566_fp16.rknn) · [RK3576](AICameraInferenceEngine/model/versions/age_v118_dinov2_small_20261117/rk3576_fp16.rknn) · [RK3588](AICameraInferenceEngine/model/versions/age_v118_dinov2_small_20261117/rk3588_fp16.rknn) · [SHA256](AICameraInferenceEngine/model/versions/age_v118_dinov2_small_20261117/SHA256SUMS.txt) · [完整說明](AICameraInferenceEngine/model/versions/age_v118_dinov2_small_20261117/README.md)

- 來源：本專案 DINOv2-Small Age 研究，V118 selected epoch 2；21,667,685 parameters。這是本頁最新新增的 Age 發布版本，**原 MobileAgeNet／ConvNeXt-Tiny 版本與下載完整保留，不自動替換 production**。
- 224×224 RGB face crop、NCHW、raw 0–255，ONNX float32，normalization 已內建。輸出 **101 類別 logits**，使用固定 softmax global median 解碼；不能沿用 Tiny 的 `sum(sigmoid(100 logits))`。解碼程式隨包提供。
- 保存的 Clean4953 ONNX CPU：MAE **5.533212195**、CS@7 **73.571573%**。115／57／38／29／20 px GT-crop CS@7 為 **76.31／75.80／72.73／73.11／67.35%**；五尺寸完整 gate 仍 **0/5 PASS**，Clean 也尚未通過原整套門檻。
- 三個 target 的 FP16 conversion 與檔案 SHA／size 均通過；ONNX 86.833 MB 另列，三個 RKNN 分別46.570／50.724／47.289 MB。RK3566／RK3576 physical runtime 仍待驗。
- RK3588 曾用同12筆TRAIN/degraded工程輸入進行50warmup/500timed：mean/P95 **74.76／81.55 ms**；Tiny同條件 **56.78／62.25 ms**。這是model-only速度，非相機FPS；V118工程輸入對ONNX年齡平均/最大差0.166667/1歲，不等於完整NPU準確度或正式hardwareparity PASS。
- 本次為新增發布，不代表精度門檻、正式產品驗收或資料商用清權已完成。

### EdgeFace-XXS — 性別首選

Repo：`AICameraInferenceEngine/model/versions/gender_v3_bounded_20261006/selected/edgeface_xxs/`

[ONNX](AICameraInferenceEngine/model/versions/gender_v3_bounded_20261006/selected/edgeface_xxs/edgeface_xxs_gender.onnx) · [RK3566](AICameraInferenceEngine/model/versions/gender_v3_bounded_20261006/selected/edgeface_xxs/rk3566/edgeface_xxs_gender_fp16.rknn) · [RK3576](AICameraInferenceEngine/model/versions/gender_v3_bounded_20261006/selected/edgeface_xxs/rk3576/edgeface_xxs_gender_fp16.rknn) · [RK3588](AICameraInferenceEngine/model/versions/gender_v3_bounded_20261006/selected/edgeface_xxs/rk3588/edgeface_xxs_gender_fp16.rknn) · [使用說明](AICameraInferenceEngine/model/versions/gender_v3_bounded_20261006/selected/edgeface_xxs/README.md)

- 來源：[EdgeFace](https://github.com/otroshi/edgeface) XXS face-recognition pretrained checkpoint；本專案以 UTKFace fine-tune，selected epoch 7。
- 112×112 RGB face crop、float32 0–255；單一 logit，threshold `0.3863107562`，大於等於門檻判為 Male。

### EdgeFace-XXS 224 — 性別備選

Repo：`AICameraInferenceEngine/model/versions/gender_edgeface_224_20261007/`

[ONNX](AICameraInferenceEngine/model/versions/gender_edgeface_224_20261007/edgeface_xxs_gender_224.onnx) · [RK3566](AICameraInferenceEngine/model/versions/gender_edgeface_224_20261007/rk3566/edgeface_xxs_gender_224_fp16.rknn) · [RK3576](AICameraInferenceEngine/model/versions/gender_edgeface_224_20261007/rk3576/edgeface_xxs_gender_224_fp16.rknn) · [RK3588](AICameraInferenceEngine/model/versions/gender_edgeface_224_20261007/rk3588/edgeface_xxs_gender_224_fp16.rknn) · [SHA256](AICameraInferenceEngine/model/versions/gender_edgeface_224_20261007/SHA256SUMS.txt) · [完整說明](AICameraInferenceEngine/model/versions/gender_edgeface_224_20261007/README.md)

- 來源同首選 EdgeFace-XXS；本專案以相同 UTKFace split 與 protocol 重新進行 224×224 fine-tune，selected epoch 8。
- 224×224 RGB face crop、float32 0–255；單一 logit，threshold `-0.1358056068`，大於等於門檻判為 Male。
- 歷史80／40／27px **GT-crop單模型** mean BAcc 88.68%，不含YuNet偵測誤差，**不能當作五尺度E2E表現**；該歷史GT比較較112低0.59pp（80px較佳，40／27px較差）。原224五尺度E2E以本頁總覽及新舊比較表為準。
- ONNX parity 與三平台 FP16 conversion PASS；RK3566／3576 physical 待驗。原版 RK3588 2026-11-17 五尺度 h115 E2E BA 65.24%、mean/P95 21.11／32.73ms，正式 parity maxprobdelta0.06671>.05；原版本與檔案保留。

### EdgeFace-XXS 224 — 最新發布版本（預測框裁切適配，新增）

Repo：`AICameraInferenceEngine/model/versions/gender224_edgeface_latest_20261118/`

[ONNX](AICameraInferenceEngine/model/versions/gender224_edgeface_latest_20261118/model.onnx) · [RK3566](AICameraInferenceEngine/model/versions/gender224_edgeface_latest_20261118/rk3566_fp16.rknn) · [RK3576](AICameraInferenceEngine/model/versions/gender224_edgeface_latest_20261118/rk3576_fp16.rknn) · [RK3588](AICameraInferenceEngine/model/versions/gender224_edgeface_latest_20261118/rk3588_fp16.rknn) · [SHA256](AICameraInferenceEngine/model/versions/gender224_edgeface_latest_20261118/SHA256SUMS.txt) · [完整說明](AICameraInferenceEngine/model/versions/gender224_edgeface_latest_20261118/README.md)

- 本專案 EdgeFace-XXS 224 預測框適配，selected epoch6；**本頁最新新增的 Gender224 發布版本，原2026-10-07 Gender224及112下載完整保留，不自動替換 production**。
- 224×224 RGB、NCHW、raw0–255，normalization `x/127.5-1` 已內建。輸出一male logit，**新threshold `-0.23273582011461258`，大於等於為Male**；不能沿用原224的−0.1358056068或112的+0.3863107562。
- 同936個UTKFace來源×五臉高640×480、每尺度Male461／Female475，原固定corrected YuNet804保存框、IoU≥.5；E2E漏檢計失敗，BA為男女Recall平均。h115男性Recall **98.48%→95.44%**、女性 **32.00%→79.58%**；不是每一性別都比原版提高。

|臉高|原224 E2E BA|最新224 E2E BA|112參考 E2E BA|
|---|---:|---:|---:|
|115px|65.2408%|**87.5118%**|81.7285%|
|57px|60.6729%|**84.6026%**|81.8655%|
|38px|67.0588%|**83.9838%**|80.5514%|
|29px|43.7723%|**55.8520%**|53.9317%|
|20px|0.2137%|**0.5295%**|0.5295%|

新舊速度比較（RK3588 core0／batch1，50warmup＋500timed，未鎖頻、不同日量測）：

|版本|Mean ms|P50 ms|P95 ms|
|---|---:|---:|---:|
|原224（2026-11-17）|21.1076|18.4632|32.7287|
|最新224（2026-11-18）|**21.0749**|18.5015|33.0933|
|112參考（2026-11-12）|7.2808|6.7831|10.9498|

- **新舊224速度大致相同**，不能由0.033ms均值差宣稱穩定加速；最新224仍約112的2.9倍延遲。Model-only包含RKNN呼叫／layout／outputreturn，不含YuNet、裁切與相機，非完整FPS。
- h115 GT-crop BA92.00%，不是原始Clean；**新版本原始清晰影像Clean尚未獨立實測**。20px只有5/936有效偵測，不把conditional100%當成整體能力。
- 指定h115 E2E≥11281.73%已達87.51%；但**完整8074 samecrop parity仍FAIL：h57預測crop maxprobdelta0.05237>.05**，該組分類一致100%，其他9組PASS；整體分類一致99.8142%。不放寬gate或把精度達標稱所有工程驗收通過。
- RK3566／3576 conversion PASS、physical待驗；研究benchmark已重用、非新盲測。資料與訓練／15格GT-conditional-E2E／完整SHA見[版本說明](AICameraInferenceEngine/model/versions/gender224_edgeface_latest_20261118/README.md)、[新舊比較CSV](AICameraInferenceEngine/model/versions/gender224_edgeface_latest_20261118/comparison.csv)。

### Head Pose v2 E1 — 頭部姿態首選

Repo：`AICameraInferenceEngine/model/versions/pose_v2_runtime/`

[ONNX](AICameraInferenceEngine/model/versions/pose_v2_runtime/onnx/head_pose_v2_e1_224x224.onnx) · [RK3566](AICameraInferenceEngine/model/versions/pose_v2_runtime/rknn/rk3566/head_pose_v2_e1_224x224_fp16.rknn) · [RK3576](AICameraInferenceEngine/model/versions/pose_v2_runtime/rknn/rk3576/head_pose_v2_e1_224x224_fp16.rknn) · [RK3588](AICameraInferenceEngine/model/versions/pose_v2_runtime/rknn/rk3588/head_pose_v2_e1_224x224_fp16.rknn) · [RK3588 實測](AICameraInferenceEngine/model/versions/pose_v2_runtime/deployment/rk3588/result.json)

- 來源：[Lightweight Head Pose Estimation](https://github.com/Shaw-git/Lightweight-Head-Pose-Estimation) 66-bin checkpoint；本專案 frozen v2 epoch 1。
- margin 0.6、224×224 RGB square-padded face crop；輸出 `roll, yaw, pitch`（度）。

## 訓練與驗證資料

| 任務 | 訓練／微調資料 | 驗證資料與分布 |
|---|---|---|
| YuNet | 官方 pretrained weight，本專案未重訓 | Phase1 每尺度 3,066 張：SCface 130、UTKFace 936、AFLW2000 2,000；模擬 115／57／38 px |
| Age | UTKFace + Remaining Lifespan；V31 train 44,269 | DEV 4,953：Child 404、Adult 3,280、Elderly 1,269；與 train 無 exact hash／identity overlap |
| Gender | UTKFace train 6,188 | validation 1,325、internal test 1,329；frozen external 936（Male 461／Female 475），各測 80／40／27 px |
| Gender224 最新適配版 | 原TRAIN照片SHA隔離後6,100，6,087 predictedcrop；半pred115半GT80/40/27，6輪 | 去除已知照片重疊後validation1,301；同external936×115／57／38／29／20五尺度；Clean未測 |
| Head Pose | 300W-LP 系列來源；train 97,698 source images | validation 24,717；train／validation source-file 與 source-group overlap 均為 0；另有 6,000 張三尺度 frozen test |

Age 群組：Child 0–17、Adult 18–54、Elderly 55–80；DEV 分布為 8.16%／66.22%／25.62%。Gender train 為 Male 44.23%、Female 55.77%。完整證據見 [Age metrics](AICameraInferenceEngine/model/versions/supervisor_age_model_comparison/metrics.csv)、[Gender V3 摘要](AICameraInferenceEngine/model/versions/gender_v3_bounded_20261006/SUPERVISOR_GENDER_V1_V2_V3_SUMMARY.zh-TW.md) 與 [Pose integrity](AICameraInferenceEngine/model/versions/pose_v2_runtime/audit/validation_integrity.json)。

## 部署狀態

- MobileAgeNet、ConvNeXt-Tiny 與 EdgeFace-XXS 尚未接入 runtime selector，也未核准自動取代 production；原 Gender224 與最新預測框適配版並列保留。最新版本E2E精度達標不等於完整parity或部署驗收通過。
- 維護現行 production 時仍需保留既有版本與設定，並明確使用 `RKNPU_ARTIFACT_SET=qualified_20260911`。
- Artifact 僅供內部／研究工程驗證；`RESEARCH_ACCESS` 不等於 `COMMERCIAL_RIGHTS`。
- 最新狀態以 [HANDOFF.md](AICameraInferenceEngine/HANDOFF.md) 最上方紀錄為準。
