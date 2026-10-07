# VAC AI Camera 模型下載與選版

本頁只列目前建議使用的臉部模型：**YuNet** 人臉偵測、**MobileAgeNet** 年齡首選、**ConvNeXt-Tiny** 年齡準確度備選、**EdgeFace-XXS 112** 性別首選、**EdgeFace-XXS 224** 性別備選，以及 **Head Pose v2 E1**。

> 更新：2026-10-07。推薦代表工程候選，不代表已切換 production，也不代表商用或再散布權利已核准。

## 模型總覽

| 任務 | 推薦模型 | Params | ONNX / RK3588 大小 | 驗證表現 | RK3588 mean / p95 | ONNX | RK3566 | RK3576 | RK3588 |
|---|---|---:|---:|---|---:|:---:|:---:|:---:|:---:|
| Face | **YuNet v1 + corrected BGR/xywh** | — | 0.32 / 1.18 MB | 115 / 57 / 38 px 偵測成功率：97.59 / 95.14 / 88.91% | 35.17 / 43.61 ms | ✅ | ✅ | ✅ | ✅ 實機 |
| Age | **MobileAgeNet（低延遲首選）** | 3.22M | 12.86 / 7.85 MB | V31 DEV：MAE 6.367、CS@7 65.80% | **12.67 / 16.68 ms** | ✅ | ✅ | ✅ | ✅ 實機 parity |
| Age | **ConvNeXt-Tiny（準確度備選）** | 27.90M | 111.66 / 57.68 MB | V31 DEV：**MAE 5.692、CS@7 69.84%** | 55.82 / 60.56 ms | ✅ | ✅ | ✅ | ✅ 實機 parity |
| Gender | **V3 EdgeFace-XXS** | 1.16M | 4.82 / 4.18 MB | 80 / 40 / 27 px BAcc：89.75 / 89.55 / 88.50% | 7.26 / 9.14 ms | ✅ | ✅ | ✅ | ✅ 實機 parity |
| Gender | EdgeFace-XXS 224（備選） | 1.16M | 4.82 / 4.40 MB | 80 / 40 / 27 px BAcc：90.28 / 88.92 / 87.35% | PENDING | ✅ | ✅ | ✅ | ✅ conversion |
| Head Pose | **v2 E1** | — | 0.85 / 1.16 MB | 6,000 張 frozen test：MAE 5.87°、±10° 75.19% | 5.43 / 7.29 ms | ✅ | ✅ | ✅ | ✅ 實機 |

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
- Frozen external mean BAcc 88.68%，較 112 首選低 0.59 個百分點；80 px 較佳，但 40／27 px 較差，因此只列為備選。
- ONNX parity 與三平台 FP16 conversion PASS；三平台 physical runtime/parity 均為 PENDING。

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
| Head Pose | 300W-LP 系列來源；train 97,698 source images | validation 24,717；train／validation source-file 與 source-group overlap 均為 0；另有 6,000 張三尺度 frozen test |

Age 群組：Child 0–17、Adult 18–54、Elderly 55–80；DEV 分布為 8.16%／66.22%／25.62%。Gender train 為 Male 44.23%、Female 55.77%。完整證據見 [Age metrics](AICameraInferenceEngine/model/versions/supervisor_age_model_comparison/metrics.csv)、[Gender V3 摘要](AICameraInferenceEngine/model/versions/gender_v3_bounded_20261006/SUPERVISOR_GENDER_V1_V2_V3_SUMMARY.zh-TW.md) 與 [Pose integrity](AICameraInferenceEngine/model/versions/pose_v2_runtime/audit/validation_integrity.json)。

## 部署狀態

- MobileAgeNet、ConvNeXt-Tiny 與 EdgeFace-XXS 尚未接入 runtime selector，也未核准自動取代 production；EdgeFace-XXS 224 僅為備選。
- 維護現行 production 時仍需保留既有版本與設定，並明確使用 `RKNPU_ARTIFACT_SET=qualified_20260911`。
- Artifact 僅供內部／研究工程驗證；`RESEARCH_ACCESS` 不等於 `COMMERCIAL_RIGHTS`。
- 最新狀態以 [HANDOFF.md](AICameraInferenceEngine/HANDOFF.md) 最上方紀錄為準。
