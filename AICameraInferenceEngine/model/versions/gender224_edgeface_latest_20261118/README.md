# Gender224 EdgeFace-XXS — 最新發布版本（預測框裁切適配）

新增日期：2026-11-18。新增發布selected epoch6；**保留原2026-10-07 Gender224與112版本，不更改production預設**。本版改善固定YuNet→Gender五尺度E2E精度；「最新版」不代表所有parity或產品驗收皆已通過。

## 模型下載

[ONNX FP32](model.onnx) · [RK3566 FP16](rk3566_fp16.rknn) · [RK3576 FP16](rk3576_fp16.rknn) · [RK3588 FP16](rk3588_fp16.rknn) · [SHA256](SHA256SUMS.txt) · [metadata](metadata.json)

|檔案|大小（十進位MB）|狀態|
|---|---:|---|
|model.onnx|4.824|ONNX parity PASS|
|rk3566_fp16.rknn|3.827|FP16 conversion PASS|
|rk3576_fp16.rknn|5.316|FP16 conversion PASS|
|rk3588_fp16.rknn|4.400|FP16 conversion PASS|

## 前處理與解碼

- 224×224 RGB、batch1 NCHW、raw0–255；ONNX輸入float32，實測RKNN呼叫uint8 NCHW。內建 `x/127.5-1`，外部不要再normalize。
- bbox採normalized xyxy，先各邊增加原寬/高0.45倍、clamp至影像界線、轉int pixel裁切，再RGB／LINEAR resize224。
- 輸出一個male logit。**新threshold=-0.23273582011461258，logit≥threshold為Male，否則Female**；不是概率threshold、不能沿用原224的−0.13580560684204102或112的+0.38631075620651245。
- RKNN只適用檔名對應SoC。三平台模型均來自同selected6 checkpoint；RK3566／3576只有conversion，實體runtime待驗。

## 新舊版本五尺度比較

同936個UTKFace來源×五臉高=4680個640×480影像，每尺度Male461／Female475。新舊224皆RK3588新Gender推論，使用同112原corrected BGR/xywh YuNet804保存框；未重跑YuNet，未替換成同事3718版本。匹配IoU≥.5、E2E漏檢計失敗、BA=(MaleRecall+FemaleRecall)/2。

|臉高|原224 E2E BA|新224 E2E BA|提升pp|112參考E2E BA|
|---|---:|---:|---:|---:|
|115|65.2408%|**87.5118%**|+22.2710|81.7285%|
|57|60.6729%|**84.6026%**|+23.9297|81.8655%|
|38|67.0588%|**83.9838%**|+16.9250|80.5514%|
|29|43.7723%|**55.8520%**|+12.0797|53.9317%|
|20|0.2137%|**0.5295%**|+0.3158|0.5295%|

h115：Male Recall **98.48%→95.44%**，Female Recall **32.00%→79.58%**，BA **65.24%→87.51%**。新版本高於112參考81.73%，但不能稱兩性每項都比原224提高。20px只有5/936有效偵測，conditional100%不能視為整體100%。

新版本h115 GT-crop BA92.00%（Male93.06%、Female90.95%）是降成115px的GT裁切，**不是原始Clean**。本版原始清晰影像Clean尚未獨立實測，不以GT代稱。全GT／conditional／E2E的15格結果見[metrics.csv](metrics.csv)，逐尺度新舊Recall見[comparison.csv](comparison.csv)。

## RK3588推論速度

|版本|Mean ms|P50 ms|P95 ms|
|---|---:|---:|---:|
|原224（2026-11-17）|21.1076|18.4632|32.7287|
|新224（2026-11-18）|21.0749|18.5015|33.0933|
|112參考（2026-11-12）|7.2808|6.7831|10.9498|

Orange Pi5Ultra RK3588／driver0.9.6／runtime2.3.2、Gender core0、batch1、50warmup＋500timed、未鎖頻。計時含RKNN呼叫/layout/outputreturn，不含YuNet／crop／camera／Python解碼，非整機FPS。不同日量測，只作描述性比較：新舊224速度大致相同，不能把0.033ms均值差當穩定加速；新224仍約112的2.9倍延遲。

## 訓練、標準與限制

- 原UTKFace TRAIN6188按照片SHA排除已知跨split重疊與重複後6100，6087個predictedcrop可用；每批男女平衡、半pred115半原GT80/40/27，6輪／1146updates／36672重抽曝光。原224selected8初始化、1301validation-only選epoch與一global threshold，不用正式936挑參。未知人物／預訓練曝光仍不保證隔離。
- 指定h115 RK3588 E2E BA≥112的81.7285%已達87.5118%；完整五尺度如表。這是固定重用的研究benchmark，非新盲測或所有真實相機距離保證。
- ONNX對selected CUDA全1301validation分類100%一致，maxlogit1.4961e−5；實機validation原parity gates PASS。
- **正式8074同cropONNX/NPU：分類一致99.8142%、meanprobdelta0.00177365、maxprobdelta0.05237050。h57預測crop max>.05，完整parity FAIL；其他9組PASS，h115 GT/det兩組PASS。** h57det分類一致100%仍不抵銷數值gate FAIL；不放寬原mean≤.01/max≤.05/agreement≥.99標準。
- Production未替換；原版本下載保留。資料／模型商用權利未由工程測試建立。

## 學術來源

骨幹沿用[EdgeFace官方程式與權重](https://github.com/otroshi/edgeface)，架構背景見[EdgeFace論文](https://arxiv.org/html/2307.01838v2)、[EdgeNeXt](https://arxiv.org/abs/2206.10589)。本輪標籤與影像來自[UTKFace作者資料](https://github.com/aicip/UTKFace/blob/master/README.md)。本次crop混合與配對驗證為本專案工程實驗，不宣稱重現所有論文；未新增FairFace／CelebA訓練影像。

## 完整性

模型使用Git LFS。下載後在本目錄執行：

```bash
git lfs pull
sha256sum -c SHA256SUMS.txt
```
