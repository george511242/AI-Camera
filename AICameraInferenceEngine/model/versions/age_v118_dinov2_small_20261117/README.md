# Age V118 DINOv2-Small — 最新發布的 Age 研究版本

新增日期：2026-11-17。此目錄新增 V118 selected epoch 2 的部署檔案，**保留原 MobileAgeNet、ConvNeXt-Tiny 與其他 Age 版本，不修改 production 預設**。「最新版」表示本次最新新增的 Age 發布版本，不代表後續編號研究候選都優於它，亦不代表完整精度門檻或商用清權已通過。

## 下載

- [ONNX FP32](model.onnx)
- [RK3566 FP16](rk3566_fp16.rknn)
- [RK3576 FP16](rk3576_fp16.rknn)
- [RK3588 FP16](rk3588_fp16.rknn)
- [SHA256 sums](SHA256SUMS.txt) · [完整 metadata](metadata.json) · [固定年齡解碼](runtime.py)

| Artifact | 大小（十進位 MB） | 狀態 |
|---|---:|---|
|model.onnx|86.833|ONNX parity PASS|
|rk3566_fp16.rknn|46.570|獨立FP16 conversion PASS|
|rk3576_fp16.rknn|50.724|獨立FP16 conversion PASS|
|rk3588_fp16.rknn|47.289|獨立FP16 conversion PASS|

參數量21,667,685。三個RKNN各小於80,000,000bytes；FP32 ONNX為86,832,972bytes，不能宣稱ONNX小於80MB。RKNN檔案只適用其標示SoC，不可跨平台共用。

## 輸入與輸出契約

- 輸入：224×224 RGB face crop、batch1、NCHW、raw0–255；ONNX為float32，normalization已內建，不再除255或額外ImageNet normalization。
- 五尺度評估使用bbox每邊margin0.45、LINEAR resize；部署時保持相同裁切。
- 輸出：101個age logits，**不同於Tiny的100 ordinal logits**。
- 固定解碼：先softmax，再取全域median（第一個CDF≥0.5的年齡）；不能使用Tiny的`sum(sigmoid(logits))`，也不是直接argmax或expected age。

```python
from runtime import decode
age = decode(logits)  # logits shape [1, 101]
```

## 準確度與限制

Clean4953（0–80歲）ONNX CPU：MAE **5.533212195歲**、CS@7 **73.571573%**。Child/Adult/Elderly分組見metadata。

|臉高|GT MAE（歲）|GT CS@7|完整GT gates|
|---|---:|---:|---|
|115px|4.882202|76.312%|FAIL|
|57px|5.106274|75.800%|FAIL|
|38px|5.343150|72.727%|FAIL|
|29px|5.545455|73.111%|FAIL|
|20px|6.130602|67.350%|FAIL|

每尺度781張0–80歲；另155張81–110歲為OOD，不混入正式分數。640×480上的五尺度GT為UTK原chip proxy框，非新人工解剖框。Clean為反覆使用的研究DEV，非盲測。正式標準overall MAE≤5.5、CS7≥75%，Child0–17/Adult18–54/Elderly55–80各CS7≥65%且absbias≤5、75–80absbias≤6；本版Clean與五GT完整門檻仍未全過。

## RK3588實測速度與比較

2026-11-05 OrangePi5Ultra/RK3588、runtime2.3.2、driver0.9.6、單core0、batch1，同12筆原TRAIN/degraded工程輸入，50warmup/500timed，未鎖頻。

|模型|mean ms|P50 ms|P95 ms|
|---|---:|---:|---:|
|ConvNeXt-Tiny epoch5|56.78|59.78|62.25|
|V118 DINOv2-Small|74.76|75.00|81.55|

V118此測量的平均延遲約為Tiny1.32倍；包含RKNN呼叫/layout轉換/outputreturn，不含YuNet/crop/相機/年齡解碼，非整機FPS。12工程輸入對ONNX的年齡平均/最大差0.166667/1歲；只是實測差值，沒有事後放寬gate宣稱正式hardwareparity PASS，也不是完整測試集的NPU MAE/CS7。RK3566/3576只有conversion，physicalruntime仍待驗；RK3588完整準確度亦待驗。

## 下載完整性

模型由Git LFS管理：

```bash
git lfs pull
sha256sum -c SHA256SUMS.txt
```

原研究來源：`age_v118_entropy_kd_qualification_20261031`，模型檔僅複製、未重新訓練或轉換。研究證據不等於商用資料/模型清權；production replacement不在本次發布範圍。
