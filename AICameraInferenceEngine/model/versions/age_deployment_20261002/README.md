# 年齡估測部署模型

## 推薦優先測試

1. **ConvNeXt-Tiny — Recommended / Primary**：目前封裝模型中準確度、大小與部署可行性的最佳實用平衡；ONNX 與 RK3588 parity 均通過。
2. **MobileAgeNet — Lightweight / Fast Recommended**：最小且目前 RK3588 最快，適合作為低延遲與輕量基準；ONNX 與 RK3588 parity 均通過。
3. **ConvNeXt-Small — Experimental**：可執行，但 RK3588 平均年齡輸出差約 1.999 年，部署 parity 尚未接受。

| Model | Role | Params | ONNX | RK3566 | RK3576 | RK3588 | RK3588 parity | Recommended |
|---|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXt-Tiny | 準確度／部署平衡 | 27.90M | ✅ | ✅ | ✅ | ✅ | ✅ PASS | ⭐ #1 |
| MobileAgeNet | 輕量／最快 | 3.22M | ✅ | ✅ | ✅ | ✅ | ✅ PASS | ⭐ #2 |
| ConvNeXt-Small | 高容量實驗 | 49.53M | ✅ | ✅ | ✅ | ✅ | ⚠️ FAIL／PENDING | Experimental |

RK3566、RK3576 欄位代表已產生獨立 FP16 artifact；兩平台實體板驗證仍待進行。只有 RK3588 已完成此包所述的實體 runtime 與 parity 測試。

## 直接下載

### 1. ConvNeXt-Tiny — Recommended / Primary

- [ONNX](convnext_tiny_age/model.onnx)
- [RK3566 FP16](convnext_tiny_age/rk3566_fp16.rknn)
- [RK3576 FP16](convnext_tiny_age/rk3576_fp16.rknn)
- [RK3588 FP16](convnext_tiny_age/rk3588_fp16.rknn)
- [metadata](convnext_tiny_age/metadata.json)
- [SHA256 sums](convnext_tiny_age/SHA256SUMS.txt)

約 27.90M parameters，準確度顯著優於 MobileAgeNet，且 PyTorch→ONNX 與 RK3588 parity 均通過。RK3588 model-only mean／p50／p95 為 55.82／59.07／60.56 ms；ONNX→RKNN 平均年齡差 0.081 年。

### 2. MobileAgeNet — Lightweight / Fast Recommended

- [ONNX](mobileagenet_age/model.onnx)
- [RK3566 FP16](mobileagenet_age/rk3566_fp16.rknn)
- [RK3576 FP16](mobileagenet_age/rk3576_fp16.rknn)
- [RK3588 FP16](mobileagenet_age/rk3588_fp16.rknn)
- [metadata](mobileagenet_age/metadata.json)
- [SHA256 sums](mobileagenet_age/SHA256SUMS.txt)

約 3.22M parameters，是三個候選中最小且 RK3588 最快者。PyTorch→ONNX 與 RK3588 parity 均通過。RK3588 model-only mean／p50／p95 為 12.67／11.86／16.68 ms；ONNX→RKNN 平均年齡差 0.131 年。

### ConvNeXt-Small — ⚠️ Experimental / Parity Not Yet Passed

- [ONNX](convnext_small_age/model.onnx)
- [RK3566 FP16](convnext_small_age/rk3566_fp16.rknn)
- [RK3576 FP16](convnext_small_age/rk3576_fp16.rknn)
- [RK3588 FP16](convnext_small_age/rk3588_fp16.rknn)
- [metadata](convnext_small_age/metadata.json)
- [SHA256 sums](convnext_small_age/SHA256SUMS.txt)

約 49.53M parameters。ONNX 與三平台 artifact 均存在，RK3588 runtime 可執行，model-only mean／p50／p95 為 76.77／75.78／83.33 ms；但 ONNX→RKNN 平均年齡差為 **1.999 年**。僅供實驗比較，不得視為 deployment-ready 或優先於前兩個模型。

## 該下載哪一個檔案？

| Hardware | Download |
|---|---|
| PC / ONNX Runtime | `model.onnx` |
| RK3566 | `rk3566_fp16.rknn` |
| RK3576 | `rk3576_fp16.rknn` |
| RK3588 | `rk3588_fp16.rknn` |

建議測試順序：ConvNeXt-Tiny → MobileAgeNet → ConvNeXt-Small（僅實驗比較）。大型 `.onnx`／`.rknn` 使用 Git LFS；clone 後請先確認已安裝 Git LFS 並執行 `git lfs pull`。

## 前處理與輸出

三個模型皆使用評估 pipeline 產生的**人臉裁切**，不接受整張場景影像；裁切後直接 resize 至 224×224。不要額外做人臉 alignment、margin 變更或外部 normalization，否則不再符合本包 parity 證據。

| Model | Input | Color / layout / dtype | Numeric range / normalization | Output / age decoding |
|---|---|---|---|---|
| ConvNeXt-Tiny | 224×224 face crop | RGB、NCHW、float32 | host 輸入 0–255；模型內含 `/255` 與 ImageNet mean/std | 100 ordinal logits；`age = sum(sigmoid(logits))` |
| MobileAgeNet | 224×224 face crop | RGB、NCHW、float32 | host 輸入 0–255；模型內含 `/255` 與 ImageNet mean/std | scalar；模型已含 `80 * sigmoid`，直接作為 0–80 歲年齡 |
| ConvNeXt-Small | 224×224 face crop | RGB、NCHW、float32 | host 輸入 0–255；模型內含 `/255` 與 ImageNet mean/std | 100 ordinal logits；`age = sum(sigmoid(logits))` |

## 轉換與相容性

- RKNN Toolkit2：2.3.2，FP16、非量化；每個 SoC 各自轉換，不可跨平台共用 RKNN。
- RK3588 實測：Orange Pi 5 Ultra、RKNN Runtime 2.3.2、driver 0.9.6、NPU core 0；50 次 warmup、500 次 model-only inference。
- RK3566／RK3576：artifact generated；physical board validation pending。不得由轉換成功推論實體 latency 或 runtime 已通過。
- 每個模型的 `conversion_rk*.json` 與 `rk3588_physical_benchmark.json` 保存完整轉換／實測證據；`metadata.json` 保存來源 checkpoint SHA256、介面及 ONNX parity。

## 完整性與權利

下載後在各模型目錄執行 `sha256sum -c SHA256SUMS.txt`。這些 artifact 僅供內部／研究工程驗證；工程可取得（`RESEARCH_ACCESS`）不代表模型或資料的商用授權已核准（`COMMERCIAL_RIGHTS` 仍未釐清）。
