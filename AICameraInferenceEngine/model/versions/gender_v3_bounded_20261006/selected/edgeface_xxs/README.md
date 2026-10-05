# Gender V3 EdgeFace-XXS

**EdgeFace-XXS 是目前 Gender 最佳模型**，同時為 bounded V3 tournament 的 `ACCURACY_WINNER` 與 `DEPLOYMENT_WINNER`。

## 下載

- [ONNX](edgeface_xxs_gender.onnx)
- [RK3566 FP16](rk3566/edgeface_xxs_gender_fp16.rknn)
- [RK3576 FP16](rk3576/edgeface_xxs_gender_fp16.rknn)
- [RK3588 FP16](rk3588/edgeface_xxs_gender_fp16.rknn)
- [SHA256 sums](SHA256SUMS.txt)

| Artifact | Status | Physical runtime |
|---|---|---|
| ONNX | parity PASS | ONNX Runtime validated |
| RK3566 FP16 | conversion PASS | PENDING |
| RK3576 FP16 | conversion PASS | PENDING |
| RK3588 FP16 | conversion PASS | PASS |

RK3588 在 Orange Pi 5 Ultra、RKNPU driver 0.9.6、RKNN Lite 2.3.2、NPU core 0 實測。完整 2,808-sample parity 的 probability mean/max delta 為 0.000993／0.049316，threshold agreement 99.751%；50 warmups 加 500 次 model-only inference 的 mean／p50／p95 latency 為 7.255／6.893／9.138 ms。

## 輸入與輸出

- Input：人臉 crop，RGB、float32、NCHW、raw 0–255；normalization 已嵌入模型。
- ONNX input size：112×112。
- Output：單一 gender logit。
- Decision threshold：`0.38631075620651245`；大於等於 threshold 判為 Male，否則為 Female。
- 不可任意變更 crop、color order、normalization 或 threshold，否則不再符合現有 parity 與 accuracy 證據。

## 完整性與限制

下載後在本目錄執行 `sha256sum -c SHA256SUMS.txt`。RK3566 與 RK3576 只有獨立轉換成功，不能宣稱實機 runtime 已通過。此模型列為最佳工程候選，不代表已變更 production runtime 預設，也不代表資料或模型的商用／再散布權利已核准。

完整 V1／V2／V3 比較與決策見 [FINAL_DECISION.md](../../FINAL_DECISION.md)。
