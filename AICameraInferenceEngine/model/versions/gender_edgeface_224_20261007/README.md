# EdgeFace-XXS Gender 224×224 experiment

This is a separately trained 224×224 candidate. It does not replace or modify
the selected 112×112 Gender V3 deployment candidate.

## Artifacts and contract

- [ONNX](edgeface_xxs_gender_224.onnx)
- [RK3566 FP16](rk3566/edgeface_xxs_gender_224_fp16.rknn)
- [RK3576 FP16](rk3576/edgeface_xxs_gender_224_fp16.rknn)
- [RK3588 FP16](rk3588/edgeface_xxs_gender_224_fp16.rknn)
- [SHA256 sums](SHA256SUMS.txt)
- Input: 224×224 RGB, float32 NCHW, raw 0–255; normalization is embedded.
- Output: one logit. A value greater than or equal to
  `-0.13580560684204102` (probability `0.4661006832`) is Male.

The run used the same official EdgeFace-XXS face-recognition pretrained source,
UTKFace splits, six gender/native-resolution cells, augmentations, seed, learning
rates, validation-only selection rule, and patience rule as Gender V3. It selected
epoch 8. This is one run, so run-to-run variance is unknown.

## Accuracy comparison

Balanced accuracy on the frozen corrected-crop benchmark:

| Native face height | 112 baseline | 224 candidate | Delta |
|---:|---:|---:|---:|
| 80 px | 89.754% | 90.277% | +0.523 pp |
| 40 px | 89.547% | 88.918% | -0.628 pp |
| 27 px | 88.497% | 87.346% | -1.152 pp |
| Mean | 89.266% | 88.681% | -0.586 pp |

The 224 model is therefore close, but it is not equally good overall and does
not supersede the 112 model. It improves the easiest 80 px condition but loses
accuracy at 40 px and 27 px. Full validation, test, external metrics, confusion
matrices, and the 112 reference are in `evaluation.json`.

## Engineering validation

- PyTorch-to-ONNX parity: PASS, max logit delta `3.33786e-06`, classification
  agreement 100% on fixed random vectors.
- RK3566/RK3576/RK3588 Toolkit2 2.3.2 FP16 conversion: PASS independently.
- Physical RK3566/RK3576/RK3588 runtime and RKNN output parity: PENDING.
- Production promotion: NOT authorized.

The first diagnostic proved that merely changing the old 112 model's input to
224 causes severe accuracy collapse. The published artifact is a genuinely
fine-tuned and freshly exported fixed-224 graph, not that metadata-only rewrite.

Commercial use and redistribution rights remain unclear; this is engineering
evidence, not rights clearance.
