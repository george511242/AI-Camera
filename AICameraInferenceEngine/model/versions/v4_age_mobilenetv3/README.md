# Age v4 - MobileNetV3-Large 112x112

Age v4 tests exactly one capacity hypothesis: replace the SSR-Net 64x64
feature extractor with ImageNet-pretrained MobileNetV3-Large at 112x112 while
retaining the leakage-safe split, runtime-faithful crop semantics, natural age
sampling, 40/35/25 distance distribution, and O1 ordinal BCE objective.

No training may start until the complete untrained deployment graph converts
successfully for both RK3566 and RK3588. Preflight artifacts are isolated under
`preflight/` and are never deployment candidates.

External preflight contract is NCHW RGB float32 0..255, `[1,3,112,112]`. Keras
MobileNetV3 built-in preprocessing maps pixels to -1..1. Output is scalar age
`[1,1]`, computed as `1 + sum(sigmoid(98 ordinal logits))`.

## Untrained deployment preflight

- TensorFlow 2.15.1 / tf2onnx 1.16.1 / ONNX opset 13
- Parameters: 3,132,002 total; 2,996,352 in the backbone
- ONNX: 12,460,937 bytes, checker PASS, `[1,3,112,112] -> [1,1]`
- RK3566 FP16: 6,790,387 bytes, config/load/build/export all returned zero
- RK3588 FP16: 7,345,779 bytes, config/load/build/export all returned zero
- No conversion reported an unsupported operator or build failure

Warnings retained for audit: tf2onnx could not apply its optional transpose
optimizer but emitted a checker-valid, ONNX Runtime-valid model; Toolkit2
reported the existing `pkg_resources` deprecation and `Unkown op target: 0`
messages while both builds completed successfully. Conversion success does not
prove RK3566 physical-device latency or full-NPU execution.

Official Keras ImageNet no-top weights were obtained through
`keras.applications.MobileNetV3Large(weights="imagenet")`. Keras uses the
`large_224_1.0_float` convolution weights with the 112x112 static input.
The immutable initialization copy is stored under `pretrained/`; training uses
that explicit path rather than relying on a per-user Keras cache.

## Training preflight

- Official initialization SHA256:
  `88252c55061fd4434ccc4c37fd7bb71c8832e9453190e4d8326a7adf58577411`
- Batch-64 forward/backward PASS; head gradient 0.9046 and backbone gradient
  0.1759 were non-zero.
- TensorFlow created the RTX 5070 Ti for the batch smoke, but the training-flow
  smoke then exited with signal 139. TensorFlow 2.15 warns that it has no CUDA
  kernels compiled for compute capability 12.0. GPU is therefore rejected as
  unsafe for this run; do not retry it during the same experiment.
- CPU training-flow smoke passed: head warm-up about 30 seconds and one
  unfrozen epoch 101.6 seconds. The smoke checkpoint reached 50.93% +/-5 and
  7.075 MAE; it is not a model-selection candidate.

Status: dual-platform conversion and all CPU smoke tests PASS. The single full
CPU run is authorized with batch 64, workers 4, seed 20260905, one warm-up
epoch, and at most 20 unfrozen epochs (estimated upper bound about 35 minutes).

## Training and validation result

The single run stopped early after 14/20 unfrozen epochs. Epoch 9 was selected
by validation +/-5. It reached 59.61% / 6.048 MAE on the same 2,144 validation
80px/40px crops, versus v3.2 55.41% / 6.311: +4.20 pp and -0.263 years.
The primary gate passed, while the preferred 0.7-year MAE gain did not.

Young buckets and 70-79/80+ improved, but 40-49, 50-59 MAE, and especially
60-69 regressed. See `validation/age_v32_v4_validation.zh-TW.md`.

Trained ONNX is 12,469,046 bytes with contract `[1,3,112,112] -> [1,1]`.
ONNX checker passed; 73-sample framework/ONNX max and mean differences were
0.000206 and 0.0000355 years. Final FP16 conversion passed for both targets:
RK3566 7,009,779 bytes and RK3588 7,311,411 bytes.

## RK3588 hardware and frozen benchmark

Orange Pi sanity passed on RKNN Runtime 2.3.2 / driver 0.9.6:
`load_rknn=0`, `init_runtime=0`, and a GT-age-10 validation crop predicted
11.352. Fifty Age-only calls measured mean 4.668 ms, p50 4.811 ms, and p95
6.057 ms on NPU core 0. The static-shape dynamic-range query warning is benign.

The single frozen 9,198-row benchmark completed in 393 seconds with zero
runtime errors. Sample keys and YuNet bbox/score/IoU/failure fields match all
earlier versions exactly. Primary 80+40px result:

| Version | E2E +/-5 | Conditional +/-5 | Conditional MAE |
|---|---:|---:|---:|
| v3.1 | 30.82% | 37.76% | 10.172 |
| v3.2 | 30.77% | 37.70% | 9.790 |
| v4 | 33.39% | 40.90% | 8.842 |

V4 improves over v3.2 by 3.21 pp and 0.947 years. This is a moderate 40-42%
result, not the predefined >=42% meaningful-success tier. It improves 40px
strongly (41.69%, +6.03 pp) and reduces old-age MAE, but 60-69 +/-5 regresses
from 41.61% to 34.90%, and 80+ remains only 10.32%.

V4 is integrated as an explicit deployment candidate, but is not the default.
Production remains Age v2 unless `AGE_MODEL_VERSION=v4` is set. The runtime
factory selects the matching RK3566/RK3588 artifact from `RKNPU_PLATFORM` and
uses the v4-specific 112x112 RGB float32 input adapter. Unknown model versions
or platforms fail during model loading. RK3566 conversion passed, but physical
RK3566 execution and latency remain unverified.

Example RK3588 opt-in:

```bash
RKNPU_PLATFORM=rk3588 AGE_MODEL_VERSION=v4 python main.py
```

Full report: `validation/age_v1_v2_v3_v31_v32_v4_rk3588_benchmark.zh-TW.md`.
