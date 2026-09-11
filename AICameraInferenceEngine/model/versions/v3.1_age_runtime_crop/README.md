# Age v3.1 - Runtime-faithful crop

## Objective

Test one hypothesis: whether matching the VAC runtime Age crop semantics during
training and validation makes internal improvements transfer to RK3588 E2E.

The architecture remains SSR-Net with BGR uint8-like `0..255` input and ONNX
contract `[1,3,64,64] -> [1,1]`. Gender, YuNet weights, Head Pose, benchmark
thresholds, and the frozen benchmark manifest are unchanged.

## Crop contract

Runtime-faithful samples use the existing VAC implementations directly:

1. YuNet bbox on the input image
2. `inference.utils.img_utils.crop(..., margin=0.45)`
3. `inference.utils.img_utils.resize(..., (64, 64))`
4. OpenCV BGR, values `0..255`, no normalization

The builder uses the existing ONNX YuNet class, whose preprocessing and
postprocessing mirror the RKNN class. Validation is built from 80px and 40px
canvas images using the same placement/JPEG rules as the frozen benchmark.

## Controlled matrix

| Experiment | Training change | Common validation |
|---|---|---|
| B0 | Historical v2 training preprocessing and uniform-decade sampling | Runtime-faithful 80/40 crops |
| B1 | B0 plus clean 160px runtime-faithful YuNet crops | Runtime-faithful 80/40 crops |
| B2 | B1 plus mild bbox/crop jitter | Runtime-faithful 80/40 crops |
| B3 | B1 plus 80/40 distance training crops | Runtime-faithful 80/40 crops |
| B4 | B1 with natural sampling instead of uniform-decade sampling | Runtime-faithful 80/40 crops |
| B5 | Only individually successful B1-B4 components | Runtime-faithful 80/40 crops |

B4 intentionally turns balancing off: v2 already had balancing enabled, so
this is the controlled ON/OFF comparison requested after v3 showed a strong
age-bucket redistribution.

The clean B1 source is a 160px face on the same 640x480 synthetic camera canvas.
Running YuNet directly on tight UTKFace chips detected only 400/1,300 validation
sources and would introduce detector-selection bias. B3 alone adds the 80px and
40px distance variants to the 160px baseline.

No B experiment may be selected by overall metrics alone. Validation reports
MAE, +/-5, signed bias, and all eight age buckets. Full E2E is run only after
screening and full training select B5 without benchmark-label feedback.

## Status

- Runtime crop cache/manifests: complete (train 16,215 successful crops,
  validation 3,438, internal test 3,481 across 160/80/40px variants)
- B0-B4 screening: complete; B1, B3, and B4 improved validation, while B2
  jitter was rejected
- B5: complete; combines runtime crops, 40/35/25 distance weighting, and
  natural age sampling. Best common validation is 52.66% +/-5 and 7.225 MAE
- ONNX verification: PASS (`[1,3,64,64] -> [1,1]`, max TF/ONNX difference
  `9.1553e-05`)
- RK3566 and RK3588 FP16 conversion: complete
- RK3588 frozen 9,198-row E2E: complete with zero runtime errors. Overall
  conditional +/-5 is 37.78% and MAE is 10.205; 80+40px is 37.76% and 10.172
- Detailed results are in `training/analysis/ablation_screening.zh-TW.md` and
  `validation/age_v1_v2_v3_v31_rk3588_benchmark.zh-TW.md`
