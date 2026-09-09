# Age v3.2 - Ordinal fast track

This version tests one structural change: replace SSR-Net scalar MAE regression
with a rank-consistent ordinal age objective. It preserves the v3.1
runtime-faithful crop, 64x64 BGR 0..255 input, natural age sampling, 40/35/25
distance weighting, split, seed, and feature extractor initialization.

The supported training range is 1 through 99, represented by 98 thresholds
`P(age > 1)` through `P(age > 98)`. Deployment output remains a scalar:
`1 + sum(threshold probabilities)`.

Only O1 (ordinal BCE) is authorized initially. O2 and sampling sweeps are not
part of this run. Frozen benchmark data must not be used for model selection.

## O1 result

O1 completed one head-only warm-up epoch and 20 unfrozen epochs on CPU in
295.9 seconds. The selected epoch-20 checkpoint has SHA256
`582012caa48ddd33f22d1cafea1c88a2951abaca8519f9e8ebc29d397cf64436`.

Runtime-faithful 80px + 40px validation:

| Scope | +/-5 | MAE | Signed error |
|---|---:|---:|---:|
| Combined | 55.41% | 6.311 | -0.796 |
| 80px | 57.31% | 6.116 | -0.830 |
| 40px | 52.60% | 6.600 | -0.747 |

Relative to v3.1 validation (52.66%, 7.225 MAE), O1 improves +/-5 by 2.75 pp
and MAE by 0.914 years. It retains the young-age gain and improves 20-29 to
55.24% / 5.546 MAE. It does not solve the oldest-age problem: 70-79 is 26.23%
/ 12.003 MAE, while the 50 validation crops aged 80+ have 0% / 18.459 MAE.

All 98 probabilities are rank-consistent by construction. Validation sample,
threshold-pair, and maximum monotonic violation are all zero.

## O2 and model selection

O2 was the single authorized auxiliary-loss run. Four initialization batches
had mean BCE/MAE losses 0.3798/17.896 and gradient norms 0.2856/15.0835.
`lambda=0.0028` made the auxiliary gradient 14.8% of the ordinal gradient.

O2 reached 55.88% / 6.279 MAE, but did not improve the high-age guardrail:
70-79 MAE worsened from 12.003 to 12.290 and 80+ MAE from 18.459 to 18.647,
with 0% +/-5 in both. O1 was therefore selected using validation only.

## Export and frozen RK3588 result

Selected O1 ONNX: `onnx/age_ordinal_v32_64x64.onnx`, 263,936 bytes, SHA256
`e52e14f1b8e090b9e7f36161830b54b9af3fe0dded10b40db7f0656f5614de76`.
Contract is `[1,3,64,64] -> [1,1]`; checker passed and 73 samples had maximum
framework/ONNX difference `5.14984e-05`.

RK3588 FP16: `rknn/rk3588/age_ordinal_v32_64x64_fp16.rknn`, 1,373,489 bytes,
SHA256 `e93a8fd28cb9af136a1a856fc588a158ad5d845931bde21e4a50716404902413`.
Config/load/build/export returned zero. Orange Pi sanity predicted 2.082 for a
GT-age-2 crop with no runtime error.

The one frozen 9,198-row benchmark completed in 384 seconds with zero runtime
errors. At 80+40px v3.2 has 37.70% conditional +/-5 and 9.790 MAE, versus v3.1
37.76% and 10.172. It improves MAE but not the primary +/-5 metric. It improves
20-29 to 49.43% and 60-69 to 41.61%, but 80+ falls to 3.17% / 20.170 MAE.
Do not promote v3.2 over v3.1 or the current production v2 without a new user
decision. No further training is authorized.

## Final calibration experiment

One source-grouped 5-fold cross-fitted PAVA isotonic calibration was evaluated
on the 2,144 validation 80px/40px runtime crops (1,279 sources). All fold-level
fit/evaluation source overlaps were zero. It reduced overall +/-5 from 55.41%
to 54.48% and worsened MAE from 6.311 to 6.378 years. Although 70-79 and 80+
improved, the predeclared overall-benefit hard stop failed.

No calibrated frozen RK3588 benchmark was run and calibration remains disabled.
See `validation/age_v32_isotonic_calibration.zh-TW.md`. Do not run another
calibration grid; a future large-gain attempt should evaluate a stronger
RKNN-compatible age model family instead of further SSR-Net tuning.
