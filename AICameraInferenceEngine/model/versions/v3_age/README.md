# Age v3

Age v3 targets improved conditional age accuracy at 80 px and 40 px while
preserving the deployable SSR-Net 64x64 input/output contract.

This version does not modify or overwrite Age v1 or Age v2 artifacts. Gender,
YuNet, and Head Pose are outside its scope.

## Status

- Phase 1 pipeline inspection: complete
- Phase 2 leakage audit: v2 split failed exact-content isolation
- Phase 3 Age v2 error analysis: complete
- Age v3 SHA256-aware split: generated and verified with seed `20260905`
- A0-A5 8-epoch screening: complete
- A6 full training: complete; selected configuration is A1 sampling only

## Layout

- `training/splits/`: immutable v3 train, validation, test, benchmark, and audit metadata
- `training/analysis/`: data distribution and leakage reports
- `experiments/`: isolated A0-A6 training outputs
- `validation/`: ONNX/RKNN and frozen benchmark results
- `onnx/`: selected Age v3 ONNX model
- `rknn/rk3566/`: RK3566 conversion
- `rknn/rk3588/`: RK3588 conversion

The frozen external benchmark manifest remains unchanged. The v3 split builder
uses exact file SHA256 values to exclude benchmark aliases, deduplicate matching
labels, and quarantine identical content with conflicting labels.

Verified split counts are 6,075 train, 1,300 validation, 1,304 internal test,
and 936 external benchmark source images. Every pair has zero `source_id` and
zero exact-content SHA256 overlap. See `training/splits/metadata.json` for
manifest hashes and `training/splits/quarantine.json` for excluded records.

The screening comparison is in
`training/analysis/ablation_screening.zh-TW.md`. Only moderate sqrt
inverse-frequency sampling improved both validation +/-5 and MAE, so A6 does
not include the other four unsuccessful changes.

A6 completed 20 epochs on CPU. Selected validation is 29.92% within five years
with 9.104 MAE; internal test is 30.06% with 8.903 MAE.

ONNX export and RK3566/RK3588 FP16 conversion are complete. TensorFlow and ONNX
agreed within `3.624e-05` over 74 fixed samples. Conversion metadata and hashes
are recorded in `validation/age_v3_conversion.json`.

RK3588 hardware validation and the frozen 9,198-row benchmark are complete with
zero runtime errors. Overall v3 conditional +/-5 is 26.05% and MAE is 12.906,
compared with v2 at 28.06% and 11.967. Age v3 is therefore not approved to
replace v2. See `validation/age_v1_v2_v3_rk3588_benchmark.zh-TW.md`.
