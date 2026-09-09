# v2 Age/Gender fine-tune

Status: first Gender and Age fine-tunes, ONNX export, frozen external GT-crop
benchmarks, and RK3566/RK3588 FP16 conversions completed. Device validation of
both v2 models remains pending.

## Compatibility contract

- Input: float32 NCHW `[1, 3, 64, 64]` ONNX interface
- Runtime pixels: OpenCV BGR uint8 values represented as float32, no scaling
- Gender output: scalar `[1, 1]`, `raw >= 0.5` means male
- Base architecture: official SSR-Net `SSR_net_general(64, [3, 3, 3], 1, 1)`
- Base weights: v1 official Wiki Gender checkpoint
- Deployment targets: RK3566 and RK3588, converted separately

## Split policy

The frozen 936-source v1 distance benchmark is excluded from training,
validation, and internal test. Remaining UTKFace `crop_part1` images are split
with seed `20260905`, stratified by gender and age decade.

UTKFace filenames do not include a reliable person identity. Source-image
isolation is guaranteed, but identity-level isolation cannot be guaranteed.
Split manifests store filenames relative to the UTKFace `crop_part1` directory;
training receives that directory explicitly through `--image-root`.

No RKNN model in this directory should be selected by the inference factory
until Orange Pi device validation is complete.

## First Gender training run

- Seed: `20260905`
- Train / validation / internal test: `6188 / 1325 / 1329`
- Frozen external benchmark: `936` source images
- Initial v1 validation balanced accuracy: `72.41%`
- Selected v2 validation balanced accuracy: `79.67%`
- Selected v2 internal-test balanced accuracy: `79.26%`
- Internal-test male recall: `80.78%`
- Internal-test female recall: `77.73%`
- Best epoch: 13; early stopped after epoch 18
- Selected weights SHA-256:
  `692b724cf0132c93fb07d01c6879293f7a67d924372f311a668b5155e84f9cff`

The internal test result is a training-stage gate, not the final v2 result.

## ONNX and frozen external benchmark

- ONNX: `onnx/gender_ssrnet_v2_64x64.onnx`
- ONNX SHA-256:
  `992c554be25fd647bbe4732c326fceae90dcb7ce7932dc9f1df9301db2faca20`
- ONNX checker: passed; input `[1, 3, 64, 64]`, output `[1, 1]`
- Frozen benchmark: 936 source images at each of `80/40/27px`
- v2 balanced accuracy: `77.6% / 79.3% / 76.9%`
- v1 balanced accuracy: `73.1% / 69.6% / 65.7%`
- Improvement over v1: `+4.5 / +9.7 / +11.2 pp`
- v2 male recall: `93.3% / 92.4% / 91.1%`
- v2 female recall: `61.9% / 66.1% / 62.7%`
- Raw results: `validation/gender_v2_onnx_gt_crop.jsonl`

The gain therefore reproduces on the frozen external benchmark and is not
limited to the internal test split.

## RKNN FP16 artifacts

Both models were converted with RKNN-Toolkit2 `2.3.2`,
`do_quantization=False`, and target-specific `target_platform` values.
For both conversions, `config`, `load_onnx`, `build`, and `export_rknn`
returned `0`.

- RK3566: `rknn/rk3566/gender_ssrnet_v2_64x64_fp16.rknn`
  - Size: 388,337 bytes
  - SHA-256:
    `1f8a6b2586015b490099ac2a99299f6c18714ef5853c999f6f0baf8a5c881ba8`
- RK3588: `rknn/rk3588/gender_ssrnet_v2_64x64_fp16.rknn`
  - Size: 508,913 bytes
  - SHA-256:
    `afbcc6374c31d93bbedfd859050d3695bf8c034926ac5122ca3a6adc1969d70f`

Same-crop simulator comparison (`validation/gender_backend_comparison.json`):

- ONNX: `0.9999602437`
- RK3566 target simulator: `1.0`
- RK3588 target simulator: `1.0`
- Each simulator absolute error versus ONNX: `3.9756e-05`
- RK3566/RK3588 simulator difference: `0.0`

RKNN-Toolkit2 emitted its deprecated `pkg_resources` warning and seven
`Unkown op target: 0` messages for each target. Conversion still completed
with all API return values equal to zero, and both simulator outputs were
finite and numerically close to ONNX. Physical-device validation is still
required before deployment selection.

## First Age training run

- Seed: `20260905`; decade-balanced sampling with the same small-face pipeline
- Selected by minimum validation MAE at epoch 17
- Validation: MAE `12.0771`, within 5 years `25.43%` (N=1,325)
- Internal test: MAE `12.1044`, within 5 years `26.03%` (N=1,329)
- Selected weights SHA-256:
  `2c68bda888f3a19246e9a4b38518113ed039f9ef3c9e8276a3cf8f33999b855f`

Frozen external v1 to v2 results at `80/40/27px`:

- MAE: `14.28/16.37/19.47` to `10.86/11.17/13.57`
- Within 5 years: `22.5/20.8/17.0%` to `31.1/30.2/23.5%`

Age v2 artifact SHA-256 values:

- ONNX: `6e2f15c6d8a9017e38398dd7f314c9f6d45d42b7acb40ef212d2a66d6bc391e2`
- RK3566 FP16: `3c24618d308a69284e1abefde509cafbc2617bfe62077e30739fabe25a408cca`
- RK3588 FP16: `0682bd8cca058470306cd6734a4b6e604a881c0efc5e67ac8c8f5be83089c807`
- Same-crop ONNX/RK3566 simulator/RK3588 simulator:
  `39.07916/39.09375/39.09375`

Training completed all 20 epochs and saved the selected weights, then failed
only while serializing NumPy float32 history values. The serializer was fixed;
validation was recovered directly from the saved weights without retraining.
