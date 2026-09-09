# v1 baseline models

## Gender SSR-Net

- Source: `shamangary/SSR-Net`
- Source commit: `f98b6cbe1c9c8c78649e5a331f94113564521525`
- Architecture: official `SSR_net_general(64, [3, 3, 3], 1, 1)`
- Checkpoint: official Wiki Gender
  `pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5`
- Checkpoint SHA256:
  `2d15bfa72869526d1e2b80385157e34a55fb5a4b184bbac5ea56d623141d7e8c`
- ONNX interface: `float32 [1,3,64,64] -> float32 [1,1]`, opset 13
- RKNN conversion: RKNN-Toolkit2 2.3.2, FP16, non-quantized,
  mean `[0,0,0]`, std `[1,1,1]`

The old VAC filename does not identify its training dataset. The official Wiki
checkpoint is the best-supported reconstruction, but its weights cannot be
proven byte-for-byte identical to the checkpoint used by the recovered RKNN.

The comparison in `validation/gender_backend_comparison.json` uses the same
lossless VAC-style BGR uint8 PNG crop for ONNX, RK3566 simulator, and RK3588
simulator. Orange Pi NPU results are recorded in
`validation/gender_rk3588_orangepi.json`.

## Age SSR-Net

- Source: `shamangary/SSR-Net`
- Source commit: `f98b6cbe1c9c8c78649e5a331f94113564521525`
- Architecture: official `SSR_net(64, [3, 3, 3], 1, 1)`
- Checkpoint: official Wiki Age
  `pre-trained/wiki/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5`
- Checkpoint SHA256:
  `f8b677f1a61fcf1233de0bd942f3172b1397fcbfe0f2784220154af1faf833e0`
- ONNX interface: `float32 [1,3,64,64] -> float32 [1,1]`, opset 13
- RKNN conversion: RKNN-Toolkit2 2.3.2, FP16, non-quantized,
  mean `[0,0,0]`, std `[1,1,1]`

## Lightweight Head Pose

- Source: `Shaw-git/Lightweight-Head-Pose-Estimation`
- Source commit: `aed1802a34cccdd7e35beb788517cc6d01319d5c`
- Architecture: official `Network(num_bins=66, M=99, bin_train=False)`
- Checkpoint: `models/model-b66.pkl`
- Checkpoint SHA256:
  `0398c6d13983bc9820dd60fcd00c2e8b655e7a78d4cd7572fa6b4a42b9064879`
- ONNX interface: `float32 [1,3,224,224]` to three scalar outputs in
  `roll, yaw, pitch` order, opset 13
- RKNN conversion: RKNN-Toolkit2 2.3.2, FP16, non-quantized,
  mean `[123.675,116.28,103.53]`, std `[58.395,57.12,57.375]`

Age and Pose simulator results are in
`validation/age_pose_backend_comparison.json`. RK3588 NPU results are in
`validation/age_pose_rk3588_orangepi.json`.

## YuNet

The YuNet ONNX is from `ShiqiYu/libfacedetection.train`. Both platform models
are FP16/non-quantized conversions made with RKNN-Toolkit2 2.3.2.
