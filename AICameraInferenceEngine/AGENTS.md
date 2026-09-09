# VAC AI Camera - Codex Instructions

## Goal

Port the existing VAC AI Camera inference engine from
Rockchip RK3566 to Orange Pi 5 Ultra RK3588.

Current priority:
Get YuNet face detection running on RK3588 NPU first.

Do not work on age/gender/head-pose until YuNet works.

## Hardware

Target:
- Orange Pi 5 Ultra
- RK3588
- ARM64
- Ubuntu 22.04
- RKNPU driver 0.9.6
- RKNN Runtime 2.3.2
- USB UVC camera

Orange Pi SSH:
orangepi@100.94.109.12

Development machine:
- Acer Predator
- WSL2 Ubuntu
- x86_64

RKNN conversion environment:

conda activate rknn-toolkit2

Python 3.12.

## Existing models

model/rknn/

- yunet_n_640_640.rknn
- age_model_1015-pa.rknn
- gender_model_0925-pa.rknn
- pose-0701-quantize-100.rknn

These models were recovered from old Git history.

The existing YuNet model was tested on RK3588.

RKNN reported:

target platform: rk3566

and failed with:

"This rknn model is for RK3566, but current platform is RK3588"

Therefore the old RK3566 RKNN model cannot be used directly.

## Current strategy

Find a compatible YuNet ONNX model.

Then:

YuNet ONNX
-> RKNN Toolkit2
-> target_platform='rk3588'
-> RK3588 .rknn
-> Orange Pi
-> NPU inference

## Important

Before converting any model:

1. Inspect inference/rknn/yunet.py
2. Determine expected input shape
3. Determine preprocessing
4. Determine expected output tensors
5. Determine postprocessing
6. Verify candidate ONNX model matches those expectations

Do not assume an arbitrary YuNet ONNX model is compatible.

Do not delete or overwrite the existing recovered RK3566 models.

Prefer minimal changes to the existing VAC code.