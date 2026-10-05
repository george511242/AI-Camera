# Gender V3 final decision

Status: `BOUNDED EXPERIMENT COMPLETE / EDGEFACE-XXS VALIDATED DEPLOYMENT CANDIDATE / PRODUCTION PROMOTION NOT AUTHORIZED`

Date: 2026-10-06 Asia/Taipei. Production factory/defaults were not changed.

## Decision

- `ACCURACY_WINNER`: **EdgeFace-XXS**. Corrected external mean three-resolution balanced accuracy is 89.266%, versus 88.522% for FaceLiVTv2-XS, 77.921% for V2, and 69.464% for V1.
- `DEPLOYMENT_WINNER`: **EdgeFace-XXS**. It is the only finalist that passes the complete RK3588 physical parity gate, is 0.695 ms faster in mean latency, and is 2,943,872 bytes smaller than FaceLiVT.
- Female-recall target: **PASS** for both V3 finalists. EdgeFace worst external Female recall is 86.105%; FaceLiVT is 88.632%. EdgeFace protects Male recall better (worst 90.456% versus 87.636%).
- Production promotion: **NOT AUTHORIZED**. Factory, resolver, shipping default, and runtime default remain unchanged.

## Unified result

| Model | External BAcc 80 / 40 / 27 | Male recall 80 / 40 / 27 | Female recall 80 / 40 / 27 | Mean BAcc | RK3588 size | RK3588 mean / p50 / p95 | Physical parity | Final status |
|---|---|---|---|---:|---:|---|---|---|
| V1 SSR-Net | 73.059 / 69.607 / 65.726% | 93.275 / 94.794 / 95.662% | 52.842 / 44.421 / 35.789% | 69.464% | 508,913 B | 1.518 / 1.258 / 2.471 ms | not re-evaluated | production baseline unchanged |
| V2 SSR-Net | 77.585 / 79.257 / 76.922% | 93.275 / 92.408 / 91.106% | 61.895 / 66.105 / 62.737% | 77.921% | 508,913 B | 1.387 / 1.262 / 2.286 ms | not re-evaluated | prior candidate; not default |
| EdgeFace-XXS | 89.754 / 89.547 / 88.497% | 90.456 / 90.672 / 90.889% | 89.053 / 88.421 / 86.105% | 89.266% | 4,180,397 B | 7.255 / 6.893 / 9.138 ms | **PASS** | **winner; not promoted** |
| FaceLiVTv2-XS | 88.979 / 88.344 / 88.242% | 87.852 / 87.636 / 87.852% | 90.105 / 89.053 / 88.632% | 88.522% | 7,124,269 B | 7.950 / 7.665 / 9.851 ms | **FAIL** max delta | finalist not selected |

V3 accuracy columns are frozen host/ONNX corrected-crop results used by the winner rule. Full RK3588 labeled inference was also run: EdgeFace physical BAcc is 89.965 / 89.652 / 88.494%; FaceLiVT is 89.193 / 88.236 / 88.134%.

## RK3588 physical evidence

Both models executed on Orange Pi 5 Ultra RK3588, Ubuntu 22.04.4, driver 0.9.6, RKNN Lite 2.3.2, batch 1, NPU core 0. Latency excludes preprocessing and uses 50 warmups plus 500 timed runs.

- EdgeFace: mean/max probability delta 0.000993/0.049316 and classification agreement 99.751% across 2,808 samples. Physical parity PASS.
- FaceLiVT: mean/max delta 0.002095/0.060464 and agreement 99.858%. Maximum delta exceeds the frozen 0.05 ceiling; physical parity FAIL.
- Neither V3 candidate reaches the preferred mean <=5 ms and p95 <=7 ms target. This records the accuracy/latency Pareto cost; both still physically execute.
- Runtime emitted static-shape and NCHW-to-NHWC layout-conversion warnings. No load, initialization, or inference error occurred.

## Artifacts and remaining platform status

EdgeFace ONNX is 4,819,687 bytes; RK3566/RK3576/RK3588 files are 3,349,357 / 5,243,309 / 4,180,397 bytes. FaceLiVT ONNX is 9,568,907 bytes; RKNN files are 5,788,397 / 7,523,053 / 7,124,269 bytes. All SHA256 values match their durable records.

ONNX parity and all three conversions pass for both finalists. RK3566/RK3576 remain `CONVERSION PASS / PHYSICAL RUNTIME PENDING`; RK3588 evidence cannot substitute for those boards.

## YuNet + winner E2E

Using preserved physical RK3588 YuNet detections and frozen EdgeFace host ONNX, detection coverage at 80/40/27 px is 97.650 / 65.598 / 1.603%; Gender conditional accuracy is 83.151 / 85.505 / 86.667%; E2E correct-Gender rate is 81.197 / 56.090 / 1.389%. The 27 px collapse is overwhelmingly YuNet coverage, not Gender classification. This is not a second physical Gender RKNN parity claim.

Machine-readable details are in `CANDIDATE_COMPARISON.csv`, `evaluation/`, and `physical_rk3588/`.
