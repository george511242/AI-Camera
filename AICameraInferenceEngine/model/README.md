# Model layout

Models are separated by provenance and version. Do not overwrite an existing
artifact; create a new version directory instead.

```text
model/
|-- original/
|   `-- rk3566/              Recovered binaries from the original project
`-- versions/
    |-- v1_baseline/
    |   |-- onnx/            Source/interchange models for this version
    |   |-- rknn/
    |   |   |-- rk3566/      RKNN-Toolkit2 builds targeting RK3566
    |   |   `-- rk3588/      RKNN-Toolkit2 builds targeting RK3588
    |   `-- validation/      Reproducible inputs and comparison results
    |-- v2_finetuned/        Current deployed Age/Gender generation
    |-- v3_age/              Age research generation; not deployed
    |-- v3.1_age_runtime_crop/ Runtime-faithful Age candidate
    |-- v3.2_age_ordinal/     Ordinal SSR-Net research; not deployed
    `-- v4_age_mobilenetv3/   MobileNetV3-Large 112x112 experiment
```

Each version must include a README recording source repository/commit,
checkpoint checksum, ONNX interface, RKNN Toolkit version, conversion settings,
target platforms, warnings, and validation results.

`original/` is archival. Runtime code should select a versioned model from
`versions/` after that model has passed validation for its target platform.
