#!/usr/bin/env python3
from pathlib import Path
import sys

from rknn.api import RKNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = PROJECT_ROOT / "model/versions/v1_baseline/onnx/yunet_n_640_640.onnx"
OUTPUT_MODEL = PROJECT_ROOT / "model/versions/v1_baseline/rknn/rk3588/yunet_n_640_640_fp16.rknn"


def require_success(step: str, result: int) -> None:
    print(f"{step} result: {result}", flush=True)
    if result != 0:
        raise RuntimeError(f"{step} failed with return value {result}")


def main() -> int:
    if not SOURCE_MODEL.is_file():
        raise FileNotFoundError(f"Source ONNX model not found: {SOURCE_MODEL}")
    if OUTPUT_MODEL.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_MODEL}")
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=True)
    try:
        config_result = rknn.config(
            target_platform="rk3588",
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
        )
        require_success("config", config_result)

        load_result = rknn.load_onnx(model=str(SOURCE_MODEL))
        require_success("load_onnx", load_result)

        build_result = rknn.build(do_quantization=False)
        require_success("build", build_result)

        export_result = rknn.export_rknn(str(OUTPUT_MODEL))
        require_success("export_rknn", export_result)
    finally:
        rknn.release()

    print(f"output: {OUTPUT_MODEL}", flush=True)
    print(f"size_bytes: {OUTPUT_MODEL.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"conversion error: {error}", file=sys.stderr, flush=True)
        raise
