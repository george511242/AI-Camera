#!/usr/bin/env python3
"""Convert an ONNX model to a non-quantized RKNN model."""

import argparse
import sys
from pathlib import Path

from rknn.api import RKNN


def require_success(step: str, result: int) -> None:
    print(f"{step} result: {result}", flush=True)
    if result != 0:
        raise RuntimeError(f"{step} failed with return value {result}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("rk3566", "rk3588"), required=True)
    parser.add_argument("--mean", nargs=3, type=float, default=[0, 0, 0])
    parser.add_argument("--std", nargs=3, type=float, default=[1, 1, 1])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"Source ONNX model not found: {args.source}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=False)
    try:
        require_success(
            "config",
            rknn.config(
                target_platform=args.target,
                mean_values=[args.mean],
                std_values=[args.std],
            ),
        )
        require_success("load_onnx", rknn.load_onnx(model=str(args.source)))
        require_success("build", rknn.build(do_quantization=False))
        require_success("export_rknn", rknn.export_rknn(str(args.output)))
    finally:
        rknn.release()

    print(f"output: {args.output}", flush=True)
    print(f"size_bytes: {args.output.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"conversion error: {error}", file=sys.stderr, flush=True)
        raise
