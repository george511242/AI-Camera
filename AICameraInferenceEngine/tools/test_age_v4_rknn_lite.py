#!/usr/bin/env python3
"""Run Age v4 RK3588 sanity inference and latency measurement."""

import argparse
import statistics
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    bgr = cv2.imread(args.crop, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to read crop: {args.crop}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    value = np.ascontiguousarray(rgb[None], dtype=np.float32)

    rknn = RKNNLite(verbose=False)
    try:
        load_result = rknn.load_rknn(args.model)
        print(f"load_rknn: {load_result}")
        if load_result != 0:
            raise RuntimeError("load_rknn failed")
        runtime_result = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        print(f"init_runtime: {runtime_result}")
        if runtime_result != 0:
            raise RuntimeError("init_runtime failed")
        for _ in range(args.warmup):
            rknn.inference(inputs=[value], data_format=["nhwc"])
        durations = []
        output = None
        for _ in range(args.runs):
            started = time.perf_counter()
            output = rknn.inference(inputs=[value], data_format=["nhwc"])
            durations.append((time.perf_counter() - started) * 1000.0)
        durations.sort()
        print(f"input_shape: {value.shape}")
        print(f"input_dtype: {value.dtype}")
        print(f"age: {float(np.asarray(output[0]).reshape(-1)[0]):.10f}")
        print(f"runs: {args.runs}")
        print(f"latency_mean_ms: {statistics.fmean(durations):.4f}")
        print(f"latency_p50_ms: {statistics.median(durations):.4f}")
        print(f"latency_p95_ms: {durations[int(0.95 * (len(durations) - 1))]:.4f}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
