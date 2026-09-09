#!/usr/bin/env python3
"""Run one Gender RKNN inference using RKNN Toolkit Lite2."""

import argparse

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--crop", required=True)
    args = parser.parse_args()

    face = cv2.imread(args.crop)
    if face is None:
        raise ValueError(f"Unable to read crop: {args.crop}")
    if face.shape != (64, 64, 3):
        raise ValueError(f"Expected crop shape (64, 64, 3), got {face.shape}")
    face = np.ascontiguousarray([face], dtype=np.uint8)

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

        outputs = rknn.inference(inputs=[face], data_format=["nhwc"])
        value = float(np.asarray(outputs).reshape(-1)[0])
        print(f"input_shape: {face.shape}")
        print(f"input_dtype: {face.dtype}")
        print(f"output_count: {len(outputs)}")
        print(f"gender_scalar: {value:.10f}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
