#!/usr/bin/env python3
"""Run Age and Head Pose RKNN models with RKNN Toolkit Lite2."""

import argparse

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def infer(model_path, image):
    rknn = RKNNLite(verbose=False)
    try:
        load_result = rknn.load_rknn(model_path)
        runtime_result = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if load_result != 0 or runtime_result != 0:
            raise RuntimeError(
                f"RKNN setup failed: load={load_result}, runtime={runtime_result}"
            )
        outputs = rknn.inference(inputs=[image], data_format=["nhwc"])
        return load_result, runtime_result, [
            float(np.asarray(output).reshape(-1)[0]) for output in outputs
        ]
    finally:
        rknn.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--age-model", required=True)
    parser.add_argument("--pose-model", required=True)
    parser.add_argument("--age-crop", required=True)
    parser.add_argument("--pose-crop", required=True)
    args = parser.parse_args()

    age = cv2.imread(args.age_crop)
    pose = cv2.imread(args.pose_crop)
    if age is None or age.shape != (64, 64, 3):
        raise ValueError(f"Invalid Age crop shape: {None if age is None else age.shape}")
    if pose is None or pose.shape != (224, 224, 3):
        raise ValueError(f"Invalid Pose crop shape: {None if pose is None else pose.shape}")

    age_input = np.ascontiguousarray([age], dtype=np.uint8)
    pose = cv2.cvtColor(pose, cv2.COLOR_BGR2RGB)
    pose_input = np.ascontiguousarray([pose], dtype=np.float32)

    age_load, age_runtime, age_output = infer(args.age_model, age_input)
    pose_load, pose_runtime, pose_output = infer(args.pose_model, pose_input)

    print(f"age_load_rknn: {age_load}")
    print(f"age_init_runtime: {age_runtime}")
    print(f"age: {age_output[0]:.10f}")
    print(f"pose_load_rknn: {pose_load}")
    print(f"pose_init_runtime: {pose_runtime}")
    print("pose_order: roll,yaw,pitch")
    print("pose: " + ",".join(f"{value:.10f}" for value in pose_output))


if __name__ == "__main__":
    main()
