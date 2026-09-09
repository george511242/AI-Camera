#!/usr/bin/env python3
"""Export the official Lightweight Head Pose checkpoint to ONNX."""

import argparse
import importlib.util
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("official_head_pose", args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.Network(num_bins=66, M=99, cuda=False, bin_train=False)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        str(args.output),
        input_names=["input"],
        output_names=["roll", "yaw", "pitch"],
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
