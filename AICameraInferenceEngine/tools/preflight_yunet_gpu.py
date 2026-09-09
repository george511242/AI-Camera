#!/usr/bin/env python3
"""One-batch CUDA forward/backward check for the pinned YuNet trainer."""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--img-prefix", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.trainer_root.resolve()))
    from yunet_train.engine import load_model_weights_only
    from yunet_train.tasks.face import (
        WIDERFaceDataset,
        YuNetCriterion,
        build_train_transforms,
        build_yunet,
        collate_face_samples,
        move_batch_to_device,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible. Run this from the normal VS Code WSL terminal.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cv2.setNumThreads(0)

    dataset = WIDERFaceDataset(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        transform=build_train_transforms(
            image_size=640,
            crop_choice=(0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 2.5),
            min_box_size=10.0,
            small_face_policy=True,
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_face_samples,
        pin_memory=True,
    )
    device = torch.device("cuda")
    model = build_yunet("yunet_n").to(device)
    load_model_weights_only(args.weights, model=model, map_location="cpu")
    criterion = YuNetCriterion(strides=(8, 16, 32))
    torch.cuda.reset_peak_memory_stats(device)

    batch = move_batch_to_device(next(iter(loader)), device)
    outputs = model(batch.images)
    losses = criterion(outputs, boxes=batch.boxes, labels=batch.labels, keypoints=batch.keypoints)
    total = sum(losses.values())
    total.backward()
    backbone_grad = max(
        float(parameter.grad.norm().item())
        for parameter in model.backbone.parameters()
        if parameter.grad is not None
    )
    head_grad = max(
        float(parameter.grad.norm().item())
        for parameter in model.bbox_head.parameters()
        if parameter.grad is not None
    )
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "dataset_samples": len(dataset),
        "batch_shape": list(batch.images.shape),
        "batch_device": str(batch.images.device),
        "losses": {key: float(value.detach().item()) for key, value in losses.items()},
        "total_loss": float(total.detach().item()),
        "max_backbone_gradient_norm": backbone_grad,
        "max_head_gradient_norm": head_grad,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
    if backbone_grad <= 0 or head_grad <= 0:
        raise RuntimeError(f"non-zero gradient check failed: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
