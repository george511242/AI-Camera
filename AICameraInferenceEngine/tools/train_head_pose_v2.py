#!/usr/bin/env python3
"""Fine-tune the lightweight head-pose model on fixed 300W-LP splits."""

import argparse
import importlib.util
import json
import os
import platform
import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_network(path):
    spec = importlib.util.spec_from_file_location("head_pose_network", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Network


def load_rows(path, root):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    usable = []
    for row in rows:
        angles = np.degrees([row["roll_rad"], row["yaw_rad"], row["pitch_rad"]])
        if np.all(np.abs(angles) <= 99.0):
            usable.append((root / row["file"], angles.astype(np.float32)))
    return usable, len(rows) - len(usable)


class PoseDataset(Dataset):
    def __init__(self, rows, training, seed):
        self.rows = rows
        self.training = training
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, angles = self.rows[index]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rng = random.Random(self.seed + index + random.randrange(1 << 20))
        if self.training:
            low_size = rng.randint(27, 96)
            image = cv2.resize(image, (low_size, low_size), interpolation=cv2.INTER_AREA)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            gain = rng.uniform(0.85, 1.15)
            bias = rng.uniform(-16.0, 16.0)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255)
        else:
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - MEAN) / STD
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        return tensor, torch.from_numpy(angles)


def angle_bins(angles, num_bins=66, maximum=99.0):
    width = 2.0 * maximum / num_bins
    return torch.clamp(((angles + maximum) / width).long(), 0, num_bins - 1)


def loss_for(logits, targets, bins):
    losses = []
    predictions = []
    for axis, output in enumerate(logits):
        expected = (torch.softmax(output, dim=1) * bins).sum(dim=1)
        losses.append(F.cross_entropy(output, angle_bins(targets[:, axis])))
        losses.append(0.001 * F.mse_loss(expected, targets[:, axis]))
        predictions.append(expected)
    return sum(losses), torch.stack(predictions, dim=1)


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but torch.cuda.is_available() is False"
        )
    return torch.device(requested)


def atomic_torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def checkpoint_state(model, optimizer, scheduler, epoch, history, best_metric, run_config):
    return {
        "format_version": 1,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "best_mean_axis_mae": best_metric,
        "run_config": run_config,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def restore_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Resume checkpoint {path} is not a complete Head Pose v2 training checkpoint"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    random.setstate(checkpoint["rng_state"]["python"])
    np.random.set_state(checkpoint["rng_state"]["numpy"])
    torch.set_rng_state(checkpoint["rng_state"]["torch"].cpu())
    if device.type == "cuda" and checkpoint["rng_state"].get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["rng_state"]["cuda"]]
        )
    return checkpoint


def evaluate(model, loader, bins, device):
    errors = []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            _, predictions = loss_for(model(images), targets, bins)
            errors.append(torch.abs(predictions - targets).cpu().numpy())
    errors = np.concatenate(errors)
    return {
        "count": int(len(errors)),
        "axis_mae_roll_yaw_pitch": errors.mean(axis=0).tolist(),
        "mean_axis_mae": float(errors.mean()),
        "all_axes_within_15": float(np.all(errors <= 15.0, axis=1).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run one forward/backward batch without updating weights, then exit",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_rows, train_excluded = load_rows(args.train_manifest, args.image_root)
    validation_rows, validation_excluded = load_rows(args.validation_manifest, args.image_root)
    train_loader = DataLoader(
        PoseDataset(train_rows, True, args.seed), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, persistent_workers=False,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        PoseDataset(validation_rows, False, args.seed), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, persistent_workers=False,
        pin_memory=device.type == "cuda",
    )

    model = load_network(args.network)(
        num_bins=66, M=99, cuda=device.type == "cuda", bin_train=True
    )
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.to(device)
    bins = model.bins.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": args.device,
        "selected_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "train_count": len(train_rows),
        "train_excluded_outside_99": train_excluded,
        "validation_count": len(validation_rows),
        "validation_excluded_outside_99": validation_excluded,
        "network_path": str(args.network.resolve()),
        "pretrained_weight_path": str(args.weights.resolve()),
        "output_directory": str(args.output_dir.resolve()),
        "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
    }
    atomic_json_save(run_config, args.output_dir / "run_config.json")
    print(json.dumps(run_config, indent=2), flush=True)

    history = []
    best_metric = float("inf")
    start_epoch = 1
    initial = None
    if args.resume:
        resumed = restore_checkpoint(args.resume, model, optimizer, scheduler, device)
        history = resumed["history"]
        best_metric = resumed["best_mean_axis_mae"]
        start_epoch = resumed["epoch"] + 1
        initial = resumed.get("run_config", {}).get("initial_validation")
        print(f"resumed_from={args.resume} next_epoch={start_epoch}", flush=True)

    preflight_rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    images, targets = next(iter(train_loader))
    images = images.to(device, non_blocking=device.type == "cuda")
    targets = targets.to(device, non_blocking=device.type == "cuda")
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    preflight_loss, _ = loss_for(model(images), targets, bins)
    preflight_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    print(f"preflight_forward_backward=ok loss={preflight_loss.item():.5f}", flush=True)
    del images, targets, preflight_loss
    random.setstate(preflight_rng["python"])
    np.random.set_state(preflight_rng["numpy"])
    torch.set_rng_state(preflight_rng["torch"])
    if preflight_rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(preflight_rng["cuda"])
    if args.smoke_test:
        print("smoke_test=complete model_state_unchanged=true", flush=True)
        return

    if initial is None:
        initial = evaluate(model, validation_loader, bins, device)
        best_metric = initial["mean_axis_mae"]
        run_config["initial_validation"] = initial
        atomic_json_save(run_config, args.output_dir / "run_config.json")
    best_path = args.output_dir / "head_pose_v2_best.pkl"
    print(f"initial_validation={json.dumps(initial)}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (images, targets) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for(model(images), targets, bins)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 200 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={total_loss / step:.5f}", flush=True)
        metrics = evaluate(model, validation_loader, bins, device)
        train_loss = total_loss / len(train_loader)
        scheduler.step(metrics["mean_axis_mae"])
        row = {"epoch": epoch, "train_loss": train_loss, "validation": metrics,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        atomic_torch_save(model.state_dict(), args.output_dir / "head_pose_v2_latest.pkl")
        if metrics["mean_axis_mae"] < best_metric:
            best_metric = metrics["mean_axis_mae"]
            atomic_torch_save(model.state_dict(), best_path)
        atomic_json_save(history, args.output_dir / "history.json")
        atomic_torch_save(
            checkpoint_state(
                model, optimizer, scheduler, epoch, history, best_metric, run_config
            ),
            args.output_dir / "head_pose_v2_resume.pt",
        )
        print(json.dumps(row), flush=True)

    summary = {
        "seed": args.seed, "epochs": args.epochs,
        "train_count": len(train_rows), "train_excluded_outside_99": train_excluded,
        "validation_count": len(validation_rows),
        "validation_excluded_outside_99": validation_excluded,
        "initial_validation": initial, "best_mean_axis_mae": best_metric,
        "best_weights": str(best_path),
    }
    atomic_json_save(summary, args.output_dir / "training_run.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
