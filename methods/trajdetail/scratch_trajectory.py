"""Fit each cohort's trajectory and local matching stages on its own patients."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.compare import cases
from src.report import case_manifest
from src.trajectory.control_external import _step, _student
from methods.trajdetail.matched_control import MatchedControl
from methods.trajdetail.train_external_curvature_prototype import Features, train_stage
from src.trajectory.validate_control import batch_item, cached_teacher, teacher_components, write_json
from src.trajectory.train_unified import _autocast, _input


def windows(manifest, cache, split):
    # Use native H,W,Z within the trajectory, as in the archived brain MRI branch.
    return [{"patient_id": patient, "context": volumes[:2].movedim(-3, -1).numpy(),
             "targets": volumes[2:].movedim(-3, -1).numpy(),
             "mask": np.ones(2, dtype=np.float32), "context_times": times[:2].numpy(),
             "target_times": times[2:].numpy()}
            for patient, volumes, times in cases(manifest, cache, split, torch.device("cpu"))]


@torch.inference_mode()
def validation_mse(model, valid, device):
    model.eval()
    model.capture_components = False
    errors = []
    for item in valid:
        batch = batch_item(item)
        with _autocast(device, True):
            forecast = model(*_input(batch, device))["predictions"]
        errors.append(float((forecast[:, -1].float() - batch["targets"][:, -1].to(device)).square().mean()))
    return float(np.mean(errors))


def fit_control(train, valid, args, device, *, windows_per_patient=1):
    if len(train) % windows_per_patient:
        raise ValueError("Training windows must be grouped by patient")
    path = args.output / "control.pt"
    model = _student("control", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.control_epochs)
    order = torch.Generator().manual_seed(42)
    augment = np.random.RandomState(42)
    teachers = {}
    for index, item in enumerate(train):
        teacher = cached_teacher(args.output, index, batch_item(item), args, allow_create=True)
        teachers[index] = {key: teacher[key] for key in ("transport_components", "source_components")}
        if (index + 1) % 10 == 0 or index + 1 == len(train):
            print(f"control teachers {index + 1}/{len(train)}", flush=True)
    best, history = float("inf"), []
    for epoch in range(1, args.control_epochs + 1):
        model.train()
        model.capture_components = True
        losses = []
        for patient in torch.randperm(len(train) // windows_per_patient, generator=order).tolist():
            index = patient * windows_per_patient + (epoch - 1) % windows_per_patient
            flips = tuple(bool(v) for v in augment.random_sample(2) < .5)
            batch = batch_item(train[index], flips)
            teacher = teacher_components(teachers[index], flips, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _step(model, batch, teacher, "control", device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"control: nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        row = {"epoch": epoch, "train_loss": float(torch.stack(losses).mean())}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.control_epochs:
            error = validation_mse(model, valid, device)
            row["validation_mse"] = error
            if error < best:
                best = error
                torch.save({"model": model.state_dict(), "epoch": epoch, "validation_mse": error}, path)
        history.append(row)
        write_json(args.output / "control_history.json", history)
        print("control", row, flush=True)
    checkpoint = torch.load(path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if not np.isclose(validation_mse(model, valid, device), best, rtol=1e-6, atol=1e-9):
        raise RuntimeError("Trajectory validation checkpoint does not replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--control-epochs", type=int, default=60)
    parser.add_argument("--teacher-steps", type=int, default=150)
    parser.add_argument("--match-epochs", type=int, default=8)
    parser.add_argument("--benefit-epochs", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    manifest, partitions = case_manifest(args.output)
    args.dataset = manifest["dataset"]
    write_json(args.output / "trajectory_config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "seed": 42, "trainable_initialization": "random", "encoder": "frozen DINOv3 ViT-B/16",
        "counts": {split: len(rows) for split, rows in partitions.items()},
    })
    train, valid = (windows(manifest, args.cache, split) for split in ("train", "validation"))
    fit_control(train, valid, args, device)
    model = MatchedControl(use_dino=True).to(device)
    checkpoint = torch.load(args.output / "control.pt", weights_only=True, map_location=device)
    model.load_control(checkpoint["model"])
    stage_args = copy.copy(args)
    stage_args.lr = .001
    # The common stage alternates two windows per patient. There is one eligible
    # window here, so both indices refer to that window and each epoch sees it once.
    paired_indices = [item for item in train for _ in range(2)]
    features = Features("lumiere", device)
    selected = {stage: train_stage(model, stage, paired_indices, valid, features, device, stage_args)
                for stage in ("matching", "benefit")}
    (args.output / "trajectory_selection.json").write_text(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
