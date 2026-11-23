"""Train the history-calibrated trajectory model with real rollout supervision."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.trajectory.external_data import ACDCJointDataset, LumiereJointDataset, PROJECT_ROOT
from src.trajectory.external_eval import save_evaluation
from src.trajectory.train import seed_all


def trajectory_loss(result, batch, *, history_weight=0.25, l1_weight=0.01,
                    primary_weight=2.0):
    """Average targets inside each patient, then average patients equally."""
    prediction = result["predictions"]
    target = batch["targets"].to(prediction)
    if prediction.shape != target.shape:
        raise ValueError("future prediction/target shapes differ")
    difference = prediction - target
    future_mse = difference.square().flatten(2).mean(2)
    future_mae = difference.abs().flatten(2).mean(2)
    weights = prediction.new_ones(prediction.shape[1])
    weights[-1] = primary_weight
    if primary_weight <= 0:
        raise ValueError("primary_weight must be positive")
    future = ((future_mse + l1_weight * future_mae) * weights).sum(1) / weights.sum()

    historical_prediction = result["history_predictions"]
    historical_target = batch["context"].to(historical_prediction)
    valid = result["history_valid"].bool()
    if historical_prediction.shape != historical_target.shape or valid.shape != historical_target.shape[:2]:
        raise ValueError("historical prediction/target/mask shapes differ")
    historical_difference = historical_prediction - historical_target
    history_error = (historical_difference.square() + l1_weight * historical_difference.abs()).flatten(2).mean(2)
    history_error = torch.where(valid, history_error, torch.zeros_like(history_error))
    history = history_error.sum(1) / valid.sum(1).clamp_min(1)
    total = future.mean() + history_weight * history.mean()
    return total, {
        "future": future.mean().detach(), "history": history.mean().detach(),
        "primary_mse": future_mse[:, -1].mean().detach(),
        "primary_mae": future_mae[:, -1].mean().detach(),
        "history_terms": valid.sum().detach(),
    }


def _autocast(device, amp):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp and device.type == "cuda" else nullcontext()


def _input(batch, device):
    return tuple(batch[name].to(device, non_blocking=True) for name in (
        "context", "mask", "context_times", "target_times"))


def _dataset(args, split, *, limit=None):
    if args.dataset == "acdc":
        return ACDCJointDataset(args.data_root, split, protocol_root=args.protocol_root,
                                augment=args.augment and split == "train", limit=limit)
    return LumiereJointDataset(args.data_root, split, augment=args.augment and split == "train", limit=limit)


@torch.inference_mode()
def predict(model, dataset, device, *, amp=False):
    model.eval()
    predictions, targets, lasts, ids, counts, times = [], [], [], [], [], []
    historical_losses = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        with _autocast(device, amp):
            result = model(*_input(batch, device))
        predictions.append(result["predictions"].float().cpu().numpy())
        targets.append(batch["targets"].numpy())
        lasts.append(result["last_context"].float().cpu().numpy())
        ids.extend(batch["patient_id"])
        counts.extend(batch["mask"].sum(1).int().tolist())
        times.append(batch["target_times"].numpy())
        _, parts = trajectory_loss(result, batch, history_weight=0.0, l1_weight=0.0)
        historical_losses.append(float(parts["history"]))
    return {"predictions": np.concatenate(predictions), "targets": np.concatenate(targets),
            "last_context": np.concatenate(lasts), "patient_ids": ids,
            "history_counts": counts, "target_times": np.concatenate(times),
            "history_mse": float(np.mean(historical_losses))}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    from methods.trajdetail.unified_model import HistoryCalibratedTrajectory

    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    device = torch.device(args.device)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    training = _dataset(args, "train", limit=args.limit)
    validation = _dataset(args, "validation")
    model = HistoryCalibratedTrajectory(base_channels=args.base_channels,
        memory_channels=args.memory_channels, max_step=args.max_step).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))

    source = output / "source_snapshot"
    source.mkdir()
    source_hashes = {}
    for name in ("unified_model.py", "train_unified.py", "external_data.py", "external_eval.py", "train.py"):
        path = Path(__file__).resolve().parent / name
        shutil.copyfile(path, source / name)
        source_hashes[name] = _sha(path)
    manifest = (Path(args.protocol_root) / "data_manifest.json" if args.dataset == "acdc"
                else Path(args.data_root) / "patient_split_manifest.json")
    import torchmetrics
    config = {**vars(args), "parameters": sum(p.numel() for p in model.parameters()),
        "train_patients": len(training), "validation_patients": len(validation),
        "model": "HistoryCalibratedTrajectory V1", "source_sha256": source_hashes,
        "manifest_sha256": _sha(manifest), "prediction": "deterministic shared data-time rollout",
        "selection": "minimum fixed-validation latest-target mean MSE; no test selection",
        "future_targets": "same frozen q-1,q; q is primary; both excluded from observed history",
        "history_supervision": "causal prediction before correction; patient-equal; single-history contributes zero",
        "inference_matches_training": True,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                    "torchmetrics": torchmetrics.__version__, "cuda": torch.version.cuda}}
    if args.dataset == "acdc":
        mask_root = Path(args.protocol_root) / "runs" / "discrete_seed42"
        config["fixed_masks_sha256"] = {s: _sha(mask_root / f"fixed_masks_{s}.json") for s in ("validation", "test")}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"config": config}), flush=True)

    best, best_epoch, history = float("inf"), None, []
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        field_names = ("velocity_rms", "source_rms", "memory_derivative_rms")
        sums = {key: 0.0 for key in ("total", "future", "history", "primary_mse", "primary_mae", "history_terms", *field_names)}
        epoch_start = time.monotonic()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, args.amp):
                result = model(*_input(batch, device))
                loss, parts = trajectory_loss(result, batch, history_weight=args.history_weight,
                    l1_weight=args.l1_weight, primary_weight=args.primary_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            optimizer.step()
            sums["total"] += float(loss.detach())
            for key, value in parts.items():
                sums[key] += float(value)
            for key in field_names:
                sums[key] += float(result[key].detach())
        scheduler.step()
        record = {"epoch": epoch, "train": {key: value / len(loader) for key, value in sums.items()},
                  "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.monotonic() - epoch_start}
        if epoch == 1 or epoch % args.val_interval == 0 or epoch == args.epochs:
            values = predict(model, validation, device, amp=args.amp)
            mse = float(np.square(values["predictions"][:, -1] - values["targets"][:, -1]).mean())
            record["validation_primary_mse"] = mse
            record["validation_history_mse"] = values["history_mse"]
            if mse < best:
                best, best_epoch = mse, epoch
                torch.save({"model": model.state_dict(), "config": config,
                    "epoch": epoch, "validation_primary_mse": mse}, output / "best.pt")
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = _dataset(args, "test")
    values = predict(model, test, device, amp=args.amp)
    evaluation = save_evaluation(output / "test_eval", values["predictions"][:, -1],
        values["targets"][:, -1], values["last_context"], values["patient_ids"],
        ssim_data_range=None if args.dataset == "acdc" else 1.0,
        title=f"{args.dataset.upper()} history-calibrated trajectory V1")
    pair_dir = output / "future_pairs"
    pair_dir.mkdir()
    for i, patient in enumerate(values["patient_ids"]):
        np.savez_compressed(pair_dir / f"{patient}.npz", prediction=values["predictions"][i],
            target=values["targets"][i], target_times=values["target_times"][i],
            history_count=values["history_counts"][i])
    result = {"best_epoch": best_epoch, "best_validation_primary_mse": best,
        "test": evaluation, "elapsed_seconds": time.monotonic() - start}
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("acdc", "lumiere"), default="acdc")
    p.add_argument("--data-root", default=None)
    p.add_argument("--protocol-root", default=str(PROJECT_ROOT / "cronos_acdc_baseline"))
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--memory-channels", type=int, default=8)
    p.add_argument("--max-step", type=float, default=1/11)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--history-weight", type=float, default=0.25)
    p.add_argument("--l1-weight", type=float, default=0.01)
    p.add_argument("--primary-weight", type=float, default=2.0)
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)
    if args.data_root is None:
        args.data_root = "/mnt/Datasets/ACDC" if args.dataset == "acdc" else str(
            PROJECT_ROOT / "cronos_lumiere_baseline" / "paper_protocol_data")
    return run(args)


if __name__ == "__main__":
    main()
