"""Train and evaluate the first ISLES joint-trajectory performance pilot."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.trajectory.data import ISLESJointDataset
from methods.trajdetail.evaluate import evaluate_arrays
from methods.trajdetail.model import JointTrajectoryFlow


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _batch_to_device(batch, device):
    context = batch["context"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    context_times = batch["context_times"].to(device, non_blocking=True)
    target_times = batch["target_times"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    return context, mask, context_times, target_times, targets


def _amp_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _flow_loss(model, batch, device, endpoint_weight=0.25, route_weight=0.01,
               use_route=False, amp=False, endpoint_steps=1,
               primary_weight=2.0, endpoint_l1_weight=0.0):
    context, mask, context_times, target_times, targets = _batch_to_device(batch, device)
    with _amp_context(device, amp):
        source = model.make_source(context, mask, context_times, target_times)
        tau = torch.rand(context.shape[0], device=device)
        x_tau = source + tau[:, None, None, None, None] * (targets - source)
        velocity = model(x_tau, tau, context, mask, context_times, target_times)
        target_weights = velocity.new_tensor([1.0, primary_weight]).reshape(1, 2, 1, 1, 1)
        weight_normalizer = target_weights.mean()
        velocity_loss = ((velocity - (targets - source)).square() * target_weights).mean() / weight_normalizer

        # Endpoint supervision reduces the train/inference discrepancy from the
        # finite Euler rollout while keeping the same patient-paired targets.
        endpoint = model.predict(context, mask, context_times, target_times,
                                 steps=endpoint_steps, source=source)
        endpoint_loss = ((endpoint - targets).square() * target_weights).mean() / weight_normalizer
        endpoint_l1 = ((endpoint - targets).abs() * target_weights).mean() / weight_normalizer
        route_loss = endpoint.new_zeros(())
        if use_route:
            transitioned = model.transition(
                endpoint[:, 0], context, mask, context_times,
                target_times[:, 0], target_times[:, 1], steps=2,
            )
            route_loss = (transitioned - endpoint[:, 1]).square().mean()
        total = (velocity_loss + endpoint_weight * endpoint_loss
                 + endpoint_l1_weight * endpoint_l1 + route_weight * route_loss)
    return total, {"velocity": velocity_loss.detach(), "endpoint": endpoint_loss.detach(),
                   "endpoint_l1": endpoint_l1.detach(),
                   "route": route_loss.detach()}


@torch.no_grad()
def predict_dataset(model, dataset, device, batch_size=4, steps=10, amp=False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    predictions, targets, lasts, ids = [], [], [], []
    model.eval()
    for batch in loader:
        context, mask, context_times, target_times, target_pair = _batch_to_device(batch, device)
        with _amp_context(device, amp):
            prediction = model.predict(context, mask, context_times, target_times, steps=steps)
        predictions.append(prediction[:, 1].cpu().numpy())
        targets.append(target_pair[:, 1].cpu().numpy())
        summary = model.history_summary(context, mask, context_times)
        lasts.append(summary["last"].cpu().numpy())
        ids.extend(batch["patient_id"])
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(lasts), ids


def run(args):
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    train_data = ISLESJointDataset(data_root, "trn", augment=args.augment)
    val_data = ISLESJointDataset(data_root, "val", augment=False)
    test_data = None

    model = JointTrajectoryFlow(base_channels=args.base_channels).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))
    config = {
        "seed": args.seed, "device": str(device), "epochs": args.epochs,
        "batch_size": args.batch_size, "base_channels": args.base_channels,
        "parameters": parameter_count, "lr": args.lr, "weight_decay": args.weight_decay,
        "endpoint_weight": args.endpoint_weight, "route_weight": args.route_weight,
        "route_warmup_epoch": args.route_warmup_epoch, "euler_steps": args.steps,
        "selection": "lowest validation primary-target mean MSE; no test selection",
        "amp": bool(args.amp), "training_endpoint_steps": args.endpoint_steps,
        "primary_target_weight": args.primary_weight,
        "endpoint_l1_weight": args.endpoint_l1_weight, "spatial_flip_augmentation": args.augment,
        "data_root": str(data_root),
        "manifest_sha256": _sha256(data_root / "manifest.json"),
        "evaluation_samples_sha256": _sha256(data_root / "evaluation_samples.json"),
        "protocol": "frozen ISLES CRONOS windows; joint labels q-1,q; q remains primary target",
        "model_file": str((Path(__file__).resolve().parent / "model.py")),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history = []
    best = float("inf")
    best_epoch = None
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"total": 0.0, "velocity": 0.0, "endpoint": 0.0,
                "endpoint_l1": 0.0, "route": 0.0}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            total, parts = _flow_loss(
                model, batch, device, args.endpoint_weight, args.route_weight,
                use_route=epoch >= args.route_warmup_epoch, amp=args.amp,
                endpoint_steps=args.endpoint_steps, primary_weight=args.primary_weight,
                endpoint_l1_weight=args.endpoint_l1_weight,
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss epoch {epoch}: {total.item()}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums["total"] += float(total.detach())
            for key, value in parts.items():
                sums[key] += float(value)
        scheduler.step()
        count = len(loader)
        record = {"epoch": epoch, "train": {k: v / count for k, v in sums.items()},
                  "lr": optimizer.param_groups[0]["lr"]}
        if epoch % args.val_interval == 0 or epoch == 1 or epoch == args.epochs:
            pred, target, last, ids = predict_dataset(model, val_data, device, args.batch_size, args.steps, args.amp)
            primary_mse = float(np.mean((pred - target) ** 2))
            record["validation_primary_mse"] = primary_mse
            record["validation_primary_nrmse"] = float(np.sqrt(primary_mse))
            if primary_mse < best:
                best = primary_mse
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "config": config,
                            "epoch": epoch, "validation_primary_mse": primary_mse}, output / "best.pt")
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if epoch == 1 or epoch % args.print_interval == 0 or "validation_primary_mse" in record:
            print(json.dumps(record), flush=True)
    if best_epoch is None:
        raise RuntimeError("no validation checkpoint was selected")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_data = ISLESJointDataset(data_root, "tst", augment=False)
    pred, target, last, ids = predict_dataset(model, test_data, device, args.batch_size, args.steps, args.amp)
    metrics = evaluate_arrays(pred, target, last, ids, output_dir=output / "test_eval", make_figure=True)
    result = {"best_epoch": best_epoch, "best_validation_primary_mse": best,
              "test": metrics, "elapsed_seconds": time.monotonic() - start}
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/mnt/Datasets/ISLES24/prepared_paper")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--endpoint-weight", type=float, default=0.25)
    parser.add_argument("--route-weight", type=float, default=0.01)
    parser.add_argument("--route-warmup-epoch", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--amp", action="store_true", help="use CUDA bfloat16 autocast")
    parser.add_argument("--endpoint-steps", type=int, default=1,
                        help="differentiable Euler steps used only for training endpoint loss")
    parser.add_argument("--primary-weight", type=float, default=2.0,
                        help="relative loss weight for the q target versus q-1")
    parser.add_argument("--endpoint-l1-weight", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true", help="random in-plane flips of all paired images")
    parser.add_argument("--val-interval", type=int, default=10)
    parser.add_argument("--print-interval", type=int, default=10)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
