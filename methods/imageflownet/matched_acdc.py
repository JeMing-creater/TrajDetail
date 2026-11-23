"""Retrain ImageFlowNet on the frozen ACDC long-horizon benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/ImageFlowNet-main"
sys.path[:0] = [str(ROOT), str(UPSTREAM)]

from src.trajectory.external_data import ACDCJointDataset
from src.trajectory.external_eval import save_evaluation
from src.trajectory.validate_control import FixedWindows
from methods.imageflownet.upstream.src.nn.imageflownet_ode import ImageFlowNetODE


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_trained_checkpoint(path):
    # The first run saved torch.__version__ as a TorchVersion object.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def latest_observation(item):
    slot = int(np.flatnonzero(item["mask"])[-1])
    delta_t = item["target_times"] - item["context_times"][slot]
    if not np.all(delta_t > 0):
        raise ValueError("Selected observation must precede both targets")
    return item["context"][slot], delta_t


def model_input(volume, device):
    # The upstream ACDC slice runner uses raw H,W order. The shared cache
    # loader presents each axial slice in W,H order.
    slices = torch.from_numpy(np.ascontiguousarray(volume.swapaxes(-1, -2)))
    slices = slices.to(device=device, dtype=torch.float32).unsqueeze(1)
    return F.interpolate(slices.clamp(0, 1) * 2 - 1, (256, 256),
                         mode="bilinear", align_corners=False)


def forward_native(model, image, delta_t, shape):
    # Canonical time is ED-to-ES in [0,1]. The source checkpoint used
    # frame_gap * 5/11, so preserve its time units by scaling by 5.
    pred = model(x=image, t=image.new_tensor([5.0 * float(delta_t)]))
    pred = F.interpolate(pred.float(), shape[::-1], mode="bilinear", align_corners=False)
    return (pred[:, 0].transpose(-1, -2) + 1) / 2


@torch.inference_mode()
def evaluate(model, dataset, device, slice_batch, store_volumes=False):
    model.eval()
    mse = []
    predictions, targets, observations, patient_ids = [], [], [], []
    for item in dataset:
        last, delta_t = latest_observation(item)
        target = item["targets"][-1]
        inputs = model_input(last, device)
        prediction = torch.cat([
            forward_native(model, image, delta_t[-1], target.shape[-2:])
            for image in inputs.split(slice_batch)
        ]).cpu().numpy()
        mse.append(float(np.square(prediction - target).mean(dtype=np.float64)))
        if store_volumes:
            predictions.append(prediction)
            targets.append(target)
            observations.append(last)
            patient_ids.append(item["patient_id"])
            if len(patient_ids) % 10 == 0:
                print(json.dumps({"phase": "test", "completed_patients": len(patient_ids),
                                  "total_patients": len(dataset)}), flush=True)
    if not store_volumes:
        return float(np.mean(mse))
    return (np.stack(predictions), np.stack(targets),
            np.stack(observations), patient_ids)


def train_epoch(model, dataset, args, optimizer, device, epoch,
                order_generator, augmentation_generator):
    model.train()
    order = torch.randperm(len(dataset) // 2, generator=order_generator).tolist()
    batch_size = args.train_slice_batch
    # Alternate the same two windows per patient as the control model.
    bank_slot = (epoch - 1) % 2
    accum = 0.0
    tick = time.monotonic()
    for update, patient_index in enumerate(order, start=1):
        item = dataset[2 * patient_index + bank_slot]
        last, delta_t = latest_observation(item)
        # Rotate through disjoint axial subsets on each occurrence of a window.
        # Every chosen subset spans the full axial extent of the volume.
        depth = last.shape[0]
        offset = (patient_index + (epoch - 1) // 2) % (depth // batch_size)
        indices = np.arange(offset, depth, depth // batch_size)
        target = item["targets"][:, indices]
        last = last[indices]
        for axis, flip in zip((-2, -1), augmentation_generator.random_sample(2) < .5):
            if flip:
                target = np.flip(target, axis=axis)
                last = np.flip(last, axis=axis)
        image = model_input(last, device)
        target = torch.from_numpy(np.ascontiguousarray(target)).to(device)
        optimizer.zero_grad(set_to_none=True)
        # Separate backward calls preserve the weighted objective and release
        # the first solver graph before the later target is evaluated.
        loss_total = 0.0
        for target_image, gap, weight in zip(target, delta_t, (1 / 3, 2 / 3)):
            prediction = forward_native(model, image, gap, target.shape[-2:])
            loss = weight * (F.mse_loss(prediction, target_image) +
                             .01 * F.l1_loss(prediction, target_image))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, update {update}")
            loss.backward()
            loss_total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        accum += loss_total
        if update == 1 or update % 20 == 0:
            print(json.dumps({"epoch": epoch, "update": update, "phase": "train",
                              "training_loss_running": accum / update,
                              "seconds": round(time.monotonic() - tick, 1)}), flush=True)
    return accum / len(order)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=ROOT /
                        "runs/trajectory_control_acdc_seed42_20261117_v3/bank.json")
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT /
                        "imageflownet_baseline/runs/acdc_seed1/best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--schedule-epochs", type=int, help="Cosine schedule length, defaults to --epochs.")
    parser.add_argument("--validate-every", type=int, default=2)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validation-only", action="store_true",
                      help="Select a checkpoint using validation and skip opening the test split.")
    mode.add_argument("--test-candidates", nargs="+", type=Path, metavar="RUN_DIR",
                      help="Evaluate the single candidate with the lowest trained validation MSE.")
    parser.add_argument("--train-slice-batch", type=int, choices=(1, 2, 4, 8, 16, 32), default=4)
    parser.add_argument("--eval-slice-batch", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    if args.schedule_epochs is not None and args.schedule_epochs < args.epochs:
        parser.error("--schedule-epochs must cover all training epochs")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.test_candidates:
        selected = select_checkpoint(args.test_candidates, args.output)
        save_test(selected, args.output, device, args.eval_slice_batch, started)
        return
    bank = json.loads(args.bank.read_text())
    manifest = ROOT / "cronos_acdc_baseline/data_manifest.json"
    split_dir = ROOT / "cronos_acdc_baseline/runs/discrete_seed42"
    if bank["manifest_sha256"] != sha256(manifest) or bank["split"] != "train":
        raise ValueError("Training bank does not match the frozen ACDC training manifest")
    dataset = FixedWindows(bank)
    validation_set = ACDCJointDataset(split="validation", augment=False)
    training_ids = {row["patient_id"] for row in bank["windows"]}
    validation_ids = {f"patient{row[2]}" for row in validation_set.sources}
    test_ids = set(map(lambda ident: f"patient{ident}",
                       json.loads(manifest.read_text())["split_patient_ids_one_based"]["test"]))
    expected_splits = (len(training_ids), len(validation_ids), len(test_ids), len(dataset))
    if expected_splits != (80, 20, 50, 160) or (training_ids & validation_ids or
            training_ids & test_ids or validation_ids & test_ids):
        raise ValueError(f"Unexpected split or overlap: {expected_splits}")
    if any(row["bank_index"] != index or row["patient_index"] != index // 2 or
           row["window_number"] != index % 2 for index, row in enumerate(bank["windows"])):
        raise ValueError("The fixed bank must contain two consecutive windows per patient")

    model = ImageFlowNetODE(device=device, in_channels=1, ode_location="all_connections")
    source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
    if (source["manifest"]["dataset"] != "acdc" or
            not np.isclose(source["manifest"]["t_multiplier"], 5 / 11)):
        raise ValueError("Initialization must be the ACDC adjacent-frame checkpoint")
    model.load_state_dict(source["model"], strict=True)
    del source
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.schedule_epochs or args.epochs)

    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pid": os.getpid(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "bank_sha256": sha256(args.bank), "init_sha256": sha256(args.init_checkpoint),
        "script_sha256": sha256(Path(__file__)), "data_manifest_sha256": sha256(manifest),
        "fixed_validation_masks_sha256": sha256(split_dir / "fixed_masks_validation.json"),
        "fixed_test_masks_sha256": sha256(split_dir / "fixed_masks_test.json"),
        "model_source_sha256": sha256(UPSTREAM / "src/nn/imageflownet_ode.py"),
        "counts": {"train": 80, "validation": 20, "test": 50, "train_windows": 160},
        "parameters": sum(p.numel() for p in model.parameters()),
        "input": "last valid observation in frozen context and requested target time minus its observation time",
        "architecture": "upstream ImageFlowNetODE, all_connections, strict load from adjacent ACDC checkpoint",
        "time_multiplier": 5.0,
        "solver": "upstream adaptive torchdiffeq odeint, rtol=atol=0.001",
        "scoring": "float32, full 32x128x128 volume, no clipping or ROI, patient means",
        "loss": "q-1 and q with weights 1:2, MSE + 0.01 MAE at 128x128",
        "selection": "minimum frozen validation latest-target patient-mean 3D MSE among fine-tuned checkpoints",
        "train_windows_per_epoch": 80,
        "source_training": "same 80 training patients, adjacent ED-ES phase pairs throughout the sequence",
    }
    write_json(args.output / "config.json", config)
    print(json.dumps({key: config[key] for key in ("pid", "device", "counts", "epochs", "parameters")}), flush=True)

    # Initial score provides a same-run reference, but selection is restricted to
    # checkpoints that actually trained on the matched long-horizon windows.
    initial_mse = evaluate(model, validation_set, device, args.eval_slice_batch)
    history = [{"epoch": 0, "validation_mse": initial_mse}]
    print(json.dumps(history[0]), flush=True)
    write_json(args.output / "history.json", history)
    best, best_epoch = float("inf"), None
    rng = np.random.RandomState(args.seed)
    order = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        tick = time.monotonic()
        training_loss = train_epoch(model, dataset, args, optimizer, device, epoch, order, rng)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": training_loss,
               "train_seconds": round(time.monotonic() - tick, 1)}
        if epoch == 1 or epoch == args.epochs or epoch % args.validate_every == 0:
            mse = evaluate(model, validation_set, device, args.eval_slice_batch)
            row["validation_mse"] = mse
            if mse < best:
                best, best_epoch = mse, epoch
                temporary = args.output / "best.tmp"
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "validation_mse": mse, "config": config}, temporary)
                temporary.replace(args.output / "best.pt")
        history.append(row)
        write_json(args.output / "history.json", history)
        print(json.dumps(row), flush=True)

    if args.validation_only:
        return
    del model, optimizer
    save_test(args.output / "best.pt", args.output, device, args.eval_slice_batch, started)


def select_checkpoint(run_dirs, output):
    candidates = []
    shared_config = None
    for run_dir in run_dirs:
        checkpoint_path = run_dir.resolve() / "best.pt"
        history = json.loads((run_dir / "history.json").read_text())
        best = min((row for row in history if row["epoch"] > 0 and "validation_mse" in row),
                   key=lambda row: row["validation_mse"])
        checkpoint = load_trained_checkpoint(checkpoint_path)
        if (best["epoch"], best["validation_mse"]) != (checkpoint["epoch"], checkpoint["validation_mse"]):
            raise ValueError(f"Best checkpoint does not match completed validation history in {run_dir}")
        config = checkpoint["config"]
        protocol = {key: config[key] for key in (
            "bank_sha256", "data_manifest_sha256", "fixed_validation_masks_sha256",
            "fixed_test_masks_sha256", "model_source_sha256", "init_sha256", "seed",
            "train_slice_batch", "eval_slice_batch", "input", "time_multiplier",
            "solver", "scoring", "loss", "selection")}
        if shared_config is not None and protocol != shared_config:
            raise ValueError(f"Candidate does not match the shared selection protocol in {run_dir}")
        shared_config = protocol
        candidates.append({"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
                           "epoch": best["epoch"], "validation_mse": best["validation_mse"],
                           "learning_rate": config["learning_rate"]})
    selected = min(candidates, key=lambda row: row["validation_mse"])
    write_json(output / "selection.json", {"selected": selected, "candidates": candidates})
    return Path(selected["checkpoint"])


def save_test(checkpoint_path, output, device, eval_slice_batch, started):
    checkpoint = load_trained_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    fixed_test_masks = ROOT / "cronos_acdc_baseline/runs/discrete_seed42/fixed_masks_test.json"
    if (sha256(fixed_test_masks) != config["fixed_test_masks_sha256"] or
            eval_slice_batch != config["eval_slice_batch"]):
        raise ValueError("Test mask file or slice batch differs from the trained protocol")
    model = ImageFlowNetODE(device=device, in_channels=1, ode_location="all_connections")
    model.load_state_dict(checkpoint["model"], strict=True)
    test_set = ACDCJointDataset(split="test", augment=False)
    volumes = evaluate(model, test_set, device, eval_slice_batch, store_volumes=True)
    eval_summary = save_evaluation(output / "test", *volumes,
                                   title="ACDC ImageFlowNet trained on matched windows")
    paired_nrmse_wins = sum(
        row["nrmse"] < copy["nrmse"]
        for row, copy in zip(eval_summary["per_patient"], eval_summary["last_context_per_patient"])
    )
    result = {"epoch": checkpoint["epoch"], "validation_mse": checkpoint["validation_mse"],
              "checkpoint": str(checkpoint_path), "n_patients": eval_summary["n_patients"],
              "mean": eval_summary["mean"], "median": eval_summary["median"],
              "copy_last": eval_summary["last_context"], "nrmse_wins_against_copy": paired_nrmse_wins,
              "total_seconds": round(time.monotonic() - started, 1)}
    write_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
