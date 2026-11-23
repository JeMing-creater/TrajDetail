"""Paired ACDC experiment for trajectory refinement with and without frozen DINOv3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from methods.trajdetail.control import ControlTrajectory
from src.trajectory.external_data import ACDCJointDataset, PROJECT_ROOT
from src.trajectory.external_eval import evaluate_metrics, save_evaluation
from methods.trajdetail.local_control import LocalControlTrajectory
from src.trajectory.train import seed_all
from src.trajectory.train_unified import _autocast, _input, trajectory_loss
from methods.trajdetail.train_unified_dino import DEFAULT_CACHE, DINO_WEIGHTS
from src.trajectory.validate_control import FixedWindows, batch_item, json_hash, write_json


CONTROL_RUN = PROJECT_ROOT / "runs/trajectory_control_acdc_seed42_20261117_v3"


def training_transport(bank, control_size):
    manifest = json.loads((CONTROL_RUN / "teacher_cache_manifest.json").read_text())
    if manifest["bank_sha256"] != json_hash(bank) or manifest["count"] != len(bank["windows"]):
        raise ValueError("teacher cache does not match the frozen training windows")
    labels = []
    for index in range(len(bank["windows"])):
        source = CONTROL_RUN / "teachers" / f"window_{index:04d}.pt"
        with source.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["files"][source.name]:
                raise ValueError(f"teacher checksum mismatch: {source}")
        teacher = torch.load(source, map_location="cpu", weights_only=True)["teacher"]
        labels.append(F.adaptive_avg_pool3d(teacher["transport_components"].float(), control_size))
    return labels


def transport_loss(result, teacher, flips):
    components = result["transport_components"].float()
    axes = tuple(axis for axis, flip in zip((3, 4), flips) if flip)
    target = teacher.flip(axes) if axes else teacher
    predicted = F.adaptive_avg_pool3d(components, target.shape[-3:])
    by_time = (predicted - target.to(predicted)).square().flatten(2).mean(dim=2)
    weights = by_time.new_ones(by_time.shape[1])
    weights[-1] = 2.
    return ((by_time * weights).sum(dim=1) / weights.sum()).mean()


def last_features(item, cache, device, flips=(False, False)):
    slot = np.flatnonzero(item["mask"])[-1]
    phase = round(float(item["context_times"][slot]) * 11)
    if phase >= int(item["primary_index"]) - 1:
        raise ValueError("DINO feature must be from the observed history")
    entry = np.load(cache / f"{item['patient_id']}.npy", mmap_mode="r", allow_pickle=False)
    if entry.shape != (12, 768, 8, 8, 8):
        raise ValueError(f"unexpected frozen DINO cache shape: {entry.shape}")
    feature = entry[phase]
    for axis, flip in zip((2, 3), flips):
        if flip:
            feature = np.flip(feature, axis=axis)
    return torch.from_numpy(feature.copy()).unsqueeze(0).to(device)


def loss_for_batch(result, batch, stability_weight):
    image_loss, parts = trajectory_loss(result, batch, history_weight=0.)
    controls = result["future_controls"]
    observed_difference = (batch["targets"].to(controls) -
                           result["last_context"][:, None]).abs().amax(dim=1, keepdim=True)
    coarse_change = F.adaptive_avg_pool3d(observed_difference, controls.shape[-3:])
    # Target pixels label the regularizer while training. The model inputs and
    # the displacement computation use only observed images and their times.
    stability = torch.exp(-coarse_change / .04)
    penalty = (stability * controls.square()).mean()
    return image_loss + stability_weight * penalty, parts["primary_mse"], penalty


@torch.inference_mode()
def evaluate(model, dataset, device, cache=None, *, save=None, title="", last_only=False):
    model.eval()
    rows, predictions, targets, anchors = [], [], [], []
    for index in range(len(dataset)):
        item = dataset[index]
        batch = batch_item(item)
        features = None if cache is None else last_features(item, cache, device)
        if last_only:
            slot = np.flatnonzero(item["mask"])[-1]
            batch["mask"].zero_()
            batch["mask"][0, slot] = 1
        inputs = _input(batch, device)
        with _autocast(device, True):
            result = model(*inputs, last_features=features) if isinstance(model, LocalControlTrajectory) else model(*inputs)
        prediction = result["predictions"][0, -1].float().cpu().numpy()
        target = item["targets"][-1]
        anchor = result["last_context"][0].float().cpu().numpy()
        error = prediction - target
        change = np.abs(target - anchor)
        high = change >= np.quantile(change, .90)
        rows.append({
            "patient_id": item["patient_id"],
            "mse": float(np.square(error).mean()),
            "high_change_mae": float(np.abs(error[high]).mean()),
            "other_mae": float(np.abs(error[~high]).mean()),
        })
        predictions.append(prediction)
        targets.append(target)
        anchors.append(anchor)
    result = {
        "mean": {key: float(np.mean([p[key] for p in rows])) for key in
                 ("mse", "high_change_mae", "other_mae")},
        "per_patient": rows,
    }
    arrays = np.stack(predictions), np.stack(targets), np.stack(anchors)
    ids = [row["patient_id"] for row in rows]
    if save is not None:
        metrics = save_evaluation(save, *arrays, ids, title=title)
        result["metrics"] = metrics["mean"]
        write_json(save / "regional_error.json", result)
    else:
        result["metrics"] = evaluate_metrics(arrays[0], arrays[1], ids)["mean"]
    return result


def load_backbone(device, checkpoint):
    model = ControlTrajectory(capture_components=False).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def fit(args):
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    bank = json.loads((CONTROL_RUN / "bank.json").read_text())
    manifest = PROJECT_ROOT / "cronos_acdc_baseline/data_manifest.json"
    if bank["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError("current manifest differs from frozen training bank")
    training = FixedWindows(bank)
    validation = ACDCJointDataset(split="validation")
    checkpoint_path = CONTROL_RUN / "students/control/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["config"]["bank_sha256"] != json_hash(bank):
        raise ValueError("trajectory control checkpoint was trained on a different window bank")
    transport_labels = training_transport(bank, (16, 32, 32)) if args.transport_weight else None
    config = {
        "seed": 42, "epochs": args.epochs, "learning_rate": args.lr,
        "stability_weight": args.stability_weight, "device": str(device), "precision": "bfloat16 autocast",
        "control_checkpoint": str(checkpoint_path), "control_epoch": checkpoint["epoch"],
        "control_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "bank_sha256": json_hash(bank), "training_windows": len(training),
        "val_patients": len(validation), "dino_cache": str(args.cache), "dino_weights": str(DINO_WEIGHTS),
        "selection": "lowest validation latest-target mean MSE, including untouched initialization",
        "test": "single frozen 50-patient score per selected checkpoint; exploratory split",
        "future_regularizer": "training targets label low change at the control resolution",
        "transport_weight": args.transport_weight,
        "transport_labels": str(CONTROL_RUN / "teacher_cache_manifest.json") if transport_labels else None,
        "model": "frozen trajectory control backbone, two bounded velocity residual coefficients per location",
    }
    write_json(args.output / "config.json", config)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for name in ("local_control.py", "train_local_control.py", "unified_model.py", "validate_control.py",
                 "external_data.py", "external_eval.py"):
        shutil.copyfile(Path(__file__).with_name(name), snapshot / name)

    baseline = load_backbone(device, checkpoint)
    baseline_val = evaluate(baseline, validation, device)
    write_json(args.output / "control_validation.json", baseline_val)
    print(json.dumps({"stage": "baseline_validation", **baseline_val["metrics"]}), flush=True)

    summary = {"control": {"validation": baseline_val["metrics"]}}
    planned_arms = (("local_mri", None), ("local_dinov3", args.cache))
    for arm, cache in planned_arms:
        out = args.output / arm
        out.mkdir()
        seed_all(42)
        model = LocalControlTrajectory(supervise_transport=transport_labels is not None).to(device)
        model.load_control(checkpoint["model"])
        trainable = [p for name, p in model.named_parameters()
                     if p.requires_grad and (cache is not None or not name.startswith("feature_"))]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        order_rng = torch.Generator().manual_seed(42)
        augment_rng = np.random.RandomState(42)
        best = baseline_val["mean"]["mse"]
        best_epoch = 0
        torch.save({"model": model.state_dict(), "epoch": 0}, out / "best.pt")
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            start = time.monotonic()
            losses, penalties, transport_errors = [], [], []
            for person in torch.randperm(80, generator=order_rng).tolist():
                index = person * 2 + (epoch - 1) % 2
                item = training[index]
                flips = tuple(bool(x) for x in augment_rng.random_sample(2) < .5)
                batch = batch_item(item, flips)
                features = None if cache is None else last_features(item, cache, device, flips)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, True):
                    result = model(*_input(batch, device), last_features=features)
                    loss, primary_mse, penalty = loss_for_batch(result, batch, args.stability_weight)
                    if transport_labels is not None:
                        auxiliary = transport_loss(result, transport_labels[index], flips)
                        loss = loss + args.transport_weight * auxiliary
                        transport_errors.append(float(auxiliary.detach()))
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise FloatingPointError(f"nonfinite training loss/gradient in {arm}")
                losses.append(float(primary_mse))
                penalties.append(float(penalty.detach()))
                optimizer.step()
            scheduler.step()
            val = evaluate(model, validation, device, cache)
            score = val["mean"]["mse"]
            entry = {"epoch": epoch, "train_mse": float(np.mean(losses)),
                     "train_stability_penalty": float(np.mean(penalties)),
                     "validation": val["metrics"], "duration_s": time.monotonic() - start}
            if transport_errors:
                entry["train_transport_mse"] = float(np.mean(transport_errors))
            history.append(entry)
            if score < best:
                best, best_epoch = score, epoch
                torch.save({"model": model.state_dict(), "epoch": epoch}, out / "best.pt")
            write_json(out / "history.json", history)
            print(json.dumps({"arm": arm, **entry}), flush=True)
        selected = torch.load(out / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(selected["model"])
        validation_report = evaluate(model, validation, device, cache)
        if not np.isclose(validation_report["mean"]["mse"], best, atol=1e-9, rtol=1e-5):
            raise RuntimeError(f"selected {arm} checkpoint did not reproduce validation MSE")
        write_json(out / "selected_validation.json", validation_report)
        summary[arm] = {"best_epoch": best_epoch, "validation": validation_report["metrics"],
                        "optimized_parameters": sum(p.numel() for p in trainable)}

    # Freeze both validation decisions before evaluating either candidate on test.
    write_json(args.output / "selection.json", summary)
    test_data = ACDCJointDataset(split="test")
    result = evaluate(baseline, test_data, device, save=args.output / "control_test",
                      title="ACDC trajectory control")
    summary["control"]["test"] = result["metrics"]
    summary["control"]["regional"] = result["mean"]
    for arm, cache in planned_arms:
        out = args.output / arm
        model = LocalControlTrajectory().to(device)
        model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True)["model"])
        report = evaluate(model, test_data, device, cache, save=out / "test", title=f"ACDC {arm}")
        summary[arm]["test"] = report["metrics"]
        summary[arm]["regional"] = report["mean"]
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"stage": "complete", "results": summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--stability-weight", type=float, default=.002)
    parser.add_argument("--transport-weight", type=float, default=0.)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.stability_weight < 0 or args.transport_weight < 0:
        parser.error("expected positive epochs and learning rate and nonnegative loss weights")
    fit(args)


if __name__ == "__main__":
    main()
