"""Train and evaluate causal local matching and paired-error routing on ACDC."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.trajectory.external_data import ACDCJointDataset, PROJECT_ROOT
from src.trajectory.external_eval import save_evaluation
from methods.trajdetail.matched_control import MatchedControl, paired_benefit
from src.trajectory.train import seed_all
from methods.trajdetail.train_local_control import CONTROL_RUN, last_features, load_backbone
from src.trajectory.train_unified import _autocast, _input, trajectory_loss
from methods.trajdetail.train_unified_dino import DEFAULT_CACHE
from src.trajectory.validate_control import FixedWindows, batch_item, json_hash, write_json


def semantic_labels(model, target, item, cache, flips):
    if model.use_dino:
        path = cache / f"{item['patient_id']}.npy"
        indices = [round(float(value) * 11) for value in item["target_times"]]
        features = np.load(path, mmap_mode="r", allow_pickle=False)[indices]
        axes = tuple(axis for axis, flip in zip((3, 4), flips) if flip)
        if axes:
            features = np.flip(features, axis=axes)
        features = torch.from_numpy(np.ascontiguousarray(features)).to(target.device)
    else:
        features = None
    with torch.no_grad():
        return model.descriptor(target.flatten(0, 1).unsqueeze(1), features, model.feature_size)[None]


def paired_loss(model, result, target):
    utility = result["utilities"]
    benefit = paired_benefit(result["base_forecasts"], result["proposals"], target, utility.shape[-3:])
    label = (benefit.detach() / model.benefit_scale).clamp(-4., 4.)
    # Weight blocks with consequential corrections above the stationary background.
    weight = 1. + label.abs().sqrt()
    return (F.smooth_l1_loss(utility, label, reduction="none", beta=.25) * weight).mean()


def region_scores(pred, target, anchor, ids):
    rows = []
    for p, t, s, patient in zip(pred, target, anchor, ids):
        change = np.abs(t - s)
        high = change >= np.quantile(change, .9)
        error = np.abs(p - t)
        rows.append({"patient_id": patient, "high_change_mae": float(error[high].mean()),
                     "other_mae": float(error[~high].mean())})
    return {key: float(np.mean([row[key] for row in rows]))
            for key in ("high_change_mae", "other_mae")}


@torch.inference_mode()
def evaluate(model, dataset, device, cache, *, output=None, prefix=""):
    model.eval()
    arms = ("base_forecasts", "proposals", "predictions") if isinstance(model, MatchedControl) else ("predictions",)
    all_preds = {arm: [] for arm in arms}
    targets, anchors, ids, history_counts = [], [], [], []
    positive, routed, beneficial, gates = [], [], [], []
    for index in range(len(dataset)):
        item = dataset[index]
        batch = batch_item(item)
        kwargs = {}
        if isinstance(model, MatchedControl) and model.use_dino:
            kwargs["last_features"] = last_features(item, cache, device)
        with _autocast(device, True):
            result = model(*_input(batch, device), **kwargs)
        for arm in arms:
            all_preds[arm].append(result[arm][0, -1].cpu().numpy())
        targets.append(item["targets"][-1])
        anchors.append(result["last_context"][0].cpu().numpy())
        ids.append(item["patient_id"])
        history_counts.append(int(np.count_nonzero(item["mask"])))
        if output is not None and "gates" in result:
            gate = result["gates"][:, -1:]
            signed = paired_benefit(result["base_forecasts"][:, -1:], result["proposals"][:, -1:],
                                    batch["targets"][:, -1:].to(device), gate.shape[-3:])
            selected = gate > 0
            positive.append(float(((signed > 0) & selected).sum()))
            routed.append(float(selected.sum()))
            beneficial.append(float((signed > 0).sum()))
            gates.append(gate[0, 0].cpu().numpy())
    report = {}
    targets, anchors = np.stack(targets), np.stack(anchors)
    for arm, values in all_preds.items():
        pred = np.stack(values)
        if not np.isfinite(pred).all():
            raise FloatingPointError(f"nonfinite forecasts in {prefix}/{arm}")
        if output is None:
            report[arm] = {"mse": float(np.square(pred - targets).mean(dtype=np.float64))}
            continue
        directory = output / arm
        figures = save_evaluation(directory, pred, targets, anchors, ids, title=f"ACDC {prefix} {arm}")
        subset_ids = {patient for patient, count in zip(ids, history_counts) if count == 1}
        single = [row["nrmse"] for row in figures["per_patient"] if row["patient_id"] in subset_ids]
        report[arm] = {**figures["mean"], **region_scores(pred, targets, anchors, ids),
                       "single_observed_count": len(single), "single_observed_nrmse": float(np.mean(single))}
    if output is not None and gates:
        stacked = np.stack(gates)
        report["routing"] = {
            "mean_gate": float(stacked.mean()), "active_fraction": float((stacked > 0).mean()),
            "positive_benefit_fraction": sum(beneficial) / stacked.size,
            "precision_at_positive_gate": sum(positive) / max(1., sum(routed)),
        }
        np.savez_compressed(output / "gates.npz", patient_ids=ids, gates=stacked)
    return report


def train_stage(model, stage, training, valid, device, args, directory):
    model.train_stage(stage)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    order_rng = torch.Generator().manual_seed(42)
    augment_rng = np.random.RandomState(42)
    arm = "proposals" if stage == "matching" else "predictions"
    initial = evaluate(model, valid, device, args.cache)
    # Always retain the untouched control as an option for the benefit head.
    # Matching must select a trained proposal so the direction has a fair test.
    best = initial[arm]["mse"] if stage == "benefit" else float("inf")
    selected = 0
    history = [{"epoch": 0, "validation": initial}]
    path = directory / f"{stage}.pt"
    torch.save({"model": model.state_dict(), "epoch": selected}, path)
    epochs = args.match_epochs if stage == "matching" else args.benefit_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr * .1)
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.monotonic()
        losses = []
        order = torch.randperm(len(training) // 2, generator=order_rng).tolist()
        for person in order:
            item = training[person * 2 + (epoch - 1) % 2]
            flips = tuple(bool(v) for v in augment_rng.random_sample(2) < .5)
            batch = batch_item(item, flips)
            target = batch["targets"].to(device)
            features = last_features(item, args.cache, device, flips) if model.use_dino else None
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, True):
                result = model(*_input(batch, device), last_features=features)
                if stage == "matching":
                    proposal = {**result, "predictions": result["proposals"]}
                    image_loss, _ = trajectory_loss(proposal, batch, history_weight=0.)
                    labels = semantic_labels(model, target, item, args.cache, flips)
                    similarity = (result["queries"] * labels).sum(dim=2)
                    loss = image_loss + .0005 * (1. - similarity.mean())
                else:
                    loss = paired_loss(model, result, target)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise FloatingPointError(f"nonfinite training update in {directory.name}/{stage}")
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        validation = evaluate(model, valid, device, args.cache)
        score = validation[arm]["mse"]
        if score < best:
            best, selected = score, epoch
            torch.save({"model": model.state_dict(), "epoch": selected}, path)
        record = {"epoch": epoch, "training_loss": float(np.mean(losses)), "validation": validation,
                  "seconds": time.monotonic() - started}
        history.append(record)
        write_json(directory / f"{stage}_history.json", history)
        print(json.dumps({"arm": directory.name, "stage": stage, **record}), flush=True)
    chosen = torch.load(path, weights_only=True, map_location=device)
    model.load_state_dict(chosen["model"])
    reproduced = evaluate(model, valid, device, args.cache)
    if not np.isclose(reproduced[arm]["mse"], best, rtol=1e-6, atol=1e-9):
        raise RuntimeError(f"{stage} checkpoint disagrees with the selected validation score")
    return {"epoch": selected, "validation": reproduced, "trainable": sum(p.numel() for p in parameters)}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    bank = json.loads((CONTROL_RUN / "bank.json").read_text())
    protocol = PROJECT_ROOT / "cronos_acdc_baseline"
    manifest = hashlib.sha256((protocol / "data_manifest.json").read_bytes()).hexdigest()
    checkpoint_path = CONTROL_RUN / "students/control/best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if manifest != bank["manifest_sha256"] or json_hash(bank) != checkpoint["config"]["bank_sha256"]:
        raise ValueError("training bank, current manifest and frozen checkpoint must agree")
    train, val = FixedWindows(bank), ACDCJointDataset(split="validation")
    config = {**vars(args), "output": str(args.output), "cache": str(args.cache),
              "seed": 42, "bank_sha256": json_hash(bank), "manifest_sha256": manifest,
              "control_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "selection": "minimum validation MSE separately for match proposal and benefit-routed proposal",
              "supervision": "future training MRI and frozen future training descriptors only label losses",
              "inference": "same visible MRI window, timestamps, and descriptor of the last visible frame"}
    write_json(args.output / "config.json", config)
    source = args.output / "source_snapshot"
    source.mkdir()
    for name in ("matched_control.py", "train_matched_control.py", "unified_model.py", "external_eval.py"):
        shutil.copyfile(Path(__file__).with_name(name), source / name)
    report = {}
    baseline = load_backbone(device, checkpoint)
    report["trajectory_control"] = {"validation": evaluate(baseline, val, device, args.cache)}
    print(json.dumps({"stage": "baseline", **report["trajectory_control"]}), flush=True)
    for name, use_dino in (("mri_matching", False), ("dinov3_matching", True)):
        seed_all(42)
        directory = args.output / name
        directory.mkdir()
        model = MatchedControl(use_dino=use_dino).to(device)
        model.load_control(checkpoint["model"])
        matching = train_stage(model, "matching", train, val, device, args, directory)
        routed = train_stage(model, "benefit", train, val, device, args, directory)
        if abs(routed["validation"]["base_forecasts"]["mse"] -
               report["trajectory_control"]["validation"]["predictions"]["mse"]) > 1e-9:
            raise RuntimeError("the shared frozen forecast changed in the matching wrapper")
        report[name] = {"matching": matching, "benefit": routed}
    write_json(args.output / "selection.json", report)
    test = ACDCJointDataset(split="test")
    report["trajectory_control"]["test"] = evaluate(baseline, test, device, args.cache,
                                                     output=args.output / "trajectory_control", prefix="control")
    for name, use_dino in (("mri_matching", False), ("dinov3_matching", True)):
        model = MatchedControl(use_dino=use_dino).to(device)
        directory = args.output / name
        chosen = torch.load(directory / "benefit.pt", weights_only=True, map_location=device)
        model.load_state_dict(chosen["model"])
        report[name]["test"] = evaluate(model, test, device, args.cache, output=directory, prefix=name)
    write_json(args.output / "summary.json", report)
    print(json.dumps({"stage": "complete", **report}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--match-epochs", type=int, default=8)
    parser.add_argument("--benefit-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=.001)
    args = parser.parse_args()
    if min(args.match_epochs, args.benefit_epochs) < 1 or args.lr <= 0:
        parser.error("epoch counts and learning rate must be positive")
    run(args)


if __name__ == "__main__":
    main()
