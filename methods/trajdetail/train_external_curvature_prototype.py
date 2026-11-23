"""Train the ACDC curvature-corrected prototype on the frozen external windows."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from methods.trajdetail.audit_acdc_sharpness import ReplayProbe, regions
from methods.trajdetail.composed_readout import curvature_corrected_warp
from src.trajectory.control_external import (
    DEFAULTS, _base_dataset, _manifest_hash, _sha256,
    fixed_training_dataset, window_protocol,
)
from src.trajectory.external_data import PROJECT_ROOT
from src.trajectory.external_eval import _save_panel
from src.metrics import score
from methods.trajdetail.matched_control import MatchedControl
from src.trajectory.train import seed_all
from methods.trajdetail.train_acdc_frequency_readout import measure
from methods.trajdetail.train_matched_control import paired_loss
from methods.trajdetail.train_transport_acdc import WEIGHTS, encode_volume, load_dino
from src.trajectory.train_unified import _autocast, _input, trajectory_loss
from src.trajectory.validate_control import batch_item, data_hash, write_json


def axial(volume, dataset):
    return np.moveaxis(volume, -1, 0) if dataset == "lumiere" else volume


class Features:
    """Cache frozen descriptors by patient and acquisition time for one run."""
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device
        self.encoder = load_dino(device)
        self.cache = {}

    def get(self, item, slot, *, training_target=False):
        images, times = ("targets", "target_times") if training_target else ("context", "context_times")
        key = item["patient_id"], float(item[times][slot])
        if key not in self.cache:
            image = np.ascontiguousarray(axial(item[images][slot], self.dataset))
            feature = encode_volume(self.encoder, image, self.device)
            if self.dataset == "lumiere":
                feature = np.moveaxis(feature, 1, -1)
            self.cache[key] = np.ascontiguousarray(feature)
        return self.cache[key]

    def observed(self, item, flips=(False, False)):
        feature = self.get(item, int(np.flatnonzero(item["mask"])[-1]))
        axes = tuple(axis for axis, flip in zip((-2, -1), flips) if flip)
        return torch.from_numpy(np.ascontiguousarray(np.flip(feature, axis=axes)))[None].to(self.device)

    def labels(self, model, target, item, flips):
        features = np.stack([self.get(item, i, training_target=True) for i in range(len(item["targets"]))])
        axes = tuple(axis for axis, flip in zip((-2, -1), flips) if flip)
        features = torch.from_numpy(np.ascontiguousarray(np.flip(features, axis=axes))).to(self.device)
        with torch.no_grad():
            return model.descriptor(target.flatten(0, 1).unsqueeze(1), features, model.feature_size)[None]


@torch.no_grad()
def validation(model, dataset, features, device):
    model.eval()
    errors = {key: [] for key in ("base_forecasts", "proposals", "predictions")}
    for item in dataset:
        batch = batch_item(item)
        target = batch["targets"][:, -1].to(device)
        with _autocast(device, True):
            result = model(*_input(batch, device), last_features=features.observed(item))
        for key in errors:
            errors[key].append(float((result[key][:, -1].float() - target).square().mean()))
    return {key: float(np.mean(values)) for key, values in errors.items()}


def train_stage(model, stage, train, valid, features, device, args):
    model.train_stage(stage)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    epochs = args.match_epochs if stage == "matching" else args.benefit_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=args.lr * .1)
    order = torch.Generator().manual_seed(42)
    augment = np.random.RandomState(42)
    arm = "proposals" if stage == "matching" else "predictions"
    initial = validation(model, valid, features, device)
    best = initial[arm] if stage == "benefit" else float("inf")
    path = args.output / f"{stage}.pt"
    torch.save({"model": model.state_dict(), "epoch": 0}, path)
    history = [{"epoch": 0, "validation_mse": initial}]
    print(json.dumps({"stage": stage, **history[0]}), flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.monotonic()
        losses = []
        for person in torch.randperm(len(train) // 2, generator=order).tolist():
            item = train[person * 2 + (epoch - 1) % 2]
            flips = tuple(bool(v) for v in augment.random_sample(2) < .5)
            batch = batch_item(item, flips)
            observed = features.observed(item, flips)
            target = batch["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, True):
                result = model(*_input(batch, device), last_features=observed)
                if stage == "matching":
                    loss, _ = trajectory_loss({**result, "predictions": result["proposals"]}, batch,
                                             history_weight=0.)
                    labels = features.labels(model, target, item, flips)
                    loss = loss + .0005 * (1. - (result["queries"] * labels).sum(dim=2).mean())
                else:
                    loss = paired_loss(model, result, target)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise FloatingPointError(f"nonfinite {args.dataset}/{stage} update")
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        errors = validation(model, valid, features, device)
        if errors[arm] < best:
            best = errors[arm]
            torch.save({"model": model.state_dict(), "epoch": epoch}, path)
        record = {"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_mse": errors,
                  "seconds": time.monotonic() - started, "cached_volumes": len(features.cache)}
        history.append(record)
        write_json(args.output / f"{stage}_history.json", history)
        print(json.dumps({"stage": stage, **record}), flush=True)
    state = torch.load(path, weights_only=True, map_location=device)
    model.load_state_dict(state["model"])
    checked = validation(model, valid, features, device)
    if not np.isclose(checked[arm], best, rtol=1e-6, atol=1e-9):
        raise RuntimeError(f"{stage} checkpoint does not reproduce its validation score")
    return {"epoch": state["epoch"], "validation_mse": checked}


def render(parts, dataset):
    anchor, _, displacement, source, *_ = (value.float() if torch.is_tensor(value) else value for value in parts)
    if dataset == "lumiere":
        # MRI cache is H,W,Z and displacement channels follow reversed cache axes.
        anchor = anchor.permute(0, 1, 4, 2, 3)
        displacement = displacement[:, [1, 2, 0]].permute(0, 1, 4, 2, 3)
        return curvature_corrected_warp(anchor, displacement).permute(0, 1, 3, 4, 2) + source
    return curvature_corrected_warp(anchor, displacement) + source


def summarize(rows):
    result = {}
    for key in rows[0]:
        if key not in ("patient_id", "query", "method"):
            values = [row[key] for row in rows if key in row]
            result[key] = {"mean": float(np.mean(values)), "median": float(np.median(values)), "n": len(values)}
    return result


def witness(dataset):
    records = []
    for item in dataset:
        valid = item["mask"] > 0
        times = item["context_times"][valid]
        if not valid.any() or not times.max() < item["target_times"].min():
            raise ValueError(f"noncausal window for {item['patient_id']}")
        records.append({"patient_id": item["patient_id"], "mask": item["mask"].tolist(),
                        "context_times": item["context_times"].tolist(),
                        "target_times": item["target_times"].tolist(),
                        "data_sha256": data_hash(batch_item(item))})
    return records


@torch.inference_mode()
def test(model, dataset, features, device, args):
    model.eval()
    model.capture_transport = True
    rows = []
    volumes = args.output / "volumes"
    volumes.mkdir(exist_ok=True)
    for index, item in enumerate(dataset):
        batch = batch_item(item)
        with _autocast(device, True):
            result = model(*_input(batch, device), last_features=features.observed(item))
        with _autocast(device, True):
            predictions = torch.stack([render(part, args.dataset)[0, 0] for part in model.transport_parts])
        predictions = predictions.float().cpu().numpy()
        anchor = result["last_context"][0].float().cpu().numpy()
        targets = item["targets"]
        if not np.array_equal(anchor, item["context"][np.flatnonzero(item["mask"])[-1]]):
            raise RuntimeError("renderer anchor differs from the last visible MRI")
        patient = item["patient_id"]
        native = result["predictions"][0].float().cpu().numpy()
        np.savez_compressed(volumes / f"{patient}.npz", prediction=predictions[-1], target=targets[-1],
                            predictions=predictions, targets=targets, last_context=anchor,
                            native_predictions=native, mask=item["mask"],
                            context_times=item["context_times"], target_times=item["target_times"])
        for query, target in enumerate(targets):
            for label, prediction in (("Curvature-corrected composed warp", predictions[query]),
                                      ("DINOv3 routed rollout", native[query]), ("LCI", anchor)):
                rows.append({"patient_id": patient, "query": query + 1, "method": label,
                             **score(prediction, target, args.dataset, device)})
        if index == len(dataset) // 2:
            _save_panel(args.output / "representative_case.png",
                        *(axial(vol, args.dataset) for vol in (predictions[-1], targets[-1], anchor)),
                        patient, args.dataset.upper())
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "per_patient.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {f"query_{query}": {label: summarize([row for row in rows
                                                 if row["query"] == query and row["method"] == label])
                               for label in dict.fromkeys(row["method"] for row in rows)}
            for query in (1, 2)}


def run(args):
    seed_all(42)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    runs = PROJECT_ROOT / "runs"
    upstream = runs / f"trajectory_control_{args.dataset}_seed42_20261117_v1"
    bank_path = upstream / "bank.json"
    checkpoint_path = upstream / "students/control/best.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    manifest = _manifest_hash(args.dataset, args.data_root)
    bank = json.loads(bank_path.read_text())
    if (bank["windows_per_patient"] != 2 or checkpoint["config"]["bank_sha256"] != _sha256(bank_path)
            or checkpoint["config"]["manifest_sha256"] != manifest):
        raise ValueError("training bank and frozen trajectory checkpoint do not match this experiment")
    train = fixed_training_dataset(args.dataset, args.data_root, bank)
    valid = _base_dataset(args.dataset, args.data_root, "validation")
    held_out = _base_dataset(args.dataset, args.data_root, "test")
    train_ids = [case["patient_id"] for case in train.base.cases]
    partitions = [train_ids, [case["patient_id"] for case in valid.cases],
                  [case["patient_id"] for case in held_out.cases]]
    if len(set(sum(partitions, []))) != sum(map(len, partitions)):
        raise ValueError("patient IDs overlap across splits")
    args.output.mkdir(parents=True, exist_ok=args.evaluate_only)
    if not args.evaluate_only:
        shutil.copyfile(__file__, args.output / "source.py")
        write_json(args.output / "config.json", {
            "dataset": args.dataset, "seed": 42, "counts": list(map(len, partitions)),
            "protocol": window_protocol(args.dataset), "train_ids": train_ids,
            "validation": witness(valid), "test": witness(held_out),
            "bank_sha256": _sha256(bank_path), "manifest_sha256": manifest,
            "trajectory_checkpoint": str(checkpoint_path), "trajectory_sha256": _sha256(checkpoint_path),
            "dino_weights": str(WEIGHTS), "dino_sha256": _sha256(WEIGHTS),
            "match_epochs": args.match_epochs, "benefit_epochs": args.benefit_epochs, "lr": args.lr,
            "dino_input": "last visible volume; training targets used only for matching supervision",
            "selection": "lowest validation final-query MSE per stage; benefit includes epoch zero",
            "render": "composed backward map; axial curvature correction; transported source",
        })
    features = Features(args.dataset, device)
    if not args.evaluate_only:
        model = MatchedControl(use_dino=True).to(device)
        model.load_control(checkpoint["model"])
        picked = {}
        for stage in ("matching", "benefit"):
            picked[stage] = train_stage(model, stage, train, valid, features, device, args)
        write_json(args.output / "selection.json", picked)
        del model
    else:
        picked = json.loads((args.output / "selection.json").read_text())
    model = ReplayProbe(use_dino=True).to(device)
    model.load_state_dict(torch.load(args.output / "benefit.pt", weights_only=True, map_location=device)["model"])
    scores = test(model, held_out, features, device, args)
    summary = {"dataset": args.dataset, "counts": list(map(len, partitions)),
               "method_selection": picked, "test": scores}
    write_json(args.output / "summary.json", summary)
    print(json.dumps({"stage": "complete", **summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DEFAULTS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--match-epochs", type=int, default=8)
    parser.add_argument("--benefit-epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if min(args.match_epochs, args.benefit_epochs) < 1 or args.lr <= 0:
        parser.error("epoch counts and learning rate must be positive")
    run(args)


if __name__ == "__main__":
    main()
