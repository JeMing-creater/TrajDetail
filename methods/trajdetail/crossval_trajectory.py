"""Fit TrajDetail's upstream and shared readouts inside each fold."""

import argparse
import hashlib
from pathlib import Path
from src.paths import PROTOCOL
from types import SimpleNamespace

import numpy as np
import torch

from src.compare import save_rows
from src.crossval_data import LEGACY, RUN, SPLITS, LegacyWindows, read, write
from methods.trajdetail.proposed import MILESTONE, run_one
from methods.trajdetail.scratch_trajectory import fit_control
from methods.trajdetail.audit_acdc_sharpness import ReplayProbe
from methods.trajdetail.evaluate_spectral_calibration import model_from
from methods.trajdetail.matched_control import MatchedControl
from src.trajectory.shared_forecast_data import CHANNELS, temporal_predictors, to_axial
from methods.trajdetail.shared_forecast_model import SharedForecast
from src.trajectory.train import seed_all
from methods.trajdetail.train_budgeted_detail import fit
from methods.trajdetail.train_external_curvature_prototype import Features, render, score, train_stage
from methods.trajdetail.train_shared_forecast import TrainingPool, Windows, train
from src.trajectory.train_unified import _autocast, _input
from src.trajectory.validate_control import batch_item, data_hash


def load_partitions(root, dataset):
    datasets = {split: LegacyWindows(root, dataset, split) for split in SPLITS}
    source = datasets["train"].source
    for windows in datasets.values():
        windows.source = source
    return datasets


def upstream(root, dataset, device, settings=None):
    path = root / dataset / "trajectory"
    path.mkdir(exist_ok=True)
    if (path / "complete.json").exists():
        return
    args = SimpleNamespace(output=path, dataset=dataset, device=str(device),
                           **(settings or dict(control_epochs=60, teacher_steps=150, match_epochs=8, benefit_epochs=6, lr=.001)))
    partitions = load_partitions(root, dataset)
    training, validation = (partitions[split] for split in SPLITS[:2])
    write(path / "recipe.json", {"seed": 42, "trainable_initialization": "random",
                                "dataset": dataset, "train_ids": [row["patient_id"] for row in training.rows],
                                "validation_ids": [row["patient_id"] for row in validation.rows],
                                **{key: value for key, value in vars(args).items() if key != "output"}})
    seed_all(42)
    fit_control(training, validation, args, device, windows_per_patient=2)
    seed_all(42)
    model = MatchedControl(use_dino=True).to(device)
    model.load_control(torch.load(path / "control.pt", map_location=device, weights_only=True)["model"])
    features = Features(dataset, device)
    selected = {stage: train_stage(model, stage, training, validation, features, device, args)
                for stage in ("matching", "benefit")}
    write(path / "complete.json", selected)


@torch.inference_mode()
def extract(root, dataset, device, splits=("train", "validation", "test")):
    checkpoint = root / dataset / "trajectory/benefit.pt"
    bank_hash = hashlib.sha256((root / dataset / "manifest.json").read_bytes()).hexdigest()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = ReplayProbe(use_dino=True).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.capture_transport = True
    features = Features(dataset, device)
    datasets = load_partitions(root, dataset)
    for branch, split, extra in (("primary", "train", False), ("extra", "train", True),
                                 ("primary", "validation", False), ("primary", "test", False)):
        if split not in splits:
            continue
        path = root / "causal_features" / branch / dataset / split
        if (path / "manifest.json").exists():
            continue
        path.mkdir(parents=True, exist_ok=True)
        windows = datasets[split]
        if split == "train":
            windows.n = 6 if extra else 2
            indices = ([j for j in range(len(windows)) if j % 6 >= 2] if extra else range(len(windows)))
        else:
            indices = range(len(windows))
        shape = to_axial(torch.as_tensor(windows[0]["targets"][0]), dataset).shape
        x = np.lib.format.open_memmap(path / "inputs.npy", mode="w+", dtype="float16",
                                     shape=(len(indices), 2, len(CHANNELS), *shape))
        y = np.lib.format.open_memmap(path / "targets.npy", mode="w+", dtype="float32",
                                     shape=(len(indices), 2, *shape))
        records = []
        for position, index in enumerate(indices):
            item = windows[index]
            batch = batch_item(item)
            context, mask, times, queries = _input(batch, device)
            visible = mask[0].bool()
            with _autocast(device, True):
                result = model(context, mask, times, queries, last_features=features.observed(item))
            anchor, previous, trend, elapsed, interval = temporal_predictors(
                context[0, visible], times[0, visible], queries[0])
            for query, parts in enumerate(model.transport_parts):
                combined = torch.cat((anchor, render(parts, dataset)[0], result["predictions"][:, query],
                                      previous, trend[query:query + 1], parts[3][0].float()))
                x[position, query] = to_axial(combined, dataset).cpu().numpy()
            y[position] = to_axial(batch["targets"][0], dataset).numpy()
            records.append({"patient_id": item["patient_id"], "window_sha256": data_hash(batch),
                            "mask": item["mask"].tolist(), "context_times": item["context_times"].tolist(),
                            "target_times": item["target_times"].tolist(), "elapsed": elapsed.tolist(),
                            "interval": float(interval)})
        x.flush()
        y.flush()
        write(path / "manifest.json", {"dataset": dataset, "split": split, "channels": CHANNELS,
              "windows_per_patient": (4 if extra else 2) if split == "train" else 1,
              "bank_sha256": bank_hash, "checkpoint_sha256": checkpoint_hash,
              "sampled_bank_sha256": bank_hash if extra else None, "windows": records})
        print(dataset, branch, split, len(records), flush=True)


def shared_readout(root, device, feature_cache=None, phases=(30, 32, 16)):
    source = feature_cache or root / "causal_features"
    cache = source / "primary"
    extra = source / "extra"
    destination = root / "legacy_readout"
    config = read(PROTOCOL / "trajdetail.json")["selected"]
    settings = dict(cache=cache, extra_cache=None, device=str(device), datasets=LEGACY,
                    slices=10, width=20, hf_weight=.25, validate_every=2, unbalanced=False,
                    decoder_only=False, test=False)
    for name, prior, epochs, lr in (("base", None, phases[0], .0006), ("voxel", "base", phases[1], .00032)):
        path = destination / name
        complete = path / "completed.json"
        if complete.exists():
            continue
        if path.exists():
            path.rename(path.with_name(f"{name}.incomplete.{path.stat().st_mtime_ns}"))
        seed_all(42)
        args = SimpleNamespace(**settings, output=path, initialize=(destination / prior / "selected.pt" if prior else None),
                               epochs=epochs, lr=lr)
        if prior:
            args.extra_cache = extra
        train(args, device)
        write(complete, {"epochs": epochs, "seed": 42})

    detail_dir = destination / "detail"
    voxel = model_from(destination / "voxel/selected.pt", device)
    detail = SharedForecast(20).to(device=device, memory_format=torch.channels_last)
    if not (detail_dir / "completed.json").exists():
        if detail_dir.exists():
            detail_dir.rename(detail_dir.with_name(f"detail.incomplete.{detail_dir.stat().st_mtime_ns}"))
        detail_dir.mkdir()
        seed_all(42)
        detail.load_state_dict(voxel.state_dict())
        training = [TrainingPool(cache, dataset, extra) for dataset in LEGACY]
        valid = [Windows(cache, dataset, "validation") for dataset in LEGACY]
        fit(detail, training, valid, SimpleNamespace(output=detail_dir, epochs=phases[2],
                    lr=.00018, slices=10, confidence=0., validate_every=2), device)
        write(detail_dir / "completed.json", {"epochs": phases[2], "seed": 42})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", type=int, choices=(1, 2, 3))
    parser.add_argument("action", choices=("upstream", "readout"))
    parser.add_argument("--dataset", choices=LEGACY)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    root = args.root / f"fold{args.fold}"
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.action == "upstream":
        if args.dataset is None:
            parser.error("upstream requires --dataset")
        upstream(root, args.dataset, device)
        extract(root, args.dataset, device)
    else:
        shared_readout(root, device, args.feature_cache)


if __name__ == "__main__":
    main()
