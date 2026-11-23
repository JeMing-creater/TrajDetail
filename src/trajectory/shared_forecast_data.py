"""Build causal forecast inputs on the frozen ACDC, LUMIERE and ISLES windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from methods.trajdetail.audit_acdc_sharpness import ReplayProbe
from src.trajectory.control_external import _base_dataset, _sha256, fixed_training_dataset, make_bank as make_external_bank
from src.trajectory.external_data import ACDCJointDataset, PROJECT_ROOT
from methods.trajdetail.train_external_curvature_prototype import Features, render
from methods.trajdetail.train_local_control import last_features
from src.trajectory.train_unified import _autocast, _input
from methods.trajdetail.train_unified_dino import DEFAULT_CACHE
from src.trajectory.validate_control import FixedWindows, batch_item, data_hash, json_hash, make_bank as make_acdc_bank, write_json


DATASETS = ("acdc", "lumiere", "isles")
RUNS = PROJECT_ROOT / "runs"
CHANNELS = ("last", "composed", "rollout", "previous", "trend", "source")


def source_paths(dataset):
    if dataset == "acdc":
        return (RUNS / "trajectory_control_acdc_seed42_20261117_v3/bank.json",
                RUNS / "acdc_curvature_prototype_20261121/benefit.pt")
    return (RUNS / f"trajectory_control_{dataset}_seed42_20261117_v1/bank.json",
            RUNS / f"curvature_prototype_{dataset}_seed42_20261121/benefit.pt")


def load_dataset(dataset, split, bank):
    if dataset == "acdc":
        return FixedWindows(bank) if split == "train" else ACDCJointDataset(split=split)
    return fixed_training_dataset(dataset, None, bank) if split == "train" else _base_dataset(dataset, None, split)


def to_axial(value, dataset):
    return value.movedim(-1, -3) if dataset == "lumiere" else value


def temporal_predictors(context, times, queries):
    """Least squares temporal slope shrunk when little history is available."""
    anchor = context[-1:]
    previous = context[-2:-1] if len(context) > 1 else anchor
    elapsed = queries - times[-1]
    centered = times - times.mean()
    interval = (times[-1] - times[0]).clamp_min(0.)
    coefficients = centered / (centered.square().sum() + .02 ** 2)
    slope = (context * coefficients[:, None, None, None]).sum(0, keepdim=True)
    delta = slope * torch.minimum(elapsed, interval).view(-1, 1, 1, 1)
    return anchor, previous, delta.clamp(-.25, .25), elapsed, interval


@torch.inference_mode()
def prepare(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    bank_path, checkpoint = source_paths(args.dataset)
    bank = json.loads(bank_path.read_text())
    model = ReplayProbe(use_dino=True).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.capture_transport = True
    feature_cache = Features(args.dataset, device) if args.dataset != "acdc" else None
    if args.dataset == "acdc":
        config = json.loads((checkpoint.parent / "upstream_config.json").read_text())
        if config["bank_sha256"] != json_hash(bank):
            raise ValueError("ACDC replay checkpoint uses a different training bank")
    else:
        config = json.loads((checkpoint.parent / "config.json").read_text())
        if config["bank_sha256"] != _sha256(bank_path):
            raise ValueError(f"{args.dataset} replay checkpoint uses a different training bank")
    if args.extra_seed is not None:
        if args.dataset == "acdc":
            bank = make_acdc_bank(windows_per_patient=args.extra_windows, seed=args.extra_seed)
        else:
            bank = make_external_bank(args.dataset, windows_per_patient=args.extra_windows, seed=args.extra_seed)
    for split in args.splits:
        started = time.monotonic()
        dataset = load_dataset(args.dataset, split, bank)
        out = args.output / args.dataset / split
        out.mkdir(parents=True, exist_ok=False)
        stub = dataset[0]["targets"]
        shape = (stub.shape[-1], *stub.shape[-3:-1]) if args.dataset == "lumiere" else stub.shape[-3:]
        x = np.lib.format.open_memmap(out / "inputs.npy", mode="w+", dtype="float16",
                                     shape=(len(dataset), 2, len(CHANNELS), *shape))
        # Keep reference voxel intensities exactly as loaded for the reported metrics.
        y = np.lib.format.open_memmap(out / "targets.npy", mode="w+", dtype="float32",
                                     shape=(len(dataset), 2, *shape))
        metadata = []
        for index, item in enumerate(dataset):
            batch = batch_item(item)
            context, mask, times, queries = _input(batch, device)
            visible = mask[0].bool()
            features = (feature_cache.observed(item) if feature_cache is not None
                        else last_features(item, DEFAULT_CACHE, device))
            with _autocast(device, True):
                result = model(context, mask, times, queries, last_features=features)
            anchor, previous, trend, elapsed, interval = temporal_predictors(
                context[0, visible], times[0, visible], queries[0])
            for query, parts in enumerate(model.transport_parts):
                curved = render(parts, args.dataset)[0]
                values = torch.cat((anchor, curved, result["predictions"][:, query],
                                    previous, trend[query:query + 1], parts[3][0].float()))
                x[index, query] = to_axial(values, args.dataset).cpu().numpy()
            y[index] = to_axial(batch["targets"][0], args.dataset).numpy()
            metadata.append({
                "patient_id": item["patient_id"], "window_sha256": data_hash(batch),
                "mask": item["mask"].tolist(), "context_times": item["context_times"].tolist(),
                "target_times": item["target_times"].tolist(),
                "elapsed": elapsed.tolist(), "interval": float(interval), "visible": int(visible.sum()),
            })
            if (index + 1) % 30 == 0 or index + 1 == len(dataset):
                print(json.dumps({"dataset": args.dataset, "split": split, "completed": index + 1,
                                  "total": len(dataset), "seconds": round(time.monotonic() - started, 1)}),
                      flush=True)
        x.flush()
        y.flush()
        write_json(out / "manifest.json", {"dataset": args.dataset, "split": split, "channels": CHANNELS,
                   "bank_sha256": _sha256(bank_path), "checkpoint_sha256": _sha256(checkpoint),
                   "sampled_bank_sha256": json_hash(bank), "extra_seed": args.extra_seed,
                   "windows_per_patient": bank['windows_per_patient'] if split == 'train' else 1,
                   "windows": metadata})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--splits", choices=("train", "validation", "test"), nargs="+",
                        default=("train", "validation", "test"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extra-seed", type=int)
    parser.add_argument("--extra-windows", type=int, default=4)
    args = parser.parse_args()
    if args.extra_seed is not None and (args.splits != ['train'] or args.extra_windows < 1):
        parser.error("additional windows are allowed only in training")
    prepare(args)


if __name__ == "__main__":
    main()
