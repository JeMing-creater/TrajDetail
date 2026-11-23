"""Rebuild both shared readout branches from the new cohorts' trajectories."""

import argparse
import hashlib
import json
from pathlib import Path
from src.paths import PROTOCOL
from types import SimpleNamespace

import numpy as np
import torch

from methods.trajdetail.proposed import MILESTONE, run_one
from src.compare import save_rows
from src.report import case_manifest
from methods.trajdetail.scratch_trajectory import windows
from methods.trajdetail.audit_acdc_sharpness import ReplayProbe
from methods.trajdetail.evaluate_spectral_calibration import model_from
from src.trajectory.shared_forecast_data import CHANNELS, temporal_predictors
from methods.trajdetail.shared_forecast_model import SharedForecast
from src.trajectory.train import seed_all
from methods.trajdetail.train_budgeted_detail import fit
from methods.trajdetail.train_external_curvature_prototype import Features, render, score
from methods.trajdetail.train_shared_forecast import TrainingPool, Windows, train
from src.trajectory.train_unified import _autocast, _input
from src.trajectory.validate_control import batch_item, write_json

DATASETS = ("mu", "dlbs")


@torch.inference_mode()
def prepare(root, raw_cache, readout_cache, device, splits):
    features = Features("lumiere", device)
    for dataset in DATASETS:
        output = root / dataset
        manifest, _ = case_manifest(output)
        checkpoint = output / "benefit.pt"
        model = ReplayProbe(use_dino=True).to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
        model.capture_transport = True
        features.cache.clear()
        for split in splits:
            items = windows(manifest, raw_cache / dataset, split)
            path = readout_cache / dataset / split
            path.mkdir(parents=True, exist_ok=True)
            shape = tuple(manifest["shape_zyx"])
            x = np.lib.format.open_memmap(path / "inputs.npy", mode="w+", dtype="float16",
                                         shape=(len(items), 1, len(CHANNELS), *shape))
            y = np.lib.format.open_memmap(path / "targets.npy", mode="w+", dtype="float32",
                                         shape=(len(items), 1, *shape))
            rows = []
            for i, item in enumerate(items):
                batch = batch_item(item)
                context, mask, times, queries = _input(batch, device)
                with _autocast(device, True):
                    prediction = model(context, mask, times, queries, last_features=features.observed(item))
                anchor, previous, trend, elapsed, interval = temporal_predictors(context[0], times[0], queries[0])
                parts, = model.transport_parts
                streams = (anchor, render(parts, "lumiere")[0], prediction["predictions"][:, 0],
                           previous, trend, parts[3][0].float())
                x[i, 0] = torch.cat(streams).movedim(-1, -3).cpu().numpy()
                y[i] = np.moveaxis(item["targets"], -1, -3)
                rows.append({"patient_id": item["patient_id"], "mask": item["mask"].tolist(),
                             "context_times": item["context_times"].tolist(),
                             "target_times": item["target_times"].tolist(), "elapsed": elapsed.tolist(),
                             "interval": float(interval)})
            x.flush()
            y.flush()
            write_json(path / "manifest.json", {"dataset": dataset, "split": split, "channels": CHANNELS,
                       "windows_per_patient": 1, "windows": rows,
                       "bank_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
                       "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
            print(dataset, split, "causal cache", len(rows), flush=True)


@torch.inference_mode()
def evaluate(detail, voxel, root, readout_cache, device, config):
    for dataset in DATASETS:
        output = root / dataset
        held_out = Windows(readout_cache, dataset, "test")
        rows = []
        for index, case in enumerate(held_out.patients()):
            _, _, corrected = run_one(detail, voxel, held_out, index, device, config)
            estimate = corrected.cpu().numpy()
            rows.append({"method": "our_method", "patient_id": case,
                         **score(estimate, held_out.targets[index, 0], dataset, device)})
            if index == len(held_out.rows) // 2:
                np.save(output / "our_method_example.npy", estimate)
        save_rows(output / "our_method_test.csv", rows)
        print(dataset, "our_method TEST", np.mean([r["nrmse"] for r in rows]), "n", len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_cache", type=Path)
    parser.add_argument("readout_cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--base-epochs", type=int, default=30)
    parser.add_argument("--voxel-epochs", type=int, default=32)
    parser.add_argument("--detail-epochs", type=int, default=16)
    train_pipeline(parser.parse_args())


def train_pipeline(args):
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    settings = dict(cache=args.readout_cache, extra_cache=None, device=args.device, datasets=DATASETS,
                    slices=10, width=20, hf_weight=.25, validate_every=2, unbalanced=False,
                    decoder_only=False, test=False)
    joint = args.output / "shared_readout"
    joint.mkdir(exist_ok=True)
    cfg_path = PROTOCOL / "trajdetail.json"
    config = json.loads(cfg_path.read_text())["selected"]
    write_json(joint / "recipe.json", {"architecture": str(MILESTONE), "high_frequency_policy": config,
               "trainable_initialization": "random and subsequent same-run stages",
               "base_epochs": args.base_epochs, "voxel_epochs": args.voxel_epochs,
               "detail_epochs": args.detail_epochs, "domains": DATASETS, "queries": 1})
    prepare(args.output, args.raw_cache, args.readout_cache, device, ("train", "validation"))
    seed_all(42)
    base_dir, voxel_dir = joint / "base", joint / "voxel"
    train(SimpleNamespace(**settings, initialize=None, epochs=args.base_epochs, lr=.0006,
                           output=base_dir), device)
    seed_all(42)
    train(SimpleNamespace(**settings, initialize=base_dir / "selected.pt", epochs=args.voxel_epochs,
                           lr=.00032, output=voxel_dir), device)

    detail_dir = joint / "detail"
    detail_dir.mkdir(exist_ok=False)
    seed_all(42)
    detail = SharedForecast(20).to(device=device, memory_format=torch.channels_last)
    checkpoint = torch.load(voxel_dir / "selected.pt", map_location=device, weights_only=True)
    detail.load_state_dict(checkpoint["model"])
    training = [TrainingPool(args.readout_cache, name, None) for name in DATASETS]
    valid = [Windows(args.readout_cache, name, "validation") for name in DATASETS]
    selected = fit(detail, training, valid, SimpleNamespace(output=detail_dir, epochs=args.detail_epochs,
                    lr=.00018, slices=10, confidence=0., validate_every=2), device)
    detail.load_state_dict(selected["model"])
    detail.eval()
    voxel = model_from(voxel_dir / "selected.pt", device)
    return joint


if __name__ == "__main__":
    main()
