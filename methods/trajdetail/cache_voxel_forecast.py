"""Replay frozen voxel trajectories on both queries with matched patient inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from src.trajectory.control_external import _base_dataset
from src.trajectory.external_data import ACDCJointDataset
from src.trajectory.shared_forecast_data import DATASETS
from methods.trajdetail.train_shared_forecast import Windows
from src.trajectory.validate_control import batch_item, data_hash, write_json


def provenance(dataset):
    if dataset == "isles":
        root = Path("cronos_isles24_baseline/runs/discrete_seed42").resolve()
        report = json.loads((root / "test_best.json").read_text())
        return Path(report["checkpoint"]), root / "source_snapshot", report["checkpoint_sha256"]
    root = Path(f"runs/patient_trajectory_{dataset}_seed42_joint_primary/cronos_matched")
    report = json.loads((root / "comparison_metadata.json").read_text())
    return Path(report["checkpoint"]), Path(report["source_snapshot"]), report["checkpoint_sha256"]


@torch.inference_mode()
def prepare(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    checkpoint, repo, expected = provenance(args.dataset)
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Checkpoint digest changed for {args.dataset}")
    sys.path.insert(0, str(repo / "src"))
    from methods.cronos import CRONOS

    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = dict(weights["args"])
    shape = tuple(config.pop("in_shape"))
    config.update(device=args.device, batch_size=1, num_workers=0)
    model = CRONOS(in_shape=shape, feature_size=config["base_channels"], **config).to(device)
    model.load_state_dict(weights["model_state_dict"], strict=True)
    model.eval()
    for split in ("validation", "test"):
        dataset = (ACDCJointDataset(split=split) if args.dataset == "acdc"
                   else _base_dataset(args.dataset, None, split))
        windows = Windows(args.cache, args.dataset, split)
        if len(windows.rows) != len(dataset):
            raise ValueError("Cached window count differs")
        directory = args.output / args.dataset / split
        directory.mkdir(parents=True, exist_ok=False)
        result = np.lib.format.open_memmap(directory / "forecast.npy", mode="w+", dtype=np.float32,
                                           shape=windows.targets.shape)
        for ix, row in enumerate(windows.rows):
            item = dataset[ix]
            batch = batch_item(item)
            if row["patient_id"] != item["patient_id"] or data_hash(batch) != row["window_sha256"]:
                raise ValueError(f"Voxel trajectory forecast received a different {split} window for {row['patient_id']}")
            observed = batch["context"].to(device).unsqueeze(2)
            times = batch["context_times"].to(device)
            queries = batch["target_times"].to(device)
            for q in (0, 1):
                pred = model.validation_step(observed, time_points=torch.cat((times, queries[:, q:q+1]), dim=1))
                data = pred.reshape(tuple(item["targets"].shape[-3:])).cpu().numpy()
                result[ix, q] = np.moveaxis(data, -1, 0) if args.dataset == "lumiere" else data
            if (ix + 1) % 10 == 0:
                print(f"{args.dataset} {split} {ix + 1}/{len(dataset)}", flush=True)
        result.flush()
        write_json(directory / "manifest.json", {"windows": windows.rows, "checkpoint_sha256": digest})
        print(f"{args.dataset} {split} complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args())
