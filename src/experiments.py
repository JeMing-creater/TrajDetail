"""Experiment configuration, fold grouping, and selected checkpoint discovery."""
from contextlib import contextmanager
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from src.paths import ROOT, SPLITS_ROOT
from src.release_io import checkpoint, write_json

DATASETS = ("acdc", "lumiere", "isles", "mu", "dlbs")
LEGACY = DATASETS[:3]
NEW = DATASETS[3:]
METHODS = ("copy", "linear", "convlstm", "simvp", "node_lstm", "vivit",
           "cronos", "imageflownet", "brlp", "delta_lfm", "log4d", "trajdetail")
REFERENCES = METHODS[:2]
SIMPLE = METHODS[2:8]
GROUPS = {"legacy": LEGACY, "mu-dlbs": NEW}
DISPLAY = dict(zip(METHODS, ("Last image", "Linear extrapolation", "ConvLSTM", "SimVP",
                            "NODE + LSTM", "ViViT", "CRONOS", "ImageFlowNet", "BrLP",
                            "Δ-LFM (Euler 50)", "4DLoG", "TrajDetail"), strict=True))


def load(path):
    return json.loads(Path(path).read_text())


def group_for(dataset, method):
    if dataset in NEW and method not in REFERENCES:
        return "mu-dlbs"
    if dataset in LEGACY and method == "trajdetail":
        return "legacy"
    return dataset


def members(group):
    return GROUPS.get(group, (group,))


def run_path(method, group, fold, run):
    if not run or not all(c.isascii() and (c.isalnum() or c in "-_") for c in run):
        raise ValueError("Run names may contain ASCII letters, numbers, hyphens and underscores.")
    return ROOT / "methods" / method / "weight" / group / f"fold{fold}" / run


def fold_manifest(dataset, fold):
    return SPLITS_ROOT / f"fold{fold}" / dataset / "manifest.json"


def fingerprint(dataset, fold):
    return hashlib.sha256(fold_manifest(dataset, fold).read_bytes()).hexdigest()


def validate_config(config, path):
    method, group = config["method"], config["group"]
    if method not in METHODS or group not in (*DATASETS, *GROUPS):
        raise ValueError(f"Invalid method or training group in {path}")
    if any(group_for(dataset, method) != group for dataset in members(group)):
        raise ValueError(f"{method} requires joint training on {group_for(members(group)[0], method)}")
    return config


@contextmanager
def exclusive(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def weight_paths(method, group):
    if method in SIMPLE:
        return {"model": f"{method}.pt"}
    if method == "trajdetail":
        readout = "legacy_readout" if group == "legacy" else "shared_readout"
        return {**{f"trajectory_{cohort}": f"{cohort}/"
                   f"{'trajectory/' if group == 'legacy' else ''}benefit.pt" for cohort in members(group)},
                "voxel": f"{readout}/voxel/selected.pt", "detail": f"{readout}/detail/selected.pt"}
    if method == "log4d":
        return {"registration": "4dlog_registration.pt", "diffusion": "4dlog_diffusion.pt"}
    if method == "delta_lfm":
        return {"autoencoder": "ae.pt", "flow": "flow.pt"}
    if method == "brlp":
        return {key: f"brlp_{key}.pt" for key in ("ae", "diffusion", "control")} | {
            "selection": "brlp_selection.json"}
    return {}


def selected_epoch(path):
    state = checkpoint(path)
    return {key: state[key] for key in ("epoch", "step", "validation_mse", "validation_nrmse",
                                       "val", "validation", "val_loss")
            if isinstance(state.get(key), (str, int, float))}


def complete(directory, config, fold, origin="trained"):
    method, group = config["method"], config["group"]
    paths = weight_paths(method, group)
    if method == "brlp" and load(directory / paths["selection"])["use_tuned_decoder"]:
        paths["decoder"] = "decoder.pt"
    weights = {}
    for key, name in paths.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Selected artifact missing: {path}")
        details = selected_epoch(path) if key == "model" else {}
        if key == "model":
            history_path = directory / f"{method}_history.json"
            history = load(history_path)
            metric = next(name for name in ("validation_mse", "validation_nrmse") if name in details)
            selected = min((row for row in history if metric in row), key=lambda row: row[metric])
            if details[metric] != selected[metric] or details["epoch"] != selected["epoch"]:
                raise ValueError(f"{path} does not match the best validation checkpoint")
        weights[key] = dict(path=name, bytes=path.stat().st_size, **details)
    write_json(directory / "complete.json", {
        "status": "complete", "checkpoint_policy": "selected on validation data",
        "origin": origin, "fold": fold, "config": config, "weights": weights,
        "splits": {dataset: fingerprint(dataset, fold) for dataset in members(group)}})


def discover(method, dataset, fold, run):
    group = group_for(dataset, method)
    path = run_path(method, group, fold, run)
    if method in REFERENCES:
        return path, {}
    if not (path / "complete.json").is_file():
        raise FileNotFoundError(f"{method}/{dataset}/fold{fold}: training is incomplete or run '{run}' is absent")
    record = load(path / "complete.json")
    if (record["status"] != "complete" or record["fold"] != fold or
            record["config"]["method"] != method or record["config"]["group"] != group or
            record["splits"].get(dataset) != fingerprint(dataset, fold)):
        raise ValueError(f"Checkpoint manifest does not match requested experiment: {path}")
    expected = weight_paths(method, group)
    if method == "brlp" and load(path / expected["selection"])["use_tuned_decoder"]:
        expected["decoder"] = "decoder.pt"
    if {key: value["path"] for key, value in record["weights"].items()} != expected:
        raise ValueError(f"Incorrect selected checkpoint paths: {path}")
    for value in record["weights"].values():
        artifact = path / value["path"]
        if not artifact.is_file() or artifact.stat().st_size != value["bytes"]:
            raise ValueError(f"Selected checkpoint is missing or damaged: {artifact}")
    return path, record
