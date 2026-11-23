"""Run the four CRONOS comparison architectures on ACDC or LUMIERE.

The model definitions are loaded from the existing ISLES adaptation without
copying or modifying that source.  This module only adapts the two public
trajectory loaders to the common ``(B,T,1,D,H,W)`` model contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.trajectory.control_external import fixed_training_dataset, make_bank
from src.trajectory.external_data import ACDCJointDataset, LumiereJointDataset
from src.trajectory.external_eval import save_evaluation


ROOT = Path(__file__).resolve().parent
MODEL_SOURCE = ROOT.parent / "cronos_isles24_baseline" / "paper_baselines" / "models.py"
MODEL_NAMES = ("convlstm", "simvp", "node_lstm", "vivit")
DEFAULTS = {
    "acdc": Path("/mnt/Datasets/ACDC"),
    "lumiere": ROOT.parent / "cronos_lumiere_baseline" / "paper_protocol_data",
}
DEFAULT_ACDC_PROTOCOL = ROOT.parent / "cronos_acdc_baseline"
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 4
DEFAULT_VAL_INTERVAL = 10


def _load_model_module():
    spec = importlib.util.spec_from_file_location("cronos_isles_paper_models", MODEL_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load baseline model source: {MODEL_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(name: str, volume_shape, *, num_frames: int, width: int = 32):
    """Build one existing CRONOS comparison architecture."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    return _load_model_module().build_forecaster(
        name, volume_shape=tuple(volume_shape), width=width, num_frames=num_frames,
    )


def adapt_item(item: Mapping, dataset_name: str) -> dict:
    """Convert one external loader item to the baseline volume axis order.

    ACDC is already ``(T,D,H,W)`` after the frozen loader transpose.  The
    LUMIERE cache is ``(T,H,W,D)``; its last axis is moved to depth so both
    datasets satisfy the model's ``(D,H,W)`` contract.  ``targets[-1]`` is the
    primary q target and no target frame is inserted into context.
    """
    if dataset_name not in {"acdc", "lumiere"}:
        raise ValueError(f"unknown dataset {dataset_name!r}")
    result = dict(item)
    for key in ("context", "targets"):
        value = np.asarray(result[key], dtype=np.float32)
        if dataset_name == "lumiere":
            if value.ndim == 4:
                value = value.transpose(0, 3, 1, 2)
            elif value.ndim == 3:
                value = value.transpose(2, 0, 1)
            else:
                raise ValueError(f"unexpected LUMIERE {key} shape {value.shape}")
        result[key] = np.ascontiguousarray(value)
    result["mask"] = np.asarray(result["mask"], dtype=np.float32)
    result["context_times"] = np.asarray(result["context_times"], dtype=np.float32)
    result["target_times"] = np.asarray(result["target_times"], dtype=np.float32)
    valid_times = result["context_times"][result["mask"].astype(bool)]
    if valid_times.size and not np.all(valid_times < result["target_times"][0]):
        raise ValueError("future target time leaked into context")
    if result["targets"].shape[0] != 2:
        raise ValueError("expected joint q-1,q targets")
    return result


def primary_target(item: Mapping) -> np.ndarray:
    """Return q only; q-1 remains available solely for protocol diagnostics."""
    targets = np.asarray(item["targets"], dtype=np.float32)
    if targets.ndim != 4 or targets.shape[0] != 2:
        raise ValueError(f"expected targets shaped (2,D,H,W), got {targets.shape}")
    return targets[-1]


def model_batch(items_or_batch, device: torch.device):
    """Create model inputs and the q target from a list or DataLoader batch."""
    if isinstance(items_or_batch, Mapping):
        context = torch.as_tensor(items_or_batch["context"], dtype=torch.float32)
        targets = torch.as_tensor(items_or_batch["targets"], dtype=torch.float32)
        context_times = torch.as_tensor(items_or_batch["context_times"], dtype=torch.float32)
        target_times = torch.as_tensor(items_or_batch["target_times"], dtype=torch.float32)
    else:
        context = torch.as_tensor(np.stack([item["context"] for item in items_or_batch]), dtype=torch.float32)
        targets = torch.as_tensor(np.stack([item["targets"] for item in items_or_batch]), dtype=torch.float32)
        context_times = torch.as_tensor(np.stack([item["context_times"] for item in items_or_batch]), dtype=torch.float32)
        target_times = torch.as_tensor(np.stack([item["target_times"] for item in items_or_batch]), dtype=torch.float32)
    if context.ndim != 5 or targets.ndim != 5:
        raise ValueError(
            "expected collated context (B,T,D,H,W), targets (B,2,D,H,W); "
            f"got {tuple(context.shape)}, {tuple(targets.shape)}"
        )
    context = context.unsqueeze(2).to(device, non_blocking=True)
    target = targets[:, -1:].to(device, non_blocking=True)
    times = torch.cat((context_times, target_times[:, -1:]), dim=1).to(device, non_blocking=True)
    return context, times, target


def _last_context(items_or_batch):
    context = torch.as_tensor(items_or_batch["context"], dtype=torch.float32)
    mask = torch.as_tensor(items_or_batch["mask"], dtype=torch.float32)
    if context.ndim == 4:
        context = context.unsqueeze(0)
        mask = mask.unsqueeze(0)
    result = []
    for sample, sample_mask in zip(context, mask):
        valid = torch.where(sample_mask > 0)[0]
        result.append(sample[valid[-1]] if len(valid) else torch.zeros_like(sample[0]))
    return torch.stack(result)


class AdaptedDataset(Dataset):
    def __init__(self, base: Dataset, dataset_name: str):
        self.base = base
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return adapt_item(self.base[index], self.dataset_name)


class CachedACDC(Dataset):
    """Keep ACDC source volumes in RAM while preserving window sampling.

    The public ACDC arrays are memory-mapped.  Re-reading the same patient for
    every epoch makes the baseline benchmark I/O-bound; copying only the source
    cases once leaves the train-time random missing-mask logic unchanged.
    """

    def __init__(self, base: ACDCJointDataset):
        self.base = base
        self.sources = []
        for source, index, patient_id in base.sources:
            cached = np.asarray(source[index:index + 1], dtype=np.float32).copy()
            self.sources.append((cached, 0, patient_id))
        self.base.sources = self.sources

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return self.base[index]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _dataset_bundle(dataset_name: str, data_root: Path, protocol_root: Path):
    if dataset_name == "acdc":
        train = ACDCJointDataset(data_root, "train", protocol_root=protocol_root)
        val = ACDCJointDataset(data_root, "validation", protocol_root=protocol_root)
        test = ACDCJointDataset(data_root, "test", protocol_root=protocol_root)
        return train, val, test, 11, (32, 128, 128)
    bank = make_bank("lumiere", data_root=data_root, windows_per_patient=2, seed=42)
    train = fixed_training_dataset("lumiere", data_root, bank)
    val = LumiereJointDataset(data_root, "validation", max_context=7)
    test = LumiereJointDataset(data_root, "test", max_context=7)
    return train, val, test, 7, (64, 96, 96), bank


def _loader(dataset, batch_size, *, shuffle, device, seed=42):
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == "cuda", generator=generator)


@torch.no_grad()
def _predict(model, dataset, device, batch_size):
    model.eval()
    predictions, targets, lasts, ids = [], [], [], []
    for batch in _loader(dataset, batch_size, shuffle=False, device=device):
        context, times, target = model_batch(batch, device)
        prediction = model(context, times)
        predictions.append(prediction[:, 0].float().cpu().numpy())
        targets.append(target[:, 0].float().cpu().numpy())
        lasts.append(_last_context(batch).numpy())
        ids.extend(batch["patient_id"])
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(lasts), ids


@torch.no_grad()
def _validation_mse(model, dataset, device, batch_size):
    prediction, target, _last, _ids = _predict(model, dataset, device, batch_size)
    return float(np.mean((prediction - target) ** 2))


def _config(dataset_name, model_name, data_root, protocol_root, output, seed, epochs,
            batch_size, width, context_frames, volume_shape, source_hashes,
            bank=None, val_interval=DEFAULT_VAL_INTERVAL):
    config = {
        "dataset": dataset_name,
        "model_name": model_name,
        "seed": seed,
        "epochs": epochs,
        "validation_interval": val_interval,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": 1.0e-4,
        "weight_decay": 0.01,
        "scheduler": "CosineAnnealingLR",
        "selection": "lowest fixed-validation primary q MSE; test untouched",
        "context_frames": context_frames,
        "volume_shape": list(volume_shape),
        "primary_target": "targets[-1] (q); targets[0] (q-1) is never used as supervision",
        "protocol": (
            "ACDC frozen validation/test masks; random training mask; context slots=11"
            if dataset_name == "acdc" else
            "LUMIERE seed42 fixed training bank, 46 patients x 2 windows; latest q window for validation/test; context slots=7"
        ),
        "data_root": str(data_root.resolve()),
        "protocol_root": str(protocol_root.resolve()),
        "output": str(output.resolve()),
        "width": width,
        "source_sha256": source_hashes,
    }
    if bank is not None:
        config["training_bank_sha256"] = hashlib.sha256(
            json.dumps(bank, sort_keys=True).encode()).hexdigest()
        config["training_bank_windows"] = len(bank["windows"])
    if dataset_name == "acdc":
        config["fixed_masks_sha256"] = {
            split: _sha256(protocol_root / "runs" / "discrete_seed42" / f"fixed_masks_{split}.json")
            for split in ("validation", "test")
        }
        config["manifest_sha256"] = _sha256(protocol_root / "data_manifest.json")
    else:
        config["manifest_sha256"] = _sha256(data_root / "patient_split_manifest.json")
    return config


def run(args):
    """Train one baseline and evaluate it; this function is intentionally opt-in."""
    if args.model not in MODEL_NAMES:
        raise ValueError(f"unknown model {args.model!r}; expected one of {MODEL_NAMES}")
    _seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=False)
    data_root = Path(args.data_root or DEFAULTS[args.dataset]).resolve()
    protocol_root = Path(args.protocol_root or DEFAULT_ACDC_PROTOCOL).resolve()
    bundle = _dataset_bundle(args.dataset, data_root, protocol_root)
    train_base, val_base, test_base, context_frames, volume_shape = bundle[:5]
    bank = bundle[5] if args.dataset == "lumiere" else None
    if args.dataset == "acdc":
        train_base, val_base, test_base = (CachedACDC(train_base), CachedACDC(val_base),
                                           CachedACDC(test_base))
    train = AdaptedDataset(train_base, args.dataset)
    val = AdaptedDataset(val_base, args.dataset)
    test = AdaptedDataset(test_base, args.dataset)
    model = build_model(args.model, volume_shape, num_frames=context_frames, width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    source_files = [Path(__file__), MODEL_SOURCE]
    for name in ("external_data.py", "control_external.py", "external_eval.py"):
        source_files.append(ROOT.parent / "patient_trajectory" / name)
    source_hashes = {str(path.resolve()): _sha256(path) for path in source_files}
    config = _config(args.dataset, args.model, data_root, protocol_root, output, args.seed,
                     args.epochs, args.batch_size, args.width, context_frames,
                     volume_shape, source_hashes, bank, args.val_interval)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if bank is not None:
        (output / "bank.json").write_text(json.dumps(bank, indent=2) + "\n")
    best, best_epoch = float("inf"), None
    history = []
    loader = _loader(train, args.batch_size, shuffle=True, device=device, seed=args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            context, times, target = model_batch(batch, device)
            prediction = model(context, times)
            loss = (prediction - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        scheduler.step()
        record = {"epoch": epoch, "train_mse": total / max(1, len(loader))}
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            record["validation_mse"] = _validation_mse(model, val, device, args.batch_size)
            if record["validation_mse"] < best:
                best, best_epoch = record["validation_mse"], epoch
                torch.save({"model_state_dict": model.state_dict(), "config": config,
                            "epoch": epoch, "validation_mse": best}, output / "best.pt")
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if best_epoch is None:
        raise RuntimeError("no validation checkpoint was selected")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    prediction, target, last, ids = _predict(model, test, device, args.batch_size)
    test_result = save_evaluation(output / "test_eval", prediction, target, last, ids,
                                  title=f"{args.dataset.upper()} {args.model} forecast")
    result = {"dataset": args.dataset, "model": args.model, "best_epoch": best_epoch,
              "best_validation_mse": best, "test": test_result}
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("acdc", "lumiere"), required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--protocol-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--val-interval", type=int, default=DEFAULT_VAL_INTERVAL)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
