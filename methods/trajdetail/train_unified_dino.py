"""Train V1 with a zero-gated frozen-DINO adapter on the fixed ACDC protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from methods.trajdetail.dino_unified_model import DINOHistoryCalibratedTrajectory
from src.trajectory.external_data import ACDCJointDataset, PROJECT_ROOT
from src.trajectory.external_eval import save_evaluation
from src.trajectory.train import seed_all
from src.trajectory.train_unified import trajectory_loss


DEFAULT_CACHE = Path("/mnt/Users/LiangJiaming/Generation_large_files_20261119/acdc_dino_features_8cube")
DINO_WEIGHTS = Path("/mnt/Users/LiangJiaming/SomeKeeper/SPkeeper/DinoV3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")


class DINOACDCDataset(Dataset):
    def __init__(self, base, cache_root: Path):
        self.base = base
        self.cache_root = Path(cache_root)
        self.augment = base.split == "train"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        features = np.load(self.cache_root / f"{item['patient_id']}.npy", mmap_mode="r")
        context_dino = np.zeros((len(item["mask"]), *features.shape[1:]), dtype=np.float32)
        for slot in np.flatnonzero(item["mask"] > 0):
            frame = int(round(float(item["context_times"][slot]) * 11))
            context_dino[slot] = features[frame]
        if self.augment:
            if np.random.random() < 0.5:
                item["context"] = np.flip(item["context"], axis=2)
                item["targets"] = np.flip(item["targets"], axis=2)
                context_dino = np.flip(context_dino, axis=3)
            if np.random.random() < 0.5:
                item["context"] = np.flip(item["context"], axis=3)
                item["targets"] = np.flip(item["targets"], axis=3)
                context_dino = np.flip(context_dino, axis=4)
        item["context"] = np.ascontiguousarray(item["context"])
        item["targets"] = np.ascontiguousarray(item["targets"])
        item["context_dino"] = np.ascontiguousarray(context_dino)
        return item


def _batch_inputs(batch, device):
    return (batch["context"].to(device, non_blocking=True),
            batch["mask"].to(device, non_blocking=True),
            batch["context_times"].to(device, non_blocking=True),
            batch["target_times"].to(device, non_blocking=True),
            batch["context_dino"].to(device, non_blocking=True))


def _sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def predict(model, dataset, device, amp=False):
    model.eval()
    predictions, targets, lasts, ids, counts, times, history_mse = [], [], [], [], [], [], []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=amp and device.type == "cuda"):
            context, mask, context_times, target_times, dino = _batch_inputs(batch, device)
            result = model(context, mask, context_times, target_times, context_dino=dino)
        predictions.append(result["predictions"].float().cpu().numpy())
        targets.append(batch["targets"].numpy())
        lasts.append(result["last_context"].float().cpu().numpy())
        ids.extend(batch["patient_id"])
        counts.extend(batch["mask"].sum(1).int().tolist())
        times.append(batch["target_times"].numpy())
        _, parts = trajectory_loss(result, batch, history_weight=0.0, l1_weight=0.0)
        history_mse.append(float(parts["history"]))
    return {"predictions": np.concatenate(predictions), "targets": np.concatenate(targets),
            "last_context": np.concatenate(lasts), "patient_ids": ids,
            "history_counts": counts, "target_times": np.concatenate(times),
            "history_mse": float(np.mean(history_mse))}


def run(args):
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    device = torch.device(args.device)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_root)
    train_base = ACDCJointDataset(args.data_root, "train", protocol_root=args.protocol_root)
    val_base = ACDCJointDataset(args.data_root, "validation", protocol_root=args.protocol_root)
    test_base = ACDCJointDataset(args.data_root, "test", protocol_root=args.protocol_root)
    training, validation, test = (DINOACDCDataset(base, cache)
                                  for base in (train_base, val_base, test_base))
    model = DINOHistoryCalibratedTrajectory(base_channels=args.base_channels,
        memory_channels=args.memory_channels, max_step=args.max_step,
        dino_channels=args.dino_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))

    source = output / "source_snapshot"
    source.mkdir()
    source_hashes = {}
    for name in ("unified_model.py", "dino_unified_model.py", "train_unified_dino.py",
                 "train_unified.py", "external_data.py", "external_eval.py", "train.py"):
        path = Path(__file__).resolve().parent / name
        shutil.copyfile(path, source / name)
        source_hashes[name] = _sha(path)
    manifest = Path(args.protocol_root) / "data_manifest.json"
    masks = Path(args.protocol_root) / "runs" / "discrete_seed42"
    import torchmetrics
    config = {**vars(args), "model": "HistoryCalibratedTrajectory V1 + zero-gated frozen-DINO adapter",
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_patients": len(training), "validation_patients": len(validation),
        "test_patients": len(test), "source_sha256": source_hashes,
        "manifest_sha256": _sha(manifest), "dino_weights_sha256": _sha(DINO_WEIGHTS),
        "dino_feature_cache": str(cache), "dino_feature_cache_shape_per_patient": [12, args.dino_channels, 8, 8, 8],
        "prediction": "same V1 shared data-time rollout; DINO enters observed memory init/correction only",
        "selection": "minimum fixed-validation latest-target mean MSE; no test selection",
        "augmentation": "identical random flips applied to MRI, targets and cached DINO maps",
        "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                    "torchmetrics": torchmetrics.__version__, "cuda": torch.version.cuda},
        "fixed_masks_sha256": {s: _sha(masks / f"fixed_masks_{s}.json") for s in ("validation", "test")}}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"config": config}), flush=True)

    best, best_epoch, history = float("inf"), None, []
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {key: 0.0 for key in ("total", "future", "history", "primary_mse", "primary_mae", "history_terms")}
        epoch_start = time.monotonic()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=args.amp and device.type == "cuda"):
                context, mask, context_times, target_times, dino = _batch_inputs(batch, device)
                result = model(context, mask, context_times, target_times, context_dino=dino)
                loss, parts = trajectory_loss(result, batch, history_weight=args.history_weight,
                    l1_weight=args.l1_weight, primary_weight=args.primary_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
            optimizer.step()
            sums["total"] += float(loss.detach())
            for key in parts:
                sums[key] += float(parts[key])
        scheduler.step()
        record = {"epoch": epoch, "train": {key: value / len(loader) for key, value in sums.items()},
                  "dino_gate_mean": float(model.dino_gate.detach().mean()),
                  "lr": optimizer.param_groups[0]["lr"],
                  "elapsed_seconds": time.monotonic() - epoch_start}
        if epoch == 1 or epoch % args.val_interval == 0 or epoch == args.epochs:
            values = predict(model, validation, device, amp=args.amp)
            mse = float(np.square(values["predictions"][:, -1] - values["targets"][:, -1]).mean())
            record["validation_primary_mse"] = mse
            record["validation_history_mse"] = values["history_mse"]
            if mse < best:
                best, best_epoch = mse, epoch
                torch.save({"model": model.state_dict(), "config": config,
                    "epoch": epoch, "validation_primary_mse": mse}, output / "best.pt")
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    values = predict(model, test, device, amp=args.amp)
    evaluation = save_evaluation(output / "test_eval", values["predictions"][:, -1],
        values["targets"][:, -1], values["last_context"], values["patient_ids"],
        ssim_data_range=None, title="ACDC V1 + DINO adapter")
    pair_dir = output / "future_pairs"
    pair_dir.mkdir()
    for i, patient in enumerate(values["patient_ids"]):
        np.savez_compressed(pair_dir / f"{patient}.npz", prediction=values["predictions"][i],
            target=values["targets"][i], target_times=values["target_times"][i],
            history_count=values["history_counts"][i])
    result = {"best_epoch": best_epoch, "best_validation_primary_mse": best,
              "test": evaluation, "elapsed_seconds": time.monotonic() - start,
              "final_dino_gate_mean": float(model.dino_gate.detach().mean())}
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/mnt/Datasets/ACDC")
    parser.add_argument("--protocol-root", default=str(PROJECT_ROOT / "cronos_acdc_baseline"))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--memory-channels", type=int, default=8)
    parser.add_argument("--dino-channels", type=int, default=768)
    parser.add_argument("--max-step", type=float, default=1 / 11)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--history-weight", type=float, default=0.25)
    parser.add_argument("--l1-weight", type=float, default=0.01)
    parser.add_argument("--primary-weight", type=float, default=2.0)
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
