"""ACDC validation run for DINOv3-conditioned unbalanced feature transport."""
from __future__ import annotations

import argparse
import os
from src.paths import DATA, ROOT
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.trajectory.external_data import ACDCJointDataset, PROJECT_ROOT
from src.trajectory.external_eval import save_evaluation
from src.trajectory.train import seed_all
from methods.trajdetail.transport_model import DINOUnbalancedTransport


WEIGHTS = Path(os.environ.get("TRAJDETAIL_DINO_WEIGHTS", DATA / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"))
DINO_REPO = Path(os.environ.get("TRAJDETAIL_DINO_REPO", ROOT / "methods/trajdetail/upstream"))


def load_dino(device):
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Set TRAJDETAIL_DINO_WEIGHTS to your DINOv3 ViT-B/16 weights: {WEIGHTS}")
    sys.path.insert(0, str(DINO_REPO))
    from dinov3.hub.backbones import dinov3_vitb16
    model = dinov3_vitb16(weights=str(WEIGHTS)).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_volume(model, volume: np.ndarray, device):
    """Axial DINOv3 features pooled to an 8-cubed feature grid."""
    slices = torch.from_numpy(np.asarray(volume, dtype=np.float32))
    slices = F.interpolate(slices[:, None], size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    chunks = []
    for start in range(0, len(slices), 16):
        x = slices[start:start + 16].to(device).expand(-1, 3, -1, -1)
        x = (x - mean) / std
        chunks.append(model.get_intermediate_layers(x, n=1, reshape=True)[0].float())
    features = torch.cat(chunks, dim=0).permute(1, 0, 2, 3)[None]
    features = F.adaptive_avg_pool3d(features, (8, 8, 8))[0]
    return features.half().cpu().numpy()


def prepare_feature_cache(datasets, cache_root: Path, device):
    cache_root.mkdir(parents=True, exist_ok=True)
    missing = []
    for dataset in datasets:
        for _, _, patient in dataset.sources:
            if not (cache_root / f"patient{patient}.npy").exists():
                missing.append((dataset, int(patient)))
    if not missing:
        return {"cached_patients": len(list(cache_root.glob("patient*.npy"))), "created": 0}
    model = load_dino(device)
    created = 0
    seen = set()
    for dataset, patient in missing:
        if patient in seen:
            continue
        seen.add(patient)
        source = next(row for row in dataset.sources if int(row[2]) == patient)
        array, array_index, _ = source
        raw = np.asarray(array[array_index], dtype=np.float32)
        if raw.shape != (12, 128, 128, 32):
            raise ValueError(f"Unexpected ACDC cached patient shape {raw.shape}")
        # Match ACDCJointDataset: cached T,H,W,D -> model D,H,W.
        series = np.ascontiguousarray(raw.transpose(0, 3, 2, 1))
        per_frame = [encode_volume(model, frame, device) for frame in series]
        np.save(cache_root / f"patient{patient}.npy", np.stack(per_frame).astype(np.float16))
        created += 1
        if created % 10 == 0:
            print(json.dumps({"dino_cache_created": created, "remaining": len(seen) - created}), flush=True)
    del model
    torch.cuda.empty_cache()
    return {"cached_patients": len(list(cache_root.glob("patient*.npy"))), "created": created}


class CachedDinoDataset(Dataset):
    def __init__(self, base, cache_root):
        self.base = base
        self.cache_root = Path(cache_root)
        self.augment = base.split == "train"
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        batch = self.base[index]
        patient = batch["patient_id"]
        feature = np.load(self.cache_root / f"{patient}.npy", mmap_mode="r")
        valid = np.flatnonzero(batch["mask"] > 0)
        if not len(valid):
            raise ValueError(f"No observed frames for {patient}")
        source_time = float(batch["context_times"][valid[-1]])
        source_index = int(round(source_time * 11))
        target_indices = np.rint(batch["target_times"] * 11).astype(np.int64)
        batch["source_dino"] = np.asarray(feature[source_index], dtype=np.float32)
        batch["target_dino"] = np.asarray(feature[target_indices], dtype=np.float32)
        batch["source_time"] = np.float32(source_time)
        if self.augment:
            if np.random.random() < 0.5:
                batch["context"] = np.flip(batch["context"], axis=2)
                batch["targets"] = np.flip(batch["targets"], axis=2)
                batch["source_dino"] = np.flip(batch["source_dino"], axis=2)
                batch["target_dino"] = np.flip(batch["target_dino"], axis=3)
            if np.random.random() < 0.5:
                batch["context"] = np.flip(batch["context"], axis=3)
                batch["targets"] = np.flip(batch["targets"], axis=3)
                batch["source_dino"] = np.flip(batch["source_dino"], axis=3)
                batch["target_dino"] = np.flip(batch["target_dino"], axis=4)
            for key in ("context", "targets", "source_dino", "target_dino"):
                batch[key] = np.ascontiguousarray(batch[key])
        return batch


def tensor(batch, key, device):
    return torch.as_tensor(batch[key], device=device)


def step_loss(model, batch, device, train=True):
    source = tensor(batch, "context", device)
    mask = tensor(batch, "mask", device)
    positions = torch.arange(source.shape[1], device=device)[None].expand_as(mask)
    source_index = torch.where(mask > 0, positions, -1).max(1).values
    image = source[torch.arange(source.shape[0], device=device), source_index][:, None]
    dino = tensor(batch, "source_dino", device)
    targets = tensor(batch, "targets", device)
    target_dino = tensor(batch, "target_dino", device)
    source_time = tensor(batch, "source_time", device)
    target_times = tensor(batch, "target_times", device)
    losses, outputs = [], []
    for k in range(2):
        delta = (target_times[:, k] - source_time).clamp_min(0)
        result = model(image, dino, delta)
        image_loss = (result["prediction"][:, 0] - targets[:, k]).square().mean()
        latent_target = model.project(target_dino[:, k].float())
        latent_loss = F.mse_loss(result["future_tokens"], latent_target.detach())
        losses.append(image_loss + 0.05 * latent_loss)
        outputs.append(result)
    loss = 0.5 * (losses[0] + 2.0 * losses[1]) / 1.5
    return loss, outputs


@torch.inference_mode()
def evaluate(model, dataset, device):
    model.eval()
    predictions, targets, lasts, ids, times = [], [], [], [], []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        loss, results = step_loss(model, batch, device, train=False)
        predictions.append(results[-1]["prediction"][:, 0].float().cpu().numpy())
        targets.append(batch["targets"][:, -1].numpy())
        context = batch["context"].numpy()
        valid = np.flatnonzero(batch["mask"].numpy()[0] > 0)
        lasts.append(context[:, valid[-1]])
        ids.extend(batch["patient_id"])
        times.append(batch["target_times"].numpy())
    return {"predictions": np.concatenate(predictions), "targets": np.concatenate(targets),
            "last_context": np.concatenate(lasts), "patient_ids": ids,
            "target_times": np.concatenate(times)}


def run(args):
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    device = torch.device(args.device)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_base = ACDCJointDataset(args.data_root, "train", protocol_root=args.protocol_root, augment=False)
    val_base = ACDCJointDataset(args.data_root, "validation", protocol_root=args.protocol_root)
    test_base = ACDCJointDataset(args.data_root, "test", protocol_root=args.protocol_root)
    cache = Path(args.cache_root)
    cache_result = prepare_feature_cache((train_base, val_base, test_base), cache, device)
    training = CachedDinoDataset(train_base, cache)
    validation = CachedDinoDataset(val_base, cache)
    test = CachedDinoDataset(test_base, cache)
    model = DINOUnbalancedTransport().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(training, batch_size=1, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    cfg = {**vars(args), "model": "DINOv3 + feature-predicted unbalanced Sinkhorn transport + residual signal decoder",
           "parameters": sum(p.numel() for p in model.parameters()),
           "split": {"train": len(training), "validation": len(validation), "test": len(test)},
           "dino_weights": str(WEIGHTS), "dino_feature_cache": str(cache), "dino_cache": cache_result,
           "transport": {"grid": 8, "epsilon": 0.08, "rho": 0.12, "sinkhorn_iterations": 12},
           "protocol": "frozen ACDC CRONOS q/masks; q-1 and q supervised; q primary; whole-volume metrics"}
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(json.dumps({"config": cfg}), flush=True)
    best, best_epoch, history = float("inf"), None, []
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = step_loss(model, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.monotonic() - start}
        if epoch == 1 or epoch % args.val_interval == 0 or epoch == args.epochs:
            values = evaluate(model, validation, device)
            mse = float(np.square(values["predictions"] - values["targets"]).mean())
            record["validation_primary_mse"] = mse
            if mse < best:
                best, best_epoch = mse, epoch
                torch.save({"model": model.state_dict(), "epoch": epoch, "validation_mse": mse}, output / "best.pt")
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    values = evaluate(model, test, device)
    metrics = save_evaluation(output / "test_eval", values["predictions"],
        values["targets"], values["last_context"], values["patient_ids"],
        ssim_data_range=None, title="ACDC DINOv3 unbalanced transport forecast")
    result = {"best_epoch": best_epoch, "best_validation_primary_mse": best,
              "test": metrics, "elapsed_seconds": time.monotonic() - start}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/mnt/Datasets/ACDC")
    parser.add_argument("--protocol-root", default=str(PROJECT_ROOT / "cronos_acdc_baseline"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", default="/mnt/Users/LiangJiaming/Generation_large_files_20261119/acdc_dino_features_8cube")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
