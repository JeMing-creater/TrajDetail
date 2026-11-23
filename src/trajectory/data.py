"""RAM-cached ISLES benchmark windows with a second supervised future image.

The primary target, masks and normalized voxels match the existing CRONOS
benchmark. This deliberately inherits that benchmark's preprocessing limits.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from torch.utils.data import Dataset

BASELINE_ROOT = Path(__file__).resolve().parents[2] / "cronos_isles24_baseline"


def normalize_percentiles(data, lower, upper):
    lo, hi = np.percentile(data, [lower, upper])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Constant or non-finite series cannot be normalized")
    clipped = np.clip(data, lo, hi)
    return ((clipped - clipped.min()) / (clipped.max() - clipped.min())).astype(np.float32)


def sample_legacy_window(rng, frames=7, distance=3):
    if not 1 <= distance < frames:
        raise ValueError("distance must be between 1 and frames-1")
    target_idx = int(rng.randint(frames, frames * 3))
    last_context = frames - distance
    mask = np.zeros(frames, dtype=np.float32)
    mask[:last_context] = rng.randint(0, 2, size=last_context)
    mask[0] = 1
    mask[last_context] = 1
    return {"target_idx": target_idx, "context_indices": list(range(target_idx - frames, target_idx)),
            "missing_mask": mask}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ISLESJointDataset(Dataset):
    def __init__(self, root="/mnt/Datasets/ISLES24/prepared_paper", split="trn",
                 augment=False, limit=None, preload_workers=4):
        self.root = Path(root)
        self.split = split
        self.augment = augment and split == "trn"
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.windows = json.loads((self.root / "evaluation_samples.json").read_text())
        self.cases = [c for c in self.manifest["cases"] if c["split"] == split]
        if limit is not None:
            self.cases = self.cases[:limit]
        if not self.cases:
            raise ValueError(f"Empty dataset {split}")
        with ThreadPoolExecutor(max_workers=preload_workers) as pool:
            self.volumes = list(pool.map(self._load, self.cases))
        print(f"RAM cache {split}: {len(self.cases)} patients, "
              f"{sum(x.nbytes for x in self.volumes) / 2**30:.3f} GiB", flush=True)

    def _load(self, case):
        arr = np.load(self.root / case["cache"], mmap_mode="r", allow_pickle=False)
        if arr.shape != (128, 128, 16, 32):
            raise ValueError(f"Unexpected volume shape {arr.shape}")
        return np.ascontiguousarray(normalize_percentiles(arr, 1, 99).transpose(3, 2, 1, 0))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        case = self.cases[index]
        window = (sample_legacy_window(np.random, 7, 3) if self.split == "trn"
                  else self.windows["cases"][case["patient_id"]])
        return self.window_item(index, window)

    def window_item(self, index, window):
        q = int(window["target_idx"])
        indices = np.asarray(window["context_indices"], dtype=np.int64)
        mask = np.asarray(window["missing_mask"], dtype=np.float32)
        if not np.array_equal(indices, np.arange(q - 7, q)):
            raise ValueError("Changed primary benchmark window")
        if mask.shape != (7,) or mask[4] != 1 or mask[5:].any():
            raise ValueError("Invalid frozen gap3 mask")
        data = self.volumes[index]
        # q-1 is a supervised future, never an observed conditioning image.
        context = data[indices] * mask[:, None, None, None]
        targets = data[[q - 1, q]]
        if self.augment:
            for axis in (2, 3):
                if np.random.random() < 0.5:
                    context = np.flip(context, axis=axis)
                    targets = np.flip(targets, axis=axis)
        times = np.linspace(0, 1, 32, dtype=np.float32)
        return {
            "context": np.ascontiguousarray(context),
            "mask": mask.copy(),
            "context_times": times[indices],
            "target_times": times[[q - 1, q]],
            "targets": np.ascontiguousarray(targets),
            "patient_id": self.cases[index]["patient_id"],
            "primary_index": q,
        }
