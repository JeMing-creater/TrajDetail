#!/usr/bin/env python3
"""Reconstruct the repository's historical ISLES image preprocessing.

This deliberately does not call the maintained isles_prepare.py: that script
builds a newer 112x112 ROI dataset. The historical paper_process branch in
isles_loader.py creates 128x128x16x32 volumes but fails to save them because
self.data_raw assignment is indented in the other branch. This independent
script preserves the historical image operations and makes a patient manifest.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def normalize_percentiles(data, lower, upper):
    lo, hi = np.percentile(data, [lower, upper])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("Constant or non-finite CTP series cannot be normalized")
    x = np.clip(data, lo, hi)
    return ((x - x.min()) / (x.max() - x.min())).astype(np.float32)


def preprocess_ctp(source, target_shape=(128, 128, 16, 32)):
    img = nib.load(str(source))
    raw_shape = tuple(img.shape)
    if len(raw_shape) != 4 or min(raw_shape) < 2:
        raise ValueError(f"Expected nonempty 4-D CTP: {source}, {raw_shape}")
    data = img.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite CTP: {source}")
    data = normalize_percentiles(data, 2, 98)
    data[data > 0.8] = 0
    foreground = np.max(data, axis=3) > 0.02
    positions = np.where(foreground)
    if not positions[0].size:
        raise ValueError(f"Empty CTP foreground: {source}")
    bounds = [(int(x.min()), int(x.max()) + 1) for x in positions]
    slices = tuple(slice(a, b) for a, b in bounds)
    data = data[slices + (slice(None),)]
    result = zoom(data, np.asarray(target_shape) / data.shape, order=3,
                  prefilter=True).astype(np.float32)
    if result.shape != tuple(target_shape) or not np.isfinite(result).all():
        raise ValueError(f"Invalid resampled CTP: {source}, {result.shape}")
    return result, {"raw_shape": list(raw_shape), "raw_zooms": list(map(float, img.header.get_zooms())),
                    "foreground_bbox": bounds, "cache_shape": list(result.shape)}


def discover_cases(raw_root):
    raw_root = Path(raw_root).resolve()
    candidates = sorted(raw_root.rglob("*ctp.nii.gz"))
    cases = {}
    for source in candidates:
        # The supplied root may be an extraction root containing both raw and
        # registered derivatives. The historical pipeline used raw_data.
        if "derivatives" in source.relative_to(raw_root).parts:
            continue
        patient_parts = [p for p in source.parent.parts if p.startswith("sub-")]
        if not patient_parts:
            raise ValueError(f"Cannot determine BIDS patient ID: {source}")
        patient = patient_parts[-1]
        if patient in cases:
            raise ValueError(f"More than one raw CTP series for {patient}: {cases[patient]}, {source}")
        cases[patient] = source
    if not cases:
        raise ValueError(f"No raw *ctp.nii.gz found below {raw_root}")
    return cases


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
            "missing_mask": mask.tolist()}


def build_evaluation_samples(cases, seed=42, frames=7, distance=3):
    result = {"seed": seed, "num_context": frames, "context_target_distance": distance,
              "protocol": "repository legacy: 7 context slots + separate future target; gap3", "cases": {}}
    # These independent RNGs keep each split stable regardless of construction
    # order, model initialization, global RNG usage, and training workers.
    for split in ("val", "tst"):
        rng = np.random.RandomState(seed)
        for case in sorted((c for c in cases if c["split"] == split), key=lambda c: c["patient_id"]):
            result["cases"][case["patient_id"]] = {"split": split, **sample_legacy_window(rng, frames, distance)}
    return result


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def _prepare_case(task):
    patient, source, output, target_shape, split = task
    stat = source.stat()
    source_identity = {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cache = output / "cases" / f"{patient}.npy"
    sidecar = cache.with_suffix(".json")
    meta = None
    if cache.is_file() and sidecar.is_file():
        old = json.loads(sidecar.read_text())
        if old.get("source") == source_identity and old.get("cache_shape") == list(target_shape):
            if _sha256(cache) == old.get("cache_sha256"):
                meta = old
    if meta is None:
        arr, meta = preprocess_ctp(source, target_shape=target_shape)
        temp = cache.with_suffix(".npy.tmp")
        with temp.open("wb") as f:
            np.save(f, arr, allow_pickle=False)
        temp.replace(cache)
        meta.update({"source": source_identity, "cache_sha256": _sha256(cache)})
        _write_json(sidecar, meta)
    return {"patient_id": patient, "split": split,
            "cache": str(cache.relative_to(output)), **meta}


def prepare_dataset(raw_root, output, split_counts=(92, 23, 34), seed=42,
                    target_shape=(128, 128, 16, 32), workers=1):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    output = Path(output).resolve()
    sources = discover_cases(raw_root)
    counts = tuple(map(int, split_counts))
    if len(counts) != 3 or min(counts) < 1 or sum(counts) != len(sources):
        raise ValueError(f"Expected {sum(counts)} cases for split {counts}, found {len(sources)}")
    if len(target_shape) != 4 or target_shape[-1] != 32:
        raise ValueError("Historical pipeline requires a 32-frame temporal cache")
    (output / "cases").mkdir(parents=True, exist_ok=True)
    ids = sorted(sources)
    permutation = np.random.RandomState(seed).permutation(len(ids))
    split_ids = {}
    start = 0
    for split, n in zip(("trn", "val", "tst"), counts):
        for idx in permutation[start:start + n]:
            split_ids[ids[int(idx)]] = split
        start += n
    records = []
    # Only small paths/metadata cross the process boundary. Each unique patient
    # owns its cache/sidecar; ordered map preserves the historical record order.
    tasks = [(patient, sources[patient], output, target_shape, split_ids[patient]) for patient in ids]
    with (ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))
          if workers > 1 else nullcontext(None)) as executor:
        results = executor.map(_prepare_case, tasks) if executor else map(_prepare_case, tasks)
        for n, record in enumerate(results, 1):
            records.append(record)
            print(f"[{n}/{len(ids)}] {record['patient_id']} {record['split']} {record['cache_shape']}", flush=True)
    manifest = {
        "dataset": "ISLES24", "raw_root": str(Path(raw_root).resolve()), "seed": seed,
        "split_counts": dict(zip(("trn", "val", "tst"), counts)), "cases": records,
        "original_author_patient_split_available": False,
        "protocol": {"implementation_reference": "Longitudinal4DMed 856a136 isles_loader.py paper_process",
                     "preprocess_percentiles": [2, 98], "zero_above_normalized_intensity": 0.8,
                     "foreground_threshold": 0.02, "target_shape_hwdt": list(target_shape),
                     "resample": "scipy.ndimage.zoom order3 prefilter=True (including temporal axis)",
                     "loader_percentiles": [1, 99], "context_slots": 7, "separate_target": True,
                     "context_target_distance": 3, "segmentation": "all-ones placeholder; lambda_roi_seg=0",
                     "paper_text_ambiguity": "Sec4.1 says 7 points last-as-target but also context T=7; repository uses 7+1",
                     "cache_bug_avoided": "historical paper_process branch never assigns self.data_raw"},
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "evaluation_samples.json", build_evaluation_samples(records, seed=seed))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4, help="Independent case preprocessing processes (default: 4)")
    p.add_argument("--split-counts", nargs=3, type=int, default=(92, 23, 34), metavar=("TRN", "VAL", "TST"))
    args = p.parse_args()
    prepare_dataset(args.raw_root, args.output, tuple(args.split_counts), args.seed, workers=args.workers)


if __name__ == "__main__":
    main()
