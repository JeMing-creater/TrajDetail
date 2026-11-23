"""Frozen ACDC and LUMIERE data adapters for two-future MRI prediction."""

from __future__ import annotations

import json
from src.paths import DATA, PROTOCOL
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path):
    return json.loads(path.read_text())


def _acdc_case_sources(data_root: Path, protocol_root: Path, split: str):
    manifest = _read_json(protocol_root / "data_manifest.json")
    if split == "test":
        ids = manifest["split_patient_ids_one_based"]["test"]
        array = np.load(data_root / "tst_dat.npy", mmap_mode="r", allow_pickle=False)
        return [(array, i, int(patient_id)) for i, patient_id in enumerate(ids)]

    merged_ids = (manifest["intermediate_train_ids_one_based"]
                  + manifest["intermediate_validation_ids_one_based"])
    selected = [(i, int(patient_id)) for i, patient_id in enumerate(merged_ids)
                if (i % 5 != 0 if split == "train" else i % 5 == 0)]
    trn = np.load(data_root / "trn_dat.npy", mmap_mode="r", allow_pickle=False)
    val = np.load(data_root / "val_dat.npy", mmap_mode="r", allow_pickle=False)
    sources = []
    for merged_index, patient_id in selected:
        if merged_index < len(trn):
            sources.append((trn, merged_index, patient_id))
        else:
            sources.append((val, merged_index - len(trn), patient_id))
    return sources


class ACDCJointDataset(Dataset):
    """ACDC cine windows retaining the frozen CRONOS q and history mask."""

    def __init__(self, data_root=None, split="train", *,
                 protocol_root=None, augment=False, limit=None):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown ACDC split: {split}")
        self.data_root = Path(data_root or DATA / "source_cache/acdc")
        self.protocol_root = Path(protocol_root or PROTOCOL / "acdc")
        self.split = split
        self.augment = bool(augment and split == "train")
        self.sources = _acdc_case_sources(self.data_root, self.protocol_root, split)
        if limit is not None:
            self.sources = self.sources[:limit]
        if not self.sources:
            raise ValueError(f"Empty ACDC split: {split}")
        self.frames = 12
        self.context_frames = self.frames - 1
        if split == "train":
            self.windows = None
        else:
            window_path = self.protocol_root / "runs" / "discrete_seed42" / (
                "fixed_masks_validation.json" if split == "validation" else "fixed_masks_test.json")
            stored = _read_json(window_path)
            patient_ids = [int(x) for x in stored["patient_ids_one_based"]]
            if [case[2] for case in self.sources] != patient_ids:
                raise ValueError(f"ACDC {split} patient order differs from frozen masks")
            self.windows = stored["samples"]
            if len(self.windows) != len(self.sources):
                raise ValueError("ACDC frozen mask cardinality mismatch")

    def __len__(self):
        return len(self.sources)

    def _sample_window(self):
        q = int(np.random.randint(7, self.frames))
        visible = q - 6
        mask = np.zeros(q, dtype=np.float32)
        mask[:visible] = np.random.randint(0, 2, size=visible)
        if not mask.any():
            mask[np.random.randint(visible)] = 1.0
        return {"target_idx": q, "missing_mask": mask.tolist()}

    def __getitem__(self, index):
        array, array_index, patient_id = self.sources[index]
        raw = np.asarray(array[array_index], dtype=np.float32)
        if raw.ndim != 4 or raw.shape[0] != 12:
            raise ValueError(f"Unexpected ACDC case shape {raw.shape}")
        # The public ACDC cache stores (T,H,W,D); the CRONOS loader uses (T,D,W,H).
        # Keep a view and copy only the observed and target frames a window uses.
        series = raw.transpose(0, 3, 2, 1)
        window = self._sample_window() if self.split == "train" else self.windows[index]
        q = int(window["target_idx"])
        source_mask = np.asarray(window["missing_mask"], dtype=np.float32)
        if not 7 <= q <= 11 or source_mask.shape != (q,):
            raise ValueError(f"Invalid ACDC frozen window q={q}, mask={source_mask.shape}")

        context = np.zeros((self.context_frames, *series.shape[1:]), dtype=np.float32)
        mask = np.zeros(self.context_frames, dtype=np.float32)
        context_times = np.zeros(self.context_frames, dtype=np.float32)
        offset = self.context_frames - q
        full_times = np.linspace(0, 1, self.frames, dtype=np.float32)
        for frame in range(q):
            slot = offset + frame
            mask[slot] = source_mask[frame]
            context_times[slot] = full_times[frame]
            if source_mask[frame] > 0:
                context[slot] = series[frame]

        targets = series[[q - 1, q]]
        if self.augment:
            for axis in (2, 3):
                if np.random.random() < 0.5:
                    context = np.flip(context, axis=axis)
                    targets = np.flip(targets, axis=axis)
        return {
            "context": np.ascontiguousarray(context),
            "mask": mask,
            "context_times": context_times,
            "target_times": full_times[[q - 1, q]],
            "targets": np.ascontiguousarray(targets),
            "patient_id": f"patient{patient_id}",
            "primary_index": q,
        }


def _normalize_lumiere(image):
    image = np.asarray(image, dtype=np.float32)
    lower, upper = np.percentile(image, [1, 99])
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("LUMIERE image is constant or non-finite")
    clipped = np.clip(image, lower, upper)
    return ((clipped - clipped.min()) / (clipped.max() - clipped.min())).astype(np.float32)


def _visit_key_order(key):
    try:
        return (0, int(key))
    except (TypeError, ValueError):
        return (1, str(key))


def _unique_visits(case):
    by_week = {}
    for key, (image, _seg, week) in case.items():
        week = float(week)
        current = by_week.get(week)
        if current is None or _visit_key_order(key) < _visit_key_order(current[0]):
            by_week[week] = (key, image)
    return [(week, _normalize_lumiere(by_week[week][1])) for week in sorted(by_week)]


class LumiereJointDataset(Dataset):
    """Irregular-visit LUMIERE windows, deduplicated by clinical week."""

    def __init__(self, data_root=None, split="train", *, max_context=7,
                 augment=False, limit=None):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown LUMIERE split: {split}")
        self.data_root = Path(data_root or DATA / "lumiere_protocol")
        self.split = split
        self.max_context = int(max_context)
        self.augment = bool(augment and split == "train")
        if self.max_context < 1:
            raise ValueError("max_context must be positive")
        manifest = _read_json(self.data_root / "patient_split_manifest.json")
        split_cases = [case for case in manifest["cases"] if case["split"] == split]
        split_cases.sort(key=lambda case: int(case["split_index"]))
        if limit is not None:
            split_cases = split_cases[:limit]

        split_arrays = {}
        self.cases = []
        for case_meta in split_cases:
            npy_split = case_meta["npy_split"]
            if npy_split not in split_arrays:
                split_arrays[npy_split] = np.load(
                    self.data_root / f"{npy_split}_lumiere.npy", allow_pickle=True)
            raw_case = split_arrays[npy_split][int(case_meta["split_index"])]
            visits = _unique_visits(raw_case)
            if len(visits) >= 3:
                self.cases.append({"patient_id": case_meta["patient_id"], "visits": visits})

        self.examples = list(range(len(self.cases)))
        if not self.examples:
            raise ValueError(f"No eligible LUMIERE examples in split: {split}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        case_index = self.examples[index]
        case = self.cases[case_index]
        visits = case["visits"]
        target_index = (int(np.random.randint(2, len(visits))) if self.split == "train"
                        else len(visits) - 1)
        history = visits[:target_index - 1][-self.max_context:]
        future = visits[target_index - 1:target_index + 1]
        shape = future[0][1].shape
        context = np.zeros((self.max_context, *shape), dtype=np.float32)
        mask = np.zeros(self.max_context, dtype=np.float32)
        context_times = np.zeros(self.max_context, dtype=np.float32)
        offset = self.max_context - len(history)
        for i, (week, image) in enumerate(history):
            context[offset + i] = image
            mask[offset + i] = 1.0
            context_times[offset + i] = week / 255.0
        targets = np.stack([image for _week, image in future]).astype(np.float32)
        target_times = np.asarray([week / 255.0 for week, _image in future], dtype=np.float32)
        if self.augment:
            for axis in (2, 3):
                if np.random.random() < 0.5:
                    context = np.flip(context, axis=axis)
                    targets = np.flip(targets, axis=axis)
        return {
            "context": np.ascontiguousarray(context),
            "mask": mask,
            "context_times": context_times,
            "target_times": target_times,
            "targets": np.ascontiguousarray(targets),
            "patient_id": case["patient_id"],
            "primary_index": target_index,
        }


__all__ = ["ACDCJointDataset", "LumiereJointDataset"]
