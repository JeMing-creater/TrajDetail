"""Patient-level folds and the existing causal windows for five cohorts."""

import argparse
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from src.paths import DATA, SPLITS_ROOT

from src.trajectory.control_external import DEFAULTS, _base_dataset, _lumiere_window_item
from src.trajectory.data import ISLESJointDataset, sample_legacy_window
from src.trajectory.external_data import ACDCJointDataset, PROJECT_ROOT


LEGACY = ("acdc", "lumiere", "isles")
NEW = ("mu", "dlbs")
DATASETS = LEGACY + NEW
SPLITS = ("train", "validation", "test")
RUN = SPLITS_ROOT
ORIGINAL = PROJECT_ROOT / "runs/new_longitudinal_scratch_seed42_20261122"
RAW_CACHE = DATA / "longitudinal"
LOCAL_SOURCE_CACHE = DATA / "source_cache"


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rng_for(*parts):
    key = hashlib.sha256("/".join(map(str, (42, *parts))).encode()).digest()
    return np.random.RandomState(int.from_bytes(key[:4], "little"))


def patient_rows(dataset):
    if dataset in NEW:
        return read(ORIGINAL / dataset / "manifest.json")["patients"]
    if dataset == "acdc":
        rows = []
        for split in SPLITS:
            ds = ACDCJointDataset(split=split)
            for index, source in enumerate(ds.sources):
                window = None if ds.windows is None else ds.windows[index]
                rows.append(dict(patient_id=f"patient{source[2]}", split=split,
                                 source_index=index, evaluation=window))
        return rows
    if dataset == "lumiere":
        return [dict(patient_id=case["patient_id"], split=split, source_index=index,
                     visits=len(case["visits"]))
                for split in SPLITS
                for index, case in enumerate(lumiere_base(split).cases)]
    manifest = read(DEFAULTS["isles"] / "manifest.json")
    windows = read(DEFAULTS["isles"] / "evaluation_samples.json")["cases"]
    name = dict(trn="train", val="validation", tst="test")
    return [dict(patient_id=row["patient_id"], split=name[row["split"]],
                 source=row, evaluation=windows.get(row["patient_id"])) for row in manifest["cases"]]


@lru_cache(maxsize=3)
def lumiere_base(split):
    return _base_dataset("lumiere", None, split)


def sample_window(dataset, row, rng, evaluation=False):
    if evaluation and row.get("evaluation") is not None:
        return row["evaluation"]
    if dataset == "acdc":
        q = int(rng.randint(7, 12))
        mask = np.zeros(q, dtype=int)
        mask[:q - 6] = rng.randint(0, 2, size=q - 6)
        if not mask.any():
            mask[rng.randint(q - 6)] = 1
        return dict(target_idx=q, missing_mask=mask.tolist())
    if dataset == "isles":
        window = sample_legacy_window(rng, 7, 3)
        return {key: np.asarray(value).tolist() for key, value in window.items()}
    if dataset == "lumiere":
        return row["visits"] - 1 if evaluation else int(rng.randint(2, row["visits"]))
    return None


def partition(dataset, rows):
    rows = sorted(rows, key=lambda row: row["patient_id"])
    ids = [row["patient_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{dataset}: repeated patient ID")
    folds = np.array_split(rng_for(dataset, "outer").permutation(len(rows)), 3)
    for fold, test in enumerate(folds, 1):
        development = np.setdiff1d(np.arange(len(rows)), test)
        development = rng_for(dataset, fold, "inner").permutation(development)
        n_validation = max(1, round(.2 * len(development)))
        groups = (development[n_validation:], development[:n_validation], test)
        selected = []
        for split, indices in zip(SPLITS, groups, strict=True):
            for index in sorted(indices):
                row = rows[index]
                generator = rng_for(dataset, row["patient_id"], "eval")
                windows = [sample_window(dataset, row, generator, evaluation=True)]
                if split == "train" and dataset in LEGACY:
                    generator = rng_for(dataset, fold, row["patient_id"], "train")
                    windows = [sample_window(dataset, row, generator) for _ in range(6)]
                selected.append({**row, "source_split": row["split"], "split": split, "windows": windows})
        yield fold, selected


def prepare(root):
    for dataset in DATASETS:
        rows = patient_rows(dataset)
        manifest = read(ORIGINAL / dataset / "manifest.json") if dataset in NEW else {}
        tests = []
        for fold, selected in partition(dataset, rows):
            path = root / f"fold{fold}" / dataset / "manifest.json"
            counts = {split: sum(row["split"] == split for row in selected) for split in SPLITS}
            data = {**manifest, "dataset": dataset, "outer_fold": fold,
                    "split_seed": 42, "counts": counts, "patients": selected}
            if path.exists():
                if read(path) != data:
                    raise ValueError(f"Fold already exists with different contents: {path}")
            else:
                write(path, data)
            print(dataset, fold, *counts.values(), flush=True)
            tests.extend(row["patient_id"] for row in selected if row["split"] == "test")
        if sorted(tests) != sorted(row["patient_id"] for row in rows):
            raise ValueError(f"{dataset}: held-out coverage differs from source population")


class LegacySource:
    def __init__(self, dataset, cache):
        self.dataset = dataset
        local = LOCAL_SOURCE_CACHE / dataset
        use_local = local.is_dir()
        self.cache = local if use_local else cache / dataset
        if dataset == "acdc":
            options = {"data_root": local} if use_local else {}
            self.original = {split: ACDCJointDataset(split=split, **options) for split in SPLITS}
        if dataset == "isles":
            self.isles = ISLESJointDataset.__new__(ISLESJointDataset)
            self.isles.root = DEFAULTS["isles"]
            self.isles.augment = False

    @lru_cache(maxsize=160)
    def ctp(self, path):
        cached = self.cache / Path(path).name
        if cached.exists():
            return np.load(cached, mmap_mode="r", allow_pickle=False)
        return self.isles._load({"cache": path})

    @lru_cache(maxsize=160)
    def lumiere(self, split, index):
        mapping = self.cache / "manifest.json"
        if not mapping.exists():
            return lumiere_base(split).cases[index]
        entry = read(mapping)[split][index]
        arrays = np.load(self.cache / entry["file"], mmap_mode="r", allow_pickle=False)
        return {"patient_id": entry["patient_id"],
                "visits": list(zip(entry["weeks"], arrays, strict=True))}

    def item(self, row, window):
        if self.dataset == "acdc":
            ds = copy.copy(self.original[row["source_split"]])
            ds.sources = [ds.sources[row["source_index"]]]
            ds.split, ds.windows = "test", [window]
            item = ds[0]
        elif self.dataset == "lumiere":
            from types import SimpleNamespace

            case = self.lumiere(row["source_split"], row["source_index"])
            item = _lumiere_window_item(SimpleNamespace(cases=[case], max_context=7), 0, window)
        else:
            self.isles.cases = [row["source"]]
            self.isles.volumes = [self.ctp(row["source"]["cache"])]
            item = self.isles.window_item(0, window)
        observed = item["context_times"][item["mask"].astype(bool)]
        if item["patient_id"] != row["patient_id"] or observed.max() >= item["target_times"].min():
            raise ValueError(f"Invalid evaluation window for {row['patient_id']}")
        return item


class LegacyWindows:
    def __init__(self, root, dataset, split, train_windows=2, dynamic=False):
        self.source = LegacySource(dataset, root.parent / "source_cache")
        manifest = read(root / dataset / "manifest.json")
        self.rows = [row for row in manifest["patients"] if row["split"] == split]
        self.dataset = dataset
        self.n = train_windows if split == "train" else 1
        self.dynamic = dynamic and split == "train"

    def __len__(self):
        return len(self.rows) * self.n

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        patient, slot = divmod(index, self.n)
        row = self.rows[patient]
        window = (sample_window(self.dataset, row, np.random) if self.dynamic
                  else row["windows"][slot])
        return self.source.item(row, window)


def cache_lumiere(root):
    cache = root / "source_cache/lumiere"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "manifest.json"
    if manifest.exists():
        return
    entries = {}
    for split in SPLITS:
        base = lumiere_base(split)
        entries[split] = []
        for case in base.cases:
            patient = case["patient_id"]
            filename = f"{patient}.npy"
            weeks, images = zip(*case["visits"], strict=True)
            if len(weeks) != len(images):
                raise ValueError(f"{patient}: volume metadata mismatch")
            with (cache / (filename + ".tmp")).open("wb") as stream:
                np.save(stream, np.stack(images), allow_pickle=False)
            (cache / (filename + ".tmp")).replace(cache / filename)
            entries[split].append({"patient_id": patient, "file": filename, "weeks": weeks})
        print("LUMIERE source cache", split, len(entries[split]), flush=True)
    write(manifest, entries)


def cache_isles(root):
    source = LegacySource("isles", root / "source_cache")
    source.cache.mkdir(parents=True, exist_ok=True)
    rows = patient_rows("isles")
    for index, row in enumerate(rows, 1):
        relative = row["source"]["cache"]
        path = source.cache / Path(relative).name
        if not path.is_file():
            array = source.isles._load({"cache": relative})
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            temporary.replace(path)
        if index % 25 == 0 or index == len(rows):
            print("ISLES'24 source cache", index, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--cache-lumiere", action="store_true")
    parser.add_argument("--cache-isles", action="store_true")
    args = parser.parse_args()
    if args.cache_lumiere:
        cache_lumiere(args.output)
    elif args.cache_isles:
        cache_isles(args.output)
    else:
        prepare(args.output)
