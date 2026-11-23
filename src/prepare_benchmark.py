"""Create canonical source caches and verify the frozen set of benchmark patients."""
import argparse
from collections import Counter
from pathlib import Path
import shutil

import numpy as np

from src.experiments import DATASETS, NEW, fold_manifest, load
from src.paths import DATA


def validate_manifests(dataset):
    all_test = []
    population = None
    for fold in range(1, 4):
        rows = load(fold_manifest(dataset, fold))["patients"]
        ids = [row["patient_id"] for row in rows]
        if len(ids) != len(set(ids)) or (population is not None and population != set(ids)):
            raise ValueError(f"Contradictory fold population for {dataset}")
        population = set(ids)
        all_test.extend(row["patient_id"] for row in rows if row["split"] == "test")
    if Counter(all_test) != Counter(population):
        raise ValueError(f"Each {dataset} patient must be held out exactly once")
    return population


def prepare_lumiere(protocol):
    from src.trajectory.external_data import LumiereJointDataset
    from src.release_io import write_json

    path = DATA / "source_cache/lumiere"
    path.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for split in ("train", "validation", "test"):
        source = LumiereJointDataset(protocol, split, max_context=7)
        mapping[split] = []
        for case in source.cases:
            weeks, arrays = zip(*case["visits"], strict=True)
            patient = case["patient_id"]
            filename = f"{patient}.npy"
            np.save(path / filename, np.stack(arrays), allow_pickle=False)
            mapping[split].append(dict(patient_id=patient, file=filename, weeks=weeks))
    write_json(path / "manifest.json", mapping)


def prepare_isles(protocol):
    from src.trajectory.data import ISLESJointDataset

    rows = load(fold_manifest("isles", 1))["patients"]
    source = ISLESJointDataset.__new__(ISLESJointDataset)
    source.root = protocol
    source.augment = False
    output = DATA / "source_cache/isles"
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = Path(row["source"]["cache"])
        local = protocol / name
        if name.is_absolute():
            local = protocol / "cases" / name.name
        volume = source._load({"cache": str(local)})
        np.save(output / name.name, volume, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("--source", type=Path, help="Prepared data directory or raw MU/DLBS source")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    dataset = args.dataset
    population = validate_manifests(dataset)
    if args.check:
        from src.inference import records

        evaluated = {patient for fold in range(1, 4) for patient, *_ in records(dataset, fold)}
        if evaluated != population:
            raise ValueError("Prepared windows differ from the archived folds")
        print(f"{dataset}: validated {len(evaluated)} unique held-out windows")
        return
    if args.source is None:
        parser.error("Provide --source or use --check.")
    if dataset == "lumiere":
        prepare_lumiere(args.source)
    elif dataset == "isles":
        prepare_isles(args.source)
    elif dataset == "acdc":
        path = DATA / "source_cache/acdc"
        path.mkdir(parents=True, exist_ok=True)
        for name in ("trn_dat.npy", "val_dat.npy", "tst_dat.npy"):
            shutil.copyfile(args.source / name, path / name)
    elif dataset in NEW:
        from src.prepare import prepare

        prepare(dataset, args.source, DATA / "longitudinal" / dataset,
                DATA / "preparation_records" / dataset, args.workers)
    print(f"Prepared {dataset} for {len(population)} benchmark patients")


if __name__ == "__main__":
    main()
