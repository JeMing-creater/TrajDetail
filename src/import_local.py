"""Register the original server's prepared data and trained benchmark artifacts."""
import argparse
import csv
from pathlib import Path
import shutil
import sys

from src.experiments import DATASETS, LEGACY, METHODS, NEW, REFERENCES, SIMPLE, complete, group_for, load, members, run_path, weight_paths
from src.paths import DATA, PROTOCOL, RESULTS, ROOT
from src.release_io import write_csv, write_json
from src.setup_upstream import install


def link(source, dest):
    if not source.exists():
        raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if dest.resolve() != source.resolve():
            raise FileExistsError(dest)
    else:
        dest.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def data(old, raw, source_cache, dino_weights, datasets):
    for name in ("lumiere", "isles", "acdc"):
        source = source_cache / name
        if not source.is_dir():
            source = old / "runs/baseline_3fold_20261122/source_cache" / name if name != "acdc" else datasets / "ACDC"
        link(source, DATA / "source_cache" / name)
    for name in NEW:
        link(raw / name, DATA / "longitudinal" / name)
    link(old / "runs/delta_lfm_dlbs_20261122/autoencoder_epoch273.pt", DATA / "delta_lfm/autoencoder_epoch273.pt")
    link(dino_weights, DATA / dino_weights.name)
    for name, path in (("isles_protocol", datasets / "ISLES24/prepared_paper"),
                       ("lumiere_protocol", old / "cronos_lumiere_baseline/paper_protocol_data")):
        if path.exists():
            link(path, DATA / name)


def author_code(old, dino):
    import torch

    third = old / "third_party"
    for name, directory in (("trajdetail", dino), ("log4d", third / "4DLoG"), ("brlp", third / "BrLP"),
                            ("imageflownet", third / "ImageFlowNet-main"),
                            ("delta_lfm", third / "Disease-Progression-World-Model-main"),
                            ("cronos", old / "cronos_isles24_baseline/runs/discrete_seed42/source_snapshot")):
        install(name, directory)
    for dataset in LEGACY:
        if dataset == "isles":
            path = old / "cronos_isles24_baseline/runs/discrete_seed42/test_best.json"
            checkpoint = Path(load(path)["checkpoint"])
        else:
            path = old / f"runs/patient_trajectory_{dataset}_seed42_joint_primary/cronos_matched/comparison_metadata.json"
            checkpoint = Path(load(path)["checkpoint"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        write_json(PROTOCOL / "cronos_local" / f"{dataset}.json", dict(state["args"]))


def weights(old, selected_methods=None):
    base = old / "runs/baseline_3fold_20261122"
    comparators = old / "runs/comparators_20261122"
    repair = old / "runs/brlp_repair_20261123"
    for fold in (1, 2, 3):
        f = f"fold{fold}"
        for method in (selected_methods or METHODS):
            for group in dict.fromkeys(group_for(dataset, method) for dataset in DATASETS):
                directory = run_path(method, group, fold, "main")
                if (directory / "complete.json").exists():
                    continue
                config = load(ROOT / "configs" / group / f"{method}.json")
                paths = weight_paths(method, group)
                if method in SIMPLE:
                    source = base / f / ("models" if group == "mu-dlbs" else group)
                    paths["history"] = f"{method}_history.json"
                elif method == "trajdetail":
                    source = base / f
                elif method == "log4d":
                    source = comparators / "4dlog" / f / group
                    paths.update({name + "_done": f"4dlog_{name}.done" for name in ("registration", "diffusion")})
                elif method == "delta_lfm":
                    source = (old / "runs/delta_lfm_dlbs_20261122/joint" / f if group == "mu-dlbs" else
                              comparators / "delta_lfm_euler50" / f / group)
                elif method == "brlp":
                    source = comparators / "brlp" / f / group
                    selected = load(repair / "selections" / f / f"{group}.json")
                    use_tuned = selected["selected"] == "las_mask_volume_decoder"
                    write_json(directory / paths.pop("selection"), {"use_tuned_decoder": use_tuned,
                               "validation_nrmse_by_cohort": selected["validation_nrmse_by_cohort"]})
                    if use_tuned:
                        link(repair / "full_volume_decoder" / f / group / "brlp_ae.pt", directory / "decoder.pt")
                else:
                    source = base / f / group
                for name in paths.values():
                    link(source / name, directory / name)
                if method == "trajdetail":
                    for dataset in members(group):
                        fname = f"{dataset}/manifest.json"
                        link(source / fname, directory / fname)
                complete(directory, config, fold, origin="original benchmark archive")
    print("Registered all trained models and validation-selected stages across three folds.")


def reference(old):
    target = RESULTS / "reference"
    target.mkdir(parents=True, exist_ok=True)
    base = old / "runs/comparators_20261122/chronological_comparison"
    repair = old / "runs/brlp_repair_20261123"
    historical = target / "historical"
    historical.mkdir(exist_ok=True)
    for name in ("chronological_nrmse.csv", "shown_cases_nrmse.csv", "five_cohorts_predictions_and_errors.pdf",
                 *(f"{dataset}_predictions_and_errors.png" for dataset in DATASETS)):
        shutil.copyfile(base / name, historical / name)
    shutil.copyfile(repair / "brlp_comparison.png", historical / "brlp_repair.png")
    shutil.copyfile(repair / "nrmse_3fold.csv", target / "brlp_repair_nrmse.csv")
    with (repair / "nrmse_3fold.csv").open() as stream:
        updated = {row["dataset"]: row for row in csv.DictReader(stream)}
    with (base / "chronological_nrmse.csv").open() as stream:
        table = list(csv.DictReader(stream))
    labels = dict(acdc="ACDC", lumiere="LUMIERE", isles="ISLES’24", mu="MU-Glioma-Post", dlbs="DLBS")
    for row in table:
        if row["method"] == "BrLP":
            row["method"] = "BrLP (LAS + validation-selected decoder)"
            for name, label in labels.items():
                row[f"{label} mean"], row[f"{label} SD"] = (updated[name][f"updated_{key}"]
                                                          for key in ("mean", "sd"))
    write_csv(target / "nrmse_3fold.csv", table)
    shutil.copyfile(target / "nrmse_3fold.csv", target / "chronological_nrmse.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--datasets", type=Path)
    parser.add_argument("--dino", type=Path)
    parser.add_argument("--dino-weights", type=Path)
    parser.add_argument("--method", choices=METHODS, nargs="+")
    parser.add_argument("--only", choices=("reference", "assets", "weights"))
    args = parser.parse_args()
    if args.only in (None, "reference"):
        reference(args.source)
    if args.only in (None, "assets"):
        if any(getattr(args, arg) is None for arg in ("raw_cache", "source_cache", "datasets", "dino", "dino_weights")):
            parser.error("The assets import requires all five data and DINO paths.")
        data(args.source, args.raw_cache, args.source_cache, args.dino_weights, args.datasets)
        author_code(args.source, args.dino)
    if args.only in (None, "weights"):
        weights(args.source, args.method)


if __name__ == "__main__":
    main()
