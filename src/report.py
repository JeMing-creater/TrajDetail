"""Audit cohort coverage and summarize paired forecast performance."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from src.compare import METHODS, save_rows

DISPLAY = {
    "copy": "Copy last",
    "linear": "Linear extrapolation",
    "convlstm": "ConvLSTM",
    "simvp": "SimVP",
    "node_lstm": "NODE + LSTM",
    "vivit": "ViViT",
    "cronos": "CRONOS",
    "imageflownet": "ImageFlowNet",
    "frozen_trajectory": "Frozen trajectory",
    "frozen_voxel": "Frozen voxel branch",
    "frozen_method": "Ours (frozen)",
    "adapted_voxel": "Adapted voxel branch",
    "adapted_method": "Ours (adapted)",
    "our_method": "Ours (joint readout + HF)",
}
REQUIRED = (*METHODS, "frozen_trajectory", "frozen_voxel", "frozen_method",
            "adapted_voxel", "adapted_method")
METRICS = ("nrmse", "mae", "ssim", "psnr", "core_edge_hf_ratio", "core_edge_hf_nrmse")


def case_manifest(path):
    protocol = json.loads((path / "manifest.json").read_text())
    unique = set()
    partitions = {split: [] for split in ("train", "validation", "test")}
    for row in protocol["patients"]:
        patient = row["patient_id"]
        if patient in unique or len(row["days"]) != 3 or min(np.diff(row["days"])) <= 0:
            raise ValueError(f"Duplicate patient or noncausal window for {patient}")
        unique.add(patient)
        partitions[row["split"]].append(row)
    counts = {split: len(rows) for split, rows in partitions.items()}
    assert all(count and count == protocol["counts"][split] for split, count in counts.items())
    return protocol, partitions


def summarize(path, protocol, partitions, required, *, scratch=False):
    patients = [row["patient_id"] for row in partitions["test"]]
    method_metrics = {}
    for method in (*METHODS, "our_method") if scratch else REQUIRED:
        filename = path / f"{method}_test.csv"
        if not filename.is_file():
            if required:
                raise FileNotFoundError(filename)
            continue
        with filename.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [row["patient_id"] for row in rows] != patients or any(row["method"] != method for row in rows):
            raise ValueError(f"Test rows do not match held-out cohort in {filename}")
        method_metrics[method] = {metric: np.asarray([float(row[metric]) for row in rows])
                                  for metric in METRICS}
        if not all(np.isfinite(x).all() for x in method_metrics[method].values()):
            raise ValueError(f"Nonfinite evaluation results in {filename}")
    rng = np.random.default_rng(42)
    index = rng.integers(0, len(patients), size=(10000, len(patients)))
    comparison = []
    for method, metrics in method_metrics.items():
        difference = metrics["nrmse"] - method_metrics["copy"]["nrmse"]
        lo, hi = np.percentile(difference[index].mean(1), [2.5, 97.5])
        comparison.append({"dataset": protocol["dataset"], "method": method, "n_test": len(patients),
                           **{key: float(values.mean()) for key, values in metrics.items()},
                           "nrmse_diff_vs_copy": float(difference.mean()),
                           "nrmse_diff_ci_lower": float(lo), "nrmse_diff_ci_upper": float(hi),
                           "nrmse_wins_vs_copy": int((difference < 0).sum())})
    if required:
        proposed = method_metrics["our_method" if scratch else "adapted_method"]["nrmse"]
        paired = []
        for baseline in METHODS if scratch else ("copy", "cronos", "imageflownet", "adapted_voxel", "frozen_method"):
            reference = method_metrics[baseline]["nrmse"]
            delta = proposed - reference
            low, high = np.percentile(delta[index].mean(axis=1), [2.5, 97.5])
            paired.append({"baseline": baseline, "n_test": len(patients),
                           "mean_delta_nrmse": delta.mean(), "ci_low": low, "ci_high": high,
                           "relative_nrmse_reduction_pct": 100 * (1 - proposed.mean()/reference.mean()),
                           "wins": int((delta < 0).sum())})
        save_rows(path / "paired_comparison.csv", paired)
    return comparison


def visualize(path, title, methods):
    inputs = np.load(path / "example_inputs_targets.npy")
    # Use the same central axial plane for every method on the fixed test case.
    depth = inputs.shape[1] // 2
    selection = [x for x in methods if x not in ("frozen_trajectory", "frozen_voxel", "adapted_voxel")]
    volumes = {"History 1": inputs[0], "History 2": inputs[1], "Target": inputs[2]}
    volumes.update({DISPLAY[method]: np.load(path / f"{method}_example.npy") for method in selection})
    cols = 4
    fig, axes = plt.subplots((len(volumes)+cols-1)//cols, cols, figsize=(9.6, 2.45*((len(volumes)+cols-1)//cols)))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < len(volumes):
            name, volume = list(volumes.items())[index]
            axis.set_title(name, fontsize=10)
            axis.imshow(volume[depth], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--scratch", action="store_true", help="Report the from-scratch joint comparison")
    args = parser.parse_args()
    rows = []
    for dataset in ("mu", "dlbs"):
        path = args.output / dataset
        if args.allow_partial and not (path / "copy_test.csv").is_file():
            continue
        protocol, partitions = case_manifest(path)
        comparison = summarize(path, protocol, partitions, required=not args.allow_partial, scratch=args.scratch)
        case = partitions["test"][len(partitions["test"])//2]
        visualize(path, f"{dataset.upper()} · {case['patient_id']} · {case['days'][2]-case['days'][1]:.0f} days",
                  [row["method"] for row in comparison])
        test_gaps = [x["days"][2]-x["days"][1] for x in partitions["test"]]
        qc = [x["registration"]["2"]["ncc_after"] for x in protocol["patients"] if x["registration"]]
        print(dataset, protocol["counts"], "median test gap (days)", np.median(test_gaps),
              "registration NCC quantiles", np.quantile(qc, [.0, .05, .5]).tolist() if qc else None)
        rows.extend(comparison)
    if not rows:
        raise ValueError("There are no completed results")
    save_rows(args.output / ("comparison_partial.csv" if args.allow_partial else "comparison.csv"), rows)


if __name__ == "__main__":
    main()
