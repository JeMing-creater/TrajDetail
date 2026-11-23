"""Run held-out evaluation and render matched images and absolute error maps."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
from contextlib import nullcontext

import numpy as np

from src.experiments import DISPLAY, METHODS, discover
from src.paths import RESULTS
from src.release_io import write_csv, write_json


def evaluate(dataset, fold, methods, device, run, limit, save_predictions=False):
    import torch

    from src.metrics import score
    from src.inference import infer, records

    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    root = RESULTS / dataset / run / f"fold{fold}" / (f"preview_{limit}" if limit else "test")
    summaries = []
    examples = {}
    sample_id = None
    for method in methods:
        directory, meta = discover(method, dataset, fold, run)
        ordered = list(itertools.islice(records(dataset, fold), limit))
        if not ordered or len({row[0] for row in ordered}) != len(ordered):
            raise ValueError("The held-out partition must contain unique patients")
        selected = ordered[len(ordered) // 2]
        if sample_id is None:
            sample_id = selected[0]
            examples.update(target=selected[1], last=selected[2])
        elif selected[0] != sample_id:
            raise ValueError(f"Case ordering differed for {method}")
        remaining = {patient: (truth, anchor) for patient, truth, anchor, _ in ordered}
        rows = []
        destination = root / method
        destination.mkdir(parents=True, exist_ok=True)
        for patient, prediction in infer(method, dataset, ordered, directory, meta, destination, device):
            target, anchor = remaining.pop(patient)
            if prediction.shape != target.shape:
                raise ValueError(f"{method}/{patient}: forecast and target do not share a grid")
            values = score(prediction.astype(np.float32), target.astype(np.float32),
                           "acdc" if dataset == "lumiere" else dataset, device)
            rows.append(dict(patient_id=patient, **values))
            if patient == sample_id:
                examples[method] = prediction
                examples[f"{method}_nrmse"] = values["nrmse"]
            if save_predictions:
                path = destination / "predictions" / f"{patient}.npy"
                path.parent.mkdir(exist_ok=True)
                np.save(path, prediction, allow_pickle=False)
        if remaining:
            raise ValueError(f"{method} did not score all expected patients")
        write_csv(destination / "metrics.csv", rows)
        metrics = {key: statistics.fmean(row[key] for row in rows if key in row)
                   for key in dict.fromkeys(key for row in rows for key in row) if key != "patient_id"}
        summaries.append(dict(dataset=dataset, fold=fold, method=method, patients=len(rows), **metrics))
        print(f"{dataset}/fold{fold} {DISPLAY[method]}: NRMSE {metrics['nrmse']:.6f} ({len(rows)} cases)", flush=True)
        del ordered
    write_csv(root / "comparison.csv", summaries)
    render(examples, methods, root / "comparison.png", f"{dataset} | Fold {fold}")
    write_json(root / "evaluation.json", {"dataset": dataset, "fold": fold, "methods": methods,
        "limited_cases": limit, "selection": "middle patient in fixed held-out order",
        "nrmse": "RMSE on the shared [0,1] analysis grid"})
    return summaries


def render(examples, methods, path, title):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    middle = len(examples["target"]) // 2
    target = examples["target"][middle]
    labels = ["Last observed", "Ground truth", *[DISPLAY[method] for method in methods]]
    images = [examples["last"][middle], target, *[examples[method][middle] for method in methods]]
    error_cap = .3
    figure, axes = plt.subplots(2, len(images), squeeze=False, figsize=(2.05 * len(images), 4.7),
                                layout="constrained")
    for i, (label, image) in enumerate(zip(labels, images, strict=True)):
        axes[0, i].imshow(image, cmap="gray", vmin=0, vmax=1)
        if i >= 2:
            label += f"\nNRMSE {examples[methods[i - 2] + '_nrmse']:.4f}"
        axes[0, i].set_title(label)
        mapping = axes[1, i].imshow(np.abs(image - target), cmap="magma", vmin=0, vmax=error_cap)
        for axis in axes[:, i]:
            axis.set_axis_off()
    axes[1, 0].set_title("Absolute error")
    figure.colorbar(mapping, ax=axes[1, :].tolist(), shrink=.78)
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(dataset):
    parser = argparse.ArgumentParser(description=f"Evaluate selected checkpoints on {dataset}.")
    parser.add_argument("--method", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--fold", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run", default="main")
    parser.add_argument("--limit", type=int, help="Preview only, no full-cohort statistic")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    from src.experiments import run_path

    run_path(args.method[0], dataset, args.fold[0], args.run)
    summaries = []
    for fold in dict.fromkeys(args.fold):
        summaries.extend(evaluate(dataset, fold, tuple(dict.fromkeys(args.method)), args.device, args.run,
                                   args.limit, args.save_predictions))
    if set(args.fold) == {1, 2, 3} and args.limit is None:
        summary = []
        for method in dict.fromkeys(args.method):
            values = [row for row in summaries if row["method"] == method]
            scores = [row["nrmse"] for row in values]
            summary.append(dict(dataset=dataset, method=method, patients=sum(row["patients"] for row in values),
                                nrmse_mean=statistics.fmean(scores), nrmse_sd=statistics.stdev(scores)))
        write_csv(RESULTS / dataset / args.run / "comparison_3fold.csv", summary)
