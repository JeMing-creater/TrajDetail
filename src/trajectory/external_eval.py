"""Whole-volume scoring and artifacts shared by the external pilot runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torchmetrics.functional import structural_similarity_index_measure


METRICS = ("mae", "mse", "nrmse", "ssim", "psnr")


def _volume_batch(value, name):
    tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32, device="cpu")
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 5 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must be [N,D,H,W] or [N,1,D,H,W]")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def _score_one(prediction, target, ssim_data_range):
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    difference = prediction - target
    mse = float(difference.square().mean().item())
    ssim_kwargs = {} if ssim_data_range is None else {"data_range": ssim_data_range}
    return {
        "mae": float(difference.abs().mean().item()),
        "mse": mse,
        "nrmse": math.sqrt(mse),
        "ssim": float(structural_similarity_index_measure(
            prediction, target, **ssim_kwargs).item()),
        "psnr": float(10.0 * math.log10(1.0 / max(mse, 1e-12))),
    }


def evaluate_metrics(prediction, target, patient_ids, *, ssim_data_range=None):
    """Calculate per-patient full-volume metrics and equal-patient aggregates."""
    pred = _volume_batch(prediction, "prediction")
    truth = _volume_batch(target, "target")
    ids = list(patient_ids)
    if pred.shape != truth.shape or len(ids) != pred.shape[0] or len(set(ids)) != len(ids):
        raise ValueError("prediction, target, and patient ID cardinalities must match")
    rows = [{"patient_id": str(patient_id), **_score_one(pred[i:i + 1], truth[i:i + 1], ssim_data_range)}
            for i, patient_id in enumerate(ids)]
    values = {name: np.asarray([row[name] for row in rows], dtype=np.float64) for name in METRICS}
    return {
        "n_patients": len(rows),
        "shape": list(pred.shape[2:]),
        "protocol": {
            "whole_volume": True,
            "masking": False,
            "clipping": False,
            "patient_weighting": "equal",
            "nrmse": "sqrt(per-patient full-volume MSE)",
            "psnr_data_range": 1.0,
            "ssim_data_range": "torchmetrics default" if ssim_data_range is None else ssim_data_range,
        },
        "mean": {name: float(values[name].mean()) for name in METRICS},
        "median": {name: float(np.median(values[name])) for name in METRICS},
        "per_patient": rows,
    }


def save_evaluation(path, prediction, target, last_context, patient_ids, *,
                    ssim_data_range=None, title="MRI forecast"):
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty evaluation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate_metrics(prediction, target, patient_ids, ssim_data_range=ssim_data_range)
    copy_result = evaluate_metrics(last_context, target, patient_ids, ssim_data_range=ssim_data_range)
    result["last_context"] = {"mean": copy_result["mean"], "median": copy_result["median"]}
    result["last_context_per_patient"] = copy_result["per_patient"]

    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    last = np.asarray(last_context, dtype=np.float32)
    volumes = output / "volumes"
    volumes.mkdir()
    for index, patient_id in enumerate(patient_ids):
        np.savez_compressed(volumes / f"{patient_id}.npz", prediction=pred[index],
                            target=truth[index], last_context=last[index])
    with (output / "per_patient.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient_id", *METRICS])
        writer.writeheader()
        writer.writerows(result["per_patient"])
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    _save_panel(output / "representative_case.png", pred[0], truth[0], last[0], str(patient_ids[0]), title)
    result["output_dir"] = str(output.resolve())
    return result


def _save_panel(path, prediction, target, last_context, patient_id, title, *, reference_title="last context"):
    import matplotlib.pyplot as plt

    depth = prediction.shape[0] // 2
    images = (last_context[depth], prediction[depth], target[depth],
              np.abs(prediction[depth] - target[depth]))
    titles = (reference_title, "prediction", "target", "absolute error")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    for axis, image, panel_title in zip(axes, images, titles):
        im = axis.imshow(image, cmap="gray" if panel_title != "absolute error" else "magma",
                         vmin=0.0 if panel_title != "absolute error" else None,
                         vmax=1.0 if panel_title != "absolute error" else None)
        axis.set_title(panel_title)
        axis.axis("off")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"{title}: {patient_id} (central slice)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


__all__ = ["evaluate_metrics", "save_evaluation"]
