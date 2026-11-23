"""Evaluation helpers for full-volume ISLES-style forecasting.

The primary score is deliberately kept independent of the frozen CRONOS
evaluator while matching its whole-volume formulas: SSIM keeps the installed
torchmetrics default data range, PSNR uses data range 1, with no clipping or
masking and equal weight per patient.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torchmetrics.functional import structural_similarity_index_measure as _ssim


METRICS = ("mae", "mse", "nrmse", "ssim", "psnr")


def _as_volume_batch(value: np.ndarray | torch.Tensor, name: str) -> torch.Tensor:
    """Normalize arrays to float32 ``[N, 1, D, H, W]`` without rescaling."""
    tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))
    tensor = tensor.detach().to(dtype=torch.float32, device="cpu")
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 5 or tensor.shape[1] != 1:
        raise ValueError(f"{name} must have shape [N,D,H,W] or [N,1,D,H,W], got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def patient_scores(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute exact full-volume patient metrics used by the primary protocol."""
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    difference = prediction - target
    mse = float(difference.square().mean().item())
    return {
        "mae": float(difference.abs().mean().item()),
        "mse": mse,
        "nrmse": math.sqrt(mse),
        # The frozen CRONOS scorer calls torchmetrics SSIM without a
        # data_range argument; retain that protocol rather than silently
        # switching to the common fixed-range variant.
        "ssim": float(_ssim(prediction, target).item()),
        "psnr": float(10.0 * math.log10(1.0 / max(mse, 1.0e-12))),
    }


def _aggregate(rows: Sequence[Mapping[str, object]], reducer: str) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty patient list")
    values = {metric: np.asarray([float(row[metric]) for row in rows], dtype=np.float64) for metric in METRICS}
    if reducer == "mean":
        return {metric: float(values[metric].mean()) for metric in METRICS}
    if reducer == "median":
        return {metric: float(np.median(values[metric])) for metric in METRICS}
    raise ValueError(f"unknown reducer: {reducer}")


def _slice_array(value: np.ndarray | torch.Tensor, index: int) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32, copy=True)
    return np.asarray(value, dtype=np.float32).copy()


def _save_panel(path: Path, prediction: np.ndarray, target: np.ndarray, last: np.ndarray, patient_id: str) -> None:
    """Save a deterministic central-slice panel; this is diagnostic only."""
    import matplotlib.pyplot as plt

    # Inputs are [D,H,W] or [1,D,H,W] after selecting one case.
    def squeeze_channel(array: np.ndarray) -> np.ndarray:
        return array[0] if array.ndim == 4 and array.shape[0] == 1 else array

    pred, gt, lci = map(squeeze_channel, (prediction, target, last))
    z = pred.shape[0] // 2
    images = (lci[z], pred[z], gt[z], np.abs(pred[z] - gt[z]))
    titles = ("last context", "prediction", "target", "absolute error")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    for axis, image, title in zip(axes, images, titles):
        im = axis.imshow(image, cmap="gray" if title != "absolute error" else "magma", vmin=0.0 if title != "absolute error" else None, vmax=1.0 if title != "absolute error" else None)
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"ISLES full-volume diagnostic: {patient_id} (central slice)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_arrays(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    last_context: np.ndarray | torch.Tensor,
    patient_ids: Iterable[str] | None = None,
    *,
    output_dir: str | Path | None = None,
    make_figure: bool = True,
) -> dict[str, object]:
    """Evaluate predictions and optionally persist a reviewable artifact.

    ``predictions``, ``targets`` and ``last_context`` have shape ``[N,D,H,W]``
    or ``[N,1,D,H,W]``.  Every patient contributes equally to ``mean`` and
    ``median`` is included for direct comparison with the ISLES primary table.
    The first case in the supplied frozen order is visualized; no case is
    selected using its ground-truth score.
    """
    pred = _as_volume_batch(predictions, "predictions")
    target = _as_volume_batch(targets, "targets")
    last = _as_volume_batch(last_context, "last_context")
    if pred.shape != target.shape or pred.shape != last.shape:
        raise ValueError(f"all arrays must share shape, got {pred.shape}, {target.shape}, {last.shape}")
    n = int(pred.shape[0])
    ids = list(patient_ids) if patient_ids is not None else [f"case_{i:04d}" for i in range(n)]
    if len(ids) != n or len(set(ids)) != n or any(not str(pid) or Path(str(pid)).name != str(pid) for pid in ids):
        raise ValueError("patient_ids must contain unique safe non-empty names")

    rows: list[dict[str, object]] = []
    copy_rows: list[dict[str, object]] = []
    for i, patient_id in enumerate(ids):
        row = {"patient_id": str(patient_id), **patient_scores(pred[i:i + 1], target[i:i + 1])}
        copy_row = {"patient_id": str(patient_id), **patient_scores(last[i:i + 1], target[i:i + 1])}
        rows.append(row)
        copy_rows.append(copy_row)
    result: dict[str, object] = {
        "protocol": {
            "data_range": {"ssim": "torchmetrics default", "psnr": 1.0},
            "whole_volume": True,
            "masking": False,
            "clipping": False,
            "patient_weighting": "equal",
            "nrmse": "sqrt(per-patient MSE)",
            "ssim": "torchmetrics functional SSIM with its default data_range",
            "psnr": "10*log10(1/MSE), data_range=1.0",
        },
        "n_patients": n,
        "shape": list(pred.shape[2:]),
        "model": {"mean": _aggregate(rows, "mean"), "median": _aggregate(rows, "median")},
        "last_context": {"mean": _aggregate(copy_rows, "mean"), "median": _aggregate(copy_rows, "median")},
        "per_patient": rows,
        "per_patient_last_context": copy_rows,
    }

    if output_dir is not None:
        out = Path(output_dir).resolve()
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty evaluation directory: {out}")
        out.mkdir(parents=True, exist_ok=True)
        volume_dir = out / "volumes"
        volume_dir.mkdir()
        for i, patient_id in enumerate(ids):
            np.savez_compressed(volume_dir / f"{patient_id}.npz", prediction=_slice_array(pred[i], i), target=_slice_array(target[i], i), last=_slice_array(last[i], i))
        with (out / "per_patient.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["patient_id", *METRICS])
            writer.writeheader()
            writer.writerows(rows)
        (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if make_figure:
            _save_panel(out / "representative_case.png", _slice_array(pred[0], 0), _slice_array(target[0], 0), _slice_array(last[0], 0), str(ids[0]))
        result["output_dir"] = str(out)
        result["representative_case"] = str(ids[0])
    return result
