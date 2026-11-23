"""Measure ACDC edge fidelity and replay the frozen trajectory without repeated image resampling."""

import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation
import torch
import torch.nn.functional as F

from . import matched_control
from src.trajectory.external_data import ACDCJointDataset, PROJECT_ROOT
from methods.trajdetail.matched_control import MatchedControl
from methods.trajdetail.train_local_control import last_features
from src.trajectory.train_unified import _autocast, _input
from methods.trajdetail.unified_model import warp_volume
from src.trajectory.validate_control import batch_item


RUNS = PROJECT_ROOT / "runs"
MATCHED = RUNS / "trajectory_control_acdc_matched_benefit_seed42"
STORED = {
    "CRONOS": RUNS / "patient_trajectory_acdc_seed42_joint_primary/cronos_matched",
    "Control": MATCHED / "dinov3_matching/base_forecasts",
    "DINO proposal": MATCHED / "dinov3_matching/proposals",
    "Current": MATCHED / "dinov3_matching/predictions",
    "Earlier gradient-loss arm": RUNS / "trajectory_control_acdc_seed42_20261117_v3/test/gradient",
}
REPLAYS = ("Gate in displacement space", "Compose base warps", "Compose all warps")
DETAIL_ARMS = ("Transported detail", "Locally aligned transported detail", "Uniform sharpening")


def spatial_average(image):
    return F.avg_pool3d(image, (1, 5, 5), stride=1, padding=(0, 2, 2), count_include_pad=False)


def detail_readouts(current, transported):
    current_detail = current - spatial_average(current)
    transported_detail = transported - spatial_average(transported)
    cross = spatial_average(current_detail * transported_detail)
    energy = (spatial_average(current_detail.square()) *
              spatial_average(transported_detail.square())).clamp_min(1e-10)
    reliability = (cross / energy.sqrt()).clamp(0., 1.)
    detail_difference = transported_detail - current_detail
    return {"Transported detail": detail_difference,
            "Locally aligned transported detail": reliability * detail_difference,
            "Uniform sharpening": current_detail}


class ReplayProbe(MatchedControl):
    """Keep learned fields on the original trajectory and vary only the image readout."""

    def _advance(self, image, memory, field, step):
        result = super()._advance(image, memory, field, step)
        if self._transport is not None and step:
            # Keep tensor references alive to prevent id reuse across fractional reads.
            _, previous_displacement, previous_source = self._transport[id(image)]
            displacement = step * field[0]
            total = displacement + warp_volume(previous_displacement, displacement, self._sampling_grid)
            source = warp_volume(previous_source, displacement, self._sampling_grid) + step * field[1]
            self._transport[id(result[0])] = result[0], total, source
        return result

    def _forecast(self, image, memory, start, queries, statistics, condition):
        shape = image.shape[-3:]
        self._transport = {id(image): (image, image.new_zeros(1, 3, *shape), torch.zeros_like(image))}
        proposals = []

        def capture(volume, displacement, base_grid=None):
            if volume.shape[1] == 1:
                proposals.append((volume, displacement))
            return warp_volume(volume, displacement, base_grid)

        try:
            with patch.object(matched_control, "warp_volume", capture):
                forecasts, extras = super()._forecast(image, memory, start, queries, statistics, condition)
            if self.capture_transport:
                self.transport_parts = []
                for index, (baseline, proposal_shift) in enumerate(proposals):
                    _, total, source = self._transport[id(baseline)]
                    gate = F.interpolate(extras["gates"][:, index:index + 1], size=shape,
                                         mode="trilinear", align_corners=False)
                    gate_shift = gate * proposal_shift
                    displacement = gate_shift + warp_volume(total, gate_shift, self._sampling_grid)
                    transported = warp_volume(image, displacement, self._sampling_grid)
                    source = warp_volume(source, gate_shift, self._sampling_grid)
                    self.transport_parts.append((image, transported, displacement, source,
                                                 forecasts[index][0], queries[index] - start))
                return forecasts, extras
            baseline, proposal_shift = proposals[-1]
            _, total, source = self._transport[id(baseline)]
            gate = F.interpolate(extras["gates"][:, -1:], size=shape, mode="trilinear", align_corners=False)
            current = forecasts[-1][0]
            single_base = warp_volume(image, total, self._sampling_grid) + source
            single_proposal = warp_volume(single_base, proposal_shift, self._sampling_grid)
            gate_shift = gate * proposal_shift
            compose_all = gate_shift + warp_volume(total, gate_shift, self._sampling_grid)
            transported = warp_volume(image, compose_all, self._sampling_grid)
            self.readouts = {
                "Control": baseline,
                "DINO proposal": extras["proposals"][:, -1:],
                "Current": current,
                "Gate in displacement space": warp_volume(baseline, gate_shift, self._sampling_grid),
                "Compose base warps": single_base + gate * (single_proposal - single_base),
                "Compose all warps": transported + warp_volume(source, gate_shift, self._sampling_grid),
            }
            self.detail = detail_readouts(current, transported)
            return forecasts, extras
        finally:
            self._transport = None

    _transport = None
    capture_transport = False


def gradient(volume):
    """Central in-plane derivatives in the shared preprocessed voxel grid."""
    dy, dx = np.gradient(volume, axis=(-2, -1))
    return dy, dx, np.hypot(dy, dx)


def regions(target):
    inside = binary_dilation(target > .06, iterations=2)
    dy, dx, strength = gradient(target)
    edge = inside & (strength >= np.quantile(strength[inside], .80))
    return (dy, dx, strength), inside, edge


def metrics(prediction, target, masks):
    (target_dy, target_dx, target_strength), inside, edge = masks
    error = prediction - target
    squared = np.square(error)
    dy, dx, strength = gradient(prediction)
    gradient_error = np.hypot(dy - target_dy, dx - target_dx)
    nonedge = inside & ~edge
    return {
        "nrmse": float(np.sqrt(squared.mean(dtype=np.float64))),
        "edge_mae": float(np.abs(error[edge]).mean(dtype=np.float64)),
        "edge_gradient_mae": float(gradient_error[edge].mean(dtype=np.float64)),
        "edge_strength_ratio": float(strength[edge].sum(dtype=np.float64) /
                                     target_strength[edge].sum(dtype=np.float64)),
        "outside_edge_mae": float(np.abs(error[nonedge]).mean(dtype=np.float64)),
        "edge_voxels": int(edge.sum()),
        "all_voxels": int(edge.size),
        "edge_sse": float(squared[edge].sum(dtype=np.float64)),
        "all_sse": float(squared.sum(dtype=np.float64)),
    }


def stored_volumes(patient, target, anchor):
    entries = {}
    for name, directory in STORED.items():
        with np.load(directory / "volumes" / f"{patient}.npz") as archive:
            if not np.array_equal(archive["target"], target) or not np.array_equal(archive["last_context"], anchor):
                raise ValueError(f"Mismatched target or last observation for {name}/{patient}")
            entries[name] = archive["prediction"]
    return entries


def final_readout(model, name):
    if name in model.readouts:
        return model.readouts[name][0, 0].cpu().numpy()
    arm, fraction = name.rsplit(" ", maxsplit=1)
    current = model.readouts["Current"]
    residual = model.detail[arm] if arm in model.detail else model.readouts[arm] - current
    return (current + float(fraction) * residual)[0, 0].cpu().numpy()


@torch.inference_mode()
def evaluate(model, split, device, cache, selected=()):
    rows = []
    dataset = ACDCJointDataset(split=split)
    for item in dataset:
        patient = item["patient_id"]
        target = item["targets"][-1]
        last = np.flatnonzero(item["mask"])[-1]
        anchor = item["context"][last]
        masks = regions(target)
        with _autocast(device, True):
            model(*_input(batch_item(item), device), last_features=last_features(item, cache, device))
        readouts = {name: value[0, 0].cpu().numpy() for name, value in model.readouts.items()}
        if split == "test":
            stored = stored_volumes(patient, target, anchor)
            for name in ("Control", "DINO proposal", "Current"):
                np.testing.assert_allclose(readouts[name], stored[name], atol=2e-6, rtol=1e-6)
            readouts.update(stored)
        if split == "validation":
            selected = tuple(f"{name} {fraction:g}" for name in (*REPLAYS, *DETAIL_ARMS)
                             for fraction in ((.25, .5) if name in REPLAYS else (.125, .25, .5, .75, 1., 1.5, 2.)))
        for name in selected:
            if name not in readouts:
                readouts[name] = final_readout(model, name)
        readouts["Last observation"] = anchor
        for name, prediction in readouts.items():
            if np.isfinite(prediction).all():
                rows.append({"patient_id": patient, "method": name, **metrics(prediction, target, masks)})
            else:
                raise ValueError(f"Nonfinite {name}/{patient}")
    return rows


def aggregates(rows):
    by_method = {}
    columns = ("nrmse", "edge_mae", "edge_gradient_mae", "edge_strength_ratio", "outside_edge_mae")
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summaries = {}
    for method, group in by_method.items():
        summaries[method] = {
            **{key: float(np.mean([row[key] for row in group])) for key in columns},
            "edge_sse_fraction": sum(row["edge_sse"] for row in group) / sum(row["all_sse"] for row in group),
            "edge_voxel_fraction": sum(row["edge_voxels"] for row in group) / sum(row["all_voxels"] for row in group),
            "patient_count": len(group),
        }
    return summaries


@torch.inference_mode()
def draw(model, rows, selected, device, cache, output):
    copies = sorted((row for row in rows if row["method"] == "Last observation"),
                    key=lambda row: row["nrmse"])
    patients = {copies[index]["patient_id"] for index in (len(copies) // 2, 3 * len(copies) // 4)}
    dataset = ACDCJointDataset(split="test")
    names = ("Target", "CRONOS", "Control", "DINO proposal", "Current")
    if selected != "Current":
        names += ("Selected readout",)
    fig, axes = plt.subplots(4, len(names), figsize=(2.05 * len(names), 8.1), layout="constrained")
    for index, item in enumerate(item for item in dataset if item["patient_id"] in patients):
        target = item["targets"][-1]
        anchor = item["context"][np.flatnonzero(item["mask"])[-1]]
        image_dict = {"Target": target, **stored_volumes(item["patient_id"], target, anchor)}
        if selected != "Current":
            with _autocast(device, True):
                model(*_input(batch_item(item), device), last_features=last_features(item, cache, device))
            image_dict["Selected readout"] = final_readout(model, selected)
        middle = range(target.shape[0] // 4, 3 * target.shape[0] // 4)
        z = max(middle, key=lambda depth: float(np.abs(target[depth] - anchor[depth]).sum()))
        detail = (slice(37, 100), slice(37, 100))
        max_image = float(np.quantile(target[z][target[z] > .06], .995))
        max_gradient = float(np.quantile(gradient(target[z][detail])[2], .985))
        for column, name in enumerate(names):
            cut = image_dict[name][z][detail]
            a, b = axes[2 * index:2 * index + 2, column]
            a.imshow(cut, cmap="gray", vmin=0, vmax=max_image, interpolation="nearest")
            b.imshow(gradient(cut)[2], cmap="magma", vmin=0, vmax=max_gradient, interpolation="nearest")
            if index == 0:
                a.set_title(name)
            if column == 0:
                a.set_ylabel(f"{item['patient_id']} · z={z}\nMRI detail")
                b.set_ylabel("Gradient magnitude")
            for axis in (a, b):
                axis.set(xticks=[], yticks=[])
    fig.suptitle("ACDC  |  Same cases, same windows, same display scales", fontsize=15)
    fig.savefig(output, dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = json.loads((MATCHED / "config.json").read_text())
    cache = Path(config["cache"])
    device = torch.device(args.device)
    model = ReplayProbe(use_dino=True).to(device).eval()
    model.load_state_dict(torch.load(MATCHED / "dinov3_matching/benefit.pt",
                                     map_location=device, weights_only=True)["model"])
    val_rows = evaluate(model, "validation", device, cache)
    val = aggregates(val_rows)
    max_nrmse = val["Current"]["nrmse"] * 1.01
    def select(prefixes):
        eligible = ["Current", *(name for name in val if name.startswith(prefixes)
                                 and val[name]["nrmse"] <= max_nrmse)]
        return min(eligible, key=lambda name: val[name]["edge_gradient_mae"])

    choices = {
        "warp_only": select(REPLAYS),
        "transported_detail": select(DETAIL_ARMS[:2]),
        "uniform_sharpening": select(DETAIL_ARMS[2:]),
    }
    print(f"Validation selections: {choices}", flush=True)
    test_rows = evaluate(model, "test", device, cache, tuple(dict.fromkeys(choices.values())))
    test = aggregates(test_rows)
    for split, rows in (("validation", val_rows), ("test", test_rows)):
        with (args.output / f"{split}_per_patient.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {"selected_on_validation": choices, "max_validation_nrmse": max_nrmse,
              "validation": val, "test": test}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    draw(model, test_rows, choices["transported_detail"], device, cache, args.output / "edge_comparison.png")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
