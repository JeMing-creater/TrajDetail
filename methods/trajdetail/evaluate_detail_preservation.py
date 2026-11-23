"""Compare two frozen, causal readouts under a validation detail floor."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from methods.trajdetail.audit_acdc_sharpness import regions
from src.trajectory.external_eval import _save_panel
from src.trajectory.shared_forecast_data import DATASETS
from methods.trajdetail.shared_forecast_model import SharedForecast
from src.metrics import score
from methods.trajdetail.train_shared_forecast import Windows, forecast_volume
from src.trajectory.validate_control import write_json


CUTOFFS = (.065, .10, .15, .20, .25)
LOW_WEIGHTS = (.25, .50, .75, 1.)
SHIFT_WEIGHTS = (.25, .50, 1.)
PATCH_SIZES = (8, 16)
PATCH_WEIGHTS = LOW_WEIGHTS
PROTECTION_LEVELS = (.55, .72, .86)
CONSTRAINT_LEVELS = (1., .55, .72)


def constrained_residuals(base, refined):
    """Approximate the closest residual that retains signed prototype detail."""
    detail = highpass(base)
    foreground = F.max_pool2d((base > .06).float()[:, None], 7, stride=1, padding=3)[:, 0] > 0
    if not foreground.any():
        return [torch.zeros_like(base) for _ in CONSTRAINT_LEVELS]
    absolute = detail.abs()
    limits = torch.quantile(absolute[foreground],
                            torch.tensor(CONSTRAINT_LEVELS[1:], device=base.device))
    masks = [foreground, *((absolute >= threshold) & foreground for threshold in limits)]
    reference = (refined - base).repeat(len(masks), 1, 1)
    shape = base.shape
    direction = (torch.stack(masks) * detail.sign()).reshape_as(reference)
    # Projected dual ascent for min ||u - residual||² / 2, s.t. sign(d) H(u) >= 0.
    # A conservative step follows from the Gaussian high-pass operator norm.
    dual = torch.zeros_like(reference)
    extrapolated_dual = dual
    for iteration in range(36):
        update = reference + highpass(direction * extrapolated_dual)
        next_dual = (extrapolated_dual - .85 * direction * highpass(update)).clamp_min(0)
        extrapolated_dual = next_dual + (iteration / (iteration + 3)) * (next_dual - dual)
        dual = next_dual
    return list((reference + highpass(direction * dual)).reshape(len(masks), *shape))


def protected_updates(base, refined):
    """Keep the detail filter's footprint fixed around prototype edges."""
    detail = highpass(base).abs()
    foreground = F.max_pool2d((base > .06).float()[:, None], 5, stride=1, padding=2)[:, 0] > 0
    if not foreground.any():
        return [torch.zeros_like(base) for _ in PROTECTION_LEVELS]
    thresholds = torch.quantile(detail[foreground],
                                torch.tensor(PROTECTION_LEVELS, device=base.device))
    residual = refined - base
    results = []
    for threshold in thresholds:
        edges = (detail >= threshold) & foreground
        # Sigma 0.8 has a three-pixel radius in the scoring implementation.
        protected = F.max_pool2d(edges.float()[:, None], 7, stride=1, padding=3)[:, 0]
        taper = gaussian(protected, 1.15)
        results.append(residual * (1 - taper) * (1 - protected))
    return results


def project_patch_bands(base, refined, size, amount):
    """Update each patch while preserving its prototype energy per detail band."""
    depth, height, width = base.shape
    if height % size or width % size:
        raise ValueError(f"Patch size {size} does not divide the image grid")
    k = torch.arange(size, dtype=base.dtype, device=base.device)
    transform = torch.cos((torch.pi / size) * k[:, None] * (k + .5)) * (2 / size) ** .5
    transform[0] /= 2 ** .5
    frequencies = (k[:, None].square() + k.square()).sqrt() / (2 * size)
    bands = [(frequencies > .125) & (frequencies <= .25), frequencies > .25]

    def patches(volume):
        return volume.reshape(depth, height // size, size, width // size, size).transpose(2, 3)

    origin = transform @ patches(base) @ transform.T
    moved = torch.lerp(origin, transform @ patches(refined) @ transform.T, amount)
    for mask in bands:
        reference = origin[..., mask]
        proposed = moved[..., mask]
        energy = torch.linalg.vector_norm(proposed, dim=-1, keepdim=True)
        scale = torch.linalg.vector_norm(reference, dim=-1, keepdim=True) / energy.clamp_min(1e-8)
        # A near-zero proposal offers no usable orientation.
        moved[..., mask] = torch.where(energy > 1e-8, proposed * scale, reference)
    delta = transform.T @ (moved - origin) @ transform
    return delta.transpose(2, 3).reshape_as(base)


def gaussian(image, sigma):
    radius = int(4 * sigma + .5)
    offsets = torch.arange(-radius, radius + 1, device=image.device)
    weights = torch.exp(-.5 * (offsets / sigma).square())
    weights /= weights.sum()
    value = image[:, None]
    for axis in (-1, -2):
        size = value.shape[axis]
        indices = torch.arange(-radius, size + radius, device=value.device) % (2 * size)
        indices = torch.minimum(indices, 2 * size - 1 - indices)
        kernel = weights.reshape(1, 1, 1, -1) if axis == -1 else weights.reshape(1, 1, -1, 1)
        value = F.conv2d(value.index_select(axis, indices), kernel)
    return value[:, 0]


def highpass(image):
    return image - gaussian(image, .8)


def low_bands(image):
    # A short cosine taper confines this update to a prescribed passband.
    height, width = image.shape[-2:]
    dy = torch.fft.fftfreq(height, device=image.device)[:, None]
    dx = torch.fft.rfftfreq(width, device=image.device)
    radius = (dy.square() + dx.square()).sqrt()
    spectrum = torch.fft.rfft2(image)
    return [
        torch.fft.irfft2(
            spectrum * ((1 + torch.cos(torch.pi * ((radius / cutoff - .75) / .25).clamp(0, 1))) / 2),
            s=(height, width),
        )
        for cutoff in CUTOFFS
    ]


def align_detail(base, refined):
    # Estimate a small displacement from coarse predicted anatomy and confidence
    # from the structure tensor. Sample only once on the native grid.
    coarse = gaussian(base, 1.15)
    guide = gaussian(refined, 1.15)
    dy, dx = torch.gradient(coarse, dim=(-2, -1))
    difference = (guide - coarse).clamp(-.08, .08)
    xx, yy, xy, bx, by = [gaussian(value, 2.) for value in
                           (dx.square(), dy.square(), dx * dy, dx * difference, dy * difference)]
    xx, yy = xx + .00015, yy + .00015
    det = xx * yy - xy.square()
    sx = ((yy * bx - xy * by) / det).clamp(-1, 1)
    sy = ((xx * by - xy * bx) / det).clamp(-1, 1)
    depth, height, width = base.shape
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, height, device=base.device),
                            torch.linspace(-1, 1, width, device=base.device), indexing="ij")
    grid = torch.stack((gx + sx * (2 / (width - 1)), gy + sy * (2 / (height - 1))), dim=-1)
    detail = highpass(base)
    shifted = F.grid_sample(detail[:, None], grid, padding_mode="border", align_corners=True)[:, 0]
    # Restore local detail amplitude lost in the fractional resampling itself.
    energy_ratio = ((gaussian(detail.square(), 2.) + 1e-6) /
                    (gaussian(shifted.square(), 2.) + 1e-6)).sqrt().clamp(.75, 1.65)
    return shifted * energy_ratio - detail


def core_mask(target):
    edge = regions(target)[2]
    _, height, width = edge.shape
    core = np.zeros_like(edge)
    core[:, round(37 * height / 128):round(100 * height / 128),
         round(37 * width / 128):round(100 * width / 128)] = True
    return edge & core


def gram(parts):
    flat = torch.stack(parts).flatten(1)
    return (flat @ flat.T / flat.shape[1]).cpu().numpy().astype(np.float64)


def readouts(predictions, pick, families):
    base = predictions["base"]
    refined = base.lerp(predictions[pick["arm"]], pick["strength"])
    if families == "patch":
        updates = [project_patch_bands(base, refined, size, weight)
                   for size, weight in itertools.product(PATCH_SIZES, PATCH_WEIGHTS)]
    elif families == "protected":
        updates = protected_updates(base, refined)
    elif families == "constrained":
        updates = constrained_residuals(base, refined)
    else:
        updates = [*low_bands(refined - base), align_detail(base, refined)]
    return base, refined, updates


def moments(predictions, target, target_array, pick, families):
    base, _, updates = readouts(predictions, pick, families)
    core = torch.as_tensor(core_mask(target_array), device=target.device)
    detail = highpass(target)[core]
    output_bands = [highpass(value)[core] for value in [base, *updates]]
    return (gram([base - target, *updates]),
            gram(output_bands),
            gram([output_bands[0] - detail, *output_bands[1:]]),
            float(detail.square().mean()))


def make_candidates(families):
    if families != "spectral":
        arm, options = {
            "patch": ("patch_projection", list(itertools.product(PATCH_SIZES, PATCH_WEIGHTS))),
            "protected": ("protected_updates", PROTECTION_LEVELS),
            "constrained": ("constrained_residual", CONSTRAINT_LEVELS),
        }[families]
        specifications = [("prototype", None, 0., 0.)]
        weights = (1.,) if families == "patch" else LOW_WEIGHTS
        specifications += [(arm, index, weight, 0.)
                           for index in range(len(options)) for weight in weights]
        coefficients = np.zeros((len(specifications), len(options) + 1), dtype=np.float64)
        coefficients[:, 0] = 1.
        for index, (_, patch_index, weight, _) in enumerate(specifications[1:], 1):
            coefficients[index, 1 + patch_index] = weight
        return specifications, coefficients
    specifications = [("prototype", None, 0., 0.)]
    specifications += [("low_only", cutoff, weight, 0.)
                       for cutoff, weight in itertools.product(CUTOFFS, LOW_WEIGHTS)]
    specifications += [("low_and_align", cutoff, weight, shift) for cutoff, weight, shift
                       in itertools.product(CUTOFFS, LOW_WEIGHTS, SHIFT_WEIGHTS)]
    coefficients = np.zeros((len(specifications), len(CUTOFFS) + 2), dtype=np.float64)
    coefficients[:, 0] = 1.
    for index, (_, cutoff, weight, shift) in enumerate(specifications[1:], 1):
        coefficients[index, 1 + CUTOFFS.index(cutoff)] = weight
        coefficients[index, -1] = shift
    return specifications, coefficients


def screen(windows, model, device, state, output, families):
    specs, weights = make_candidates(families)
    summaries = {}
    for name, volume in windows.items():
        patients = []
        for index in range(len(volume.rows)):
            target_array = volume.targets[index, 1]
            prediction = forecast_volume(model, volume, index, 1, device)
            matrices = moments(prediction, torch.as_tensor(target_array.copy(), device=device),
                               target_array, state["validation"][name], families)
            nrmse, strength, signed_error = [
                np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", weights, matrix, weights), 0))
                for matrix in matrices[:3]]
            norm = np.sqrt(matrices[-1])
            patients.append(np.stack([nrmse, strength / norm, signed_error / norm], axis=-1))
        summaries[name] = np.stack(patients).mean(axis=0)
        print(f"Validation complete for {name} ({len(patients)} patients)", flush=True)
    keys = ("nrmse", "core_edge_hf_ratio", "core_edge_hf_nrmse")
    records = []
    for i, (arm, cutoff, weight, shift) in enumerate(specs):
        per_domain = {name: dict(zip(keys, values[i].tolist())) for name, values in summaries.items()}
        relative = [values[i, 0] / values[0, 0] for values in summaries.values()]
        fidelity = [values[i, 2] / values[0, 2] for values in summaries.values()]
        floor = min(values[i, 1] / values[0, 1] for values in summaries.values())
        if arm == "patch_projection":
            size, projection_amount = list(itertools.product(PATCH_SIZES, PATCH_WEIGHTS))[cutoff]
            settings = {"update_index": cutoff, "size": size, "projection_amount": projection_amount}
        elif arm == "protected_updates":
            settings = {"update_index": cutoff, "edge_quantile": PROTECTION_LEVELS[cutoff]}
        elif arm == "constrained_residual":
            settings = {"update_index": cutoff, "edge_quantile": CONSTRAINT_LEVELS[cutoff]}
        else:
            settings = {"cutoff": cutoff, "shift": shift}
        records.append({"arm": arm, **settings, "weight": weight,
                        "metrics": per_domain, "mean_relative_nrmse": float(np.mean(relative)),
                        "mean_relative_hf_error": float(np.mean(fidelity)),
                        "all_nrmse_improve": bool(max(relative) < 1.), "min_relative_hf_strength": float(floor)})
    selected = {}
    # Fit shared parameters on validation only. Use a 0.5% safety margin
    # for the per-domain mean, and report the best unconstrained floor
    # diagnostic if no candidate can reach it.
    for arm in dict.fromkeys(spec[0] for spec in specs[1:]):
        family = [entry for entry in records if entry["arm"] == arm]
        feasible = [entry for entry in family if entry["all_nrmse_improve"]
                    and entry["min_relative_hf_strength"] >= 1.005]
        chosen = (min(feasible, key=lambda row: row["mean_relative_nrmse"]) if feasible else
                  max(family, key=lambda row: (row["min_relative_hf_strength"], -row["mean_relative_nrmse"])))
        selected[arm] = {**chosen, "passes_validation": bool(feasible)}
    write_json(output / "validation.json", {"selected": selected, "candidates": records})
    print(json.dumps(selected), flush=True)
    return selected


def evaluate(windows, model, device, state, output, selected, families):
    rows = []
    for name, volume in windows.items():
        for index, patient in enumerate(volume.patients()):
            query = 1
            target = volume.targets[index, query].copy()
            prediction = forecast_volume(model, volume, index, query, device)
            base, refined, updates = readouts(prediction, state["validation"][name], families)
            outputs = {"prototype": base, "refined": refined}
            for arm, parameters in selected.items():
                if "update_index" in parameters:
                    outputs[arm] = base + parameters["weight"] * updates[parameters["update_index"]]
                else:
                    band = updates[CUTOFFS.index(parameters["cutoff"])]
                    outputs[arm] = base + parameters["weight"] * band + parameters["shift"] * updates[-1]
            for arm, tensor in outputs.items():
                image = tensor.cpu().numpy()
                native = (np.moveaxis(image, 0, -1), np.moveaxis(target, 0, -1)) if name == "lumiere" else (image, target)
                rows.append({"dataset": name, "patient_id": patient, "method": arm,
                             **score(*native, name, device)})
            if index == len(volume.rows) // 2:
                for arm in selected:
                    _save_panel(output / f"{name}_{arm}.png", outputs[arm].cpu().numpy(),
                                target, base.cpu().numpy(), patient, name.upper(), reference_title="previous prototype")
        print(f"Test complete for {name} ({len(volume.rows)} patients)", flush=True)
    with (output / "test_per_patient.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--families", choices=("spectral", "patch", "protected", "constrained"), default="spectral")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    # Use full precision for the small validation covariance matrices.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    state = torch.load(args.checkpoint, weights_only=True, map_location=device)
    model = SharedForecast(state["width"]).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(state["model"])
    model.eval()
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    development, test = ({name: Windows(args.cache, name, split) for name in DATASETS}
                         for split in ("validation", "test"))
    for name in DATASETS:
        used = config["patients"][name]
        if (development[name].patients() != used["validation_ids"] or
                set(test[name].patients()) & set(used["train_ids"] + used["validation_ids"]) or
                any(volume.metadata["bank_sha256"] != used["training_bank_sha256"] or
                    volume.metadata["checkpoint_sha256"] != used["upstream_checkpoint_sha256"]
                    for volume in (development[name], test[name]))):
            raise ValueError(f"Frozen split or cache provenance mismatch for {name}")
    with torch.inference_mode():
        selection = screen(development, model, device, state, args.output, args.families)
        evaluate(test, model, device, state, args.output, selection, args.families)


if __name__ == "__main__":
    main()
