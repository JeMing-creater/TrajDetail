"""Preserve signed detail while selecting how much to update coarse anatomy."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from methods.trajdetail.evaluate_detail_preservation import core_mask, gaussian, highpass
from methods.trajdetail.evaluate_phase_detail import summary
from methods.trajdetail.evaluate_spectral_calibration import model_from, raw_volume
from src.trajectory.external_eval import _save_panel
from src.trajectory.shared_forecast_data import DATASETS
from methods.trajdetail.train_budgeted_detail import forecast
from src.metrics import score
from methods.trajdetail.train_shared_forecast import Windows
from src.trajectory.validate_control import write_json


def causal_core(base, proposal):
    """Use a forecast-defined foreground, eroding away the outer boundary."""
    support = ((base > .04) | (proposal > .04)).float()[:, None]
    local = F.avg_pool2d(support, 7, stride=1, padding=3, count_include_pad=False)
    return local[:, 0] > .75


def correction_directions(raw, own, base):
    high_raw, high_own, high_base = (highpass(x) for x in (raw, own, base))
    local_agreement = gaussian(high_raw * high_base, 1.5)
    local_power = (gaussian(high_raw.square(), 1.5) * gaussian(high_base.square(), 1.5)).sqrt()
    stable = (local_agreement / local_power.clamp_min(1.e-6)).clamp(0., 1.)
    return (
        high_raw,  # Supplies signed detail even where anatomy changes.
        high_own,
        high_raw * (.15 + .85 * stable),
        high_own * (.15 + .85 * stable),
    )


def history_directions(raw, own, base, history):
    """Place recovery where history supports the same anatomical detail."""
    last, earlier = history
    earlier_change = gaussian((last - earlier).abs(), 2.)
    anticipated_change = gaussian((raw - last).abs(), 2.)
    uncertainty = earlier_change + anticipated_change
    # Normalize per patient so the same rule applies to the three modalities.
    foreground = causal_core(base, raw)
    strength = torch.quantile(uncertainty[foreground], .75) if foreground.any() else uncertainty.mean()
    confidence = torch.exp(-uncertainty / strength.clamp_min(.003))
    guide = highpass(last)
    own_detail, new_detail = highpass(own), highpass(raw)
    agreements = [(gaussian(current * guide, 1.2) /
                   (gaussian(current.square(), 1.2) * gaussian(guide.square(), 1.2) + 1.e-10).sqrt()).clamp(0, 1)
                  for current in (own_detail, new_detail)]
    return (
        confidence * own_detail, confidence * new_detail,
        confidence * agreements[0] * own_detail, confidence * agreements[1] * new_detail,
    )


def causal_history(window, index, query, device):
    return [torch.from_numpy(window.inputs[index, query, j].copy()).to(device=device, dtype=torch.float32)
            for j in (0, 3)]


def through_plane(field):
    # Operate only on the estimated update so the original detail is untouched.
    previous = torch.cat((field[:1], field[:-1]))
    following = torch.cat((field[1:], field[-1:]))
    return (previous + following - 2. * field) / 4


def energy_moments(current, direction, mask):
    values = current[mask]
    updates = direction[mask]
    return values.square().mean(), (values * updates).mean(), updates.square().mean()


def min_gain(current_power, cross, carrier_power, required):
    gap = (required - current_power).clamp_min(0)
    root = (cross.square() + carrier_power * gap).sqrt() + cross
    return (gap / root.clamp_min(1.e-9)).clamp(0., 3.)


def coarse_fields(raw, own):
    delta = raw - own
    wide = gaussian(delta, 2.4)
    narrow = gaussian(delta, .9)
    return wide, narrow - wide, delta - narrow


def prespecified(history=False):
    if history:
        coefficients = np.array([[1., a, b, c, z] for a, b, c, z in itertools.product(
            (.8, 1.), (.75, 1.), (0., .35, .7), (0., .5, 1.))])
        return coefficients, np.array((.97, .99, 1.01, 1.03))
    coefficients = np.array([[1., a, b, c] for a, b, c in
                             itertools.product((.65, 1.), (.5, 1.), (0., .35, .7, 1.))])
    levels = np.array((.95, .97, .99, 1.01))
    return coefficients, levels


def readout_fields(raw, own, base, window, index, query, device, use_history):
    fields = coarse_fields(raw, own)
    directions = correction_directions(raw, own, base)
    if use_history:
        fields = (*fields, through_plane(raw - base))
        directions = (*directions, *history_directions(raw, own, base,
                                                       causal_history(window, index, query, device)))
    return (own, *fields), directions


def compare(model, raw_model, window, index, device, coefficients, levels, use_history):
    base, own = forecast(model, window, index, 1, device, (.99,))
    raw = raw_volume(raw_model, window, index, 1, device)
    target_array = window.targets[index, 1].copy()
    target = torch.from_numpy(target_array).to(device)
    tissue = torch.from_numpy(core_mask(target_array)).to(device)
    support = causal_core(base, raw)
    if not support.any():
        support = torch.ones_like(support)
    fields, directions = readout_fields(raw, own, base, window, index, 1, device, use_history)
    filtered = torch.stack([highpass(image) for image in fields])
    own_coarse = torch.einsum("ck,k...->c...", coefficients, torch.stack(fields))
    own_detail = torch.einsum("ck,k...->c...", coefficients, filtered)
    reference = highpass(base)[support].square().mean()
    target_detail = highpass(target)[tissue]
    target_energy = target_detail.square().mean().clamp_min(1.e-10)
    base_rms = (base - target).square().mean().sqrt()
    base_strength = (highpass(base)[tissue].square().mean() / target_energy).sqrt()
    coarse_error = (own_coarse - target)
    baseline_mse = coarse_error.square().mean(dim=(1, 2, 3))
    detail_tissue = own_detail[:, tissue]
    base_detail_power = detail_tissue.square().mean(1)
    moments = []
    for direction in directions:
        detail_direction = highpass(direction)
        evaluated = detail_direction[tissue]
        energy = detail_direction[support].square().mean()
        # The current and cross terms need one value for each proposal.
        current = own_detail[:, support].square().mean(1)
        cross = (own_detail[:, support] * detail_direction[support]).mean(1)
        gain = min_gain(current[:, None], cross[:, None], energy, reference * levels.square())
        mse = baseline_mse[:, None] + 2 * gain * (coarse_error * direction).mean((1, 2, 3))[:, None]
        mse += gain.square() * direction.square().mean()
        power = base_detail_power[:, None] + 2 * gain * (detail_tissue * evaluated).mean(1)[:, None]
        power += gain.square() * evaluated.square().mean()
        moments.append(torch.stack((mse.clamp_min(0.).sqrt(), (power / target_energy).clamp_min(0.).sqrt()), dim=-1))
    return torch.stack((base_rms, base_strength)).cpu().numpy(), torch.stack(moments).cpu().numpy()


def choose(model, raw, windows, device, output, use_history=False):
    coeff, levels = prespecified(use_history)
    coef_dev = torch.as_tensor(coeff, device=device, dtype=torch.float32)
    levels_dev = torch.as_tensor(levels, device=device, dtype=torch.float32)
    results = {}
    for window in windows:
        scores = [compare(model, raw, window, i, device, coef_dev, levels_dev, use_history)
                  for i in range(len(window.rows))]
        base = np.mean([entry[0] for entry in scores], axis=0)
        results[window.dataset] = np.mean([entry[1] for entry in scores], axis=0) / base
        print(f"Validation complete for {window.dataset}", flush=True)
    candidates = []
    for d, c, l in np.ndindex(8 if use_history else 4, len(coeff), len(levels)):
        relative = {ds: values[d, c, l].tolist() for ds, values in results.items()}
        values = np.array(list(relative.values()))
        violation = max(0., .95 - float(values[:, 1].min()), float(values[:, 1].max()) - 1.05)
        candidates.append({"direction": d, "coefficients": coeff[c, 1:].tolist(), "history": use_history,
                           "level": float(levels[l]), "ratios": relative, "violation": violation,
                           "nrmse_ratio": float(values[:, 0].mean())})
    selection = min(candidates, key=lambda row: (row["violation"] > 0.,
                                                row["nrmse_ratio"] + 3 * row["violation"]))
    write_json(output / "validation.json", {"selected": selection, "candidates": candidates})
    print(json.dumps(selection), flush=True)
    return selection


def evaluate(model, raw_model, windows, device, output, selection):
    rows = []
    for window in windows:
        ds = window.dataset
        for index, patient in enumerate(window.patients()):
            for q in (0, 1):
                base, own = forecast(model, window, index, q, device, (.99,))
                raw = raw_volume(raw_model, window, index, q, device)
                support = causal_core(base, raw)
                if not support.any():
                    support = torch.ones_like(support)
                fields, directions = readout_fields(raw, own, base, window, index, q, device,
                                                     selection.get("history", False))
                direction = directions[selection["direction"]]
                prediction = sum(c * field for c, field in
                                 zip((1., *selection["coefficients"]), fields, strict=True))
                moments = energy_moments(highpass(prediction), highpass(direction), support)
                budget = highpass(base)[support].square().mean() * selection["level"] ** 2
                prediction = prediction + min_gain(*moments, budget) * direction
                target = window.targets[index, q].copy()
                if q == 1 and index == len(window.rows) // 2:
                    _save_panel(output / f"{ds}_signed_readout.png", prediction.cpu().numpy(), target,
                                base.cpu().numpy(), patient, ds.upper(), reference_title="prototype")
                for name, pred in (("prototype", base), ("signed_readout", prediction)):
                    volumes = (pred.cpu().numpy(), target)
                    if ds == "lumiere":
                        volumes = tuple(np.moveaxis(x, 0, -1) for x in volumes)
                    rows.append({"dataset": ds, "query": q+1, "patient_id": patient,
                                 "method": name, **score(*volumes, ds, device)})
        print(f"Test complete for {ds}", flush=True)
    for name, records in (("test_per_patient", rows), ("comparison", summary(rows))):
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "checkpoint", "proposal", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    model, raw = (model_from(path, device) for path in (args.checkpoint, args.proposal))
    validation, held_out = ([Windows(args.cache, ds, split) for ds in DATASETS]
                            for split in ("validation", "test"))
    with torch.inference_mode():
        selection = choose(model, raw, validation, device, args.output, args.history)
        evaluate(model, raw, held_out, device, args.output, selection)


if __name__ == "__main__":
    main()
