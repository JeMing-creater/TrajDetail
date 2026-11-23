"""Restore high-frequency energy through the gradient of a causal band norm."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from methods.trajdetail.evaluate_detail_preservation import core_mask, gaussian, gram, highpass
from methods.trajdetail.evaluate_phase_detail import summary
from methods.trajdetail.evaluate_signed_readout import causal_core, energy_moments, min_gain
from methods.trajdetail.evaluate_spectral_calibration import model_from, raw_volume
from src.trajectory.external_eval import _save_panel
from src.trajectory.shared_forecast_data import DATASETS
from methods.trajdetail.train_budgeted_detail import forecast
from src.metrics import score
from methods.trajdetail.train_shared_forecast import Windows
from src.trajectory.validate_control import write_json


def forecasts(model, raw_model, window, i, q, device, anatomical=False):
    base, own = forecast(model, window, i, q, device, (.99,))
    unrestricted = raw_volume(raw_model, window, i, q, device)
    support = causal_core(base, unrestricted)
    if not support.any():
        support = torch.ones_like(support)
    if anatomical:
        # Landmark masks come only from causal image forecasts.
        initial_masks = [
            torch.as_tensor(core_mask(image.cpu().numpy()), device=device)
            for image in (base, own, unrestricted)
        ]
        masks = (support, initial_masks[1] | initial_masks[2],
                 initial_masks[0] | initial_masks[1] | initial_masks[2])
        return base, own, unrestricted, tuple(mask if mask.any() else support for mask in masks)
    return base, own, unrestricted, (support,)


def band_directions(base, prior, posterior, support):
    raw_detail, own_detail = highpass(posterior), highpass(prior)
    window = gaussian(support.float(), .9)
    # First-order minimum-norm direction for ||mask * H(x)|| squared.
    gradient = highpass(raw_detail * window)
    reference_gradient = highpass(own_detail * window)
    base_energy = highpass(base)[support].square().mean().clamp_min(1e-10)
    raw_level = (raw_detail[support].square().mean() / base_energy).sqrt()
    raw_confidence = torch.sigmoid(11. * (raw_level - .69))
    return (raw_detail, gradient, reference_gradient,
            reference_gradient.lerp(gradient, raw_confidence)), raw_confidence, base_energy


def modes():
    # Blend signed proposals more freely if the unrestricted head already
    # preserves anatomy. Where it drops strong texture, lift it conservatively.
    return tuple(itertools.product(range(4), (0., .35, .7, 1.), (False, True)))


def initial(prior, posterior, bands, confidence, chosen):
    _, weight, dynamic = chosen
    amount = confidence * weight if dynamic else weight
    return prior + bands[0] + amount * bands[1]


@torch.inference_mode()
def select(model, raw, windows, device, output, domain_calibration=False, anatomical=False):
    policy = modes()
    levels = (.92, .95, .97, .99, 1.01, 1.03, 1.06)
    tensor_levels = torch.as_tensor(levels, device=device)
    measured = {}
    for window in windows:
        patients, bases = [], []
        for i in range(len(window.rows)):
            base, prior, posterior, supports = forecasts(model, raw, window, i, 1, device, anatomical)
            low = gaussian(posterior-prior, .9)
            bands = low, posterior - prior - low
            truth = window.targets[i, 1].copy()
            target = torch.from_numpy(truth).to(device)
            mask = torch.from_numpy(core_mask(truth)).to(device)
            scale = float(highpass(target)[mask].square().mean().sqrt())
            bases.append((float((base-target).square().mean().sqrt()),
                          float(highpass(base)[mask].square().mean().sqrt()) / scale))
            rows = []
            for support in supports:
                commands, confidence, power = band_directions(base, prior, posterior, support)
                scores = []
                for setting in policy:
                    direction = commands[setting[0]]
                    proposal = initial(prior, posterior, bands, confidence, setting)
                    current = highpass(proposal)
                    gradient = highpass(direction)
                    alpha = min_gain(*energy_moments(current, gradient, support),
                                     power * tensor_levels.square()).cpu().numpy()
                    coef = np.column_stack((np.ones_like(alpha), alpha))
                    fields = (gram((proposal-target, direction)), gram((current[mask], gradient[mask])))
                    scores.append(np.column_stack([
                        np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", coef, m, coef), 0.))
                        / (scale if idx else 1.) for idx, m in enumerate(fields)
                    ]))
                rows.append(scores)
            patients.append(rows)
        measured[window.dataset] = np.mean(patients, axis=0) / np.mean(bases, axis=0)
        print(f"Validation complete for {window.dataset}", flush=True)
    choices = []
    for a, i, j in np.ndindex(3 if anatomical else 1, len(policy), len(levels)):
        data = {name: values[a, i, j].tolist() for name, values in measured.items()}
        values = np.asarray(list(data.values()))
        violation = max(0., .95 - float(values[:, 1].min()), float(values[:, 1].max()) - 1.05)
        choices.append({"policy": policy[i], "level": levels[j], "causal_mask": a,
                        "validation_ratios": data,
                        "violations": violation, "nrmse_rel": float(values[:, 0].mean())})
    if domain_calibration:
        def domain_risk(row, name):
            error, strength = row["validation_ratios"][name]
            violation = max(0., .9625 - strength, strength - 1.0375)
            return violation > 0, error + 3 * violation

        selected = {name: min(choices, key=lambda row: domain_risk(row, name)) for name in measured}
    else:
        selected = min(choices, key=lambda row: (row["violations"] > 0.,
                                                 row["nrmse_rel"] + 3 * row["violations"]))
    write_json(output / "validation.json", {"selected": selected, "candidates": choices,
                                             "domain_calibration": domain_calibration})
    print(json.dumps(selected), flush=True)
    return selected


@torch.inference_mode()
def evaluate(model, raw, windows, device, output, selected):
    result = []
    for window in windows:
        dataset = window.dataset
        config = selected.get(dataset, selected)
        for i, patient in enumerate(window.patients()):
            for q in (0, 1):
                support_index = config.get("causal_mask", 0)
                base, prior, posterior, supports = forecasts(model, raw, window, i, q, device, support_index != 0)
                support = supports[support_index]
                directions, confidence, power = band_directions(base, prior, posterior, support)
                low = gaussian(posterior - prior, .9)
                before = initial(prior, posterior, (low, posterior-prior-low), confidence, config["policy"])
                direction = directions[config["policy"][0]]
                gain = min_gain(*energy_moments(highpass(before), highpass(direction), support),
                                power * config["level"]**2)
                after = before + gain * direction
                target = window.targets[i, q].copy()
                for method, volume in (("prototype", base), ("minimum_norm_detail", after)):
                    values = (volume.cpu().numpy(), target)
                    if dataset == "lumiere":
                        values = tuple(np.moveaxis(a, 0, -1) for a in values)
                    result.append({"dataset": dataset, "query": q + 1, "patient_id": patient,
                                   "method": method, **score(*values, dataset, device)})
                if q == 1 and i == len(window.rows) // 2:
                    _save_panel(output / f"{dataset}_minimum_norm_detail.png", after.cpu().numpy(), target,
                                base.cpu().numpy(), patient, dataset.upper(), reference_title="prototype")
        print(f"Test complete for {dataset}", flush=True)
    for name, entries in (("comparison", summary(result)), ("test_per_patient", result)):
        with (output / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "checkpoint", "proposal", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--domain-calibration", action="store_true")
    parser.add_argument("--anatomical", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    model, raw = (model_from(path, device) for path in (args.checkpoint, args.proposal))
    development, test = ([Windows(args.cache, ds, split) for ds in DATASETS]
                         for split in ("validation", "test"))
    selected = select(model, raw, development, device, args.output, args.domain_calibration, args.anatomical)
    evaluate(model, raw, test, device, args.output, selected)


if __name__ == "__main__":
    main()
