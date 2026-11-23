"""Train the proposal through its final detail-constrained image readout."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from methods.trajdetail.evaluate_detail_preservation import core_mask, highpass
from methods.trajdetail.evaluate_phase_detail import fast_metrics, summary
from src.trajectory.external_eval import _save_panel
from src.trajectory.shared_forecast_data import DATASETS
from methods.trajdetail.shared_forecast_model import SharedForecast
from src.trajectory.train import seed_all
from src.metrics import score
from methods.trajdetail.train_shared_forecast import TrainingPool, Windows, batches, train_change_scale
from src.trajectory.train_unified import _autocast
from src.trajectory.validate_control import write_json


LEVELS = (.90, .97, 1.04, 1.11)


def patch_average(value):
    return F.avg_pool2d(value[:, None], 11, 1, 5, count_include_pad=False)[:, 0]


def budgeted_readout(base, proposal, ratio, *, confidence=0.):
    """Use the proposal's signed detail locally, with a prior for weak edges."""
    prior = highpass(base)
    predicted = highpass(proposal)
    prior_energy = patch_average(prior.square())
    prediction_energy = patch_average(predicted.square())
    product = patch_average(prior * predicted)
    coherence = (product / (prior_energy * prediction_energy + 1.e-11).sqrt()).clamp(0, 1)
    fallback = (1 - coherence) * .35
    carrier = predicted + fallback * prior * ((prediction_energy + 1.e-6) /
                                                (prior_energy + 1.e-6)).sqrt()
    carrier = carrier * F.max_pool2d((base > .06).float()[:, None], 7, 1, 3)[:, 0]
    # Solve the local quadratic energy constraint in the image metric's band.
    directional = highpass(carrier)
    power = patch_average(directional.square()).clamp_min(1.e-8)
    cross = patch_average(predicted * directional)
    excess = (ratio * ratio * prior_energy - prediction_energy).clamp_min(0.)
    # The stabilized positive root is the smallest allowed change.
    scale = (excess / ((cross.square() + power * excess + 1.e-16).sqrt() + cross)
             .clamp_min(1.e-8)).clamp(0, 3.)
    # Neighboring pixel budgets overlap, so smooth the coefficient before synthesis.
    if confidence:
        # Let reliable edges carry the budget. Mixing gains locally protects
        # fast changing or underdetermined structures from excess sharpening.
        reliability = (.15 + .85 * coherence).pow(confidence)
        amplitude = patch_average(scale * reliability) / patch_average(reliability).clamp_min(.03)
        coefficient = amplitude * reliability / patch_average(reliability).clamp_min(.03)
    else:
        coefficient = patch_average(scale)
    return proposal + coefficient * carrier


def objective(pred, target, base, change_scale):
    error = (pred - target).square().mean() / change_scale ** 2
    details = highpass(pred)
    expected = highpass(target)
    reference = highpass(base)
    aspect = (expected.abs() / (patch_average(expected.square()) + 1.e-5).sqrt()).clamp(0, 3) + 1
    foreground = F.max_pool2d((target > .06).float()[:, None], 5, 1, 2)[:, 0]
    weights = (aspect * foreground).detach()
    signed = (weights * (details - expected).square()).mean() / (weights * expected.square()).mean().clamp_min(1.e-5)
    energy = ((weights * details.square()).mean() / (weights * reference.square()).mean().clamp_min(1.e-7)).sqrt()
    band_limit = (.97 - energy).relu().square() + (energy - 1.03).relu().square()
    return error + .12 * signed + 1.5 * band_limit


def outputs(model, inputs, condition):
    base = inputs[:, 4].float()
    with _autocast(inputs.device, True):
        raw, _ = model(inputs, condition)
    return base, raw[:, 0].float()


def forecast(model, volume, index, query, device, ratios, *, confidence=0.):
    results = [[] for _ in range(len(ratios) + 1)]
    for sections in np.array_split(np.arange(volume.depth), max(1, (volume.depth + 15) // 16)):
        x, c, _ = volume.pick(index, np.full(len(sections), query), sections, device)
        base, proposal = outputs(model, x, c)
        results[0].append(base)
        for i, ratio in enumerate(ratios, 1):
            results[i].append(budgeted_readout(base, proposal, ratio, confidence=confidence))
    return [torch.cat(tensors) for tensors in results]


@torch.inference_mode()
def validate(model, validation, device, confidence):
    model.eval()
    results = {}
    for volume in validation:
        patients = []
        for index in range(len(volume.rows)):
            target_array = volume.targets[index, -1].copy()
            target = torch.as_tensor(target_array, device=device)
            mask = torch.as_tensor(core_mask(target_array), device=device)
            signed = highpass(target)[mask]
            scale = signed.square().mean().sqrt().clamp_min(1.e-8)
            candidates = forecast(model, volume, index, volume.num_queries - 1, device,
                                  LEVELS, confidence=confidence)
            patients.append(torch.stack([fast_metrics(im, target, mask, signed, scale)
                                         for im in candidates]).cpu().numpy())
        results[volume.dataset] = np.stack(patients).mean(axis=0)
    candidates = []
    for index, level in enumerate(LEVELS, 1):
        relative = np.stack([rows[index] / rows[0] for rows in results.values()])
        violation = max(0., .95 - float(relative[:, 1].min()), float(relative[:, 1].max()) - 1.05)
        candidates.append({"level": level, "benchmark": {
            name: rows[index].tolist() for name, rows in results.items()},
            "nrmse_ratio": float(relative[:, 0].mean()),
            "worst_nrmse_ratio": float(relative[:, 0].max()),
            "strength_ratios": dict(zip(results, relative[:, 1].tolist())),
            "hf_error_ratio": float(relative[:, 2].mean()),
            "violation": violation})
    return candidates


def ranking(entry):
    return (entry["violation"] > 0., entry["nrmse_ratio"] + 3 * entry["violation"],
            entry["worst_nrmse_ratio"])


def fit(model, training, validation, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr / 10)
    train_scales = [train_change_scale(pool.primary) for pool in training]
    rng = np.random.default_rng(42)
    checkpoint = args.output / "selected.pt"
    history, best = [], None
    for epoch in range(args.epochs + 1):
        started = time.monotonic()
        loss_items = []
        if epoch:
            model.train()
            for dataset, case, queries, sections, flips in batches(training, rng, args.slices, epoch):
                inputs, condition, target = training[dataset].pick(case, queries, sections, device, flips)
                optimizer.zero_grad(set_to_none=True)
                base, proposal = outputs(model, inputs, condition)
                prediction = budgeted_readout(base, proposal, 1., confidence=args.confidence)
                loss = objective(prediction, target[:, 0], base, train_scales[dataset])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"Nonfinite gradient at epoch {epoch}")
                optimizer.step()
                loss_items.append(float(loss.detach()))
            schedule.step()
        row = {"epoch": epoch, "loss": float(np.mean(loss_items)) if loss_items else None}
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            candidates = validate(model, validation, device, args.confidence)
            chosen = min(candidates, key=ranking)
            row["candidates"] = candidates
            row["selected"] = chosen
            if best is None or ranking(chosen) < ranking(best):
                best = chosen
                temp = checkpoint.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "selection": chosen, "epoch": epoch}, temp)
                temp.replace(checkpoint)
        row["seconds"] = round(time.monotonic() - started, 1)
        history.append(row)
        write_json(args.output / "history.json", history)
        print(json.dumps({key: value for key, value in row.items() if key != "candidates"}), flush=True)
    return torch.load(checkpoint, map_location=device, weights_only=True)


@torch.inference_mode()
def evaluate(model, test, args, state, device):
    model.load_state_dict(state["model"])
    model.eval()
    write_json(args.output / "selection.json", {"epoch": state["epoch"], **state["selection"]})
    rows = []
    for window in test:
        name = window.dataset
        for index, patient in enumerate(window.patients()):
            for query in (0, 1):
                base, final = forecast(model, window, index, query, device, (state["selection"]["level"],),
                                       confidence=args.confidence)
                target = window.targets[index, query].copy()
                if query == 1 and index == len(window.rows) // 2:
                    _save_panel(args.output / f"{name}_budgeted_detail.png", final.cpu().numpy(), target,
                                base.cpu().numpy(), patient, name.upper(), reference_title="prototype")
                for label, pred in (("prototype", base), ("budgeted_detail", final)):
                    image = pred.cpu().numpy()
                    arrays = tuple(np.moveaxis(vol, 0, -1) for vol in (image, target)) if name == "lumiere" else (image, target)
                    rows.append({"dataset": name, "patient_id": patient, "query": query + 1,
                                 "method": label, **score(*arrays, name, device)})
        print(f"Test completed for {name}", flush=True)
    for filename, records in (("test_per_patient", rows), ("comparison", summary(rows))):
        with (args.output / f"{filename}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--extra-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--validate-every", type=int, default=2)
    parser.add_argument("--slices", type=int, default=10)
    parser.add_argument("--lr", type=float, default=.00018)
    parser.add_argument("--confidence", type=float, default=0.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_all(42)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    width = state.get("width")
    if width is None:
        width = json.loads((args.checkpoint.parent / "config.json").read_text())["width"]
    model = SharedForecast(width).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(state["model"])
    training = [TrainingPool(args.cache, name, args.extra_cache) for name in DATASETS]
    validation, test = ([Windows(args.cache, name, split) for name in DATASETS]
                        for split in ("validation", "test"))
    for pool, val, tst in zip(training, validation, test, strict=True):
        identities = [set(volume.patients()) for volume in (pool, val, tst)]
        if any(identities[i] & identities[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"Patient split overlap for {val.dataset}")
        if any(pool.primary.metadata[k] != volume.metadata[k] for volume in (val, tst)
               for k in ("bank_sha256", "checkpoint_sha256")):
            raise ValueError(f"Mismatched cache provenance for {val.dataset}")
    write_json(args.output / "config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "seed": 42, "width": width,
        "loss_terms": ["normalized voxel MSE", "normalized signed high-frequency error", "relative band energy"],
    })
    chosen = fit(model, training, validation, args, device)
    evaluate(model, test, args, chosen, device)


if __name__ == "__main__":
    main()
