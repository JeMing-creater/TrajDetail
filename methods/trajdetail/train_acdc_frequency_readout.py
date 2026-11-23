"""Train and audit a signed high-frequency readout on the fixed ACDC windows."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F

from methods.trajdetail.acdc_frequency_readout import FrequencyReadout, ResidualFlowReadout, band
from methods.trajdetail.audit_acdc_sharpness import MATCHED, ReplayProbe, aggregates, metrics, regions, stored_volumes
from methods.trajdetail.composed_readout import curvature_corrected_warp
from src.trajectory.external_data import ACDCJointDataset
from src.trajectory.train import seed_all
from methods.trajdetail.train_composed_readout import figures, inputs, load_cached_inputs, unpack
from methods.trajdetail.train_local_control import CONTROL_RUN
from src.trajectory.train_unified import _autocast
from src.trajectory.validate_control import FixedWindows, batch_item, json_hash, write_json

DETAIL_AMOUNTS = (.5, .75, 1., 1.25, 1.5)


def roi_weights(reference):
    weights = torch.full_like(reference, .20)
    weights[..., 37:100, 37:100] = 1.
    return weights


def objective(prediction, target, baseline, texture_weight):
    diff = prediction - target
    mse = diff.square().flatten(1).mean(1)
    budget = (baseline - target).square().flatten(1).mean(1)
    pred_detail, target_detail = band(prediction), band(target)
    weight = roi_weights(prediction) * (1. + (target_detail.abs() / .025).clamp(0, 2))
    signed_detail = (weight * torch.sqrt((pred_detail - target_detail).square() + .006**2)).mean()
    def local_rms(value):
        return F.avg_pool3d(value.square(), (1, 7, 7), stride=1, padding=(0, 3, 3),
                            count_include_pad=False).add(1e-6).sqrt()
    calibrated = roi_weights(prediction) * (local_rms(pred_detail) - local_rms(target_detail)).abs()
    return (mse.mean() + 1.8 * (mse - budget * 1.008).relu().mean() +
            .35 * (weight * diff.square()).mean() + .10 * signed_detail +
            texture_weight * calibrated.mean())


def signals(target):
    target_detail = target - gaussian_filter(target, (0, .8, .8))
    core = np.zeros(target.shape, dtype=bool)
    core[:, 37:100, 37:100] = True
    masks = regions(target)
    _, inside, edge = masks
    return target_detail, core & inside, core & edge, masks


def measure(prediction, target, reference):
    target_detail, core, core_edge, masks = reference
    detail = prediction - gaussian_filter(prediction, (0, .8, .8))
    energy = np.square(target_detail[core_edge]).sum(dtype=np.float64)
    stats = metrics(prediction, target, masks)
    stats.update({
        "core_edge_hf_ratio": np.sqrt(np.square(detail[core_edge]).sum(dtype=np.float64) / energy),
        "core_edge_hf_nrmse": np.sqrt(np.square((detail - target_detail)[core_edge]).sum(dtype=np.float64) / energy),
        "core_foreground_hf_rmse": np.sqrt(np.square((detail - target_detail)[core]).mean(dtype=np.float64)),
        "core_nrmse": np.sqrt(np.square((prediction - target)[core]).mean(dtype=np.float64)),
    })
    return stats


def summarize(rows):
    result = aggregates(rows)
    for method, summary in result.items():
        group = [row for row in rows if row["method"] == method]
        for key in ("core_edge_hf_ratio", "core_edge_hf_nrmse", "core_foreground_hf_rmse", "core_nrmse"):
            summary[key] = float(np.mean([entry[key] for entry in group]))
    return result


@torch.no_grad()
def validate(head, validation, cached, device):
    head.eval()
    rows = []
    for index, case in enumerate(validation):
        tensors = unpack(cached[0][index], cached[1][index], device)
        target = case["targets"][-1]
        reference = signals(target)
        original = tensors[-2][-1, 0].cpu().numpy()
        rows.append({"method": "Original", **measure(original, target, reference)})
        with _autocast(device, True):
            base, detail = head.components(*tensors)
        base, detail = base[-1, 0].cpu().numpy(), detail[-1, 0].cpu().numpy()
        for amount in DETAIL_AMOUNTS:
            rows.append({"method": str(amount), **measure(base + amount * detail, target, reference)})
    return summarize(rows)


def rank(stats, amount, minimum_gain=.035):
    base, candidate = stats["Original"], stats[str(amount)]
    excess = max(0., candidate["nrmse"] / base["nrmse"] - 1.005)
    hf_error = candidate["core_edge_hf_nrmse"] / base["core_edge_hf_nrmse"]
    deficit = max(0., minimum_gain - candidate["core_edge_hf_ratio"] + base["core_edge_hf_ratio"])
    if minimum_gain > .035:
        return bool(excess), excess, deficit, hf_error
    return bool(excess), excess, hf_error + .7 * deficit


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(replay, head, dataset, cache, device, output):
    rows, visuals = [], []
    label = "Frequency synthesis"
    for case in dataset:
        target = case["targets"][-1]
        reference = signals(target)
        value, elapsed = inputs(replay, case, cache, device)
        tensors = unpack(value.cpu().numpy(), elapsed, device)
        anchor = tensors[0][-1, 0].cpu().numpy()
        patient = case["patient_id"]
        stored = stored_volumes(patient, target, anchor)
        with _autocast(device, True):
            if head is not None:
                prediction = head(*tensors)[-1, 0].cpu().numpy()
        if head is None:
            composed = (tensors[1] + tensors[3])[-1, 0].cpu().numpy()
            curvature = (curvature_corrected_warp(tensors[0], tensors[2]) + tensors[3])[-1, 0].cpu().numpy()
            predictions = {"Last observation": anchor, "Recurrent warp": stored["Control"],
                           "DINO proposal": stored["DINO proposal"], "Original": stored["Current"],
                           "Single composed warp": composed, "Curvature-corrected composed warp": curvature,
                           "CRONOS": stored["CRONOS"]}
            comparisons = {"CRONOS": stored["CRONOS"], "Trajectory + DINOv3 gate": stored["Current"],
                           "Curvature-corrected composed warp": curvature}
            visuals.append((float(np.square(target - anchor).mean()), patient, target, anchor, comparisons))
        else:
            predictions = {"CRONOS": stored["CRONOS"], "Original": stored["Current"], label: prediction}
            visuals.append((float(np.square(target - anchor).mean()), patient, target, anchor, predictions))
        np.testing.assert_allclose(tensors[-2][-1, 0].cpu().numpy(), stored["Current"], atol=2e-6, rtol=1e-6)
        for name, image in predictions.items():
            rows.append({"patient_id": patient, "method": name, **measure(image, target, reference)})
    write_rows(output / "test_per_patient.csv", rows)
    if visuals:
        figures(visuals, output)
    return summarize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--features", type=Path, default=Path("runs/acdc_composed_readout_20261120"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--texture-weight", type=float, default=.20)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--align-detail", action="store_true")
    parser.add_argument("--residual-flow", action="store_true")
    parser.add_argument("--minimum-gain", type=float, default=.035)
    args = parser.parse_args()
    if args.align_detail and args.residual_flow:
        parser.error("--align-detail and --residual-flow configure separate readouts")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    seed_all(42)
    config = json.loads((MATCHED / "config.json").read_text())
    bank = json.loads((CONTROL_RUN / "bank.json").read_text())
    if config["bank_sha256"] != json_hash(bank):
        raise ValueError("Unexpected windows for frozen forecast features")
    replay = ReplayProbe(use_dino=True).to(device).eval()
    replay.load_state_dict(torch.load(MATCHED / "dinov3_matching/benefit.pt",
                                      weights_only=True, map_location=device)["model"])
    replay.capture_transport = True
    head, best = None, None
    if not args.audit_only:
        training, validation = FixedWindows(bank), ACDCJointDataset(split="validation")
        cached_train = load_cached_inputs(training, args.features / "train_input.npy")
        cached_val = load_cached_inputs(validation, args.features / "validation_input.npy")
        head = (ResidualFlowReadout() if args.residual_flow
                else FrequencyReadout(align_detail=args.align_detail)).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=.00065)
        rng = np.random.default_rng(42)
        for epoch in range(1, args.epochs + 1):
            head.train()
            losses = []
            for index in rng.permutation(len(training)):
                flips = tuple(rng.random(2) < .5)
                values = unpack(cached_train[0][index], cached_train[1][index], device, flips)
                target = batch_item(training[index], flips)["targets"][0, :, None].to(device)
                opt.zero_grad(set_to_none=True)
                with _autocast(device, True):
                    if args.residual_flow:
                        prediction, drift, true_drift = head.training_pair(values, target)
                        drift_loss = (roi_weights(target) * (drift - true_drift).square()).mean()
                    else:
                        prediction, drift_loss = head(*values), 0.
                    loss = objective(prediction, target, values[-2], args.texture_weight) + .65 * drift_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite frequency synthesis objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
                opt.step()
                losses.append(float(loss.detach()))
            scores = validate(head, validation, cached_val, device)
            amount = min(DETAIL_AMOUNTS, key=lambda scale: rank(scores, scale, args.minimum_gain))
            candidate = {"epoch": epoch, "amount": amount, "rank": rank(scores, amount, args.minimum_gain), "validation": scores}
            print(json.dumps({"epoch": epoch, "loss": np.mean(losses), "amount": amount,
                              "rank": candidate["rank"], "metrics": {name: {
                                  measure: round(data[measure], 6) for measure in
                                  ("nrmse", "core_edge_hf_ratio", "core_edge_hf_nrmse")}
                                  for name, data in scores.items()}}), flush=True)
            if best is None or candidate["rank"] < best["rank"]:
                best = candidate
                torch.save(head.state_dict(), args.output / "selected.pt")
                write_json(args.output / "selection.json", best)
        head.load_state_dict(torch.load(args.output / "selected.pt", weights_only=True, map_location=device))
        head.detail_amount = best["amount"]
        head.eval()
    summary = evaluate(replay, head, ACDCJointDataset(split="test"), Path(config["cache"]), device, args.output)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output / "summary.json", {"settings": settings, "seed": 42, "selection": best,
               "bank_sha256": config["bank_sha256"], "test": summary})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
