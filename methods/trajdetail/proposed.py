"""Run the frozen trajectory and readout, with optional training of the readout."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.compare import cases, save_rows
from methods.trajdetail.audit_acdc_sharpness import ReplayProbe
from methods.trajdetail.evaluate_minimum_norm_detail import band_directions, forecasts, initial
from methods.trajdetail.evaluate_signed_readout import energy_moments, min_gain
from methods.trajdetail.evaluate_spectral_calibration import model_from
from src.trajectory.shared_forecast_data import temporal_predictors
from src.trajectory.train import seed_all
from methods.trajdetail.train_budgeted_detail import objective, outputs
from methods.trajdetail.train_external_curvature_prototype import Features, render, score
from src.trajectory.train_unified import _autocast
from methods.trajdetail.evaluate_detail_preservation import gaussian, highpass

ROOT = Path(__file__).resolve().parents[2]
MILESTONE = ROOT / "runs/milestones/2026-11-21_anatomical_edge_shared_seed42"


class CachedWindows:
    def __init__(self, records):
        self.records = records
        self.depth = records[0][1].shape[-3]

    def pick(self, case, queries, sections, device, flips=()):
        _, features, condition, target = self.records[case]
        indices = (np.asarray(sections)[:, None] + np.arange(-1, 2)).clip(0, self.depth-1)
        images = features[:, indices].permute(1, 0, 2, 3, 4).flatten(1, 2).float().to(device)
        labels = target[sections, None].to(device)
        conditions = condition.to(device)[None].expand(len(sections), -1)
        positions = torch.as_tensor(sections, dtype=conditions.dtype, device=device)
        conditions = torch.cat((conditions, (2 * positions / (self.depth-1) - 1)[:, None]), dim=1)
        axes = [axis for axis, flip in zip((-2, -1), flips) if flip]
        if axes:
            images, labels = images.flip(axes), labels.flip(axes)
        return images, conditions, labels


def load_trajectory(device):
    model = ReplayProbe(use_dino=True).to(device).eval()
    checkpoint = MILESTONE / "upstream/lumiere/benefit.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    model.capture_transport = True
    return model, Features("lumiere", device)


@torch.inference_mode()
def build_cache(records, device, cache, trajectory=None):
    cache.mkdir(exist_ok=True, parents=True)
    model, encoder = trajectory or load_trajectory(device)
    result = []
    for index, (case, vol, times) in enumerate(records, 1):
        path = cache / f"{case}.pt"
        if path.exists():
            result.append(torch.load(path, map_location="cpu", weights_only=True))
            continue
        history = vol[:2].movedim(-3, -1).contiguous()
        item = {"context": history.cpu().numpy(), "mask": np.ones(2),
                "context_times": times[:2].cpu().numpy(), "patient_id": case}
        descriptor = encoder.observed(item)
        with _autocast(device, True):
            rollout = model(history[None], times.new_ones(1, 2), times[None, :2],
                            times[None, 2:], last_features=descriptor)["predictions"]
        anchor, previous, trend, elapsed, interval = temporal_predictors(history, times[:2], times[2:])
        parts, = model.transport_parts
        streams = torch.cat((anchor, render(parts, "lumiere")[0], rollout[:, 0], previous,
                             trend, parts[3][0].float())).movedim(-1, -3).contiguous().half()
        # Freeze the previously trained LUMIERE indicators for both external MRI cohorts.
        condition = torch.stack((elapsed[0], interval, times.new_tensor(2 / 7), times[1],
                                 times[2], *times.new_tensor([0., 1., 0.])))
        record = case, streams.cpu(), condition.cpu(), vol[2].cpu()
        torch.save(record, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        result.append(record)
        if index % 20 == 0:
            print("trajectory cache", index, len(records), flush=True)
    return CachedWindows(result)


@torch.no_grad()
def run_one(detail, voxel, windows, i, device, config, query=0):
    base, prior, posterior, supports = forecasts(detail, voxel, windows, i, query, device, True)
    support = supports[config["causal_mask"]]
    directions, confidence, power = band_directions(base, prior, posterior, support)
    low = gaussian(posterior - prior, .9)
    before = initial(prior, posterior, (low, posterior - prior - low), confidence, config["policy"])
    direction = directions[config["policy"][0]]
    gain = min_gain(*energy_moments(highpass(before), highpass(direction), support),
                    power * config["level"] ** 2)
    return base, before, before + gain * direction


def test(detail, voxel, windows, dataset, device, output, config, names):
    rows = [[] for _ in names]
    for i, (case, _, _, truth) in enumerate(windows.records):
        estimates = run_one(detail, voxel, windows, i, device, config)
        for name, estimate, result in zip(names, estimates, rows):
            if not name:
                continue
            estimate = estimate.cpu().numpy()
            result.append({"method": name, "patient_id": case, **score(estimate, truth.numpy(), dataset, device)})
            if i == len(windows.records) // 2:
                np.save(output / f"{name}_example.npy", estimate)
    for name, result in zip(names, rows):
        if name:
            save_rows(output / f"{name}_test.csv", result)
            print(name, "TEST", np.mean([row["nrmse"] for row in result]), "n", len(result), flush=True)


@torch.no_grad()
def val_nrmse(detail, voxel, windows, device, config):
    detail.eval()
    voxel.eval()
    return float(np.mean([float((run_one(detail, voxel, windows, i, device, config)[-1]
                               - truth.to(device)).square().mean().sqrt())
                          for i, (_, _, _, truth) in enumerate(windows.records)]))


def fine_tune(detail, voxel, train, valid, device, config, epochs, output):
    seed_all(42)
    rng = np.random.default_rng(42)
    scale = float(np.sqrt(np.mean([(x[0].float() - y).square().mean().item() for _, x, _, y in train.records])))
    parameters = list(detail.parameters()) + list(voxel.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    best = val_nrmse(detail, voxel, valid, device, config)
    history = [{"epoch": 0, "val_nrmse": best}]
    checkpoint = output / "adapted_readout.pt"

    def save(epoch):
        torch.save({"detail": detail.state_dict(), "voxel": voxel.state_dict(),
                    "epoch": epoch, "validation_nrmse": best}, checkpoint)

    save(0)
    for epoch in range(1, epochs + 1):
        detail.train()
        voxel.train()
        for case in rng.permutation(len(train.records)):
            sections = rng.choice(train.depth, 10, replace=False)
            images, condition, target = train.pick(case, None, sections, device, rng.random(2) < .5)
            optimizer.zero_grad(set_to_none=True)
            base, details = outputs(detail, images, condition)
            _, volume = outputs(voxel, images, condition)
            loss = objective(details, target[:, 0], base, scale) + (
                volume - target[:, 0]).square().mean() / scale ** 2
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError(epoch)
            optimizer.step()
        nrmse = val_nrmse(detail, voxel, valid, device, config)
        history.append({"epoch": epoch, "val_nrmse": nrmse})
        print("adapted readout", history[-1], flush=True)
        if nrmse < best:
            best = nrmse
            save(epoch)
    state = torch.load(checkpoint, weights_only=True, map_location=device)
    detail.load_state_dict(state["detail"])
    voxel.load_state_dict(state["voxel"])
    if not np.isclose(val_nrmse(detail, voxel, valid, device, config), best, rtol=1e-6):
        raise ValueError("Best readout checkpoint did not replay")
    (output / "adapted_readout_history.json").write_text(json.dumps(history, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    protocol = json.loads((args.output / "manifest.json").read_text())
    splits = ("test", "train", "validation") if args.epochs else ("test",)
    volumes = {split: list(cases(protocol, args.cache, split, device)) for split in splits}
    trajectory = load_trajectory(device)
    config = json.loads((MILESTONE / "results/validation.json").read_text())["selected"]
    detail, voxel = (model_from(MILESTONE / "models" / branch / "selected.pt", device)
                     for branch in ("detail_constrained", "unrestricted"))
    windows = build_cache(volumes["test"], device, args.cache / "trajectory", trajectory)
    test(detail, voxel, windows, protocol["dataset"], device, args.output, config,
         ("frozen_trajectory", "frozen_voxel", "frozen_method"))
    if args.epochs:
        train, valid = (build_cache(volumes[split], device, args.cache / "trajectory", trajectory)
                        for split in ("train", "validation"))
        fine_tune(detail, voxel, train, valid, device, config, args.epochs, args.output)
        test(detail, voxel, windows, protocol["dataset"], device, args.output, config,
             (None, "adapted_voxel", "adapted_method"))


if __name__ == "__main__":
    main()
