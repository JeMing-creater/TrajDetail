"""Train single-sampling readouts on the fixed ACDC patient and frame windows."""

import argparse
import csv
import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import torch

from methods.trajdetail.audit_acdc_sharpness import (MATCHED, ReplayProbe, aggregates, gradient,
                                   metrics, regions, stored_volumes)
from methods.trajdetail.boundary_profile import BoundaryProfile, BoundaryTransport
from methods.trajdetail.composed_readout import BandwiseReadout, ComposedReadout, TransportedDetailReadout, training_objective
from src.trajectory.external_data import ACDCJointDataset
from src.trajectory.train import seed_all
from methods.trajdetail.train_local_control import CONTROL_RUN, last_features
from src.trajectory.train_unified import _autocast, _input
from src.trajectory.validate_control import FixedWindows, batch_item, json_hash, write_json


BLENDS = (.25, .5, .75, 1.)
ARMS = {"pixel": 0., "pixel_edge": .08}


@torch.no_grad()
def inputs(model, item, cache, device):
    batch = batch_item(item)
    with _autocast(device, True):
        model(*_input(batch, device), last_features=last_features(item, cache, device))
    times = [parts[-1] for parts in model.transport_parts]
    values = torch.stack([torch.cat(parts[:-1], dim=1)[0] for parts in model.transport_parts])
    return values, times


def precalculate(model, dataset, cache, device, path):
    shape = (len(dataset), 2, 7, *dataset[0]["targets"].shape[-3:])
    values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    times = []
    for i in range(len(dataset)):
        value, elapsed = inputs(model, dataset[i], cache, device)
        values[i] = value.cpu().numpy()
        times.append(elapsed)
        if (i + 1) % 20 == 0:
            print(f"{path.stem}: {i + 1}/{len(dataset)}", flush=True)
    values.flush()
    return values, times


def load_cached_inputs(dataset, path):
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    expected = (len(dataset), 2, 7, *dataset[0]["targets"].shape[-3:])
    if values.shape != expected or values.dtype != np.float16:
        raise ValueError(f"Unexpected trajectory features in {path}")
    times = []
    for item in dataset:
        anchor_slot = np.flatnonzero(item["mask"])[-1]
        times.append(item["target_times"].astype(np.float64) - float(item["context_times"][anchor_slot]))
    return values, times


@torch.no_grad()
def select_transport_detail(reference, dataset, cached, device):
    candidates = {}
    for filtered in (True, False):
        # Evaluate the strongest candidate once and interpolate fractional variants.
        head = TransportedDetailReadout(reference, strength=1., filtered=filtered)
        candidates[filtered] = validate(head, dataset, cached, device)
    baseline = candidates[True]["0.0"]
    eligible = []
    for filtered, values in candidates.items():
        for strength, score in values.items():
            if (float(strength) > 0 and score["nrmse"] <= baseline["nrmse"] * 1.0025 and
                    score["edge_gradient_mae"] < baseline["edge_gradient_mae"] and
                    abs(1. - score["edge_strength_ratio"]) < abs(1. - baseline["edge_strength_ratio"])):
                eligible.append({"strength": float(strength), "filtered": filtered,
                                 "validation": score, "baseline": baseline})
    if not eligible:
        return None
    return min(eligible, key=lambda entry: entry["validation"]["edge_gradient_mae"])


def unpack(value, time, device, flips=()):
    axis = [dimension for dimension, flip in zip((-2, -1), flips) if flip]
    sample = torch.as_tensor(np.array(value), device=device, dtype=torch.float32)
    if axis:
        sample = sample.flip(axis)
    anchor, transported, shift, source, previous = sample.split((1, 1, 3, 1, 1), dim=1)
    if axis:
        signs = sample.new_tensor([-1 if flips[1] else 1, -1 if flips[0] else 1, 1])
        shift = shift * signs.view(1, 3, 1, 1, 1)
    return anchor, transported, shift, source, previous, sample.new_tensor(time).reshape(-1, 1, 1, 1, 1)


@torch.no_grad()
def validate(head, dataset, cached, device):
    head.eval()
    rows = []
    for i, item in enumerate(dataset):
        args = unpack(cached[0][i], cached[1][i], device)
        with _autocast(device, True):
            full = head(*args)[-1, 0].cpu().numpy()
        previous = args[-2][-1, 0].cpu().numpy()
        target = item["targets"][-1]
        masks = regions(target)
        for strength in (0., *BLENDS):
            prediction = previous + strength * (full - previous)
            rows.append({"method": str(strength), **metrics(prediction, target, masks)})
    return aggregates(rows)


def fit(dataset, cached, validation, cached_validation, device, output, epochs, build_head, arms):
    best = None
    for arm, edge_weight in arms.items():
        seed_all(42)
        head = build_head()
        parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=.001)
        rng = np.random.default_rng(42)
        for epoch in range(1, epochs + 1):
            head.train()
            losses = []
            for index in rng.permutation(len(dataset)):
                item = dataset[index]
                flips = tuple(rng.random(2) < .5)
                args = unpack(cached[0][index], cached[1][index], device, flips)
                target = batch_item(item, flips)["targets"][0, :, None].to(device)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, True):
                    loss = training_objective(head(*args), target, edge_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss during {arm} epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            stats = validate(head, validation, cached_validation, device)
            print(json.dumps({"arm": arm, "epoch": epoch, "loss": float(np.mean(losses)),
                              "baseline": stats["0.0"]["nrmse"],
                              "validation": {k: {n: round(v[n], 7) for n in
                                                 ("nrmse", "edge_gradient_mae", "edge_strength_ratio")}
                                             for k, v in stats.items()}}, separators=(",", ":")), flush=True)
            baseline = stats["0.0"]
            for strength in BLENDS:
                score = stats[str(strength)]
                if (score["nrmse"] > baseline["nrmse"] * 1.0025 or
                        score["edge_gradient_mae"] > baseline["edge_gradient_mae"] or
                        abs(1. - score["edge_strength_ratio"]) >= abs(1. - baseline["edge_strength_ratio"])):
                    continue
                rank = (score["edge_gradient_mae"], abs(1. - score["edge_strength_ratio"]))
                if best is None or rank < tuple(best["rank"]):
                    best = {"arm": arm, "epoch": epoch, "strength": strength, "rank": rank,
                            "validation": score, "baseline": baseline}
                    torch.save(head.state_dict(), output / "selected.pt")
                    write_json(output / "selection.json", best)
    return best


@torch.no_grad()
def test(model, head, dataset, selection, cache, device, output, method="Composed readout",
         *, additional_heads=None):
    rows, images = [], []
    for index, item in enumerate(dataset):
        value, time = inputs(model, item, cache, device)
        args = unpack(value.cpu().numpy(), time, device)
        target = item["targets"][-1]
        anchor = args[0][-1, 0].cpu().numpy()
        stored = stored_volumes(item["patient_id"], target, anchor)
        previous = stored["Current"]
        np.testing.assert_allclose(args[-2][-1, 0].cpu().numpy(), previous, atol=2e-6, rtol=1e-6)
        with _autocast(device, True):
            full = head(*args)[-1, 0].cpu().numpy()
            additional = {name: other(*args)[-1, 0].cpu().numpy()
                          for name, other in (additional_heads or {}).items()}
        predictions = {"CRONOS": stored["CRONOS"], "Current": previous, **additional,
                       method: previous + selection["strength"] * (full - previous)}
        masks = regions(target)
        for label, prediction in predictions.items():
            rows.append({"patient_id": item["patient_id"], "method": label,
                         **metrics(prediction, target, masks)})
        copy_error = float(np.mean(np.square(target - anchor)))
        images.append((copy_error, item["patient_id"], target, anchor, predictions))
    with (output / "test_per_patient.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures(images, output)
    return aggregates(rows)


def figures(images, output):
    images.sort(key=lambda value: value[0])
    names = ("Target", *images[0][-1])
    fig, axes = plt.subplots(6, len(names), figsize=(2.25 * len(names), 11), layout="constrained")
    for index, rank in enumerate((.25, .5, .75)):
        _, patient, target, anchor, predictions = images[int(len(images) * rank)]
        center = range(len(target) // 4, 3 * len(target) // 4)
        z = max(center, key=lambda d: float(np.abs(target[d] - anchor[d]).sum()))
        imgs = {"Target": target, **predictions}
        cut = (z, slice(37, 100), slice(37, 100))
        high = float(np.quantile(target[z][target[z] > .06], .995))
        high_gradient = float(np.quantile(gradient(target[cut])[2], .985))
        for column, name in enumerate(names):
            a, b = axes[2 * index:2 * index + 2, column]
            a.imshow(imgs[name][cut], cmap="gray", vmin=0, vmax=high, interpolation="nearest")
            b.imshow(gradient(imgs[name][cut])[2], cmap="magma", vmin=0, vmax=high_gradient,
                     interpolation="nearest")
            if index == 0:
                a.set_title(fill(name, width=23), fontsize=11)
            if column == 0:
                a.set_ylabel(f"{patient}, z={z}")
                b.set_ylabel("Gradient")
            for axis in (a, b):
                axis.set(xticks=[], yticks=[])
    fig.suptitle("ACDC | Matched test cases at three levels of motion")
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument("--curvature", action="store_true")
    parser.add_argument("--reuse-inputs", type=Path)
    parser.add_argument("--bandwise-reference", type=Path)
    parser.add_argument("--detail-only-reference", type=Path)
    parser.add_argument("--boundary-profile", choices=("shift", "width", "joint"))
    parser.add_argument("--boundary-transport", choices=("center", "normal"))
    parser.add_argument("--transport-reference", type=Path)
    args = parser.parse_args()
    if bool(args.boundary_transport) != bool(args.transport_reference):
        parser.error("--boundary-transport and --transport-reference must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    config = json.loads((MATCHED / "config.json").read_text())
    bank = json.loads((CONTROL_RUN / "bank.json").read_text())
    if config["bank_sha256"] != json_hash(bank):
        raise ValueError("Training bank differs from the frozen model")
    cache = Path(config["cache"])
    model = ReplayProbe(use_dino=True).to(device).eval()
    model.load_state_dict(torch.load(MATCHED / "dinov3_matching/benefit.pt",
                                     map_location=device, weights_only=True)["model"])
    model.capture_transport = True
    training, validation = FixedWindows(bank), ACDCJointDataset(split="validation")
    arms = {"pixel_edge": .08} if args.boundary_profile or args.boundary_transport else ARMS
    write_json(args.output / "config.json",
               {"curvature": args.curvature, "epochs": args.epochs, "seed": 42,
                "bank_sha256": config["bank_sha256"], "reused_inputs": str(args.reuse_inputs),
                "nrmse_relative_tolerance": .0025, "edge_weights": arms,
                "boundary_profile": args.boundary_profile,
                "boundary_transport": args.boundary_transport,
                "transport_reference": str(args.transport_reference),
                "bandwise_reference": str(args.bandwise_reference),
                "detail_only_reference": str(args.detail_only_reference)})
    if args.reuse_inputs:
        cached = load_cached_inputs(training, args.reuse_inputs / "train_input.npy")
        cached_validation = load_cached_inputs(validation, args.reuse_inputs / "validation_input.npy")
    else:
        cached = precalculate(model, training, cache, device, args.output / "train_input.npy")
        cached_validation = precalculate(model, validation, cache, device, args.output / "validation_input.npy")
    def build_head():
        if args.boundary_profile:
            return BoundaryProfile(use_shift=args.boundary_profile in ("shift", "joint"),
                                   use_width=args.boundary_profile in ("width", "joint")).to(device)
        if args.boundary_transport:
            reference = ComposedReadout(curvature=True).to(device)
            state = torch.load(args.transport_reference, map_location=device, weights_only=True)
            reference.load_state_dict(state)
            return BoundaryTransport(reference, use_offsets=args.boundary_transport == "normal").to(device)
        composed = ComposedReadout(curvature=args.curvature).to(device)
        if args.bandwise_reference:
            state = torch.load(args.bandwise_reference, map_location=device, weights_only=True)
            composed.load_state_dict(state)
            return BandwiseReadout(composed).to(device)
        return composed

    if args.detail_only_reference:
        reference = build_head().eval()
        reference.load_state_dict(torch.load(args.detail_only_reference, map_location=device, weights_only=True))
        selection = select_transport_detail(reference, validation, cached_validation, device)
        if selection is not None:
            head = TransportedDetailReadout(reference, strength=1., filtered=selection["filtered"]).eval()
            write_json(args.output / "selection.json", selection)
    else:
        selection = fit(training, cached, validation, cached_validation, device, args.output,
                        args.epochs, build_head, arms)
        if selection is not None:
            head = build_head().eval()
            head.load_state_dict(torch.load(args.output / "selected.pt", map_location=device, weights_only=True))
    if selection is None:
        print("No learned readout reached the prespecified validation criteria", flush=True)
        write_json(args.output / "summary.json", {"selection": None})
        return
    method = "Composed readout"
    if args.boundary_profile:
        method = f"Boundary {args.boundary_profile}"
    if args.boundary_transport:
        method = f"Detail transport ({args.boundary_transport})"
    summary = {"selection": selection, "test": test(model, head, ACDCJointDataset(split="test"),
                                                    selection, cache, device, args.output, method)}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
