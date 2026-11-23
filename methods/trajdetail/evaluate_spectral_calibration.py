"""Calibrate coarse intensity from the unrestricted forecast after detail synthesis."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from methods.trajdetail.evaluate_detail_preservation import core_mask, gaussian, gram, highpass
from methods.trajdetail.evaluate_phase_detail import summary
from src.trajectory.shared_forecast_data import DATASETS
from methods.trajdetail.shared_forecast_model import SharedForecast
from methods.trajdetail.train_budgeted_detail import forecast, outputs
from src.metrics import score
from methods.trajdetail.train_shared_forecast import Windows
from src.trajectory.validate_control import write_json


def candidates():
    # Calibration stays low-dimensional and shared across modalities. The
    # unrestricted optimizer supplies changes the detail-constrained model
    # missed, while the coarse filter shields the recovered detail.
    return list(itertools.product((.97, .99, 1.015), (0.8, 1.5, 2.5, 4.),
                                  (0., .2, .4, .6, .8, 1.)))


def raw_volume(model, window, i, q, device):
    values = []
    for z in np.array_split(np.arange(window.depth), max(1, (window.depth + 15) // 16)):
        x, time, _ = window.pick(i, np.full(len(z), q), z, device)
        _, prediction = outputs(model, x, time)
        values.append(prediction)
    return torch.cat(values)


def validate(model, raw, windows, device, output):
    specs = candidates()
    levels, widths, strengths = (list(dict.fromkeys(entry[i] for entry in specs)) for i in range(3))
    coeff = np.array([[1., weight] for weight in strengths])
    domains = {}
    for window in windows:
        cases = []
        for index in range(len(window.rows)):
            base, *images = forecast(model, window, index, 1, device, levels)
            unrestricted = raw_volume(raw, window, index, 1, device)
            native = window.targets[index, 1].copy()
            target = torch.from_numpy(native).to(device)
            edge = torch.from_numpy(core_mask(native)).to(device)
            detail = highpass(target)[edge]
            norm = detail.square().mean().sqrt().item()
            origin = (float((base-target).square().mean().sqrt()),
                      float(highpass(base)[edge].square().mean().sqrt()) / norm)
            measured = []
            for image in images:
                detail_image = highpass(image)[edge]
                for width in widths:
                    change = gaussian(unrestricted - image, width)
                    bands = (gram((image-target, change)), gram((detail_image, highpass(change)[edge])))
                    measured.extend(np.stack([
                        np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", coeff, g, coeff), 0)) / (norm if j else 1.)
                        for j, g in enumerate(bands)], axis=1))
            cases.append((origin, measured))
        domains[window.dataset] = (np.mean([x[0] for x in cases], 0), np.mean([x[1] for x in cases], 0))
    results = []
    for index, (level, width, weight) in enumerate(specs):
        relative = {ds: scores[index] / original for ds, (original, scores) in domains.items()}
        r = np.stack(list(relative.values()))
        violation = max(0., .95 - float(r[:, 1].min()), float(r[:, 1].max()) - 1.05)
        results.append({"level": level, "width": width, "weight": weight,
                        "strength_violation": violation, "mean_relative_nrmse": float(r[:, 0].mean()),
                        "ratios": {ds: ratio.tolist() for ds, ratio in relative.items()}})
    selected = min(results, key=lambda row: (row["strength_violation"] > 0.,
                                             row["mean_relative_nrmse"] + 3 * row["strength_violation"]))
    write_json(output / "validation.json", {"selected": selected, "candidates": results})
    print(json.dumps(selected), flush=True)
    return selected


def test(model, raw, windows, device, output, selection):
    rows = []
    for window in windows:
        ds = window.dataset
        for index, patient in enumerate(window.patients()):
            for query in (0, 1):
                base, pred = forecast(model, window, index, query, device, (selection["level"],))
                unrestricted = raw_volume(raw, window, index, query, device)
                final = pred + selection["weight"] * gaussian(unrestricted - pred, selection["width"])
                target = window.targets[index, query].copy()
                for method, image in (("prototype", base), ("spectral_calibration", final)):
                    volumes = (image.cpu().numpy(), target)
                    if ds == "lumiere":
                        volumes = tuple(np.moveaxis(vol, 0, -1) for vol in volumes)
                    rows.append({"dataset": ds, "query": query + 1, "patient_id": patient,
                                 "method": method, **score(*volumes, ds, device)})
        print(f"Test completed for {ds}", flush=True)
    for filename, records in (("test_per_patient", rows), ("comparison", summary(rows))):
        with (output / f"{filename}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def model_from(path, device):
    checkpoint = torch.load(path, weights_only=True, map_location=device)
    width = json.loads((path.parent / "config.json").read_text())["width"]
    model = SharedForecast(width).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "checkpoint", "proposal", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    model, raw = (model_from(path, device) for path in (args.checkpoint, args.proposal))
    validation, held_out = ([Windows(args.cache, ds, split) for ds in DATASETS]
                            for split in ("validation", "test"))
    with torch.inference_mode():
        selected = validate(model, raw, validation, device, args.output)
        test(model, raw, held_out, device, args.output, selected)


if __name__ == "__main__":
    main()
