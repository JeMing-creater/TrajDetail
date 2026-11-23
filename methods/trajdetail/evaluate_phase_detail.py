"""Screen detail decoders on validation, then score the selected arms on test."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from methods.trajdetail.evaluate_detail_preservation import core_mask, highpass
from src.trajectory.external_eval import _save_panel
from methods.trajdetail.phase_detail import LocalSpectrum, detail_direction, edge_projection
from src.trajectory.shared_forecast_data import DATASETS
from methods.trajdetail.shared_forecast_model import SharedForecast
from src.metrics import score
from methods.trajdetail.train_shared_forecast import Windows, forecast_volume
from src.trajectory.validate_control import write_json


METRICS = ("nrmse", "core_edge_hf_ratio", "core_edge_hf_nrmse")
ARMS = ("phase_only", "phase_amplitude", "signed_detail")


def settings():
    yield {"arm": "prototype"}
    yield {"arm": "proposal"}
    for cutoff, phase, amplitude in itertools.product(
            (.10, .16, .22, .28, .34), (0., .5, 1.), (-.15, 0., .25, .5)):
        if phase or amplitude:
            yield {"arm": "phase_amplitude" if amplitude else "phase_only",
                   "cutoff": cutoff, "phase": phase, "amplitude": amplitude}
        else:
            yield {"arm": "spectral_partition", "cutoff": cutoff, "phase": phase, "amplitude": amplitude}
    for trust, kernel, gain in itertools.product((0., .65, 1.25), (1.25, 2.5), (.20, .40, .60, .80, 1.)):
        yield {"arm": "signed_detail", "trust": trust, "kernel": kernel, "gain": gain}


def projection_settings():
    yield {"arm": "prototype"}
    yield {"arm": "proposal"}
    for anchor, proxy, ratio, passes, weight in itertools.product(
            ("prototype", "phase"), ("union", "prototype", "prediction"),
            (.9, .975, 1.05, 1.125), (0, 1), (0., .35, .65)):
        if anchor == "phase" and not weight:
            continue
        yield {"arm": "edge_projection", "anchor": anchor, "proxy": proxy,
               "ratio": ratio, "passes": passes, "weight": weight}


def predictions(base, proposal, configurations):
    spectrum = None
    directions = {}
    projected_anchors = {}
    for config in configurations:
        arm = config["arm"]
        if arm == "prototype":
            yield base
        elif arm == "proposal":
            yield proposal
        elif arm == "signed_detail":
            key = (config["kernel"], config["trust"])
            if key not in directions:
                directions[key] = detail_direction(base, proposal, *key)
            yield proposal + config["gain"] * directions[key]
        elif arm == "edge_projection":
            label = config["anchor"]
            if label not in projected_anchors:
                if label == "prototype":
                    projected_anchors[label] = highpass(base)
                else:
                    spectrum = spectrum or LocalSpectrum(base, proposal, 16)
                    projected_anchors[label] = highpass(spectrum.forecast(.16, .5, 0.))
            yield edge_projection(base, proposal, projected_anchors[label], config["weight"],
                                  config["proxy"], config["ratio"], config["passes"])
        else:
            if spectrum is None:
                spectrum = LocalSpectrum(base, proposal, 16)
            yield spectrum.forecast(config["cutoff"], config["phase"], config["amplitude"])


def predict(model, window, index, query, state, device):
    series = forecast_volume(model, window, index, query, device)
    pick = state["validation"][window.dataset]
    original = series["base"]
    proposal = original.lerp(series[pick["arm"]], pick["strength"])
    return original, proposal


def fast_metrics(prediction, truth, mask, truth_detail, denominator):
    intensity_error = (prediction - truth).square().mean().sqrt()
    detail = highpass(prediction)[mask]
    strength = detail.square().mean().sqrt() / denominator
    detail_error = (detail - truth_detail).square().mean().sqrt() / denominator
    return torch.stack((intensity_error, strength, detail_error))


def screen(model, validation, state, device, output, family):
    configs = list(projection_settings() if family == "projection" else settings())
    average = {}
    for window in validation:
        patients = []
        for index in range(len(window.rows)):
            original, proposed = predict(model, window, index, 1, state, device)
            target_array = window.targets[index, 1]
            target = torch.as_tensor(target_array.copy(), device=device)
            mask = torch.as_tensor(core_mask(target_array), device=device)
            target_detail = highpass(target)[mask]
            norm = target_detail.square().mean().sqrt()
            values = [fast_metrics(pred, target, mask, target_detail, norm)
                      for pred in predictions(original, proposed, configs)]
            patients.append(torch.stack(values).cpu().numpy())
        average[window.dataset] = np.stack(patients).mean(axis=0)
        print(f"Validation completed for {window.dataset}", flush=True)

    report = []
    for index, config in enumerate(configs):
        ratios = np.stack([rows[index] / rows[0] for rows in average.values()])
        report.append({"config": config, "metrics": {
            name: dict(zip(METRICS, rows[index].tolist())) for name, rows in average.items()},
            "nrmse_ratio": float(ratios[:, 0].mean()),
            "max_nrmse_ratio": float(ratios[:, 0].max()),
            "hf_error_ratio": float(ratios[:, 2].mean()),
            "min_strength_ratio": float(ratios[:, 1].min()),
            "max_strength_ratio": float(ratios[:, 1].max()),
        })
    selected = {}
    for arm in dict.fromkeys(config["arm"] for config in configs[2:]):
        candidates = [row for row in report if row["config"]["arm"] == arm]
        feasible = [row for row in candidates if row["min_strength_ratio"] >= .95
                    and row["max_strength_ratio"] <= 1.05]
        if feasible:
            chosen = min(feasible, key=lambda row: row["nrmse_ratio"])
        else:
            chosen = min(candidates, key=lambda row: max(.95 - row["min_strength_ratio"],
                                                         row["max_strength_ratio"] - 1.05))
        selected[arm] = {**chosen, "passes_validation_strength": bool(feasible)}
        print(json.dumps({arm: selected[arm]}), flush=True)
    write_json(output / "validation.json", {"selected": selected, "all_candidates": report})
    return selected


def summary(rows):
    records = []
    for keys in dict.fromkeys((row["dataset"], row["query"], row["method"]) for row in rows):
        selected = [row for row in rows if (row["dataset"], row["query"], row["method"]) == keys]
        dataset, query, method = keys
        records.append({"dataset": dataset, "query": query, "method": method, "n": len(selected), **{
            key: float(np.mean([row[key] for row in selected]))
            for key in (*METRICS, "ssim", "mae", "psnr")}})
    return records


def evaluate(model, test, state, device, output, selection):
    rows = []
    configs = [{"arm": "prototype"}, {"arm": "proposal"}] + [record["config"] for record in selection.values()]
    for window in test:
        name = window.dataset
        for index, patient in enumerate(window.patients()):
            for query in (0, 1):
                base, proposal = predict(model, window, index, query, state, device)
                target = window.targets[index, query].copy()
                for config, pred in zip(configs, predictions(base, proposal, configs), strict=True):
                    image = pred.cpu().numpy()
                    if query == 1 and index == len(window.rows) // 2 and config["arm"] in selection:
                        _save_panel(output / f'{name}_{config["arm"]}.png', image, target, base.cpu().numpy(),
                                    patient, name.upper(), reference_title="prototype")
                    arrays = tuple(np.moveaxis(vol, 0, -1) for vol in (image, target)) if name == "lumiere" else (image, target)
                    rows.append({"dataset": name, "patient_id": patient, "query": query + 1,
                                 "method": config["arm"], **score(*arrays, name, device)})
        print(f"Test completed for {name}", flush=True)
    for filename, records in (("test_per_patient", rows), ("comparison", summary(rows))):
        with (output / f"{filename}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--family", choices=("projection", "spectrum"), default="spectrum")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    upstream = json.loads((args.checkpoint.parent / "config.json").read_text())
    state = torch.load(args.checkpoint, weights_only=True, map_location=device)
    model = SharedForecast(state["width"]).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(state["model"])
    model.eval()
    validation, test = ([Windows(args.cache, dataset, split) for dataset in DATASETS]
                        for split in ("validation", "test"))
    for val, tst in zip(validation, test, strict=True):
        used = upstream["patients"][val.dataset]
        if (val.patients() != used["validation_ids"] or
            set(tst.patients()) & set(used["train_ids"] + used["validation_ids"]) or
            any(window.metadata["bank_sha256"] != used["training_bank_sha256"] or
                window.metadata["checkpoint_sha256"] != used["upstream_checkpoint_sha256"]
                for window in (val, tst))):
            raise ValueError(f"Forecast provenance changed for {val.dataset}")
    with torch.inference_mode():
        selection = screen(model, validation, state, device, args.output, args.family)
        evaluate(model, test, state, device, args.output, selection)


if __name__ == "__main__":
    main()
