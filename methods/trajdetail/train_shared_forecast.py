"""Train one readout on the three frozen longitudinal benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from src.trajectory import shared_forecast_data
from src.trajectory.shared_forecast_data import CHANNELS, DATASETS
from methods.trajdetail.shared_forecast_model import SharedForecast, forecast_loss
from src.trajectory.train import seed_all
from src.trajectory.external_eval import _save_panel
from methods.trajdetail.train_external_curvature_prototype import score, summarize
from src.trajectory.train_unified import _autocast
from src.trajectory.validate_control import write_json


STRENGTHS = (0., .35, .70, 1.)
ARMS = ("fusion", "full")
DOMAIN_CODES = {name: np.eye(3, dtype=np.float32)[index]
                for group in (DATASETS, ("mu", "dlbs")) for index, name in enumerate(group)}


class Windows:
    def __init__(self, root, dataset, split):
        path = root / dataset / split
        meta = json.loads((path / "manifest.json").read_text())
        if (meta["dataset"], meta["split"], tuple(meta["channels"])) != (dataset, split, CHANNELS):
            raise ValueError(f"Unexpected cache protocol at {path}")
        self.metadata = meta
        self.rows = meta["windows"]
        self.inputs = np.load(path / "inputs.npy", mmap_mode="r", allow_pickle=False)
        self.targets = np.load(path / "targets.npy", mmap_mode="r", allow_pickle=False)
        self.dataset = dataset
        self.split = split
        self.depth = self.inputs.shape[-3]
        self.num_queries = self.targets.shape[1]
        self.windows_per_patient = meta.get("windows_per_patient", 2 if split == "train" else 1)
        self.condition = []
        for row in self.rows:
            observed = np.array(row["context_times"])[np.array(row["mask"]) > 0]
            if not observed.max() < min(row["target_times"]):
                raise ValueError(f"noncausal window {row['patient_id']}")
            domain = DOMAIN_CODES[dataset]
            self.condition.append(np.array([[row["elapsed"][q], row["interval"], len(observed) / 7.,
                                             observed[-1], row["target_times"][q], *domain]
                                            for q in range(self.num_queries)], dtype=np.float32))
        if len(self.inputs) != len(self.rows) or len(self.targets) != len(self.rows):
            raise ValueError(f"cache length mismatch at {path}")

    def pick(self, case, queries, sections, device, flips=()):
        depth = self.depth
        adjacent = np.clip(sections[:, None] + np.arange(-1, 2), 0, depth - 1)
        # Advanced indexing puts the sampled slice and neighborhood axes first.
        features = self.inputs[case, queries[:, None], :, adjacent].transpose(0, 2, 1, 3, 4)
        features = features.reshape(len(sections), -1, *features.shape[-2:])
        targets = self.targets[case, queries, sections][:, None]
        condition = self.condition[case][queries]
        z = 2. * sections.astype(np.float32) / (depth - 1) - 1.
        condition = np.column_stack((condition, z))
        axes = tuple(axis for axis, flip in zip((-2, -1), flips) if flip)
        features = np.ascontiguousarray(np.flip(features, axis=axes))
        targets = np.ascontiguousarray(np.flip(targets, axis=axes))
        return tuple(torch.from_numpy(value).to(device=device, dtype=torch.float32)
                     for value in (features, condition, targets))

    def patients(self):
        if self.split != "train":
            return [row["patient_id"] for row in self.rows]
        n = self.windows_per_patient
        ids = [row["patient_id"] for row in self.rows[::n]]
        if len(self.rows) != len(ids) * n or any(
                row["patient_id"] != ids[i // n] for i, row in enumerate(self.rows)):
            raise ValueError("training windows must be grouped by patient")
        return ids


class TrainingPool:
    def __init__(self, root, dataset, extra_root):
        self.primary = Windows(root, dataset, "train")
        self.extra = Windows(extra_root, dataset, "train") if extra_root else None
        self.dataset = dataset
        self.depth = self.primary.depth
        self.n_fixed = self.primary.windows_per_patient
        self.n_extra = self.extra.windows_per_patient if self.extra else 0
        self.windows_per_patient = self.n_fixed + self.n_extra
        if self.extra and (self.extra.patients() != self.primary.patients() or any(
                self.extra.metadata[key] != self.primary.metadata[key]
                for key in ("bank_sha256", "checkpoint_sha256"))):
            raise ValueError("additional windows must use the same training patient split and base checkpoint")

    def patients(self):
        return self.primary.patients()

    def pick(self, case, *args):
        patient, index = divmod(case, self.windows_per_patient)
        if index < self.n_fixed:
            return self.primary.pick(patient * self.n_fixed + index, *args)
        return self.extra.pick(patient * self.n_extra + index - self.n_fixed, *args)


def train_change_scale(windows):
    errors = []
    for i in range(len(windows.rows)):
        anchor = windows.inputs[i, -1, 0, ::4].astype("float32")
        target = windows.targets[i, -1, ::4]
        errors.append(float(np.square(anchor - target).mean()))
    return float(np.sqrt(np.mean(errors)))


def batches(windows, rng, n_slices, epoch, unbalanced=False):
    schedules = [rng.permutation(len(dataset.patients())) for dataset in windows]
    if unbalanced:
        assignments = [(i, int(patient)) for i, order in enumerate(schedules) for patient in order]
    else:
        # Each domain contributes equally even though the number of patients differs.
        assignments = [(i, int(order[position % len(order)]))
                       for position in range(max(map(len, schedules))) for i, order in enumerate(schedules)]
    for i in rng.permutation(len(assignments)):
        domain, patient = assignments[i]
        volume = windows[domain]
        n = volume.windows_per_patient
        window_id = patient * n + (epoch + patient) % n
        axes = np.arange(volume.depth)
        sections = rng.choice(axes, n_slices, replace=n_slices > len(axes))
        n_queries = volume.primary.num_queries if isinstance(volume, TrainingPool) else volume.num_queries
        queries = (np.zeros(n_slices, dtype=np.intp) if n_queries == 1
                   else rng.choice(2, n_slices, p=(1 / 3, 2 / 3)))
        flips = rng.random(2) < .5
        yield domain, window_id, queries, sections, flips


@torch.inference_mode()
def forecast_volume(model, windows, index, query, device, chunk=24):
    predictions = {key: [] for key in ("base", *ARMS)}
    for sections in np.array_split(np.arange(windows.depth), max(1, (windows.depth + chunk - 1) // chunk)):
        x, condition, _ = windows.pick(index, np.full(len(sections), query), sections, device)
        with _autocast(device, True):
            full, fusion = model(x, condition)
        for key, value in (("base", x[:, 4:5]), ("fusion", fusion), ("full", full)):
            predictions[key].append(value[:, 0])
    return {key: torch.cat(value) for key, value in predictions.items()}


@torch.inference_mode()
def validation(model, windows, device):
    model.eval()
    summaries = {}
    for volume in windows:
        errors = {key: np.zeros((len(volume.rows), len(STRENGTHS))) for key in ARMS}
        for i in range(len(volume.rows)):
            predictions = forecast_volume(model, volume, i, volume.num_queries - 1, device)
            truth = torch.tensor(volume.targets[i, -1], device=device)
            base = predictions["base"]
            for key in ARMS:
                candidate = torch.stack([base.lerp(predictions[key], amount) for amount in STRENGTHS])
                errors[key][i] = (candidate - truth).square().flatten(1).mean(1).sqrt().cpu().numpy()
        candidates = {key: dict(zip(map(str, STRENGTHS), values.mean(0).tolist()))
                      for key, values in errors.items()}
        arm, strength, best = min((key, float(amount), value) for key, amounts in candidates.items()
                                  for amount, value in amounts.items() if value == min(
                                      metric for values in candidates.values() for metric in values.values()))
        summaries[volume.dataset] = {"arm": arm, "strength": strength, "nrmse": best,
                                    "prototype_nrmse": candidates["full"]["0.0"], "candidates": candidates}
    return summaries


def train(args, device):
    paths = args.output
    paths.mkdir(parents=True, exist_ok=False)
    sources = (Path(__file__), Path(__file__).with_name("shared_forecast_model.py"),
               Path(shared_forecast_data.__file__))
    for source in sources:
        shutil.copyfile(source, paths / source.name)
    training = [TrainingPool(args.cache, name, args.extra_cache) for name in args.datasets]
    valid = [Windows(args.cache, name, "validation") for name in args.datasets]
    for trn, val in zip(training, valid, strict=True):
        if set(trn.patients()) & set(val.patients()):
            raise ValueError("patients overlap between training and validation")
    scales = np.asarray([train_change_scale(volume.primary) for volume in training])
    model = SharedForecast(args.width).to(device=device, memory_format=torch.channels_last)
    if args.initialize:
        initial = torch.load(args.initialize, map_location=device, weights_only=True)
        model.load_state_dict(initial["model"])
    if args.decoder_only:
        for module in (model.stem, model.level1, model.level2, model.level3):
            module.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * .08)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["change_scales"] = dict(zip(args.datasets, scales.tolist()))
    config["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    config["trainable_parameters"] = sum(parameter.numel() for parameter in trainable)
    config["patients"] = {pool.dataset: {"train": len(pool.patients()), "validation": len(val.patients()),
                                         "train_ids": pool.patients(), "validation_ids": val.patients(),
                                         "training_bank_sha256": pool.primary.metadata["bank_sha256"],
                                         "upstream_checkpoint_sha256": pool.primary.metadata["checkpoint_sha256"],
                                         "extra_bank_sha256": pool.extra.metadata["sampled_bank_sha256"] if pool.extra else None}
                          for pool, val in zip(training, valid, strict=True)}
    write_json(paths / "config.json", config)
    best = float("inf")
    history = []
    rng = np.random.default_rng(42)
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        losses = []
        for domain, case, query, sections, flips in batches(training, rng, args.slices, epoch, args.unbalanced):
            x, condition, target = training[domain].pick(case, query, sections, device, flips)
            x = x.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, True):
                prediction, fusion = model(x, condition)
                # The augmentation distribution already implements weights 1:2 for the two queries.
                loss = forecast_loss(prediction, fusion, target, float(scales.mean() / scales[domain]),
                                     args.hf_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        scheduler.step()
        row = {"epoch": epoch, "train_loss": float(torch.stack(losses).mean()),
               "seconds": round(time.monotonic() - started, 2)}
        if epoch == 1 or epoch % args.validate_every == 0 or epoch == args.epochs:
            values = validation(model, valid, device)
            val_score = float(np.mean([row["nrmse"] / row["prototype_nrmse"] for row in values.values()]))
            row["validation"] = values
            row["validation_mean_relative_nrmse"] = val_score
            if val_score < best:
                best = val_score
                path = paths / "selected.pt"
                temporary = path.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "epoch": epoch, "width": args.width,
                            "validation": values, "relative_nrmse": best}, temporary)
                temporary.replace(path)
        history.append(row)
        temporary_history = paths / "history.tmp"
        write_json(temporary_history, history)
        temporary_history.replace(paths / "history.json")
        print(json.dumps({**row, "validation": {name: {
            key: values[key] for key in ("arm", "strength", "nrmse", "prototype_nrmse")}
            for name, values in row.get("validation", {}).items()}}), flush=True)


@torch.inference_mode()
def test(args, device):
    config = json.loads((args.output / "config.json").read_text())
    state = torch.load(args.output / "selected.pt", map_location=device, weights_only=True)
    model = SharedForecast(state["width"]).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(state["model"])
    model.eval()
    report, details = {}, []
    for name in config["datasets"]:
        volume = Windows(args.cache, name, "test")
        used = config["patients"][name]
        if (set(volume.patients()) & set(used["train_ids"] + used["validation_ids"]) or
                volume.metadata["bank_sha256"] != used["training_bank_sha256"] or
                volume.metadata["checkpoint_sha256"] != used["upstream_checkpoint_sha256"]):
            raise ValueError("test split overlaps with development data or changed provenance")
        pick = state["validation"][name]
        for index, patient in enumerate(volume.patients()):
            for query in (0, 1):
                candidates = forecast_volume(model, volume, index, query, device)
                chosen = candidates["base"].lerp(candidates[pick["arm"]], pick["strength"])
                truth = volume.targets[index, query]
                if index == len(volume.rows) // 2 and query == 1:
                    _save_panel(args.output / f"{name}_representative_case.png", chosen.cpu().numpy(),
                                truth, candidates["base"].cpu().numpy(), patient,
                                f"{name.upper()} refinement", reference_title="previous prototype")
                for label, volume_data in (("shared", chosen), ("prototype", candidates["base"])):
                    prediction = volume_data.cpu().numpy()
                    if name == "lumiere":
                        prediction, target = np.moveaxis(prediction, 0, -1), np.moveaxis(truth, 0, -1)
                    else:
                        target = truth
                    details.append({"dataset": name, "patient_id": patient, "query": query + 1, "method": label,
                                    **score(prediction, target, name, device)})
        report[name] = {"patients": len(volume.rows), "config": pick, "metrics": {
            str(q): {label: summarize([{key: value for key, value in row.items() if key != "dataset"}
                                      for row in details if row["dataset"] == name and row["query"] == q
                                      and row["method"] == label]) for label in ("shared", "prototype")}
            for q in (1, 2)}}
        print(json.dumps({"dataset": name, "patients": len(volume.rows),
                          "test_nrmse": report[name]["metrics"]["2"]["shared"]["nrmse"]["mean"]}), flush=True)
    with (args.output / "test_per_patient.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    write_json(args.output / "test_summary.json", {"epoch": state["epoch"], "results": report})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--extra-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=DATASETS)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--slices", type=int, default=8)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--lr", type=float, default=.0006)
    parser.add_argument("--hf-weight", type=float, default=.25)
    parser.add_argument("--validate-every", type=int, default=2)
    parser.add_argument("--unbalanced", action="store_true")
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.slices, args.validate_every) <= 0 or args.width < 4 or args.width % 4:
        parser.error("positive epoch and slice counts and a width divisible by four are required")
    if args.decoder_only and not args.initialize:
        parser.error("decoder-only training needs a trained shared checkpoint")
    seed_all(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    if args.test:
        test(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
