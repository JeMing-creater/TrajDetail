"""Matched trajectory-control training for LUMIERE and ISLES'24."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.trajectory.data import ISLESJointDataset, sample_legacy_window
from src.trajectory.external_data import LumiereJointDataset, PROJECT_ROOT
from src.trajectory.external_eval import evaluate_metrics, save_evaluation
from src.trajectory.train import seed_all
from src.trajectory.train_unified import _autocast, _input, trajectory_loss
from src.trajectory.validate_control import (batch_item, cached_teacher, component_loss,
    data_hash, image_gradient_loss, pair_diagnostics, solve_one, teacher_args,
    teacher_components, write_json)


from src.paths import DATA

DEFAULTS = {
    "lumiere": DATA / "lumiere_protocol",
    "isles": DATA / "isles_protocol",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_dataset(dataset_name, data_root, split):
    root = Path(data_root or DEFAULTS[dataset_name])
    if dataset_name == "lumiere":
        return LumiereJointDataset(root, split, max_context=7)
    split_map = {"train": "trn", "validation": "val", "test": "tst"}
    return ISLESJointDataset(root, split_map[split], augment=False)


def _manifest_hash(dataset_name, data_root):
    root = Path(data_root or DEFAULTS[dataset_name])
    if dataset_name == "lumiere":
        return _sha256(root / "patient_split_manifest.json")
    return hashlib.sha256(( _sha256(root / "manifest.json") +
                            _sha256(root / "evaluation_samples.json")).encode()).hexdigest()


def _base_root(dataset_name, base):
    return base.data_root if dataset_name == "lumiere" else base.root


def window_protocol(dataset_name):
    if dataset_name == "lumiere":
        return "LUMIERE latest two visits; up to seven prior unique weeks as context"
    if dataset_name == "isles":
        return "ISLES frozen gap-3 windows; q-1 and q targets, q primary"
    raise ValueError(f"Unknown dataset: {dataset_name}")


def make_bank(dataset_name, data_root=None, windows_per_patient=2, seed=42):
    """Create a deterministic train-only bank matching the source loaders."""
    if dataset_name not in DEFAULTS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if windows_per_patient < 1:
        raise ValueError("windows_per_patient must be positive")
    base = _base_dataset(dataset_name, data_root, "train")
    rng = np.random.RandomState(seed)
    records = []
    for patient_index, case in enumerate(base.cases):
        patient_id = case["patient_id"]
        for window_number in range(windows_per_patient):
            if dataset_name == "lumiere":
                target_index = int(rng.randint(2, len(case["visits"])))
                window = {"target_index": target_index}
            else:
                window = sample_legacy_window(rng, frames=7, distance=3)
                window["missing_mask"] = window["missing_mask"].tolist()
            records.append({"bank_index": len(records), "patient_index": patient_index,
                "patient_id": patient_id, "window_number": window_number, "window": window})
    return {"schema": 1, "dataset": dataset_name, "seed": seed,
        "windows_per_patient": windows_per_patient, "split": "train",
        "manifest_sha256": _manifest_hash(dataset_name, data_root),
        "data_identity_scope": "manifest-only; source image array bytes are not hashed",
        "windows": records}


def rescore_saved_test_protocol(root):
    """Score saved ISLES outputs with CRONOS' default SSIM range, without inference."""
    test_root = Path(root) / "test"
    arms = {}
    reference_ids = None
    reference_targets = None
    for arm in ("endpoint", "gradient", "control"):
        files = sorted((test_root / arm / "volumes").glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no saved test volumes for {arm}")
        values = [np.load(path, allow_pickle=False) for path in files]
        patient_ids = [path.stem for path in files]
        targets = np.stack([value["target"] for value in values])
        if reference_ids is None:
            reference_ids, reference_targets = patient_ids, targets
        elif patient_ids != reference_ids or not np.array_equal(targets, reference_targets):
            raise ValueError("saved test cases/targets differ across arms")
        prediction = np.stack([value["prediction"] for value in values])
        last = np.stack([value["last_context"] for value in values])
        score = evaluate_metrics(prediction, targets, patient_ids)
        copy_score = evaluate_metrics(last, targets, patient_ids)
        arms[arm] = {"mean": score["mean"], "median": score["median"],
            "last_context_mean": copy_score["mean"],
            "last_context_median": copy_score["median"],
            "n_patients": score["n_patients"], "protocol": score["protocol"]}
    result = {"dataset": "ISLES24",
        "ssim": "TorchMetrics default data_range=None, matching CRONOS public evaluator",
        "predictions_rerun": False, "arms": arms}
    write_json(test_root / "cronos_metric_protocol.json", result)
    return result


def _lumiere_window_item(base, index, target_index):
    case = base.cases[index]
    visits = case["visits"]
    if not 2 <= target_index < len(visits):
        raise ValueError(f"Invalid LUMIERE target index {target_index}")
    history = visits[:target_index - 1][-base.max_context:]
    future = visits[target_index - 1:target_index + 1]
    shape = future[0][1].shape
    context = np.zeros((base.max_context, *shape), dtype=np.float32)
    mask = np.zeros(base.max_context, dtype=np.float32)
    context_times = np.zeros(base.max_context, dtype=np.float32)
    offset = base.max_context - len(history)
    for slot, (week, image) in enumerate(history, start=offset):
        context[slot] = image
        mask[slot] = 1.0
        context_times[slot] = week / 255.0
    targets = np.stack([image for _, image in future]).astype(np.float32)
    target_times = np.asarray([week / 255.0 for week, _ in future], dtype=np.float32)
    return {"context": context, "mask": mask, "context_times": context_times,
        "target_times": target_times, "targets": targets,
        "patient_id": case["patient_id"], "primary_index": target_index}


class FixedTrainingWindows(Dataset):
    def __init__(self, dataset_name, base, bank):
        self.dataset_name = dataset_name
        self.base = base
        self.bank = bank
        if bank["dataset"] != dataset_name or bank["manifest_sha256"] != _manifest_hash(dataset_name, _base_root(dataset_name, base)):
            raise ValueError("training window bank belongs to different data")
        for row in bank["windows"]:
            if row["patient_id"] != base.cases[row["patient_index"]]["patient_id"]:
                raise ValueError("bank patient order changed")

    def __len__(self):
        return len(self.bank["windows"])

    def __getitem__(self, bank_index):
        row = self.bank["windows"][bank_index]
        if self.dataset_name == "lumiere":
            return _lumiere_window_item(self.base, row["patient_index"],
                                        int(row["window"]["target_index"]))
        return self.base.window_item(row["patient_index"], row["window"])


def fixed_training_dataset(dataset_name, data_root, bank):
    base = _base_dataset(dataset_name, data_root, "train")
    if dataset_name == "lumiere":
        manifest_sha = _manifest_hash(dataset_name, data_root)
    else:
        manifest_sha = _manifest_hash(dataset_name, data_root)
    if bank["manifest_sha256"] != manifest_sha:
        raise ValueError("training bank manifest does not match current data")
    return FixedTrainingWindows(dataset_name, base, bank)


def _load_bank(args):
    root = Path(args.output)
    bank = make_bank(args.dataset, args.data_root, windows_per_patient=2, seed=42)
    path = root / "bank.json"
    if path.exists() and json.loads(path.read_text()) != bank:
        raise ValueError(f"stored training bank changed: {path}")
    if not path.exists():
        write_json(path, bank)
    return bank


def _student(arm, device):
    from methods.trajdetail.control import ControlTrajectory
    seed_all(42)
    return ControlTrajectory(capture_components=arm == "control").to(device)


def _step(model, batch, teacher, arm, device):
    with _autocast(device, device.type == "cuda"):
        result = model(*_input(batch, device))
        loss, parts = trajectory_loss(result, batch)
        extra = loss * 0
        if arm == "control":
            extra = component_loss(result, teacher)
        elif arm == "gradient":
            extra = image_gradient_loss(result["predictions"], batch["targets"].to(device))
        total = loss + .1 * extra
    return total, {**parts, "extra": extra.detach(), "total": total.detach()}


def _snapshot(output):
    output.mkdir(parents=True, exist_ok=True)
    source = output / "source_snapshot"
    source.mkdir(exist_ok=True)
    files = ("control_external.py", "control.py", "data.py", "external_data.py",
        "external_eval.py", "train.py", "train_unified.py", "validate_control.py")
    hashes = {}
    for name in files:
        path = Path(__file__).resolve().parent / name
        shutil.copyfile(path, source / name)
        hashes[name] = _sha256(path)
    return hashes


@torch.inference_mode()
def _evaluate(model, dataset, device, *, ssim_data_range):
    model.eval()
    predictions, targets, lasts, ids, times, diagnostics = [], [], [], [], [], []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        with _autocast(device, device.type == "cuda"):
            result = model(*_input(batch, device))
        prediction = result["predictions"][:, -1].float().cpu().numpy()[0]
        target = batch["targets"][:, -1].numpy()[0]
        last = result["last_context"].float().cpu().numpy()[0]
        delta_prediction, delta_target = (prediction - last).ravel(), (target - last).ravel()
        diagnostics.append({"patient_id": batch["patient_id"][0],
            "history_count": int(batch["mask"].sum()),
            "change_rms": float(np.sqrt(np.mean(delta_prediction ** 2))),
            "true_change_rms": float(np.sqrt(np.mean(delta_target ** 2))),
            "change_cosine": float(delta_prediction.dot(delta_target) /
                max(np.linalg.norm(delta_prediction) * np.linalg.norm(delta_target), 1e-12)),
            **{key: float(result[key]) for key in ("velocity_rms", "source_rms", "memory_derivative_rms")}})
        predictions.append(prediction); targets.append(target); lasts.append(last)
        ids.extend(batch["patient_id"]); times.append(batch["target_times"].numpy()[0])
    prediction = np.stack(predictions); target = np.stack(targets); last = np.stack(lasts)
    summary = evaluate_metrics(prediction, target, ids, ssim_data_range=ssim_data_range)
    summary["last_context"] = evaluate_metrics(last, target, ids,
        ssim_data_range=ssim_data_range)["mean"]
    summary["diagnostics"] = diagnostics
    summary["diagnostic_means"] = {key: float(np.mean([row[key] for row in diagnostics]))
        for key in diagnostics[0] if key not in ("patient_id", "history_count")}
    return summary, prediction, target, last, ids, np.stack(times)


def pilot(args):
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    bank = _load_bank(args)
    data = fixed_training_dataset(args.dataset, args.data_root, bank)
    indices = [row["bank_index"] for row in bank["windows"] if row["window_number"] == 0][:8]
    rows = []
    for index in indices:
        batch = batch_item(data[index])
        last_index = int(batch["mask"][0].nonzero()[-1, 0])
        anchor = batch["context"][:, last_index:last_index + 1]
        row = {"bank_index": index, "patient_id": batch["patient_id"][0],
            "last": pair_diagnostics(anchor.expand_as(batch["targets"]), batch["targets"], anchor)}
        for mode in ("joint", "source_only", "no_action"):
            value = solve_one(batch, args, mode)
            row[mode] = {**pair_diagnostics(value["predictions"], batch["targets"], anchor),
                "solver_seconds": value["timing_seconds"]}
        rows.append(row)
        write_json(root / "teacher_pilot_progress.json", rows)
        print(json.dumps({"teacher_pilot": row}), flush=True)
    means = {mode: {"latest_mse": float(np.mean([r[mode]["mse_by_time"][-1] for r in rows])),
        "edge_mse": float(np.mean([r[mode]["edge_mse"] for r in rows]))}
        for mode in ("last", "joint", "source_only", "no_action")}
    passed = (means["joint"]["latest_mse"] < .75 * means["last"]["latest_mse"]
        and means["joint"]["edge_mse"] < means["last"]["edge_mse"]
        and all(r["joint"]["mse_by_time"][-1] < r["last"]["mse_by_time"][-1] for r in rows))
    result = {"dataset": args.dataset, "bank_sha256": _sha256(root / "bank.json"),
        "indices": indices, "solver": teacher_args(args), "means": means,
        "representation_gate_passed": passed,
        "gate": "joint latest MSE <0.75*LCI; lower edge MSE; all selected windows improve",
        "gate_is_informational": True, "forecasting_evidence": False, "cases": rows}
    write_json(root / "teacher_pilot.json", result)
    print(json.dumps({"pilot_gate": passed, "means": means}), flush=True)


def cache(args):
    root = Path(args.output); bank = _load_bank(args)
    data = fixed_training_dataset(args.dataset, args.data_root, bank)
    timings = []
    for index in range(len(data)):
        batch = batch_item(data[index])
        value = cached_teacher(root, index, batch, args, allow_create=True)
        timings.append(value["timing_seconds"])
        if index % 10 == 0 or index + 1 == len(data):
            print(json.dumps({"teacher_cache": index + 1, "total": len(data),
                "solver_seconds_sum": sum(timings)}), flush=True)
    teachers = {path.name: _sha256(path) for path in sorted((root / "teachers").glob("*.pt"))}
    write_json(root / "teacher_cache_manifest.json", {"dataset": args.dataset,
        "bank_sha256": _sha256(root / "bank.json"), "count": len(teachers),
        "solver": teacher_args(args), "solver_seconds_sum": sum(timings), "files": teachers})


def train_arm(args):
    root = Path(args.output); bank = _load_bank(args)
    data = fixed_training_dataset(args.dataset, args.data_root, bank)
    if not (root / "teacher_cache_manifest.json").exists():
        raise RuntimeError("complete teacher cache before training the matched arms")
    output = root / "students" / args.arm
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    validation = _base_dataset(args.dataset, args.data_root, "validation")
    model = _student(args.arm, device)
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
        for p in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    order_rng = torch.Generator().manual_seed(42)
    flip_rng = np.random.RandomState(42)
    patient_count = len(data.base.cases)
    params = sum(p.numel() for p in model.parameters())
    config = {"dataset": args.dataset, "arm": args.arm, "seed": 42,
        "epochs": epochs, "updates": epochs * patient_count,
        "train_patients": patient_count, "windows_per_patient": 2,
        "parameters": params, "initial_state_sha256": initial_hash,
        "bank_sha256": _sha256(root / "bank.json"), "extra_weight": .1,
        "teacher_solver": teacher_args(args), "teacher_cost_excluded_from_student_budget": True,
        "history_weight": .25, "l1_weight": .01, "primary_weight": 2,
        "optimizer": "AdamW", "lr": 2e-4, "weight_decay": 1e-4,
        "selection": "minimum validation latest-target mean MSE, epochs 1/5/.../60",
        "data_root": str(Path(args.data_root or DEFAULTS[args.dataset]).resolve()),
        "manifest_sha256": _manifest_hash(args.dataset, args.data_root),
        "data_identity_scope": "manifest-only; source image array bytes are not hashed",
        "source_sha256": _snapshot(output), "test_evaluated": False}
    write_json(output / "config.json", config)
    fixed_windows = [row["bank_index"] for row in bank["windows"] if row["window_number"] == 0]
    teacher_memory = {}

    def get_teacher(index):
        if index not in teacher_memory:
            teacher_memory[index] = cached_teacher(root, index, batch_item(data[index]), args)
        return teacher_memory[index]

    best, best_epoch, history = float("inf"), None, []
    exposure = hashlib.sha256(); start_time = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train(); model.capture_components = args.arm == "control"
        order = torch.randperm(patient_count, generator=order_rng).tolist()
        sums = {}; tic = time.monotonic()
        for patient in order:
            index = patient * 2 + (epoch - 1) % 2
            flips = tuple(bool(x) for x in (flip_rng.random_sample(2) < .5))
            batch = batch_item(data[index], flips)
            exposure.update(f"{index}:{flips}:".encode()); exposure.update(data_hash(batch).encode())
            teacher = teacher_components(get_teacher(index), flips, device) if args.arm == "control" else None
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _step(model, batch, teacher, args.arm, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite student loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite student gradient")
            optimizer.step()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.) + float(value)
        scheduler.step()
        row = {"epoch": epoch, "train": {key: value / patient_count for key, value in sums.items()},
            "seconds": time.monotonic() - tic, "exposure_sha256": exposure.hexdigest()}
        if epoch == 1 or epoch % 5 == 0:
            model.capture_components = False
            summary, *_ = _evaluate(model, validation, device,
                ssim_data_range=1.0 if args.dataset == "isles" else None)
            mse = summary["mean"]["mse"]
            row["validation_latest_mse"] = mse
            if mse < best:
                best, best_epoch = mse, epoch
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch},
                           output / "best.pt")
        history.append(row); write_json(output / "history.json", history)
        print(json.dumps({"arm": args.arm, **row}), flush=True)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"]); model.capture_components = False
    summary, prediction, target, last, ids, times = _evaluate(model, validation, device,
        ssim_data_range=1.0 if args.dataset == "isles" else None)
    summary.update(best_epoch=best_epoch, best_validation_mse=best,
        elapsed_seconds=time.monotonic() - start_time,
        exposure_sha256=exposure.hexdigest(), parameters=params,
        initial_state_sha256=initial_hash, test_evaluated=False)
    np.savez_compressed(output / "validation_predictions.npz", prediction=prediction,
        target=target, last_context=last, patient_ids=np.asarray(ids), target_times=times)
    write_json(output / "summary.json", summary)
    print(json.dumps({"completed": args.arm, "validation": summary["mean"],
        "best_epoch": best_epoch, "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


def test(args):
    root = Path(args.output); bank = _load_bank(args)
    if not all((root / "students" / arm / "best.pt").exists()
               for arm in ("endpoint", "gradient", "control")):
        raise RuntimeError("all three validation-selected student checkpoints are required")
    device = torch.device(args.device); dataset = _base_dataset(args.dataset, args.data_root, "test")
    output = root / "test"
    output.mkdir(parents=True, exist_ok=False)
    from methods.trajdetail.control import ControlTrajectory
    results = {}
    for arm in ("endpoint", "gradient", "control"):
        arm_root = root / "students" / arm
        checkpoint = torch.load(arm_root / "best.pt", map_location=device, weights_only=False)
        model = ControlTrajectory(capture_components=False).to(device)
        model.load_state_dict(checkpoint["model"])
        summary, prediction, target, last, ids, times = _evaluate(model, dataset, device,
            ssim_data_range=1.0 if args.dataset == "isles" else None)
        evaluation = save_evaluation(output / arm, prediction, target, last, ids,
            ssim_data_range=1.0 if args.dataset == "isles" else None,
            title=f"{args.dataset.upper()} trajectory-control test: {arm}")
        summary["checkpoint_epoch"] = checkpoint["epoch"]
        summary["checkpoint_validation_mse"] = json.loads((arm_root / "summary.json").read_text())["best_validation_mse"]
        summary["test_selection"] = "checkpoint selected only on validation; test evaluated once"
        summary["protocol_note"] = ("Exploratory: same fixed test windows used by prior benchmark runs. "
            "ISLES preprocessing uses whole-series percentiles/min-max statistics." if args.dataset == "isles"
            else "Exploratory: target scans are normalized independently to [0,1], matching prior runs.")
        write_json(output / arm / "diagnostic_summary.json", summary)
        results[arm] = {"mean": evaluation["mean"], "median": evaluation["median"],
            "last_context_mean": evaluation["last_context"]["mean"],
            "checkpoint_epoch": checkpoint["epoch"],
            "diagnostic_means": summary["diagnostic_means"]}
        print(json.dumps({"test_arm": arm, **results[arm]}), flush=True)
    comparison = {"dataset": args.dataset, "arms": results,
        "patient_count": len(dataset), "bank_sha256": _sha256(root / "bank.json"),
        "test_selection": "validation-only checkpoint selection; one test pass",
        "window_protocol": window_protocol(args.dataset),
        "initial_metric_protocol": ("SSIM data_range=1.0; use cronos_metric_protocol.json for CRONOS-matched default SSIM"
                                    if args.dataset == "isles" else "TorchMetrics default SSIM range"),
        "development_status": "Exploratory: data preprocessing/test windows match prior benchmark runs."}
    write_json(output / "comparison.json", comparison)
    protocol_comparison = (rescore_saved_test_protocol(root) if args.dataset == "isles" else None)
    _write_report(root, comparison, protocol_comparison)


def _write_report(root, comparison, protocol_comparison=None):
    arms = comparison["arms"]
    if comparison["dataset"] == "isles" and protocol_comparison is not None:
        rows = [("LCI", protocol_comparison["arms"]["endpoint"]["last_context_median"], "-"),
            ("CRONOS epoch 80", {"mae": 0.011026379186660051, "nrmse": 0.04439223375887158,
                "ssim": 0.9789654016494751, "psnr": 27.055623054504395}, "-"),
            *[(arm, protocol_comparison["arms"][arm]["median"], arms[arm]["checkpoint_epoch"])
              for arm in ("endpoint", "gradient", "control")]]
        lines = ["# ISLES trajectory-control test results", "",
            "Metrics use patient medians; SSIM was rescored from saved predictions with CRONOS' default `data_range=None`. No model inference was rerun.",
            "The initial fixed-range results remain in `comparison.json`; this table uses `cronos_metric_protocol.json`.", "",
            "| Method | N | MAE | NRMSE | SSIM | PSNR | Best epoch |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for name, metrics, epoch in rows:
            lines.append(f"| {name} | 34 | {metrics['mae']:.6f} | {metrics['nrmse']:.6f} | {metrics['ssim']:.6f} | {metrics['psnr']:.3f} | {epoch} |")
        lines += ["", "![Control test case](control/representative_case.png)", ""]
        (root / "test" / "report.md").write_text("\n".join(lines))
        return
    lines = [f"# {comparison['dataset'].upper()} trajectory-control test results", "",
        "| Method | N | MAE | NRMSE | SSIM | PSNR | Best epoch |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("endpoint", "gradient", "control"):
        m = arms[arm]["mean"]
        lines.append(f"| {arm} | {comparison['patient_count']} | {m['mae']:.6f} | {m['nrmse']:.6f} | {m['ssim']:.6f} | {m['psnr']:.3f} | {arms[arm]['checkpoint_epoch']} |")
    lines += ["", "Patient-median metrics are also in `test/comparison.json`; ISLES literature comparisons should use medians.",
        "", "![Test case](endpoint/representative_case.png)", ""]
    (root / "test" / "report.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pilot", "cache", "train", "test"))
    parser.add_argument("--dataset", choices=tuple(DEFAULTS), required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--teacher-steps", type=int, default=150)
    parser.add_argument("--arm", choices=("endpoint", "gradient", "control"), default="endpoint")
    args = parser.parse_args(argv)
    if args.data_root is None:
        args.data_root = str(DEFAULTS[args.dataset])
    torch.set_num_threads(4)
    seed_all(42)
    {"pilot": pilot, "cache": cache, "train": train_arm, "test": test}[args.phase](args)


if __name__ == "__main__":
    main()
