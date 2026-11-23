"""Training-only control teachers and matched causal forecasting experiments.

The fixed-window protocol deliberately differs from the archived random-window
run. Only the freshly matched endpoint arm is an attribution baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from src.trajectory.external_data import ACDCJointDataset, PROJECT_ROOT
from src.trajectory.train import seed_all
from src.trajectory.train_unified import _autocast, _input, trajectory_loss


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def component_loss(result, teacher, primary_weight=2.0):
    losses = []
    for key in ("transport_components", "source_components"):
        prediction = result[key]
        target = teacher[key].detach().to(prediction)
        if prediction.shape != target.shape:
            raise ValueError(f"component shapes differ for {key}")
        error = (prediction - target).square().flatten(2).mean(2)
        weights = error.new_ones(error.shape[1]); weights[-1] = primary_weight
        losses.append(((error * weights).sum(1) / weights.sum()).mean())
    return sum(losses)


def image_gradient_loss(prediction, target, primary_weight=2.0):
    differences = []
    for axis in (2, 3, 4):
        if prediction.shape[axis] > 1:
            differences.append((prediction.diff(dim=axis) - target.diff(dim=axis)).square().flatten(2).mean(2))
    if not differences:
        return prediction.sum() * 0
    error = torch.stack(differences).mean(0)
    weights = error.new_ones(error.shape[1]); weights[-1] = primary_weight
    return ((error * weights).sum(1) / weights.sum()).mean()


def make_bank(windows_per_patient=2, seed=42):
    data = ACDCJointDataset(split="train", augment=False)
    previous = np.random.get_state()
    np.random.seed(seed)
    records = []
    try:
        for index, (_, _, patient_id) in enumerate(data.sources):
            for number in range(windows_per_patient):
                window = data._sample_window()
                records.append({"bank_index": len(records), "patient_index": index,
                    "patient_id": f"patient{patient_id}", "window_number": number, "window": window})
    finally:
        np.random.set_state(previous)
    manifest = PROJECT_ROOT / "cronos_acdc_baseline/data_manifest.json"
    return {"schema": 1, "seed": seed, "windows_per_patient": windows_per_patient,
        "split": "train", "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "windows": records}


class FixedWindows(ACDCJointDataset):
    def __init__(self, bank):
        super().__init__(split="train", augment=False)
        self.bank = bank
        self.forced_window = None
        for row in bank["windows"]:
            if row["patient_id"] != f"patient{self.sources[row['patient_index']][2]}":
                raise ValueError("bank patient order changed")

    def _sample_window(self):
        if self.forced_window is None:
            raise RuntimeError("bank window must be selected explicitly")
        return self.forced_window

    def __getitem__(self, bank_index):
        record = self.bank["windows"][bank_index]
        self.forced_window = record["window"]
        try:
            value = super().__getitem__(record["patient_index"])
        finally:
            self.forced_window = None
        return value

    def __len__(self):
        return len(self.bank["windows"])


def batch_item(item, flips=(False, False)):
    batch = {}
    for key, value in item.items():
        if isinstance(value, np.ndarray):
            if key in ("context", "targets"):
                for axis, flip in zip((2, 3), flips):
                    if flip:
                        value = np.flip(value, axis=axis)
            batch[key] = torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0)
        elif key == "patient_id":
            batch[key] = [value]
        else:
            batch[key] = value
    return batch


def data_hash(batch):
    digest = hashlib.sha256(batch["patient_id"][0].encode())
    for key in ("context", "mask", "context_times", "target_times", "targets"):
        digest.update(batch[key].numpy().tobytes())
    return digest.hexdigest()


def teacher_args(args):
    return {"steps": args.teacher_steps, "lr_u": .15, "lr_r": .03,
        "lambda_u": 1e-5, "lambda_grad": 1e-4, "lambda_r": .01,
        "primary_weight": 2.0,
        "max_step": 1/11, "record_every": 25}


def solve_one(batch, args, mode="joint"):
    from methods.trajdetail.control import solve_teacher
    last = int(batch["mask"][0].nonzero()[-1, 0])
    return solve_teacher(batch["context"][:, last:last+1].to(args.device),
        float(batch["context_times"][0, last]), batch["target_times"][0].double().tolist(),
        batch["targets"].to(args.device), source_only=mode == "source_only",
        no_action=mode == "no_action", **teacher_args(args))


def cached_teacher(root, index, batch, args, *, allow_create=False):
    path = root / "teachers" / f"window_{index:04d}.pt"
    expected = {"data_sha256": data_hash(batch), "solver": teacher_args(args)}
    if path.exists():
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value["identity"] != expected:
            raise ValueError(f"teacher configuration/data mismatch: {path}")
        return value["teacher"]
    if not allow_create:
        raise FileNotFoundError(path)
    value = solve_one(batch, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": expected, "teacher": value}, path)
    return value


def teacher_components(teacher, flips, device):
    output = {}
    for key in ("transport_components", "source_components"):
        value = teacher[key]
        for axis, flip in zip((3, 4), flips):
            if flip:
                value = value.flip(axis)
        output[key] = value.to(device)
    return output


def pair_diagnostics(prediction, target, last):
    p, y, x = (v.detach().float().cpu() for v in (prediction, target, last))
    dp, dy = (p - x).flatten(2), (y - x).flatten(2)
    return {"mse_by_time": (p-y).square().flatten(2).mean(2)[0].tolist(),
        "mae_by_time": (p-y).abs().flatten(2).mean(2)[0].tolist(),
        "edge_mse": float(image_gradient_loss(p, y)),
        "change_rms": float(dp.square().mean().sqrt()),
        "true_change_rms": float(dy.square().mean().sqrt()),
        "change_cosine": float((dp*dy).sum() / (dp.norm()*dy.norm()).clamp_min(1e-12))}


def sources_snapshot(output):
    folder = output / "source_snapshot"; folder.mkdir(exist_ok=True)
    hashes = {}
    for name in ("validate_control.py", "control.py", "unified_model.py", "diagnose_unified.py",
                 "train_unified.py", "external_data.py", "external_eval.py", "train.py"):
        source = Path(__file__).parent / name
        shutil.copyfile(source, folder / name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    return hashes


def get_bank(root):
    path = root / "bank.json"
    bank = make_bank()
    if path.exists() and json.loads(path.read_text()) != bank:
        raise ValueError("stored window bank changed")
    if not path.exists():
        write_json(path, bank)
    return bank


def pilot(args):
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    bank = get_bank(root); data = FixedWindows(bank)
    rows, panels = [], []
    indices = [w["bank_index"] for w in bank["windows"] if w["window_number"] == 0][:8]
    for index in indices:
        batch = batch_item(data[index])
        last_index = int(batch["mask"][0].nonzero()[-1, 0])
        anchor = batch["context"][:, last_index:last_index+1]
        record = {"bank_index": index, "patient_id": batch["patient_id"][0],
            "last": pair_diagnostics(anchor.expand_as(batch["targets"]), batch["targets"], anchor)}
        images = [anchor[0, 0].numpy(), batch["targets"][0, -1].numpy()]
        for mode in ("joint", "source_only", "no_action"):
            if mode == "joint":
                value = cached_teacher(root, index, batch, args, allow_create=True)
            else:
                path = root / "pilot" / f"{index:04d}_{mode}.pt"
                if path.exists():
                    value = torch.load(path, map_location="cpu", weights_only=False)
                else:
                    value = solve_one(batch, args, mode)
                    path.parent.mkdir(exist_ok=True)
                    torch.save(value, path)
            record[mode] = {**pair_diagnostics(value["predictions"], batch["targets"], anchor),
                "solver_seconds": value["timing_seconds"], "last_record": value["records"][-1]}
            images.append(value["predictions"][0, -1].numpy())
        rows.append(record); panels.append((record["patient_id"], images))
        write_json(root / "teacher_pilot_progress.json", rows)
        print(json.dumps({"teacher_pilot": record}), flush=True)
    means = {mode: {"latest_mse": float(np.mean([r[mode]["mse_by_time"][-1] for r in rows])),
        "edge_mse": float(np.mean([r[mode]["edge_mse"] for r in rows]))}
        for mode in ("last", "joint", "source_only", "no_action")}
    # Conservative representation gate; it does not establish predictability.
    passed = (means["joint"]["latest_mse"] < .75 * means["last"]["latest_mse"]
              and means["joint"]["edge_mse"] < means["last"]["edge_mse"]
              and all(r["joint"]["mse_by_time"][-1] < r["last"]["mse_by_time"][-1] for r in rows))
    result = {"bank_sha256": json_hash(bank), "indices": indices, "solver": teacher_args(args),
        "means": means, "representation_gate_passed": passed, "gate": "joint latest MSE <0.75*LCI, mean edge MSE <LCI, all 8 latest MSE <LCI",
        "forecasting_evidence": False, "cases": rows, "source_sha256": sources_snapshot(root)}
    write_json(root / "teacher_pilot.json", result)
    montage(root / "teacher_pilot.png", panels, ["Last observed", "True future", "Joint teacher", "Source-only teacher", "No-action teacher"])
    print(json.dumps({"pilot_gate": passed, "means": means}), flush=True)


def cache(args):
    root = Path(args.output); bank = get_bank(root); data = FixedWindows(bank)
    gate = json.loads((root / "teacher_pilot.json").read_text())
    if not gate["representation_gate_passed"]:
        raise RuntimeError("teacher representation gate failed; inspect solver before full experiment")
    timings = []
    for index in range(len(data)):
        batch = batch_item(data[index])
        value = cached_teacher(root, index, batch, args, allow_create=True)
        timings.append(value["timing_seconds"])
        if index % 10 == 0 or index + 1 == len(data):
            print(json.dumps({"teacher_cache": index+1, "total": len(data), "solver_seconds_sum": sum(timings)}), flush=True)
    records = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((root / "teachers").glob("*.pt"))}
    write_json(root / "teacher_cache_manifest.json", {"bank_sha256": json_hash(bank), "count": len(records),
        "solver": teacher_args(args), "solver_seconds_sum": sum(timings), "files": records})


def make_student(arm, device):
    from methods.trajdetail.control import ControlTrajectory
    seed_all(42)
    return ControlTrajectory(capture_components=arm == "control").to(device)


def step_loss(model, batch, teacher, arm, device):
    with _autocast(device, True):
        result = model(*_input(batch, device))
        loss, parts = trajectory_loss(result, batch)
        extra = loss * 0
        if arm == "control":
            extra = component_loss(result, teacher)
        elif arm == "gradient":
            extra = image_gradient_loss(result["predictions"], batch["targets"].to(device))
        total = loss + .1 * extra
    return total, {**parts, "extra": extra.detach(), "total": total.detach()}


def fit(args):
    from methods.trajdetail.diagnose_unified import validation_diagnostics
    root = Path(args.output); bank = get_bank(root); data = FixedWindows(bank)
    small = args.phase == "student-pilot"
    output = root / ("student_pilot" if small else "students") / args.arm
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if not small and not (root / "teacher_cache_manifest.json").exists():
        raise RuntimeError("full cache must be completed before matched student training")
    teacher_cache = {}
    def get_teacher(index):
        if index not in teacher_cache:
            teacher_cache[index] = cached_teacher(root, index, batch_item(data[index]), args)
        return teacher_cache[index]
    model = make_student(args.arm, device)
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if small else 2e-4, weight_decay=0 if small else 1e-4)
    epochs = 30 if small else 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = np.random.RandomState(42); order_rng = torch.Generator().manual_seed(42)
    config = {"arm": args.arm, "small_training_fit": small, "seed": 42, "epochs": epochs,
        "updates": 240 if small else 4800, "parameters": sum(p.numel() for p in model.parameters()),
        "initial_state_sha256": initial_hash, "bank_sha256": json_hash(bank), "extra_weight": .1,
        "teacher_solver": teacher_args(args), "teacher_cost_excluded_from_student_budget": True,
        "history_weight": .25, "l1_weight": .01, "primary_weight": 2,
        "lr": optimizer.param_groups[0]["lr"], "weight_decay": optimizer.param_groups[0]["weight_decay"],
        "selection": "final fixed update only" if small else "frozen validation latest mean MSE, epochs1/5/.../60",
        "test_evaluated": False, "source_sha256": sources_snapshot(output)}
    write_json(output / "config.json", config)
    fixed = [w["bank_index"] for w in bank["windows"] if w["window_number"] == 0][:8]
    history, best, best_epoch = [], float("inf"), None
    exposure = hashlib.sha256(); start_time = time.monotonic()
    validation = None if small else ACDCJointDataset(split="validation")
    for epoch in range(1, epochs+1):
        model.train(); model.capture_components = args.arm == "control"
        order = torch.randperm(8 if small else 80, generator=order_rng).tolist()
        sums = {}; tic = time.monotonic()
        for person in order:
            index = fixed[person] if small else person * 2 + (epoch-1) % 2
            flips = (False, False) if small else tuple(bool(x) for x in (rng.random_sample(2) < .5))
            batch = batch_item(data[index], flips)
            exposure.update(f"{index}:{flips}:".encode()); exposure.update(data_hash(batch).encode())
            teacher = teacher_components(get_teacher(index), flips, device) if args.arm == "control" else None
            optimizer.zero_grad(set_to_none=True)
            loss, parts = step_loss(model, batch, teacher, args.arm, device)
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
        row = {"epoch": epoch, "train": {k: v/len(order) for k,v in sums.items()},
            "seconds": time.monotonic()-tic, "exposure_sha256": exposure.hexdigest()}
        if not small and (epoch == 1 or epoch % 5 == 0):
            model.capture_components = False
            summary, *_ = validation_diagnostics(model, validation, device)
            mse = summary["mean"]["mse"]; row["validation_mse"] = mse
            if mse < best:
                best, best_epoch = mse, epoch
                torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, output / "best.pt")
        history.append(row); write_json(output / "history.json", history)
        print(json.dumps({"arm": args.arm, "small": small, **row}), flush=True)
    model.capture_components = False
    if small:
        model.eval(); reports = []
        with torch.inference_mode():
            for index in fixed:
                batch = batch_item(data[index])
                with _autocast(device, True):
                    result = model(*_input(batch, device))
                reports.append(pair_diagnostics(result["predictions"], batch["targets"], result["last_context"][:, None]))
        summary = {"training_cases_only": True, "cases": reports,
            "mean_latest_mse": float(np.mean([v["mse_by_time"][-1] for v in reports])),
            "mean_edge_mse": float(np.mean([v["edge_mse"] for v in reports]))}
        torch.save({"model": model.state_dict(), "config": config}, output / "final.pt")
    else:
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        summary, prediction, target, last, ids = validation_diagnostics(model, validation, device)
        for label, single in (("single_history", True), ("multiple_history", False)):
            selected = {d["patient_id"] for d in summary["diagnostics"] if (d["history_count"] == 1) == single}
            values = [d for d in summary["per_patient"] if d["patient_id"] in selected]
            summary[label] = {"n": len(values), "mean": {k: float(np.mean([v[k] for v in values]))
                for k in ("mae", "mse", "nrmse", "ssim", "psnr")}}
        summary.update(best_epoch=best_epoch, best_validation_mse=best)
        np.savez_compressed(output / "validation_predictions.npz", prediction=prediction, target=target,
            last_context=last, patient_ids=ids)
    summary.update(elapsed_seconds=time.monotonic()-start_time, exposure_sha256=exposure.hexdigest(),
        parameters=config["parameters"], initial_state_sha256=initial_hash, test_evaluated=False)
    write_json(output / "summary.json", summary)
    print(json.dumps({"completed": args.arm, "small": small, "summary": summary}), flush=True)


def montage(path, panels, titles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), len(titles), figsize=(3*len(titles), 2.4*len(panels)), squeeze=False)
    for row, (patient, images) in enumerate(panels):
        for col, image in enumerate(images):
            axes[row, col].imshow(image[image.shape[0]//2], cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_axis_off()
            axes[row, col].set_title((titles[col]+"\n" if row == 0 else "") + patient, fontsize=9)
    fig.suptitle("Fixed patient order; central slices; identical display range [0,1]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .985)); fig.savefig(path, dpi=140); plt.close(fig)


def report(args):
    root = Path(args.output); arms = ("endpoint", "gradient", "control")
    records = {a: json.loads((root / "students" / a / "summary.json").read_text()) for a in arms}
    configs = {a: json.loads((root / "students" / a / "config.json").read_text()) for a in arms}
    histories = {a: json.loads((root / "students" / a / "history.json").read_text()) for a in arms}
    exposure_equal = all([r["exposure_sha256"] for r in histories[a]] == [r["exposure_sha256"] for r in histories["endpoint"]] for a in arms)
    initial_equal = len({configs[a]["initial_state_sha256"] for a in arms}) == 1
    parameters_equal = len({configs[a]["parameters"] for a in arms}) == 1
    if not (exposure_equal and initial_equal and parameters_equal):
        raise AssertionError("matched comparison failed")
    output = {"arms": {a: {k: records[a][k] for k in ("mean", "median", "best_epoch", "diagnostic_means", "single_history", "multiple_history")} for a in arms},
        "all_epoch_exposures_equal": exposure_equal, "initial_states_equal": initial_equal,
        "parameter_counts_equal": parameters_equal, "test_evaluated": False,
        "teacher_cache": json.loads((root / "teacher_cache_manifest.json").read_text())}
    predictions = {a: np.load(root / "students" / a / "validation_predictions.npz") for a in arms}
    baseline = predictions["endpoint"]
    if not all(np.array_equal(predictions[a]["patient_ids"], baseline["patient_ids"]) and np.array_equal(predictions[a]["target"], baseline["target"]) for a in arms):
        raise AssertionError("validation cases/targets changed across arms")
    from src.trajectory.external_eval import evaluate_metrics
    output["copy"] = evaluate_metrics(baseline["last_context"], baseline["target"], baseline["patient_ids"].tolist())["mean"]
    panels = []
    for i in range(4):
        panels.append((str(baseline["patient_ids"][i]), [baseline["last_context"][i], baseline["target"][i],
            *[predictions[a]["prediction"][i] for a in arms]]))
    montage(root / "validation_comparison.png", panels, ["Last observed", "True future", "Endpoint", "+Gradient", "+Control"])
    output["paired_latest_mse_wins"] = {a: int(np.sum(np.square(predictions[a]["prediction"]-baseline["target"]).mean((1,2,3)) <
        np.square(baseline["prediction"]-baseline["target"]).mean((1,2,3)))) for a in ("gradient", "control")}
    write_json(root / "comparison.json", output)
    lines = ["# ACDC trajectory-control supervision validation", "", "Matched single-seed development validation; no new test evaluation.", "",
        "| Arm | MAE | NRMSE | SSIM | PSNR | Best epoch |", "|---|---:|---:|---:|---:|---:|"]
    for arm in arms:
        m = records[arm]["mean"]
        lines.append(f"| {arm} | {m['mae']:.6f} | {m['nrmse']:.6f} | {m['ssim']:.6f} | {m['psnr']:.3f} | {records[arm]['best_epoch']} |")
    lines += ["", f"All 60 epoch exposure hashes equal: {exposure_equal}; initial weights equal: {initial_equal}; parameters equal: {parameters_equal}.",
        "", "Regularized teacher vs unregularized teacher student training is not isolated in this three-arm experiment.",
        "Teacher reconstruction is training-only representation evidence; causal validation determines forecasting benefit.", "",
        "![Fixed-order central slices](validation_comparison.png)", ""]
    (root / "report.md").write_text("\n".join(lines))
    print(json.dumps({"comparison": {a: records[a]["mean"] for a in arms},
        "exposure_equal": exposure_equal, "initial_equal": initial_equal}), flush=True)


def test(args):
    """Evaluate validation-selected checkpoints once on the frozen ACDC test split.

    This is an exploratory evaluation because the same test split was used by
    earlier development diagnostics; no checkpoint or hyperparameter is chosen
    from these outputs.
    """
    from methods.trajdetail.diagnose_unified import validation_diagnostics
    from src.trajectory.external_eval import save_evaluation
    from methods.trajdetail.control import ControlTrajectory

    root = Path(args.output)
    source_root = Path(args.source_run or args.output)
    output = root / "test"
    output.mkdir(parents=True, exist_ok=True)
    data = ACDCJointDataset(split="test")
    device = torch.device(args.device)
    results = {}
    for arm in ("endpoint", "gradient", "control"):
        arm_root = source_root / "students" / arm
        checkpoint = torch.load(arm_root / "best.pt", map_location=device, weights_only=False)
        selected_summary = json.loads((arm_root / "summary.json").read_text())
        model = ControlTrajectory(base_channels=16, memory_channels=8, max_step=1 / 11,
                                  capture_components=False).to(device)
        model.load_state_dict(checkpoint["model"])
        summary, prediction, target, last, ids = validation_diagnostics(model, data, device)
        arm_output = output / arm
        evaluation = save_evaluation(arm_output, prediction, target, last, ids,
            title=f"ACDC exploratory test: {arm} trajectory-control arm")
        summary["checkpoint_epoch"] = checkpoint.get("epoch")
        summary["checkpoint_validation_mse"] = selected_summary["best_validation_mse"]
        summary["test_selection"] = "checkpoint selected only on frozen validation; test labels not used"
        summary["development_status"] = "exploratory: this ACDC test split was used by earlier diagnostics"
        write_json(arm_output / "diagnostic_summary.json", summary)
        results[arm] = {"mean": evaluation["mean"], "median": evaluation["median"],
                        "diagnostic_means": summary["diagnostic_means"],
                        "checkpoint_epoch": summary["checkpoint_epoch"]}
        print(json.dumps({"test_arm": arm, **results[arm]}), flush=True)
    output_data = {"arms": results, "test_selection": "validation-only checkpoint selection",
        "development_status": "exploratory: same test split used by earlier development diagnostics",
        "n_patients": len(data), "test_evaluated": True}
    write_json(output / "comparison.json", output_data)
    lines = ["# Exploratory ACDC test evaluation", "",
        "Checkpoints were selected only on the frozen validation split. This is not an untouched confirmatory test because this split was used by earlier development diagnostics.", "",
        "| Arm | MAE | NRMSE | SSIM | PSNR | Checkpoint epoch |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm in ("endpoint", "gradient", "control"):
        m = results[arm]["mean"]
        lines.append(f"| {arm} | {m['mae']:.6f} | {m['nrmse']:.6f} | {m['ssim']:.6f} | {m['psnr']:.3f} | {results[arm]['checkpoint_epoch']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("pilot", "cache", "student-pilot", "train", "report", "test"))
    parser.add_argument("--output", default="runs/trajectory_control_acdc_seed42_20261117")
    parser.add_argument("--source-run", default=None,
                        help="checkpoint run for test phase; defaults to --output")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-steps", type=int, default=150)
    parser.add_argument("--arm", choices=("endpoint", "gradient", "control"), default="endpoint")
    args = parser.parse_args(); torch.set_num_threads(4)
    seed_all(42)
    {"pilot": pilot, "cache": cache, "student-pilot": fit, "train": fit,
     "report": report, "test": test}[args.phase](args)


if __name__ == "__main__":
    main()
