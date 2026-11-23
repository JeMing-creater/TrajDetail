"""Restore selected weights and infer from the released patient windows."""
from pathlib import Path

import numpy as np
import torch

from src import compare
from src.crossval_baselines import for_model, legacy_cronos, predict as predict_legacy, to_axial
from src.crossval_data import LegacyWindows, RAW_CACHE
from src.experiments import LEGACY, REFERENCES, SIMPLE, fold_manifest, load, members
from src.paths import DATA, PROTOCOL, SPLITS_ROOT
from src.release_io import checkpoint


def records(dataset, fold, device="cpu"):
    if dataset in LEGACY:
        windows = LegacyWindows(SPLITS_ROOT / f"fold{fold}", dataset, "test")
        for item in windows:
            visible = np.flatnonzero(item["mask"])
            yield (item["patient_id"], to_axial(item["targets"][-1], dataset),
                   to_axial(item["context"][visible[-1]], dataset), item)
    else:
        for patient, sequence, times in compare.cases(load(fold_manifest(dataset, fold)),
                                                      RAW_CACHE / dataset, "test", device):
            yield patient, sequence[2].cpu().numpy(), sequence[1].cpu().numpy(), (sequence, times)


def simple(method, dataset, ordered, directory, meta, device):
    model = None
    if method in SIMPLE:
        state = checkpoint(directory / f"{method}.pt")
        chosen = meta["weights"]["model"]
        for key in ("epoch", "validation_nrmse", "validation_mse"):
            if key in chosen and state.get(key) != chosen[key]:
                raise ValueError(f"Loaded checkpoint does not match the validation selection for {method}")
        example = ordered[0][3]
        if method == "cronos" and dataset in LEGACY:
            model, _ = legacy_cronos(dataset, device)
        elif method == "cronos":
            model = compare.PaperC3UNet(input_channels=2, base_channels=16).to(device)
        elif method == "imageflownet":
            model = compare.imageflownet(device)
        else:
            if dataset in LEGACY:
                formatted = for_model(example, dataset)
                shape, frames = formatted["targets"].shape[1:], len(formatted["context"])
            else:
                shape, frames = example[0].shape[-3:], 2
            model = compare.build_forecaster(method, volume_shape=shape, num_frames=frames,
                width=meta["config"]["training"].get("width", 32)).to(device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
    for patient, _, _, item in ordered:
        with torch.inference_mode():
            if dataset in LEGACY:
                value = predict_legacy(model, item, method, dataset, device)
            else:
                value = compare.predict(method, model, *(part.to(device) for part in item)).cpu().numpy()
        yield patient, value


def trajdetail(dataset, ordered, directory, device):
    from methods.trajdetail.audit_acdc_sharpness import ReplayProbe
    from methods.trajdetail.proposed import run_one
    from methods.trajdetail.scratch_trajectory import windows
    from methods.trajdetail.shared_forecast_model import SharedForecast
    from methods.trajdetail.train_external_curvature_prototype import Features, render
    from src.trajectory.shared_forecast_data import temporal_predictors, to_axial as as_axial
    from src.trajectory.train_unified import _autocast, _input
    from src.trajectory.validate_control import batch_item
    from methods.trajdetail.train_shared_forecast import Windows

    earlier = dataset in LEGACY
    readout = directory / ("legacy_readout" if earlier else "shared_readout")
    upstream = directory / dataset
    if earlier:
        upstream /= "trajectory"
    config = load(PROTOCOL / "trajdetail.json")["selected"]
    model = ReplayProbe(use_dino=True).to(device).eval()
    model.load_state_dict(checkpoint(upstream / "benefit.pt")["model"])
    model.capture_transport = True
    features = Features(dataset if earlier else "lumiere", device)
    voxel, detail = [SharedForecast(20).to(device=device, memory_format=torch.channels_last).eval()
                     for _ in range(2)]
    for decoder, stage in ((voxel, "voxel"), (detail, "detail")):
        decoder.load_state_dict(checkpoint(readout / stage / "selected.pt")["model"], strict=True)
    if not earlier:
        order = [item[0] for item in ordered]
        inputs = windows(load(directory / dataset / "manifest.json"), RAW_CACHE / dataset, "test")
        inputs = {item["patient_id"]: item for item in inputs}
        evaluated = [(patient, inputs[patient]) for patient in order]
    else:
        evaluated = [(patient, item) for patient, _, _, item in ordered]

    from src.live_window import LiveWindow
    for patient, item in evaluated:
        with torch.inference_mode():
            context, mask, times, query = _input(batch_item(item), device)
            with _autocast(device, True):
                prediction = model(context, mask, times, query, last_features=features.observed(item))
            visible = mask[0].bool()
            anchor, previous, trend, elapsed, interval = temporal_predictors(
                context[0, visible], times[0, visible], query[0])
            which = len(model.transport_parts) - 1
            parts = model.transport_parts[which]
            streams = torch.cat((anchor, render(parts, dataset if earlier else "lumiere")[0],
                                 prediction["predictions"][:, which], previous, trend[which:which+1],
                                 parts[3][0].float()))
            axial = as_axial(streams, dataset) if earlier else streams.movedim(-1, -3)
            features.cache.clear()
            live = LiveWindow(axial.half().cpu().numpy(), elapsed[which].item(), float(interval),
                              item, which, dataset)
            _, _, result = run_one(detail, voxel, live, 0, device, config)
            yield patient, result.cpu().float().numpy()


def heavy(method, dataset, ordered, directory, meta, destination, limit):
    from src.comparator_windows import legacy_record, load_partitions

    fold = meta["fold"]
    config = meta["config"]
    group = config["group"]
    if dataset in LEGACY:
        test = [legacy_record(item, dataset, False) for _, _, _, item in ordered]
    else:
        scale = load(fold_manifest(dataset, fold))["time_unit_days"] / 365.25
        test = [(patient, item[0], item[1] * scale) for patient, _, _, item in ordered]
    torch.set_float32_matmul_precision("high")
    if method == "log4d":
        from methods.log4d.log4d_adapter import train_and_predict

        for name in ("registration", "diffusion"):
            if not (directory / f"4dlog_{name}.done").is_file():
                raise FileNotFoundError(f"Missing 4DLoG training completion mark for {name}")
        yield from train_and_predict([], [], test, directory, settings=config["training"])
    elif method == "brlp":
        from methods.brlp.pipeline import evaluate

        parts = load_partitions(SPLITS_ROOT, fold, group, splits=("train",))
        training = [record for cohort in members(group) for record in parts[cohort]["train"]]
        # Fit deterministic image-derived covariates on the training partition only.
        yield from evaluate(directory, training, test, dataset, config["training"])
    elif method == "delta_lfm":
        from methods.delta_lfm.delta_lfm_dlbs import evaluate_integrated, load_ae
        from src.training import delta_settings

        ae = load_ae()
        args = delta_settings(config, directory, "acdc" if dataset == "lumiere" else dataset,
                              result_dir=destination, prediction_dtype=np.float32)
        evaluate_integrated(ae, [(p, v, t / 10) for p, v, t in test], args)
        for patient, _, _ in test:
            yield patient, np.load(destination / "predictions" / f"{patient}.npy")


def infer(method, dataset, ordered, directory, meta, destination, device):
    if method in (*REFERENCES, *SIMPLE):
        yield from simple(method, dataset, ordered, directory, meta, device)
    elif method == "trajdetail":
        yield from trajdetail(dataset, ordered, directory, device)
    else:
        if device.type != "cuda":
            raise ValueError(f"{method} requires CUDA")
        yield from heavy(method, dataset, ordered, directory, meta, destination, len(ordered))
