"""Measure reconstruction, sampling, and background error for the BrLP adapter."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from methods.brlp.brlp_adapter import AUTHOR, amp, conditions, denoise, fit_progression, latent_inputs, pad, restore
from src.comparator_windows import legacy_record
from src.compare import cases
from src.crossval_data import LEGACY, NEW, RAW_CACHE, RUN, LegacyWindows, read
from src.release_io import OUTPUT

GROUPS = LEGACY + ("mu-dlbs",)
SCHEDULE = dict(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=.0015, beta_end=.0205)


def case_iter(group, fold, split):
    root = RUN / f"fold{fold}"
    for cohort in NEW if group == "mu-dlbs" else (group,):
        if cohort in NEW:
            manifest = read(root / cohort / "manifest.json")
            scale = manifest["time_unit_days"] / 365.25
            yield cohort, ((p, v, t * scale) for p, v, t in cases(manifest, RAW_CACHE / cohort, split, "cpu"))
        else:
            windows = LegacyWindows(root, cohort, split, train_windows=1)
            yield cohort, (legacy_record(windows[i], cohort, False) for i in range(len(windows)))


def collect_records(group, fold, split):
    return {cohort: list(data) for cohort, data in case_iter(group, fold, split)}


def output_path(group, fold):
    return OUTPUT / "brlp" / f"fold{fold}" / group


@torch.no_grad()
def decode(ae, latent, shape):
    with amp():
        image = ae.decode(latent[None])[0, 0].float()
    d, h, w = shape
    return image[:d, :h, :w]


def stats(pred, target, anchor):
    observed_background = (anchor.abs() <= 1e-6)
    covered_background = observed_background & (target.abs() <= 1e-6)
    error = (pred.clamp(0, 1) - target).square()
    rms = lambda x: x.mean().sqrt().item() if x.numel() else float("nan")
    return dict(nrmse=rms(error), nonbackground_rmse=rms(error[~covered_background]),
                background_mse_share=(error[covered_background].sum() / error.sum().clamp_min(1e-12)).item(),
                below_zero=(pred < 0).float().mean().item(), above_one=(pred > 1).float().mean().item(),
                change_rms=rms((pred.clamp(0, 1) - anchor).square()))


def mask_exterior(prediction, anchor):
    support = (anchor.abs() > 1e-6).float()[None, None]
    # Preserve the whole volume when the observed image contains no zero background.
    support = F.max_pool3d(support, kernel_size=5, stride=1, padding=2)[0, 0].bool()
    return prediction.masked_fill(~support, 0)


@torch.no_grad()
def samples(model, control, anchor, covariates, seed, count, start=None, steps=50):
    from generative.networks.schedulers import DDIMScheduler

    scheduler = DDIMScheduler(**SCHEDULE, clip_sample=False)
    scheduler.set_timesteps(steps)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn((count, *anchor.shape), device="cuda", generator=generator)
    timesteps = scheduler.timesteps
    if start is not None:
        timesteps = timesteps[timesteps <= start]
        t = torch.full((count,), int(timesteps[0]), device="cuda", dtype=torch.long)
        z = scheduler.add_noise(anchor.expand_as(z), z, t)
    c = covariates.expand(count, -1)
    cond = conditions(anchor.expand_as(z), c)
    z_outputs = []
    for noisy, context, conditioning in zip(z.split(4), c.split(4), cond.split(4), strict=True):
        for step in timesteps:
            time = torch.full((len(noisy),), int(step), device="cuda", dtype=torch.long)
            with amp():
                noise = denoise(model, control, noisy, time, context[:, None], conditioning)
            noisy, _ = scheduler.step(noise, int(step), noisy)
        z_outputs.append(noisy)
    return torch.cat(z_outputs)


@torch.no_grad()
def scale_latents(ae, train, trajectory):
    latents, _ = latent_inputs(ae, train, trajectory, True)
    return latents[:, 1].std().clamp_min(1e-5).reciprocal()


@torch.no_grad()
def diagnose(ae, model, control, data, trajectory, scale, cohort, fold, count):
    indices = np.linspace(0, len(data) - 1, min(len(data), count), dtype=int)
    selected = [data[i] for i in indices]
    z, covariates = latent_inputs(ae, selected, trajectory, True)
    for index, ((patient, volumes, _), latent, c) in enumerate(zip(selected, z, covariates, strict=True)):
        source, target = volumes[1:].cuda()
        shape = target.shape
        copy = source
        ae_oracle = decode(ae, latent[1], shape)
        ae_copy = decode(ae, latent[0], shape)
        with amp():
            native_latent = ae.encode(pad(target[None, None], 8))[0].float()[0]
        ae_oracle_native = decode(ae, native_latent, shape)
        seed = 42 + index
        pure = samples(model, control, latent[0] * scale, c, seed, 10)
        unconditioned = samples(model, None, latent[0] * scale, c, seed, 1)
        wrong = z[(index + 1) % len(z), 0] * scale
        wrong_anchor = samples(model, control, wrong, c, seed, 1)
        results = {"copy": copy, "autoencoder_oracle": ae_oracle, "autoencoder_oracle_pad8": ae_oracle_native,
                   "autoencoder_copy": ae_copy, "without_control": decode(ae, unconditioned[0] / scale, shape),
                   "different_patient_control": decode(ae, wrong_anchor[0] / scale, shape),
                   "baseline": decode(ae, pure[0] / scale, shape),
                   "las10": decode(ae, pure.mean(0) / scale, shape)}
        results["las10_mask"] = mask_exterior(results["las10"], source)
        for t in (160, 320, 500):
            refined = samples(model, control, latent[0] * scale, c, seed, 1, t)
            results[f"warm{t}"] = decode(ae, refined[0] / scale, shape)
        for name, image in results.items():
            yield {"dataset": cohort, "fold": fold, "patient_id": patient, "variant": name,
                   **stats(image, target, source)}


def load_models(group, fold):
    sys.path.insert(0, str(AUTHOR))
    from brlp import networks

    path = output_path(group, fold)
    models = []
    for factory, name in ((networks.init_autoencoder, "ae"), (networks.init_latent_diffusion, "diffusion"),
                          (networks.init_controlnet, "control")):
        model = factory().cuda().eval().requires_grad_(False)
        restore(path / f"brlp_{name}.pt", model)
        models.append(model)
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    ae, model, control = load_models(args.group, args.fold)
    train = [row for group in collect_records(args.group, args.fold, "train").values() for row in group]
    trajectory = fit_progression(train)
    scale = scale_latents(ae, train, trajectory)
    del train
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = None
        for cohort, records in case_iter(args.group, args.fold, "validation"):
            for entry in diagnose(ae, model, control, list(records), trajectory, scale, cohort, args.fold, args.count):
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=entry.keys())
                    writer.writeheader()
                writer.writerow(entry)
                stream.flush()
                if entry["variant"] == "baseline":
                    print(cohort, entry["patient_id"], entry["nrmse"], flush=True)


if __name__ == "__main__":
    main()
