"""Train and score the existing baseline architectures on one fixed manifest."""
import argparse
import csv
import json
import os
from pathlib import Path
from methods.copy.model import forecast as persistence
from methods.linear.model import forecast as extrapolate
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.cronos_isles24_baseline.paper_baselines.models import build_forecaster
from methods.cronos.flow_ops import interpolate_context_times, shared_velocity_ode
from methods.cronos.paper_c3 import PaperC3UNet
from src.trajectory.train import seed_all
from src.metrics import score


def cases(manifest, cache, split, device):
    for row in manifest["patients"]:
        if row["split"] != split:
            continue
        vol = torch.from_numpy(np.load(cache / (row["patient_id"] + ".npy"))).to(device)
        timestamps = np.array(row["days"], dtype=np.float32)
        times = torch.as_tensor((timestamps - timestamps[0]) / manifest["time_unit_days"], device=device)
        if not bool(torch.all(times.diff() > 0)):
            raise ValueError(row)
        yield row["patient_id"], vol, times


def amp(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def imageflownet(device):
    from methods.imageflownet.upstream.src.nn.imageflownet_ode import ImageFlowNetODE
    return ImageFlowNetODE(device=device, in_channels=1, ode_location="all_connections").to(device)


def imageflownet_slice(model, image, gap):
    height, width = image.shape[-2:]
    padding = (0, (-width) % 32, 0, (-height) % 32)
    input_slice = F.pad(image[:, None] * 2 - 1, padding, value=-1.)
    prediction = model(x=input_slice, t=gap.reshape(1) * 5)
    return (prediction[:, 0, :height, :width].float() + 1) / 2


def predict(method, model, vol, times):
    if method == "copy":
        return persistence(vol[1])
    if method == "linear":
        return extrapolate(vol[1], vol[0], ((times[2] - times[1]) /
                (times[1] - times[0])).clamp(max=2.)).clamp(0, 1)
    if method == "cronos":
        with amp(vol.device):
            return shared_velocity_ode(model, vol[None, :2, None], times[None, :2],
                                       times[None, 2], "continuous", steps=10)[0, 0].float()
    if method == "imageflownet":
        return torch.cat([imageflownet_slice(model, sections, times[2] - times[1])
                          for sections in vol[1].split(8)])
    with amp(vol.device) if method != "node_lstm" else torch.autocast(vol.device.type, enabled=False):
        return model(vol[None, :2, None], times[None])[0, 0].float()


def training_loss(method, model, vol, times, rng, epoch):
    if method == "cronos":
        tau = torch.rand(1, device=vol.device)
        state = torch.lerp(vol[:2], vol[2], tau[0])[None]
        velocity = model(state, interpolate_context_times(times[None, :2], times[None, 2], tau))
        return (velocity.float() - (vol[2] - vol[:2])[None]).square().mean()
    if method == "imageflownet":
        sections = rng.choice(vol.shape[1], 8, replace=False)
        gap = times[2] - times[1]
        if epoch <= 2:
            gap = torch.zeros_like(gap)
            target = vol[1, sections]
        else:
            target = vol[2, sections]
        return (imageflownet_slice(model, vol[1, sections], gap) - target).square().mean()
    return (predict(method, model, vol, times) - vol[2]).square().mean()


@torch.inference_mode()
def validate(method, model, data):
    model.eval()
    return float(np.mean([float((predict(method, model, vol, times) - vol[2]).square().mean().sqrt())
                          for _, vol, times in data]))


def train(method, train_set, validation, device, epochs, output, volume_shape, *, domain_balanced=False, learning_rate=None, weight_decay=1e-4, width=32):
    seed_all(42)
    rng = np.random.default_rng(42)
    if method == "cronos":
        model = PaperC3UNet(input_channels=2, base_channels=16).to(device)
    elif method == "imageflownet":
        model = imageflownet(device)
    else:
        model = build_forecaster(method, volume_shape=volume_shape, width=width, num_frames=2).to(device)
    lr = learning_rate if learning_rate is not None else (1e-4 if method in ("cronos", "imageflownet") else 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*.05)
    best, history = float("inf"), []
    checkpoint = output / f"{method}.pt"

    def validation_nrmse():
        if domain_balanced:
            return float(np.mean([validate(method, model, cohort) for cohort in validation]))
        return validate(method, model, validation)

    for epoch in range(1, epochs+1):
        model.train()
        if method == "imageflownet" and epoch == 3:
            model.freeze_time_independent()
        losses = []
        if domain_balanced:
            orders = [rng.permutation(len(cohort)) for cohort in train_set]
            pool = [cohort[order[i % len(order)]] for i in range(max(map(len, orders)))
                    for cohort, order in zip(train_set, orders, strict=True)]
        else:
            pool = train_set
        for index in rng.permutation(len(pool)):
            _, vol, times = pool[index]
            if rng.random() < .5:
                vol = vol.flip(-1)
            optimizer.zero_grad(set_to_none=True)
            with amp(device) if method == "cronos" else torch.autocast(device.type, enabled=False):
                loss = training_loss(method, model, vol, times, rng, epoch)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError(f"{method}, epoch {epoch}")
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
        record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if (epoch == 1 or epoch % 2 == 0 or epoch == epochs) and (
                method != "imageflownet" or epoch >= 3):
            nrmse = validation_nrmse()
            record["validation_nrmse"] = nrmse
            if nrmse < best:
                torch.save({"epoch": epoch, "model": model.state_dict(), "validation_nrmse": nrmse},
                           checkpoint)
                best = nrmse
        history.append(record)
        print(method, record, flush=True)
    state = torch.load(checkpoint, weights_only=True, map_location=device)
    model.load_state_dict(state["model"])
    if not np.isclose(validation_nrmse(), best, rtol=1e-6):
        raise ValueError("Best validation checkpoint was not reproduced")
    (output / f"{method}_history.json").write_text(json.dumps(history, indent=2))
    return model.eval()


def save_rows(path, rows):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@torch.inference_mode()
def test(method, model, data, dataset, output):
    rows = []
    for i, (case, vol, times) in enumerate(data):
        prediction = predict(method, model, vol, times)
        pred, truth = prediction.cpu().numpy(), vol[2].cpu().numpy()
        rows.append({"method": method, "patient_id": case, **score(pred, truth, dataset, vol.device)})
        if i == len(data)//2:
            np.save(output / f"{method}_example.npy", pred)
            if method == "copy":
                np.save(output / "example_inputs_targets.npy", vol.cpu().numpy())
    save_rows(output / f"{method}_test.csv", rows)
    print(method, "TEST", np.mean([row["nrmse"] for row in rows]), "n", len(rows), flush=True)


METHODS = ("copy", "linear", "convlstm", "simvp", "node_lstm", "vivit", "cronos", "imageflownet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    manifest = json.loads((args.output / "manifest.json").read_text())
    training, validation, held_out = (list(cases(manifest, args.cache, split, device))
                                      for split in ("train", "validation", "test"))
    for method in args.methods:
        model = None if method in ("copy", "linear") else train(
            method, training, validation, device, args.epochs, args.output, manifest["shape_zyx"])
        test(method, model, held_out, manifest["dataset"], args.output)


if __name__ == "__main__":
    main()
