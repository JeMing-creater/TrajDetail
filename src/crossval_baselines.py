"""Fit the archived comparison architectures in one cross-validation fold."""

import argparse
import fcntl
from pathlib import Path
import sys

import numpy as np
import torch

from src.cronos_isles24_baseline.paper_baselines.models import build_forecaster
from src.cronos_three_dataset_baseline.external_paper_baselines import adapt_item, model_batch
from src.compare import METHODS, save_rows
from src.crossval_data import LEGACY, RUN, SPLITS, LegacyWindows, read, write
from src.trajectory.train import seed_all
from src.metrics import score


def for_model(item, dataset):
    return item if dataset == "isles" else adapt_item(item, dataset)


def legacy_cronos(dataset, device):
    from src.paths import PROTOCOL
    from methods.cronos.upstream.methods.cronos import CRONOS

    config = read(PROTOCOL / "cronos" / f"{dataset}.json")
    shape = tuple(config.pop("in_shape"))
    config.update(device=str(device), num_workers=0)
    model = CRONOS(in_shape=shape, feature_size=config["base_channels"], **config).to(device)
    return model, config


def flow_inputs(item, dataset, device):
    from methods.imageflownet.matched_acdc import latest_observation

    anchor, gaps = latest_observation(item)
    raw = np.moveaxis(anchor, -1, 0) if dataset == "lumiere" else anchor.swapaxes(-1, -2)
    x = torch.from_numpy(np.ascontiguousarray(raw)).to(device)[:, None]
    scale = {"acdc": 5., "lumiere": 5., "isles": 5.}[dataset]
    return x, float(gaps[-1]) * scale


def flow_predict(model, raw, gap, batch=8):
    from torch.nn.functional import interpolate

    result = []
    for sections in raw.split(batch):
        x = interpolate(sections * 2 - 1, (256, 256), mode="bilinear", align_corners=False)
        prediction = model(x=x, t=x.new_tensor([gap]))
        y = interpolate(prediction.float(), raw.shape[-2:], mode="bilinear", align_corners=False)
        result.append((y[:, 0] + 1) / 2)
    return torch.cat(result)


def to_axial(result, dataset):
    return np.moveaxis(result, -1, 0) if dataset == "lumiere" else result


@torch.inference_mode()
def predict(model, item, method, dataset, device):
    if method in ("copy", "linear"):
        visible = np.flatnonzero(item["mask"])
        observed = item["context"][visible]
        if method == "copy" or len(visible) < 2:
            return to_axial(observed[-1], dataset)
        times = item["context_times"][visible]
        ratio = min(2., (item["target_times"][-1] - times[-1]) / (times[-1] - times[-2]))
        return to_axial(np.clip(observed[-1] + (observed[-1] - observed[-2]) * ratio, 0, 1), dataset)
    if method == "imageflownet":
        raw, gap = flow_inputs(item, dataset, device)
        estimate = flow_predict(model, raw, gap).cpu().numpy()
        return estimate if dataset == "lumiere" else estimate.swapaxes(-1, -2)
    if method == "cronos":
        batch = native_batch([item], device)
        context_time = torch.cat((batch["context_time"], batch["target_time"]), dim=1)
        forecast = model.validation_step(batch["context"], time_points=context_time)
        return to_axial(forecast.reshape(item["targets"].shape[1:]).cpu().numpy(), dataset)
    context, times, _ = model_batch([for_model(item, dataset)], device)
    return model(context, times)[0, 0].float().cpu().numpy()


def native_batch(items, device):
    def stack(key):
        return torch.as_tensor(np.stack([item[key] for item in items]), device=device)

    context = stack("context")[:, :, None]
    targets = stack("targets")[:, -1:, None]
    return {"context": context, "target_img": targets,
            "context_time": stack("context_times"), "target_time": stack("target_times")[:, -1:],
            "context_seg": None, "target_seg": None}


def fit(method, dataset, partitions, device, output, smoke=False, settings=None):
    seed_all(42)
    train, validation, _ = partitions
    if method == "cronos":
        model, config = legacy_cronos(dataset, device)
        batch_size, epochs, interval = 4, 500, 10
        lr, decay = 1e-4, .01
    elif method == "imageflownet":
        from methods.imageflownet.matched_acdc import ImageFlowNetODE

        model = ImageFlowNetODE(device=device, in_channels=1, ode_location="all_connections").to(device)
        for block in model.ode_list:
            block.activation_checkpointing = True
        config = {"resolution": 256, "time_scale": 5., "identity_epochs": 2,
                  "ode_activation_checkpointing": True}
        batch_size, epochs, interval = 1, 10, 2
        lr, decay = 1e-4, 1e-4
    else:
        one = for_model(train[0], dataset)
        width = (settings or {}).get("width", 32)
        model = build_forecaster(method, volume_shape=one["targets"].shape[1:],
                                 width=width, num_frames=len(one["context"])).to(device)
        config = {"width": width, "num_frames": len(one["context"])}
        batch_size, epochs, interval = 4, 500, 10
        lr, decay = 1e-4, .01
    if settings:
        epochs, interval, batch_size = (settings[k] for k in ("epochs", "validation_interval", "batch_size"))
        lr, decay = (settings[k] for k in ("learning_rate", "weight_decay"))
    if smoke:
        epochs, interval, batch_size = (3 if method == "imageflownet" else 1), 1, 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    write(output / f"{method}_recipe.json",
          {"epochs": epochs, "batch_size": batch_size, "validation_interval": interval,
           "learning_rate": lr, "weight_decay": decay, "seed": 42, "initialization": "random",
           "smoke": smoke, "model": config})
    best = float("inf")
    history = []
    sample_rng = np.random.default_rng(42)
    for epoch in range(1, epochs + 1):
        model.train()
        if method == "imageflownet" and epoch == 3:
            model.freeze_time_independent()
        losses = []
        indices = sample_rng.permutation(len(train.rows)) * train.n
        if train.n > 1:
            indices += (epoch - 1) % train.n
        if smoke:
            indices = indices[:1]
        for start in range(0, len(indices), batch_size):
            items = [train[int(index)] for index in indices[start:start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            if method == "cronos":
                loss = model.training_step(native_batch(items, device), start)
            elif method == "imageflownet":
                raw, gap = flow_inputs(items[0], dataset, device)
                indices_z = sample_rng.choice(len(raw), min(len(raw), 4), replace=False)
                raw = raw[indices_z]
                target = items[0]["targets"][-1]
                axial = to_axial(target, dataset)
                if dataset != "lumiere":
                    axial = axial.swapaxes(-1, -2)
                target = (raw[:, 0] if epoch <= 2 else
                          torch.from_numpy(np.ascontiguousarray(axial[indices_z])).to(device))
                loss = (flow_predict(model, raw, 0. if epoch <= 2 else gap, batch=4) - target).square().mean()
            else:
                context, times, target = model_batch([for_model(item, dataset) for item in items], device)
                loss = (model(context, times) - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{dataset}, {method}, epoch {epoch}: nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        schedule.step()
        record = {"epoch": epoch, "train_mse": float(np.mean(losses))}
        if (epoch % interval == 0 or epoch == epochs) and (method != "imageflownet" or epoch >= 3):
            model.eval()
            errors = []
            for i in range(min(1, len(validation)) if smoke else len(validation)):
                item = validation[i]
                pred = predict(model, item, method, dataset, device)
                errors.append(float(np.mean((pred - to_axial(item["targets"][-1], dataset)) ** 2)))
            record["validation_mse"] = float(np.mean(errors))
            if record["validation_mse"] < best:
                best = record["validation_mse"]
                temp = output / f"{method}.tmp"
                torch.save({"epoch": epoch, "model": model.state_dict(), "validation_mse": best}, temp)
                temp.replace(output / f"{method}.pt")
        history.append(record)
        write(output / f"{method}_history.json", history)
        print(dataset, method, record, flush=True)
    model.load_state_dict(torch.load(output / f"{method}.pt", map_location=device, weights_only=True)["model"])
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", type=int, choices=(1, 2, 3))
    parser.add_argument("dataset", choices=LEGACY)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--root", type=Path, default=RUN)
    parser.add_argument("--device", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    fold_root = args.root / f"fold{args.fold}"
    dataset, method = args.dataset, args.method
    output = fold_root / dataset
    if args.smoke:
        output = output / "smoke"
        output.mkdir(exist_ok=True)
    final = output / f"{method}_test.csv"
    with (output / f".{method}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        train_and_evaluate(args, fold_root, output, final, device)


def train_and_evaluate(args, fold_root, output, final, device):
    if final.exists():
        return
    dataset, method = args.dataset, args.method
    # Sampling mirrors the original baseline adapters. LUMIERE alternates a fixed pair.
    dynamic = dataset != "lumiere" and method != "imageflownet"
    partitions = [LegacyWindows(fold_root, dataset, split, train_windows=2 if not dynamic else 1,
                                dynamic=dynamic) for split in SPLITS]
    for window in partitions[1:]:
        window.source = partitions[0].source
    model = (None if method in ("copy", "linear") else
             fit(method, dataset, partitions, device, output, args.smoke))
    held_out = partitions[-1]
    rows = []
    for i in range(1 if args.smoke else len(held_out)):
        item = held_out[i]
        prediction = predict(model, item, method, dataset, device)
        target = to_axial(item["targets"][-1], dataset)
        # Both arrays already have the axial orientation used by the five-cohort comparison.
        rows.append({"method": method, "patient_id": item["patient_id"],
                     **score(prediction, target, "acdc" if dataset == "lumiere" else dataset, device)})
    temporary = final.with_suffix(".tmp")
    save_rows(temporary, rows)
    temporary.replace(final)
    print(dataset, method, "TEST", len(rows), np.mean([row["nrmse"] for row in rows]), flush=True)


if __name__ == "__main__":
    main()
