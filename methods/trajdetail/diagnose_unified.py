"""Controlled V1 interventions and retraining. Original V1 remains immutable.

`normalized` changes velocity's numerical units to normalized sampling-grid
coordinates per unit data time. It is preconditioning, not an added physical law.
Variants instantiate the original parameters before optional intervention or
pruning, preserving initialization and RNG consumption for the remaining model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from methods.trajdetail.unified_model import HistoryCalibratedTrajectory, warp_volume
from src.trajectory.train_unified import _autocast, _input, trajectory_loss, predict
from src.trajectory.external_data import ACDCJointDataset
from src.trajectory.external_eval import evaluate_metrics, save_evaluation
from src.trajectory.train import seed_all


def _zero_output(_module, _inputs, output):
    return output * 0


class ZeroMemoryCorrection(torch.nn.Module):
    """Parameter-free zero update that never reads potentially invalid values."""

    def __init__(self, memory_channels):
        super().__init__()
        self.memory_channels = memory_channels

    def forward(self, value):
        return value.new_zeros(value.shape[0], self.memory_channels, *value.shape[2:])


class DiagnosticTrajectory(HistoryCalibratedTrajectory):
    def __init__(self, *, velocity_units="voxel", transport=True, source=True,
                 correction=True, sampling="incremental", group_norm=False,
                 prune_correction=False, **kwargs):
        super().__init__(**kwargs)
        if velocity_units not in ("voxel", "normalized"):
            raise ValueError("velocity_units must be voxel or normalized")
        self.velocity_units = velocity_units
        if sampling not in ("incremental", "composed"):
            raise ValueError("sampling must be incremental or composed")
        self.sampling = sampling
        if group_norm:
            # Batch size 1: normalize each image's spatial features, never
            # across patients/time. Conv initial weights remain unchanged.
            for name in ("encoder", "dynamics"):
                layers = []
                for layer in getattr(self, name):
                    layers.append(layer)
                    if isinstance(layer, torch.nn.Conv3d):
                        groups = math.gcd(4, layer.out_channels)
                        layers.append(torch.nn.GroupNorm(groups, layer.out_channels))
                setattr(self, name, torch.nn.Sequential(*layers))
        self.transport_enabled = transport
        self.source_enabled = source
        if not correction:
            self.correction.register_forward_hook(_zero_output)
        if prune_correction:
            self.correction = ZeroMemoryCorrection(self.memory_channels)

    def _vector_field(self, image, memory, now, statistics):
        velocity, source, derivative = super()._vector_field(image, memory, now, statistics)
        if self.velocity_units == "normalized":
            d, h, w = image.shape[2:]
            velocity = velocity * velocity.new_tensor([(w-1)/2, (h-1)/2, (d-1)/2])[None, :, None, None, None]
        if not self.transport_enabled:
            velocity = velocity * 0
        if not self.source_enabled:
            source = source * 0
        statistics[-1] = torch.stack([value.detach().float().square().mean()
                                     for value in (velocity, source, derivative)])
        return velocity, source, derivative

    def _advance(self, image, memory, field, step):
        if not self.transport_enabled:
            # Disabling transport must also disable its interpolation roundoff.
            _, source, derivative = field
            return image + step * source, memory + step * derivative
        return super()._advance(image, memory, field, step)

    def _trajectory(self, image, memory, start, queries, statistics):
        if self.sampling == "incremental":
            return super()._trajectory(image, memory, start, queries, statistics)
        # Backward maps compose as d_new(x)=h*u(x)+d_old(x-h*u(x)).
        # Reconstruct the acquisition image from its anchor at every step;
        # only the smooth coordinate/source fields are incrementally sampled.
        anchor = image
        zero_displacement = image.new_zeros(image.shape[0], 3, *image.shape[2:])
        states = [(image, memory, zero_displacement, torch.zeros_like(image))]
        plans = []
        for query in queries:
            ratio = (query-start)/self.max_step
            nearest = round(ratio)
            n = nearest if abs(ratio-nearest) <= 1e-5 else math.floor(ratio)
            remainder = 0.0 if abs(ratio-nearest) <= 1e-5 else query-start-n*self.max_step
            plans.append((n, remainder))

        def advance(state, field, step):
            current, mem, displacement, signal = state
            velocity, source, derivative = field
            if self.transport_enabled:
                local = step * velocity
                displacement = local + warp_volume(displacement, local, self._sampling_grid)
                signal = warp_volume(signal, local, self._sampling_grid) + step * source
                current = warp_volume(anchor, displacement, self._sampling_grid) + signal
            else:
                signal = signal + step * source
                current = anchor + signal
            return current, mem + step * derivative, displacement, signal

        fields = {}
        for index in range(max(n for n, _ in plans)):
            fields[index] = self._vector_field(*states[-1][:2], start+index*self.max_step, statistics)
            states.append(advance(states[-1], fields[index], self.max_step))
        outputs = []
        for index, remainder in plans:
            if remainder == 0:
                outputs.append(states[index][:2])
            else:
                if index not in fields:
                    fields[index] = self._vector_field(*states[index][:2], start+index*self.max_step, statistics)
                outputs.append(advance(states[index], fields[index], remainder)[:2])
        return outputs


VARIANTS = {
    "original": {},
    "no_correction": {"correction": False},
    "no_transport": {"transport": False},
    "no_source": {"source": False},
    "scaled": {"velocity_units": "normalized"},
    "scaled_no_source": {"velocity_units": "normalized", "source": False},
    "scaled_no_correction": {"velocity_units": "normalized", "correction": False},
    "composed": {"sampling": "composed"},
    "scaled_composed": {"velocity_units": "normalized", "sampling": "composed"},
    "normalized_backbone": {"group_norm": True},
    "normalized_pruned": {"group_norm": True, "prune_correction": True},
}


def make_model(variant, config=None):
    config = config or {}
    return DiagnosticTrajectory(base_channels=config.get("base_channels", 16),
        memory_channels=config.get("memory_channels", 8),
        max_step=config.get("max_step", 1/11), **VARIANTS[variant])


def dataset(split, augment=False):
    return ACDCJointDataset("/mnt/Datasets/ACDC", split, augment=augment)


@torch.inference_mode()
def validation_diagnostics(model, data, device, amp=True, last_only=False):
    model.eval()
    pred, target, lasts, ids, diagnostics = [], [], [], [], []
    for batch in DataLoader(data, batch_size=1, shuffle=False):
        if last_only:
            mask = batch["mask"]
            for row in mask:
                last = row.nonzero()[-1, 0]
                row.zero_()
                row[last] = 1
        with _autocast(device, amp):
            result = model(*_input(batch, device))
        p = result["predictions"][:, -1].float().cpu().numpy()[0]
        y = batch["targets"][:, -1].numpy()[0]
        x = result["last_context"].float().cpu().numpy()[0]
        dp, dy = (p-x).ravel(), (y-x).ravel()
        diagnostics.append({"patient_id": batch["patient_id"][0],
            "history_count": int(batch["mask"].sum()),
            "bias": float((p-y).mean()), "change_bias": float(dp.mean()),
            "change_rms": float(np.sqrt(np.mean(dp**2))),
            "true_change_rms": float(np.sqrt(np.mean(dy**2))),
            "change_cosine": float(dp.dot(dy)/max(np.linalg.norm(dp)*np.linalg.norm(dy), 1e-12)),
            **{name: float(result[name]) for name in ("velocity_rms", "source_rms", "memory_derivative_rms")}})
        pred.append(p); target.append(y); lasts.append(x); ids.extend(batch["patient_id"])
    summary = evaluate_metrics(np.stack(pred), np.stack(target), ids)
    summary["diagnostics"] = diagnostics
    summary["diagnostic_means"] = {name: float(np.mean([d[name] for d in diagnostics]))
                                  for name in diagnostics[0] if name not in ("patient_id", "history_count")}
    return summary, np.stack(pred), np.stack(target), np.stack(lasts), ids


def interventions(args):
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    data = dataset("validation")
    summaries = {}
    for variant in ("original", "no_correction", "no_transport", "no_source", "last_only"):
        model = make_model("original" if variant == "last_only" else variant, checkpoint["config"]).to(args.device)
        model.load_state_dict(checkpoint["model"])
        result, *_ = validation_diagnostics(model, data, torch.device(args.device), last_only=variant == "last_only")
        summaries[variant] = result
        print(json.dumps({"intervention": variant, "mean": result["mean"], "median": result["median"],
                          "diagnostic_means": result["diagnostic_means"]}), flush=True)
        (output / "interventions_validation.json").write_text(json.dumps(summaries, indent=2)+"\n")
    return summaries


def overfit(args):
    seed_all(42)
    batch = next(iter(DataLoader(dataset("train"), batch_size=1, shuffle=False)))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    device = torch.device(args.device)
    for variant in args.overfit_variants:
        seed_all(42)
        model = make_model(variant, {"base_channels":args.base_channels}).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
        records = []
        for step in range(args.steps+1):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, True):
                result = model(*_input(batch, device))
                loss, parts = trajectory_loss(result, batch)
            if step % 25 == 0 or step == args.steps:
                record = {"step": step, "loss": float(loss.detach()),
                    **{k: float(v) for k,v in parts.items()},
                    **{k: float(result[k]) for k in ("velocity_rms", "source_rms")}}
                records.append(record)
                print(json.dumps({"overfit": variant, **record}), flush=True)
            if step < args.steps:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                if not torch.isfinite(norm): raise FloatingPointError("non-finite overfit gradient")
                optimizer.step()
        summaries[variant] = {"patient_id": batch["patient_id"], "mask": batch["mask"].tolist(),
                              "target_times": batch["target_times"].tolist(), "records": records}
        torch.save({"model": model.state_dict(), "variant": variant}, output/f"overfit_{variant}.pt")
        (output/"overfit_training.json").write_text(json.dumps(summaries, indent=2)+"\n")


def train_variant(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    seed_all(42)
    model = make_model(args.variant, {"base_channels":args.base_channels}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    training, validation = dataset("train", augment=True), dataset("validation")
    loader = DataLoader(training, batch_size=1, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(42), pin_memory=True)
    # Sampling/augmentation use NumPy; reset independently of model init.
    np.random.seed(42)
    config = {"variant": args.variant, "settings": VARIANTS[args.variant], "seed":42,
        "epochs":args.epochs, "base_channels":args.base_channels, "memory_channels":8, "max_step":1/11,
        "lr":2e-4, "weight_decay":1e-4, "amp":True, "augment":True,
        "history_weight":.25,"l1_weight":.01,"primary_weight":2,
        "selection":"minimum frozen validation latest-target mean MSE",
        "parameters":sum(p.numel() for p in model.parameters()), "test_evaluated":False,
        "source_sha256":{}}
    snapshot=output/"source_snapshot"; snapshot.mkdir()
    for name in ("diagnose_unified.py","unified_model.py","train_unified.py","external_data.py","external_eval.py","train.py"):
        src=Path(__file__).parent/name
        shutil.copyfile(src,snapshot/name)
        config["source_sha256"][name]=hashlib.sha256(src.read_bytes()).hexdigest()
    for name, path in {
        "manifest":Path("cronos_acdc_baseline/data_manifest.json"),
        "validation_masks":Path("cronos_acdc_baseline/runs/discrete_seed42/fixed_masks_validation.json"),
        "test_masks":Path("cronos_acdc_baseline/runs/discrete_seed42/fixed_masks_test.json")}.items():
        config[name+"_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    (output/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    history=[]; best=float("inf"); best_epoch=None; start=time.monotonic()
    exposure=hashlib.sha256()
    for epoch in range(1,args.epochs+1):
        model.train(); sums={}; tic=time.monotonic()
        for batch in loader:
            # Include full contexts/targets to audit that flips/windows match.
            exposure.update(batch["patient_id"][0].encode())
            for key in ("mask","target_times","context","targets"):
                exposure.update(batch[key].numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device,True):
                result=model(*_input(batch,device))
                loss,parts=trajectory_loss(result,batch)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite loss")
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            if not torch.isfinite(norm): raise FloatingPointError("non-finite gradient")
            optimizer.step()
            for k,v in {"loss":loss.detach(), **parts, **{k:result[k] for k in ("velocity_rms","source_rms")}}.items():
                sums[k]=sums.get(k,0)+float(v)
        scheduler.step()
        row={"epoch":epoch,"train":{k:v/len(loader) for k,v in sums.items()},
             "seconds":time.monotonic()-tic,"exposure_sha256":exposure.hexdigest()}
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            values=predict(model,validation,device,amp=True)
            mse=float(np.square(values["predictions"][:,-1]-values["targets"][:,-1]).mean())
            row["validation_mse"]=mse
            if mse<best:
                best=mse; best_epoch=epoch
                torch.save({"model":model.state_dict(),"config":config,"epoch":epoch,
                            "validation_primary_mse":mse},output/"best.pt")
        history.append(row)
        (output/"history.json").write_text(json.dumps(history,indent=2)+"\n")
        print(json.dumps({"variant":args.variant,**row}),flush=True)
    ckpt=torch.load(output/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(ckpt["model"])
    summary,p,y,x,ids=validation_diagnostics(model,validation,device)
    summary.update(best_epoch=best_epoch,best_validation_mse=best,
                   elapsed_seconds=time.monotonic()-start,exposure_sha256=exposure.hexdigest())
    (output/"validation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(output/"validation_predictions.npz",prediction=p,target=y,last_context=x,patient_ids=ids)
    print(json.dumps({"completed":args.variant,"best_epoch":best_epoch,"mean":summary["mean"],
                      "median":summary["median"],"diagnostic_means":summary["diagnostic_means"]}),flush=True)


def evaluate_final(args):
    ckpt=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    config=ckpt["config"]
    model=make_model(config["variant"],config).to(args.device)
    model.load_state_dict(ckpt["model"])
    values=predict(model,dataset("test"),torch.device(args.device),amp=True)
    result=save_evaluation(args.output,values["predictions"][:,-1],values["targets"][:,-1],
        values["last_context"],values["patient_ids"],title="ACDC normalized-coordinate trajectory")
    pair=Path(args.output)/"future_pairs"; pair.mkdir()
    for i,patient in enumerate(values["patient_ids"]):
        np.savez_compressed(pair/f"{patient}.npz",prediction=values["predictions"][i],
                           target=values["targets"][i],target_times=values["target_times"][i],
                           history_count=values["history_counts"][i])
    print(json.dumps({"test_mean":result["mean"],"test_median":result["median"]}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("interventions","overfit","train","test"))
    p.add_argument("--output",required=True)
    p.add_argument("--device",default="cuda:1")
    p.add_argument("--checkpoint",default="runs/patient_trajectory_acdc_seed42_unified_v1/best.pt")
    p.add_argument("--variant",choices=tuple(VARIANTS),default="scaled")
    p.add_argument("--steps",type=int,default=300)
    p.add_argument("--overfit-variants",nargs="+",choices=tuple(VARIANTS),
                   default=["original","scaled","scaled_no_source"])
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--base-channels",type=int,default=16)
    args=p.parse_args(); torch.set_num_threads(4)
    {"interventions":interventions,"overfit":overfit,"train":train_variant,"test":evaluate_final}[args.mode](args)


if __name__=="__main__": main()
