"""Replay BrLP with latent averaging and observed-image-only repair variants."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from methods.brlp.brlp_adapter import fit_progression, latent_inputs, restore
from methods.brlp.brlp_audit import (GROUPS, case_iter, collect_records, decode, load_models,
                                        mask_exterior, output_path, samples, scale_latents, stats)
from src.release_io import write_csv


@torch.inference_mode()
def forecasts(ae, model, control, record, trajectory, scale, variant, count, steps, start, seed, decoder):
    _, volumes, _ = record
    source = volumes[1].cuda()
    z, c = latent_inputs(ae, [record], trajectory, include_future=False)
    anchor = z[0, 0]
    latent = samples(model, control, anchor * scale, c[0], seed, count,
                     start=start if variant in ("warm", "warm_residual") else None, steps=steps)
    generated = latent.mean(0) / scale
    estimate = decode(ae, generated, source.shape)
    alternate = decode(decoder, generated, source.shape) if decoder is not None else None
    if variant == "warm_residual":
        estimate = source + (estimate - decode(ae, anchor, source.shape))
    elif variant == "las_mask":
        estimate = mask_exterior(estimate, source)
        if alternate is not None:
            alternate = mask_exterior(alternate, source)
    return estimate, alternate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--variant", choices=("las", "las_mask", "warm", "warm_residual"), required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--start", type=int, default=160)
    parser.add_argument("--decoder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or not 1 <= args.steps <= 1000 or not 0 <= args.start < 1000:
        parser.error("Invalid sampler parameters")
    name = f"{args.variant}_volume_decoder" if args.decoder else args.variant
    output = args.output / name / f"fold{args.fold}" / args.group / args.split
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"group": args.group, "fold": args.fold, "split": args.split, "variant": args.variant,
                "samples": args.samples, "steps": args.steps, "start": args.start}
    if args.decoder:
        protocol["decoder"] = str(args.decoder.resolve())
    recipe = output / "protocol.json"
    if recipe.is_file() and json.loads(recipe.read_text()) != protocol:
        raise ValueError(f"Existing run uses a different sampling protocol: {recipe}")
    recipe.write_text(json.dumps(protocol, indent=2) + "\n")
    cohorts = [cohort for cohort, _ in case_iter(args.group, args.fold, args.split)]
    if all((output / cohort / "complete").exists() for cohort in cohorts):
        return
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    ae, model, control = load_models(args.group, args.fold)
    decoder = None
    if args.decoder:
        if args.variant != "las_mask":
            parser.error("The whole-volume decoder evaluates the LAS+mask variant")
        from brlp import networks

        reference = torch.load(output_path(args.group, args.fold) / "brlp_ae.pt", map_location="cpu",
                               weights_only=True, mmap=True)["model"]
        updated = torch.load(args.decoder, map_location="cpu", weights_only=True, mmap=True)["model"]
        for key in reference.keys() - {k for k in reference if k.startswith(("post_quant_conv.", "decoder."))}:
            if not torch.equal(reference[key], updated[key]):
                raise ValueError(f"Updated encoder parameter would change the latent distribution: {key}")
        decoder = networks.init_autoencoder().cuda().eval().requires_grad_(False)
        restore(args.decoder, decoder)
    training = [row for rows in collect_records(args.group, args.fold, "train").values() for row in rows]
    trajectory = fit_progression(training)
    scale = scale_latents(ae, training, trajectory)
    del training
    for cohort, records in case_iter(args.group, args.fold, args.split):
        path = output / cohort
        cached = path / "predictions"
        cached.mkdir(parents=True, exist_ok=True)
        score_path = path / "cases.csv"
        existing = {}
        if score_path.is_file():
            with score_path.open(newline="") as stream:
                existing = {row["patient_id"]: row for row in csv.DictReader(stream)}
        rows = []
        for position, record in enumerate(records, 1):
            patient, volumes, _ = record
            pred_path = cached / f"{patient}.npy"
            if patient in existing and pred_path.is_file():
                rows.append(existing[patient])
                continue
            seed = int.from_bytes(hashlib.sha256(f"{cohort}/{patient}".encode()).digest()[:4], "big")
            original, repaired = forecasts(ae, model, control, record, trajectory, scale, args.variant,
                                            args.samples, args.steps, args.start, seed, decoder)
            prediction = original if repaired is None else repaired
            values = stats(prediction, volumes[2].cuda(), volumes[1].cuda())
            if repaired is not None:
                original_stats = stats(original, volumes[2].cuda(), volumes[1].cuda())
                values["original_decoder_nrmse"] = original_stats["nrmse"]
            temp = pred_path.with_suffix(".tmp.npy")
            np.save(temp, prediction.clamp(0, 1).cpu().numpy().astype(np.float16), allow_pickle=False)
            temp.replace(pred_path)
            rows.append(dict(patient_id=patient, **values))
            write_csv(score_path, rows + [row for key, row in existing.items()
                                          if key not in {done["patient_id"] for done in rows}])
            if position == 1 or position % 10 == 0:
                print(cohort, args.fold, args.variant, args.split, position, values["nrmse"], flush=True)
        write_csv(score_path, rows)
        (path / "complete").touch()
        print(cohort, args.fold, args.variant, args.split, len(rows),
              float(np.mean([float(row["nrmse"]) for row in rows])), flush=True)


if __name__ == "__main__":
    main()
