"""Fine-tune the BrLP autoencoder on the same volumes seen by its diffusion model."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from methods.brlp.brlp_adapter import AUTHOR, amp, checkpoint, pad, restore
from methods.brlp.brlp_audit import GROUPS, collect_records, decode, output_path


@torch.inference_mode()
def validate(model, records):
    errors = {}
    model.eval()
    for cohort, rows in records.items():
        per_patient = []
        for _, volumes, _ in rows:
            image = volumes[2].cuda()
            with amp():
                latent = model.encode(pad(image[None, None], 32))[0].float()[0]
            reconstruction = decode(model, latent, image.shape).clamp(0, 1)
            per_patient.append((image - reconstruction).square().mean().sqrt().item())
        errors[cohort] = float(np.mean(per_patient))
    return errors


def fit(model, train, validation, path, steps, lr, decoder_only):
    if decoder_only:
        model.requires_grad_(False)
        model.post_quant_conv.requires_grad_(True)
        model.decoder.requires_grad_(True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    frames = [(index, frame) for index, (_, _, times) in enumerate(train)
              for frame in (range(1, 3) if times[0] == times[1] else range(3))]
    generator = np.random.default_rng(42)
    best = float("inf")
    with path.with_name("whole_volume_reconstruction.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["updates", *validation])
        writer.writeheader()

        for step in range(steps + 1):
            if step % 100 == 0 or step == steps:
                errors = validate(model, validation)
                writer.writerow({"updates": step, **errors})
                stream.flush()
                print(f"Whole-volume reconstruction, step {step}: {errors}", flush=True)
                mean = float(np.mean(list(errors.values())))
                if mean < best:
                    best = mean
                    checkpoint(path, model, mean, step)
                model.train()
            if step == steps:
                break
            if step % len(frames) == 0:
                order = generator.permutation(len(frames))
            patient, frame = frames[order[step % len(frames)]]
            image = train[patient][1][frame:frame + 1, None].cuda()
            d, h, w = image.shape[-3:]
            optimizer.zero_grad(set_to_none=True)
            with amp():
                prediction, mu, sigma = model(pad(image, 32))
            reconstruction = prediction[..., :d, :h, :w].float()
            loss = F.l1_loss(reconstruction, image)
            if not decoder_only:
                # Scale the KL coefficient by the original patch's maximum latent size.
                regularity = .5 * (mu.float().square() + sigma.float().square() - 1
                                   - 2 * sigma.float().clamp_min(1e-6).log()).mean()
                loss += (1e-6 * 3 * 6 ** 3) * regularity
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite AE loss at iteration {step + 1}")
            loss.backward()
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(AUTHOR))
    from brlp import networks

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    stage = "full_volume_decoder" if args.decoder_only else "full_volume_ae"
    out = args.output / stage / f"fold{args.fold}" / args.group
    out.mkdir(parents=True, exist_ok=True)
    path = out / "brlp_ae.pt"
    model = networks.init_autoencoder().cuda()
    restore(output_path(args.group, args.fold) / "brlp_ae.pt", model)
    train = [row for rows in collect_records(args.group, args.fold, "train").values() for row in rows]
    validation = collect_records(args.group, args.fold, "validation")
    fit(model, train, validation, path, args.steps, args.lr, args.decoder_only)


if __name__ == "__main__":
    main()
