"""Run the released Δ-LFM direct-change recipe on the fixed DLBS patient folds."""

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.losses import PatchAdversarialLoss, PerceptualLoss

from src.compare import cases
from src.metrics import score

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT / "methods/delta_lfm/upstream"
sys.path.insert(0, str(AUTHOR))
from methods.delta_lfm.upstream.src.autoencoder.maisi.scripts.utils import define_instance
from methods.delta_lfm.upstream.src.change_f1 import change_f1
from methods.delta_lfm.upstream.src.model3D.networks import init_large_latent_diffusion, init_patch_discriminator


def dump(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def normalize(x):
    mean = x.mean((-1, -2, -3), keepdim=True)
    std = x.std((-1, -2, -3), keepdim=True).clamp_min(1e-5)
    return (x - mean) / std, mean, std


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def patch_triplet(volume, rng, augment):
    x = volume[:, None].cuda().float()
    if augment:
        a, b, c = rng.uniform(-np.pi / 36, np.pi / 36, size=3)
        sa, sb, sc, ca, cb, cc = np.sin(a), np.sin(b), np.sin(c), np.cos(a), np.cos(b), np.cos(c)
        transform = x.new_tensor([[cb*cc, sa*sb*cc-ca*sc, ca*sb*cc+sa*sc],
                                  [cb*sc, sa*sb*sc+ca*cc, ca*sb*sc-sa*cc],
                                  [-sb, sa*cb, ca*cb]])
        xyz = x.new_tensor(x.shape[-3:][::-1])
        transform = transform * xyz[None] / xyz[:, None]
        offset = x.new_tensor(rng.uniform(-3, 3, size=3)) * 2 / xyz
        affine = torch.cat((transform, offset[:, None]), dim=1)[None].expand(3, -1, -1)
        grid = F.affine_grid(affine, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
    # Legacy cardiac and perfusion grids have fewer than 48 axial slices.
    limits = tuple(min(side, 48) for side in x.shape[-3:])
    starts = tuple(int(rng.integers(side - length + 1))
                   for side, length in zip(x.shape[-3:], limits, strict=True))
    return x[(..., *(slice(start, start + length) for start, length in zip(starts, limits, strict=True)))]


def angular_order(mu):
    first, middle, last = mu.float().flatten(1)
    segment1, segment2 = middle - first, last - middle
    if first.equal(middle):
        return mu.new_zeros(())
    angle = 1 - F.cosine_similarity(segment1, segment2, dim=0)
    chord = last - first
    total = chord.square().sum().clamp_min(1e-10)
    order = F.relu(.2 - (segment1 * chord).sum() / total)
    order = order + F.relu(.2 - (segment2 * chord).sum() / total)
    return .35 * angle + .2 * order


def ae_loss(x, recon, mu, sigma, perceptual):
    gt, pred = x[-1] - x[0], recon[-1] - recon[0]
    mask = (x[-1] > .05) | (x[0] > .05)
    # Observable change energy weights are built on the training triplet.
    energy = (gt.abs() / gt.abs().quantile(.95).clamp_min(1e-5)).clamp(max=1)
    weight = energy * mask
    gt_centered = gt - (gt * weight).sum() / weight.sum().clamp_min(1)
    pred_centered = pred - (pred * weight).sum() / weight.sum().clamp_min(1)
    direction = 1 - F.cosine_similarity(
        (pred_centered * weight.sqrt()).flatten(), (gt_centered * weight.sqrt()).flatten(), dim=0)
    change = ((pred - gt).abs() * (1 + weight)).mean()
    kl = .5 * (mu.square() + sigma.square() - 2 * sigma.clamp_min(1e-6).log() - 1).flatten(1).sum(1).mean()
    padding = tuple(value for side in reversed(x.shape[-3:]) for value in (0, max(0, 32 - side)))
    return (F.l1_loss(recon, x) + 1e-7 * kl + .3 * (change + direction)
            + .3 * perceptual(F.pad(recon, padding) * 2 - 1, F.pad(x, padding) * 2 - 1) + angular_order(mu))


def load_ae(weight=None):
    with (AUTHOR / "src/autoencoder/maisi/inference_args.pkl").open("rb") as file:
        init = pickle.load(file)
    init.autoencoder_def["num_splits"] = 1
    init.autoencoder_def["norm_float16"] = False
    model = define_instance(init, "autoencoder_def")
    if weight is not None:
        model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    return model.cuda()


@torch.inference_mode()
def validate_ae(model, validation, cohort_break):
    model.eval()
    errors = []
    for _, volumes, _ in validation:
        raw = volumes[:, None].cuda().float()
        x, mean, std = normalize(raw)
        with autocast():
            z, _ = model.encode(x)
            recon = model.decode(z).float() * std + mean
        errors.append((recon.clamp(0, 1) - raw).square().flatten(1).mean(1).sqrt().mean().item())
    return cohort_average(errors, cohort_break)


def balanced_order(size, split, rng):
    if split is None:
        return rng.permutation(size)
    first, second = rng.permutation(split), rng.permutation(size - split) + split
    count = max(len(first), len(second))
    pool = np.stack((np.resize(first, count), np.resize(second, count)), axis=1).ravel()
    return rng.permutation(pool)


def cohort_average(values, split):
    if split is None:
        return float(np.mean(values))
    return float((np.mean(values[:split]) + np.mean(values[split:])) / 2)


def fit_ae(model, train, validation, args):
    checkpoint = args.output / "ae.pt"
    log_file = args.output / "ae_history.json"
    if checkpoint.exists() and log_file.exists():
        history = json.loads(log_file.read_text())
        if history[-1]["step"] == args.ae_steps:
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
            return
    rng = np.random.default_rng(args.seed)
    perceptual = PerceptualLoss(3, "squeeze", is_fake_3d=True, fake_3d_ratio=.2).cuda().eval()
    perceptual.requires_grad_(False)
    discriminator = init_patch_discriminator().cuda()
    adversarial = PatchAdversarialLoss(criterion="least_squares")
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    best, history = math.inf, []

    def evaluate(step):
        nonlocal best
        error = validate_ae(model, validation, args.val_break)
        row = {"step": step, "rmse": error}
        history.append(row)
        if error < best:
            best = error
            torch.save({"model": model.state_dict(), **row}, checkpoint)
        dump(log_file, history)
        print("autoencoder validation", row, flush=True)
        model.train()

    evaluate(0)
    order = balanced_order(len(train), args.train_break, rng)
    for step in range(1, args.ae_steps + 1):
        if step > 1 and (step - 1) % len(order) == 0:
            order = balanced_order(len(train), args.train_break, rng)
        _, volumes, _ = train[order[(step - 1) % len(order)]]
        x = patch_triplet(volumes, rng, augment=True)
        norm, mean, std = normalize(x)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            decoded, mu, sigma = model(norm)
        recon = decoded.float() * std + mean
        strength = .1 * min(1., max(0, step - 400) / 1200)
        loss = ae_loss(x, recon, mu.float(), sigma.float(), perceptual)
        padding = tuple(value for side in reversed(x.shape[-3:]) for value in (0, max(0, 32 - side)))
        if strength:
            discriminator.requires_grad_(False)
            with autocast():
                loss = loss + strength * adversarial(discriminator(F.pad(recon, padding))[-1], True, False)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"AE loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()
        if strength:
            discriminator.requires_grad_(True)
            disc_opt.zero_grad(set_to_none=True)
            with autocast():
                fake = adversarial(discriminator(F.pad(recon.detach(), padding))[-1], False, True)
                real = adversarial(discriminator(F.pad(x, padding))[-1], True, True)
            ((fake + real) * .5).backward()
            disc_opt.step()
        if step == 1 or step % 25 == 0:
            print(f"autoencoder step {step}/{args.ae_steps}: {loss.item():.5f}", flush=True)
        if step == args.ae_steps or step % args.ae_validate == 0:
            evaluate(step)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])


@torch.inference_mode()
def latents(model, group, include_target=True):
    model.eval()
    result = {}
    for patient, volumes, _ in group:
        raw = volumes[:, None].cuda().float()
        with autocast():
            z, _ = model.encode(normalize(raw if include_target else raw[:2])[0])
        result[patient] = z.float()
    return result


def pairs(group, encoded, include_all_pairs):
    indices = ((0, 1), (0, 2), (1, 2)) if include_all_pairs else ((1, 2),)
    rows = []
    for patient, _, times in group:
        for anchor, target in indices:
            elapsed = float((times[target] - times[anchor]) * 10)
            z = encoded[patient]
            previous_dt = float((times[1] - times[0]) * 10) if anchor else 0.
            rows.append((z[anchor], z[anchor-1] if anchor else z[anchor], previous_dt, elapsed, z[target]))
    return rows


def predict_delta(model, anchor, previous, previous_dt, elapsed, stats, delta=None, sigma=0.,
                  *, state=None, flow_time=None):
    shape = (-1, 1, 1, 1, 1)
    times = torch.stack((elapsed, torch.log1p(elapsed)), dim=1)
    cov = (times - stats[0]) / stats[1]
    context = F.pad((elapsed / 100)[:, None], (8, 1))[:, None]
    labels = elapsed.long().clamp(0, 14)
    present = (previous_dt > 0).float().view(shape)
    previous_dt = previous_dt.view(shape)
    old_rate = (anchor - previous) / previous_dt.clamp_min(.25) * present
    history = [old_rate, previous_dt.expand_as(anchor[:, :1]), present.expand_as(anchor[:, :1])]
    if delta is not None:
        t = torch.rand(anchor.shape[0], device=anchor.device).clamp(.001, .999)
        x = torch.lerp(torch.randn_like(anchor) * sigma, delta, t.view(shape))
    else:
        t = torch.full_like(elapsed, 0. if flow_time is None else flow_time)
        x = torch.zeros_like(anchor) if state is None else state
    inputs = torch.cat([x, anchor, *history, torch.zeros_like(anchor), torch.zeros_like(anchor[:, :2])], dim=1)
    with autocast():
        return model(inputs, t, context=context, class_labels=labels,
                     cond_vec=F.pad(cov, (0, 8))).float()


def fit_loss(pred, delta, fine_tune):
    gls = (pred - delta).square()
    if not fine_tune:
        return gls.mean()
    magnitude = delta.abs().mean(1, keepdim=True)
    weight = 1 + 5 * magnitude / magnitude.amax((2, 3, 4), keepdim=True).clamp_min(1e-6)
    gt, generated = delta.flatten(1), pred.flatten(1)
    tau = (.25 * torch.quantile(gt.abs(), .99, dim=1, keepdim=True)).clamp_min(1e-6)
    g = gt.abs() / (gt.abs() + tau)
    p = generated.abs() / (generated.abs() + tau)
    rec = ((g * gt * generated).sum(1) / ((g * gt.square()).sum(1) + 1e-9)).clamp(0, 1)
    pre = ((p * gt * generated).sum(1) / ((p * generated.square()).sum(1) + 1e-9)).clamp(0, 1)
    cf1 = 2 * pre * rec / (pre + rec + 1e-9)
    return (gls * weight).sum() / weight.expand_as(gls).sum() + .5 * (1 - cf1).mean()


def forecast(model, ae, z, volumes, times, baseline=None):
    anchor, previous = z[1:2], z[:1]
    history, horizon = ((times[1] - times[0]).reshape(1) * 10,
                        (times[2] - times[1]).reshape(1) * 10)
    if baseline is None:
        baseline = decode_reference(ae, anchor, volumes[1])
    raw, dec, slope = baseline
    with autocast():
        generated = ae.decode(anchor + predict_delta(model, anchor, previous, history, horizon, forecast.stats))
    return (raw + slope * (generated.float().squeeze() - dec)).clamp(0, 1)


def decode_reference(ae, anchor, raw):
    raw = raw.cuda().float()
    with autocast():
        decoded = ae.decode(anchor).float().squeeze()
    support = raw > .05
    if not support.any():
        support = torch.ones_like(raw, dtype=torch.bool)
    x, y = decoded[support], raw[support]
    xc = x - x.mean()
    slope = (xc * (y - y.mean())).sum() / xc.square().sum().clamp_min(1e-6)
    return raw, decoded, slope


@torch.inference_mode()
def evaluate_flow(model, ae, patients, encoded, baselines, output=None, cohort_break=None, integrated_dataset="lumiere"):
    model.eval()
    rows = []
    for patient, volumes, times in patients:
        pred = forecast(model, ae, encoded[patient], volumes, times.cuda(), baselines[patient])
        truth = volumes[2].float()
        rmse = (pred - truth.cuda()).square().mean().sqrt().item()
        if output is None:
            rows.append(rmse)
        else:
            guess = pred.cpu().numpy()
            target = truth.numpy()
            row = score(guess, target, integrated_dataset, torch.device("cuda"))
            row.update(change_f1(guess, target, volumes[1].float().numpy()))
            rows.append({"patient_id": patient, **row})
            np.save(output / f"{patient}.npy", guess.astype(np.float16))
    return cohort_average(rows, cohort_break) if output is None else rows


def save_evaluation(rows, output, epoch):
    partial = output / "test_cases.csv.part"
    with partial.open("w") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metrics = {key: float(np.mean([row[key] for row in rows if key in row]))
               for key in fields if key != "patient_id"}
    dump(output / "test_summary.json", {"test_count": len(rows), "epoch": epoch, **metrics})
    partial.replace(output / "test_cases.csv")
    print("test", metrics, flush=True)


@torch.inference_mode()
def evaluate_integrated(ae, test, args):
    ae_state = torch.load(args.output / "ae.pt", map_location="cpu", weights_only=True)
    ae.load_state_dict(ae_state["model"])
    ae.eval()
    with torch.device("meta"):
        model = init_large_latent_diffusion(in_channels=20, use_image=False, out_channels=4,
                                            num_class_embeds=15, cond_dim=10)
    state = torch.load(getattr(args, "flow_checkpoint", args.output / "flow.pt"),
                       map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state["model"], assign=True)
    model.cuda().eval()
    stats = tuple(value.cuda() for value in state["stats"])
    encoded = latents(ae, test, include_target=False)
    if args.replay_direct:
        forecast.stats = stats
        destination = args.result_dir or (args.output / args.eval_cohort if args.with_mu else args.output)
        predictions = destination / "predictions"
        predictions.mkdir(parents=True, exist_ok=True)
        baselines = {p: decode_reference(ae, encoded[p][1:2], v[1]) for p, v, _ in test}
        rows = evaluate_flow(model, ae, test, encoded, baselines, output=predictions,
                             integrated_dataset=args.eval_cohort)
        save_evaluation(rows, destination, int(state["epoch"]))
        return
    output = args.output / f"euler{args.eval_steps}"
    if args.with_mu:
        output = output / args.eval_cohort
    if args.result_dir is not None:
        output = args.result_dir
    predictions = output / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(0, len(test), args.eval_batch):
        group = test[index:index+args.eval_batch]
        previous, anchor = torch.stack([encoded[patient] for patient, _, _ in group], dim=1)
        times = torch.stack([time for _, _, time in group]).cuda() * 10
        history = times[:, 1] - times[:, 0]
        horizon = times[:, 2] - times[:, 1]
        state_delta = torch.zeros_like(anchor)
        with autocast():
            for step in range(args.eval_steps):
                t = step / args.eval_steps
                denoised = predict_delta(model, anchor, previous, history, horizon, stats,
                                         state=state_delta, flow_time=t)
                velocity = (denoised - state_delta) / (1 - t)
                state_delta = state_delta + velocity / args.eval_steps
            decoded = ae.decode(anchor + state_delta).float()
        for position, (patient, volumes, _) in enumerate(group):
            raw, ref, slope = decode_reference(ae, anchor[position:position+1], volumes[1])
            guess = (raw + slope * (decoded[position, 0] - ref)).clamp(0, 1).cpu().numpy()
            target = volumes[2].float().numpy()
            row = score(guess, target, args.eval_cohort, torch.device("cuda"))
            row.update(change_f1(guess, target, volumes[1].float().numpy()))
            rows.append({"patient_id": patient, **row})
            np.save(predictions / f"{patient}.npy", guess.astype(getattr(args, "prediction_dtype", np.float16)))
        print(f"Euler {args.eval_steps}: {len(rows)}/{len(test)}", flush=True)
    save_evaluation(rows, output, int(state["epoch"]))


def fit_flow(ae, train, validation, test, args):
    train_latents = latents(ae, train)
    val_latents, test_latents = [latents(ae, group, include_target=False) for group in (validation, test)]
    rows = pairs(train, train_latents, args.all_train_pairs)
    data = [torch.stack(field) if torch.is_tensor(field[0]) else torch.tensor(field, device="cuda")
            for field in zip(*rows)]
    anchor, prev, prev_t, elapsed, target = data
    delta = target - anchor
    sigma = delta.std().item()
    time_feats = torch.stack((elapsed, torch.log1p(elapsed)), dim=1)
    forecast.stats = (time_feats.mean(0), time_feats.std(0).clamp_min(1e-5))
    model = init_large_latent_diffusion(in_channels=20, use_image=False, out_channels=4,
                                        num_class_embeds=15, cond_dim=10).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=1e-6, fused=True)
    with torch.inference_mode():
        val_ref, test_ref = [{p: decode_reference(ae, zs[p][1:2], v[1]) for p, v, _ in group}
                             for group, zs in ((validation, val_latents), (test, test_latents))]
    checkpoint = getattr(args, "flow_checkpoint", args.output / "flow.pt")
    history = []
    rng = np.random.default_rng(args.seed)
    train_break = args.train_break
    if train_break is not None and args.all_train_pairs:
        train_break *= 3
    best = math.inf
    print(f"flow sigma={sigma:.5f}, pairs={len(rows)}, parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    for epoch in range(1, args.base_epochs + args.fine_epochs + 1):
        fine_tune = epoch > args.base_epochs
        if epoch == args.base_epochs + 1:
            optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=1e-6, fused=True)
        model.train()
        losses = []
        order = (torch.randperm(len(rows), device="cuda") if train_break is None else
                 torch.as_tensor(balanced_order(len(rows), train_break, rng), device="cuda"))
        for i, indices in enumerate(order.split(args.batch_size)):
            optimizer.zero_grad(set_to_none=True)
            count = len(indices)
            step_loss = 0.
            for micro in indices.split(args.microbatch):
                pr = predict_delta(model, anchor[micro], prev[micro], prev_t[micro], elapsed[micro],
                                   forecast.stats, delta=delta[micro], sigma=sigma)
                loss = fit_loss(pr, delta[micro], fine_tune)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"flow loss at epoch {epoch}: {loss.item()}")
                (loss * (len(micro) / count)).backward()
                step_loss += loss.item() * len(micro) / count
            optimizer.step()
            losses.append(step_loss)
            if i == 0 and epoch == 1:
                print(f"flow first batch loss={step_loss:.5f}", flush=True)
        val = evaluate_flow(model, ae, validation, val_latents, val_ref, cohort_break=args.val_break)
        row = {"epoch": epoch, "phase": "fine_tune" if fine_tune else "base",
               "loss": float(np.mean(losses)), "validation_rmse": val}
        history.append(row)
        dump(args.output / "flow_history.json", history)
        print("flow validation", row, flush=True)
        if fine_tune and val < best:
            best = val
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save({"model": model.state_dict(), "stats": forecast.stats, **row}, temporary)
            temporary.replace(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    if getattr(args, "legacy_euler", False):
        return
    test_groups = [("dlbs", args.output, test)] if not args.with_mu else [
        ("dlbs", args.output / "dlbs", test[:args.test_break]),
        ("mu", args.output / "mu", test[args.test_break:])]
    for cohort, report_dir, group in test_groups:
        output = report_dir / "predictions"
        output.mkdir(parents=True, exist_ok=True)
        evaluated = evaluate_flow(model, ae, group, test_latents, test_ref, output=output, integrated_dataset=cohort)
        save_evaluation(evaluated, report_dir, int(state["epoch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--ae-steps", type=int, default=600)
    parser.add_argument("--ae-validate", type=int, default=200)
    parser.add_argument("--base-epochs", type=int, default=30)
    parser.add_argument("--fine-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-train-pairs", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--replay-direct", action="store_true")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--with-mu", action="store_true")
    parser.add_argument("--eval-cohort", choices=("dlbs", "mu"), default="dlbs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    manifest_path = ROOT / f"runs/baseline_3fold_20261122/fold{args.fold}/dlbs/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["time_unit_days"] != 3652.5:
        raise ValueError("Elapsed-time conversion expects the shared DLBS manifest in decades")
    cache = Path("/dev/shm/generation_new_longitudinal_20261121/dlbs")
    partitions = ("train", "validation", "test")
    train, validation, test = [list(cases(manifest, cache, part, torch.device("cpu")))
                               for part in partitions]
    args.train_break = args.val_break = args.test_break = None
    if args.with_mu:
        mu_manifest = json.loads((manifest_path.parents[1] / "mu/manifest.json").read_text())
        if manifest["shape_zyx"] != mu_manifest["shape_zyx"]:
            raise ValueError("DLBS and MU must have the same analysis grid")
        scale = mu_manifest["time_unit_days"] / manifest["time_unit_days"]
        mu = [[(patient, vol, times * scale)
               for patient, vol, times in cases(mu_manifest, cache.with_name("mu"), part, torch.device("cpu"))]
              for part in partitions]
        args.train_break, args.val_break, args.test_break = len(train), len(validation), len(test)
        train, validation, test = [group + other for group, other in zip((train, validation, test), mu, strict=True)]
    ids = [row[0] for part in (train, validation, test) for row in part]
    if len(ids) != len(set(ids)):
        raise ValueError("DLBS patient folds overlap")
    ae = load_ae(args.weight)
    if args.preflight:
        train, validation, test = train[:2], validation[:2], test[:2]
        args.train_break = args.val_break = args.test_break = None
    if args.eval_steps or args.replay_direct:
        if args.with_mu and not args.preflight:
            test = test[:args.test_break] if args.eval_cohort == "dlbs" else test[args.test_break:]
        evaluate_integrated(ae, test, args)
        return
    dump(args.output / "protocol.json", {**vars(args), "output": str(args.output),
         "weight": str(args.weight), "manifest": str(manifest_path),
         "counts": [len(train), len(validation), len(test)],
         "raw_support": "last observed MRI only", "flow_channels": [256, 512, 768]})
    fit_ae(ae, train, validation, args)
    ae.eval().requires_grad_(False)
    fit_flow(ae, train, validation, test, args)


if __name__ == "__main__":
    main()
