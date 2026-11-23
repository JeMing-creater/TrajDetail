"""BrLP's KL autoencoder, latent diffusion and ControlNet on common 3-D windows."""

from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from src.comparator_windows import checkpoint_path

AUTHOR = Path(__file__).resolve().parents[2] / "methods/brlp/upstream/src"


def progression_summary(volume):
    """Five fractions of the volume above fixed intensity thresholds."""
    thresholds = volume.new_tensor([.05, .15, .25, .40, .60])
    return (volume[None] > thresholds[:, None, None, None]).float().mean((1, 2, 3))


def observed_features(volumes, times):
    previous, current = (progression_summary(volume) for volume in volumes[:2])
    history, horizon = times.diff()
    has_history = (history > 0).float()
    time = torch.stack((history.log1p(), horizon.log1p(), has_history))
    return torch.cat((torch.ones_like(time[:1]), time, current, (current - previous) * has_history)), time


def fit_progression(train):
    features = torch.stack([observed_features(volumes, times)[0] for _, volumes, times in train])
    targets = torch.stack([progression_summary(volumes[2]) for _, volumes, _ in train])
    penalty = torch.eye(features.shape[1]) * .01
    penalty[0, 0] = 0
    return torch.linalg.solve(features.T @ features + penalty, features.T @ targets)


def amp():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def pad(volume, multiple):
    pads = tuple(value for size in reversed(volume.shape[-3:]) for value in (0, (-size) % multiple))
    return F.pad(volume, pads)


def checkpoint(path, model, metric, step):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "loss": metric, "step": step}, temporary)
    temporary.replace(path)


def restore(path, model):
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state["model"])


def fit_autoencoder(model, discriminator, train, validation, path, steps, rng):
    from monai.losses import PatchAdversarialLoss, PerceptualLoss

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=5e-5)
    perceptual = PerceptualLoss(3, "squeeze", is_fake_3d=True, fake_3d_ratio=.2).cuda().eval().requires_grad_(False)
    adv = PatchAdversarialLoss(criterion="least_squares")
    best = float("inf")
    frames = [(index, frame) for index, (_, volumes, times) in enumerate(train)
              for frame in (range(1, 3) if times[0] == times[1] else range(3))]
    order = rng.permutation(len(frames))
    for step in range(1, steps + 1):
        if step > 1 and (step - 1) % len(order) == 0:
            order = rng.permutation(len(frames))
        patient, frame = frames[order[(step - 1) % len(order)]]
        volume = train[patient][1][frame:frame+1, None].cuda()
        limits = tuple(min(side, 48) for side in volume.shape[-3:])
        starts = [rng.integers(side - limit + 1) for side, limit in zip(volume.shape[-3:], limits)]
        x = pad(volume[(..., *(slice(start, start + limit) for start, limit in zip(starts, limits)))], 8)
        optimizer.zero_grad(set_to_none=True)
        with amp():
            decoded, mean, std = model(x)
        loss = F.l1_loss(decoded.float(), x)
        loss += 1e-6 * (.5 * (mean.float().square() + std.float().square() - 1
                              - 2 * std.float().clamp_min(1e-6).log()).flatten(1).sum(1).mean())
        padding = tuple(value for side in reversed(x.shape[-3:]) for value in (0, max(0, 32 - side)))
        loss += .01 * perceptual(F.pad(decoded.float(), padding), F.pad(x, padding))
        if step > steps // 2:
            discriminator.requires_grad_(False)
            with amp():
                loss += .01 * adv(discriminator(F.pad(decoded, padding))[-1], True, False)
        loss.backward()
        optimizer.step()
        if step > steps // 2:
            discriminator.requires_grad_(True)
            disc_optimizer.zero_grad(set_to_none=True)
            with amp():
                disc_loss = (adv(discriminator(F.pad(decoded.detach(), padding))[-1], False, True)
                             + adv(discriminator(F.pad(x, padding))[-1], True, True)) / 2
            disc_loss.backward()
            disc_optimizer.step()
        if step == steps or step % max(1, steps // 3) == 0:
            model.eval()
            with torch.no_grad(), amp():
                errors = []
                for _, volumes, _ in validation:
                    images = volumes[1:, None].cuda()
                    source = pad(images, 8)
                    rec = model.decode(model.encode(source)[0])
                    d, h, w = images.shape[-3:]
                    errors.append(F.l1_loss(rec[..., :d, :h, :w].float(), images).item())
            value = float(np.mean(errors))
            print(f"BrLP AE {step}/{steps}, validation MAE {value:.6f}", flush=True)
            if value < best:
                best = value
                checkpoint(path, model, best, step)
            model.train()
    restore(path, model)


def denoise(model, control, z, time, covariates, condition):
    if control is None:
        return model(x=z, timesteps=time, context=covariates).float()
    down, middle = control(x=z, timesteps=time, context=covariates, controlnet_cond=condition)
    return model(x=z, timesteps=time, context=covariates,
                 down_block_additional_residuals=down, mid_block_additional_residual=middle).float()


@torch.no_grad()
def latent_inputs(autoencoder, records, progression, include_future):
    latent, covariates = [], []
    for _, volumes, times in records:
        features, time = observed_features(volumes, times)
        prediction = (features @ progression).clamp(0, 1)
        covariates.append(torch.cat((time, prediction)))
        with amp():
            inputs = pad(volumes[1:, None] if include_future else volumes[1:2, None], 32).cuda()
            latent.append(autoencoder.encode(inputs)[0].float())
    return torch.stack(latent), torch.stack(covariates).cuda()


def conditions(anchor, covariates):
    elapsed = covariates[:, 0].reshape(-1, 1, 1, 1, 1)
    return torch.cat((anchor, elapsed.expand_as(anchor[:, :1])), dim=1)


def fit_denoiser(model, control, train, validation, path, epochs, batch, scheduler):
    component = model if control is None else control
    if control is not None:
        model.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(component.parameters(), lr=2.5e-5, fused=True)
    best = float("inf")

    def objective(data, index, generator=None):
        z, c = (value[index] for value in data)
        anchor, target = z.unbind(dim=1)
        noise = torch.randn(target.shape, device="cuda", generator=generator)
        t = torch.randint(1000, (len(index),), device="cuda", generator=generator)
        noisy = scheduler.add_noise(target, noise, t)
        with amp():
            pred = denoise(model, control, noisy, t, c[:, None], conditions(anchor, c))
        return F.mse_loss(pred, noise)

    for epoch in range(1, epochs + 1):
        component.train()
        for indices in torch.randperm(len(train[0]), device="cuda").split(batch):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(train, indices)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BrLP denoising loss")
            loss.backward()
            optimizer.step()
        if epoch == epochs or epoch % 5 == 0 or epoch == 1:
            component.eval()
            generator = torch.Generator(device="cuda").manual_seed(42)
            with torch.no_grad():
                vals = [objective(validation, indices, generator).item()
                        for indices in torch.arange(len(validation[0]), device="cuda").split(batch)]
            val = float(np.mean(vals))
            print(f"BrLP {'ControlNet' if control is not None else 'latent diffusion'} {epoch}/{epochs}, val {val:.6f}", flush=True)
            if val < best:
                best = val
                checkpoint(path, component, best, epoch)
    restore(path, component)
    component.eval()


def train_and_predict(train, validation, test, output, smoke, settings=None):
    settings = settings or {}
    sys.path.insert(0, str(AUTHOR))
    from brlp import networks
    from generative.networks.schedulers import DDIMScheduler, DDPMScheduler

    autoencoder = networks.init_autoencoder().cuda()
    ae_path = output / "brlp_ae.pt"
    ae_done = output / "brlp_ae.done"
    if not ae_done.exists():
        fit_autoencoder(autoencoder, networks.init_patch_discriminator().cuda(), train, validation, ae_path,
                        2 if smoke else settings.get("ae_steps", 600), np.random.default_rng(42))
        ae_done.touch()
    restore(ae_path, autoencoder)
    autoencoder.eval().requires_grad_(False)
    trajectory = fit_progression(train)
    tr, va = [latent_inputs(autoencoder, split, trajectory, True) for split in (train, validation)]
    scale = tr[0][:, 1].std().clamp_min(1e-5).reciprocal()
    tr, va = [(z * scale, c) for z, c in (tr, va)]
    model = networks.init_latent_diffusion().cuda()
    schedule = dict(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=.0015, beta_end=.0205)
    scheduler = DDPMScheduler(**schedule)
    diffusion_path = checkpoint_path(output, "brlp_diffusion.pt")
    diffusion_done = output / "brlp_diffusion.done"
    if not diffusion_done.exists():
        fit_denoiser(model, None, tr, va, diffusion_path, 1 if smoke else settings.get("diffusion_epochs", 30), 2, scheduler)
        diffusion_done.touch()
    restore(diffusion_path, model)
    control = networks.init_controlnet().cuda()
    control_path = checkpoint_path(output, "brlp_control.pt")
    control_done = output / "brlp_control.done"
    if not control_done.exists():
        control.load_state_dict(model.state_dict(), strict=False)
        fit_denoiser(model, control, tr, va, control_path, 1 if smoke else settings.get("control_epochs", 15), 2, scheduler)
        control_done.touch()
    restore(control_path, control)
    model.eval()
    control.eval()
    if not test:
        return
    scheduler = DDIMScheduler(**schedule, clip_sample=False)
    scheduler.set_timesteps(5 if smoke else 50)
    inputs = latent_inputs(autoencoder, test, trajectory, False)
    generator = torch.Generator(device="cuda").manual_seed(42)
    with torch.no_grad():
        for start in range(0, len(test), 4):
            anchor, c = inputs[0][start:start+4, 0] * scale, inputs[1][start:start+4]
            z = torch.randn(anchor.shape, device="cuda", generator=generator)
            cond = conditions(anchor, c)
            for step in scheduler.timesteps:
                time = torch.full((len(z),), int(step), device="cuda", dtype=torch.long)
                with amp():
                    noise = denoise(model, control, z, time, c[:, None], cond)
                z, _ = scheduler.step(noise, int(step), z)
            for offset, latent in enumerate(z.split(1)):
                patient, volumes, _ = test[start+offset]
                with amp():
                    image = autoencoder.decode(latent / scale)[0, 0].float()
                d, h, w = volumes.shape[-3:]
                yield patient, image[:d, :h, :w].clamp(0, 1).cpu().numpy()
