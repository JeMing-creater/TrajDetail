"""4DLoG registration and spatiotemporal diffusion on the common causal windows."""

from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from methods.brlp.brlp_adapter import amp, checkpoint, restore

AUTHOR = Path(__file__).resolve().parents[2] / "methods/log4d/upstream"


def smoothness(field):
    return sum(field.diff(dim=axis).square().mean() for axis in (-1, -2, -3)) / 3


def fit_registration(model, train, validation, path, epochs):
    from velocity.losses import MSE

    similarity = MSE().loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(42)
    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for index in rng.permutation(len(train)):
            _, volume, times = train[index]
            source = volume[:1, None].cuda()
            target = (volume[2:] if times[0] == times[1] else volume[1:])[:, None].cuda()
            optimizer.zero_grad(set_to_none=True)
            warped, _, field, _ = model(source.expand_as(target), target)
            loss = similarity(target, warped) + .01 * smoothness(field)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite 4DLoG registration loss")
            loss.backward()
            optimizer.step()
        if epoch == epochs or epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                vals = []
                for _, volume, _ in validation:
                    source, target = volume[[0, 2], None].cuda().split(1)
                    warped, _, field, _ = model(source, target)
                    vals.append((similarity(target, warped) + .01 * smoothness(field)).item())
            value = float(np.mean(vals))
            print(f"4DLoG registration {epoch}/{epochs}, validation {value:.6f}", flush=True)
            if value < best:
                best = value
                checkpoint(path, model, best, epoch)
    restore(path, model)


def pad_cube(volume, size):
    d, h, w = volume.shape[-3:]
    return F.pad(volume, (0, size-w, 0, size-h, 0, size-d))


@torch.no_grad()
def velocity_inputs(registration, records, size, include_future):
    fields, gradients, times = [], [], []
    for _, volume, time in records:
        source = volume[:1, None].cuda()
        # Compute the gradient in the observed image, on the velocity grid.
        grad = torch.stack(torch.gradient(source[:, 0], dim=(-3, -2, -1)), dim=1)
        grad = F.interpolate(grad, size=tuple(s // 4 for s in volume.shape[-3:]), mode="trilinear", align_corners=False)
        gradients.append(pad_cube(grad[0], size))
        seen = volume[1:3 if include_future else 2, None].cuda()
        _, _, velocity, _ = registration(source.expand_as(seen), seen)
        if time[0] == time[1]:
            velocity[0].zero_()
        fields.append(pad_cube(velocity, size).tanh().transpose(0, 1))
        times.append(time)
    return torch.stack(fields), torch.stack(gradients), torch.stack(times).cuda()


def model_output(diffusion, x, time, gradient, visit_times):
    label = torch.zeros(len(x), dtype=torch.long, device=x.device)
    with amp():
        return diffusion.denoiser(x, time, label, gradient, visit_times).float()


def fit_diffusion(diffusion, train, validation, path, epochs):
    opt = torch.optim.AdamW(diffusion.parameters(), lr=1e-4, fused=True)
    best = float("inf")
    batch = 4

    def objective(data, indices, rng=None):
        velocity, gradient, visit_times = (value[indices] for value in data)
        time = torch.randint(diffusion.num_timesteps, (len(indices),), device="cuda", generator=rng)
        noise = torch.randn(velocity.shape, device="cuda", generator=rng)
        noisy = diffusion.q_sample(velocity, time, noise)
        # The first slot is the observed registration and stays clean in both
        # training and prediction. Only the future velocity receives noise.
        noisy[:, :, 0] = velocity[:, :, 0]
        estimate = model_output(diffusion, noisy, time, gradient, visit_times)
        return F.l1_loss(estimate[:, :, 1], noise[:, :, 1])

    for epoch in range(1, epochs + 1):
        diffusion.train()
        for index in torch.randperm(len(train[0]), device="cuda").split(batch):
            opt.zero_grad(set_to_none=True)
            loss = objective(train, index)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite 4DLoG diffusion loss")
            loss.backward()
            opt.step()
        if epoch == epochs or epoch % 5 == 0 or epoch == 1:
            diffusion.eval()
            rng = torch.Generator(device="cuda").manual_seed(42)
            with torch.no_grad():
                values = [objective(validation, index, rng).item()
                          for index in torch.arange(len(validation[0]), device="cuda").split(batch)]
            value = float(np.mean(values))
            print(f"4DLoG diffusion {epoch}/{epochs}, validation {value:.6f}", flush=True)
            if value < best:
                best = value
                checkpoint(path, diffusion, best, epoch)
    restore(path, diffusion)


@torch.no_grad()
def sample_svf(diffusion, history, gradient, visit_times, seed):
    """DDPM predictor and one adaptive Langevin correction per active step."""
    rng = torch.Generator(device="cuda").manual_seed(seed)

    def noise():
        return torch.randn(history[:, :, 0].shape, device="cuda", generator=rng)

    future = noise()
    for step in range(diffusion.num_timesteps - 1, -1, -1):
        t = torch.full((len(history),), step, dtype=torch.long, device="cuda")
        noisy = torch.cat((history, future[:, :, None]), dim=2)
        pred = model_output(diffusion, noisy, t, gradient, visit_times)[:, :, 1]
        x0 = diffusion.predict_start_from_noise(future, t=t, noise=pred).clamp(-.9999, .9999)
        future, _, logvar = diffusion.q_posterior(x0, future, t)
        if step:
            future += (0.5 * logvar).exp() * noise()
            noisy[:, :, 1] = future
            sigma = diffusion.sqrt_one_minus_alphas_cumprod[step].clamp_min(1e-8)
            score = -model_output(diffusion, noisy, t, gradient, visit_times)[:, :, 1] / sigma
            norm = score.flatten(1).norm(dim=1).clamp_min(1e-8)
            step_size = (2 * (.16 * sigma / norm).square()).clamp(1e-6, 1e-3).view(-1, 1, 1, 1, 1)
            future = future + step_size * score + (2 * step_size).sqrt() * noise()
    return future.clamp(-.9999, .9999).atanh()


def train_and_predict(train, validation, test, output, smoke=False, settings=None):
    settings = settings or {}
    sys.path.insert(0, str(AUTHOR))
    from diffusion_m import DVIT_Diffusion
    from dvit import DViT
    from velocity.networks import DIGIT_reg

    image_shape = tuple((train or test)[0][1].shape[-3:])
    model = DIGIT_reg(inshape=image_shape, nb_unet_features=[[16, 32, 32, 16], [16, 32, 32, 16, 16, 3]],
                      int_downsize=8, use_skip=True).cuda()
    registration_path = output / "4dlog_registration.pt"
    registration_done = output / "4dlog_registration.done"
    if not registration_done.exists():
        fit_registration(model, train, validation, registration_path, 1 if smoke else settings.get("registration_epochs", 30))
        registration_done.touch()
    restore(registration_path, model)
    model.eval().requires_grad_(False)
    d, h, w = (side // 4 for side in image_shape)
    size = max(d, h, w)
    size += (-size) % 4
    denoiser = DViT(patch_size=4, embed_dim=192, img_size=size, num_heads=6,
                   depth=6, num_classes=1, num_frames=2)
    diffusion = DVIT_Diffusion(denoiser, channels=3, image_size=size, timesteps=5 if smoke else settings.get("sampling_steps", 100),
                               num_frames=2).cuda()
    path = output / "4dlog_diffusion.pt"
    diffusion_done = output / "4dlog_diffusion.done"
    if not diffusion_done.exists():
        tr, val = [velocity_inputs(model, split, size, True) for split in (train, validation)]
        fit_diffusion(diffusion, tr, val, path, 1 if smoke else settings.get("diffusion_epochs", 60))
        diffusion_done.touch()
    restore(path, diffusion)
    diffusion.eval()
    inputs = velocity_inputs(model, test, size, False)
    for start in range(0, len(test), 4):
        history, gradient, visit_times = (value[start:start+4] for value in inputs)
        velocity = sample_svf(diffusion, history, gradient, visit_times, 42 + start)
        with torch.no_grad():
            images = torch.stack([volume[0] for _, volume, _ in test[start:start+4]], dim=0)[:, None].cuda()
            recovered = model.get_full_deform(velocity[..., :d, :h, :w], images)[0]
        for (patient, _, _), image in zip(test[start:start+4], recovered[:, 0].cpu().numpy(), strict=True):
            yield patient, image.clip(0, 1)
