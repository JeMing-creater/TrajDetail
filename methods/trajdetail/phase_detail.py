"""Causal phase and amplitude readout for a frozen longitudinal forecast."""

import math

import torch
import torch.nn.functional as F

from methods.trajdetail.evaluate_detail_preservation import gaussian, highpass


class LocalSpectrum:
    """Half-overlapped in-plane sine windows with normalized overlap-add."""

    def __init__(self, base, proposal, size):
        self.shape = base.shape
        self.size = size
        self.stride = size // 2
        height, width = self.shape[-2:]
        self.pad = (self.stride, self.stride + (-width) % self.stride,
                    self.stride, self.stride + (-height) % self.stride)
        self.extent = (height + sum(self.pad[2:]), width + sum(self.pad[:2]))
        axis = torch.arange(size, device=base.device, dtype=base.dtype)
        taper = torch.sin(math.pi * (axis + .5) / size)
        self.window = taper[:, None] * taper
        self.base = self.analysis(base)
        self.proposal = self.analysis(proposal)
        self.norm = F.fold(self.window.square().flatten().view(1, -1, 1).expand(1, -1, self.base.shape[1]),
                           self.extent, size, stride=self.stride)[0, 0]
        dy = torch.fft.fftfreq(size, device=base.device)[:, None]
        dx = torch.fft.rfftfreq(size, device=base.device)
        self.radius = (dy.square() + dx.square()).sqrt()

    def analysis(self, volume):
        patches = F.unfold(F.pad(volume[:, None], self.pad, mode="reflect"), self.size, stride=self.stride)
        patches = patches.transpose(1, 2).reshape(*patches.shape[::2], self.size, self.size)
        return torch.fft.rfft2(patches * self.window, norm="ortho")

    def synthesis(self, spectrum):
        patches = torch.fft.irfft2(spectrum, s=(self.size, self.size), norm="ortho") * self.window
        columns = patches.flatten(-2).transpose(1, 2)
        value = F.fold(columns, self.extent, self.size, stride=self.stride)[:, 0] / self.norm
        y, x = self.pad[2], self.pad[0]
        return value[:, y:y + self.shape[-2], x:x + self.shape[-1]]

    def forecast(self, cutoff, phase, amplitude):
        """Blend angular offsets and magnitude independently in detail bands."""
        weight = ((self.radius - cutoff * .7) / (cutoff * .5)).clamp(0, 1)
        origin = self.base
        guide = self.proposal
        delta = torch.angle(guide * origin.conj())
        # Keep the real-valued Nyquist coefficients real.
        delta[..., 0, 0] = 0
        delta[..., 0, -1] = 0
        delta[..., self.stride, 0] = 0
        delta[..., self.stride, -1] = 0
        magnitude = origin.abs()
        magnitude = magnitude.lerp(guide.abs(), amplitude)
        detail = torch.polar(magnitude, torch.angle(origin) + phase * delta)
        # The proposal supplies coarse intensity, the anchor supplies detail.
        return self.synthesis(guide + weight * (detail - guide))


def detail_direction(base, proposal, kernel, trust):
    """Signed proposed edges plus a prototype fallback when phase is unreliable."""
    origin = highpass(base)
    new = highpass(proposal)
    local_mean = lambda value: gaussian(value, kernel)
    a = (local_mean(origin.square()) + 1.e-7).sqrt()
    b = (local_mean(new.square()) + 1.e-7).sqrt()
    support = (a.square() / (a.square() + .000015)).clamp(0., 1.)
    agreement = (local_mean(origin * new) / (a * b)).clamp(0., 1.)
    # Blend relative to local energy, so weak predicted bands do not win by size.
    direction = (agreement * trust) * (new * (a / b).clamp(.5, 5.)) + (1 - agreement * trust) * origin
    return highpass(direction * support)


def edge_projection(base, proposal, anchor, weight, proxy, ratio, passes):
    """Make the smallest one-direction edit required by a causal edge budget."""
    reference = highpass(base)
    observed = highpass(proposal)
    foreground = F.max_pool2d((base > .06).float()[:, None], 5, 1, 2)[:, 0] > 0
    if proxy == "union":
        support = torch.maximum(reference.abs(), observed.abs())
    elif proxy == "prediction":
        support = observed.abs()
    else:
        support = reference.abs()
    threshold = torch.quantile(support[foreground], .80)
    mask = foreground & (support >= threshold)
    desired = reference[mask].square().mean().sqrt() * ratio
    a = (gaussian(anchor.square(), 2.) + 1.e-6).sqrt()
    b = (gaussian(observed.square(), 2.) + 1.e-6).sqrt()
    direction = observed.lerp(anchor * (b / a).clamp(.25, 4.), weight) * foreground
    for _ in range(passes):
        direction = highpass(direction)
    shape = highpass(direction)[mask]
    x = observed[mask]
    power = shape.square().mean().clamp_min(1e-10)
    cross = (shape * x).mean()
    offset = desired.square() - x.square().mean()
    gain = offset / ((cross.square() + power * offset).clamp_min(0).sqrt() + cross).clamp_min(1e-9)
    return proposal + torch.where(offset > 0, gain.clamp(0, 8.), torch.zeros_like(gain)) * direction
