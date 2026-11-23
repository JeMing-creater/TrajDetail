"""Render a transported MRI from the last observation and a separate signal field."""

import torch
from torch import nn
import torch.nn.functional as F

from methods.trajdetail.unified_model import warp_volume


def local_average(image: torch.Tensor) -> torch.Tensor:
    return F.avg_pool3d(image, (1, 5, 5), stride=1, padding=(0, 2, 2), count_include_pad=False)


def curvature_corrected_warp(anchor: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Apply an in-plane correction for the curvature lost in a subvoxel linear sample."""
    padded = F.pad(anchor, (1, 1, 1, 1, 0, 0), mode="replicate")
    dx2 = padded[..., 1:-1, 2:] + padded[..., 1:-1, :-2] - 2 * anchor
    dy2 = padded[..., 2:, 1:-1] + padded[..., :-2, 1:-1] - 2 * anchor
    image, cx, cy = warp_volume(torch.cat((anchor, dx2, dy2), dim=1), displacement).split(1, dim=1)
    phase = displacement[:, :2] - displacement[:, :2].floor()
    alpha_x, alpha_y = (phase * (1. - phase)).split(1, dim=1)
    return image - .5 * (alpha_x * cx + alpha_y * cy)


class ComposedReadout(nn.Module):
    """Calibrate displacement and signal at each query of a frozen trajectory."""

    def __init__(self, *, curvature=False):
        super().__init__()
        self.render = curvature_corrected_warp if curvature else warp_volume
        self.geometry_and_signal = nn.Sequential(
            nn.Conv3d(9, 16, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(16, 24, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(24, 24, 3, padding=1), nn.SiLU(),
            nn.Conv3d(24, 4, 3, padding=1),
        )
        nn.init.zeros_(self.geometry_and_signal[-1].weight)
        nn.init.zeros_(self.geometry_and_signal[-1].bias)
        self.register_buffer("displacement_limit", torch.tensor([2., 2., .6]).view(1, 3, 1, 1, 1))

    def forward(self, anchor, transported, displacement, source, previous, elapsed):
        # Use the old readout only for low-frequency intensity calibration.
        # Anatomical details come from the last observed scan at the composed coordinates.
        baseline = transported + source
        coarse_signal = local_average(previous - baseline)
        context = torch.cat((anchor, transported, source, previous, coarse_signal,
                             displacement / 4., elapsed.expand_as(transported)), dim=1)
        fields = self.geometry_and_signal(context)
        fields = F.interpolate(fields.float(), size=anchor.shape[-3:],
                               mode="trilinear", align_corners=False)
        scale = (elapsed * 11. / 7.).clamp(0., 2.)
        geometry = scale * self.displacement_limit * fields[:, :3].tanh()
        signal = scale * .20 * fields[:, 3:].tanh()
        image = self.render(anchor, displacement + geometry) + source + coarse_signal + signal
        return image


class BandwiseReadout(nn.Module):
    """Control the contribution of the rendered intensity and detail independently."""

    def __init__(self, reference):
        super().__init__()
        self.reference = reference.requires_grad_(False)
        self.reliability = nn.Sequential(
            nn.Conv3d(9, 16, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(16, 16, 3, padding=1), nn.SiLU(),
            nn.Conv3d(16, 3, 3, padding=1),
        )
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)

    def forward(self, anchor, transported, displacement, source, previous, elapsed):
        with torch.no_grad():
            rendered = self.reference(anchor, transported, displacement, source, previous, elapsed)
        delta = rendered - previous
        low = local_average(delta)
        detail = delta - low
        texture = previous - local_average(previous)
        correlation = local_average(texture * (texture + detail))
        scale = (local_average(texture.square()) *
                 local_average((texture + detail).square())).clamp_min(1e-8).sqrt()
        agreement = (correlation / scale).clamp(-1., 1.)
        features = torch.cat((previous, low, detail, agreement, source,
                              displacement / 4., elapsed.expand_as(previous)), dim=1)
        corrections = self.reliability(features)
        corrections = F.interpolate(corrections.float(), size=anchor.shape[-3:],
                                    mode="trilinear", align_corners=False)
        low_gain = .5 * corrections[:, :1].tanh()
        detail_gain = (corrections[:, 1:2] - 1.1).sigmoid()
        signal = .05 * corrections[:, 2:].tanh() * (elapsed * 11. / 7.).clamp(0., 2.)
        return previous + low_gain * low + detail_gain * detail + signal


class TransportedDetailReadout(nn.Module):
    """Replace only the detail band and attenuate it when local alignment is poor."""

    def __init__(self, reference, strength, filtered):
        super().__init__()
        self.reference = reference.requires_grad_(False)
        self.strength = strength
        self.filtered = filtered

    def forward(self, anchor, transported, displacement, source, previous, elapsed):
        rendered = self.reference(anchor, transported, displacement, source, previous, elapsed)
        previous_detail = previous - local_average(previous)
        rendered_detail = rendered - local_average(rendered)
        update = rendered_detail - previous_detail
        if self.filtered:
            cross = local_average(previous_detail * rendered_detail)
            energy = (local_average(previous_detail.square()) *
                      local_average(rendered_detail.square())).clamp_min(1e-10).sqrt()
            confidence = (cross / energy).clamp(0., 1.).square()
            update = confidence * update
        return previous + self.strength * update


def in_plane_gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ((value[..., 2:, 1:-1] - value[..., :-2, 1:-1]) * .5,
            (value[..., 1:-1, 2:] - value[..., 1:-1, :-2]) * .5)


def training_objective(prediction, target, edge_weight: float):
    weights = prediction.new_tensor([1., 2.])
    if prediction.shape[0] != weights.numel():
        raise ValueError("Training windows must have exactly two query frames")
    difference = prediction - target
    pixel = (difference.square() + .01 * difference.abs()).flatten(1).mean(1)
    if edge_weight:
        py, px = in_plane_gradient(prediction)
        ty, tx = in_plane_gradient(target)
        strength = (ty.square() + tx.square()).sqrt()
        focus = (strength / .06).clamp(max=3.)
        # Direction-sensitive matching penalizes strong edges in the wrong place.
        error = ((py - ty).square() + (px - tx).square() + 1e-6).sqrt() - 1e-3
        edge = (focus * error).flatten(1).sum(1) / focus.flatten(1).sum(1).clamp_min(1.)
        pixel = pixel + edge_weight * edge
    return (pixel * weights).sum() / weights.sum()
