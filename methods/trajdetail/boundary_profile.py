"""Correct the location and transition width of predicted MRI boundaries."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from methods.trajdetail.composed_readout import local_average
from methods.trajdetail.unified_model import warp_volume


def derivatives(image: Tensor):
    padded = F.pad(image.float(), (1, 1, 1, 1, 0, 0), mode="replicate")
    center = padded[..., 1:-1, 1:-1]
    left, right = padded[..., 1:-1, :-2], padded[..., 1:-1, 2:]
    top, bottom = padded[..., :-2, 1:-1], padded[..., 2:, 1:-1]
    dx = .5 * (right - left)
    dy = .5 * (bottom - top)
    dxx = right + left - 2 * center
    dyy = bottom + top - 2 * center
    dxy = .25 * (padded[..., 2:, 2:] + padded[..., :-2, :-2] -
                 padded[..., 2:, :-2] - padded[..., :-2, 2:])
    return dx, dy, dxx, dyy, dxy


def boundary_basis(image, transported):
    dx, dy, dxx, dyy, dxy = derivatives(image)
    smooth = F.avg_pool3d(image.float(), (1, 3, 3), stride=1, padding=(0, 1, 1),
                         count_include_pad=False)
    sx, sy, *_ = derivatives(smooth)
    tx, ty, *_ = derivatives(transported)
    norm = (sx.square() + sy.square()).add(1e-6).sqrt()
    nx, ny = sx / norm, sy / norm
    slope = nx * dx + ny * dy
    curvature = nx.square() * dxx + 2 * nx * ny * dxy + ny.square() * dyy
    evidence = nx * tx + ny * ty
    support = norm / (norm + .025)
    return slope, curvature, evidence, support


class BoundaryProfile(nn.Module):
    """Predict bounded shifts and profile compression along the local normal."""

    def __init__(self, *, use_shift, use_width):
        super().__init__()
        self.use_shift = use_shift
        self.use_width = use_width
        self.head = nn.Sequential(
            nn.Conv3d(10, 12, (1, 5, 5), padding=(0, 2, 2)), nn.SiLU(),
            nn.Conv3d(12, 12, (3, 3, 3), padding=1), nn.SiLU(),
            nn.Conv3d(12, 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, anchor, transported, displacement, source, previous, elapsed):
        slope, curvature, evidence, support = boundary_basis(previous, transported)
        features = torch.cat((previous, transported, slope, curvature, evidence, support,
                              displacement / 4., elapsed.expand_as(previous)), dim=1)
        offset, width = self.head(features).float().split(1, dim=1)
        age = (11. * elapsed / 7.).clamp(0., 1.)
        # Translation uses the normal derivative. Width correction uses the
        # normal curvature, with opposite intensity updates on either side.
        correction = torch.zeros_like(previous)
        if self.use_shift:
            correction = correction + .65 * offset.tanh() * slope
        if self.use_width:
            agreement = (evidence / (slope.abs() + .025)).clamp(0., 1.)
            correction = correction - 1.5 * width.tanh() * curvature * (.25 + .75 * agreement)
        return previous + age * support * correction


class BoundaryTransport(nn.Module):
    """Choose a local normal offset before transferring the observed detail band."""

    def __init__(self, reference, *, use_offsets=True):
        super().__init__()
        self.reference = reference.requires_grad_(False)
        self.use_offsets = use_offsets
        self.head = nn.Sequential(
            nn.Conv3d(11, 12, (1, 5, 5), padding=(0, 2, 2)), nn.SiLU(),
            nn.Conv3d(12, 12, 3, padding=1), nn.SiLU(),
            nn.Conv3d(12, 4, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, anchor, transported, displacement, source, previous, elapsed):
        with torch.no_grad():
            rendered = self.reference(anchor, transported, displacement, source, previous, elapsed)
            previous_detail = previous - local_average(previous)
            detail = rendered - local_average(rendered)
            smooth = F.avg_pool3d(previous.float(), (1, 3, 3), stride=1, padding=(0, 1, 1),
                                 count_include_pad=False)
            dx, dy, *_ = derivatives(smooth)
            norm = (dx.square() + dy.square()).add(1e-6).sqrt()
            direction = torch.cat((dx / norm, dy / norm, torch.zeros_like(norm)), dim=1)
            candidates = (warp_volume(detail, -.65 * direction), detail,
                          warp_volume(detail, .65 * direction))
            updates = torch.cat([candidate - previous_detail for candidate in candidates], dim=1)
            texture_energy = local_average(previous_detail.square())
            agreement = torch.cat([
                local_average(previous_detail * candidate) /
                (texture_energy * local_average(candidate.square())).clamp_min(1e-10).sqrt()
                for candidate in candidates
            ], dim=1).clamp(0., 1.)
        features = torch.cat((previous, previous_detail, norm, updates, agreement,
                              source, elapsed.expand_as(previous)), dim=1)
        state = self.head(features).float()
        prior = state.new_tensor([-2., 0., -2.]).view(1, 3, 1, 1, 1)
        weights = (state[:, :3] + prior).softmax(dim=1) if self.use_offsets else None
        confidence = (.25 + .25 * state[:, 3:].tanh()) * (elapsed * 11.).clamp(0., 1.)
        supported = agreement.square() * updates
        update = ((weights * supported).sum(dim=1, keepdim=True) if weights is not None
                  else supported[:, 1:2])
        return previous + confidence * update
