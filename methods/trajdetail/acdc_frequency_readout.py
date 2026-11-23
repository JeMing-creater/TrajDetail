"""Native-resolution synthesis of missing detail on a frozen patient trajectory."""

import torch
from torch import nn
import torch.nn.functional as F

from methods.trajdetail.composed_readout import curvature_corrected_warp
from methods.trajdetail.unified_model import warp_volume


def blur(image):
    """In-plane, separable binomial filter with a width close to 0.8 voxel."""
    padded = F.pad(image.float(), (1, 1, 1, 1, 0, 0), mode="replicate")
    horizontal = .25 * padded[..., :-2] + .5 * padded[..., 1:-1] + .25 * padded[..., 2:]
    return .25 * horizontal[..., :-2, :] + .5 * horizontal[..., 1:-1, :] + .25 * horizontal[..., 2:, :]


def band(image):
    return image.float() - blur(image)


class FrequencyReadout(nn.Module):
    """Use the transported anchor for texture evidence and synthesize signed updates."""

    def __init__(self, *, align_detail=False):
        super().__init__()
        self.align_detail = align_detail
        self.encoder = nn.Sequential(
            nn.Conv3d(11, 20, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(20, 24, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(24, 24, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv3d(24, 12, 1), nn.SiLU(),
        )
        self.detail = nn.Sequential(
            nn.Conv3d(23, 20, (1, 5, 5), padding=(0, 2, 2)), nn.SiLU(),
            nn.Conv3d(20, 16, (1, 3, 3), padding=(0, 1, 1)), nn.SiLU(),
            nn.Conv3d(16, 5 if align_detail else 2, 1),
        )
        self.signal = nn.Conv3d(12, 1, 1)
        for layer in (self.detail[-1], self.signal):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.detail_amount = 1.
        self.coordinate_gain = 1.

    def components(self, anchor, transported, displacement, source, previous, elapsed):
        rendered = curvature_corrected_warp(anchor, displacement)
        past_detail = band(rendered)
        previous_detail = band(previous)
        features = torch.cat((anchor, transported, rendered, previous, source,
                              displacement / 4., past_detail, previous_detail,
                              elapsed.expand_as(previous)), dim=1)
        context = self.encoder(features)
        signal = F.interpolate(self.signal(context).float(), size=anchor.shape[-3:],
                               mode="trilinear", align_corners=False)
        context = F.interpolate(context, size=anchor.shape[-3:], mode="trilinear",
                                align_corners=False)
        output = self.detail(torch.cat((features, context), dim=1)).float()
        generated, evidence = output[:, :1], output[:, 1:2]
        if self.align_detail:
            # Estimate a small local phase correction in the fine-detail branch.
            # Coarse position and intensity retain the original trajectory.
            limits = output.new_tensor([1.3, 1.3, .25]).view(1, 3, 1, 1, 1)
            shift = self.coordinate_gain * limits * output[:, 2:].tanh()
            aligned_displacement = shift + warp_volume(displacement, shift)
            past_detail = band(curvature_corrected_warp(anchor, aligned_displacement))
        time_weight = (elapsed * 11. / 6.).clamp(0., 1.)
        # Predict signed details independently of transport. The second output
        # can reuse aligned detail carried by the original observed image.
        transport_update = evidence.tanh() * (past_detail - previous_detail)
        if self.align_detail:
            # This term is already a frequency-band difference. Removing the
            # slow gate envelope avoids filtering away its mid frequencies twice.
            transport_update = transport_update - blur(blur(transport_update))
            update = band(.20 * generated.tanh()) + transport_update
        else:
            update = band(.20 * generated.tanh() + transport_update)
        return previous + time_weight * .12 * signal.tanh(), time_weight * update

    def forward(self, *args):
        base, detail = self.components(*args)
        return base + self.detail_amount * detail


class ResidualFlowReadout(nn.Module):
    """Flow matching in image intensities for the missing detail band."""

    def __init__(self, steps=4):
        super().__init__()
        self.steps = steps
        self.detail_amount = 1.
        self.encoder = nn.Sequential(
            nn.Conv3d(11, 20, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(20, 24, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(24, 24, 3, padding=1), nn.SiLU(),
            nn.Conv3d(24, 12, 1),
        )
        self.context_detail = nn.Sequential(nn.Conv3d(23, 16, (1, 3, 3), padding=(0, 1, 1)), nn.SiLU())
        self.velocity = nn.Sequential(
            nn.Conv3d(19, 20, (1, 3, 3), padding=(0, 1, 1)), nn.SiLU(),
            nn.Conv3d(20, 16, (1, 3, 3), padding=(0, 1, 1)), nn.SiLU(),
            nn.Conv3d(16, 1, 1),
        )
        self.signal = nn.Conv3d(12, 1, 1)
        for layer in (self.velocity[-1], self.signal):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def condition(self, anchor, transported, displacement, source, previous, elapsed):
        rendered = curvature_corrected_warp(anchor, displacement)
        previous_detail, transported_detail = band(previous), band(rendered)
        time_weight = (elapsed * 11. / 6.).clamp(0., 1.)
        features = torch.cat((anchor, transported, rendered, previous, source,
                              displacement / 4., previous_detail, transported_detail,
                              elapsed.expand_as(previous)), dim=1)
        context = self.encoder(features)
        x0 = .40 * (transported_detail - previous_detail)
        signal = F.interpolate(self.signal(context).float(), size=anchor.shape[-3:],
                               mode="trilinear", align_corners=False)
        context = F.interpolate(context, size=anchor.shape[-3:], mode="trilinear", align_corners=False)
        detail_context = self.context_detail(torch.cat((features, context), dim=1)).float()
        return previous + time_weight * .12 * signal.tanh(), x0, detail_context, time_weight

    def rate(self, state, flow_time, condition):
        features = torch.cat((condition, state, band(state),
                              flow_time.expand_as(state)), dim=1)
        return band(.33 * self.velocity(features).float().tanh())

    def training_pair(self, values, target):
        base, start, condition, time_weight = self.condition(*values)
        end = band(target - base.detach())
        time_shape = (len(target), 1, 1, 1, 1)
        time = torch.rand(time_shape, device=target.device)
        # Endpoint examples also supervise the initial velocity used at inference.
        time = time * (torch.rand(time_shape, device=target.device) > .25)
        state = torch.lerp(start, end, time)
        velocity = self.rate(state, time, condition)
        endpoint = base + time_weight * (state + (1. - time) * velocity)
        return endpoint, velocity, end - start

    def components(self, *values):
        base, state, condition, time_weight = self.condition(*values)
        for step in range(self.steps):
            velocity = self.rate(state, state.new_tensor(step / self.steps), condition)
            state = state + velocity / self.steps
        return base, time_weight * state

    def forward(self, *values):
        base, detail = self.components(*values)
        return base + self.detail_amount * detail
