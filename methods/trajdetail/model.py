"""Small whole-volume, two-target conditional flow and temporal transition.

The model uses only the observed history as conditioning. The two queried
future volumes are a joint flow state, not teacher-forced observations. Its
zero-initialized flow starts at last-context prediction. ``noise_std=0`` gives
a deterministic history-source flow; it is not a probabilistic sample claim.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return math.gcd(channels, 8)


class ConditionedBlock(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels_in), channels_in)
        self.conv1 = nn.Conv3d(channels_in, channels_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels_out), channels_out)
        self.conv2 = nn.Conv3d(channels_out, channels_out, 3, padding=1)
        self.film = nn.Linear(time_dim, channels_out * 2)
        self.skip = nn.Identity() if channels_in == channels_out else nn.Conv3d(channels_in, channels_out, 1)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        y = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(embedding).chunk(2, dim=1)
        y = self.norm2(y) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        return self.skip(x) + self.conv2(F.silu(y))


class JointTrajectoryFlow(nn.Module):
    """Joint two-time 3-D image flow, with an independent history-aware ODE.

    Inputs use ``context[B,T,D,H,W]``, binary ``mask[B,T]``, timestamps
    ``context_times[B,T]`` and sorted future query times ``target_times[B,2]``.
    The network preserves the caller's time units. Absolute target timestamps
    are retained, which matters for perfusion phase; relative gaps are also
    represented. Callers must give missing slots mask=0, even if LOCF-filled.
    """

    def __init__(self, context_frames: int = 7, base_channels: int = 24, time_dim: int = 96):
        super().__init__()
        self.context_frames = context_frames
        self.base_channels = base_channels
        self.time_dim = time_dim
        # Mask, absolute observed times, relative observed times, absolute and
        # relative query times, plus Fourier features of flow time.
        time_inputs = 3 * context_frames + 4 + 17
        self.time_encoder = nn.Sequential(
            nn.Linear(time_inputs, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim), nn.SiLU()
        )
        self.register_buffer("frequencies", 2 ** torch.arange(8).float() * math.pi)
        channels = [base_channels * n for n in (1, 2, 3, 4)]
        # Clean history, summary, noisy/flow image, and image relative to LCI.
        self.stem = nn.Conv3d(context_frames + 7, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList(ConditionedBlock(c, c, time_dim) for c in channels)
        strides = ((1, 2, 2), (2, 2, 2), (2, 2, 2))
        self.down = nn.ModuleList(
            nn.Conv3d(channels[i], channels[i + 1], 3, stride=strides[i], padding=1)
            for i in range(3)
        )
        self.decoder = nn.ModuleList(
            ConditionedBlock(channels[i + 1] + channels[i], channels[i], time_dim)
            for i in (2, 1, 0)
        )
        self.flow_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.flow_head = nn.Conv3d(channels[0], 2, 3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

        # Independent, deliberately smaller temporal transition. Its history
        # features act as patient memory and are reused across integration steps.
        route_channels = max(8, base_channels // 2)
        self.history_memory = nn.Sequential(
            nn.Conv3d(context_frames + 3, route_channels, 3, padding=1),
            nn.GroupNorm(_groups(route_channels), route_channels), nn.SiLU(),
            nn.Conv3d(route_channels, route_channels, 3, padding=1), nn.SiLU(),
        )
        self.transition_stem = nn.Conv3d(route_channels + 5, route_channels, 3, padding=1)
        self.transition_block = ConditionedBlock(route_channels, route_channels, time_dim)
        self.transition_head = nn.Conv3d(route_channels, 1, 3, padding=1)
        nn.init.zeros_(self.transition_head.weight)
        nn.init.zeros_(self.transition_head.bias)

    @staticmethod
    def history_summary(context: Tensor, mask: Tensor, context_times: Tensor) -> dict[str, Tensor]:
        """Compute clean causal context statistics; masked values are ignored."""
        valid = mask.to(dtype=context.dtype)
        clean = torch.where(mask[:, :, None, None, None] > 0, context, 0.0)
        count = valid.sum(dim=1).clamp_min(1)
        slot_ids = torch.arange(context.shape[1], device=context.device)[None]
        last_index = torch.where(mask > 0, slot_ids, -1).amax(dim=1).clamp_min(0)
        batch_ids = torch.arange(context.shape[0], device=context.device)
        last = clean[batch_ids, last_index]
        last_time = context_times[batch_ids, last_index]
        mean = clean.sum(dim=1) / count[:, None, None, None]
        safe_times = torch.where(mask > 0, context_times, 0.0)
        mean_time = safe_times.sum(dim=1) / count
        centered_time = torch.where(mask > 0, context_times - mean_time[:, None], 0.0)
        time_variance = centered_time.square().sum(dim=1)
        slope = (clean * centered_time[:, :, None, None, None]).sum(dim=1)
        slope = slope / time_variance.clamp_min(1e-5)[:, None, None, None]
        slope = torch.where((time_variance > 1e-5)[:, None, None, None], slope, 0.0)
        # Clip this conditioning feature only, never the generated image.
        slope = slope.clamp(-4.0, 4.0)
        return {"clean": clean, "last": last, "mean": mean, "trend": slope, "last_time": last_time}

    def _time_embedding(self, tau: Tensor, mask: Tensor, context_times: Tensor,
                        target_times: Tensor, last_time: Tensor) -> Tensor:
        tau = tau.reshape(-1).to(context_times)
        angles = tau[:, None] * self.frequencies.to(tau)
        absolute_context = torch.where(mask > 0, context_times, 0.0)
        relative_context = torch.where(mask > 0, context_times - last_time[:, None], 0.0)
        features = torch.cat((
            mask.to(context_times), absolute_context, relative_context,
            target_times, target_times - last_time[:, None], tau[:, None],
            angles.sin(), angles.cos(),
        ), dim=1)
        return self.time_encoder(features)

    def forward(self, x_tau: Tensor, tau: Tensor, context: Tensor, mask: Tensor,
                context_times: Tensor, target_times: Tensor) -> Tensor:
        if context.shape[1] != self.context_frames or x_tau.shape[1] != 2:
            raise ValueError("Expected configured history length and exactly two joint future volumes")
        summary = self.history_summary(context, mask, context_times)
        reference = summary["last"][:, None]
        statistics = torch.stack((summary["last"], summary["mean"], summary["trend"]), dim=1)
        x = torch.cat((x_tau, x_tau - reference, summary["clean"], statistics), dim=1)
        embedding = self._time_embedding(tau, mask, context_times, target_times, summary["last_time"])
        x = self.stem(x)
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x, embedding)
            if i < 3:
                skips.append(x)
                x = self.down[i](x)
        for block, skip in zip(self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1), embedding)
        return self.flow_head(F.silu(self.flow_norm(x)))

    def make_source(self, context: Tensor, mask: Tensor, context_times: Tensor,
                    target_times: Tensor, noise_std: float = 0.0,
                    generator: torch.Generator | None = None) -> Tensor:
        last = self.history_summary(context, mask, context_times)["last"]
        source = last[:, None].expand(-1, 2, -1, -1, -1).clone()
        if noise_std:
            # Full-rank two-query covariance with positive temporal correlation.
            # Query consistency under arbitrary additions is not claimed here.
            shared = torch.randn(last.shape, device=last.device, dtype=last.dtype, generator=generator)
            separate = torch.randn(source.shape, device=last.device, dtype=last.dtype, generator=generator)
            noise = (shared[:, None] + separate) / math.sqrt(2.0)
            source = source + float(noise_std) * noise
        return source

    def predict(self, context: Tensor, mask: Tensor, context_times: Tensor,
                target_times: Tensor, steps: int = 10, noise_std: float = 0.0,
                generator: torch.Generator | None = None, source: Tensor | None = None) -> Tensor:
        """Differentiable Euler rollout. Use inference_mode externally for eval."""
        if steps < 1:
            raise ValueError("steps must be positive")
        x = self.make_source(context, mask, context_times, target_times, noise_std, generator) if source is None else source
        for step in range(steps):
            tau = context.new_full((context.shape[0],), step / steps)
            x = x + self(x, tau, context, mask, context_times, target_times) / steps
        return x

    def transition(self, image_a: Tensor, context: Tensor, mask: Tensor,
                   context_times: Tensor, time_a: Tensor, time_b: Tensor,
                   steps: int = 2) -> Tensor:
        """Integrate an independently parameterized clinical-time velocity.

        No future ground-truth volume enters this function. Both paths retain
        the clean historical memory; this first prototype has deterministic
        memory, and is not a claim of full augmented-state stochastic closure.
        """
        if steps < 1:
            raise ValueError("steps must be positive")
        summary = self.history_summary(context, mask, context_times)
        statistics = torch.stack((summary["last"], summary["mean"], summary["trend"]), dim=1)
        memory = self.history_memory(torch.cat((summary["clean"], statistics), dim=1))
        state = image_a[:, None] if image_a.ndim == 4 else image_a
        dt = (time_b - time_a) / steps
        for step in range(steps):
            now = time_a + step * dt
            # Only current time and one step ahead; no direct full-future decoder.
            query = torch.stack((now, now + dt), dim=1)
            embedding = self._time_embedding(torch.ones_like(now), mask, context_times, query, summary["last_time"])
            features = torch.cat((state, state - summary["last"][:, None], statistics, memory), dim=1)
            features = self.transition_stem(features)
            rate = self.transition_head(F.silu(self.transition_block(features, embedding)))
            state = state + dt[:, None, None, None, None] * rate
        return state[:, 0] if image_a.ndim == 4 else state


__all__ = ["JointTrajectoryFlow"]
