from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from src.forecaster_input import unpack_context as _unpack_context


class _InceptionMixer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        branch = channels // 4
        if branch < 1 or channels % 8:
            raise ValueError("SimVP width must be divisible by 8")
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, branch, 1),
            nn.Conv2d(channels, branch, 3, padding=1),
            nn.Conv2d(channels, branch, 3, padding=2, dilation=2),
            nn.Conv2d(channels, branch, 5, padding=2),
        ])
        self.mix = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.mix(torch.cat([layer(x) for layer in self.branches], dim=1))))


class SimVPForecaster(nn.Module):
    """SimVP-style CNN encoder, latent translator, and volume decoder."""

    def __init__(self, volume_shape, width=32, num_frames=7, num_mixer_blocks=4):
        super().__init__()
        self.volume_shape = tuple(volume_shape)
        self.num_frames = num_frames
        depth, _, _ = self.volume_shape
        latent = 2 * width
        self.encoder = nn.Sequential(
            nn.Conv2d(num_frames * depth, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, latent, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(latent, latent, 3, stride=2, padding=1), nn.GELU(),
        )
        self.translator = nn.Sequential(*[_InceptionMixer(latent) for _ in range(num_mixer_blocks)])
        self.decoder = nn.Sequential(
            nn.Conv2d(latent, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, depth, 3, padding=1),
        )

    def forward(self, context, context_time=None):
        frames = _unpack_context(context, self.volume_shape)
        batch, steps, depth, height, width = frames.shape
        if steps != self.num_frames:
            raise ValueError(f"expected {self.num_frames} context frames, got {steps}")
        features = self.translator(self.encoder(frames.reshape(batch, steps * depth, height, width)))
        features = F.interpolate(features, size=(height, width), mode="bilinear", align_corners=False)
        prediction = self.decoder(features)
        return torch.sigmoid(prediction).unsqueeze(1)
