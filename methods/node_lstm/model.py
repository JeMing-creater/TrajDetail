from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchdiffeq import odeint
from src.forecaster_input import unpack_context as _unpack_context


class _LatentDynamics(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_size, latent_size), nn.Tanh(),
            nn.Linear(latent_size, latent_size),
        )

    def forward(self, _time, state):
        return 0.1 * self.network(state)

class NodeLSTMForecaster(nn.Module):
    """ODE-LSTM adaptation with a CNN volume encoder and decoder."""

    def __init__(self, volume_shape, width=32, num_frames=7, latent_size=None):
        super().__init__()
        self.volume_shape = tuple(volume_shape)
        self.num_frames = num_frames
        slices, height, image_width = self.volume_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(slices, width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * width, latent_size or max(64, 4 * width)),
        )
        latent = latent_size or max(64, 4 * width)
        self.dynamics = _LatentDynamics(latent)
        self.cell = nn.LSTMCell(latent, latent)
        base_h, base_w = max(4, height // 16), max(4, image_width // 16)
        self.base_size = (base_h, base_w)
        self.decoder_in = nn.Linear(latent, 4 * width * base_h * base_w)
        self.decoder_mid = nn.Sequential(
            nn.Conv2d(4 * width, 2 * width, 3, padding=1), nn.GELU(),
            nn.Conv2d(2 * width, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, slices, 3, padding=1),
        )

    def _advance(self, hidden, t0, t1):
        if float((t1 - t0).detach().abs().item()) <= 1e-7:
            return hidden
        times = torch.stack((t0, t1)).to(device=hidden.device, dtype=hidden.dtype)
        return odeint(self.dynamics, hidden, times, method="rk4")[-1]

    def forward(self, context, context_time=None):
        frames = _unpack_context(context, self.volume_shape)
        batch, steps, slices, height, image_width = frames.shape
        if steps != self.num_frames:
            raise ValueError(f"expected {self.num_frames} context frames, got {steps}")
        if context_time is None:
            context_time = torch.linspace(0.0, 1.0, steps + 1, device=frames.device, dtype=frames.dtype)
            context_time = context_time.unsqueeze(0).expand(batch, -1)
        if context_time.shape != (batch, steps + 1):
            raise ValueError(f"context_time must have shape {(batch, steps + 1)}")

        encoded = self.encoder(frames.reshape(batch * steps, slices, height, image_width))
        encoded = encoded.reshape(batch, steps, -1)
        predicted_states = []
        for sample in range(batch):
            hidden = frames.new_zeros(1, self.cell.hidden_size)
            cell = torch.zeros_like(hidden)
            previous_time = context_time[sample, 0]
            for step in range(steps):
                current_time = context_time[sample, step]
                hidden = self._advance(hidden, previous_time, current_time)
                observed = frames[sample, step].abs().sum() > 0
                if bool(observed):
                    hidden, cell = self.cell(encoded[sample, step:step + 1], (hidden, cell))
                previous_time = current_time
            target_time = context_time[sample, steps]
            hidden = self._advance(hidden, previous_time, target_time)
            predicted_states.append(hidden)
        hidden = torch.cat(predicted_states, dim=0)

        base_h, base_w = self.base_size
        features = self.decoder_in(hidden).reshape(batch, -1, base_h, base_w)
        features = F.interpolate(features, size=(max(8, height // 4), max(8, image_width // 4)),
                                 mode="bilinear", align_corners=False)
        features = F.interpolate(features, size=(height, image_width), mode="bilinear", align_corners=False)
        prediction = self.decoder_mid(features)
        return torch.sigmoid(prediction).unsqueeze(1)
