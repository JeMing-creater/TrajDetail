from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from src.forecaster_input import unpack_context as _unpack_context


class _ConvLSTMCell2d(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, 3, padding=1)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Two-layer ConvLSTM predictor with depth represented as image channels."""

    def __init__(self, volume_shape, width=32, num_frames=7):
        super().__init__()
        self.volume_shape = tuple(volume_shape)
        self.num_frames = num_frames
        depth = self.volume_shape[0]
        self.cell1 = _ConvLSTMCell2d(depth, width)
        self.cell2 = _ConvLSTMCell2d(width, width)
        self.head = nn.Conv2d(width, depth, 1)

    def forward(self, context, context_time=None):
        frames = _unpack_context(context, self.volume_shape)
        if frames.shape[1] != self.num_frames:
            raise ValueError(f"expected {self.num_frames} context frames, got {frames.shape[1]}")
        batch, _, depth, height, width = frames.shape
        h1 = frames.new_zeros(batch, self.cell1.hidden_channels, height, width)
        c1 = torch.zeros_like(h1)
        h2 = frames.new_zeros(batch, self.cell2.hidden_channels, height, width)
        c2 = torch.zeros_like(h2)
        for frame in frames.unbind(dim=1):
            h1, c1 = self.cell1(frame, (h1, c1))
            h2, c2 = self.cell2(h1, (h2, c2))
        return torch.sigmoid(self.head(h2)).reshape(batch, 1, depth, height, width)
