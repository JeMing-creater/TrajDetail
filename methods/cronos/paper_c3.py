"""Paper-narrative CRONOS C.3 3D U-Net reconstruction.

The appendix's single-channel velocity head conflicts with the T-state flow
equations. This implementation predicts one shared velocity per image channel
and broadcasts it over the context states, matching the shared-velocity
description while keeping the state dimension used by Algorithm 2.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dimension: int, max_period: int = 10000) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    phase = timesteps.reshape(-1, 1).float() * frequencies.reshape(1, -1)
    embedding = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
    if dimension % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class FiLMResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dimension: int):
        super().__init__()
        if out_channels % 8:
            raise ValueError("C.3 GroupNorm(8) requires feature widths divisible by 8")
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.film = nn.Linear(time_dimension, 2 * out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        scale, shift = self.film(time_embedding).chunk(2, dim=1)
        h = h + scale[:, :, None, None, None] * h + shift[:, :, None, None, None]
        h = self.conv2(F.silu(h))
        h = self.norm2(h)
        return F.silu(h + self.skip(x))


class WindowAttention3D(nn.Module):
    def __init__(self, channels: int, window_size: int = 8, heads: int = 4, head_dimension: int = 64):
        super().__init__()
        self.window_size = window_size
        self.attention_dimension = heads * head_dimension
        self.norm = nn.LayerNorm(self.attention_dimension)
        self.project_in = nn.Linear(channels, self.attention_dimension)
        self.attention = nn.MultiheadAttention(
            self.attention_dimension, num_heads=heads, batch_first=True
        )
        self.project_out = nn.Linear(self.attention_dimension, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        window = self.window_size
        pad_d = (-depth) % window
        pad_h = (-height) % window
        pad_w = (-width) % window
        padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        valid = F.pad(
            x.new_ones((batch, 1, depth, height, width)),
            (0, pad_w, 0, pad_h, 0, pad_d),
        )
        padded_depth, padded_height, padded_width = padded.shape[-3:]
        n_depth, n_height, n_width = (
            padded_depth // window,
            padded_height // window,
            padded_width // window,
        )

        windows = padded.permute(0, 2, 3, 4, 1).reshape(
            batch, n_depth, window, n_height, window, n_width, window, channels
        )
        windows = windows.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
            batch * n_depth * n_height * n_width, window**3, channels
        )
        valid_windows = valid.permute(0, 2, 3, 4, 1).reshape(
            batch, n_depth, window, n_height, window, n_width, window, 1
        )
        valid_windows = valid_windows.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
            batch * n_depth * n_height * n_width, window**3
        )

        tokens = self.project_in(windows)
        normalized = self.norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid_windows.bool(),
            need_weights=False,
        )
        tokens = windows + self.project_out(attended)
        tokens = tokens.reshape(
            batch, n_depth, n_height, n_width, window, window, window, channels
        )
        output = tokens.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            batch, channels, padded_depth, padded_height, padded_width
        )
        return output[:, :, :depth, :height, :width]


class UpStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dimension: int):
        super().__init__()
        self.up_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.block1 = FiLMResBlock(out_channels + skip_channels, out_channels, time_dimension)
        self.block2 = FiLMResBlock(out_channels, out_channels, time_dimension)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, time_embedding)
        return self.block2(x, time_embedding)


class PaperC3UNet(nn.Module):
    """Four-scale 3D U-Net matching the paper's stated C.3 dimensions."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 1,
        base_channels: int = 32,
        channel_mult=(1, 1, 2, 4),
        attention_window: int = 8,
        attention_heads: int = 4,
        attention_head_dimension: int = 64,
    ):
        super().__init__()
        if len(channel_mult) != 4:
            raise ValueError("The paper C.3 backbone has four scales")
        if base_channels % 8 or any(base_channels * mult % 8 for mult in channel_mult):
            raise ValueError("All feature widths must be divisible by GroupNorm(8)")
        channels = [base_channels * mult for mult in channel_mult]
        time_dimension = base_channels * 4
        self.stem = nn.Conv3d(input_channels, base_channels, kernel_size=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dimension),
            nn.SiLU(),
            nn.Linear(time_dimension, time_dimension),
        )
        self.encoder = nn.ModuleList(
            [FiLMResBlock(base_channels, channels[0], time_dimension)]
            + [
                FiLMResBlock(channels[index], channels[index], time_dimension)
                for index in range(1, 4)
            ]
        )
        self.downsample = nn.ModuleList(
            [
                nn.Conv3d(channels[index], channels[index + 1], kernel_size=3, stride=2, padding=1)
                for index in range(3)
            ]
        )
        self.middle1 = FiLMResBlock(channels[-1], channels[-1], time_dimension)
        self.attention = WindowAttention3D(
            channels[-1], attention_window, attention_heads, attention_head_dimension
        )
        self.middle2 = FiLMResBlock(channels[-1], channels[-1], time_dimension)
        self.decoder = nn.ModuleList(
            [
                UpStage(channels[3], channels[2], channels[2], time_dimension),
                UpStage(channels[2], channels[1], channels[1], time_dimension),
                UpStage(channels[1], channels[0], channels[0], time_dimension),
            ]
        )
        self.head = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels, output_channels, kernel_size=1),
        )
    def forward(self, x: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected (B,C,D,H,W), got {tuple(x.shape)}")
        if x.shape[1] != self.stem.in_channels:
            raise ValueError(f"Expected {self.stem.in_channels} input channels, got {x.shape[1]}")
        raw_time_embedding = timestep_embedding(flow_time.reshape(-1), self.time_mlp[0].in_features)
        if flow_time.ndim == 1:
            raw_time_embedding = raw_time_embedding.reshape(x.shape[0], -1)
        elif flow_time.ndim == 2:
            raw_time_embedding = raw_time_embedding.reshape(x.shape[0], flow_time.shape[1], -1).mean(dim=1)
        else:
            raise ValueError(f"Expected scalar or per-context times, got {tuple(flow_time.shape)}")
        time_embedding = self.time_mlp(raw_time_embedding)
        h = self.stem(x)
        skips = []
        for index, block in enumerate(self.encoder):
            h = block(h, time_embedding)
            skips.append(h)
            if index < len(self.downsample):
                h = self.downsample[index](h)
        h = self.middle1(h, time_embedding)
        h = self.attention(h)
        h = self.middle2(h, time_embedding)
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            h = block(h, skip, time_embedding)
        return self.head(h)


def broadcast_shared_velocity(velocity: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Broadcast a C-channel velocity to a (B,T,C,D,H,W) flow state."""
    if states.ndim != 6 or velocity.ndim != 5:
        raise ValueError("Expected velocity (B,C,D,H,W) and state (B,T,C,D,H,W)")
    if velocity.shape[0] != states.shape[0] or velocity.shape[1:] != states.shape[2:]:
        raise ValueError("Velocity channels and spatial shape must match the flow state")
    return velocity.unsqueeze(1).expand_as(states)
