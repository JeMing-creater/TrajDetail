from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from src.forecaster_input import unpack_context as _unpack_context


class ViViTForecaster(nn.Module):
    """Factorized video-transformer adaptation with a learned future query."""

    def __init__(self, volume_shape, width=32, num_frames=7, patch_size=8, depth=2, num_heads=4):
        super().__init__()
        self.volume_shape = tuple(volume_shape)
        self.num_frames = num_frames
        self.patch_size = patch_size
        slices, height, image_width = self.volume_shape
        if height % patch_size or image_width % patch_size:
            raise ValueError("ViViT patch_size must divide the two in-plane dimensions")
        dim = max(32, width * 4)
        grid_h, grid_w = height // patch_size, image_width // patch_size
        self.grid_h, self.grid_w = grid_h, grid_w
        n_patches = grid_h * grid_w
        self.patch_embed = nn.Conv2d(slices, dim, patch_size, stride=patch_size)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, n_patches, dim))
        self.time_pos = nn.Parameter(torch.zeros(1, num_frames + 1, dim))
        self.future_query = nn.Parameter(torch.zeros(1, 1, 1, dim))
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=4 * dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=4 * dim,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=depth)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=depth)
        self.patch_head = nn.Linear(dim, slices * patch_size * patch_size)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.future_query, std=0.02)

    def forward(self, context, context_time=None):
        frames = _unpack_context(context, self.volume_shape)
        batch, steps, slices, height, image_width = frames.shape
        if steps != self.num_frames:
            raise ValueError(f"expected {self.num_frames} context frames, got {steps}")
        patches = self.patch_embed(frames.reshape(batch * steps, slices, height, image_width))
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch, steps, self.grid_h * self.grid_w, -1)
        tokens = tokens + self.spatial_pos
        tokens = self.spatial_encoder(tokens.reshape(batch * steps, self.grid_h * self.grid_w, -1))
        tokens = tokens.reshape(batch, steps, self.grid_h * self.grid_w, -1).transpose(1, 2)

        query = self.future_query.expand(batch, self.grid_h * self.grid_w, 1, -1)
        temporal = torch.cat((tokens, query), dim=2)
        temporal = temporal + self.time_pos[:, None, :steps + 1]
        temporal = self.temporal_encoder(temporal.reshape(batch * self.grid_h * self.grid_w, steps + 1, -1))
        future_tokens = temporal[:, -1].reshape(batch, self.grid_h * self.grid_w, -1)

        patch_values = self.patch_head(future_tokens)
        patch_values = patch_values.reshape(
            batch, self.grid_h, self.grid_w, slices, self.patch_size, self.patch_size,
        )
        prediction = patch_values.permute(0, 3, 1, 4, 2, 5).reshape(batch, slices, height, image_width)
        return torch.sigmoid(prediction).unsqueeze(1)
