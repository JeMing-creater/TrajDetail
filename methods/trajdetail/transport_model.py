"""DINO-conditioned unbalanced feature transport for ACDC cine forecasting."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from methods.trajdetail.unified_model import warp_volume


def unbalanced_sinkhorn(cost: Tensor, source_mass: Tensor, target_mass: Tensor,
                        epsilon: float = 0.08, rho: float = 0.12,
                        iterations: int = 12) -> Tensor:
    """Entropic unbalanced OT plan with relaxed source and target marginals."""
    if cost.ndim != 3 or source_mass.shape != cost.shape[:1] + cost.shape[1:2]:
        if source_mass.shape != (cost.shape[0], cost.shape[1]):
            raise ValueError("source_mass must have shape [B,N]")
    if target_mass.shape != (cost.shape[0], cost.shape[2]):
        raise ValueError("target_mass must have shape [B,M]")
    log_kernel = -cost.float() / epsilon
    log_a = source_mass.float().clamp_min(1e-8).log()
    log_b = target_mass.float().clamp_min(1e-8).log()
    tau = rho / (rho + epsilon)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = tau * (log_a - torch.logsumexp(log_kernel + log_v[:, None, :], dim=2))
        log_v = tau * (log_b - torch.logsumexp(log_kernel + log_u[:, :, None], dim=1))
    return torch.exp(log_kernel + log_u[:, :, None] + log_v[:, None, :])


class DINOUnbalancedTransport(nn.Module):
    """Predicts future DINO tokens, transports anatomy, and synthesizes residual signal."""

    def __init__(self, dino_channels: int = 768, latent_channels: int = 48,
                 grid_size: int = 8, time_scale: float = 1.0):
        super().__init__()
        self.grid_size = int(grid_size)
        self.time_scale = float(time_scale)
        self.project = nn.Conv3d(dino_channels, latent_channels, 1, bias=False)
        self.future = nn.Sequential(
            nn.Conv3d(latent_channels + 2, latent_channels, 3, padding=1), nn.SiLU(),
            nn.Conv3d(latent_channels, latent_channels, 3, padding=1), nn.SiLU(),
            nn.Conv3d(latent_channels, latent_channels, 3, padding=1))
        self.mass = nn.Conv3d(latent_channels, 1, 1)
        self.residual = nn.Sequential(
            nn.Conv3d(latent_channels * 2 + 2, 32, 3, padding=1), nn.SiLU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv3d(32, 1, 3, padding=1))
        nn.init.zeros_(self.future[-1].weight)
        nn.init.zeros_(self.future[-1].bias)
        nn.init.zeros_(self.mass.weight)
        nn.init.zeros_(self.mass.bias)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, image: Tensor, dino: Tensor, delta: Tensor) -> dict[str, Tensor]:
        if image.ndim != 5 or dino.ndim != 5:
            raise ValueError("image and dino must be [B,C,D,H,W]")
        batch = image.shape[0]
        size = (self.grid_size,) * 3
        with torch.autocast(device_type=image.device.type, enabled=False):
            source_tokens = F.adaptive_avg_pool3d(dino.float(), size)
            source = self.project(source_tokens)
            image_low = F.interpolate(image.float(), size=size, mode="trilinear", align_corners=False)
            time = delta.float().reshape(batch, 1, 1, 1, 1).expand(-1, 1, *size) / self.time_scale
            predicted = source + self.future(torch.cat((source, image_low, time), dim=1))
            src_flat = F.normalize(source.flatten(2).transpose(1, 2), dim=-1)
            dst_flat = F.normalize(predicted.flatten(2).transpose(1, 2), dim=-1)
            feature_cost = 1 - torch.bmm(src_flat, dst_flat.transpose(1, 2)).clamp(-1, 1)
            coords = torch.stack(torch.meshgrid(*[
                torch.linspace(-1, 1, n, device=image.device) for n in size
            ], indexing="ij"), dim=-1)[..., (2, 1, 0)].reshape(-1, 3)
            spatial_cost = torch.cdist(coords[None], coords[None]).square()
            cost = feature_cost + 0.25 * spatial_cost
            source_mass = torch.softmax(self.mass(source).flatten(1), dim=1)
            target_mass = torch.softmax(self.mass(predicted).flatten(1), dim=1)
            plan = unbalanced_sinkhorn(cost, source_mass, target_mass)
            transported_mass = plan.sum(2)
            barycenter = torch.bmm(plan, coords[None].expand(batch, -1, -1))
            barycenter = barycenter / transported_mass.unsqueeze(-1).clamp_min(1e-8)
            delta_xyz = barycenter - coords[None]
            unmatched = (1 - transported_mass / source_mass.clamp_min(1e-8)).clamp(0, 1)
            flow_low = delta_xyz.reshape(batch, *size, 3).permute(0, 4, 1, 2, 3)
            flow_full = F.interpolate(flow_low, size=image.shape[2:], mode="trilinear", align_corners=False)
            scale = image.new_tensor([(image.shape[-1] - 1) / 2,
                                      (image.shape[-2] - 1) / 2,
                                      (image.shape[-3] - 1) / 2]).view(1, 3, 1, 1, 1)
            warped = warp_volume(image.float(), flow_full * scale)
            unmatched_low = unmatched.reshape(batch, 1, *size)
            dec_in = torch.cat((source, predicted, image_low, unmatched_low), dim=1)
            residual_low = self.residual(dec_in)
            residual_full = F.interpolate(residual_low, size=image.shape[2:], mode="trilinear", align_corners=False)
            prediction = warped + residual_full
        return {"prediction": prediction, "warped": warped, "residual": residual_full,
                "future_tokens": predicted, "source_tokens": source,
                "source_mass": source_mass, "target_mass": target_mass,
                "transport_plan": plan, "unmatched": unmatched_low, "flow": flow_full}


__all__ = ["DINOUnbalancedTransport", "unbalanced_sinkhorn"]
