"""History-calibrated trajectory with frozen-DINO feature adapters."""

import torch
from torch import nn
import torch.nn.functional as F

from methods.trajdetail.unified_model import HistoryCalibratedTrajectory, _identity_grid, warp_volume


def local_feature_displacement(latest, previous, *, radius=1, temperature=0.07):
    """Estimate latest-minus-previous displacement by local cosine matching."""
    if latest.shape != previous.shape or latest.ndim != 5:
        raise ValueError("feature maps must share [B,C,D,H,W] shape")
    if radius < 0 or temperature <= 0:
        raise ValueError("radius must be nonnegative and temperature positive")
    latest = F.normalize(latest.float(), dim=1, eps=1e-6)
    previous = F.normalize(previous.float(), dim=1, eps=1e-6)
    batch, _channels, depth, height, width = latest.shape
    padded = F.pad(previous, (radius,) * 6, mode="replicate")
    offsets, scores = [], []
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                candidate = padded[:, :, radius + dz:radius + dz + depth,
                    radius + dy:radius + dy + height,
                    radius + dx:radius + dx + width]
                scores.append((latest * candidate).sum(dim=1))
                offsets.append((-dx, -dy, -dz))
    weights = torch.softmax(torch.stack(scores, dim=1) / temperature, dim=1)
    offset_tensor = latest.new_tensor(offsets).view(1, -1, 3, 1, 1, 1)
    flow = (weights[:, :, None] * offset_tensor).sum(dim=1)
    confidence = weights.max(dim=1).values[:, None]
    return flow, confidence


class DINOHistoryCalibratedTrajectory(HistoryCalibratedTrajectory):
    def __init__(self, base_channels: int = 16, memory_channels: int = 8,
                 max_step: float = 1 / 11, dino_channels: int = 768):
        super().__init__(base_channels=base_channels, memory_channels=memory_channels,
                         max_step=max_step, dino_channels=dino_channels)


class DINOInnovationTrajectory(DINOHistoryCalibratedTrajectory):
    """Assimilate matched MRI and DINO prediction errors into patient memory."""

    def __init__(self, base_channels=16, memory_channels=8, max_step=1 / 11,
                 dino_channels=768, innovation_channels=32):
        super().__init__(base_channels=base_channels, memory_channels=memory_channels,
                         max_step=max_step, dino_channels=dino_channels)
        self.semantic_predictor = nn.Sequential(
            nn.Conv3d(base_channels + memory_channels + 1, innovation_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(innovation_channels, base_channels, 1),
        )
        self.semantic_gate = nn.Conv3d(2 * base_channels, base_channels, 1)
        self.semantic_correction = nn.Sequential(
            nn.Conv3d(memory_channels + 3 * base_channels + 1,
                      base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels, memory_channels, 3, padding=1),
        )
        nn.init.zeros_(self.semantic_correction[-1].weight)
        nn.init.zeros_(self.semantic_correction[-1].bias)

    def _update_memory(self, memory, observed_features, observed_image_features,
                       predicted_image_features, observed_dino, delta_time):
        if observed_dino is None:
            raise ValueError("DINO innovation updates require observed DINO features")
        dt_plane = self._plane(memory, delta_time)
        projected_observation = self.dino_projection(observed_dino.float())
        projected_observation = F.interpolate(projected_observation,
            size=memory.shape[2:], mode="trilinear", align_corners=False)
        predicted_semantic = self.semantic_predictor(torch.cat((
            predicted_image_features, memory, dt_plane), dim=1))
        image_innovation = observed_image_features - predicted_image_features
        semantic_innovation = projected_observation - predicted_semantic
        reliability = torch.sigmoid(self.semantic_gate(torch.cat((
            image_innovation.abs(), semantic_innovation.abs()), dim=1)))
        update = self.semantic_correction(torch.cat((memory, observed_image_features,
            image_innovation, reliability * semantic_innovation, dt_plane), dim=1))
        auxiliary = F.smooth_l1_loss(predicted_semantic, projected_observation.detach())
        return update, auxiliary


class DINODeformationRefiner(DINOHistoryCalibratedTrajectory):
    """Refine V1 forecasts with a DINO-conditioned displacement and residual."""

    def __init__(self, base_channels: int = 16, memory_channels: int = 8,
                 max_step: float = 1 / 11, dino_channels: int = 768,
                 refiner_channels: int = 32, max_displacement: float = 3.0,
                 appearance_residual: bool = True):
        super().__init__(base_channels=base_channels, memory_channels=memory_channels,
                         max_step=max_step, dino_channels=dino_channels)
        self.max_displacement = float(max_displacement)
        self.appearance_residual = bool(appearance_residual)
        self.refiner_dino = nn.Conv3d(dino_channels, refiner_channels, 1)
        self.refiner = nn.Sequential(
            nn.Conv3d(refiner_channels + 3, refiner_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(refiner_channels, refiner_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(refiner_channels, 4 if self.appearance_residual else 3, 1),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def forward(self, context, mask, context_times, target_times, context_dino=None,
                base_result=None):
        result = (base_result if base_result is not None else
                  super().forward(context, mask, context_times, target_times,
                                  context_dino=context_dino))
        batch, slots = mask.shape
        latest = torch.arange(slots, device=mask.device).expand(batch, -1)
        latest = latest.masked_fill(mask <= 0, -1).max(dim=1).values
        rows = torch.arange(batch, device=mask.device)
        latest_dino = context_dino[rows, latest].float()
        latest_time = context_times[rows, latest].float()
        prediction = result["predictions"]
        batch_size, queries, depth, height, width = prediction.shape
        grid_shape = (1, depth, height, width, 3)
        if self._sampling_grid.shape != grid_shape or self._sampling_grid.device != prediction.device:
            self._sampling_grid = _identity_grid(prediction[:, :1])
        coarse = latest_dino.shape[-3:]

        dino_features = self.refiner_dino(latest_dino)
        dino_features = F.interpolate(dino_features, size=coarse, mode="trilinear",
                                       align_corners=False)
        anchor = result["last_context"].unsqueeze(1).expand(-1, queries, -1, -1, -1)
        image_features = torch.stack((prediction, prediction - anchor), dim=2)
        image_features = image_features.reshape(batch_size * queries, 2, depth, height, width)
        image_features = F.interpolate(image_features, size=coarse, mode="trilinear",
                                       align_corners=False)
        dino_features = dino_features[:, None].expand(-1, queries, -1, -1, -1, -1)
        dino_features = dino_features.reshape(batch_size * queries, -1, *coarse)
        delta = (target_times.float() - latest_time[:, None]).reshape(batch_size * queries, 1, 1, 1, 1)
        delta = delta.expand(-1, -1, *coarse)
        raw = self.refiner(torch.cat((dino_features, image_features, delta), dim=1))

        displacement = torch.tanh(raw[:, :3]) * self.max_displacement
        residual = raw[:, 3:4] if self.appearance_residual else torch.zeros_like(raw[:, :1])
        full_shape = (depth, height, width)
        displacement = F.interpolate(displacement, size=full_shape, mode="trilinear",
                                     align_corners=False)
        residual = F.interpolate(residual, size=full_shape, mode="trilinear",
                                 align_corners=False)
        flat_prediction = prediction.reshape(batch_size * queries, 1, depth, height, width)
        moved = warp_volume(flat_prediction, displacement, self._sampling_grid)
        identity = warp_volume(flat_prediction, torch.zeros_like(displacement),
                               self._sampling_grid)
        deformation_correction = moved - identity
        refined = flat_prediction + deformation_correction + residual
        result["unrefined_predictions"] = prediction
        result["predictions"] = refined.reshape(batch_size, queries, depth, height, width)
        result["deformation_correction"] = deformation_correction.reshape(
            batch_size, queries, depth, height, width)
        result["predicted_displacement"] = displacement.reshape(
            batch_size, queries, 3, depth, height, width)
        result["predicted_residual"] = residual.reshape(
            batch_size, queries, depth, height, width)
        return result


class DINOConfidenceGatedDeformationRefiner(DINODeformationRefiner):
    """Apply the learned deformation only where anatomy-conditioned confidence is high."""

    def __init__(self, base_channels=16, memory_channels=8, max_step=1 / 11,
                 dino_channels=768, refiner_channels=32, max_displacement=3.0):
        super().__init__(base_channels=base_channels, memory_channels=memory_channels,
            max_step=max_step, dino_channels=dino_channels,
            refiner_channels=refiner_channels, max_displacement=max_displacement,
            appearance_residual=False)
        self.gate_dino = nn.Conv3d(dino_channels, refiner_channels, 1)
        self.gate_head = nn.Sequential(
            nn.Conv3d(refiner_channels + 4, refiner_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(refiner_channels, refiner_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(refiner_channels, 1, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, 3.0)

    def forward(self, context, mask, context_times, target_times, context_dino=None,
                base_result=None, disable_confidence_gate=False):
        result = super().forward(context, mask, context_times, target_times,
            context_dino=context_dino, base_result=base_result)
        prediction = result["unrefined_predictions"]
        raw_correction = result["deformation_correction"]
        batch, queries, depth, height, width = prediction.shape
        latest = torch.arange(mask.shape[1], device=mask.device).expand(batch, -1)
        latest = latest.masked_fill(mask <= 0, -1).max(dim=1).values
        rows = torch.arange(batch, device=mask.device)
        latest_dino = context_dino[rows, latest].float()
        latest_time = context_times[rows, latest].float()
        coarse = latest_dino.shape[-3:]
        anatomy = F.interpolate(self.gate_dino(latest_dino), size=coarse,
            mode="trilinear", align_corners=False)
        image_inputs = torch.stack((prediction, raw_correction, raw_correction.abs()), dim=2)
        image_inputs = image_inputs.reshape(batch * queries, 3, depth, height, width)
        image_inputs = F.interpolate(image_inputs, size=coarse, mode="trilinear",
            align_corners=False)
        anatomy = anatomy[:, None].expand(-1, queries, -1, -1, -1, -1).reshape(
            batch * queries, -1, *coarse)
        horizon = target_times.float() - latest_time[:, None]
        time_plane = horizon.reshape(batch * queries, 1, 1, 1, 1).expand(-1, -1, *coarse)
        gate = torch.sigmoid(self.gate_head(torch.cat((anatomy, image_inputs, time_plane), dim=1)))
        gate = F.interpolate(gate, size=(depth, height, width), mode="trilinear",
            align_corners=False).reshape(batch, queries, depth, height, width)
        if disable_confidence_gate:
            gate = torch.ones_like(gate)
        result["raw_deformation_correction"] = raw_correction
        result["deformation_gate"] = gate
        result["deformation_correction"] = gate * raw_correction
        result["predictions"] = prediction + result["deformation_correction"]
        return result


class DINOTimeAwareDeformationRefiner(DINOHistoryCalibratedTrajectory):
    """Forecast temporal displacement from the two latest valid observations."""

    def __init__(self, base_channels: int = 16, memory_channels: int = 8,
                 max_step: float = 1 / 11, dino_channels: int = 768,
                 refiner_channels: int = 32, max_displacement: float = 3.0):
        super().__init__(base_channels=base_channels, memory_channels=memory_channels,
                         max_step=max_step, dino_channels=dino_channels)
        self.max_displacement = float(max_displacement)
        self.static_dino = nn.Conv3d(dino_channels, refiner_channels, 1)
        self.motion_dino = nn.Conv3d(dino_channels, refiner_channels, 1)
        self.temporal_flow_head = nn.Sequential(
            nn.Conv3d(2 * refiner_channels + 5, refiner_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(refiner_channels, refiner_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(refiner_channels, 3, 1),
        )
        nn.init.zeros_(self.temporal_flow_head[-1].weight)
        nn.init.zeros_(self.temporal_flow_head[-1].bias)

    @staticmethod
    def _latest_pair(mask):
        batch, slots = mask.shape
        indices = torch.arange(slots, device=mask.device).expand(batch, -1)
        latest = indices.masked_fill(mask <= 0, -1).max(dim=1).values
        previous = indices.masked_fill((mask <= 0) | (indices >= latest[:, None]), -1)
        previous = previous.max(dim=1).values
        has_pair = previous >= 0
        previous = torch.where(has_pair, previous, latest)
        return latest, previous, has_pair

    def temporal_features(self, context_dino, mask, context_times, target_times):
        latest_idx, previous_idx, has_pair = self._latest_pair(mask)
        rows = torch.arange(mask.shape[0], device=mask.device)
        latest = context_dino[rows, latest_idx].float()
        previous = context_dino[rows, previous_idx].float()
        latest_time = context_times[rows, latest_idx].float()
        previous_time = context_times[rows, previous_idx].float()
        delta = (latest_time - previous_time).clamp_min(1e-6)
        velocity = (latest - previous) / delta[:, None, None, None, None]
        velocity = velocity * has_pair[:, None, None, None, None]
        horizon = target_times.float() - latest_time[:, None]
        return latest, velocity, horizon

    def forward(self, context, mask, context_times, target_times, context_dino=None,
                disable_temporal_motion=False):
        result = super().forward(context, mask, context_times, target_times,
                                 context_dino=context_dino)
        batch, queries, depth, height, width = result["predictions"].shape
        latest_idx, previous_idx, has_pair = self._latest_pair(mask)
        rows = torch.arange(mask.shape[0], device=mask.device)
        latest_dino, velocity_dino, horizon = self.temporal_features(
            context_dino, mask, context_times, target_times)
        latest_image = context[rows, latest_idx].float()
        previous_image = context[rows, previous_idx].float()
        image_velocity = (latest_image - previous_image) / (
            context_times[rows, latest_idx] - context_times[rows, previous_idx]
        ).clamp_min(1e-6)[:, None, None, None]
        image_velocity = image_velocity * has_pair[:, None, None, None]
        image_change = latest_image - previous_image
        if disable_temporal_motion:
            image_velocity = torch.zeros_like(image_velocity)
            image_change = torch.zeros_like(image_change)

        coarse = latest_dino.shape[-3:]
        static = F.interpolate(self.static_dino(latest_dino), size=coarse,
                               mode="trilinear", align_corners=False)
        motion = F.interpolate(self.motion_dino(velocity_dino), size=coarse,
                               mode="trilinear", align_corners=False)
        if disable_temporal_motion:
            motion = torch.zeros_like(motion)
        prediction = result["predictions"]
        anchor = result["last_context"]
        images = torch.stack((prediction,
            prediction - anchor[:, None],
            image_change[:, None].expand(-1, queries, -1, -1, -1),
            image_velocity[:, None].expand(-1, queries, -1, -1, -1)), dim=2)
        images = images.reshape(batch * queries, 4, depth, height, width)
        images = F.interpolate(images, size=coarse, mode="trilinear", align_corners=False)
        static = static[:, None].expand(-1, queries, -1, -1, -1, -1)
        motion = motion[:, None].expand(-1, queries, -1, -1, -1, -1)
        static = static.reshape(batch * queries, -1, *coarse)
        motion = motion.reshape(batch * queries, -1, *coarse)
        time_plane = horizon.reshape(batch * queries, 1, 1, 1, 1).expand(-1, -1, *coarse)
        raw_flow = self.temporal_flow_head(torch.cat((static, motion, images, time_plane), dim=1))
        flow = torch.tanh(raw_flow) * self.max_displacement
        flow = F.interpolate(flow, size=(depth, height, width), mode="trilinear",
                             align_corners=False)
        flat_prediction = prediction.reshape(batch * queries, 1, depth, height, width)
        moved = warp_volume(flat_prediction, flow, self._sampling_grid)
        identity = warp_volume(flat_prediction, torch.zeros_like(flow), self._sampling_grid)
        correction = moved - identity
        result["unrefined_predictions"] = prediction
        result["predictions"] = (flat_prediction + correction).reshape(
            batch, queries, depth, height, width)
        result["deformation_correction"] = correction.reshape(
            batch, queries, depth, height, width)
        result["predicted_displacement"] = flow.reshape(
            batch, queries, 3, depth, height, width)
        result["predicted_residual"] = torch.zeros_like(prediction)
        result["temporal_motion_features"] = velocity_dino
        return result


class DINOCorrespondenceDeformationRefiner(DINOHistoryCalibratedTrajectory):
    """Extrapolate local DINO correspondences, with a learned flow correction."""

    def __init__(self, base_channels=16, memory_channels=8, max_step=1 / 11,
                 dino_channels=768, refiner_channels=32, match_channels=64,
                 max_displacement=3.0, match_radius=1, match_temperature=0.07):
        super().__init__(base_channels=base_channels, memory_channels=memory_channels,
                         max_step=max_step, dino_channels=dino_channels)
        self.max_displacement = float(max_displacement)
        self.match_radius = int(match_radius)
        self.match_temperature = float(match_temperature)
        self.match_projection = nn.Conv3d(dino_channels, match_channels, 1, bias=False)
        self.static_projection = nn.Conv3d(dino_channels, refiner_channels, 1)
        self.correspondence_gate = nn.Parameter(torch.zeros(()))
        self.flow_correction = nn.Sequential(
            nn.Conv3d(refiner_channels + 7, refiner_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(refiner_channels, refiner_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(refiner_channels, 3, 1),
        )
        nn.init.zeros_(self.flow_correction[-1].weight)
        nn.init.zeros_(self.flow_correction[-1].bias)

    def forward(self, context, mask, context_times, target_times, context_dino=None,
                disable_correspondence=False):
        result = super().forward(context, mask, context_times, target_times,
                                 context_dino=context_dino)
        batch, queries, depth, height, width = result["predictions"].shape
        latest_idx, previous_idx, has_pair = DINOTimeAwareDeformationRefiner._latest_pair(mask)
        rows = torch.arange(batch, device=mask.device)
        latest_dino = context_dino[rows, latest_idx].float()
        previous_dino = context_dino[rows, previous_idx].float()
        dt = (context_times[rows, latest_idx] - context_times[rows, previous_idx]).float().clamp_min(1e-6)
        latest_match = self.match_projection(latest_dino)
        previous_match = self.match_projection(previous_dino)
        observed_flow, confidence = local_feature_displacement(latest_match,
            previous_match, radius=self.match_radius, temperature=self.match_temperature)
        observed_flow = observed_flow * has_pair[:, None, None, None, None]
        confidence = confidence * has_pair[:, None, None, None, None]
        coarse = latest_dino.shape[-3:]
        voxel_scale = observed_flow.new_tensor([
            (width - 1) / max(coarse[2] - 1, 1),
            (height - 1) / max(coarse[1] - 1, 1),
            (depth - 1) / max(coarse[0] - 1, 1),
        ]).view(1, 3, 1, 1, 1)
        displacement_rate = observed_flow * voxel_scale / dt[:, None, None, None, None]
        static = F.interpolate(self.static_projection(latest_dino), size=coarse,
                               mode="trilinear", align_corners=False)
        confidence = confidence.to(static.dtype)
        prediction = result["predictions"]
        anchor = result["last_context"]
        images = torch.stack((prediction, prediction - anchor[:, None]), dim=2)
        images = images.reshape(batch * queries, 2, depth, height, width)
        images = F.interpolate(images, size=coarse, mode="trilinear", align_corners=False)
        static = static[:, None].expand(-1, queries, -1, -1, -1, -1).reshape(
            batch * queries, -1, *coarse)
        prior = displacement_rate[:, None] * (target_times.float() -
            context_times[rows, latest_idx, None].float())[:, :, None, None, None, None]
        prior = prior.expand(-1, queries, -1, -1, -1, -1)
        conf = confidence[:, None].expand(-1, queries, -1, -1, -1, -1)
        if disable_correspondence:
            prior = torch.zeros_like(prior)
            conf = torch.zeros_like(conf)
        prior_coarse = F.interpolate(prior.reshape(batch * queries, 3, *coarse),
            size=coarse, mode="trilinear", align_corners=False)
        prior_input = torch.tanh(prior_coarse / self.max_displacement)
        conf = conf.reshape(batch * queries, 1, *coarse)
        horizon = (target_times.float() - context_times[rows, latest_idx, None].float())
        time_plane = horizon.reshape(batch * queries, 1, 1, 1, 1).expand(-1, -1, *coarse)
        raw = self.flow_correction(torch.cat((static, prior_input, conf, images, time_plane), dim=1))
        correction = torch.tanh(raw) * self.max_displacement
        correction = F.interpolate(correction, size=(depth, height, width),
                                   mode="trilinear", align_corners=False)
        prior_full = F.interpolate(prior.reshape(batch * queries, 3, *coarse),
            size=(depth, height, width), mode="trilinear", align_corners=False)
        flow = self.max_displacement * torch.tanh(
            (self.correspondence_gate * prior_full + correction) / self.max_displacement)
        base = prediction.reshape(batch * queries, 1, depth, height, width)
        moved = warp_volume(base, flow, self._sampling_grid)
        identity = warp_volume(base, torch.zeros_like(flow), self._sampling_grid)
        result["unrefined_predictions"] = prediction
        result["predictions"] = (base + (moved - identity)).reshape(
            batch, queries, depth, height, width)
        result["deformation_correction"] = (moved - identity).reshape(
            batch, queries, depth, height, width)
        result["predicted_displacement"] = flow.reshape(
            batch, queries, 3, depth, height, width)
        result["predicted_residual"] = torch.zeros_like(prediction)
        result["predicted_correspondence_flow"] = prior.reshape(
            batch, queries, 3, *coarse)
        result["correspondence_confidence"] = conf.reshape(
            batch, queries, 1, *coarse)
        return result


__all__ = ["DINOHistoryCalibratedTrajectory", "DINOInnovationTrajectory", "DINODeformationRefiner",
           "DINOConfidenceGatedDeformationRefiner",
           "DINOTimeAwareDeformationRefiner", "DINOCorrespondenceDeformationRefiner",
           "local_feature_displacement"]
