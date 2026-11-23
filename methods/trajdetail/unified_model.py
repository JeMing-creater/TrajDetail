"""History-calibrated, deterministic whole-volume trajectory (framework V1).

The image state stays at acquisition resolution; spatial memory and the shared
3-D vector field use an encoder pyramid ending at roughly D/2, H/4, W/4.
Observed images correct memory only after their causal prior is recorded.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _identity_grid(image: Tensor) -> Tensor:
    axes = [torch.linspace(-1, 1, n, device=image.device, dtype=torch.float32)
            if n > 1 else torch.zeros(1, device=image.device) for n in image.shape[2:]]
    zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)


def warp_volume(image: Tensor, displacement: Tensor, base_grid: Tensor | None = None) -> Tensor:
    """Backward-sample ``[N,C,D,H,W]`` with x/y/z displacements in voxels.

    Positive x velocity moves image content toward increasing W: sample at
    x-displacement. Border replication defines the field of view boundary.
    Float32 sampling avoids autocast grid/image dtype mismatch. Repeated
    trilinear warps can smooth textures; this numerical effect is not diffusion
    learned by the model and must be checked when interpreting long rollouts.
    """
    if image.ndim != 5 or displacement.shape != (image.shape[0], 3, *image.shape[2:]):
        raise ValueError("Expected image [N,C,D,H,W] and displacement [N,3,D,H,W]")
    with torch.autocast(device_type=image.device.type, enabled=False):
        base_grid = _identity_grid(image) if base_grid is None else base_grid.float()
        depth, height, width = image.shape[2:]
        scale = image.new_tensor([2 / (width - 1) if width > 1 else 0,
                                  2 / (height - 1) if height > 1 else 0,
                                  2 / (depth - 1) if depth > 1 else 0], dtype=torch.float32)
        grid = base_grid - displacement.float().movedim(1, -1) * scale
        return F.grid_sample(image.float(), grid, mode="bilinear",
                             padding_mode="border", align_corners=True)


class HistoryCalibratedTrajectory(nn.Module):
    """One shared dynamics operator for historical priors and future prediction.

    Inputs: context[B,T,D,H,W], binary mask[B,T], context_times[B,T], and
    target_times[B,K]. Valid observations must be in chronological slot order.
    Times use the caller's normalized data-time units. Queries may be unordered
    and must be at or after the last valid observation (the cutoff).

    A fixed step grid is anchored at the cutoff. Fractional queries read the
    vector field at their preceding fixed grid point, without changing that
    field or the states used by other queries. The step conditioning channel is
    always max_step, including fractional reads; target times never enter it.
    """

    def __init__(self, base_channels: int = 16, memory_channels: int = 8,
                 max_step: float = 1 / 11, dino_channels: int | None = None):
        super().__init__()
        if not math.isfinite(max_step) or not 0 < max_step <= 1 / 11:
            raise ValueError("max_step must be finite and in (0, 1/11]")
        if base_channels < 1 or memory_channels < 1:
            raise ValueError("Channel counts must be positive")
        self.max_step = float(max_step)
        self.base_channels = base_channels
        self.memory_channels = memory_channels
        self.dino_channels = dino_channels
        self.encoder = nn.Sequential(
            nn.Conv3d(1, base_channels, 3, stride=(1, 2, 2), padding=1), nn.SiLU(),
            nn.Conv3d(base_channels, base_channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(base_channels, base_channels, 3, padding=1), nn.SiLU(),
        )
        self.initialize_memory = nn.Conv3d(base_channels + 1, memory_channels, 1)
        self.correction = nn.Sequential(
            nn.Conv3d(memory_channels + 2 * base_channels + 1, base_channels, 3, padding=1),
            nn.SiLU(), nn.Conv3d(base_channels, memory_channels, 3, padding=1),
        )
        self.dynamics = nn.Sequential(
            nn.Conv3d(base_channels + memory_channels + 2, base_channels, 3, padding=1),
            nn.SiLU(), nn.Conv3d(base_channels, base_channels, 3, padding=1), nn.SiLU(),
        )
        self.velocity_head = nn.Conv3d(base_channels, 3, 1)
        self.source_head = nn.Conv3d(base_channels, 1, 1)
        self.memory_head = nn.Conv3d(base_channels, memory_channels, 1)
        # A linear, signed source can create or remove signal everywhere. It is
        # unconstrained and can explain changes also explained by transport;
        # neither branch is an identified physiological quantity. No clipping.
        for head in (self.velocity_head, self.source_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.register_buffer("_sampling_grid", torch.empty(0), persistent=False)
        # Initialize the optional adapter after every V1 layer so enabling DINO
        # does not change V1's seeded initialization.
        self.dino_projection = (nn.Conv3d(dino_channels, base_channels, 1)
                                if dino_channels is not None else None)
        self.dino_gate = (nn.Parameter(torch.zeros(1, base_channels, 1, 1, 1))
                          if dino_channels is not None else None)

    def _encode_observation(self, image: Tensor, dino: Tensor | None = None) -> Tensor:
        encoded = self.encoder(image)
        if self.dino_projection is None:
            if dino is not None:
                raise ValueError("This model was created without a DINO adapter")
            return encoded
        if dino is None or dino.ndim != 5 or dino.shape[0] != image.shape[0]:
            raise ValueError("DINO-enabled model expects features [B,C,D,H,W]")
        if dino.shape[1] != self.dino_channels:
            raise ValueError(f"Expected {self.dino_channels} DINO channels, got {dino.shape[1]}")
        projected = self.dino_projection(dino.float())
        projected = F.interpolate(projected, size=encoded.shape[2:], mode="trilinear",
                                  align_corners=False).to(encoded.dtype)
        return encoded + self.dino_gate.to(encoded.dtype) * projected

    @staticmethod
    def _plane(reference: Tensor, value: float) -> Tensor:
        return reference.new_full((reference.shape[0], 1, *reference.shape[2:]), value)

    def _vector_field(self, image: Tensor, memory: Tensor, now: float,
                      statistics: list[Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.encoder(image)
        features = self.dynamics(torch.cat((encoded, memory, self._plane(memory, now),
                                           self._plane(memory, self.max_step)), dim=1))
        velocity = self.velocity_head(features)
        source = self.source_head(features)
        derivative = self.memory_head(features)
        statistics.append(torch.stack([x.detach().float().square().mean()
                                       for x in (velocity, source, derivative)]))
        velocity = F.interpolate(velocity.float(), size=image.shape[2:], mode="trilinear", align_corners=False)
        source = F.interpolate(source.float(), size=image.shape[2:], mode="trilinear", align_corners=False)
        return velocity, source, derivative

    def _advance(self, image: Tensor, memory: Tensor, field: tuple[Tensor, Tensor, Tensor],
                 step: float) -> tuple[Tensor, Tensor]:
        if step == 0:
            return image, memory
        velocity, source, derivative = field
        image = warp_volume(image, step * velocity, self._sampling_grid) + step * source
        return image, memory + step * derivative

    def _update_memory(self, memory: Tensor, observed_features: Tensor,
                       observed_image_features: Tensor, predicted_image_features: Tensor,
                       observed_dino: Tensor | None, delta_time: float):
        error = observed_features - predicted_image_features
        update = self.correction(torch.cat((memory, observed_features, error,
            self._plane(memory, delta_time)), dim=1))
        return update, None

    def _trajectory(self, image: Tensor, memory: Tensor, start: float,
                    queries: list[float], statistics: list[Tensor], *, vector_field=None) -> list[tuple[Tensor, Tensor]]:
        evaluate_field = self._vector_field if vector_field is None else vector_field
        plans = []
        for query in queries:
            gap = query - start
            ratio = gap / self.max_step
            nearest = round(ratio)
            # Float32 data timestamps are not exact multiples of Python's
            # max_step. Snap grid endpoints to avoid an extra microscopic warp.
            if abs(ratio - nearest) <= 1e-5:
                full_steps, remainder = nearest, 0.0
            else:
                full_steps = int(math.floor(ratio))
                remainder = gap - full_steps * self.max_step
            plans.append((full_steps, remainder))
        # Full states and fields are shared. A fractional read never feeds back
        # into the canonical trajectory, even when another query is inserted.
        states = [(image, memory)]
        fields = {}
        for index in range(max(n for n, _ in plans)):
            fields[index] = evaluate_field(*states[-1], start + index * self.max_step, statistics)
            states.append(self._advance(*states[-1], fields[index], self.max_step))
        outputs = []
        for index, remainder in plans:
            if remainder == 0:
                outputs.append(states[index])
            else:
                if index not in fields:
                    fields[index] = evaluate_field(*states[index], start + index * self.max_step, statistics)
                outputs.append(self._advance(*states[index], fields[index], remainder))
        return outputs

    def _forecast(self, image, memory, start, queries, statistics, condition):
        return self._trajectory(image, memory, start, queries, statistics), {}

    def forward(self, context: Tensor, mask: Tensor, context_times: Tensor,
                target_times: Tensor, context_dino: Tensor | None = None, *,
                forecast_condition=None) -> dict[str, Tensor]:
        if context.ndim != 5 or any(n < 1 for n in context.shape):
            raise ValueError("context must have nonempty dimensions [B,T,D,H,W]")
        batch, slots = context.shape[:2]
        if mask.shape != (batch, slots) or context_times.shape != (batch, slots):
            raise ValueError("mask and context_times must have shape [B,T]")
        if target_times.ndim != 2 or target_times.shape[0] != batch or target_times.shape[1] < 1:
            raise ValueError("target_times must have nonempty shape [B,K]")
        if self.dino_channels is not None:
            if context_dino is None or context_dino.shape[:2] != context.shape[:2]:
                raise ValueError("DINO-enabled model expects context_dino [B,T,C,D,H,W]")
            if context_dino.shape[2] != self.dino_channels:
                raise ValueError(f"Expected {self.dino_channels} DINO channels")
        elif context_dino is not None:
            raise ValueError("context_dino supplied to model without a DINO adapter")
        # Transfer small metadata once; no accelerator-to-host synchronization
        # occurs inside temporal integration loops. Masked timestamps may be NaN.
        mask_rows = mask.detach().cpu().tolist()
        time_rows = context_times.detach().double().cpu().tolist()
        query_rows = target_times.detach().double().cpu().tolist()
        indices = []
        for row_mask, row_times, row_queries in zip(mask_rows, time_rows, query_rows):
            if any(value not in (0, 1) for value in row_mask):
                raise ValueError("mask must be binary")
            valid = [index for index, value in enumerate(row_mask) if value]
            if not valid:
                raise ValueError("Every patient needs at least one valid observation")
            observed_times = [row_times[index] for index in valid]
            if not all(math.isfinite(value) for value in observed_times + row_queries):
                raise ValueError("Valid observation and query times must be finite")
            if any(b < a for a, b in zip(observed_times, observed_times[1:])):
                raise ValueError("Valid history must be in chronological order")
            if any(value < observed_times[-1] for value in row_queries):
                raise ValueError("Future query times must be at or after the prediction cutoff")
            indices.append(valid)
        grid_shape = (1, *context.shape[2:], 3)
        if self._sampling_grid.shape != grid_shape or self._sampling_grid.device != context.device:
            self._sampling_grid = _identity_grid(context[:, :1])
        predictions, histories, history_valid, last_context, statistics = [], [], [], [], []
        semantic_losses, forecast_auxiliary = [], {}
        for patient, valid in enumerate(indices):
            first = valid[0]
            image = context[patient:patient + 1, first:first + 1].float()
            now = time_rows[patient][first]
            first_dino = None if context_dino is None else context_dino[patient:patient + 1, first].float()
            features = self._encode_observation(image, first_dino)
            memory = self.initialize_memory(torch.cat((features, self._plane(features, now)), dim=1))
            priors = [torch.zeros_like(image[0, 0]) for _ in range(slots)]
            scored = torch.zeros(slots, device=context.device, dtype=torch.bool)
            for index in valid[1:]:
                next_time = time_rows[patient][index]
                image, memory = self._trajectory(image, memory, now, [next_time], statistics)[0]
                priors[index] = image[0, 0]  # Recorded before reading this observation.
                scored[index] = True
                observed = context[patient:patient + 1, index:index + 1].float()
                observed_dino = None if context_dino is None else context_dino[patient:patient + 1, index].float()
                observed_features = self._encode_observation(observed, observed_dino)
                observed_image_features = self.encoder(observed)
                predicted_image_features = self.encoder(image)
                update, auxiliary = self._update_memory(memory, observed_features,
                    observed_image_features, predicted_image_features, observed_dino,
                    next_time - now)
                if auxiliary is not None:
                    semantic_losses.append(auxiliary)
                memory = memory + update
                image, now = observed, next_time
            last_context.append(image[0, 0])
            condition = None if forecast_condition is None else forecast_condition[patient:patient + 1]
            future, auxiliary = self._forecast(image, memory, now, query_rows[patient], statistics, condition)
            for name, value in auxiliary.items():
                forecast_auxiliary.setdefault(name, []).append(value)
            predictions.append(torch.stack([state[0][0, 0] for state in future]))
            histories.append(torch.stack(priors))
            history_valid.append(scored)
        rms = torch.stack(statistics).mean(dim=0).sqrt() if statistics else context.new_zeros(3)
        result = {"predictions": torch.stack(predictions),
                "history_predictions": torch.stack(histories),
                "history_valid": torch.stack(history_valid),
                "last_context": torch.stack(last_context),
                "velocity_rms": rms[0], "source_rms": rms[1], "memory_derivative_rms": rms[2]}
        if semantic_losses:
            result["semantic_prediction_loss"] = torch.stack(semantic_losses).mean()
        result.update({name: torch.cat(values) for name, values in forecast_auxiliary.items()})
        return result


__all__ = ["HistoryCalibratedTrajectory", "warp_volume"]
