"""Learn a spatial velocity residual inside the frozen ACDC control trajectory."""
import torch
from torch import nn
import torch.nn.functional as F

from methods.trajdetail.control import ControlTrajectory, _plans
from methods.trajdetail.unified_model import HistoryCalibratedTrajectory, warp_volume


class LocalControlTrajectory(ControlTrajectory):
    def __init__(self, *, feature_channels=768, local_channels=12, max_residual=4.,
                 supervise_transport=False, **kwargs):
        super().__init__(capture_components=False, **kwargs)
        if local_channels < 1 or max_residual <= 0:
            raise ValueError("local_channels and max_residual must be positive")
        self.local_input = nn.Sequential(nn.Conv3d(5, local_channels, 3, padding=1), nn.SiLU())
        self.local_refinement = nn.Sequential(nn.Conv3d(local_channels, local_channels, 3, padding=1), nn.SiLU())
        self.local_displacement = nn.Conv3d(local_channels, 6, 1)
        nn.init.zeros_(self.local_displacement.weight)
        nn.init.zeros_(self.local_displacement.bias)
        self.feature_projection = nn.Conv3d(feature_channels, local_channels, 1, bias=False)
        self.feature_normalization = nn.GroupNorm(1, local_channels, affine=False)
        self.max_residual = max_residual
        self.supervise_transport = supervise_transport

    def load_control(self, state_dict):
        current = self.state_dict()
        expected = {key for key in current if not key.startswith(("local_", "feature_"))}
        if expected != set(state_dict):
            raise ValueError("checkpoint does not match the frozen trajectory backbone")
        self.load_state_dict(state_dict, strict=False)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(("local_", "feature_")))

    def forward(self, context, mask, context_times, target_times, *, last_features=None):
        return HistoryCalibratedTrajectory.forward(
            self, context, mask, context_times, target_times, forecast_condition=last_features)

    def _forecast(self, image, memory, start, queries, statistics, condition):
        size = memory.shape[-3:]
        geometry = [image]
        for dim in (2, 3, 4):
            gradient = image.diff(dim=dim).abs()
            pad = [0] * 6
            pad[2 * (4 - dim) + 1] = 1
            geometry.append(F.pad(gradient, pad))
        local_inputs = F.adaptive_avg_pool3d(torch.cat(geometry, dim=1), size)
        features = self.local_input(torch.cat((local_inputs, self._plane(memory, start)), dim=1))
        if condition is not None:
            projected = self.feature_normalization(self.feature_projection(condition.float()))
            features = features + F.interpolate(projected, size=size, mode="trilinear",
                                                 align_corners=False).to(features.dtype)
        controls = self.local_displacement(self.local_refinement(features)).float().tanh()
        first, later = controls.chunk(2, dim=1)
        first, later = (F.interpolate(value * self.max_residual, size=image.shape[-3:],
                                      mode="trilinear", align_corners=False)
                        for value in (first, later))
        capture_transport = self.training and self.supervise_transport
        velocities = {}

        def vector_field(current_image, current_memory, now, current_statistics):
            velocity, source, derivative = self._vector_field(
                current_image, current_memory, now, current_statistics)
            elapsed = min(max(now - start, 0.0), 1.0)
            velocity = velocity + torch.lerp(first, later, elapsed)
            if capture_transport:
                velocities[round((now - start) / self.max_step)] = velocity
            return velocity, source, derivative

        future = HistoryCalibratedTrajectory._trajectory(
            self, image, memory, start, queries, statistics, vector_field=vector_field)
        auxiliary = {"future_controls": controls}
        if capture_transport:
            plans = _plans(start, queries, self.max_step)
            transported = [image]
            for index in range(max(steps for steps, _ in plans)):
                transported.append(warp_volume(transported[-1], self.max_step * velocities[index],
                                               self._sampling_grid))
            endpoints = [warp_volume(transported[index], remainder * velocities[index],
                                      self._sampling_grid) if remainder else transported[index]
                         for index, remainder in plans]
            auxiliary["transport_components"] = torch.cat(endpoints, dim=1)
        return future, auxiliary
