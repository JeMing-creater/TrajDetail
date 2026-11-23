"""Local feature correspondence and paired-error routing for the frozen trajectory."""

import torch
from torch import nn
import torch.nn.functional as F

from methods.trajdetail.control import ControlTrajectory, _plans
from methods.trajdetail.unified_model import HistoryCalibratedTrajectory, _identity_grid, warp_volume


def paired_benefit(base, proposal, target, shape):
    """MSE saved by the correction, in local blocks of the common image grid."""
    batch, count, *spatial = base.shape
    difference = (base - target).square() - (proposal - target).square()
    return F.adaptive_avg_pool3d(difference.reshape(batch * count, 1, *spatial),
                                 shape).reshape(batch, count, *shape)


def neighbourhood(volume):
    """Return the centered voxel and its six face neighbours in x/y/z pairs."""
    padded = F.pad(volume, (1, 1, 1, 1, 1, 1), mode="replicate")
    return torch.stack((volume, padded[..., 1:-1, 1:-1, 2:], padded[..., 1:-1, 1:-1, :-2],
                        padded[..., 1:-1, 2:, 1:-1], padded[..., 1:-1, :-2, 1:-1],
                        padded[..., 2:, 1:-1, 1:-1], padded[..., :-2, 1:-1, 1:-1]), dim=1)


def expected_direction(scores):
    """Convert probabilities on six neighbour positions and zero to x/y/z."""
    probabilities = scores.float().softmax(dim=1)
    direction = probabilities[:, 1::2] - probabilities[:, 2::2]
    return direction, probabilities


def frozen_geometry(image):
    gradients = []
    for dim in (2, 3, 4):
        pad = [0] * 6
        pad[2 * (4 - dim) + 1] = 1
        gradients.append(F.pad(image.diff(dim=dim), pad))
    return torch.cat((image, *gradients), dim=1)


class MatchedControl(ControlTrajectory):
    """Query the transported MRI or DINO token atlas at each future readout."""

    def __init__(self, *, use_dino, feature_channels=768, feature_size=(8, 16, 16),
                 benefit_scale=2e-4, **kwargs):
        super().__init__(capture_components=False, **kwargs)
        self.use_dino = use_dino
        self.feature_size = feature_size
        self.benefit_scale = benefit_scale
        self.descriptor_channels = 32 if use_dino else 28
        if use_dino:
            rng = torch.Generator().manual_seed(314159)
            projection, _ = torch.linalg.qr(torch.randn(feature_channels, self.descriptor_channels, generator=rng))
            self.register_buffer("descriptor_basis", projection.T[:, :, None, None, None])
        self.match_query = nn.Sequential(nn.Conv3d(self.memory_channels + 11, 24, 3, padding=1), nn.SiLU(),
                                         nn.Conv3d(24, 24, 3, padding=1), nn.SiLU())
        self.match_descriptor = nn.Conv3d(24, self.descriptor_channels, 1)
        self.match_prior = nn.Conv3d(24, 7, 1)
        nn.init.zeros_(self.match_prior.weight)
        nn.init.zeros_(self.match_prior.bias)
        self.benefit_head = nn.Sequential(nn.Conv3d(24 + 4, 16, 3, padding=1), nn.SiLU(),
                                          nn.Conv3d(16, 12, 3, padding=1), nn.SiLU(), nn.Conv3d(12, 1, 1))
        nn.init.zeros_(self.benefit_head[-1].weight)
        nn.init.zeros_(self.benefit_head[-1].bias)
        self.register_buffer("max_speed", torch.tensor([4., 4., 1.5]).view(1, 3, 1, 1, 1))

    def load_control(self, state_dict):
        unfilled, extra = self.load_state_dict(state_dict, strict=False)
        if extra or any(not name.startswith(("match_", "benefit_", "descriptor_basis", "max_speed"))
                        for name in unfilled):
            raise ValueError("checkpoint does not match the frozen trajectory")
        self.train_stage("matching")

    def train_stage(self, stage):
        if stage not in {"matching", "benefit"}:
            raise ValueError(f"unknown training stage: {stage}")
        prefix = "match_" if stage == "matching" else "benefit_"
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))

    def descriptor(self, image, features, size):
        with torch.autocast(device_type=image.device.type, enabled=False):
            if self.use_dino:
                if features is None:
                    raise ValueError("DINOv3 matching needs observed patch descriptors")
                projected = F.conv3d(F.normalize(features.float(), dim=1), self.descriptor_basis)
                projected = F.interpolate(projected, size=size, mode="trilinear", align_corners=True)
            else:
                geometry = F.adaptive_avg_pool3d(frozen_geometry(image.float()), size)
                projected = neighbourhood(geometry - geometry.mean(dim=(2, 3, 4), keepdim=True)).flatten(1, 2)
            return F.normalize(projected, dim=1, eps=1e-5)

    def forward(self, context, mask, context_times, target_times, *, last_features=None):
        return HistoryCalibratedTrajectory.forward(
            self, context, mask, context_times, target_times, forecast_condition=last_features)

    def _coarse_readouts(self, image, memory, start, queries, statistics, size):
        """Share the MRI rollout and carry its backward map on the smaller atlas grid."""
        plans = _plans(start, queries, self.max_step)
        reference = image.new_empty(1, 1, *size)
        coordinates = _identity_grid(reference).movedim(-1, 1)
        ratios = image.new_tensor([(low - 1) / max(high - 1, 1)
                                   for low, high in zip(reversed(size), reversed(image.shape[-3:]))])
        ratios = ratios.view(1, 3, 1, 1, 1)
        states, fields = [(image, memory, coordinates)], {}

        def field(index):
            if index not in fields:
                value = self._vector_field(*states[index][:2], start + index * self.max_step, statistics)
                velocity = F.interpolate(value[0], size=size, mode="trilinear", align_corners=True) * ratios
                fields[index] = value, velocity
            return fields[index]

        def advance(index, step):
            value, velocity = field(index)
            current_image, current_memory, current_coordinates = states[index]
            prediction, new_memory = self._advance(current_image, current_memory, value, step)
            return prediction, new_memory, warp_volume(current_coordinates, step * velocity)

        for index in range(max(index for index, _ in plans)):
            states.append(advance(index, self.max_step))
        return [advance(index, fraction) if fraction else states[index] for index, fraction in plans]

    def _forecast(self, image, memory, start, queries, statistics, condition):
        size = tuple(min(lattice, dimension) for lattice, dimension in zip(self.feature_size, image.shape[-3:]))
        with torch.no_grad():
            atlas = self.descriptor(image, condition, size)
            coarse = self._coarse_readouts(image, memory, start, queries, statistics, size)
            anchor_geom = F.adaptive_avg_pool3d(frozen_geometry(image), size)
        predictions, proposals, bases, query_descriptors, utility_maps, gates = [], [], [], [], [], []
        matching_evidence = []
        for (baseline, memory_at_t, backward), time in zip(coarse, queries):
            elapsed = time - start
            # Each query reads the same canonical trajectory and uses the full
            # elapsed time to turn the local readout velocity into displacement.
            features = torch.cat((F.adaptive_avg_pool3d(memory_at_t.float(), size),
                                  F.adaptive_avg_pool3d(frozen_geometry(baseline), size),
                                  anchor_geom, self._plane(anchor_geom, start),
                                  self._plane(anchor_geom, time), self._plane(anchor_geom, elapsed)), dim=1)
            query_state = self.match_query(features)
            query = F.normalize(self.match_descriptor(query_state).float(), dim=1, eps=1e-5)
            atlas_at_t = F.grid_sample(atlas, backward.movedim(1, -1), align_corners=True,
                                      padding_mode="border")
            candidates = F.normalize(neighbourhood(atlas_at_t), dim=2, eps=1e-5)
            correspondence = (query[:, None] * candidates).sum(dim=2)
            direction, probabilities = expected_direction(5. * correspondence + self.match_prior(query_state).float())
            displacement = F.interpolate(elapsed * self.max_speed * direction, size=image.shape[-3:],
                                         mode="trilinear", align_corners=True)
            proposal = warp_volume(baseline, -displacement, self._sampling_grid) if elapsed else baseline
            evidence = torch.cat((probabilities[:, :1], direction.norm(dim=1, keepdim=True),
                                  (correspondence[:, 1:].amax(dim=1, keepdim=True) - correspondence[:, :1]),
                                  F.adaptive_avg_pool3d((proposal - baseline).abs(), size)), dim=1)
            utility = self.benefit_head(torch.cat((query_state.detach().float(), evidence.detach()), dim=1)).float()
            gate = (2. * utility).clamp(0., 1.)
            fine_gate = F.interpolate(gate, size=image.shape[-3:], mode="trilinear", align_corners=False)
            predictions.append((baseline + fine_gate * (proposal - baseline), memory_at_t))
            proposals.append(proposal)
            bases.append(baseline)
            query_descriptors.append(query)
            utility_maps.append(utility)
            gates.append(gate)
            if getattr(self, "capture_matching_evidence", False):
                entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
                matching_evidence.append(torch.cat((
                    (1 - entropy / 1.945910149).clamp(0, 1),
                    correspondence.amax(dim=1, keepdim=True),
                    gate, evidence[:, 1:2],
                ), dim=1))
        extras = {"proposals": torch.cat(proposals, dim=1), "base_forecasts": torch.cat(bases, dim=1),
                             "queries": torch.stack(query_descriptors, dim=1),
                             "utilities": torch.cat(utility_maps, dim=1), "gates": torch.cat(gates, dim=1)}
        if matching_evidence:
            self.matching_evidence = torch.stack(matching_evidence, dim=1)
        return predictions, extras
