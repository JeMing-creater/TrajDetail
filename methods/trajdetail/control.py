"""Differentiable low-resolution controls and source/transport diagnostics."""

from __future__ import annotations

import math
import time
from typing import Iterable

import torch
from torch import Tensor
import torch.nn.functional as F

from methods.trajdetail.diagnose_unified import DiagnosticTrajectory
from methods.trajdetail.unified_model import warp_volume


def _plans(start: float, queries: Iterable[float], max_step: float) -> list[tuple[int, float]]:
    plans = []
    for query in queries:
        gap = float(query) - float(start)
        if not math.isfinite(gap) or gap < 0:
            raise ValueError("queries must be finite and at or after start")
        ratio = gap / max_step
        nearest = round(ratio)
        if abs(ratio - nearest) <= 1e-5:
            plans.append((max(0, nearest), 0.0))
        else:
            full = int(math.floor(ratio))
            plans.append((full, gap - full * max_step))
    return plans


def _lowres_shape(anchor: Tensor) -> tuple[int, int, int]:
    d, h, w = anchor.shape[-3:]
    return max(1, (d + 1) // 2), max(1, (h + 3) // 4), max(1, (w + 3) // 4)


def _upsample(value: Tensor, size: tuple[int, int, int]) -> Tensor:
    return F.interpolate(value.float(), size=size, mode="trilinear", align_corners=False)


def _control_durations(plans: list[tuple[int, float]], max_step: float) -> list[float]:
    if not plans:
        return []
    last_index = max(index for index, _ in plans)
    # Controls are integrated over the actual furthest horizon. A control that
    # serves both a fractional read and a later grid endpoint still gets its
    # full canonical duration.
    last_remainder = max((remainder for index, remainder in plans if index == last_index), default=0.0)
    count = last_index + int(last_remainder > 0)
    durations = [max_step] * count
    if last_remainder > 0:
        durations[-1] = last_remainder
    return durations


def rollout_controls(anchor: Tensor, start: float, queries: Iterable[float],
                     u: Tensor | None = None, r: Tensor | None = None,
                     *, max_step: float = 1 / 11) -> dict[str, Tensor | list[float]]:
    """Roll out voxel controls defined on the encoder's low-resolution grid.

    ``u`` has shape ``[N,3,D/2,H/4,W/4]`` and ``r`` has shape
    ``[N,1,D/2,H/4,W/4]``. The fields are independently upsampled for each
    canonical step. Fractional queries read the preceding field without
    feeding back into the canonical trajectory.
    """
    if anchor.ndim != 5 or anchor.shape[1] != 1 or anchor.shape[0] < 1:
        raise ValueError("anchor must have shape [B,1,D,H,W]")
    if not math.isfinite(max_step) or max_step <= 0 or max_step > 1 / 11:
        raise ValueError("max_step must be finite and in (0, 1/11]")
    query_list = [float(value) for value in queries]
    plans = _plans(start, query_list, max_step)
    required = max((index + (remainder > 0) for index, remainder in plans), default=0)
    low_shape = _lowres_shape(anchor)
    device, dtype = anchor.device, anchor.dtype
    if u is None:
        u = torch.zeros(required, 3, *low_shape, device=device, dtype=dtype)
    if r is None:
        r = torch.zeros(required, 1, *low_shape, device=device, dtype=dtype)
    if u.ndim != 5 or r.ndim != 5 or u.shape[1] != 3 or r.shape[1] != 1:
        raise ValueError("u and r must have shapes [N,3,d,h,w] and [N,1,d,h,w]")
    if u.shape[0] < required or r.shape[0] < required:
        raise ValueError("controls do not cover the requested horizon")
    if tuple(u.shape[2:]) != low_shape or tuple(r.shape[2:]) != low_shape:
        raise ValueError("controls must use the low-resolution D/2,H/4,W/4 grid")

    image_shape = tuple(anchor.shape[2:])
    transport = anchor[:, 0].float()
    source = torch.zeros_like(transport)
    states: list[tuple[Tensor, Tensor]] = [(transport, source)]
    fields: dict[int, tuple[Tensor, Tensor]] = {}

    def field(index: int) -> tuple[Tensor, Tensor]:
        if index not in fields:
            fields[index] = (_upsample(u[index:index + 1], image_shape),
                             _upsample(r[index:index + 1], image_shape))
        return fields[index]

    for index in range(max((n for n, _ in plans), default=0)):
        uf, rf = field(index)
        old_m, old_r = states[-1]
        displacement = max_step * uf
        new_m = warp_volume(old_m.unsqueeze(1), displacement).squeeze(1)
        new_r = warp_volume(old_r.unsqueeze(1), displacement).squeeze(1) + max_step * rf.squeeze(1)
        states.append((new_m, new_r))

    outputs_m, outputs_r = [], []
    for index, remainder in plans:
        if remainder == 0:
            current_m, current_r = states[index]
        else:
            uf, rf = field(index)
            current_m, current_r = states[index]
            displacement = remainder * uf
            current_m = warp_volume(current_m.unsqueeze(1), displacement).squeeze(1)
            current_r = warp_volume(current_r.unsqueeze(1), displacement).squeeze(1) + remainder * rf.squeeze(1)
        outputs_m.append(current_m)
        outputs_r.append(current_r)
    if outputs_m:
        transport_out, source_out = torch.stack(outputs_m, dim=1), torch.stack(outputs_r, dim=1)
    else:
        empty = anchor.new_empty(anchor.shape[0], 0, *anchor.shape[2:])
        transport_out = source_out = empty
    return {"predictions": transport_out + source_out,
            "transport_components": transport_out,
            "source_components": source_out,
            "durations": _control_durations(plans, max_step),
            "step_starts": [float(start) + i * max_step for i in range(required)]}


def _action_cost(u: Tensor, r: Tensor, durations: list[float], *,
                 lambda_u: float, lambda_grad: float, lambda_r: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not durations:
        zero = u.sum() * 0
        return zero, zero, zero, zero
    weights = u.new_tensor(durations).view(-1, 1, 1, 1, 1)
    # Amplitude is a nonnegative magnitude; both velocity directions incur
    # action cost while the control itself remains signed.
    positive_u = u.abs().mean(dim=(1, 2, 3, 4))
    grad_per_step = torch.stack([((u[i, :, 1:] - u[i, :, :-1]).abs().mean() if u.shape[2] > 1 else u[i].sum() * 0) +
                                 ((u[i, :, :, 1:] - u[i, :, :, :-1]).abs().mean() if u.shape[3] > 1 else u[i].sum() * 0) +
                                 ((u[i, :, :, :, 1:] - u[i, :, :, :, :-1]).abs().mean() if u.shape[4] > 1 else u[i].sum() * 0)
                                 for i in range(len(durations))])
    r_per_step = r.abs().mean(dim=(1, 2, 3, 4))
    u_cost = (positive_u * weights[:, 0, 0, 0, 0]).sum() * lambda_u
    grad_cost = (grad_per_step * weights[:, 0, 0, 0, 0]).sum() * lambda_grad
    r_cost = (r_per_step * weights[:, 0, 0, 0, 0]).sum() * lambda_r
    return u_cost + grad_cost + r_cost, u_cost, grad_cost, r_cost


def solve_teacher(anchor: Tensor, start: float, queries: Iterable[float], targets: Tensor,
                  *, steps: int = 150, lr_u: float = 0.15, lr_r: float = 0.03,
                  lambda_u: float = 1e-5, lambda_grad: float = 1e-4,
                  lambda_r: float = 0.01, source_only: bool = False,
                  no_action: bool = False, return_controls: bool = True,
                  max_step: float = 1 / 11, record_every: int = 10,
                  primary_weight: float = 2.0) -> dict:
    """Fit a free low-resolution transport/source control to future targets."""
    query_list = [float(value) for value in queries]
    if anchor.ndim != 5 or anchor.shape[1] != 1 or targets.shape != (anchor.shape[0], len(query_list), *anchor.shape[2:]):
        raise ValueError("anchor/targets must have shapes [B,1,D,H,W] and [B,K,D,H,W]")
    if anchor.shape[0] != 1:
        raise ValueError("teacher fitting currently supports batch size one")
    plans = _plans(start, query_list, max_step)
    required = max((index + int(remainder > 0) for index, remainder in plans), default=0)
    low_shape = _lowres_shape(anchor)
    u = torch.zeros(required, 3, *low_shape, device=anchor.device, dtype=anchor.dtype, requires_grad=not (no_action or source_only))
    r = torch.zeros(required, 1, *low_shape, device=anchor.device, dtype=anchor.dtype, requires_grad=not no_action)
    params = []
    if not no_action and not source_only:
        params.append({"params": [u], "lr": lr_u})
    if not no_action:
        params.append({"params": [r], "lr": lr_r})
    optimizer = torch.optim.Adam(params) if params else None
    durations = _control_durations(plans, max_step)
    records = []
    tic = time.monotonic()
    for iteration in range(max(0, int(steps))):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        result = rollout_controls(anchor, start, query_list, u, r, max_step=max_step)
        prediction = result["predictions"]
        target = targets.to(device=prediction.device, dtype=prediction.dtype)
        difference = prediction - target
        mse_by_time = difference.square().flatten(2).mean(2)
        mae_by_time = difference.abs().flatten(2).mean(2)
        weights = prediction.new_ones(prediction.shape[1])
        if primary_weight <= 0:
            raise ValueError("primary_weight must be positive")
        weights[-1] = primary_weight
        mse = (mse_by_time * weights).sum(1).mean() / weights.sum()
        mae = (mae_by_time * weights).sum(1).mean() / weights.sum()
        action, u_cost, grad_cost, r_cost = _action_cost(u, r, durations,
                                                         lambda_u=lambda_u,
                                                         lambda_grad=lambda_grad,
                                                         lambda_r=lambda_r)
        loss = mse + 0.01 * mae + action
        if optimizer is not None and loss.requires_grad:
            loss.backward()
            optimizer.step()
        if iteration == 0 or iteration == steps - 1 or (record_every and (iteration + 1) % record_every == 0):
            records.append({"step": iteration + 1, "loss": float(loss.detach()),
                            "future_mse": float(mse.detach()), "future_mae": float(mae.detach()),
                            "action": float(action.detach()), "u_action": float(u_cost.detach()),
                            "gradient_action": float(grad_cost.detach()), "r_action": float(r_cost.detach())})
    result = rollout_controls(anchor, start, query_list, u, r, max_step=max_step)
    output = {"predictions": result["predictions"].detach().cpu(),
              "transport_components": result["transport_components"].detach().cpu(),
              "source_components": result["source_components"].detach().cpu(),
              "records": records,
              "config": {"steps": int(steps), "lr_u": lr_u, "lr_r": lr_r,
                         "lambda_u": lambda_u, "lambda_grad": lambda_grad, "lambda_r": lambda_r,
                         "source_only": source_only, "no_action": no_action, "max_step": max_step,
                         "gradient_spacing": "low_resolution_index", "primary_weight": primary_weight},
              "timing_seconds": time.monotonic() - tic}
    output["controls"] = {"u": u.detach().cpu(), "r": r.detach().cpu()} if return_controls else None
    return output


class ControlTrajectory(DiagnosticTrajectory):
    """The normalized-pruned student with optional exact M/R decomposition."""

    def __init__(self, *args, capture_components: bool = True, **kwargs):
        kwargs.setdefault("group_norm", True)
        kwargs.setdefault("prune_correction", True)
        super().__init__(*args, **kwargs)
        self.capture_components = bool(capture_components)

    def _capture_future_components(self, context: Tensor, mask: Tensor,
                                   context_times: Tensor, target_times: Tensor) -> tuple[Tensor, Tensor]:
        mask_rows = mask.detach().cpu().tolist()
        time_rows = context_times.detach().double().cpu().tolist()
        query_rows = target_times.detach().double().cpu().tolist()
        self._sampling_grid = self._sampling_grid if self._sampling_grid.numel() else None
        if self._sampling_grid is None or self._sampling_grid.shape != (1, *context.shape[2:], 3) or self._sampling_grid.device != context.device:
            from methods.trajdetail.unified_model import _identity_grid
            self._sampling_grid = _identity_grid(context[:, :1])
        all_m, all_r = [], []
        for patient, row_mask in enumerate(mask_rows):
            valid = [i for i, value in enumerate(row_mask) if value]
            first = valid[0]
            image = context[patient:patient + 1, first:first + 1].float()
            now = time_rows[patient][first]
            features = self.encoder(image)
            memory = self.initialize_memory(torch.cat((features, self._plane(features, now)), dim=1))
            for index in valid[1:]:
                next_time = time_rows[patient][index]
                # The diagnostic field head expects one slot for its field
                # statistics; component capture does not expose those stats.
                image, memory = self._trajectory(image, memory, now, [next_time], [None])[0]
                observed = context[patient:patient + 1, index:index + 1].float()
                observed_features = self.encoder(observed)
                error = observed_features - self.encoder(image)
                update = self.correction(torch.cat((memory, observed_features, error,
                                                    self._plane(memory, next_time - now)), dim=1))
                memory = memory + update
                image, now = observed, next_time
            plans = _plans(now, query_rows[patient], self.max_step)
            states_i = [(image, memory)]
            states_m = [image[:, 0]]
            states_r = [torch.zeros_like(image[:, 0])]
            fields = {}
            for step_index in range(max((n for n, _ in plans), default=0)):
                fields[step_index] = self._vector_field(*states_i[-1], now + step_index * self.max_step, [None])
                current_i, current_mem = self._advance(*states_i[-1], fields[step_index], self.max_step)
                uf, sf, _ = fields[step_index]
                if self.transport_enabled:
                    current_m = warp_volume(states_m[-1].unsqueeze(1), self.max_step * uf).squeeze(1)
                    current_source = warp_volume(states_r[-1].unsqueeze(1), self.max_step * uf).squeeze(1) + self.max_step * sf.squeeze(1)
                else:
                    current_m, current_source = states_m[-1], states_r[-1] + self.max_step * sf.squeeze(1)
                states_i.append((current_i, current_mem)); states_m.append(current_m); states_r.append(current_source)
            outputs_m, outputs_r = [], []
            for index, remainder in plans:
                if remainder == 0:
                    outputs_m.append(states_m[index]); outputs_r.append(states_r[index]); continue
                if index not in fields:
                    fields[index] = self._vector_field(*states_i[index], now + index * self.max_step, [None])
                uf, sf, _ = fields[index]
                current_m, current_source = states_m[index], states_r[index]
                if self.transport_enabled:
                    current_m = warp_volume(current_m.unsqueeze(1), remainder * uf).squeeze(1)
                    current_source = warp_volume(current_source.unsqueeze(1), remainder * uf).squeeze(1) + remainder * sf.squeeze(1)
                else:
                    current_source = current_source + remainder * sf.squeeze(1)
                outputs_m.append(current_m); outputs_r.append(current_source)
            all_m.append(torch.stack(outputs_m, dim=1)); all_r.append(torch.stack(outputs_r, dim=1))
        return torch.cat(all_m, dim=0), torch.cat(all_r, dim=0)

    def forward(self, context: Tensor, mask: Tensor, context_times: Tensor,
                target_times: Tensor) -> dict[str, Tensor]:
        result = super().forward(context, mask, context_times, target_times)
        if self.capture_components:
            transport, source = self._capture_future_components(context, mask, context_times, target_times)
            result["transport_components"] = transport
            result["source_components"] = source
        return result


__all__ = ["ControlTrajectory", "rollout_controls", "solve_teacher"]
