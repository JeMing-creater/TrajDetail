"""Shared-flow training and inference operations for the CRONOS paper head."""

import torch


def interpolate_context_times(context_times: torch.Tensor, target_times: torch.Tensor, tau: torch.Tensor):
    target_times = target_times.reshape(context_times.shape[0], 1)
    tau = tau.reshape(context_times.shape[0], 1)
    return (1 - tau) * context_times + tau * target_times


def masked_flow_mse(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None):
    error = (predicted - target).square()
    if valid is None:
        return error.mean()
    if valid.ndim == 4:
        valid = valid[:, None]
    if valid.ndim != 5:
        raise ValueError(f"Expected valid mask (B,1,D,H,W) or (B,D,H,W), got {tuple(valid.shape)}")
    weights = valid[:, None].to(error.dtype).expand_as(error)
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("No valid voxels in flow loss")
    return (error * weights).sum() / denominator


@torch.no_grad()
def shared_velocity_ode(
    model,
    context: torch.Tensor,
    context_time: torch.Tensor | None = None,
    target_time: torch.Tensor | None = None,
    flow_conditioning: str = "discrete",
    steps: int = 10,
):
    if context.ndim != 6:
        raise ValueError(f"Expected (B,T,C,D,H,W) context, got {tuple(context.shape)}")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if flow_conditioning not in {"discrete", "continuous"}:
        raise ValueError(f"Unknown CRONOS time mode: {flow_conditioning}")
    if flow_conditioning == "continuous" and (context_time is None or target_time is None):
        raise ValueError("Continuous CRONOS requires context and target timestamps")

    state = context.clone()
    batch_size, num_context, channels = state.shape[:3]
    step_size = 1.0 / steps
    for index in range(steps):
        tau = torch.full((batch_size,), index * step_size, device=state.device, dtype=state.dtype)
        if flow_conditioning == "continuous":
            time_condition = interpolate_context_times(context_time, target_time, tau)
        else:
            time_condition = tau
        velocity = model(state.reshape(batch_size, num_context * channels, *state.shape[-3:]), time_condition)
        state = state + step_size * velocity.unsqueeze(1).expand_as(state)
    return state.mean(dim=1)
