import torch


def unpack_context(context: torch.Tensor, volume_shape: tuple[int, int, int]):
    if context.ndim != 6:
        raise ValueError(f"context must have shape (B,T,1,D,H,W), got {tuple(context.shape)}")
    if context.shape[2] != 1:
        raise ValueError(f"expected a single image channel, got {context.shape[2]}")
    if tuple(context.shape[-3:]) != tuple(volume_shape):
        raise ValueError(f"expected volume shape {volume_shape}, got {tuple(context.shape[-3:])}")
    return context[:, :, 0]
