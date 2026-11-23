"""A shared multiscale decoder for transported detail and signed intensity updates."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def lowpass(image):
    return F.avg_pool2d(F.pad(image, (1, 1, 1, 1), mode="replicate"), 3, stride=1)


def conv(channels_in, channels_out):
    return nn.Sequential(nn.Conv2d(channels_in, channels_out, 3, padding=1),
                         nn.GroupNorm(4, channels_out), nn.SiLU())


class SharedForecast(nn.Module):
    """Fuse competing causal forecasts, then predict low- and high-frequency residuals."""

    def __init__(self, width=24):
        super().__init__()
        self.stem = nn.Sequential(conv(18, width), conv(width, width))
        self.level1 = nn.Sequential(conv(width, width * 2), conv(width * 2, width * 2))
        self.level2 = nn.Sequential(conv(width * 2, width * 3), conv(width * 3, width * 3))
        self.level3 = nn.Sequential(conv(width * 3, width * 4), conv(width * 4, width * 4))
        self.condition = nn.Sequential(nn.Linear(9, width * 2), nn.SiLU(), nn.Linear(width * 2, width * 8))
        self.decode2 = nn.Sequential(conv(width * 7, width * 3), conv(width * 3, width * 3))
        self.decode1 = nn.Sequential(conv(width * 5, width * 2), conv(width * 2, width * 2))
        self.decode0 = nn.Sequential(conv(width * 3, width), conv(width, width))
        self.route = nn.Conv2d(width, 3, 1)
        self.intensity = nn.Conv2d(width * 2, 1, 1)
        self.detail = nn.Conv2d(width, 1, 1)
        for head in (self.route, self.intensity, self.detail):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.route.bias.copy_(torch.tensor([-1.5, 1.5, -1.5]))

    def forward(self, slices, time_condition):
        def down(x):
            return F.avg_pool2d(x, 2)

        def up(x, ref):
            return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

        full = self.stem(slices)
        half = self.level1(down(full))
        quarter = self.level2(down(half))
        coarse = self.level3(down(quarter))
        scale, shift = self.condition(time_condition).chunk(2, dim=-1)
        coarse = coarse * (1. + scale[:, :, None, None]) + shift[:, :, None, None]
        quarter_dec = self.decode2(torch.cat((up(coarse, quarter), quarter), dim=1))
        half_dec = self.decode1(torch.cat((up(quarter_dec, half), half), dim=1))
        full_dec = self.decode0(torch.cat((up(half_dec, full), full), dim=1))

        candidates = slices[:, (1, 4, 7)].float()
        weights = self.route(full_dec).float().softmax(dim=1)
        fused = (weights * candidates).sum(dim=1, keepdim=True)
        signal = .30 * up(self.intensity(half_dec).float().tanh(), fused)
        detail = .16 * self.detail(full_dec).float().tanh()
        prediction = fused + signal + detail - lowpass(detail)
        return prediction, fused


def forecast_loss(prediction, fusion, target, weight, hf_weight):
    error = prediction - target
    detail_error = error - lowpass(error)
    # Normalizing the dataset weight balances strong and weak intensity changes.
    per_slice = (error.square().mean((1, 2, 3)) + .01 * error.abs().mean((1, 2, 3))
                 + hf_weight * detail_error.square().mean((1, 2, 3))
                 + .1 * (fusion - target).square().mean((1, 2, 3)))
    return (weight * per_slice).mean()
