"""Whole-volume metrics for the shared image grid and center edge region."""
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from methods.trajdetail.audit_acdc_sharpness import regions
from methods.trajdetail.train_acdc_frequency_readout import measure
from src.trajectory.external_eval import _score_one


def score(prediction, target, dataset, device):
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite scored prediction")
    pred, truth = (torch.as_tensor(np.ascontiguousarray(vol), device=device)[None, None]
                   for vol in (prediction, target))
    row = _score_one(pred, truth, None)
    pred_axial, truth_axial = (np.moveaxis(vol, -1, 0) if dataset == "lumiere" else vol
                              for vol in (prediction, target))
    masks = regions(truth_axial)
    core = np.zeros(truth_axial.shape, dtype=bool)
    _, h, w = core.shape
    core[:, round(37*h/128):round(100*h/128), round(37*w/128):round(100*w/128)] = True
    target_detail = truth_axial - gaussian_filter(truth_axial, (0, .8, .8))
    core_edge = core & masks[2]
    if np.any(target_detail[core_edge]) and np.any(core & masks[1]):
        sharpness = measure(pred_axial, truth_axial, (target_detail, core & masks[1], core_edge, masks))
        row.update({key: sharpness[key] for key in (
            "core_edge_hf_ratio", "core_edge_hf_nrmse", "core_foreground_hf_rmse", "core_nrmse")})
    return row
