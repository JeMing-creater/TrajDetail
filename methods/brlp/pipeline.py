"""BrLP training and the validation-selected whole-volume decoder."""
import hashlib
import sys

import numpy as np
import torch

from methods.brlp import brlp_adapter as adapter
from methods.brlp.brlp_audit import scale_latents
from methods.brlp.brlp_full_volume import fit as fit_decoder
from methods.brlp.brlp_repair import forecasts
from src.experiments import load
from src.release_io import write_json


def fit(train_by_cohort, valid_by_cohort, directory, settings):
    train = [record for cohort in train_by_cohort.values() for record in cohort]
    valid = [record for cohort in valid_by_cohort.values() for record in cohort]
    list(adapter.train_and_predict(train, valid, [], directory, False, settings=settings))
    ae, model, control = open_models(directory)
    decoder = new_decoder(directory / "brlp_ae.pt")
    fit_decoder(decoder, train, valid_by_cohort, directory / "decoder.pt",
                settings["decoder_steps"], settings["decoder_lr"], True)
    decoder = new_decoder(directory / "decoder.pt")
    trajectory = adapter.fit_progression(train)
    scale = scale_latents(ae, train, trajectory)
    per_cohort = {}
    for dataset, records in valid_by_cohort.items():
        pair_scores = []
        for record in records:
            pair = predict(ae, model, control, record, trajectory, scale, dataset, settings, decoder)
            target = record[1][2].cuda()
            pair_scores.append([float((image - target).square().mean().sqrt()) for image in pair])
        per_cohort[dataset] = np.mean(pair_scores, axis=0).tolist()
    base, tuned = np.mean(list(per_cohort.values()), axis=0)
    write_json(directory / "brlp_selection.json", {"use_tuned_decoder": bool(tuned < base),
        "validation_nrmse_by_cohort": per_cohort})


def new_decoder(path):
    sys.path.insert(0, str(adapter.AUTHOR))
    from brlp import networks

    decoder = networks.init_autoencoder().cuda().eval().requires_grad_(False)
    adapter.restore(path, decoder)
    return decoder


def open_models(directory):
    sys.path.insert(0, str(adapter.AUTHOR))
    from brlp import networks

    models = []
    for factory, name in ((networks.init_autoencoder, "ae"), (networks.init_latent_diffusion, "diffusion"),
                          (networks.init_controlnet, "control")):
        model = factory().cuda().eval().requires_grad_(False)
        adapter.restore(directory / f"brlp_{name}.pt", model)
        models.append(model)
    return models


def predict(ae, model, control, record, trajectory, scale, dataset, settings, decoder):
    patient = record[0]
    seed = int.from_bytes(hashlib.sha256(f"{dataset}/{patient}".encode()).digest()[:4], "big")
    return forecasts(ae, model, control, record, trajectory, scale, "las_mask",
                     settings["latent_samples"], settings["sampling_steps"], 0, seed, decoder)


def evaluate(directory, train, test, dataset, settings):
    ae, model, control = open_models(directory)
    decoder = (new_decoder(directory / "decoder.pt")
               if load(directory / "brlp_selection.json")["use_tuned_decoder"] else None)
    trajectory = adapter.fit_progression(train)
    scale = scale_latents(ae, train, trajectory)
    for record in test:
        estimate, repaired = predict(ae, model, control, record, trajectory, scale, dataset, settings, decoder)
        yield record[0], (estimate if repaired is None else repaired).cpu().float().numpy()
