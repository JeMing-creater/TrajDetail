"""Train the original cohort adapters without opening held-out targets."""
from pathlib import Path
from types import SimpleNamespace

import torch

from src.experiments import LEGACY, NEW, REFERENCES, SIMPLE, members, load
from src.paths import DATA, PROTOCOL, SPLITS_ROOT


def delta_settings(config, directory, dataset, **extra):
    return SimpleNamespace(output=directory, seed=42, train_break=None, val_break=None, test_break=None,
                           all_train_pairs=False, legacy_euler=True, replay_direct=False,
                           with_mu=config["group"] == "mu-dlbs", eval_cohort=dataset,
                           **config["training"], **extra)


def delta_data(partitions, cohorts, split):
    order = ("dlbs", "mu") if set(cohorts) == set(NEW) else cohorts
    return [(patient, vol, times / 10)
            for cohort in order for patient, vol, times in partitions[cohort][split]]


def train_experiment(config, directory, fold, device_name):
    from src.crossval_baselines import fit
    from src.crossval_data import RAW_CACHE, LegacyWindows
    from src import compare

    torch.set_num_threads(2)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    method, group, settings = (config[key] for key in ("method", "group", "training"))
    if method in REFERENCES:
        return
    if method in SIMPLE and group in LEGACY:
        dynamic = group != "lumiere" and method != "imageflownet"
        training, valid = (LegacyWindows(directory, group, split, train_windows=1 if dynamic else 2,
                                         dynamic=dynamic) for split in ("train", "validation"))
        valid.source = training.source
        fit(method, group, (training, valid, None), device, directory, settings=settings)
        return
    if method in SIMPLE:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        manifests = [load(directory / cohort / "manifest.json") for cohort in NEW]
        training, valid = ([list(compare.cases(manifest, RAW_CACHE / cohort, split, device))
                            for cohort, manifest in zip(NEW, manifests, strict=True)]
                           for split in ("train", "validation"))
        compare.train(method, training, valid, device, settings["epochs"], directory,
                      manifests[0]["shape_zyx"], domain_balanced=True,
                      learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"],
                      width=settings.get("width", 32))
        return
    if method == "trajdetail":
        from methods.trajdetail import crossval_trajectory, scratch_readout, scratch_trajectory

        phases = (settings["base_epochs"], settings["voxel_epochs"], settings["detail_epochs"])
        if group == "legacy":
            for cohort in LEGACY:
                crossval_trajectory.upstream(directory, cohort, device, settings["trajectory"])
                crossval_trajectory.extract(directory, cohort, device, ("train", "validation"))
            crossval_trajectory.shared_readout(directory, device, phases=phases)
        else:
            from src.trajectory.train import seed_all

            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            for cohort in NEW:
                path = directory / cohort
                manifest = load(path / "manifest.json")
                training, valid = (scratch_trajectory.windows(manifest, RAW_CACHE / cohort, split)
                                   for split in ("train", "validation"))
                stage = SimpleNamespace(output=path, dataset=cohort, device=str(device), **settings["trajectory"])
                seed_all(42)
                scratch_trajectory.fit_control(training, valid, stage, device)
                seed_all(42)
                model = scratch_trajectory.MatchedControl(use_dino=True).to(device)
                model.load_control(torch.load(path / "control.pt", map_location=device, weights_only=True)["model"])
                features = scratch_trajectory.Features("lumiere", device)
                paired = [item for item in training for _ in range(2)]
                for which in ("matching", "benefit"):
                    scratch_trajectory.train_stage(model, which, paired, valid, features, device, stage)
            scratch_readout.train_pipeline(SimpleNamespace(output=directory, device=device_name, raw_cache=RAW_CACHE,
                readout_cache=directory / "new_causal_features", base_epochs=phases[0],
                voxel_epochs=phases[1], detail_epochs=phases[2]))
        return

    if device.type != "cuda":
        raise ValueError(f"{method} uses CUDA in its reference adapter")
    from src.comparator_windows import load_partitions

    cohorts = members(group)
    parts = load_partitions(SPLITS_ROOT, fold, group, splits=("train", "validation"))
    training, valid = ([row for cohort in cohorts for row in parts[cohort][split]]
                       for split in ("train", "validation"))
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    if method == "log4d":
        from methods.log4d.log4d_adapter import train_and_predict

        list(train_and_predict(training, valid, [], directory, settings=settings))
    elif method == "brlp":
        from methods.brlp.pipeline import fit

        fit({cohort: parts[cohort]["train"] for cohort in cohorts},
            {cohort: parts[cohort]["validation"] for cohort in cohorts}, directory, settings)
    elif method == "delta_lfm":
        from methods.delta_lfm.delta_lfm_dlbs import fit_ae, fit_flow, load_ae

        args = delta_settings(config, directory, cohorts[0], flow_checkpoint=directory / "flow.pt")
        if group == "mu-dlbs":
            args.train_break, args.val_break = (len(parts["dlbs"][s]) for s in ("train", "validation"))
        if not (DATA / "delta_lfm" / "autoencoder_epoch273.pt").is_file():
            raise FileNotFoundError("Place the Δ-LFM author MAISI checkpoint at TRAJDETAIL_DATA/delta_lfm/autoencoder_epoch273.pt")
        model = load_ae(DATA / "delta_lfm" / "autoencoder_epoch273.pt")
        training, valid = [delta_data(parts, cohorts, s) for s in ("train", "validation")]
        fit_ae(model, training, valid, args)
        fit_flow(model.eval().requires_grad_(False), training, valid, [], args)
