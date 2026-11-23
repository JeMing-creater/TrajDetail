"""Read the exact images and times available to the existing three-fold comparison."""

from pathlib import Path
import shutil

import numpy as np
import torch

from src.compare import cases
from src.crossval_baselines import to_axial
from src.crossval_data import LegacyWindows, NEW, RAW_CACHE, SPLITS, read

CHECKPOINT_CACHE = Path("/tmp/generation_comparator_checkpoints_20261122")


def load_partitions(root, fold, dataset, *, all_history=False, smoke=False, splits=SPLITS):
    fold_root = root / f"fold{fold}"
    cohorts = NEW if dataset == "mu-dlbs" else (dataset,)
    rows = {cohort: {} for cohort in cohorts}
    for cohort in cohorts:
        for split in splits:
            if cohort in NEW:
                manifest = read(fold_root / cohort / "manifest.json")
                scale = manifest["time_unit_days"] / 365.25
                records = [(patient, volume, times.float() * scale)
                           for patient, volume, times in cases(manifest, RAW_CACHE / cohort, split, "cpu")]
                rows[cohort][split] = records[:2] if smoke else records
            else:
                windows = LegacyWindows(fold_root, cohort, split, train_windows=1)
                indices = range(min(2, len(windows)) if smoke else len(windows))
                rows[cohort][split] = [legacy_record(windows[index], cohort, all_history) for index in indices]
    return rows


def legacy_record(item, dataset, all_history):
    visible = np.flatnonzero(item["mask"])
    if not all_history:
        visible = visible[-2:]
        if len(visible) == 1:
            visible = np.repeat(visible, 2)
    times = np.append(item["context_times"][visible], item["target_times"][-1])
    if not len(visible) or times[-1] <= times[-2]:
        raise ValueError(f"Invalid causal window for {item['patient_id']}")
    images = [item["context"][index] for index in visible] + [item["targets"][-1]]
    images = np.stack([to_axial(volume, dataset) for volume in images]).astype(np.float32, copy=False)
    # LUMIERE timestamps are archived in weeks / 255. ACDC and ISLES'24
    # have intra-exam phases, so keep their normalized exam time coordinate.
    scale = 255 / (365.25 / 7) if dataset == "lumiere" else 1
    return item["patient_id"], torch.from_numpy(images), torch.as_tensor((times - times[0]) * scale)


def checkpoint_path(output, name):
    return output / name


def archive_checkpoint(source, destination):
    if source == destination:
        return
    temp = destination.with_suffix(".archive.part")
    shutil.copyfile(source, temp)
    temp.replace(destination)
    source.unlink()
