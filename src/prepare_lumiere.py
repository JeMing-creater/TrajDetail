"""Rebuild LUMIERE from registered skull-stripped T1 scans and the frozen visit list."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from src.experiments import load
from src.paths import DATA, PROTOCOL
from src.release_io import write_json
from src.trajectory.external_data import _normalize_lumiere


RELATIVE_IMAGE = Path("DeepBraTumIA-segmentation/atlas/skull_strip/t1_skull_strip.nii.gz")


def prepare_volume(path):
    volume = nib.load(path).get_fdata()
    clipped = np.clip(volume, *np.percentile(volume, (1, 99)))
    if clipped.max() <= clipped.min():
        raise ValueError(f"Constant MRI: {path}")
    normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min())
    return zoom(normalized, np.divide((96, 96, 64), volume.shape))


def match_visits(case, raw_root):
    patient = case["patient_id"]
    output = []
    for entry in case["visits"]:
        filename = raw_root / patient / entry["directory"] / RELATIVE_IMAGE
        volume = _normalize_lumiere(prepare_volume(filename))
        if hashlib.sha256(volume.tobytes()).hexdigest() != entry["prepared_sha256"]:
            raise ValueError(f"Preprocessed grid differs from the benchmark: {filename}")
        output.append((entry["week"], volume))
    return output


def prepare(raw_root, output, workers):
    cases = load(PROTOCOL / "lumiere_raw.json")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {split: [] for split in ("train", "validation", "test")}
    cases.sort(key=lambda x: (x["split"], x["split_index"]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        prepared = pool.map(lambda case: match_visits(case, raw_root), cases)
        for case, visits in zip(cases, prepared, strict=True):
            weeks, volumes = zip(*visits, strict=True)
            filename = f"{case['patient_id']}.npy"
            np.save(output / filename, np.stack(volumes), allow_pickle=False)
            manifest[case["split"]].append(dict(patient_id=case["patient_id"], file=filename, weeks=weeks))
    write_json(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Imaging directory containing Patient-NNN folders")
    parser.add_argument("--output", type=Path, default=DATA / "source_cache/lumiere")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.raw_root, args.output, args.workers)


if __name__ == "__main__":
    main()
