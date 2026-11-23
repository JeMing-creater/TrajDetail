#!/usr/bin/env python3
"""Parallel path-compatible ACDC preprocessing for the CRONOS public loader."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


RESIZE_SHAPE = (128, 128, 32, 12)
ARRAY_NAMES = (
    "trn_dat.npy", "val_dat.npy", "tst_dat.npy",
    "trn_seg.npy", "val_seg.npy", "tst_seg.npy",
)


def process_patient(patient_dir, resize_shape=RESIZE_SHAPE):
    patient_dir = Path(patient_dir)
    patient_id = int(patient_dir.name.removeprefix("patient"))
    image_path = patient_dir / f"patient{patient_id:03d}_4d.nii.gz"
    info = {}
    for line in (patient_dir / "Info.cfg").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if key in {"ED", "ES"}:
                info[key] = int(value.strip())
    ed_idx, es_idx = info["ED"], info["ES"]

    image = nib.load(str(image_path)).get_fdata()[..., ed_idx - 1:es_idx]
    resized_image = zoom(image, zoom=np.divide(resize_shape, image.shape))

    seg_paths = glob(str(patient_dir / f"patient{patient_id:03d}_*_gt.nii.gz"))
    if len(seg_paths) < 2:
        raise ValueError(f"Expected two phase segmentations for patient{patient_id:03d}")
    segs = [nib.load(path).get_fdata() for path in seg_paths]
    resized_segs = [
        zoom(seg, zoom=np.divide(resize_shape[:3], seg.shape))
        for seg in segs[:2]
    ]
    return patient_id - 1, resized_image, np.stack(resized_segs, axis=-1), {
        "patient_id": patient_id,
        "ed_index_1based": ed_idx,
        "es_index_1based": es_idx,
        "source_frames": int(image.shape[-1]),
    }


def split_indices(patient_count, val_split=0):
    if patient_count != 150:
        raise ValueError(f"CRONOS ACDC protocol expects 150 patients, got {patient_count}")
    if val_split not in range(5):
        raise ValueError("val_split must be in [0, 4]")
    permutation = np.random.RandomState(0).permutation(100)
    intermediate = np.concatenate((permutation[:90], permutation[90:]))
    indices = np.arange(100)
    train = intermediate[indices % 5 != val_split]
    val = intermediate[indices % 5 == val_split]
    test = np.arange(100, 150)
    return train.tolist(), val.tolist(), test.tolist()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_array(path, value):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def prepare(dataset_root, output_dir, manifest_path, workers=8, val_split=0):
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ARRAY_NAMES):
        raise FileExistsError(f"Refusing to overwrite existing ACDC arrays in {output_dir}")

    patient_dirs = sorted(
        (Path(path) for path in glob(str(dataset_root / "*" / "patient*"))),
        key=lambda path: int(path.name.removeprefix("patient")),
    )
    if len(patient_dirs) != 150:
        raise ValueError(f"Expected 150 ACDC patient folders, found {len(patient_dirs)}")

    all_images = np.zeros((150,) + RESIZE_SHAPE, dtype=np.float32)
    all_segs = np.zeros((150,) + RESIZE_SHAPE[:3] + (2,), dtype=np.float32)
    patient_metadata = [None] * 150
    with ProcessPoolExecutor(max_workers=workers) as pool:
        work = pool.map(process_patient, map(str, patient_dirs))
        for index, image, seg, metadata in tqdm(work, total=len(patient_dirs)):
            if image.shape != RESIZE_SHAPE:
                raise ValueError(f"{metadata['patient_id']}: unexpected image shape {image.shape}")
            if seg.shape != RESIZE_SHAPE[:3] + (2,):
                raise ValueError(f"{metadata['patient_id']}: unexpected seg shape {seg.shape}")
            all_images[index] = image
            all_segs[index] = seg
            patient_metadata[index] = metadata

    all_images = np.transpose(all_images, (0, 4, 1, 2, 3))
    all_images -= np.amin(all_images, axis=(2, 3, 4), keepdims=True)
    all_images /= np.amax(all_images, axis=(2, 3, 4), keepdims=True)

    trn_indices, val_indices, tst_indices = split_indices(150, val_split)
    rng = np.random.RandomState(0)
    permutation = rng.permutation(100)
    trn_intermediate, val_intermediate = permutation[:90], permutation[90:]
    splits = {
        "trn_dat.npy": all_images[trn_intermediate[:90]],
        "val_dat.npy": all_images[val_intermediate],
        "tst_dat.npy": all_images[100:],
        "trn_seg.npy": all_segs[trn_intermediate[:90]],
        "val_seg.npy": all_segs[val_intermediate],
        "tst_seg.npy": all_segs[100:],
    }
    for name, array in splits.items():
        _save_array(output_dir / name, array)

    metadata = {
        "dataset": "ACDC",
        "raw_root": str(dataset_root),
        "preprocessing": {
            "crop": "[..., ED-1:ES]",
            "resize_shape_xyzt": RESIZE_SHAPE,
            "interpolation": "scipy.ndimage.zoom default cubic interpolation",
            "normalization": "per-frame spatial min-max to [0,1]",
            "prepare_seed": 0,
            "effective_validation_fold": val_split,
            "workers": workers,
        },
        "split_indices_zero_based": {
            "train": trn_indices,
            "validation": val_indices,
            "test": tst_indices,
        },
        "split_patient_ids_one_based": {
            "train": [i + 1 for i in trn_indices],
            "validation": [i + 1 for i in val_indices],
            "test": [i + 1 for i in tst_indices],
        },
        "intermediate_train_ids_one_based": [int(i) + 1 for i in trn_intermediate],
        "intermediate_validation_ids_one_based": [int(i) + 1 for i in val_intermediate],
        "patient_phase_metadata": patient_metadata,
        "arrays": {
            name: {
                "path": str(output_dir / name),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "size_bytes": int((output_dir / name).stat().st_size),
            }
            for name, array in splits.items()
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/Datasets/ACDC/database"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/Datasets/ACDC"))
    parser.add_argument("--manifest", type=Path, default=Path("data_manifest.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-split", type=int, default=0)
    args = parser.parse_args()
    result = prepare(args.dataset_root, args.output_dir, args.manifest, args.workers, args.val_split)
    print(json.dumps({"arrays": result["arrays"], "split_patient_ids_one_based": result["split_patient_ids_one_based"]}, indent=2))


if __name__ == "__main__":
    main()
