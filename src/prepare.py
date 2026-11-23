"""Prepare patient-disjoint two-history, one-future MRI windows."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path

from src.paths import PROTOCOL
import re

import nibabel as nib
import numpy as np
import openpyxl
from scipy.ndimage import zoom, gaussian_filter
import SimpleITK as sitk

SPLITS = ("train", "validation", "test")
SHAPE = (48, 80, 80)
SCALE_DAYS = {"mu": 1785., "dlbs": 3652.5}


def candidates(dataset, root):
    if dataset == "mu":
        book = openpyxl.load_workbook(next((root / "metadata").glob("*Clinical*.xlsx")),
                                      read_only=True, data_only=True)
        rows = iter(book["MU Glioma Post"].values)
        columns = {i: int(re.search(r"Timepoint_(\d+)", name)[1])
                   for i, name in enumerate(next(rows)) if name and "MRI (Timepoint_" in name}
        timing = {row[0]: {point: float(row[i]) for i, point in columns.items()
                           if isinstance(row[i], (int, float)) and np.isfinite(row[i])}
                  for row in rows if isinstance(row[0], str) and row[0].startswith("PatientID_")}
        book.close()
        images = {}
        inventory = json.loads((root / "metadata/download_manifest.json").read_text())
        for entry in inventory["files"]:
            relative = Path(entry["target"])
            if relative.name.endswith("_brain_t1c.nii.gz"):
                images.setdefault(relative.parts[1], []).append(root / relative)
        for patient, paths in sorted(images.items()):
            dated = {}
            for file in sorted(paths):
                point = int(file.parent.name.split("_")[-1])
                day = timing.get(patient, {}).get(point)
                if day is not None:
                    dated.setdefault(day, file)
            yield patient, sorted(dated.items()), len(paths)
    else:
        inventory_path = PROTOCOL / "dlbs_T1_inventory.json"
        inventory = json.loads(inventory_path.read_text())["T1_paths"]
        images = {}
        for relative in sorted(inventory, key=lambda p: ("_run-1_" not in p, p)):
            patient, wave, *_ = relative.split("/")
            images.setdefault(patient, {}).setdefault(int(wave[-1]), root / relative)
        with (root / "participants.tsv").open() as stream:
            times = {row["participant_id"]: row for row in csv.DictReader(stream, delimiter="\t")}
        for patient, by_wave in sorted(images.items()):
            values = []
            for wave, file in sorted(by_wave.items()):
                field = {1: None, 2: "MRIW1toW2", 3: "MRIW1toW3"}[wave]
                try:
                    year = 0. if field is None else float(times[patient][field])
                except (ValueError, TypeError, KeyError):
                    continue
                if np.isfinite(year) and (wave == 1 or year > 0):
                    values.append((year * 365.25, file))
            if len(values) != len({day for day, _ in values}) or values != sorted(values):
                values = []
            yield patient, values, len(by_wave)


def normalize(array):
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("invalid voxels")
    foreground = array[array > 0]
    if foreground.size < 5000:
        raise ValueError("empty foreground")
    high = np.percentile(foreground, 99)
    if high <= 0 or np.std(foreground) <= 1e-7:
        raise ValueError("constant MRI")
    return np.clip(array / high, 0, 1).astype(np.float32)


def mu_images(files):
    images = [nib.as_closest_canonical(nib.load(file)) for file in files]
    anchor = images[1]
    if any(im.shape != anchor.shape or not np.allclose(im.affine, anchor.affine, atol=1e-3) for im in images):
        raise ValueError("different atlas grids")
    bounds = np.argwhere(anchor.get_fdata(dtype=np.float32) > 0)
    extent = tuple(slice(max(0, lo - 6), min(limit, hi + 7))
                   for lo, hi, limit in zip(bounds.min(0), bounds.max(0), anchor.shape))
    result = []
    for image in images:
        array = normalize(image.get_fdata(dtype=np.float32)[extent].transpose(2, 1, 0))
        factor = tuple(out / old for out, old in zip(SHAPE, array.shape))
        filtered = gaussian_filter(array, [max(0., (.5 / f) - .5) for f in factor])
        result.append(zoom(filtered, factor, order=1))
    return result, {}


def correlation(fixed, moving):
    mask = (fixed > .08) & (moving > .08)
    return float(np.corrcoef(fixed[mask], moving[mask])[0, 1]) if mask.sum() > 5000 else -1.


def dlbs_images(files):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    images = []
    for file in files:
        image = sitk.DICOMOrient(sitk.ReadImage(str(file), sitk.sitkFloat32), "RAS")
        image_array = normalize(sitk.GetArrayFromImage(image))
        norm = sitk.GetImageFromArray(image_array)
        norm.CopyInformation(image)
        smoothed = sitk.SmoothingRecursiveGaussian(norm, .9)
        images.append(smoothed)
    anchor = images[1]
    direction = np.array(anchor.GetDirection()).reshape(3, 3)
    center = np.asarray(anchor.TransformContinuousIndexToPhysicalPoint(
        [(n - 1) / 2. for n in anchor.GetSize()]))
    size = tuple(reversed(SHAPE))
    spacing = (2.8, 2.8, 3.8)
    origin = center - direction @ ((np.asarray(size) - 1) * spacing / 2.)
    reference = sitk.Image(size, sitk.sitkFloat32)
    reference.SetSpacing(spacing)
    reference.SetDirection(anchor.GetDirection())
    reference.SetOrigin(origin.tolist())
    fixed = sitk.Resample(anchor, reference)
    fixed_array = sitk.GetArrayFromImage(fixed)
    result, qc = [], {}
    for index, image in enumerate(images):
        if index == 1:
            result.append(fixed_array)
            continue
        moving = sitk.Resample(image, reference)
        unaligned = sitk.GetArrayFromImage(moving)
        baseline = correlation(fixed_array, unaligned)
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(32)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(.40, seed=42)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsRegularStepGradientDescent(1.5, .02, 90)
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetShrinkFactorsPerLevel((2, 1))
        registration.SetSmoothingSigmasPerLevel((2, 0))
        initial = sitk.Euler3DTransform()
        initial.SetCenter(center.tolist())
        registration.SetInitialTransform(initial)
        transform = registration.Execute(fixed, moving)
        aligned = sitk.GetArrayFromImage(sitk.Resample(image, reference, transform))
        corrected = correlation(fixed_array, aligned)
        result.append(aligned)
        qc[str(index)] = {"ncc_before": baseline, "ncc_after": corrected,
                          "rigid_parameters": list(transform.GetParameters())}
    return result, qc


def process(task):
    dataset, patient, dated, output = task
    dates, files = zip(*dated)
    record = {"patient_id": patient, "days": dates, "files": [str(x) for x in files]}
    try:
        volumes, qc = (mu_images if dataset == "mu" else dlbs_images)(files)
        array = np.stack(volumes).astype(np.float32)
        if array.shape != (3, *SHAPE) or not np.isfinite(array).all():
            raise ValueError("invalid resampled stack")
        path = output / (patient + ".npy")
        np.save(path, array)
        record["registration"] = qc
        return record
    except Exception as exc:
        record["preprocessing_error"] = str(exc)
        return record


def prepare(dataset, raw, scratch, output, workers):
    scratch.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    jobs, excluded = [], []
    for patient, visits, n_images in candidates(dataset, raw):
        counts["source_patients"] += 1
        counts["at_least_three_scans"] += n_images >= 3
        if len(visits) >= 3:
            jobs.append((dataset, patient, visits[:3], scratch))
        else:
            excluded.append({"patient_id": patient, "image_visits": n_images, "dated_visits": len(visits)})
    counts["at_least_three_dated_scans"] = len(jobs)
    print(dataset, dict(counts), flush=True)
    patients = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(process, jobs, chunksize=1), 1):
            if "preprocessing_error" in result:
                excluded.append(result)
            else:
                patients.append(result)
            if i % 12 == 0 or i == len(jobs):
                print(f"{dataset}: {i}/{len(jobs)} ready, {len(patients)} passed", flush=True)
    order = np.random.default_rng(42).permutation(len(patients))
    n_eval = round(len(order) * .15)
    for i, index in enumerate(order):
        patients[index]["split"] = SPLITS[0 if i < len(order) - 2*n_eval else (1 if i < len(order)-n_eval else 2)]
    counts.update(patient["split"] for patient in patients)
    manifest = {"dataset": dataset, "shape_zyx": SHAPE, "seed": 42,
                "time_unit_days": SCALE_DAYS[dataset], "counts": dict(counts),
                "patients": patients, "excluded": excluded}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(dataset, dict(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SCALE_DAYS)
    parser.add_argument("raw", type=Path)
    parser.add_argument("scratch", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    prepare(args.dataset, args.raw, args.scratch, args.output, args.workers)
