"""Atomic records and fold-level verification helpers."""
import csv
import json
import os
from pathlib import Path

from src.paths import ROOT

OUTPUT = ROOT / "results"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", newline="") as stream:
        cols = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def checkpoint(path):
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
