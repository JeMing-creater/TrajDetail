"""Paths for data acquired separately and for reproducible experiment outputs."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TRAJDETAIL_DATA", ROOT.parent / "TrajDetail_data")).resolve()
SPLITS_ROOT = ROOT / "configs/splits"
PROTOCOL = ROOT / "configs/protocol"
RESULTS = ROOT / "results"
