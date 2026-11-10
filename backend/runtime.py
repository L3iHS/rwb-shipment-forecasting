"""Disk-backed serving arrays; use the project's unchanged feature functions in small route batches."""

import json
import mmap
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("RWB_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from rwb.data import Panel, load_panel
from rwb.features import Cumsums, cumsums

PREPARED = Path(os.environ.get("RWB_PREPARED", ROOT / "prepared"))


def mapped_cumsums(panel: Panel, directory: Path, batch: int = 32) -> Cumsums:
    """Identical per-route sums, backed by reclaimable disk pages rather than a second full RAM copy."""
    directory.mkdir(parents=True, exist_ok=True)
    r, t = panel.y.shape
    shapes = {"y": (r, t + 1), "y2": (r, t + 1), "s": (6, r, t + 1), "x": (r, t), "xc": (r, t + 1)}
    arrays = {k: np.lib.format.open_memmap(directory / f"{k}.npy", mode="w+", shape=shape,
                                          dtype="float32" if k == "x" else "float64") for k, shape in shapes.items()}
    for start in range(0, r, batch):
        end = min(start + batch, r)
        part = Panel(panel.times, panel.routes[start:end], panel.y[start:end], panel.s[:, start:end], panel.test)
        values = cumsums(part)
        for name, array in arrays.items():
            if name == "s":
                array[:, start:end] = getattr(values, name)
            else:
                array[start:end] = getattr(values, name)
            array.flush()
            if hasattr(array._mmap, "madvise"):
                array._mmap.madvise(mmap.MADV_DONTNEED)
    return Cumsums(**arrays)


def prepare(directory: Path = PREPARED) -> None:
    """Run during image build, never on the small production server."""
    directory.mkdir(parents=True, exist_ok=True)
    panel = load_panel(cache=False)
    for name, values in {"times": panel.times.to_numpy(), "routes": panel.routes, "y": panel.y, "s": panel.s}.items():
        np.save(directory / f"panel_{name}.npy", values, allow_pickle=False)
    mapped_cumsums(panel, directory / "cums")
    (directory / "info.json").write_text(json.dumps({"routes": panel.n_routes, "times": panel.n_times,
                                                     "last_observation": str(panel.times[-1])}) + "\n")


def load(directory: Path = PREPARED) -> tuple[Panel, Cumsums]:
    read = lambda path: np.load(path, mmap_mode="r", allow_pickle=False)
    panel = Panel(pd.DatetimeIndex(read(directory / "panel_times.npy")), read(directory / "panel_routes.npy"),
                  read(directory / "panel_y.npy"), read(directory / "panel_s.npy"), pd.DataFrame())
    sums = Cumsums(**{k: read(directory / "cums" / f"{k}.npy") for k in ("y", "y2", "s", "x", "xc")})
    return panel, sums


if __name__ == "__main__":
    prepare()
