"""Load the competition files into dense route x time arrays."""

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
STEP = pd.Timedelta(minutes=30)
HORIZON = 8
STATUS_COLS = [f"status_{i}" for i in range(1, 7)]
CACHE_VERSION = 1
log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Panel:
    times: pd.DatetimeIndex  # (T,)
    routes: np.ndarray  # (R,)
    y: np.ndarray  # (R, T) target_1h
    s: np.ndarray  # (6, R, T) statuses
    test: pd.DataFrame  # id, route_id, timestamp

    @property
    def n_routes(self) -> int:
        return self.y.shape[0]

    @property
    def n_times(self) -> int:
        return self.y.shape[1]


def load_panel(cache: bool = True) -> Panel:
    cache_path = ROOT / "data/processed/panel.npz"
    test = pd.read_parquet(RAW / "test_solo_track.parquet")
    train_path = RAW / "train_solo_track.parquet"
    train_digest = _sha256(train_path) if cache else None
    if cache and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as z:
                if z["cache_version"].item() == CACHE_VERSION and z["train_sha256"].item() == train_digest:
                    times, routes, y, s = pd.DatetimeIndex(z["times"]), z["routes"], z["y"], z["s"]
                    if y.shape == (len(routes), len(times)) and s.shape == (6, len(routes), len(times)):
                        return Panel(times, routes, y, s, test)
            log.info("Panel cache is outdated; rebuilding from raw data")
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            log.warning("Panel cache is unreadable or has no source fingerprint; rebuilding")

    train = pd.read_parquet(train_path)
    routes = np.sort(train["route_id"].unique())
    times = pd.date_range(train["timestamp"].min(), train["timestamp"].max(), freq=STEP)
    r_idx = np.searchsorted(routes, train["route_id"].to_numpy())
    t_idx = ((train["timestamp"] - times[0]) / STEP).to_numpy().astype(np.int64)
    # The grid is complete (checked by scripts/profile_data.py); NaN marks any gap anyway.
    y = np.full((len(routes), len(times)), np.nan, dtype=np.float32)
    y[r_idx, t_idx] = train["target_1h"].to_numpy(np.float32)
    s = np.full((6, len(routes), len(times)), np.nan, dtype=np.float32)
    for k, col in enumerate(STATUS_COLS):
        s[k, r_idx, t_idx] = train[col].to_numpy(np.float32)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Replace atomically: an interrupted writer must not expose a partial cache.
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, prefix="panel_", suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            np.savez(temporary, times=times.to_numpy(), routes=routes, y=y, s=s,
                     cache_version=CACHE_VERSION, train_sha256=train_digest)
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
    return Panel(times, routes, y, s, test)


def wape_rbias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    denom = y_true.sum()
    if denom <= 0:
        raise ValueError("Sum of y_true must be positive")
    return float(np.abs(y_pred - y_true).sum() / denom + abs(y_pred.sum() / denom - 1))


def make_submission(panel: Panel, pred: np.ndarray) -> pd.DataFrame:
    """pred: (R, HORIZON) forecasts from the last train timestamp."""
    last = panel.times[-1]
    steps = ((panel.test["timestamp"] - last) / STEP).to_numpy().astype(int)
    r = np.searchsorted(panel.routes, panel.test["route_id"].to_numpy())
    if not ((steps >= 1) & (steps <= HORIZON)).all() or not (panel.routes[r] == panel.test["route_id"].to_numpy()).all():
        raise ValueError("Test rows do not match the forecast grid")
    out = pd.DataFrame({"id": panel.test["id"].to_numpy(), "y_pred": np.clip(pred[r, steps - 1], 0, None)})
    out = out.sort_values("id").reset_index(drop=True)
    assert out["id"].is_unique and np.isfinite(out["y_pred"]).all()
    return out
