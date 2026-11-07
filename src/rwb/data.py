"""Load the competition files into dense route x time arrays."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
STEP = pd.Timedelta(minutes=30)
HORIZON = 8
STATUS_COLS = [f"status_{i}" for i in range(1, 7)]


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
    if cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        return Panel(pd.DatetimeIndex(z["times"]), z["routes"], z["y"], z["s"], test)

    train = pd.read_parquet(RAW / "train_solo_track.parquet")
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
        np.savez(cache_path, times=times.to_numpy(), routes=routes, y=y, s=s)
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
