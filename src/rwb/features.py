"""Feature matrices for direct multi-horizon forecasting.

Every feature for origin index ``o`` uses only columns ``<= o`` of the panel, so
the same builder serves training, backtests and the final test forecast.
Rows are ordered origin-major: row ``i * R + r`` is origin ``origins[i]``, route ``r``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rwb.data import STEP, Panel

DAY = 48
WEEK = 7 * DAY
LAGS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 36]
ROLL = [2, 4, 8, 16, 24, 48, 96, 336, 672]
STATUS_ROLL = [1, 2, 4, 8, 48]
MAX_LOOKBACK = 4 * WEEK + DAY


@dataclass
class Cumsums:
    y: np.ndarray  # (R, T+1)
    y2: np.ndarray  # (R, T+1)
    s: np.ndarray  # (6, R, T+1)
    x: np.ndarray  # (R, T) reconstructed half-hour shipments
    xc: np.ndarray  # (R, T+1) cumsum of x


def half_hour_shipments(y: np.ndarray, window: int = WEEK) -> np.ndarray:
    """Recover 30-minute volumes x from overlapping hourly sums y[t] = x[t] + x[t-1].

    The alternating sum A gives x[t] = A[t] + (-1)^t c with one unknown constant per route.
    x >= 0 bounds c from both sides and in this data the bounds meet within a few units,
    so c is the midpoint of the bounds over a trailing window (causal, tolerant to rare glitches).
    """

    y = np.nan_to_num(y.astype(np.float64))
    T = y.shape[1]
    A = np.empty_like(y)
    A[:, 0] = y[:, 0]
    for t in range(1, T):
        A[:, t] = y[:, t] - A[:, t - 1]
    sgn = np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    lo_src = pd.DataFrame(np.where(sgn > 0, -A, np.nan).T)
    hi_src = pd.DataFrame(np.where(sgn < 0, A, np.nan).T)
    lo = lo_src.rolling(window, min_periods=1).max().ffill().to_numpy().T
    hi = hi_src.rolling(window, min_periods=1).min().ffill().to_numpy().T
    # Before both bounds exist, fall back to whichever side is known.
    c = np.where(np.isnan(lo), hi, np.where(np.isnan(hi), lo, 0.5 * (lo + hi)))
    c = np.nan_to_num(c)
    x = A + sgn[None, :] * c
    return np.clip(x, 0, None).astype(np.float32)


def cumsums(panel: Panel) -> Cumsums:
    def cs(a: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(a.astype(np.float64))
        pad = np.zeros(a.shape[:-1] + (1,))
        return np.concatenate([pad, np.cumsum(a, axis=-1)], axis=-1)

    x = half_hour_shipments(panel.y)
    return Cumsums(cs(panel.y), cs(panel.y.astype(np.float64) ** 2), cs(panel.s), x, cs(x))


def _flat(a: np.ndarray) -> np.ndarray:
    """(R, n_o) -> (n_o * R,) in origin-major order."""
    return np.ascontiguousarray(a.T).ravel()


def _roll_mean(c: np.ndarray, origins: np.ndarray, w: int) -> np.ndarray:
    return (c[..., origins + 1] - c[..., origins + 1 - w]) / w


def time_parts(panel: Panel, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Target times of the test forecast lie past the last train timestamp, so extend the grid.
    ts = pd.DatetimeIndex(panel.times[0] + STEP * np.asarray(idx))
    slot = (ts.hour * 2 + ts.minute // 30).to_numpy()
    return slot, ts.day_of_week.to_numpy()


def base_forecast(panel: Panel, cums: Cumsums, origins: np.ndarray, h: int) -> np.ndarray:
    """Seasonal baseline (R, n_o): blend of the last day level and the same slot over a week.

    The model learns a correction on top of it, which keeps the target scale-free across routes.
    """
    y = panel.y
    t = origins + h
    slot7 = np.mean([y[:, t - k * DAY] for k in range(1, 8)], axis=0)
    rm48 = _roll_mean(cums.y, origins, DAY)
    return 0.5 * rm48 + 0.5 * slot7


def build(panel: Panel, cums: Cumsums, origins: np.ndarray, h: int,
          global_feats: bool = False, flow_feats: bool = False) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return X (n_o*R, F) float32, feature names and base (n_o*R,) for horizon ``h``."""
    if origins.min() < MAX_LOOKBACK:
        raise ValueError("Origin too early for the lookback window")
    y = panel.y
    R = panel.n_routes
    t = origins + h
    cols: dict[str, np.ndarray] = {}

    base = base_forecast(panel, cums, origins, h)
    scale = _roll_mean(cums.y, origins, WEEK) + 1.0

    cols["route"] = np.repeat(np.arange(R)[:, None], len(origins), axis=1).astype(np.float64)
    slot_o, dow_o = time_parts(panel, origins)
    slot_t, dow_t = time_parts(panel, t)
    for name, v in [("slot_o", slot_o), ("dow_o", dow_o), ("slot_t", slot_t), ("dow_t", dow_t)]:
        cols[name] = np.broadcast_to(v[None, :].astype(np.float64), (R, len(origins)))
    cols["log_scale"] = np.log(scale)

    # Everything below is divided by the route scale so trees can share splits across routes.
    for k in LAGS:
        cols[f"lag_{k}"] = y[:, origins - k] / scale
    cols["d1"] = (y[:, origins] - y[:, origins - 1]) / scale
    cols["d2"] = (y[:, origins] - y[:, origins - 2]) / scale
    for w in ROLL:
        m = _roll_mean(cums.y, origins, w)
        cols[f"rm_{w}"] = m / scale
        if w >= 4:
            m2 = _roll_mean(cums.y2, origins, w)
            cols[f"rs_{w}"] = np.sqrt(np.maximum(m2 - m * m, 0)) / scale

    # Same future slot in previous days and weeks.
    for k in range(1, 8):
        cols[f"sd_{k}"] = y[:, t - k * DAY] / scale
    for k in range(1, 5):
        cols[f"sw_{k}"] = y[:, t - k * WEEK] / scale
    same_slot = np.stack([y[:, t - k * DAY] for k in range(1, 15)])
    cols["slot7"] = same_slot[:7].mean(axis=0) / scale
    # Medians resist single bursty or empty half-hours better than means.
    cols["slot7_med"] = np.median(same_slot[:7], axis=0) / scale
    cols["slot14_med"] = np.median(same_slot, axis=0) / scale
    cols["slot7_q25"] = np.quantile(same_slot[:7], 0.25, axis=0) / scale
    cols["slot7_q75"] = np.quantile(same_slot[:7], 0.75, axis=0) / scale
    recent = np.stack([y[:, origins - k] for k in range(48)])
    cols["rmed_8"] = np.median(recent[:8], axis=0) / scale
    cols["rmed_48"] = np.median(recent, axis=0) / scale
    cols["slotw4"] = np.mean([y[:, t - k * WEEK] for k in range(1, 5)], axis=0) / scale
    # Near-future window around the target slot yesterday / last week (smoother than one point).
    for back, nm in [(DAY, "yday"), (WEEK, "lweek")]:
        cols[f"{nm}_win"] = _roll_mean(cums.y, t - back + 2, 5) / scale
        # How today's recent level compares to the same hours back then.
        recent = _roll_mean(cums.y, origins, 8)
        past = _roll_mean(cums.y, origins - back, 8)
        cols[f"ratio_{nm}"] = (recent + 1) / (past + 1)
        cols[f"{nm}_trend"] = (y[:, t - back] - y[:, origins - back]) / scale

    # Half-hour shipments: y[t] = x[t] + x[t-1], so y[o+1] already contains the known x[o].
    x = cums.x
    for k in range(9):
        cols[f"x_lag_{k}"] = x[:, origins - k] / scale
    for w in [4, 8, 48]:
        cols[f"x_rm_{w}"] = _roll_mean(cums.xc, origins, w) / scale
    cols["x_known"] = (x[:, origins] if h == 1 else np.zeros_like(scale)) / scale
    for part, tt in [("a", t), ("b", t - 1)]:
        cols[f"x{part}_sd1"] = x[:, tt - DAY] / scale
        cols[f"x{part}_sw1"] = x[:, tt - WEEK] / scale
        cols[f"x{part}_slot7"] = np.mean([x[:, tt - k * DAY] for k in range(1, 8)], axis=0) / scale
        cols[f"x{part}_slotw4"] = np.mean([x[:, tt - k * WEEK] for k in range(1, 5)], axis=0) / scale

    sc = cums.s
    for k in range(6):
        for w in STATUS_ROLL:
            cols[f"s{k + 1}_r{w}"] = _roll_mean(sc[k], origins, w) / scale
    for grp, ks in [("cur", [0, 1, 2]), ("prev", [3, 4, 5])]:
        tot = sum(_roll_mean(sc[k], origins, 4) for k in ks)
        cols[f"s_{grp}_r4"] = tot / scale
        tot_day = sum(_roll_mean(sc[k], origins - DAY, 4) for k in ks)
        cols[f"s_{grp}_ratio_yday"] = (tot + 1) / (tot_day + 1)

    if global_feats:
        # Network-wide level shocks (all routes summed): a busy or slow day shows up everywhere.
        gc = cums.y.sum(axis=0)

        def g(o: np.ndarray, w: int) -> np.ndarray:
            return (gc[o + 1] - gc[o + 1 - w]) / w

        for w in [4, 8, 16]:
            for back, nm in [(DAY, "yday"), (WEEK, "lweek")]:
                v = g(origins, w) / g(origins - back, w)
                cols[f"g_{nm}_r{w}"] = np.broadcast_to(v[None, :], (R, len(origins)))
        v = g(origins, DAY) / g(origins - DAY, WEEK)
        cols["g_day_vs_week"] = np.broadcast_to(v[None, :], (R, len(origins)))

    if flow_feats:
        # Backlog: items that passed a stage minus volume already shipped, in volume units.
        # Statuses count pieces while the target is volume, so convert with last week's ratio.
        xw = _roll_mean(cums.xc, origins, WEEK)
        for k in range(5):
            ratio = xw / (_roll_mean(sc[k], origins, WEEK) + 1e-3)
            for w in [4, 8, 16, 48, 96]:
                flow = ratio * _roll_mean(sc[k], origins, w) - _roll_mean(cums.xc, origins, w)
                cols[f"bl{k + 1}_{w}"] = w * flow / scale

    cols["base"] = base / scale
    names = list(cols)
    X = np.empty((len(origins) * R, len(names)), dtype=np.float32)
    for j, nm in enumerate(names):
        X[:, j] = _flat(np.asarray(cols[nm]))
    return X, names, _flat(base)


def flat_scale(cums: Cumsums, origins: np.ndarray) -> np.ndarray:
    return _flat(_roll_mean(cums.y, origins, WEEK) + 1.0)


def flat_target(panel: Panel, origins: np.ndarray, h: int) -> np.ndarray:
    return _flat(panel.y[:, origins + h].astype(np.float64))
