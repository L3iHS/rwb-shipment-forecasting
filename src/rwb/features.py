"""Feature matrices for direct multi-horizon forecasting.

Every feature for origin index ``o`` uses only columns ``<= o`` of the panel, so
the same builder serves training, backtests and the final test forecast.
Rows are ordered origin-major: row ``i * R + r`` is origin ``origins[i]``, route ``r``.
"""

import re
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


EXTRA_SETS = ("res", "ratio", "sdiff", "lowrank", "xlowrank")
# "lowrank" and "xlowrank" take an optional rank and profile length, e.g. "lowrank12" or "lowrank12d42";
# rank 0 keeps only the raw profile.
_EXTRA_RE = re.compile(r"^(res|ratio|sdiff|(x?lowrank)(\d*)(?:d(\d+))?)$")
PROFILE_DAYS = 28
PROFILE_RANK = 6


def _day_chain_cumsum(y: np.ndarray) -> np.ndarray:
    """cs[:, t] = sum over k >= 0 of y[:, t - k*DAY]: running total along the same clock slot."""
    R, T = y.shape
    n = -(-T // DAY)
    pad = np.zeros((R, n * DAY))
    pad[:, :T] = np.nan_to_num(y.astype(np.float64))
    return pad.reshape(R, n, DAY).cumsum(axis=1).reshape(R, n * DAY)[:, :T]


def lowrank_profiles(y: np.ndarray, origins: np.ndarray, rank: int = PROFILE_RANK,
                     days: int = PROFILE_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """Next-day profile per route from the last ``days`` days, raw and rank-reduced across routes.

    Returns two (n_o, R, DAY) arrays indexed by step ahead j-1 for j = 1..DAY. The rank-reduced
    version keeps the intraday shapes shared by many routes and drops route-specific noise.
    """
    if origins.min() < days * DAY:
        raise ValueError(f"Origin too early for a {days}-day profile")
    cs = _day_chain_cumsum(y)
    j = np.arange(1, DAY + 1)
    idx = origins[:, None] + j[None, :]  # (n_o, DAY) future slots
    # cs at a negative index would wrap around; the slot k=days back is idx - days*DAY >= 1 here.
    prof = (cs[:, idx - DAY] - np.where(idx - (days + 1) * DAY >= 0, cs[:, np.maximum(idx - (days + 1) * DAY, 0)], 0.0)) / days
    prof = np.transpose(prof, (1, 0, 2)).astype(np.float32)
    if rank == 0:
        return prof, prof
    level = prof.mean(axis=2, keepdims=True) + 1.0
    u, s, vt = np.linalg.svd(prof / level, full_matrices=False)
    k = rank
    smooth = np.einsum("nrk,nk,nkd->nrd", u[:, :, :k], s[:, :k], vt[:, :k, :]) * level
    return prof, smooth


def build(panel: Panel, cums: Cumsums, origins: np.ndarray, h: int,
          global_feats: bool = False, flow_feats: bool = False,
          extra: tuple[str, ...] = ()) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return X (n_o*R, F) float32, feature names and base (n_o*R,) for horizon ``h``.

    ``extra`` switches on optional feature sets from ``EXTRA_SETS``.
    """
    unknown = [e for e in extra if not _EXTRA_RE.match(e)]
    if unknown:
        raise ValueError(f"Unknown feature sets: {sorted(unknown)}")
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

    if "res" in extra:
        # How far off the seasonal base was for this route: at the same target slot on past days,
        # and for the last few hours. Persistent level shifts show up here before the base adapts.
        errs = np.stack([y[:, t - k * DAY] - base_forecast(panel, cums, origins - k * DAY, h) for k in range(1, 8)])
        cols["res_d1"] = errs[0] / scale
        cols["res_w1"] = errs[6] / scale
        cols["res_mean7"] = errs.mean(axis=0) / scale
        cols["res_med7"] = np.median(errs, axis=0) / scale
        recent_err = np.stack([y[:, origins - j] - base_forecast(panel, cums, origins - j - h, h) for j in range(4)])
        cols["res_recent1"] = recent_err[0] / scale
        cols["res_recent4"] = recent_err.mean(axis=0) / scale

    if "ratio" in extra:
        # Today's level against yesterday / last week over longer spans than ratio_* (4 hours).
        for w in [16, 24, 48]:
            now = _roll_mean(cums.y, origins, w)
            for back, nm in [(DAY, "yday"), (WEEK, "lweek")]:
                cols[f"ratio_{nm}_{w}"] = (now + 1) / (_roll_mean(cums.y, origins - back, w) + 1)

    if "sdiff" in extra:
        # Step-level dynamics of the pipeline statuses that lead shipments, relative to their own level.
        for k in range(3):
            s_k = np.nan_to_num(panel.s[k].astype(np.float64))
            lvl = _roll_mean(sc[k], origins, WEEK) + 1.0
            for j in [1, 2, 4]:
                cols[f"s{k + 1}_diff{j}"] = (s_k[:, origins] - s_k[:, origins - j]) / lvl
            for j in [1, 2, 3]:
                cols[f"s{k + 1}_lag{j}"] = s_k[:, origins - j] / lvl

    for e in extra:
        m = _EXTRA_RE.match(e)
        if not m or not m.group(2):
            continue
        # 28-day same-slot profile (longer than slot7/slot14) and its cross-route low-rank smoothing.
        is_x = m.group(2) == "xlowrank"
        rank = int(m.group(3)) if m.group(3) else PROFILE_RANK
        days = int(m.group(4)) if m.group(4) else PROFILE_DAYS
        src, tag = (cums.x, f"xprof{days}") if is_x else (y, f"prof{days}")
        prof, smooth = lowrank_profiles(src, origins, rank, days)
        # For x the target hour is x[t] + x[t-1], i.e. steps h and h-1 of the half-hour profile.
        steps = [h - 1, h - 2] if is_x else [h - 1]
        p_h = sum(prof[:, :, j] for j in steps if j >= 0).T
        s_h = sum(smooth[:, :, j] for j in steps if j >= 0).T
        cols[tag] = p_h / scale
        if rank > 0:
            cols[f"{tag}_lr"] = s_h / scale
            cols[f"{tag}_lr_ratio"] = (s_h + 1) / (p_h + 1)

    cols["base"] = base / scale
    names = list(cols)
    X = np.empty((len(origins) * R, len(names)), dtype=np.float32)
    for j, nm in enumerate(names):
        X[:, j] = _flat(np.asarray(cols[nm]))
    return X, names, _flat(base)


def flat_scale(cums: Cumsums, origins: np.ndarray) -> np.ndarray:
    return _flat(_roll_mean(cums.y, origins, WEEK) + 1.0)


def flat_day_mean(cums: Cumsums, origins: np.ndarray) -> np.ndarray:
    """Trailing 24-hour mean of y, the level half of the seasonal base."""
    return _flat(_roll_mean(cums.y, origins, DAY))


def flat_target(panel: Panel, origins: np.ndarray, h: int) -> np.ndarray:
    return _flat(panel.y[:, origins + h].astype(np.float64))


def flat_x(cums: Cumsums, origins: np.ndarray, h: int) -> np.ndarray:
    """Reconstructed half-hour shipments x[o+h] in origin-major order."""
    return _flat(cums.x[:, origins + h].astype(np.float64))
