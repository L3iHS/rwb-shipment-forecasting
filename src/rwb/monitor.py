"""Quality control of the running forecast: daily metrics, alerts and a data fallback.

The leaderboard taught the main lesson: one day's total can be off by 1-3% while point accuracy
is unchanged, and that alone moves the score more than any model change. So the level of the
forecast (signed bias) is tracked separately from point accuracy, and drift raises an alert
before a planner notices too few trucks.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rwb.data import Panel
from rwb.features import DAY, MAX_LOOKBACK, Cumsums
from rwb.models import naive


@dataclass
class Thresholds:
    bias: float = 0.02  # |mean signed bias| over the window that calls for a level recalibration
    window: int = 3  # days
    wape_ratio: float = 1.15  # day WAPE above its trailing median by this factor
    coverage_min: float = 0.7  # for the nominal 80% band


class DataUnavailableError(ValueError):
    """Input history cannot support a reliable model or seasonal fallback."""


def daily_metrics(pred: np.ndarray, truth: np.ndarray, times: pd.DatetimeIndex,
                  lo: np.ndarray | None = None, hi: np.ndarray | None = None) -> pd.DataFrame:
    """One row per forecast origin: WAPE, signed bias, score and (if given) band coverage."""
    s = truth.sum(axis=(1, 2))
    df = pd.DataFrame({
        "origin": times,
        "wape": np.abs(pred - truth).sum(axis=(1, 2)) / s,
        "bias": pred.sum(axis=(1, 2)) / s - 1,
    })
    df["score"] = df["wape"] + df["bias"].abs()
    if lo is not None and hi is not None:
        df["coverage"] = ((truth >= lo) & (truth <= hi)).mean(axis=(1, 2))
    return df


def alerts(df: pd.DataFrame, th: Thresholds | None = None) -> pd.DataFrame:
    """Rules a duty analyst would act on; each row names the day, the signal and the action."""
    th = th or Thresholds()
    out = []
    bias_avg = df["bias"].rolling(th.window, min_periods=th.window).mean()
    wape_ref = df["wape"].shift(1).rolling(7, min_periods=3).median()
    for i, r in df.iterrows():
        if abs(bias_avg.iloc[i]) > th.bias:
            side = "занижает" if bias_avg.iloc[i] < 0 else "завышает"
            out.append((r["origin"], "уровень", f"{th.window} дня подряд {side} в среднем на {abs(bias_avg.iloc[i]):.1%}",
                        "перекалибровать множитель уровня"))
        if np.isfinite(wape_ref.iloc[i]) and r["wape"] > th.wape_ratio * wape_ref.iloc[i]:
            out.append((r["origin"], "точность", f"WAPE {r['wape']:.3f} против обычных {wape_ref.iloc[i]:.3f}",
                        "проверить данные за день, при повторе переобучить"))
        if "coverage" in df and r["coverage"] < th.coverage_min:
            out.append((r["origin"], "диапазоны", f"в 80%-диапазон попало {r['coverage']:.0%}", "пересчитать диапазоны"))
    return pd.DataFrame(out, columns=["origin", "signal", "detail", "action"])


def stale_routes(panel: Panel, origin: int, hours: int = 4) -> np.ndarray:
    """Check the full model lookback, including statuses; zero is a valid observation."""
    if origin < MAX_LOOKBACK:
        raise DataUnavailableError(f"Need at least {MAX_LOOKBACK + 1} history points")
    start = origin - max(MAX_LOOKBACK, 2 * hours)
    y, s = panel.y[:, start:origin + 1], panel.s[:, :, start:origin + 1]
    return (~np.isfinite(y) | (y < 0)).any(axis=1) | (~np.isfinite(s) | (s < 0)).any(axis=(0, 2))


def with_fallback(pred: np.ndarray, panel: Panel, cums: Cumsums, origin: int) -> tuple[np.ndarray, np.ndarray]:
    """Replace forecasts of routes with missing recent data by the seasonal baseline.

    pred: (R, H) model forecast from ``origin``. Returns the patched forecast and the route mask.
    """
    bad = stale_routes(panel, origin)
    if bad.any():
        recent = panel.y[:, origin - DAY + 1:origin + 1]
        slots = np.array([origin + np.arange(1, 9) - k * DAY for k in range(1, 8)])
        seasonal = panel.y[:, slots]
        usable = (np.isfinite(recent) & (recent >= 0)).all(axis=1) & (np.isfinite(seasonal) & (seasonal >= 0)).all(axis=(1, 2))
        if (bad & ~usable).any():
            routes = panel.routes[bad & ~usable].tolist()
            raise DataUnavailableError(f"No trustworthy seasonal fallback for routes {routes[:10]}")
        base = naive(panel, np.array([origin]), "base", cums)[0]
        pred = np.where(bad[:, None], base, pred)
    return pred, bad
