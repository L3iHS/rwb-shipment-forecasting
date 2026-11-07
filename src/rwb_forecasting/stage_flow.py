"""Fixed paired Ridge adjustment from three observed warehouse-stage flows.

The extra signal is a trailing-centered cumulative flow difference, not a
measured queue inventory. The H1 floor is the causal feasible-half midpoint,
not an independently observed half-hour shipment. All horizon totals of the
supplied anchor are retained; its route-level H1 unknown part may be adjusted.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .halfhour import halfhour_bounds

_VOLUME_LAGS = (0, 1, 2, 3, 4, 6, 8, 12)
_STATUS_LAGS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12)
_STATUS_COLUMNS = ("status_1", "status_2", "status_3")


def _cumulative(values):
    result = np.empty((len(values) + 1, *values.shape[1:]), dtype=np.float64)
    result[0] = 0
    np.cumsum(values, axis=0, out=result[1:])
    return result


class StageFlowPanel:
    """A complete prefix, in time-by-route order, ending at the origin."""

    def __init__(self, timestamps, hourly, statuses, route_ids):
        self.times = pd.DatetimeIndex(timestamps).as_unit("ns")
        self.hourly = np.asarray(hourly, dtype=np.float64)
        self.statuses = np.asarray(statuses, dtype=np.float64)
        self.routes = np.asarray(route_ids)
        if (self.hourly.ndim != 2 or 0 in self.hourly.shape
                or self.statuses.shape != (*self.hourly.shape, 3)
                or len(self.times) != len(self.hourly)
                or self.routes.ndim != 1 or len(self.routes) != self.hourly.shape[1]
                or pd.Index(self.routes).hasnans or pd.Index(self.routes).has_duplicates
                or not np.array_equal(self.routes, np.sort(self.routes))
                or not np.isfinite(self.statuses).all() or (self.statuses < 0).any()):
            raise ValueError("Expected a nonempty complete time/route/three-stage panel")
        if (self.times.tz is not None or self.times.hasnans
                or not self.times.equals(pd.date_range(self.times[0], periods=len(self.times), freq="30min").as_unit("ns"))
                or self.times[0] != self.times[0].floor("30min")):
            raise ValueError("Expected an aligned, regular, timezone-naive 30-minute grid")
        lower, upper = halfhour_bounds(self.hourly, 0.5)
        self.half_hour = (lower + upper) / 2
        self.slots = (self.times.hour * 2 + self.times.minute // 30).to_numpy()
        self.weekdays = self.times.day_of_week.to_numpy()
        self.c_hourly = _cumulative(self.hourly)
        self.c_half = _cumulative(self.half_hour)
        self.c_status = _cumulative(self.statuses)

        balance = np.cumsum(self.statuses[:, :, :2] - self.statuses[:, :, 1:], axis=0)
        cumulative_balance = _cumulative(balance)
        self.excess = np.zeros_like(balance)
        self.excess[47:] = balance[47:] - (cumulative_balance[48:] - cumulative_balance[:-48]) / 48
        self.c_excess = _cumulative(self.excess)

    def features(self, sources):
        """Return the fixed 73 base columns followed by six centered-flow columns."""
        t = np.asarray(sources)
        if (t.ndim != 1 or len(t) == 0 or t.dtype.kind not in "iu"
                or t.min() < 336 or t.max() >= len(self.times)):
            raise ValueError("Feature source outside the available prefix")

        def mean(cumulative, length):
            return (cumulative[t + 1] - cumulative[t + 1 - length]) / length

        scale = np.maximum(mean(self.c_hourly, 336), 1000.)
        unit = np.maximum(mean(self.c_status, 336), 1.)
        columns = [values[t - lag] / scale
                   for values in (self.hourly, self.half_hour) for lag in _VOLUME_LAGS]
        columns.extend((self.statuses[t - lag] / unit)[:, :, j]
                       for lag in _STATUS_LAGS for j in range(3))
        for length in (6, 12, 24):
            columns.extend([mean(self.c_hourly, length) / scale, mean(self.c_half, length) / scale])
            average_status = mean(self.c_status, length) / unit
            columns.extend(average_status[:, :, j] for j in range(3))
        for horizon in range(1, 9):
            columns.append(sum(self.hourly[t + horizon - 48 * day] for day in range(1, 8)) / (7 * scale))
        for phase in (self.slots[t] / 48, self.weekdays[t] / 7):
            columns.extend(np.broadcast_to(value[:, None], scale.shape)
                           for value in (np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)))
        base = np.stack(columns, axis=-1)
        flow_unit = np.maximum((self.c_status[t + 1, :, :2] - self.c_status[t + 1 - 336, :, :2]) / 336, 1.)
        extras = [self.excess[t] / flow_unit, self.excess[t - 1] / flow_unit,
                  (self.c_excess[t + 1] - self.c_excess[t - 5]) / (6 * flow_unit)]
        return np.concatenate([base, *extras], axis=-1), scale

    def labels(self, sources, scale, cutoff):
        """Normalized raw hourly labels, with the source half removed from H1."""
        t = np.asarray(sources)
        if (t.ndim != 1 or len(t) == 0 or t.dtype.kind not in "iu"
                or t.min() < 0 or cutoff >= len(self.times) or cutoff < 0
                or t.max() + 8 > cutoff or scale.shape != (len(t), len(self.routes))):
            raise ValueError("Fit label after cutoff or outside the prefix")
        labels = np.stack([self.hourly[t + horizon] for horizon in range(1, 9)], axis=-1)
        labels[:, :, 0] -= self.half_hour[t]
        return labels / scale[:, :, None]


def load_stage_flow_panel(train_path: str | Path, origin, route_ids) -> StageFlowPanel:
    """Read only the observed prefix, in bounded weekly chunks, from raw Parquet."""
    origin = pd.Timestamp(origin)
    routes = np.asarray(route_ids)
    if origin.tz is not None or pd.isna(origin) or origin != origin.floor("30min"):
        raise ValueError("Origin must be an aligned timezone-naive timestamp")
    observed = pd.read_parquet(train_path, columns=["timestamp"],
                               filters=[("timestamp", "<=", origin.to_pydatetime())], use_threads=False)
    if observed.empty or observed.timestamp.isna().any() or pd.Timestamp(observed.timestamp.max()) != origin:
        raise ValueError("Origin must be the final timestamp of the loaded observed prefix")
    start = pd.Timestamp(observed.timestamp.min())
    del observed
    times = pd.date_range(start, origin, freq="30min").as_unit("ns")
    hourly = np.empty((len(times), len(routes)), dtype=np.float64)
    statuses = np.empty((len(times), len(routes), 3), dtype=np.float64)
    for first in range(0, len(times), 7 * 48):
        window = times[first:first + 7 * 48]
        frame = pd.read_parquet(train_path, columns=["timestamp", "route_id", "target_1h", *_STATUS_COLUMNS],
                                filters=[("timestamp", ">=", window[0].to_pydatetime()),
                                         ("timestamp", "<=", window[-1].to_pydatetime())], use_threads=False)
        frame = frame.sort_values(["timestamp", "route_id"])
        if (len(frame) != len(window) * len(routes)
                or not np.array_equal(frame.timestamp.to_numpy(dtype="datetime64[ns]"), np.repeat(window.to_numpy(), len(routes)))
                or not np.array_equal(frame.route_id.to_numpy(), np.tile(routes, len(window)))):
            raise ValueError("Train must have exactly one row per requested route and observed half hour")
        hourly[first:first + len(window)] = frame.target_1h.to_numpy().reshape(-1, len(routes))
        for j, column in enumerate(_STATUS_COLUMNS):
            statuses[first:first + len(window), :, j] = frame[column].to_numpy().reshape(-1, len(routes))
    del frame
    return StageFlowPanel(times, hourly, statuses, routes)


def _fit_pair(panel, sources, cutoff):
    """Weighted, route-centered Ridge; both heads share the same training moments."""
    sources = np.asarray(sources)
    if (sources.ndim != 1 or sources.dtype.kind not in "iu" or len(sources) < 2
            or sources.min() < 336 or sources.max() + 8 > cutoff or cutoff >= len(panel.times)):
        raise ValueError("Training sources or labels outside the available cutoff")
    routes = len(panel.routes)
    sx = np.zeros((routes, 79)); sy = np.zeros((routes, 8)); sw = np.zeros(routes)
    gram = np.zeros((79, 79)); rhs = np.zeros((79, 8))
    for first in range(0, len(sources), 8):
        t = sources[first:first + 8]
        x, weight = panel.features(t)
        labels = panel.labels(t, weight, cutoff)
        sx += (x * weight[:, :, None]).sum(0)
        sy += (labels * weight[:, :, None]).sum(0)
        sw += weight.sum(0)
        sqrt = np.sqrt(weight.ravel())[:, None]
        a = x.reshape(-1, 79) * sqrt
        b = labels.reshape(-1, 8) * sqrt
        gram += a.T @ a
        rhs += a.T @ b
    mx = sx / sw[:, None]; my = sy / sw[:, None]
    gram -= sx.T @ mx
    rhs -= sx.T @ my
    total = sw.sum()
    row_mean = total / (len(sources) * routes)
    std = np.maximum(np.sqrt(np.maximum(gram.diagonal(), 0.) / total), 1e-7)
    gram = gram / (std[:, None] * std[None, :] * row_mean)
    rhs = rhs / (std[:, None] * row_mean)
    heads = []
    for width in (73, 79):
        coef = np.linalg.solve(gram[:width, :width] + 100 * np.eye(width), rhs[:width]) / std[:width, None]
        heads.append((coef, my - mx[:, :width] @ coef))
    return heads


def predict_stage_flow(panel: StageFlowPanel, anchor):
    """Fit the fixed 14-day morning pair and adjust an eight-horizon anchor."""
    i = len(panel.times) - 1
    base = np.asarray(anchor, dtype=np.float64)
    if (panel.slots[i] != 21 or base.shape != (len(panel.routes), 8)
            or not np.isfinite(base).all() or (base < 0).any()):
        raise ValueError("Expected a finite nonnegative route-by-eight anchor at a 10:30 origin")
    start = i - 8 - 14 * 48
    if start < 336:
        raise ValueError("Insufficient history for the fixed 336-step features and 14-day training window")
    sources = np.arange(start, i - 8 + 1, dtype=int)
    sources = sources[(panel.slots[sources] >= 16) & (panel.slots[sources] <= 26)]
    heads = _fit_pair(panel, sources, i)
    x, scale = panel.features(np.array([i]))
    predicted = []
    for coef, intercept in heads:
        values = (x[:, :, :len(coef)] @ coef + intercept[None]) * scale[:, :, None]
        values[:, :, 0] += panel.half_hour[i]
        predicted.append(np.maximum(values, 0)[0])
    basic, extra = predicted

    floor = np.zeros_like(base)
    floor[:, 0] = panel.half_hour[i]
    room = base - floor
    if room.min() < -1e-7:
        raise ValueError("Anchor H1 is below its causal feasible-half midpoint contribution")
    room = np.maximum(room, 0)
    adjusted = np.maximum(room + extra - basic, 0)
    target = room.sum(0)
    denominator = adjusted.sum(0)
    selected = denominator > 0
    out = room.copy()
    out[:, selected] = adjusted[:, selected] * (target[selected] / denominator[selected])
    out += floor
    if not np.isfinite(out).all() or (out < 0).any() or (out[:, 0] < panel.half_hour[i] - 1e-7).any():
        raise ValueError("Invalid adjusted forecast")
    np.testing.assert_allclose(out.sum(0), base.sum(0), rtol=5e-14, atol=1e-6)
    audit = dict(origin=str(panel.times[i]), first_training_source=str(panel.times[sources[0]]),
                 last_training_source=str(panel.times[sources[-1]]),
                 latest_label_timestamp=str(panel.times[sources[-1] + 8]),
                 latest_feature_timestamp=str(panel.times[i]), training_sources=len(sources),
                 base_features=73, augmented_features=79,
                 zero_column_fallbacks=int((~selected).sum()),
                 horizon_totals_max_absolute_drift=float(np.max(abs(out.sum(0) - base.sum(0)))),
                 h1_floor="Forward-only feasible-half midpoint, rounding tolerance 0.5")
    return out, audit
