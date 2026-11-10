"""Weekly direct H2–H8 route totals, reconciled to an anchor's temporal margins.

This is the fixed weekly_route_window_l1 recipe: one Saturday10:30 LightGBM400
fit, up to42days of mature sources at09:30/10:30/11:30, physical-scale weighted
L1, and105 causal features. Its route masses enter a fixed0.5 reconciliation;
H1 and the anchor's seven H2–H8 horizon totals are retained.
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from .halfhour import halfhour_bounds
from .route_reconciliation import reconcile_temporal

_COLUMNS = ["target_1h", *[f"status_{i}" for i in range(1, 7)]]
_PARAMETERS = dict(objective="regression_l1", learning_rate=.04, num_leaves=31,
                   min_data_in_leaf=120, feature_fraction=.9, bagging_fraction=.9,
                   bagging_freq=1, lambda_l2=1., seed=42, num_threads=1, verbosity=-1)


class RouteWindowPanel:
    """Complete observed integer-count prefix, without future rows."""

    def __init__(self, timestamps, counts, route_ids):
        self.times = pd.DatetimeIndex(timestamps).as_unit("ns")
        self.raw = np.asarray(counts)
        self.route_ids = np.asarray(route_ids)
        self.n = len(self.times)
        self.routes = len(self.route_ids)
        if (not self.n or not self.routes or self.raw.shape != (self.n, self.routes, 7)
                or self.raw.dtype != np.int32 or (self.raw < 0).any()
                or self.route_ids.ndim != 1 or pd.Index(self.route_ids).hasnans
                or not np.array_equal(self.route_ids, np.unique(self.route_ids))):
            raise ValueError("Expected complete nonnegative int32 hourly/status counts and sorted unique routes")
        if (self.times.tz is not None or self.times.hasnans
                or self.times[0] != self.times[0].floor("30min")
                or not self.times.equals(pd.date_range(self.times[0], periods=self.n, freq="30min").as_unit("ns"))):
            raise ValueError("Expected an aligned, regular, timezone-naive half-hour grid")
        self.slot = (self.times.hour * 2 + self.times.minute // 30).to_numpy()
        self.weekday = self.times.day_of_week.to_numpy()
        self.cum = np.zeros((self.n + 1, self.routes, 7))
        np.cumsum(self.raw, axis=0, dtype=np.float64, out=self.cum[1:])
        low, high = halfhour_bounds(self.raw[:, :, 0], .5)
        # Same addition/division/cast as the original; avoid two full float64 temporaries.
        np.add(low, high, out=low)
        low /= 2
        self.half = low.astype(np.float32)
        del low, high
        self.ch = np.zeros((self.n + 1, self.routes))
        np.cumsum(self.half, axis=0, dtype=np.float64, out=self.ch[1:])

    def prefix(self, origin):
        """A shared-array view; every retained cumulative element is causal."""
        index = self.times.get_loc(pd.Timestamp(origin))
        view = copy.copy(self)
        view.n = index + 1
        for name in ("times", "raw", "slot", "weekday", "half"):
            setattr(view, name, getattr(self, name)[:index + 1])
        for name in ("cum", "ch"):
            setattr(view, name, getattr(self, name)[:index + 2])
        return view

    def features(self, source):
        """The original105 float32 columns in their fixed order."""
        if not 14 * 48 <= source < self.n:
            raise ValueError("Insufficient observed history for14-day aligned features")
        cumulative = self.cum
        mean = (cumulative[source + 1] - cumulative[source + 1 - 336]) / 336
        scale = np.maximum(mean[:, 0], 1000)
        unit = np.maximum(mean[:, 1:], 1)
        columns = [np.log(scale)]
        for lag in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12):
            columns.extend((self.raw[source - lag, :, 0] / scale, self.half[source - lag] / scale))
            columns.extend(self.raw[source - lag, :, j + 1] / unit[:, j] for j in range(3))
        for lag in (0, 1, 4):
            columns.extend(self.raw[source - lag, :, j + 1] / unit[:, j] for j in (3, 4, 5))
        for window in (4, 8, 16, 48):
            recent = (cumulative[source + 1] - cumulative[source + 1 - window]) / window
            columns.extend((recent[:, 0] / scale,
                            (self.ch[source + 1] - self.ch[source + 1 - window]) / (window * scale)))
            columns.extend(recent[:, j + 1] / unit[:, j] for j in range(6))
        historical = np.stack([(cumulative[source - 48 * day + 9, :, 0]
                                - cumulative[source - 48 * day + 2, :, 0]) / (7 * scale)
                               for day in range(1, 15)])
        columns.extend([historical[0], historical[1], historical[6], historical[13],
                        historical[:7].mean(0), np.median(historical[:7], axis=0),
                        historical.mean(0), historical.std(0),
                        np.mean(self.raw[source - 47:source + 1, :, 0] == 0, axis=0)])
        for j in (1, 2):
            difference = cumulative[source + 1, :, j] - cumulative[source + 1, :, j + 1]
            mean48 = (cumulative[source - 46:source + 2, :, j]
                      - cumulative[source - 46:source + 2, :, j + 1]).mean(0)
            columns.append((difference - mean48) / unit[:, j - 1])
        columns.extend([np.full(self.routes, self.slot[source]), np.full(self.routes, self.weekday[source])])
        numeric = np.asarray(np.column_stack(columns), dtype=np.float32)
        if numeric.shape != (self.routes, 105) or not np.isfinite(numeric).all():
            raise ValueError("Expected105 finite route-window columns")
        return numeric, scale

    def supervised(self, sources):
        sources = np.asarray(sources)
        if (sources.ndim != 1 or not len(sources) or sources.dtype.kind not in "iu"
                or sources.min() < 14 * 48 or sources.max() + 8 >= self.n):
            raise ValueError("Each training source and all eight labels must mature by the prefix cutoff")
        rows = len(sources) * self.routes
        numeric = np.empty((rows, 105), dtype=np.float32)
        target = np.empty(rows)
        weight = np.empty(rows)
        for k, source in enumerate(sources):
            section = slice(k * self.routes, (k + 1) * self.routes)
            numeric[section], scale = self.features(int(source))
            target[section] = (self.cum[source + 9, :, 0] - self.cum[source + 2, :, 0]) / (7 * scale)
            weight[section] = scale
        return numeric, target, weight


def load_route_window_panel(train_path: str | Path, origin, route_ids) -> RouteWindowPanel:
    """Load only observed raw counts in weekly chunks and release the last batch."""
    origin = pd.Timestamp(origin)
    routes = np.asarray(route_ids)
    if pd.isna(origin) or origin.tz is not None or origin != origin.floor("30min"):
        raise ValueError("Expected an aligned timezone-naive origin")
    observed = pd.read_parquet(train_path, columns=["timestamp"],
                               filters=[("timestamp", "<=", origin.to_pydatetime())], use_threads=False)
    if observed.empty or observed.timestamp.isna().any() or pd.Timestamp(observed.timestamp.max()) != origin:
        raise ValueError("Origin must be the final timestamp of the loaded observed prefix")
    start = pd.Timestamp(observed.timestamp.min())
    del observed
    times = pd.date_range(start, origin, freq="30min").as_unit("ns")
    counts = np.empty((len(times), len(routes), 7), dtype=np.int32)
    for first in range(0, len(times), 336):
        window = times[first:first + 336]
        frame = pd.read_parquet(train_path, columns=["route_id", "timestamp", *_COLUMNS],
                                filters=[("timestamp", ">=", window[0].to_pydatetime()),
                                         ("timestamp", "<=", window[-1].to_pydatetime())], use_threads=False)
        frame = frame.sort_values(["timestamp", "route_id"])
        if (len(frame) != len(window) * len(routes)
                or not np.array_equal(frame.timestamp.to_numpy(dtype="datetime64[ns]"), np.repeat(window.to_numpy(), len(routes)))
                or not np.array_equal(frame.route_id.to_numpy(), np.tile(routes, len(window)))):
            raise ValueError("Train must contain every requested route once per observed half hour")
        values = frame[_COLUMNS].to_numpy()
        if (not np.isfinite(values).all() or values.min() < 0 or values.max() > np.iinfo(np.int32).max
                or (values.dtype.kind not in "iu" and not np.equal(values, np.floor(values)).all())):
            raise ValueError("Expected finite nonnegative integer counts in int32 range")
        counts[first:first + len(window)] = values.reshape(-1, len(routes), 7)
        del values, frame
    return RouteWindowPanel(times, counts, routes)


def fit_weekly_route_window(panel: RouteWindowPanel):
    """One fixed400-tree CPU1 model, fitted only at Saturday10:30."""
    import lightgbm as lgb

    if panel.times[-1].weekday() != 5 or int(panel.slot[-1]) != 21:
        raise ValueError("Weekly fit requires a Saturday10:30 prefix")
    index = panel.n - 1
    sources = np.arange(max(14 * 48, index - 8 - 42 * 48), index - 8 + 1)
    sources = sources[np.isin(panel.slot[sources], [19, 21, 23])]
    numeric, target, weight = panel.supervised(sources)
    audit = dict(fit_origin=str(panel.times[-1]), latest_label_timestamp=str(panel.times[sources[-1] + 8]),
                 first_source=str(panel.times[sources[0]]), last_source=str(panel.times[sources[-1]]),
                 training_sources=len(sources), rows=len(numeric), features=numeric.shape[1])
    dataset = lgb.Dataset(numeric, label=target, weight=weight, free_raw_data=True)
    model = lgb.train(dict(_PARAMETERS), dataset, num_boost_round=400)
    if model.current_iteration() != 400:
        raise ValueError("Incomplete400-tree route-window model")
    return model, audit


def predict_weekly_route_window(panel: RouteWindowPanel, model, fit_audit, anchor):
    """Use a mature weekly model and preserve anchor H1 and seven temporal totals."""
    if not (pd.Timestamp(fit_audit["latest_label_timestamp"]) <= pd.Timestamp(fit_audit["fit_origin"])
            <= panel.times[-1] < pd.Timestamp(fit_audit["fit_origin"]) + pd.Timedelta(days=7)):
        raise ValueError("Weekly inference/label cutoff mismatch")
    if int(panel.slot[-1]) != 21:
        raise ValueError("The fixed recipe requires a10:30 prediction origin")
    numeric, scale = panel.features(panel.n - 1)
    totals = np.maximum(model.predict(numeric, num_iteration=400, num_threads=1), 0) * 7 * scale
    base = np.ascontiguousarray(anchor, dtype=np.float64)
    prediction, decision = reconcile_temporal(base, totals)
    np.testing.assert_array_equal(prediction[:, 0].view("u8"), base[:, 0].view("u8"))
    return prediction, totals, decision
