"""Explicit reproduction of Vlado's pinned T_h1s/v3 feature implementation.

Adapted from L3iHS/rwb-shipment-forecasting, participant vlado, commit
bd61c6d560f4a838dc436fbc13139fa42ccb99be, experiments/exp_gbm.py,
Git blob d09105b0cf3f31c442d7743db24f50c554cd6786. No original CLI,
validation loop, searches, file loading, model fitting or output code is run.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

SOURCE_COMMIT = "bd61c6d560f4a838dc436fbc13139fa42ccb99be"
SOURCE_GIT_BLOB = "d09105b0cf3f31c442d7743db24f50c554cd6786"
SOURCE_SHA256 = "300d99ee8861bde092110ca72543d5be2e13c1343fbed567cfb95f4a58ae471f"
SOURCE_URL = f"https://github.com/L3iHS/rwb-shipment-forecasting/blob/{SOURCE_COMMIT}/experiments/exp_gbm.py"
DAY = 48
HORIZON = 8
MIN_SOURCE = 29 * DAY


def fixed_recipe():
    """Fresh configuration values; caller mutation cannot alter feature behavior."""
    return dict(tag="T_h1s", h1_struct=True, algo="lgb", level="mean", clip=True,
        feats="v3", train_days=42, origin_slots=[16, 26], objective="tweedie",
        rounds=[600], lr=.05, leaves=63, min_leaf=200, route_cat=False,
        per_h=False, seeds=[42], tweedie_p=1.3)


@dataclass
class V3Context:
    times: pd.DatetimeIndex
    routes: np.ndarray
    raw_target: np.ndarray
    statuses: np.ndarray
    target: np.ndarray
    target_cumulative: np.ndarray
    status_cumulative: np.ndarray
    global_target: np.ndarray
    global_target_cumulative: np.ndarray
    half_midpoint: np.ndarray
    half_width: np.ndarray
    global_status: np.ndarray
    global_status_cumulative: np.ndarray
    slots: np.ndarray
    weekdays: np.ndarray

    @classmethod
    def from_history(cls, history):
        times = pd.DatetimeIndex(history.timestamps)
        routes = np.asarray(history.routes)
        raw = np.array(history.target, dtype=np.float64, copy=True)
        statuses = np.array(history.statuses, dtype=np.float64, copy=True)
        if (len(times) < MIN_SOURCE + HORIZON + 1 or not times.is_unique
                or not times.is_monotonic_increasing
                or raw.shape != (len(times), len(routes))
                or statuses.shape != (len(times), len(routes), 6)
                or not np.array_equal(routes, np.arange(len(routes)))
                or not np.isfinite(raw).all() or not np.isfinite(statuses).all()
                or (raw < 0).any() or (statuses < 0).any()):
            raise ValueError("Complete sorted nonnegative route panel required")
        if not np.all(np.diff(times.as_unit("ns").asi8) == 30 * 60 * 10**9):
            raise ValueError("A regular half-hour grid is required")
        slots = (times.hour * 2 + times.minute // 30).to_numpy()
        weekdays = times.day_of_week.to_numpy()
        target = raw.copy()
        for start in np.where(slots == 0)[0]:
            if start < 7 * DAY:
                continue
            cap = 4 * np.median(raw[start - 7 * DAY:start], 0) + 1
            end = min(start + DAY, len(times))
            target[start:end] = np.minimum(raw[start:end], cap)
        cumulative = np.vstack([np.zeros((1, len(routes))), np.cumsum(target, 0)])
        status_cumulative = np.concatenate([np.zeros((1, len(routes), 6)), np.cumsum(statuses, 0)])
        global_target = target.sum(1)
        global_cumulative = np.concatenate([[0], np.cumsum(global_target)])
        alternating = np.zeros(raw.shape)
        lower = np.zeros(raw.shape)
        upper = raw.copy()
        for lag in range(16):
            shifted = np.zeros(raw.shape)
            shifted[lag:] = raw[:len(times) - lag]
            alternating = alternating + ((-1) ** lag) * shifted
            if lag % 2 == 0:
                upper = np.minimum(upper, alternating)
            else:
                lower = np.maximum(lower, alternating)
        feasible = lower <= upper
        midpoint = np.where(feasible, (lower + upper) / 2, raw / 2)
        width = np.where(feasible, upper - lower, raw)
        midpoint[:16] = raw[:16] / 2
        global_status = statuses.sum(1)
        global_status_cumulative = np.vstack([np.zeros((1, 6)), np.cumsum(global_status, 0)])
        return cls(times, routes, raw, statuses, target, cumulative, status_cumulative,
                   global_target, global_cumulative, midpoint, width, global_status,
                   global_status_cumulative, slots, weekdays)

    @property
    def origin(self):
        return self.times[-1]

    def target_mean(self, source, window):
        return (self.target_cumulative[source + 1] - self.target_cumulative[source + 1 - window]) / window

    def status_mean(self, source, window):
        return (self.status_cumulative[source + 1] - self.status_cumulative[source + 1 - window]) / window

    def global_mean(self, source, window):
        return (self.global_target_cumulative[source + 1] - self.global_target_cumulative[source + 1 - window]) / window

    def global_status_sum(self, source, window, columns):
        return (self.global_status_cumulative[source + 1, columns]
                - self.global_status_cumulative[source + 1 - window, columns]).sum()

    def global_status_trend(self, source, window, columns):
        current = self.global_status_sum(source, window, columns)
        reference = np.mean([self.global_status_sum(source - day * DAY, window, columns) for day in range(1, 8)])
        return np.log((current + 1) / (reference + 1))

    def training_sources(self):
        last = len(self.times) - 1
        first = max(last - HORIZON - 42 * DAY, MIN_SOURCE)
        sources = np.asarray([source for source in range(first, last - HORIZON + 1)
                              if 16 <= self.slots[source] <= 26], dtype=np.int64)
        if not len(sources) or sources[-1] + HORIZON >= len(self.times):
            raise ValueError("Only fully matured eight-step labels may enter fitting")
        return sources

    def features(self, source):
        if type(source) is not int or not MIN_SOURCE <= source < len(self.times):
            raise ValueError("Source requires the complete causal lookback")
        level = self.target_mean(source, 7 * DAY) + 1.
        values = {}
        for lag in range(8):
            values[f"lag{lag}"] = self.target[source - lag] / level
        for window, name in [(6, "m3h"), (DAY, "m1d"), (3 * DAY, "m3d"),
                             (7 * DAY, "m7d"), (14 * DAY, "m14d"), (28 * DAY, "m28d")]:
            values[name] = self.target_mean(source, window) / level
        values["sd1d"] = self.target[source - DAY + 1:source + 1].std(0) / level
        values["loglvl"] = np.log(level)
        values["zero_share1d"] = (self.target[source - DAY + 1:source + 1] == 0).mean(0)
        for status in range(6):
            for lag in range(8):
                values[f"s{status + 1}_l{lag}"] = self.statuses[source - lag, :, status] / level
            values[f"s{status + 1}_m1d"] = self.status_mean(source, DAY)[:, status] / level
            values[f"s{status + 1}_m3h"] = self.status_mean(source, 6)[:, status] / level
            values[f"s{status + 1}_ratio"] = self.status_mean(source, 6)[:, status] / (self.status_mean(source, 7 * DAY)[:, status] + 1)
        values["cur_sum"] = self.statuses[source, :, :3].sum(1) / level
        values["prev_sum"] = self.statuses[source, :, 3:6].sum(1) / level
        for window in (6, 12, 24):
            # Original feature names preserved; these are throughput proxies, not queue inventories.
            values[f"backlog3_{window}"] = (self.status_mean(source, window)[:, 2] - self.target_mean(source, window) / 2) * window / level
            values[f"backlog12_{window}"] = (self.status_mean(source, window)[:, :2].sum(1) - self.target_mean(source, window)) * window / level
        for lag in range(4):
            values[f"x{lag}"] = self.half_midpoint[source - lag] / level
        values["xshare"] = self.half_midpoint[source] / (self.raw_target[source] + 1)
        values["xwidth"] = self.half_width[source] / level
        values["bl1_16"] = self.status_mean(source, 16)[:, 0] / (self.status_mean(source, 7 * DAY)[:, 0] + 1) - self.target_mean(source, 16) / (self.target_mean(source, 7 * DAY) + 1)
        stage5 = self.global_status[:, 4]
        values["gp5_d1d"] = (stage5[source] - stage5[source - DAY]) / (stage5[source - 7 * DAY + 1:source + 1].mean() + 1)
        global_history = self.global_target
        values["g_ytd6"] = np.log((global_history[source - 5:source + 1].sum() + 1) / (np.mean([global_history[source - 5 - day * DAY:source + 1 - day * DAY].sum() for day in range(1, 8)]) + 1))
        values["g_scur_td6"] = self.global_status_trend(source, 6, [0, 1, 2])
        values["g_sprev_td6"] = self.global_status_trend(source, 6, [3, 4, 5])
        values["g_last"] = self.global_target[source] / self.global_mean(source, 7 * DAY)
        values["g_m3h"] = self.global_mean(source, 6) / self.global_mean(source, 7 * DAY)
        values["g_m1d"] = self.global_mean(source, DAY) / self.global_mean(source, 7 * DAY)
        base = pd.DataFrame(values)
        blocks = []
        for horizon in range(1, HORIZON + 1):
            frame = base.copy(); target_time = source + horizon
            frame["h"] = horizon
            frame["slot_t"] = (self.slots[source] + horizon) % DAY
            frame["dow_t"] = self.weekdays[source] if self.slots[source] + horizon < DAY else (self.weekdays[source] + 1) % 7
            frame["slot_o"] = self.slots[source]
            frame["yday"] = self.target[target_time - DAY] / level
            frame["y2day"] = self.target[target_time - 2 * DAY] / level
            frame["yweek"] = self.target[target_time - 7 * DAY] / level
            frame["slotmean7"] = np.mean([self.target[target_time - day * DAY] for day in range(1, 8)], 0) / level
            frame["slotmed7"] = np.median([self.target[target_time - day * DAY] for day in range(1, 8)], 0) / level
            frame["slotmean14"] = np.mean([self.target[target_time - day * DAY] for day in range(1, 15)], 0) / level
            frame["dowslot4"] = np.mean([self.target[target_time - week * 7 * DAY] for week in range(1, 5)], 0) / level
            frame["g_slot7"] = np.mean([self.global_target[target_time - day * DAY] for day in range(1, 8)]) / self.global_mean(source, 7 * DAY)
            frame["g_dowslot4"] = np.mean([self.global_target[target_time - week * 7 * DAY] for week in range(1, 5)]) / self.global_mean(source, 7 * DAY)
            frame["yday_nb"] = (self.target[target_time - DAY - 1] + self.target[target_time - DAY + 1]) / 2 / level
            frame["xslot7"] = np.mean([self.half_midpoint[target_time - day * DAY] for day in range(1, 8)], 0) / level
            frame["xslot7_prev"] = np.mean([self.half_midpoint[target_time - 1 - day * DAY] for day in range(1, 8)], 0) / level
            frame["known"] = self.half_midpoint[source] / level if horizon == 1 else 0.
            frame["h1_naive"] = (self.half_midpoint[source] + np.mean([self.half_midpoint[source + 1 - day * DAY] for day in range(1, 8)], 0)) / level if horizon == 1 else 0.
            frame["route_id"] = np.arange(len(self.routes))
            blocks.append(frame)
        result = pd.concat(blocks, ignore_index=True)
        if (result.shape != (len(self.routes) * HORIZON, 124)
                or not np.isfinite(result.to_numpy()).all()
                or not np.isfinite(level).all() or (level <= 0).any()):
            raise ValueError("Finite route-aligned source features and positive scales required")
        return result, level
