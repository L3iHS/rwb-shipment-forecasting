from dataclasses import dataclass

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=30)
HORIZON = 8
STATUS_COLUMNS = [f"status_{index}" for index in range(1, 7)]


@dataclass(frozen=True)
class ForecastData:
    """A complete time-by-route panel. Predictors receive only a history slice."""

    timestamps: pd.DatetimeIndex
    routes: np.ndarray
    statuses: np.ndarray
    target: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "ForecastData":
        required = {"timestamp", "route_id", "target_1h", *STATUS_COLUMNS}
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError("Train must be nonempty and contain all required columns")
        if frame[list(required)].isna().any().any():
            raise ValueError("Missing train values")
        if frame.duplicated(["timestamp", "route_id"]).any():
            raise ValueError("Duplicate route/timestamp")
        ordered = frame.sort_values(["timestamp", "route_id"])
        times = pd.DatetimeIndex(ordered["timestamp"].unique())
        routes = np.sort(ordered["route_id"].unique())
        if len(ordered) != len(times) * len(routes):
            raise ValueError("Incomplete route/time grid")
        if len(times) < 2 or not (times[1:] - times[:-1] == STEP).all():
            raise ValueError("Expected a regular 30-minute grid")
        values = ordered[[*STATUS_COLUMNS, "target_1h"]].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Statuses and target must be finite and nonnegative")
        values = values.reshape(len(times), len(routes), 7)
        return cls(times, routes, values[:, :, :6], values[:, :, 6])

    def history(self, origin: pd.Timestamp) -> "ForecastData":
        index = self.timestamps.get_indexer([origin])[0]
        if index < 0:
            raise ValueError("Forecast origin must be an observed timestamp")
        return ForecastData(
            self.timestamps[: index + 1], self.routes,
            self.statuses[: index + 1], self.target[: index + 1],
        )
