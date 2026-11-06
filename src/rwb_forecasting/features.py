"""Causal features aligned to each supervised source timestamp."""

import numpy as np
import pandas as pd

from .data import ForecastData, STATUS_COLUMNS

FEATURE_SETS = ("status", "lags", "rolling", "calendar")
TARGET_LAGS = (0, 1, 2, 3, 6, 12, 48, 336)
ROLLING_WINDOWS = (6, 12, 48)
CALENDAR_COLUMNS = ("slot_of_day", "day_of_week")


def feature_spec(feature_set: str) -> dict:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    level = FEATURE_SETS.index(feature_set)
    numeric = list(STATUS_COLUMNS)
    categorical = ["route_id"]
    if level >= 1:
        numeric.extend(f"target_lag_{lag}" for lag in TARGET_LAGS)
    if level >= 2:
        numeric.extend(f"target_mean_{window}" for window in ROLLING_WINDOWS)
    if level >= 3:
        categorical.extend(CALENDAR_COLUMNS)
    return {
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "lookback_steps": max(TARGET_LAGS) if level >= 1 else 0,
        "target_lag_steps": list(TARGET_LAGS) if level >= 1 else [],
        "inclusive_mean_windows_steps": list(ROLLING_WINDOWS) if level >= 2 else [],
        "calendar_reference": "source timestamp" if level >= 3 else None,
    }


def feature_frame(history: ForecastData, indices: np.ndarray, feature_set: str = "status") -> pd.DataFrame:
    spec = feature_spec(feature_set)
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Source indices must be a nonempty integer vector")
    if (indices < spec["lookback_steps"]).any() or (indices >= len(history.timestamps)).any():
        raise ValueError("Source features require unavailable history")
    features = pd.DataFrame(history.statuses[indices].reshape(-1, 6), columns=STATUS_COLUMNS)
    features["route_id"] = np.tile(history.routes, len(indices))
    for lag in spec["target_lag_steps"]:
        features[f"target_lag_{lag}"] = history.target[indices - lag].reshape(-1)
    if spec["inclusive_mean_windows_steps"]:
        # Prefix sums at source+1 include no target after that source, even
        # though later known observations may also exist in this history slice.
        cumulative = np.vstack([np.zeros((1, len(history.routes))), np.cumsum(history.target, axis=0)])
        for window in spec["inclusive_mean_windows_steps"]:
            means = (cumulative[indices + 1] - cumulative[indices + 1 - window]) / window
            features[f"target_mean_{window}"] = means.reshape(-1)
    if spec["calendar_reference"]:
        timestamps = history.timestamps[indices]
        features["slot_of_day"] = np.repeat(timestamps.hour * 2 + timestamps.minute // 30, len(history.routes))
        features["day_of_week"] = np.repeat(timestamps.day_of_week, len(history.routes))
    return features
