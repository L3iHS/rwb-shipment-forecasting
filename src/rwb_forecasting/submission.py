"""Fit on known train history and map forecasts to competition test IDs."""

import numpy as np
import pandas as pd

from .data import ForecastData, HORIZON, STEP
from .models import predict_ridge


def make_submission(
    data: ForecastData, test: pd.DataFrame, train_days: int = 14,
    alpha: float = 4.0, feature_set: str = "calendar",
) -> tuple[pd.DataFrame, dict]:
    required = ["id", "route_id", "timestamp"]
    if test.empty or not set(required).issubset(test.columns):
        raise ValueError("Test must contain id, route_id and timestamp")
    if test[required].isna().any().any() or not test["id"].is_unique:
        raise ValueError("Test IDs must be unique and keys must not be missing")
    origin = data.timestamps[-1]
    times = pd.date_range(origin + STEP, periods=HORIZON, freq=STEP)
    expected = pd.MultiIndex.from_product([data.routes, times], names=["route_id", "timestamp"])
    actual = pd.MultiIndex.from_frame(test[["route_id", "timestamp"]])
    if len(actual) != len(expected) or not actual.is_unique or not actual.difference(expected).empty:
        raise ValueError("Test must contain exactly the next eight points for every known route")

    prediction, audit = predict_ridge(data, train_days, alpha, feature_set)
    if prediction.shape != (len(data.routes), HORIZON):
        raise ValueError("Unexpected prediction shape")
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid prediction values")
    forecasts = pd.Series(prediction.reshape(-1), index=expected)
    output = pd.DataFrame({"id": test["id"].to_numpy(), "y_pred": forecasts.reindex(actual).to_numpy()})
    audit.update({"origin": str(origin), "test_first_timestamp": str(times[0]), "test_last_timestamp": str(times[-1])})
    return output.sort_values("id").reset_index(drop=True), audit
