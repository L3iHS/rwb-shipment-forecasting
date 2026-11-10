"""Causal linear candidates for a fixed-origin, eight-step forecast."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import ForecastData, HORIZON, STATUS_COLUMNS
from .block_ridge import fit_block_ridge


TARGET_LAGS = (0, 1, 2, 3, 6, 12)
MEAN_WINDOWS = (6, 12, 48)
STATUS_LAGS = (1, 2, 3, 6, 12)
STATUS_MEAN_WINDOWS = (6, 12)
LOOKBACK = 336
NUMERIC_COLUMNS = [
    *STATUS_COLUMNS,
    *(f"target_lag_{lag}" for lag in TARGET_LAGS),
    *(f"target_mean_{window}" for window in MEAN_WINDOWS),
    *(f"target_aligned_{period}_{h}" for period in (48, 336) for h in range(1, HORIZON + 1)),
]


def numeric_columns(status_dynamics: bool = False) -> list[str]:
    return [*NUMERIC_COLUMNS, *([
        *(f"{name}_lag_{lag}" for name in STATUS_COLUMNS for lag in STATUS_LAGS),
        *(f"{name}_mean_{window}" for name in STATUS_COLUMNS for window in STATUS_MEAN_WINDOWS),
    ] if status_dynamics else [])]


def aligned_features(history: ForecastData, indices: np.ndarray, status_dynamics: bool = False) -> pd.DataFrame:
    """Read past targets at the same clock time as every future horizon."""
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer)
            or (indices < LOOKBACK).any() or (indices >= len(history.timestamps)).any()):
        raise ValueError("Source features require a nonempty integer vector with available history")
    routes = len(history.routes)
    columns = {name: history.statuses[indices, :, k].reshape(-1) for k, name in enumerate(STATUS_COLUMNS)}
    for lag in TARGET_LAGS:
        columns[f"target_lag_{lag}"] = history.target[indices - lag].reshape(-1)
    cumulative = np.vstack([np.zeros((1, routes)), np.cumsum(history.target, axis=0)])
    for window in MEAN_WINDOWS:
        columns[f"target_mean_{window}"] = ((cumulative[indices + 1] - cumulative[indices + 1 - window]) / window).reshape(-1)
    for period in (48, 336):
        for h in range(1, HORIZON + 1):
            columns[f"target_aligned_{period}_{h}"] = history.target[indices + h - period].reshape(-1)
    if status_dynamics:
        for k, name in enumerate(STATUS_COLUMNS):
            for lag in STATUS_LAGS:
                columns[f"{name}_lag_{lag}"] = history.statuses[indices - lag, :, k].reshape(-1)
            status_cumulative = np.vstack([np.zeros((1, routes)), np.cumsum(history.statuses[:, :, k], axis=0)])
            for window in STATUS_MEAN_WINDOWS:
                columns[f"{name}_mean_{window}"] = ((status_cumulative[indices + 1] - status_cumulative[indices + 1 - window]) / window).reshape(-1)
    times = history.timestamps[indices]
    columns["route_id"] = np.tile(history.routes, len(indices))
    columns["slot_of_day"] = np.repeat(times.hour * 2 + times.minute // 30, routes)
    columns["day_of_week"] = np.repeat(times.day_of_week, routes)
    return pd.DataFrame(columns)


def source_indices(history: ForecastData, train_days: int, clock_radius: int | None) -> np.ndarray:
    if train_days < 1 or (clock_radius is not None and not 0 <= clock_radius <= 24):
        raise ValueError("Invalid training window or clock radius")
    last = len(history.timestamps) - HORIZON - 1
    first = max(LOOKBACK, last - train_days * 48)
    indices = np.arange(first, last + 1)
    if clock_radius is not None and len(indices):
        times = history.timestamps[indices]
        source_slots = times.hour * 2 + times.minute // 30
        origin = history.timestamps[-1]
        origin_slot = origin.hour * 2 + origin.minute // 30
        distances = np.abs(source_slots - origin_slot)
        indices = indices[np.minimum(distances, 48 - distances) <= clock_radius]
    if not len(indices):
        raise ValueError("Insufficient history for causal training labels and features")
    return indices


def per_route_array(frame: pd.DataFrame, routes: int, status_dynamics: bool = False) -> np.ndarray:
    """Numeric features and four periodic calendar components, route-major."""
    numeric = frame[numeric_columns(status_dynamics)].to_numpy()
    slot = frame["slot_of_day"].to_numpy() * (2 * np.pi / 48)
    weekday = frame["day_of_week"].to_numpy() * (2 * np.pi / 7)
    values = np.column_stack([numeric, np.sin(slot), np.cos(slot), np.sin(weekday), np.cos(weekday)])
    return values.reshape(-1, routes, values.shape[1]).transpose(1, 0, 2)


def fit_per_route(features: pd.DataFrame, labels: np.ndarray, current: pd.DataFrame,
                  routes: int, alpha: float, status_dynamics: bool = False) -> np.ndarray:
    x = per_route_array(features, routes, status_dynamics)
    y = labels.reshape(-1, routes, HORIZON).transpose(1, 0, 2)
    mean_x, mean_y = x.mean(axis=1, keepdims=True), y.mean(axis=1, keepdims=True)
    scale = x.std(axis=1, keepdims=True)
    scale[scale < 1e-12] = 1
    x = (x - mean_x) / scale
    y = y - mean_y
    gram = x.transpose(0, 2, 1) @ x
    gram[:, np.arange(gram.shape[1]), np.arange(gram.shape[1])] += alpha
    coefficients = np.linalg.solve(gram, x.transpose(0, 2, 1) @ y)
    now = (per_route_array(current, routes, status_dynamics) - mean_x) / scale
    return (now @ coefficients + mean_y)[:, 0, :]


def normalize_forecast_totals(prediction: np.ndarray, totals: np.ndarray, fallback_volumes: np.ndarray) -> np.ndarray:
    """Allocate a separately forecast total, including zero individual forecasts."""
    prediction, totals, fallback_volumes = map(np.asarray, (prediction, totals, fallback_volumes))
    if (prediction.ndim != 2 or 0 in prediction.shape or totals.shape != (prediction.shape[1],)
            or fallback_volumes.shape != (prediction.shape[0],)
            or any(not np.isfinite(x).all() or (x < 0).any() for x in (prediction, totals, fallback_volumes))):
        raise ValueError("Forecast normalization requires finite nonnegative arrays with aligned shapes")
    with np.errstate(over="ignore"):
        summed = prediction.sum(axis=0)
    if not np.isfinite(summed).all():
        raise ValueError("Individual forecast totals overflow")
    shares = np.divide(prediction, summed, out=np.zeros_like(prediction, dtype=float), where=summed > 0)
    if (summed == 0).any():
        maximum = fallback_volumes.max(initial=0)
        weights = fallback_volumes / maximum if maximum > 0 else np.ones(prediction.shape[0])
        weights = weights / weights.sum()
        shares[:, summed == 0] = weights[:, None]
    return shares * totals


def calibrate_aggregate(history: ForecastData, prediction: np.ndarray, solver: str = "lsqr") -> tuple[np.ndarray, dict]:
    """Estimate eight aggregate totals from known history, never future labels."""
    aggregate = ForecastData(history.timestamps, np.array([0]), history.statuses.sum(axis=1, keepdims=True),
                             history.target.sum(axis=1, keepdims=True))
    totals, audit = predict_linear_candidate(aggregate, train_days=28, alpha=20.0, status_dynamics=True, solver=solver)
    normalized = normalize_forecast_totals(prediction, totals[0], history.target[-48:].mean(axis=0))
    return normalized, {"model": "aggregate_ridge", **audit,
                        "forecast_totals": totals[0].tolist(), "normalization": "per-horizon prediction shares"}


def predict_linear_candidate(history: ForecastData, *, family: str = "global", train_days: int = 14,
                             alpha: float = 4.0, clock_radius: int | None = None,
                             status_dynamics: bool = False, aggregate_calibration: bool = False,
                             solver: str = "lsqr") -> tuple[np.ndarray, dict]:
    if family not in {"global", "per_route"} or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Unknown family or invalid regularization")
    if solver not in {"lsqr", "block_gram"}:
        raise ValueError("Unknown numerical solver")
    indices = source_indices(history, train_days, clock_radius)
    features = aligned_features(history, indices, status_dynamics)
    labels = np.column_stack([history.target[indices + h].reshape(-1) for h in range(1, HORIZON + 1)])
    current = aligned_features(history, np.array([len(history.timestamps) - 1]), status_dynamics)
    if family == "global" and solver == "lsqr":
        preprocessing = ColumnTransformer([
            ("numeric", StandardScaler(), numeric_columns(status_dynamics)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), ["route_id", "slot_of_day", "day_of_week"]),
        ])
        model = Pipeline([("preprocessing", preprocessing), ("ridge", Ridge(alpha=alpha, solver="lsqr", tol=1e-6))])
        model.fit(features, labels)
        forecast = model.predict(current)
    elif family == "global":
        numeric = numeric_columns(status_dynamics)
        categorical = ["route_id", "slot_of_day", "day_of_week"]
        scaler, encoder = StandardScaler(), OneHotEncoder(handle_unknown="ignore")
        scaled = scaler.fit_transform(features[numeric])
        encoded = encoder.fit_transform(features[categorical])
        model = fit_block_ridge(scaled, encoded, labels, alpha=alpha)
        forecast = model.predict(scaler.transform(current[numeric]), encoder.transform(current[categorical]))
    else:
        forecast = fit_per_route(features, labels, current, len(history.routes), alpha, status_dynamics)
    audit = {
        "family": family, "train_days": train_days, "alpha": alpha, "clock_radius": clock_radius,
        "training_rows": len(features), "training_sources": len(indices),
        "first_source_timestamp": str(history.timestamps[indices[0]]),
        "last_source_timestamp": str(history.timestamps[indices[-1]]),
        "latest_label_timestamp": str(history.timestamps[indices[-1] + HORIZON]),
        "earliest_feature_timestamp": str(history.timestamps[indices[0] + 1 - 336]),
        "origin": str(history.timestamps[-1]), "numeric_feature_count": len(numeric_columns(status_dynamics)),
        "status_dynamics": status_dynamics,
        "solver": solver if family == "global" else "batched_gram",
        "preprocessing": "fit on training rows only",
    }
    forecast = np.maximum(forecast, 0)
    if aggregate_calibration:
        forecast, audit["aggregate_model"] = calibrate_aggregate(history, forecast, solver=solver)
    return forecast, audit
