import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import ForecastData, HORIZON
from .features import feature_frame, feature_spec

SIMPLE_MODELS = ("last_value", "mean_last_3h", "seasonal_day", "seasonal_week")
MODEL_NAMES = (*SIMPLE_MODELS, "ridge_status")


def predict_simple(history: ForecastData, name: str) -> np.ndarray:
    if name not in SIMPLE_MODELS:
        raise ValueError(f"Unknown simple model: {name}")
    needed = {"last_value": 1, "mean_last_3h": 6, "seasonal_day": 48, "seasonal_week": 336}[name]
    if len(history.timestamps) < needed:
        raise ValueError(f"Insufficient history for {name}")
    if name == "last_value":
        return np.repeat(history.target[-1, :, None], HORIZON, axis=1)
    if name == "mean_last_3h":
        return np.repeat(history.target[-6:].mean(axis=0)[:, None], HORIZON, axis=1)
    start = len(history.timestamps) - needed
    return history.target[start : start + HORIZON].T.copy()


def training_samples(
    history: ForecastData, train_days: int = 14, feature_set: str = "status",
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Only use labels already known at the forecast origin, including its fact."""
    if train_days < 1 or len(history.timestamps) <= HORIZON:
        raise ValueError("Insufficient history or invalid training window")
    spec = feature_spec(feature_set)
    last_source = len(history.timestamps) - 1 - HORIZON
    first_source = max(spec["lookback_steps"], last_source - train_days * 48)
    if first_source > last_source:
        raise ValueError("Insufficient history for the selected features and future labels")
    indices = np.arange(first_source, last_source + 1)
    features = feature_frame(history, indices, feature_set)
    labels = np.column_stack([
        history.target[indices + step].reshape(-1) for step in range(1, HORIZON + 1)
    ])
    audit = {
        "training_rows": len(features),
        "first_source_timestamp": str(history.timestamps[first_source]),
        "last_source_timestamp": str(history.timestamps[last_source]),
        "latest_label_timestamp": str(history.timestamps[last_source + HORIZON]),
        "feature_set": feature_set,
        "earliest_feature_timestamp": str(history.timestamps[first_source - spec["lookback_steps"]]),
        "numeric_feature_count": len(spec["numeric_columns"]),
        "categorical_feature_count": len(spec["categorical_columns"]),
    }
    return features, labels, audit


def predict_ridge(
    history: ForecastData, train_days: int = 14, alpha: float = 4.0, feature_set: str = "status",
) -> tuple[np.ndarray, dict]:
    features, labels, audit = training_samples(history, train_days, feature_set)
    spec = feature_spec(feature_set)
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), spec["numeric_columns"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), spec["categorical_columns"]),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("ridge", Ridge(alpha=alpha, solver="lsqr", tol=1e-6)),
    ])
    model.fit(features, labels)
    current = feature_frame(history, np.array([len(history.timestamps) - 1]), feature_set)
    prediction = np.maximum(model.predict(current), 0)
    return prediction, audit
