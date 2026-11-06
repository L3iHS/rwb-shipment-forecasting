"""Common fixed-origin evaluation for independently implemented predictors."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import ForecastData, HORIZON
from .metrics import evaluate

Predictor = Callable[[ForecastData], tuple[np.ndarray, dict]]


def validate_training_boundary(audit: dict, origin: pd.Timestamp) -> None:
    """Reject future fitted labels in any declared component of a predictor."""
    if "latest_label_timestamp" in audit:
        latest_label = pd.Timestamp(audit["latest_label_timestamp"])
        if pd.isna(latest_label) or latest_label > origin:
            raise ValueError("Training labels exceed the forecast origin or have an invalid timestamp")
    for component in audit.values():
        if isinstance(component, dict):
            validate_training_boundary(component, origin)


def compare_models(
    data: ForecastData, origins: Sequence[pd.Timestamp], predictors: Mapping[str, Predictor],
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Each predictor receives a prefix; future truth is used only by the scorer."""
    if len(origins) == 0 or len(set(origins)) != len(origins) or not predictors:
        raise ValueError("Unique nonempty origins and predictors are required")
    reserved = {"origin", "route_id", "timestamp", "horizon_step", "y_true"}
    if any(not name or name in reserved for name in predictors):
        raise ValueError("Invalid predictor name")
    positions = data.timestamps.get_indexer(origins)
    if (positions < 0).any() or (positions + HORIZON >= len(data.timestamps)).any():
        raise ValueError("Each origin requires a complete future horizon")
    folds, horizons, blocks, audits = [], [], [], []
    for origin, index in zip(origins, positions, strict=True):
        history = data.history(origin)
        forecasts = {}
        for name, predictor in predictors.items():
            if progress:
                progress(f"{origin}: fitting {name}")
            prediction, audit = predictor(history)
            if prediction.shape != (len(data.routes), HORIZON):
                raise ValueError(f"Unexpected prediction shape for {name}")
            validate_training_boundary(audit, origin)
            forecasts[name] = prediction
            audits.append({**audit, "origin": str(origin), "model": name})
        truth = data.target[index + 1:index + HORIZON + 1].T
        block = pd.DataFrame({
            "origin": origin, "route_id": np.repeat(data.routes, HORIZON),
            "timestamp": np.tile(data.timestamps[index + 1:index + HORIZON + 1], len(data.routes)),
            "horizon_step": np.tile(np.arange(1, HORIZON + 1), len(data.routes)),
            "y_true": truth.reshape(-1),
        })
        for name, prediction in forecasts.items():
            folds.append({"origin": origin, "model": name, "points": truth.size, **evaluate(truth, prediction)})
            for step in range(HORIZON):
                horizons.append({"origin": origin, "model": name, "horizon_step": step + 1,
                                 **evaluate(truth[:, step], prediction[:, step])})
            block[name] = prediction.reshape(-1)
        blocks.append(block)
    fold_frame = pd.DataFrame(folds)
    prediction_frame = pd.concat(blocks, ignore_index=True)
    summaries = []
    for name in predictors:
        values = fold_frame.loc[fold_frame["model"] == name, "score"]
        summaries.append({"model": name, "mean_score": values.mean(),
                          "std_score": values.std(ddof=0), "min_score": values.min(),
                          "max_score": values.max()})
    return {"folds": fold_frame, "by_horizon": pd.DataFrame(horizons),
            "summary": pd.DataFrame(summaries).sort_values("mean_score"),
            "predictions": prediction_frame, "audits": audits}
