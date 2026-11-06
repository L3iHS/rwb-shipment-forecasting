from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .data import ForecastData, HORIZON, STEP
from .metrics import evaluate
from .models import MODEL_NAMES, SIMPLE_MODELS, predict_ridge, predict_simple


def daily_origins(data: ForecastData, count: int = 7, time: str = "10:30") -> list[pd.Timestamp]:
    if count < 1:
        raise ValueError("At least one origin is required")
    hours, minutes = map(int, time.split(":"))
    candidate = data.timestamps[-1].normalize() + pd.Timedelta(hours=hours, minutes=minutes)
    if candidate + HORIZON * STEP > data.timestamps[-1]:
        candidate -= pd.Timedelta(days=1)
    origins = list(pd.date_range(end=candidate, periods=count, freq="D"))
    if origins[0] not in data.timestamps or any(origin not in data.timestamps for origin in origins):
        raise ValueError("Requested origins are outside the observed grid")
    return origins


def backtest(
    data: ForecastData, origins: Sequence[pd.Timestamp], train_days: int = 14,
    alpha: float = 4.0, progress: Callable[[str], None] | None = None,
) -> dict[str, pd.DataFrame | list[dict]]:
    folds, by_horizon, predictions, audits = [], [], [], []
    if not origins or len(set(origins)) != len(origins):
        raise ValueError("Origins must be nonempty and unique")
    for origin in origins:
        index = data.timestamps.get_indexer([origin])[0]
        if index < 0 or index + HORIZON >= len(data.timestamps):
            raise ValueError("Validation origin requires a full future horizon")
        history = data.history(origin)
        forecasts = {name: predict_simple(history, name) for name in SIMPLE_MODELS}
        if progress:
            progress(f"{origin}: fitting Ridge on history ending at origin")
        forecasts["ridge_status"], audit = predict_ridge(history, train_days, alpha)
        audit["origin"] = str(origin)
        audits.append(audit)
        # Future facts are read only for scoring after predictions are produced.
        future_times = data.timestamps[index + 1 : index + HORIZON + 1]
        truth = data.target[index + 1 : index + HORIZON + 1].T
        block = pd.DataFrame({
            "origin": origin,
            "route_id": np.repeat(data.routes, HORIZON),
            "timestamp": np.tile(future_times, len(data.routes)),
            "horizon_step": np.tile(np.arange(1, HORIZON + 1), len(data.routes)),
            "y_true": truth.reshape(-1),
        })
        for name, prediction in forecasts.items():
            folds.append({"origin": origin, "model": name, "points": truth.size, **evaluate(truth, prediction)})
            for step in range(HORIZON):
                by_horizon.append({
                    "origin": origin, "model": name, "horizon_step": step + 1,
                    **evaluate(truth[:, step], prediction[:, step]),
                })
            block[name] = prediction.reshape(-1)
        predictions.append(block)
        if progress:
            progress(f"{origin}: scored {truth.size} points per model")

    fold_frame = pd.DataFrame(folds)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    winners = fold_frame.loc[fold_frame.groupby("origin")["score"].idxmin(), "model"].value_counts()
    summary = []
    for name in MODEL_NAMES:
        values = fold_frame.loc[fold_frame["model"] == name, "score"]
        pooled = evaluate(prediction_frame["y_true"].to_numpy(), prediction_frame[name].to_numpy())
        summary.append({
            "model": name, "mean_score": values.mean(), "std_score": values.std(ddof=0),
            "min_score": values.min(), "max_score": values.max(),
            "fold_wins": int(winners.get(name, 0)),
            "pooled_wape": pooled["wape"], "pooled_signed_bias": pooled["signed_bias"],
            "pooled_score": pooled["score"],
        })
    return {
        "folds": fold_frame, "by_horizon": pd.DataFrame(by_horizon),
        "summary": pd.DataFrame(summary).sort_values("mean_score").reset_index(drop=True),
        "predictions": prediction_frame, "audits": audits,
    }
