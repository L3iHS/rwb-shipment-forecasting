"""Strict scoring of archived forecasts; never trains a forecasting model."""

import numpy as np


def checked_pair(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 3 or 0 in pred.shape:
        raise ValueError("Expected matching nonempty (origins, routes, horizons) arrays")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Forecasts and truth must be finite")
    if (pred < 0).any() or (truth < 0).any():
        raise ValueError("Forecasts and truth must be nonnegative")
    if (truth.sum(axis=(1, 2)) <= 0).any():
        raise ValueError("Each origin must have positive total truth")
    return pred, truth


def per_origin(pred: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    pred, truth = checked_pair(pred, truth)
    total = truth.sum(axis=(1, 2))
    wape = np.abs(pred - truth).sum(axis=(1, 2)) / total
    bias = pred.sum(axis=(1, 2)) / total - 1
    return {"wape": wape, "signed_bias": bias, "abs_bias": np.abs(bias), "score": wape + np.abs(bias)}


def summarize(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    values = per_origin(pred, truth)
    pred, truth = checked_pair(pred, truth)
    total = truth.sum()
    return {
        **{f"mean_{key}": float(value.mean()) for key, value in values.items()},
        "worst_score": float(values["score"].max()),
        "pooled_score": float(np.abs(pred - truth).sum() / total + abs(pred.sum() / total - 1)),
    }


def choose_scale(pred: np.ndarray, truth: np.ndarray, grid: np.ndarray) -> float:
    """Select on calibration arrays only; callers keep evaluation arrays separate."""
    pred, truth = checked_pair(pred, truth)
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or (grid <= 0).any():
        raise ValueError("Scale grid must contain finite positive values")
    scores = [summarize(pred * scale, truth)["mean_score"] for scale in grid]
    return float(grid[int(np.argmin(scores))])


def check_chronology(calibration_times, evaluation_times) -> None:
    cal, ev = np.asarray(calibration_times), np.asarray(evaluation_times)
    if not len(cal) or not len(ev) or cal.max() >= ev.min():
        raise ValueError("Calibration must end before evaluation starts")


def available_calibration_mask(origins: np.ndarray, evaluation_start: int, horizon: int) -> np.ndarray:
    """Only fully observed target windows may calibrate the first evaluation forecast."""
    mask = np.asarray(origins) + horizon <= evaluation_start
    if not mask.any():
        raise ValueError("No fully observed calibration windows before evaluation")
    return mask


def check_truth(saved: np.ndarray, actual: np.ndarray, name: str) -> None:
    """Fail rather than silently aligning archives by shape alone."""
    if saved.shape != actual.shape or not np.array_equal(saved, actual):
        raise ValueError(f"Archived truth does not match raw data ordering: {name}")
