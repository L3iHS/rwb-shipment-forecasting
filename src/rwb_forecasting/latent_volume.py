"""Causal reconstruction of half-hour volumes from overlapping hourly totals."""

import numpy as np
from scipy.optimize import lsq_linear, nnls

from .halfhour import halfhour_bounds


def infer_half_hour(target: np.ndarray, rounding_tolerance: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return forward-only midpoint estimates and explicit interval widths."""
    lower, upper = halfhour_bounds(target, rounding_tolerance)
    return (lower + upper) / 2, upper - lower


def hour_from_half_predictions(current_half: np.ndarray, future_half: np.ndarray) -> np.ndarray:
    """Combine the known current half-hour with nonnegative future half-hours."""
    current_half, future_half = np.asarray(current_half, float), np.asarray(future_half, float)
    if (future_half.ndim != 2 or 0 in future_half.shape
            or current_half.shape != (future_half.shape[0],)
            or any(not np.isfinite(x).all() or (x < 0).any() for x in (current_half, future_half))):
        raise ValueError("Half-hour predictions must be finite, nonnegative and route-aligned")
    return future_half + np.column_stack([current_half, future_half[:, :-1]])


def project_hourly_forecasts(current_half: np.ndarray, forecasts: np.ndarray) -> np.ndarray:
    """Least-squares projection onto nonnegative latent half-hour forecasts."""
    current_half, forecasts = np.asarray(current_half, float), np.asarray(forecasts, float)
    if (forecasts.ndim != 2 or 0 in forecasts.shape or current_half.shape != (forecasts.shape[0],)
            or any(not np.isfinite(x).all() or (x < 0).any() for x in (current_half, forecasts))):
        raise ValueError("Forecasts must be finite, nonnegative and route-aligned")
    horizon = forecasts.shape[1]
    matrix = np.eye(horizon) + np.eye(horizon, k=-1)
    future_half = np.empty_like(forecasts)
    for route in range(len(forecasts)):
        target = forecasts[route].copy()
        target[0] -= current_half[route]
        future_half[route], _ = nnls(matrix, target)
    return hour_from_half_predictions(current_half, future_half)


def project_hourly_interval(lower: np.ndarray, upper: np.ndarray, forecasts: np.ndarray) -> np.ndarray:
    """Project using the full current-volume interval rather than a fixed midpoint."""
    lower, upper, forecasts = np.asarray(lower, float), np.asarray(upper, float), np.asarray(forecasts, float)
    if (forecasts.ndim != 2 or 0 in forecasts.shape or lower.shape != (len(forecasts),)
            or upper.shape != lower.shape or (upper < lower).any()
            or any(not np.isfinite(x).all() or (x < 0).any() for x in (lower, upper, forecasts))):
        raise ValueError("Projection requires finite nonnegative aligned bounds and forecasts")
    horizon = forecasts.shape[1]
    matrix = np.eye(horizon, horizon + 1) + np.eye(horizon, horizon + 1, k=1)
    projected = np.empty_like(forecasts)
    for route in range(len(forecasts)):
        if lower[route] == upper[route]:
            projected[route] = project_hourly_forecasts(lower[route:route+1], forecasts[route:route+1])[0]
            continue
        bounds_lower = np.r_[lower[route], np.zeros(horizon)]
        bounds_upper = np.r_[upper[route], np.full(horizon, np.inf)]
        result = lsq_linear(matrix, forecasts[route], bounds=(bounds_lower, bounds_upper), method="bvls", tol=1e-10)
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError("Bounded half-hour projection did not converge")
        projected[route] = matrix @ result.x
    return projected
