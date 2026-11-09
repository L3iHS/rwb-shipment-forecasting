"""Reconcile overlapping hourly forecasts into nonnegative half-hour flows."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

FLOW_METHOD = "nonnegative_hourly_least_squares_v1"


@dataclass
class ReconciledFlow:
    half_hour: np.ndarray
    hourly: np.ndarray
    cumulative: np.ndarray
    hourly_adjustment: np.ndarray


def reconcile_hourly(pred: np.ndarray, x0: np.ndarray) -> ReconciledFlow:
    """Minimise ||A*x - (pred - [x0,0,...])||² subject to x >= 0.

    x0 is a causal estimate of the preceding half-hour, not a separately observed quantity.
    All future horizons used here are predictions, never actual future observations.
    """
    pred, x0 = np.asarray(pred, dtype=np.float64), np.asarray(x0, dtype=np.float64)
    if pred.ndim < 2 or not pred.shape[-1] or pred.shape[:-1] != x0.shape:
        raise ValueError("Expected forecasts (..., horizons) and matching x0 leading shape")
    if not np.isfinite(pred).all() or not np.isfinite(x0).all() or (pred < 0).any() or (x0 < 0).any():
        raise ValueError("Hourly forecasts and estimated x0 must be finite and nonnegative")
    horizon = pred.shape[-1]
    rhs = pred.reshape(-1, horizon).copy()
    rhs[:, 0] -= x0.ravel()
    flows = np.empty_like(rhs)
    flows[:, 0] = rhs[:, 0]
    for h in range(1, horizon):
        flows[:, h] = rhs[:, h] - flows[:, h - 1]
    constrained = np.flatnonzero((flows < 0).any(axis=1))
    design = np.eye(horizon) + np.eye(horizon, k=-1)
    for i in constrained:
        flows[i], _ = nnls(design, rhs[i], maxiter=10 * horizon)
    half_hour = flows.reshape(pred.shape)
    hourly = half_hour + np.concatenate([x0[..., None], half_hour[..., :-1]], axis=-1)
    return ReconciledFlow(half_hour, hourly, np.cumsum(half_hour, axis=-1), hourly - pred)


def legacy_cumulative_inflow(pred: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Frozen v1 approximation, retained solely for historical comparison."""
    return np.clip((np.cumsum(pred, axis=-1) + 0.5 * pred - x0[..., None]) / 2, 0, None)
