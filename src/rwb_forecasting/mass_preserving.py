"""Convex route redistribution that preserves incumbent horizon totals."""

from collections.abc import Callable

import numpy as np

from .data import ForecastData, HORIZON
from .experiments import validate_training_boundary

Predictor = Callable[[ForecastData], tuple[np.ndarray, dict]]


def _forecast(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if (array.ndim != 2 or not len(array) or array.shape[1] != HORIZON
            or not np.isfinite(array).all() or (array < 0).any()):
        raise ValueError("Expected a finite nonnegative route-by-eight forecast")
    return array


def mass_preserving_blend(incumbent: np.ndarray, alternative: np.ndarray,
                         weight: float = 0.5) -> tuple[np.ndarray, dict]:
    """Keep H1; blend H2-H8 shares after matching predicted horizon totals.

    Totals come from predictions only. Preservation is algebraically exact;
    floating-point summation drift is recorded in the audit. A zero alternative
    total cannot define shares, so that horizon retains the incumbent.
    """
    incumbent = _forecast(incumbent)
    alternative = _forecast(alternative)
    if (incumbent.shape != alternative.shape or not np.isfinite(weight) or not 0 <= weight <= 1):
        raise ValueError("Matching forecast shapes and a finite convex weight are required")
    incumbent_totals = incumbent.sum(axis=0)
    alternative_totals = alternative.sum(axis=0)
    if not np.isfinite(incumbent_totals).all() or not np.isfinite(alternative_totals).all():
        raise ValueError("Forecast totals must be finite")
    valid = alternative_totals[1:] > 0
    normalized = incumbent[:, 1:].copy()
    if valid.any():
        selected = alternative[:, 1:][:, valid]
        shares = selected / alternative_totals[1:][valid]
        normalized[:, valid] = shares * incumbent_totals[1:][valid]
    result = incumbent.copy()
    result[:, 1:] = (1 - weight) * incumbent[:, 1:] + weight * normalized
    result_totals = result.sum(axis=0)
    drift = result_totals - incumbent_totals
    relative_drift = np.divide(drift, incumbent_totals, out=np.zeros_like(drift), where=incumbent_totals != 0)
    return result, {
        "mode": "mass-preserving latent route-share blend; H1 unchanged",
        "weight": float(weight), "fitted_parameters": False,
        "normalization_source": "incumbent predicted per-horizon totals; no future observations",
        "incumbent_predicted_totals": incumbent_totals.tolist(),
        "alternative_predicted_totals": alternative_totals.tolist(),
        "result_predicted_totals": result_totals.tolist(),
        "maximum_absolute_total_drift": float(np.abs(drift).max()),
        "maximum_relative_total_drift": float(np.abs(relative_drift).max()),
        "first_horizon_exact": bool(np.array_equal(result[:, 0], incumbent[:, 0])),
        "zero_alternative_fallback_horizons": (np.flatnonzero(~valid) + 2).tolist(),
    }


def predict_mass_preserving(
    history: ForecastData, incumbent_predictor: Predictor,
    alternative_predictor: Predictor, weight: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Reconstruct both causal forecasts; cached validation OOF is not used."""
    incumbent, incumbent_audit = incumbent_predictor(history)
    alternative, alternative_audit = alternative_predictor(history)
    origin = history.timestamps[-1]
    validate_training_boundary(incumbent_audit, origin)
    validate_training_boundary(alternative_audit, origin)
    if np.asarray(incumbent).shape != (len(history.routes), HORIZON):
        raise ValueError("Incumbent forecast does not match history routes")
    result, audit = mass_preserving_blend(incumbent, alternative, weight)
    return result, {
        **audit, "origin": str(origin),
        "incumbent_model": incumbent_audit, "alternative_model": alternative_audit,
    }
