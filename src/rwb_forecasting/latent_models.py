"""Latent half-hour regression and incumbent forecast postprocessing."""

from collections.abc import Callable

import numpy as np

from .data import ForecastData, HORIZON
from .halfhour import halfhour_bounds
from .latent_volume import hour_from_half_predictions, infer_half_hour, project_hourly_interval
from .linear_candidates import predict_linear_candidate

Predictor = Callable[[ForecastData], tuple[np.ndarray, dict]]


def _check_forecast(prediction: np.ndarray, routes: int) -> np.ndarray:
    prediction = np.asarray(prediction, float)
    if (prediction.shape != (routes, HORIZON) or not np.isfinite(prediction).all()
            or (prediction < 0).any()):
        raise ValueError("Expected finite nonnegative route-by-eight forecast")
    return prediction


def predict_latent_ridge(history: ForecastData, train_days: int = 14, alpha: float = 4.0,
                         rounding_tolerance: float = 0.5) -> tuple[np.ndarray, dict]:
    latent, widths = infer_half_hour(history.target, rounding_tolerance)
    latent_history = ForecastData(history.timestamps, history.routes, history.statuses, latent)
    future_half, audit = predict_linear_candidate(
        latent_history, family="global", train_days=train_days, alpha=alpha,
        clock_radius=None, status_dynamics=True, solver="block_gram",
    )
    hourly = hour_from_half_predictions(latent[-1], future_half)
    audit.update({
        "target_representation": "causal half-hour midpoint",
        "rounding_tolerance": rounding_tolerance,
        "latent_inference_cutoff": str(history.timestamps[-1]),
        "latent_feature_lookback_timestamp": audit["earliest_feature_timestamp"],
        "earliest_feature_timestamp": str(history.timestamps[0]),
        "latent_source_causality": "forward-only bounds; no backward revision",
        "current_half_width_mean": float(widths[-1].mean()),
        "current_half_width_max": float(widths[-1].max()),
        "hour_reconstruction": "adjacent half-hour sum; first uses current midpoint",
    })
    return _check_forecast(hourly, len(history.routes)), audit


def combine_first_horizon(incumbent: np.ndarray, direct: np.ndarray) -> np.ndarray:
    incumbent = np.asarray(incumbent, float)
    if incumbent.ndim != 2:
        raise ValueError("Expected a route-by-eight incumbent")
    incumbent = _check_forecast(incumbent, len(incumbent))
    direct = _check_forecast(direct, len(incumbent))
    result = incumbent.copy()
    result[:, 0] = direct[:, 0]
    return result


def hybrid_first_horizon(history: ForecastData, incumbent_predictor: Predictor,
                         train_days: int = 14, alpha: float = 4.0,
                         rounding_tolerance: float = 0.5) -> tuple[np.ndarray, dict]:
    incumbent, incumbent_audit = incumbent_predictor(history)
    direct, direct_audit = predict_latent_ridge(history, train_days, alpha, rounding_tolerance)
    return combine_first_horizon(incumbent, direct), {
        "mode": "latent H1; incumbent H2-H8", "origin": str(history.timestamps[-1]),
        "incumbent_model": incumbent_audit, "latent_model": direct_audit,
    }


def project_prediction(history: ForecastData, incumbent: np.ndarray,
                       rounding_tolerance: float = 0.5) -> tuple[np.ndarray, dict]:
    incumbent = _check_forecast(incumbent, len(history.routes))
    lower, upper = halfhour_bounds(history.target, rounding_tolerance)
    result = project_hourly_interval(lower[-1], upper[-1], incumbent)
    audit = {
        "mode": "bounded nonnegative half-hour projection", "origin": str(history.timestamps[-1]),
        "rounding_tolerance": rounding_tolerance, "latent_inference_cutoff": str(history.timestamps[-1]),
        "earliest_feature_timestamp": str(history.timestamps[0]),
        "latent_source_causality": "forward-only bounds; no backward revision",
        "current_half_width_mean": float((upper[-1] - lower[-1]).mean()),
        "current_half_width_max": float((upper[-1] - lower[-1]).max()),
        "projection_solver": "bounded-variable least squares; tol=1e-10",
        "learned_parameters": False,
    }
    return _check_forecast(result, len(history.routes)), audit


def project_incumbent(history: ForecastData, incumbent_predictor: Predictor,
                      rounding_tolerance: float = 0.5) -> tuple[np.ndarray, dict]:
    incumbent, incumbent_audit = incumbent_predictor(history)
    prediction, audit = project_prediction(history, incumbent, rounding_tolerance)
    audit["incumbent_model"] = incumbent_audit
    return prediction, audit
