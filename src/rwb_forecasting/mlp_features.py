"""Causal latent MLP training arrays; importing this module never imports Torch."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import ForecastData, HORIZON
from .latent_volume import infer_half_hour, hour_from_half_predictions
from .linear_candidates import aligned_features, numeric_columns, source_indices
from .experiments import validate_training_boundary
from .mlp_spec import statistical_spec, specification_sha, feature_names

NUMERIC = [*numeric_columns(True), "slot_sin", "slot_cos", "weekday_sin", "weekday_cos", "log_source_scale"]
assert NUMERIC == feature_names()

def source_scales(history: ForecastData, sources):
    sources = np.asarray(sources)
    if (sources.ndim != 1 or not len(sources) or not np.issubdtype(sources.dtype, np.integer)
            or (sources < 47).any() or (sources >= len(history.timestamps)).any()):
        raise ValueError("Raw hourly scaling requires observed sources and48past steps")
    raw = history.target[:int(sources.max()) + 1]
    prefix = np.vstack([np.zeros((1, len(history.routes))), np.cumsum(raw, axis=0)])
    scale = np.maximum((prefix[sources + 1] - prefix[sources - 47]) / 48, 1000.0)
    if not np.isfinite(scale).all() or (scale < 1000).any():
        raise ValueError("Source scales must be finite raw hourly means floored at1000")
    return scale

def raw_numeric(history, latent_history, sources):
    """The unscaled78-column matrix is source-prefix invariant."""
    frame = aligned_features(latent_history, sources, status_dynamics=True)
    scale = source_scales(history, sources).reshape(-1)
    relative = frame[numeric_columns(True)].to_numpy(dtype=np.float64) / scale[:, None]
    slot = frame.slot_of_day.to_numpy(dtype=float) * (2 * np.pi / 48)
    weekday = frame.day_of_week.to_numpy(dtype=float) * (2 * np.pi / 7)
    numeric = np.column_stack([relative, np.sin(slot), np.cos(slot),
                               np.sin(weekday), np.cos(weekday), np.log(scale)])
    if numeric.shape != (len(frame), 78) or not np.isfinite(numeric).all():
        raise ValueError("Expected78finite numeric features")
    return numeric, scale, frame.route_id.to_numpy()

def prepare_training(history: ForecastData):
    """All supervised labels are observed at or before the supplied origin."""
    spec = statistical_spec()
    latent, widths = infer_half_hour(history.target, spec["rounding_tolerance"])
    latent_history = ForecastData(history.timestamps, history.routes, history.statuses, latent)
    sources = source_indices(latent_history, spec["train_days"], None)
    numeric, scale, training_route_ids = raw_numeric(history, latent_history, sources)
    now, current_scale, inference_route_ids = raw_numeric(history, latent_history,
                                                        np.array([len(history.timestamps) - 1]))
    # Only source features enter this fitted transform; labels and inference rows do not.
    scaler = StandardScaler(copy=False)
    x_train = scaler.fit_transform(numeric).astype(np.float32)
    x_infer = scaler.transform(now).astype(np.float32)
    labels = np.column_stack([latent[sources + h].reshape(-1) / scale for h in range(1, HORIZON + 1)]).astype(np.float32)
    weights = (scale / scale.mean()).astype(np.float32)
    categories = np.unique(training_route_ids)
    index = pd.Index(categories)
    route_train = index.get_indexer(training_route_ids).astype(np.int64)
    route_infer = index.get_indexer(inference_route_ids).astype(np.int64)
    if ((route_train < 0).any() or (route_infer < 0).any() or len(categories) != len(history.routes)
            or not np.isfinite(x_train).all() or not np.isfinite(x_infer).all()
            or not np.isfinite(labels).all() or (labels < 0).any()
            or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Invalid supervised tensors or unseen inference route")
    audit = {"origin": str(history.timestamps[-1]), "train_days": 14, "training_sources": len(sources),
        "training_rows": len(x_train), "first_source_timestamp": str(history.timestamps[sources[0]]),
        "last_source_timestamp": str(history.timestamps[sources[-1]]),
        "latest_label_timestamp": str(history.timestamps[sources[-1] + HORIZON]),
        "latest_feature_timestamp": str(history.timestamps[-1]),
        "earliest_feature_timestamp": str(history.timestamps[0]),
        "source_causality": "forward-only half bounds; every raw feature/scaling source-prefix only",
        "normalization": spec["normalization"], "feature_names": NUMERIC,
        "numeric_features": 78, "route_categories": len(categories), "scaler_fit_rows": len(x_train),
        "scaler_fit_last_timestamp": str(history.timestamps[sources[-1]]),
        "weight_denominator_training_mean_scale": float(scale.mean()),
        "current_half_width_mean": float(widths[-1].mean()), "specification_sha256": specification_sha()}
    validate_training_boundary(audit, history.timestamps[-1])
    payload = {"x_train": x_train, "y_train": labels, "weight_train": weights,
        "route_train": route_train, "x_infer": x_infer, "route_infer": route_infer,
        "route_ids": np.asarray(history.routes, dtype=np.int64),
        "route_categories": categories.astype(np.int64),
        "source_timestamps_ns": history.timestamps[sources].to_numpy(dtype="datetime64[ns]").astype(np.int64)}
    state = {"scaler_mean": scaler.mean_.copy(), "scaler_scale": scaler.scale_.copy(),
        "scaler_var": scaler.var_.copy(), "scaler_samples": np.array(scaler.n_samples_seen_, dtype=np.int64),
        "route_categories": categories.astype(np.int64), "scale_infer": current_scale.astype(np.float64),
        "current_half": latent[-1].astype(np.float64)}
    return payload, state, audit

def reconstruct_hourly(normalized_half, state):
    point = np.asarray(normalized_half)
    scale, current = np.asarray(state["scale_infer"]), np.asarray(state["current_half"])
    if (point.shape != (len(scale), HORIZON) or current.shape != scale.shape
            or not np.isfinite(point).all() or (point < 0).any()
            or not np.isfinite(scale).all() or (scale < 1000).any()
            or not np.isfinite(current).all() or (current < 0).any()):
        raise ValueError("Expected finite nonnegative normalized half forecasts and aligned causal native state")
    half = point.astype(np.float64) * scale[:, None]
    return hour_from_half_predictions(current, half)
