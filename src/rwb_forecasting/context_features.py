"""Source-causal global operating context and latent half-hour summaries."""
import numpy as np
import pandas as pd

from .data import ForecastData
from .halfhour import halfhour_bounds

HALFHOUR_LAGS = (0, 1, 2, 3, 6, 12)


def global_context_frame(history: ForecastData, indices: np.ndarray) -> pd.DataFrame:
    """Return shared context per source, repeated for each route in source order."""
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('Source indices must be a nonempty integer vector')
    if (indices < 336).any() or (indices >= len(history.timestamps)).any():
        raise ValueError('Global context requires available 7-day source history')
    stop = int(indices.max()) + 1
    target = history.target[:stop].mean(axis=1)
    statuses = history.statuses[:stop].mean(axis=1)
    target_sum = np.r_[0., np.cumsum(target)]
    status_sum = np.vstack([np.zeros((1, 6)), np.cumsum(statuses, axis=0)])
    means = {
        window: (target_sum[indices + 1] - target_sum[indices + 1 - window]) / window
        for window in (6, 48, 336)
    }
    scale = np.maximum(means[48], 1000.)
    frame = pd.DataFrame({
        'global_target_current_ratio': target[indices] / scale,
        'global_target_recent_ratio': means[6] / scale,
        'global_target_week_ratio': means[48] / np.maximum(means[336], 1000.),
        'global_target_day_ratio': target[indices] / np.maximum(target[indices - 48], 1000.),
    })
    recent_status = (status_sum[indices + 1] - status_sum[indices + 1 - 6]) / 6
    for status in range(6):
        frame[f'global_status_{status + 1}_level'] = statuses[indices, status] / scale
        frame[f'global_status_{status + 1}_momentum'] = (
            (statuses[indices, status] - recent_status[:, status])
            / np.maximum(recent_status[:, status], 1000.)
        )
    return frame.loc[frame.index.repeat(len(history.routes))].reset_index(drop=True)


def halfhour_feature_frame(
    history: ForecastData, indices: np.ndarray, scale: np.ndarray | None = None,
) -> pd.DataFrame:
    """Use forward feasible bounds, never future zero anchors or smoothing."""
    indices = np.asarray(indices)
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError('Source indices must be a nonempty integer vector')
    if (indices < max(HALFHOUR_LAGS)).any() or (indices >= len(history.timestamps)).any():
        raise ValueError('Half-hour features require available source history')
    lower, upper = halfhour_bounds(history.target[:int(indices.max()) + 1], rounding_tolerance=0.5)
    midpoint = (lower + upper) / 2
    row_count = len(indices) * len(history.routes)
    denominator = np.ones(row_count) if scale is None else np.asarray(scale, dtype=np.float64)
    if denominator.shape != (row_count,) or not np.isfinite(denominator).all() or (denominator <= 0).any():
        raise ValueError('Scale must be a finite positive value for every feature row')
    features = {
        f'halfhour_midpoint_lag_{lag}': midpoint[indices - lag].reshape(-1) / denominator
        for lag in HALFHOUR_LAGS
    }
    features['halfhour_width'] = (upper[indices] - lower[indices]).reshape(-1) / denominator
    return pd.DataFrame(features)
