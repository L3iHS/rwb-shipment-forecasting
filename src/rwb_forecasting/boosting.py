"""Pooled-horizon gradient boosting with features available at each source."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .data import ForecastData, HORIZON, STEP
from .features import feature_frame

@dataclass(frozen=True)
class BoostConfig:
    name: str = 'hgb_raw_sq'
    normalize: bool = False
    loss: str = 'squared_error'
    calibrate: bool = False
    advanced: bool = False
    direct: bool = False
    residual: bool = False
    train_days: int = 14
    source_stride: int = 8
    max_iter: int = 150
    max_leaf_nodes: int = 31
    learning_rate: float = 0.08
    l2_regularization: float = 10.0

def pooled_features(history: ForecastData, indices: np.ndarray, normalize: bool=False, advanced: bool=False):
    """Return horizon-major rows, scale per row; every observation <= source."""
    indices = np.asarray(indices)
    base = feature_frame(history, indices, 'calendar').drop(columns='route_id')
    r = len(history.routes)
    # The scale is local to each historical source, never fitted using later facts.
    scale = np.maximum(base['target_mean_48'].to_numpy(), 1000.0)
    numeric = [c for c in base.columns if c not in ('slot_of_day', 'day_of_week')]
    if normalize:
        base[numeric] = base[numeric].div(scale, axis=0)
    base['log_scale_24h'] = np.log1p(scale)
    for window in (6,):
        cumulative = np.concatenate([np.zeros((1, r, 6)), np.cumsum(history.statuses, axis=0)])
        means = ((cumulative[indices + 1] - cumulative[indices + 1 - window]) / window).reshape(-1, 6)
        if normalize:
            means /= scale[:, None]
        for s in range(6):
            base[f'status_{s + 1}_mean_{window}'] = means[:, s]
    if advanced:
        target_cumulative = np.vstack([np.zeros((1, r)), np.cumsum(history.target, axis=0)])
        weekmean = ((target_cumulative[indices + 1] - target_cumulative[indices + 1 - 336]) / 336).reshape(-1)
        base['target_mean_336'] = weekmean / scale if normalize else weekmean
        for lag in (1, 2, 3, 6, 12):
            values = history.statuses[indices - lag].reshape(-1, 6)
            if normalize:
                values = values / scale[:, None]
            for s in range(6):
                base[f'status_{s + 1}_lag_{lag}'] = values[:, s]
    blocks = []
    for h in range(1, HORIZON + 1):
        frame = base.copy()
        frame['horizon'] = h
        for period in (48, 336):
            values = history.target[indices + h - period].reshape(-1)
            frame[f'aligned_target_{period}'] = values / scale if normalize else values
        if advanced:
            # Every seasonal reference is before the source: horizon <= 8 < 48.
            daily = np.stack([history.target[indices + h - 48 * day].reshape(-1) for day in range(1, 8)])
            for (name, values) in [('mean', daily.mean(axis=0)), ('median', np.median(daily, axis=0)), ('std', daily.std(axis=0))]:
                frame[f'aligned_daily_{name}_7'] = values / scale if normalize else values
        future_times = history.timestamps[indices] + h * STEP
        frame['forecast_slot'] = np.repeat(future_times.hour * 2 + future_times.minute // 30, r)
        frame['forecast_weekday'] = np.repeat(future_times.day_of_week, r)
        blocks.append(frame)
    return (pd.concat(blocks, ignore_index=True).astype(np.float32), np.tile(scale, HORIZON))

def predict_boosting(history: ForecastData, config: BoostConfig=BoostConfig()):
    if config.train_days < 1 or config.source_stride < 1:
        raise ValueError('Positive train_days and source_stride required')
    last = len(history.timestamps) - 1 - HORIZON
    first = max(336, last - config.train_days * 48)
    if first > last:
        raise ValueError('Insufficient training history')
    indices = np.arange(last, first - 1, -config.source_stride)[::-1]
    if config.residual and (not config.advanced):
        raise ValueError('Residual predictions require advanced seasonal profile')
    (X, scales) = pooled_features(history, indices, config.normalize, config.advanced)
    y = np.concatenate([history.target[indices + h].reshape(-1) for h in range(1, HORIZON + 1)])
    (current, current_scales) = pooled_features(history, np.array([len(history.timestamps) - 1]), config.normalize, config.advanced)
    fit_y = y / scales if config.normalize else y.copy()
    if config.residual:
        fit_y -= X['aligned_daily_mean_7'].to_numpy()
    factors = np.ones(HORIZON)
    block_size = len(indices) * len(history.routes)
    prediction_blocks = []
    horizons = range(HORIZON) if config.direct else (None,)
    for horizon in horizons:
        fit_slice = slice(horizon * block_size, (horizon + 1) * block_size) if horizon is not None else slice(None)
        pred_slice = slice(horizon * len(history.routes), (horizon + 1) * len(history.routes)) if horizon is not None else slice(None)
        model = HistGradientBoostingRegressor(loss=config.loss, max_iter=config.max_iter, max_leaf_nodes=config.max_leaf_nodes, learning_rate=config.learning_rate, l2_regularization=config.l2_regularization, max_bins=127, early_stopping=False, random_state=42)
        model.fit(X.iloc[fit_slice], fit_y[fit_slice])
        prediction = model.predict(current.iloc[pred_slice])
        if config.residual:
            prediction += current.iloc[pred_slice]['aligned_daily_mean_7'].to_numpy()
        if config.normalize:
            prediction *= current_scales[pred_slice]
        if config.calibrate:
            # Only known training labels are used; held-out fold truth is unavailable.
            fitted = model.predict(X.iloc[fit_slice])
            if config.residual:
                fitted += X.iloc[fit_slice]['aligned_daily_mean_7'].to_numpy()
            if config.normalize:
                fitted *= scales[fit_slice]
            calibration_horizons = (horizon,) if horizon is not None else range(HORIZON)
            for h in calibration_horizons:
                lo = 0 if horizon is not None else h * block_size
                denominator = np.maximum(fitted[lo:lo + block_size], 0).sum()
                numerator = y[h * block_size:(h + 1) * block_size].sum()
                factors[h] = np.clip(numerator / denominator, 0.8, 1.25) if denominator > 0 else 1.0
            prediction *= np.repeat(factors[horizon:horizon + 1] if horizon is not None else factors, len(history.routes))
        prediction_blocks.append(prediction)
    pred = np.concatenate(prediction_blocks)
    pred = np.maximum(pred.reshape(HORIZON, len(history.routes)).T, 0)
    audit = {'model': config.name, 'config': asdict(config), 'training_rows': len(X), 'training_sources': len(indices), 'first_source_timestamp': str(history.timestamps[indices[0]]), 'last_source_timestamp': str(history.timestamps[indices[-1]]), 'latest_label_timestamp': str(history.timestamps[indices[-1] + HORIZON]), 'earliest_feature_timestamp': str(history.timestamps[indices[0] - 336]), 'calibration_factors': factors.tolist(), 'fit_early_stopping': False}
    return (pred, audit)
