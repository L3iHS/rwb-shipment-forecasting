"""Opt-in causal scale weighting for the normalized direct HGB model.

The incumbent implementation is deliberately unchanged. Source scale weights
adjust the empirical objective; they are not forecast-time bias calibration.
"""

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .boosting import BoostConfig, pooled_features, predict_boosting
from .data import ForecastData, HORIZON


@dataclass(frozen=True)
class WeightedBoostConfig:
    base: BoostConfig = BoostConfig()
    scale_weight_power: float = 0.0


def source_scale_weights(scales: np.ndarray, power: float) -> np.ndarray:
    """Positive, mean-one weights from causal training source scales alone.

    The common normalizer uses only training source scales and preserves the
    average sample-weight mass, keeping the L2 penalty on a comparable scale.
    With normalized responses, power 2 cancels squared-loss scale factors;
    power 1 cancels absolute-loss scale factors.
    """
    values = np.asarray(scales, dtype=np.float64)
    if (values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
            or (values <= 0).any()):
        raise ValueError("Source scales must be a finite positive vector")
    if not np.isfinite(power) or not 0 <= power <= 2:
        raise ValueError("Scale weight power must be finite and between 0 and 2")
    # Dividing before the power avoids overflowing physical-volume squares.
    weights = (values / values.max()) ** power
    weights /= weights.mean()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Source scale weights must remain finite and positive")
    return weights


def predict_weighted_boosting(
    history: ForecastData, config: WeightedBoostConfig = WeightedBoostConfig(),
    *, first_horizon: int = 1,
) -> tuple[np.ndarray, dict]:
    """Predict horizons ``first_horizon..8`` with source-local scale weights.

    Skipping H1 permits a composite research runner to reuse an independently
    verified incumbent H1 forecast without fitting a component it will discard.
    All rows remain horizon-major, as in the unchanged pooled feature builder.
    """
    base = config.base
    if type(first_horizon) is not int or not 1 <= first_horizon <= HORIZON:
        raise ValueError("First horizon must be an integer between 1 and 8")
    # This path guarantees the existing public default's prediction and audit.
    if config.scale_weight_power == 0 and first_horizon == 1:
        return predict_boosting(history, base)
    if not base.normalize or not base.direct or base.calibrate:
        raise ValueError("Weighted boosting requires normalized direct models without calibration")
    if base.train_days < 1 or base.source_stride < 1:
        raise ValueError("Positive train_days and source_stride required")
    if base.residual and not base.advanced:
        raise ValueError("Residual predictions require advanced seasonal profile")
    last = len(history.timestamps) - 1 - HORIZON
    first = max(336, last - base.train_days * 48)
    if first > last:
        raise ValueError("Insufficient training history")
    indices = np.arange(last, first - 1, -base.source_stride)[::-1]
    features, scales = pooled_features(
        history, indices, base.normalize, base.advanced, base.global_context, base.halfhour,
    )
    current, current_scales = pooled_features(
        history, np.array([len(history.timestamps) - 1]), base.normalize,
        base.advanced, base.global_context, base.halfhour,
    )
    target = np.concatenate([
        history.target[indices + step].reshape(-1) for step in range(1, HORIZON + 1)
    ])
    fit_target = target / scales
    if base.residual:
        fit_target -= features["aligned_daily_mean_7"].to_numpy()
    weights = source_scale_weights(scales, config.scale_weight_power)
    block_size = len(indices) * len(history.routes)
    predictions = []
    for horizon in range(first_horizon - 1, HORIZON):
        fit_slice = slice(horizon * block_size, (horizon + 1) * block_size)
        pred_slice = slice(horizon * len(history.routes), (horizon + 1) * len(history.routes))
        model = HistGradientBoostingRegressor(
            loss=base.loss, max_iter=base.max_iter, max_leaf_nodes=base.max_leaf_nodes,
            learning_rate=base.learning_rate, l2_regularization=base.l2_regularization,
            max_bins=127, early_stopping=False, random_state=42,
        )
        fit_kwargs = ({"sample_weight": weights[fit_slice]}
                      if config.scale_weight_power != 0 else {})
        model.fit(features.iloc[fit_slice], fit_target[fit_slice], **fit_kwargs)
        prediction = model.predict(current.iloc[pred_slice])
        if base.residual:
            prediction += current.iloc[pred_slice]["aligned_daily_mean_7"].to_numpy()
        prediction *= current_scales[pred_slice]
        predictions.append(prediction)
    prediction = np.maximum(np.stack(predictions, axis=1), 0)
    audit = {
        "model": base.name, "config": asdict(config),
        "training_rows": block_size * len(predictions), "training_sources": len(indices),
        "first_fitted_horizon": first_horizon,
        "first_source_timestamp": str(history.timestamps[indices[0]]),
        "last_source_timestamp": str(history.timestamps[indices[-1]]),
        "latest_label_timestamp": str(history.timestamps[indices[-1] + HORIZON]),
        "earliest_feature_timestamp": str(history.timestamps[indices[0] - 336]),
        "weight_source": "source-local trailing 48-step target mean, floor 1000",
        "weight_normalization": "mean one over training source scales only",
        "weight_min": float(weights.min()), "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "weight_latest_source_timestamp": str(history.timestamps[indices[-1]]),
        "calibration_factors": np.ones(HORIZON).tolist(), "fit_early_stopping": False,
    }
    if base.halfhour:
        audit["earliest_feature_timestamp"] = str(history.timestamps[0])
        audit["halfhour_rounding_tolerance"] = 0.5
        audit["halfhour_method"] = "forward feasible bounds; no backward smoothing"
    return prediction, audit
