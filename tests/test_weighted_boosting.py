from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from rwb_forecasting.boosting import BoostConfig, pooled_features, predict_boosting
from rwb_forecasting.data import ForecastData
from rwb_forecasting.weighted_boosting import (
    WeightedBoostConfig, predict_weighted_boosting, source_scale_weights,
)


def panel():
    times = pd.date_range("2025-01-01", periods=500, freq="30min")
    step = np.arange(500)[:, None]
    target = (1500 + 400 * np.sin(step / 11) + 2 * step) * np.array([1, 3, 9])[None, :]
    statuses = np.repeat(target[:, :, None], 6, axis=2) * np.arange(1, 7)
    return ForecastData(times, np.array([2, 9, 12]), statuses, target)


def base_config():
    return BoostConfig(normalize=True, advanced=True, direct=True,
                       max_iter=3, max_leaf_nodes=3, source_stride=8)


def test_zero_power_preserves_incumbent_prediction_and_audit_exactly():
    data, base = panel(), base_config()
    with threadpool_limits(limits=1):
        expected, expected_audit = predict_boosting(data, base)
        actual, actual_audit = predict_weighted_boosting(data, WeightedBoostConfig(base, 0))
    np.testing.assert_array_equal(actual, expected)
    assert actual_audit == expected_audit


def test_skipping_h1_does_not_change_unweighted_remaining_horizons():
    data, base = panel(), base_config()
    with threadpool_limits(limits=1):
        expected, _ = predict_boosting(data, base)
        actual, audit = predict_weighted_boosting(data, WeightedBoostConfig(base, 0), first_horizon=2)
    np.testing.assert_array_equal(actual, expected[:, 1:])
    assert audit["first_fitted_horizon"] == 2


def test_configured_weighted_hybrid_matches_screen_horizon_fits():
    from dataclasses import asdict
    from rwb_forecasting.candidates import configured_predictor

    data, base = panel(), base_config()
    spec = {"kind": "weighted_boosting", "params": {
        "base": asdict(base), "scale_weight_power": 1.0,
    }}
    with threadpool_limits(limits=1):
        full, audit = configured_predictor(spec)(data)
        later, _ = predict_weighted_boosting(
            data, WeightedBoostConfig(base, 1), first_horizon=2,
        )
    np.testing.assert_array_equal(full[:, 1:], later)
    assert audit["config"]["scale_weight_power"] == 1.0


def test_weights_cancel_normalized_physical_loss_factors():
    scales = np.array([1000., 3000., 9000.])
    errors = np.array([-400., 200., 1500.])
    squared = np.sum(source_scale_weights(scales, 2) * (errors / scales) ** 2)
    absolute = np.sum(source_scale_weights(scales, 1) * np.abs(errors / scales))
    np.testing.assert_allclose(squared, np.sum(errors ** 2) / np.mean(scales ** 2))
    np.testing.assert_allclose(absolute, np.sum(np.abs(errors)) / scales.mean())


def test_scale_weight_features_ignore_all_observations_after_source():
    data, indices = panel(), np.array([360])
    _, full_scales = pooled_features(data, indices, True, True)
    prefix = data.history(data.timestamps[360])
    _, prefix_scales = pooled_features(prefix, indices, True, True)
    for power in (0, 1, 2):
        np.testing.assert_array_equal(source_scale_weights(full_scales, power),
                                      source_scale_weights(prefix_scales, power))


@pytest.mark.parametrize("loss,power", [("squared_error", 1), ("squared_error", 2),
                                        ("absolute_error", 1)])
def test_weighted_prediction_is_finite_and_label_causal(loss, power):
    data, base = panel(), replace(base_config(), loss=loss)
    with threadpool_limits(limits=1):
        prediction, audit = predict_weighted_boosting(data, WeightedBoostConfig(base, power),
                                                    first_horizon=2)
    assert prediction.shape == (3, 7)
    assert np.isfinite(prediction).all() and (prediction >= 0).all()
    assert pd.Timestamp(audit["latest_label_timestamp"]) <= data.timestamps[-1]
    assert pd.Timestamp(audit["weight_latest_source_timestamp"]) <= data.timestamps[-9]
    assert audit["weight_mean"] == pytest.approx(1)
    assert audit["calibration_factors"] == [1.] * 8


def test_weights_reach_the_correct_direct_horizon_rows(monkeypatch):
    import rwb_forecasting.weighted_boosting as module

    fitted = []

    class SpyRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, features, target, sample_weight):
            fitted.append((features.copy(), target.copy(), sample_weight.copy()))

        def predict(self, features):
            return np.zeros(len(features))

    monkeypatch.setattr(module, "HistGradientBoostingRegressor", SpyRegressor)
    data, base = panel(), base_config()
    predict_weighted_boosting(data, WeightedBoostConfig(base, 1), first_horizon=2)
    assert len(fitted) == 7
    for horizon, (features, target, weights) in enumerate(fitted, start=2):
        assert (features["horizon"] == horizon).all()
        scales = np.expm1(features["log_scale_24h"].to_numpy(dtype=float))
        np.testing.assert_allclose(weights, source_scale_weights(scales, 1), rtol=2e-6)
        np.testing.assert_array_equal(weights, fitted[0][2])


@pytest.mark.parametrize("scales,power", [([], 1), ([0, 1], 1), ([1, np.nan], 1),
                                         ([1, 2], -1), ([1, 2], 3), ([1, 2], np.nan)])
def test_invalid_weight_inputs_are_rejected(scales, power):
    with pytest.raises(ValueError):
        source_scale_weights(scales, power)
