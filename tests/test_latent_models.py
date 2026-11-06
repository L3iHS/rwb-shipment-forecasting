import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.data import ForecastData
from rwb_forecasting.latent_models import combine_first_horizon, hybrid_first_horizon, predict_latent_ridge, project_incumbent


@pytest.fixture
def panel():
    rng = np.random.default_rng(417)
    times = pd.date_range("2025-08-01", periods=1000, freq="30min")
    latent = rng.uniform(50, 100, (1000, 2))
    latent[::40] = 0
    latent[1::40] = 0
    target = np.rint(latent + np.vstack([np.zeros((1, 2)), latent[:-1]]))
    statuses = np.repeat(latent[:, :, None], 6, axis=2) + np.arange(6)
    return ForecastData(times, np.array([3, 19]), statuses, target)


def test_latent_model_is_causal_and_labels_end_at_origin(panel):
    origin = panel.timestamps[900]
    changed_target, changed_statuses = panel.target.copy(), panel.statuses.copy()
    changed_target[901:] = 1e12
    changed_statuses[901:] = 1e12
    changed = ForecastData(panel.timestamps, panel.routes, changed_statuses, changed_target)
    actual, audit = predict_latent_ridge(panel.history(origin), train_days=2)
    other, _ = predict_latent_ridge(changed.history(origin), train_days=2)
    np.testing.assert_array_equal(actual, other)
    assert actual.shape == (2, 8) and np.isfinite(actual).all() and (actual >= 0).all()
    assert pd.Timestamp(audit["latest_label_timestamp"]) <= origin
    assert pd.Timestamp(audit["latent_inference_cutoff"]) == origin


def test_hybrid_reproduces_real_inference_by_calling_both_predictors(panel):
    calls = []
    incumbent = np.arange(16, dtype=float).reshape(2, 8)
    def predictor(history):
        calls.append(history.timestamps[-1])
        return incumbent.copy(), {"latest_label_timestamp": str(history.timestamps[-1])}
    direct, _ = predict_latent_ridge(panel, train_days=2)
    result, audit = hybrid_first_horizon(panel, predictor, train_days=2)
    assert calls == [panel.timestamps[-1]]
    np.testing.assert_array_equal(result[:, 0], direct[:, 0])
    np.testing.assert_array_equal(result[:, 1:], incumbent[:, 1:])
    assert "incumbent_model" in audit and "latent_model" in audit


def test_projection_predictor_calls_incumbent_and_preserves_nested_audit(panel):
    calls = []
    def predictor(history):
        calls.append(history.timestamps[-1])
        return np.full((2, 8), 50.), {"latest_label_timestamp": str(history.timestamps[-1])}
    result, audit = project_incumbent(panel, predictor)
    assert calls == [panel.timestamps[-1]]
    assert result.shape == (2, 8) and np.isfinite(result).all() and (result >= 0).all()
    assert "incumbent_model" in audit


@pytest.mark.parametrize("bad", [np.zeros(8), np.zeros((2, 7)), np.full((2, 8), np.nan), -np.ones((2, 8))])
def test_hybrid_rejects_bad_prediction(bad):
    with pytest.raises(ValueError):
        combine_first_horizon(bad, np.zeros((2, 8)))
