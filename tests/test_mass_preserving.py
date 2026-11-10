import numpy as np
import pandas as pd
import pytest

from rwb_forecasting.data import ForecastData
from rwb_forecasting.experiments import compare_models, validate_training_boundary
from rwb_forecasting.mass_preserving import mass_preserving_blend, predict_mass_preserving
from rwb_forecasting.metrics import evaluate


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
def test_convex_blend_preserves_totals_bias_h1_and_inputs(weight):
    rng = np.random.default_rng(40)
    incumbent = rng.lognormal(10, 3, (1000, 8))
    alternative = rng.lognormal(9, 2, (1000, 8))
    original = incumbent.copy(), alternative.copy()
    truth = rng.lognormal(10, 3, (1000, 8))
    result, audit = mass_preserving_blend(incumbent, alternative, weight)
    np.testing.assert_array_equal(result[:, 0], incumbent[:, 0])
    np.testing.assert_allclose(result.sum(axis=0), incumbent.sum(axis=0), rtol=2e-14)
    np.testing.assert_array_equal(incumbent, original[0])
    np.testing.assert_array_equal(alternative, original[1])
    assert evaluate(truth, result)["signed_bias"] == pytest.approx(evaluate(truth, incumbent)["signed_bias"], abs=2e-14)
    assert audit["first_horizon_exact"]
    assert audit["maximum_relative_total_drift"] < 2e-14
    assert (result >= 0).all()
    normalized = alternative[:, 1:] / alternative[:, 1:].sum(axis=0) * incumbent[:, 1:].sum(axis=0)
    np.testing.assert_allclose(result[:, 1:], (1 - weight) * incumbent[:, 1:] + weight * normalized)


def test_zero_alternative_horizon_retains_incumbent_and_zero_mass_stays_zero():
    incumbent = np.arange(1, 25, dtype=float).reshape(3, 8)
    incumbent[:, 3] = 0
    alternative = np.ones((3, 8))
    alternative[:, 2] = 0
    result, audit = mass_preserving_blend(incumbent, alternative, 1.0)
    np.testing.assert_array_equal(result[:, 2], incumbent[:, 2])
    np.testing.assert_array_equal(result[:, 3], 0)
    assert audit["zero_alternative_fallback_horizons"] == [3]
    np.testing.assert_allclose(result.sum(axis=0), incumbent.sum(axis=0))


def test_source_causal_wrapper_receives_history_and_retains_nested_audits():
    times = pd.date_range("2025-01-01", periods=30, freq="30min")
    target = np.arange(1, 91, dtype=float).reshape(30, 3)
    data = ForecastData(times, np.array([2, 9, 12]), np.ones((30, 3, 6)), target)
    seen = []
    def incumbent(history):
        seen.append(history.timestamps[-1])
        return np.repeat(history.target[-1, :, None], 8, axis=1), {
            "latest_label_timestamp": str(history.timestamps[-1]), "component": "incumbent"}
    def alternative(history):
        seen.append(history.timestamps[-1])
        return np.repeat(history.target[-2, ::-1, None], 8, axis=1), {
            "nested": {"latest_label_timestamp": str(history.timestamps[-2])}, "component": "alternative"}
    predictor = lambda history: predict_mass_preserving(history, incumbent, alternative, 0.5)
    origin = times[18]
    result = compare_models(data, [origin], {"blend": predictor})
    assert seen == [origin, origin]
    audit = result["audits"][0]
    assert audit["incumbent_model"]["component"] == "incumbent"
    assert audit["alternative_model"]["component"] == "alternative"
    validate_training_boundary(audit, origin)


def test_wrapper_rejects_future_labels_in_either_component():
    times = pd.date_range("2025-01-01", periods=10, freq="30min")
    data = ForecastData(times, np.array([0, 1]), np.ones((10, 2, 6)), np.ones((10, 2)))
    good = lambda history: (np.ones((2, 8)), {"latest_label_timestamp": str(times[-1])})
    bad = lambda history: (np.ones((2, 8)), {"nested": {"latest_label_timestamp": str(times[-1] + pd.Timedelta(minutes=30))}})
    for incumbent, alternative in [(good, bad), (bad, good)]:
        with pytest.raises(ValueError, match="exceed"):
            predict_mass_preserving(data, incumbent, alternative)


@pytest.mark.parametrize("damage", ["negative", "nan", "shape", "weight", "total_overflow"])
def test_invalid_forecasts_or_weight_rejected(damage):
    incumbent = np.ones((3, 8))
    alternative = np.ones((3, 8))
    weight = 0.5
    if damage == "negative":
        alternative[0, 0] = -1
    elif damage == "nan":
        incumbent[0, 0] = np.nan
    elif damage == "shape":
        alternative = np.ones((2, 8))
    elif damage == "weight":
        weight = 1.01
    elif damage == "total_overflow":
        incumbent[:] = np.finfo(float).max
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        mass_preserving_blend(incumbent, alternative, weight)
